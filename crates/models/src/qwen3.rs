//! Qwen3 denso en GPU: pesos desde `.brasa`, KV cache preasignada y forward por capas.
//!
//! Por capa (T tokens desde la posición `pos0`):
//!   h = RMSNorm(x); q, k, v = W·h (k y v se escriben directo en la caché de la capa)
//!   q, k = RMSNorm por cabeza (QK-norm); q, k = RoPE; o = atención(q, K, V)
//!   x += Wo·o; h = RMSNorm(x); x += Wdown·(silu(Wgate·h) · Wup·h)
//! Activaciones en f32; KV cache en f32, f16 o Q8 (`Limits.kv`, ADR 0009): K y V se calculan en un
//! scratch f32 y `qk_norm_rope_store` aplica QK-norm y RoPE y las escribe en la caché, en un solo
//! dispatch. Todos los buffers se asignan en `Qwen3::load`; el forward solo encola dispatches.

use std::path::Path;

use brasa_kernels::{
    AttnShape, KV_ALIGN, Kernels, QMatrix, RopeTable, WeightType, decode_partials_len,
    gqa_supported,
};
use brasa_memory::planner::{ModelShape, SessionShape, buffer_bytes};
use brasa_metal::{Arg, Buffer, Command, Context};
use brasa_quant::{BrasaFile, QType};

pub use brasa_kernels::KvType;

use crate::{Error, Result};

/// Hiperparámetros de `config.json` que usa el forward.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub vocab: usize,
    pub eps: f32,
    pub rope_theta: f64,
    pub max_position: usize,
}

impl Config {
    pub fn from_json(c: &serde_json::Value) -> Result<Self> {
        let u = |k: &str| {
            c[k].as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| Error(format!("config.json: falta {k}")))
        };
        if c["model_type"] != "qwen3" {
            return Err(Error(format!("model_type {} no es qwen3", c["model_type"])));
        }
        if c["tie_word_embeddings"] != true {
            return Err(Error("se espera tie_word_embeddings = true".into()));
        }
        Ok(Self {
            hidden: u("hidden_size")?,
            layers: u("num_hidden_layers")?,
            heads: u("num_attention_heads")?,
            kv_heads: u("num_key_value_heads")?,
            head_dim: u("head_dim")?,
            ffn: u("intermediate_size")?,
            vocab: u("vocab_size")?,
            eps: c["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            rope_theta: c["rope_theta"].as_f64().unwrap_or(1e6),
            max_position: u("max_position_embeddings")?,
        })
    }

    pub fn q_dim(&self) -> usize {
        self.heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }
}

/// Forma de memoria del modelo en `path`, leyendo solo el encabezado (no carga pesos).
pub fn model_shape(path: &Path) -> Result<(ModelShape, Config)> {
    let f = BrasaFile::open(path)?;
    if f.family() != "qwen3" {
        return Err(Error(format!("familia {} no soportada", f.family())));
    }
    let cfg = Config::from_json(f.config())?;
    let shape = ModelShape {
        tensor_bytes: f.tensors().iter().map(|t| t.nbytes as u64).collect(),
        layers: cfg.layers,
        hidden: cfg.hidden,
        heads: cfg.heads,
        kv_heads: cfg.kv_heads,
        head_dim: cfg.head_dim,
        ffn: cfg.ffn,
        vocab: cfg.vocab,
    };
    Ok((shape, cfg))
}

impl Limits {
    /// Forma de sesión para el planner.
    pub fn session_shape(&self) -> SessionShape {
        SessionShape {
            ctx: self.ctx,
            max_tokens: self.max_tokens,
            max_logit_rows: self.max_logit_rows,
            kv_block_bytes: self.kv.block_bytes(),
        }
    }
}

/// Bytes realmente reservados en buffers Metal, por categoría (para validar el planner).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Allocated {
    pub weights: u64,
    pub kv: u64,
    pub workspace: u64,
}

/// Matriz cuantizada en GPU.
#[derive(Debug)]
struct Matrix {
    data: Buffer<u8>,
    qtype: WeightType,
    rows: usize,
    cols: usize,
}

impl Matrix {
    fn q(&self) -> QMatrix<'_> {
        QMatrix {
            data: &self.data,
            qtype: self.qtype,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

#[derive(Debug)]
struct Layer {
    attn_norm: Buffer<f32>,
    wq: Matrix,
    wk: Matrix,
    wv: Matrix,
    wo: Matrix,
    q_norm: Buffer<f32>,
    k_norm: Buffer<f32>,
    ffn_norm: Buffer<f32>,
    gate: Matrix,
    up: Matrix,
    down: Matrix,
}

/// Tamaños de la sesión, fijados al cargar (el planner de memoria los elige en T1.8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Contexto máximo (posiciones en la KV cache).
    pub ctx: usize,
    /// Tokens máximos por forward (tamaño del bloque de prefill).
    pub max_tokens: usize,
    /// Filas de logits que se pueden pedir en un forward (1 para generar; más para teacher forcing).
    pub max_logit_rows: usize,
    /// Tipo de la KV cache (ADR 0009).
    pub kv: KvType,
}

/// Buffers de trabajo, preasignados.
#[derive(Debug)]
struct Workspace {
    x: Buffer<f32>,
    h: Buffer<f32>,
    q: Buffer<f32>,
    /// K y V de los tokens nuevos antes de escribirse en la caché.
    k_new: Buffer<f32>,
    v_new: Buffer<f32>,
    attn: Buffer<f32>,
    gate: Buffer<f32>,
    up: Buffer<f32>,
    ids: Buffer<u32>,
    norm_out: Buffer<f32>,
    /// Parciales de la atención de decode (un token).
    partials: Buffer<f32>,
    logits: Buffer<f32>,
}

/// KV cache `[layers, ctx, kv_heads, head_dim]` para K y para V, en el tipo de `Limits.kv`
/// (f16 se guarda como bits en `u16`).
#[derive(Debug)]
enum KvCache {
    F32 {
        k: Buffer<f32>,
        v: Buffer<f32>,
    },
    F16 {
        k: Buffer<u16>,
        v: Buffer<u16>,
    },
    /// Bytes: filas de 128 int8 + 4 escalas f16 (ADR 0009).
    Q8 {
        k: Buffer<u8>,
        v: Buffer<u8>,
    },
}

impl KvCache {
    fn new(ctx: &Context, kv: KvType, len: usize) -> Result<Self> {
        Ok(match kv {
            KvType::F32 => KvCache::F32 {
                k: ctx.buffer(len)?,
                v: ctx.buffer(len)?,
            },
            KvType::F16 => KvCache::F16 {
                k: ctx.buffer(len)?,
                v: ctx.buffer(len)?,
            },
            KvType::Q8_0 => KvCache::Q8 {
                k: ctx.buffer(kv.bytes(len))?,
                v: ctx.buffer(kv.bytes(len))?,
            },
        })
    }

    /// K y V a partir del elemento `off` (en Q8, múltiplo de 128: el byte de esa fila).
    fn args(&self, off: usize) -> (Arg<'_>, Arg<'_>) {
        match self {
            KvCache::F32 { k, v } => (Arg::buf_at(k, off), Arg::buf_at(v, off)),
            KvCache::F16 { k, v } => (Arg::buf_at(k, off), Arg::buf_at(v, off)),
            KvCache::Q8 { k, v } => {
                let b = KvType::Q8_0.bytes(off);
                (Arg::buf_at(k, b), Arg::buf_at(v, b))
            }
        }
    }

    fn bytes(&self) -> u64 {
        match self {
            KvCache::F32 { k, v } => {
                buffer_bytes(k.byte_len() as u64) + buffer_bytes(v.byte_len() as u64)
            }
            KvCache::F16 { k, v } => {
                buffer_bytes(k.byte_len() as u64) + buffer_bytes(v.byte_len() as u64)
            }
            KvCache::Q8 { k, v } => {
                buffer_bytes(k.byte_len() as u64) + buffer_bytes(v.byte_len() as u64)
            }
        }
    }
}

/// Capas que van en el primer command buffer de cada forward (ver `Qwen3::forward`).
const FIRST_COMMAND_LAYERS: usize = 2;

/// Modelo cargado en GPU con su KV cache y workspace.
#[derive(Debug)]
pub struct Qwen3 {
    pub cfg: Config,
    pub limits: Limits,
    kernels: Kernels,
    embed: Matrix,
    final_norm: Buffer<f32>,
    layers: Vec<Layer>,
    rope: RopeTable,
    kv: KvCache,
    ws: Workspace,
}

fn load_f32(ctx: &Context, f: &BrasaFile, name: &str) -> Result<Buffer<f32>> {
    let t = f
        .tensor(name)
        .ok_or_else(|| Error(format!("falta el tensor {name}")))?;
    if t.dtype != QType::F32 {
        return Err(Error(format!("{name}: se esperaba f32")));
    }
    let data: Vec<f32> = f
        .data(t)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    Ok(ctx.buffer_from(&data)?)
}

fn load_matrix(
    ctx: &Context,
    f: &BrasaFile,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Matrix> {
    let t = f
        .tensor(name)
        .ok_or_else(|| Error(format!("falta el tensor {name}")))?;
    let qtype = match t.dtype {
        QType::Q4_0 => WeightType::Q4_0,
        QType::Q8_0 => WeightType::Q8_0,
        QType::F32 => return Err(Error(format!("{name}: se esperaba un tensor cuantizado"))),
    };
    if t.shape != [rows, cols] {
        return Err(Error(format!(
            "{name}: forma {:?}, se esperaba [{rows}, {cols}]",
            t.shape
        )));
    }
    Ok(Matrix {
        data: ctx.buffer_from(f.data(t))?,
        qtype,
        rows,
        cols,
    })
}

impl Qwen3 {
    /// Carga los pesos de `path` (`.brasa`) en GPU y reserva KV cache y workspace para `limits`.
    pub fn load(ctx: &Context, path: &Path, limits: Limits) -> Result<Self> {
        let f = BrasaFile::open(path)?;
        if f.family() != "qwen3" {
            return Err(Error(format!("familia {} no soportada", f.family())));
        }
        let cfg = Config::from_json(f.config())?;
        if limits.ctx > cfg.max_position {
            return Err(Error(format!(
                "contexto {} mayor que el máximo del modelo ({})",
                limits.ctx, cfg.max_position
            )));
        }
        let (h, qd, kvd, ffn) = (cfg.hidden, cfg.q_dim(), cfg.kv_dim(), cfg.ffn);
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let n = |s: &str| format!("model.layers.{i}.{s}");
            layers.push(Layer {
                attn_norm: load_f32(ctx, &f, &n("input_layernorm.weight"))?,
                wq: load_matrix(ctx, &f, &n("self_attn.q_proj.weight"), qd, h)?,
                wk: load_matrix(ctx, &f, &n("self_attn.k_proj.weight"), kvd, h)?,
                wv: load_matrix(ctx, &f, &n("self_attn.v_proj.weight"), kvd, h)?,
                wo: load_matrix(ctx, &f, &n("self_attn.o_proj.weight"), h, qd)?,
                q_norm: load_f32(ctx, &f, &n("self_attn.q_norm.weight"))?,
                k_norm: load_f32(ctx, &f, &n("self_attn.k_norm.weight"))?,
                ffn_norm: load_f32(ctx, &f, &n("post_attention_layernorm.weight"))?,
                gate: load_matrix(ctx, &f, &n("mlp.gate_proj.weight"), ffn, h)?,
                up: load_matrix(ctx, &f, &n("mlp.up_proj.weight"), ffn, h)?,
                down: load_matrix(ctx, &f, &n("mlp.down_proj.weight"), h, ffn)?,
            });
        }
        let embed = load_matrix(ctx, &f, "model.embed_tokens.weight", cfg.vocab, h)?;
        if embed.qtype != WeightType::Q8_0 {
            return Err(Error("la tabla de embeddings debe ser q8_0".into()));
        }
        let final_norm = load_f32(ctx, &f, "model.norm.weight")?;
        let rope = RopeTable::new(ctx, cfg.rope_theta, cfg.head_dim, limits.ctx)?;

        let t = limits.max_tokens;
        // Capacidad por capa alineada para flash_attention (lee bloques de KV_ALIGN posiciones).
        let kv_len = cfg.layers * limits.ctx.next_multiple_of(KV_ALIGN) * kvd;
        let ws = Workspace {
            x: ctx.buffer(t * h)?,
            h: ctx.buffer(t * h)?,
            q: ctx.buffer(t * qd)?,
            k_new: ctx.buffer(t * kvd)?,
            v_new: ctx.buffer(t * kvd)?,
            attn: ctx.buffer(t * qd)?,
            gate: ctx.buffer(t * ffn)?,
            up: ctx.buffer(t * ffn)?,
            ids: ctx.buffer(t)?,
            norm_out: ctx.buffer(t * h)?,
            partials: ctx.buffer(decode_partials_len(cfg.heads, limits.ctx))?,
            logits: ctx.buffer(limits.max_logit_rows * cfg.vocab)?,
        };
        Ok(Self {
            kernels: Kernels::new(ctx)?,
            kv: KvCache::new(ctx, limits.kv, kv_len)?,
            cfg,
            limits,
            embed,
            final_norm,
            layers,
            rope,
            ws,
        })
    }

    /// Memoria reservada en buffers Metal (redondeada a páginas como la asigna Metal).
    pub fn allocated(&self) -> Allocated {
        fn b<T: brasa_metal::Element>(x: &Buffer<T>) -> u64 {
            buffer_bytes(x.byte_len() as u64)
        }
        let m = |x: &Matrix| b(&x.data);
        let mut weights = m(&self.embed) + b(&self.final_norm);
        for l in &self.layers {
            weights += b(&l.attn_norm) + b(&l.q_norm) + b(&l.k_norm) + b(&l.ffn_norm);
            weights += [&l.wq, &l.wk, &l.wv, &l.wo, &l.gate, &l.up, &l.down]
                .iter()
                .map(|x| m(x))
                .sum::<u64>();
        }
        let w = &self.ws;
        let workspace = b(&w.x)
            + b(&w.h)
            + b(&w.q)
            + b(&w.k_new)
            + b(&w.v_new)
            + b(&w.attn)
            + b(&w.gate)
            + b(&w.up)
            + b(&w.ids)
            + b(&w.norm_out)
            + b(&w.partials)
            + b(&w.logits)
            + b(&self.rope.cos)
            + b(&self.rope.sin);
        Allocated {
            weights,
            kv: self.kv.bytes(),
            workspace,
        }
    }

    fn matmul<'a>(
        &self,
        cmd: &mut Command<'a>,
        w: &'a Matrix,
        x: Arg<'a>,
        y: Arg<'a>,
        tokens: usize,
    ) {
        // GEMV (un simdgroup por fila) para pocos tokens; GEMM simple para prefill.
        if tokens <= 8 {
            self.kernels.gemv(cmd, w.q(), x, y, tokens);
        } else {
            self.kernels.gemm(cmd, w.q(), x, y, tokens);
        }
    }

    /// Encola la capa `li` para `tokens` tokens desde `pos0`. Entrada y salida en `ws.x`.
    fn encode_layer<'a>(&'a self, cmd: &mut Command<'a>, li: usize, tokens: usize, pos0: usize) {
        let c = &self.cfg;
        let (k, ws, l) = (&self.kernels, &self.ws, &self.layers[li]);
        let kvd = c.kv_dim();
        // Vista de la caché de esta capa desde la posición pos0.
        let cap = self.limits.ctx.next_multiple_of(KV_ALIGN);
        let kv_off = (li * cap + pos0) * kvd;
        let layer_off = li * cap * kvd;

        k.rms_norm(
            cmd,
            Arg::buf(&ws.x),
            Arg::buf(&l.attn_norm),
            Arg::buf(&ws.h),
            tokens,
            c.hidden,
            c.eps,
        );
        self.matmul(cmd, &l.wq, Arg::buf(&ws.h), Arg::buf(&ws.q), tokens);
        self.matmul(cmd, &l.wk, Arg::buf(&ws.h), Arg::buf(&ws.k_new), tokens);
        self.matmul(cmd, &l.wv, Arg::buf(&ws.h), Arg::buf(&ws.v_new), tokens);
        // QK-norm, RoPE y K/V a la caché (en su tipo), en un dispatch (T3.5).
        let kvt = self.limits.kv;
        let shape = AttnShape {
            tokens,
            hq: c.heads,
            hkv: c.kv_heads,
            dim: c.head_dim,
            pos0,
            kv: kvt,
        };
        let (k_dst, v_dst) = self.kv.args(kv_off);
        k.qk_norm_rope_store(
            cmd,
            kvt,
            [Arg::buf(&ws.q), Arg::buf(&ws.k_new), Arg::buf(&ws.v_new)],
            [Arg::buf(&l.q_norm), Arg::buf(&l.k_norm)],
            c.eps,
            &self.rope,
            [k_dst, v_dst],
            shape,
        );
        let (kc, vc) = self.kv.args(layer_off);
        if tokens == 1 && gqa_supported(c.heads, c.kv_heads) {
            k.decode_attention_lanes(
                cmd,
                Arg::buf(&ws.q),
                kc,
                vc,
                &ws.partials,
                Arg::buf(&ws.attn),
                shape,
            );
        } else if tokens == 1 {
            k.decode_attention(
                cmd,
                Arg::buf(&ws.q),
                kc,
                vc,
                &ws.partials,
                Arg::buf(&ws.attn),
                shape,
            );
        } else {
            k.flash_attention(cmd, Arg::buf(&ws.q), kc, vc, Arg::buf(&ws.attn), shape);
        }
        self.matmul(cmd, &l.wo, Arg::buf(&ws.attn), Arg::buf(&ws.h), tokens);
        k.add(cmd, &ws.x, &ws.h, &ws.x, tokens * c.hidden);

        k.rms_norm(
            cmd,
            Arg::buf(&ws.x),
            Arg::buf(&l.ffn_norm),
            Arg::buf(&ws.h),
            tokens,
            c.hidden,
            c.eps,
        );
        self.matmul(cmd, &l.gate, Arg::buf(&ws.h), Arg::buf(&ws.gate), tokens);
        self.matmul(cmd, &l.up, Arg::buf(&ws.h), Arg::buf(&ws.up), tokens);
        k.swiglu(cmd, &ws.gate, &ws.up, &ws.gate, tokens * c.ffn);
        self.matmul(cmd, &l.down, Arg::buf(&ws.gate), Arg::buf(&ws.h), tokens);
        k.add(cmd, &ws.x, &ws.h, &ws.x, tokens * c.hidden);
    }

    fn check_tokens(&self, tokens: usize, pos0: usize) -> Result<()> {
        if tokens == 0 || tokens > self.limits.max_tokens {
            return Err(Error(format!(
                "{tokens} tokens por forward; el máximo es {}",
                self.limits.max_tokens
            )));
        }
        if pos0 + tokens > self.limits.ctx {
            return Err(Error(format!(
                "posición {} fuera del contexto ({})",
                pos0 + tokens,
                self.limits.ctx
            )));
        }
        Ok(())
    }

    /// Corre las capas `layers` sobre el estado oculto `x: [tokens, hidden]` y devuelve la salida.
    /// Escribe la KV cache de esas capas. Pensado para validar capas contra la referencia (T1.5).
    pub fn run_layers(
        &mut self,
        ctx: &Context,
        x: &[f32],
        pos0: usize,
        layers: std::ops::Range<usize>,
    ) -> Result<Vec<f32>> {
        let tokens = x.len() / self.cfg.hidden;
        self.check_tokens(tokens, pos0)?;
        self.ws.x.as_mut_slice()[..x.len()].copy_from_slice(x);
        let mut cmd = ctx.command()?;
        for li in layers {
            self.encode_layer(&mut cmd, li, tokens, pos0);
        }
        cmd.commit_and_wait()?;
        Ok(self.ws.x.as_slice()[..x.len()].to_vec())
    }

    /// Forward completo de `ids` desde `pos0`. Escribe en `logits` las filas de las últimas
    /// `logit_rows` posiciones (`[logit_rows, vocab]`); 1 para generar. Devuelve el tiempo de GPU.
    pub fn forward(
        &mut self,
        ctx: &Context,
        ids: &[u32],
        pos0: usize,
        logit_rows: usize,
        logits: &mut [f32],
    ) -> Result<brasa_metal::GpuTiming> {
        let tokens = ids.len();
        self.check_tokens(tokens, pos0)?;
        if logit_rows == 0 || logit_rows > tokens.min(self.limits.max_logit_rows) {
            return Err(Error(format!("logit_rows {logit_rows} inválido")));
        }
        let (c, vocab) = (&self.cfg, self.cfg.vocab);
        assert_eq!(logits.len(), logit_rows * vocab, "buffer de logits");
        self.ws.ids.as_mut_slice()[..tokens].copy_from_slice(ids);
        // Las primeras capas van en un comando aparte que se envía enseguida: la GPU arranca
        // mientras la CPU codifica el resto (T3.5; en decode, codificar todo tomaba ~0,5 ms).
        let mut head = ctx.command()?;
        self.kernels.embed(
            &mut head,
            self.embed.q(),
            &self.ws.ids,
            Arg::buf(&self.ws.x),
            tokens,
        );
        let split = FIRST_COMMAND_LAYERS.min(c.layers);
        for li in 0..split {
            self.encode_layer(&mut head, li, tokens, pos0);
        }
        let head = head.commit();
        let mut cmd = ctx.command()?;
        for li in split..c.layers {
            self.encode_layer(&mut cmd, li, tokens, pos0);
        }
        let first = tokens - logit_rows;
        self.kernels.rms_norm(
            &mut cmd,
            Arg::buf_at(&self.ws.x, first * c.hidden),
            Arg::buf(&self.final_norm),
            Arg::buf(&self.ws.norm_out),
            logit_rows,
            c.hidden,
            c.eps,
        );
        // lm_head atado a la tabla de embeddings (q8_0).
        self.kernels.gemv(
            &mut cmd,
            self.embed.q(),
            Arg::buf(&self.ws.norm_out),
            Arg::buf(&self.ws.logits),
            logit_rows,
        );
        let tail = cmd.commit_and_wait()?;
        let first_part = head.wait()?;
        logits.copy_from_slice(&self.ws.logits.as_slice()[..logit_rows * vocab]);
        Ok(brasa_metal::GpuTiming {
            gpu_seconds: first_part.gpu_seconds + tail.gpu_seconds,
        })
    }
}

#[cfg(test)]
mod tests {
    /// `brasa-memory` no depende de `brasa-kernels` y copia sus constantes: tienen que coincidir.
    #[test]
    fn constantes_del_planner_iguales_a_las_de_los_kernels() {
        use brasa_memory::planner;
        assert_eq!(planner::DECODE_CHUNK, brasa_kernels::DECODE_CHUNK);
        assert_eq!(planner::KV_ALIGN, brasa_kernels::KV_ALIGN);
    }
}
