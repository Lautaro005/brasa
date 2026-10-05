//! Qwen3 denso en GPU: pesos desde `.brasa`, KV cache preasignada y forward por capas.
//!
//! Por capa (T tokens desde la posición `pos0`):
//!   h = RMSNorm(x); q, k, v = W·h (k y v se escriben directo en la caché de la capa)
//!   q, k = RMSNorm por cabeza (QK-norm); q, k = RoPE; o = atención(q, K, V)
//!   x += Wo·o; h = RMSNorm(x); x += Wdown·(silu(Wgate·h) · Wup·h)
//! Activaciones y KV en f32 (fase 1). Todos los buffers se asignan en `Qwen3::load`; el forward
//! solo encola dispatches.

use std::path::Path;

use brasa_kernels::{AttnShape, Kernels, QMatrix, RopeTable, WeightType};
use brasa_metal::{Arg, Buffer, Command, Context};
use brasa_quant::{BrasaFile, QType};

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
}

/// Buffers de trabajo, preasignados.
#[derive(Debug)]
struct Workspace {
    x: Buffer<f32>,
    h: Buffer<f32>,
    q: Buffer<f32>,
    attn: Buffer<f32>,
    gate: Buffer<f32>,
    up: Buffer<f32>,
    scores: Buffer<f32>,
    ids: Buffer<u32>,
    norm_out: Buffer<f32>,
    logits: Buffer<f32>,
}

/// KV cache f32: `[layers, ctx, kv_heads, head_dim]` para K y para V.
#[derive(Debug)]
struct KvCache {
    k: Buffer<f32>,
    v: Buffer<f32>,
}

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
        let kv_len = cfg.layers * limits.ctx * kvd;
        let ws = Workspace {
            x: ctx.buffer(t * h)?,
            h: ctx.buffer(t * h)?,
            q: ctx.buffer(t * qd)?,
            attn: ctx.buffer(t * qd)?,
            gate: ctx.buffer(t * ffn)?,
            up: ctx.buffer(t * ffn)?,
            scores: ctx.buffer(t * cfg.heads * limits.ctx)?,
            ids: ctx.buffer(t)?,
            norm_out: ctx.buffer(t * h)?,
            logits: ctx.buffer(limits.max_logit_rows * cfg.vocab)?,
        };
        Ok(Self {
            kernels: Kernels::new(ctx)?,
            kv: KvCache {
                k: ctx.buffer(kv_len)?,
                v: ctx.buffer(kv_len)?,
            },
            cfg,
            limits,
            embed,
            final_norm,
            layers,
            rope,
            ws,
        })
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
        let kv_off = (li * self.limits.ctx + pos0) * kvd;
        let layer_off = li * self.limits.ctx * kvd;

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
        self.matmul(
            cmd,
            &l.wk,
            Arg::buf(&ws.h),
            Arg::buf_at(&self.kv.k, kv_off),
            tokens,
        );
        self.matmul(
            cmd,
            &l.wv,
            Arg::buf(&ws.h),
            Arg::buf_at(&self.kv.v, kv_off),
            tokens,
        );
        // QK-norm: RMSNorm por cabeza, en el lugar.
        k.rms_norm(
            cmd,
            Arg::buf(&ws.q),
            Arg::buf(&l.q_norm),
            Arg::buf(&ws.q),
            tokens * c.heads,
            c.head_dim,
            c.eps,
        );
        k.rms_norm(
            cmd,
            Arg::buf_at(&self.kv.k, kv_off),
            Arg::buf(&l.k_norm),
            Arg::buf_at(&self.kv.k, kv_off),
            tokens * c.kv_heads,
            c.head_dim,
            c.eps,
        );
        k.rope_neox(
            cmd,
            Arg::buf(&ws.q),
            &self.rope,
            tokens,
            c.heads,
            c.head_dim,
            pos0,
        );
        k.rope_neox(
            cmd,
            Arg::buf_at(&self.kv.k, kv_off),
            &self.rope,
            tokens,
            c.kv_heads,
            c.head_dim,
            pos0,
        );
        k.attention(
            cmd,
            Arg::buf(&ws.q),
            Arg::buf_at(&self.kv.k, layer_off),
            Arg::buf_at(&self.kv.v, layer_off),
            &ws.scores,
            Arg::buf(&ws.attn),
            AttnShape {
                tokens,
                hq: c.heads,
                hkv: c.kv_heads,
                dim: c.head_dim,
                pos0,
            },
        );
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
    /// `logit_rows` posiciones (`[logit_rows, vocab]`); 1 para generar.
    pub fn forward(
        &mut self,
        ctx: &Context,
        ids: &[u32],
        pos0: usize,
        logit_rows: usize,
        logits: &mut [f32],
    ) -> Result<()> {
        let tokens = ids.len();
        self.check_tokens(tokens, pos0)?;
        if logit_rows == 0 || logit_rows > tokens.min(self.limits.max_logit_rows) {
            return Err(Error(format!("logit_rows {logit_rows} inválido")));
        }
        let (c, vocab) = (&self.cfg, self.cfg.vocab);
        assert_eq!(logits.len(), logit_rows * vocab, "buffer de logits");
        self.ws.ids.as_mut_slice()[..tokens].copy_from_slice(ids);
        let mut cmd = ctx.command()?;
        self.kernels.embed(
            &mut cmd,
            self.embed.q(),
            &self.ws.ids,
            Arg::buf(&self.ws.x),
            tokens,
        );
        for li in 0..c.layers {
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
        cmd.commit_and_wait()?;
        logits.copy_from_slice(&self.ws.logits.as_slice()[..logit_rows * vocab]);
        Ok(())
    }
}
