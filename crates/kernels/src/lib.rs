//! Fuentes .metal y registro de variantes por chip.
//!
//! Cada kernel tiene: fuente en `src/metal/`, referencia CPU en [`reference`], test de
//! equivalencia en `tests/` y microbenchmark en `benches/` (regla 2 de CLAUDE.md, ADR 0004).
//! Activaciones en f32. Tolerancias (distancia a la referencia CPU, que acumula en f64):
//!
//! | Kernel | Tolerancia |
//! |---|---|
//! | `add_f32`, `embed_q8_0` | exacto (0 ULP) |
//! | `swiglu_f32` | error relativo ≤ 1e-6 |
//! | `rms_norm_f32`, `softmax_f32` | error relativo ≤ 1e-5 |
//! | `rope_neox_f32` | `|err| ≤ 1e-6 · (|a| + |b|)` del par rotado |
//! | `gemv/gemm_q4_0/q8_0_f32` | `|err| ≤ 1e-5 · Σ_k |w_k · x_k|` |
//! | atención (`attention`, `flash_attention`, `decode_attention*`) | `|err| ≤ 1e-5 · max_j |v_j|` por componente de salida |
//! | `store_kv` | f32 exacto; f16 igual a `f32_to_f16` (redondeo al par) |
//!
//! La KV cache puede ser f32 o f16 ([`KvType`], ADR 0009). En f16 la referencia CPU recibe K y V
//! ya redondeados a f16, así que la tolerancia de la atención no cambia.

use brasa_metal::{Arg, Buffer, Command, Context, MetalError, Pipeline};
use brasa_quant::{KV_Q8_DIM, KV_Q8_ROW};

pub mod launch;
pub mod reference;
pub mod testutil;

pub use launch::{GEMV_NR, GEMV_SG_DEFAULT, GemvOp, LANES_SG_DEFAULT, Launch, LaunchError};

/// Fuentes MSL embebidas en el binario.
pub mod sources {
    pub const ELEMENTWISE: &str = include_str!("metal/elementwise.metal");
    pub const NORM: &str = include_str!("metal/norm.metal");
    pub const SOFTMAX: &str = include_str!("metal/softmax.metal");
    pub const ROPE: &str = include_str!("metal/rope.metal");
    pub const EMBED: &str = include_str!("metal/embed.metal");
    pub const MATMUL: &str = include_str!("metal/matmul.metal");
    pub const ATTENTION: &str = include_str!("metal/attention.metal");
    pub const MATMUL_TILED: &str = include_str!("metal/matmul_tiled.metal");
    pub const FLASH_ATTENTION: &str = include_str!("metal/flash_attention.metal");
    pub const DECODE_ATTENTION: &str = include_str!("metal/decode_attention.metal");
    pub const KV: &str = include_str!("metal/kv.metal");
    pub const QKV: &str = include_str!("metal/qkv.metal");
    /// Acceso a la KV cache por tipo; [`super::kv_source`] lo antepone a los kernels de atención.
    pub const KV_ACCESS: &str = include_str!("metal/kv_access.metal");

    /// Todas las fuentes con su nombre de archivo. La versión de los kernels del fingerprint de
    /// tuning (ADR 0026) es un hash de esta lista: un archivo nuevo tiene que sumarse acá.
    pub const ALL: &[(&str, &str)] = &[
        ("elementwise.metal", ELEMENTWISE),
        ("norm.metal", NORM),
        ("softmax.metal", SOFTMAX),
        ("rope.metal", ROPE),
        ("embed.metal", EMBED),
        ("matmul.metal", MATMUL),
        ("attention.metal", ATTENTION),
        ("matmul_tiled.metal", MATMUL_TILED),
        ("flash_attention.metal", FLASH_ATTENTION),
        ("decode_attention.metal", DECODE_ATTENTION),
        ("kv.metal", KV),
        ("qkv.metal", QKV),
        ("kv_access.metal", KV_ACCESS),
    ];
}

/// Fuente de un kernel de atención para el tipo de KV `kv`: el `#define` del tipo, el acceso a
/// la caché (`kv_access.metal`) y `source`.
pub fn kv_source(kv: KvType, source: &str) -> String {
    let define = match kv {
        KvType::F32 => "",
        KvType::F16 => "#define KV_F16 1\n",
        KvType::Q8_0 => "#define KV_Q8 1\n",
    };
    format!("{define}{}\n{source}", sources::KV_ACCESS)
}

/// Hilos por threadgroup para kernels elemento a elemento.
const ELEMENTWISE_TG: usize = 256;
/// Hilos por threadgroup de los kernels por filas (norm, softmax); debe coincidir con `TG` en MSL.
const ROW_TG: usize = 256;
/// Filas por threadgroup de los GEMV (un simdgroup por fila); `ROWS_PER_TG` en MSL.
const GEMV_ROWS_PER_TG: usize = 4;

/// Esquema de cuantización de una matriz de pesos.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightType {
    Q4_0,
    Q8_0,
    /// Tabla de embeddings atada (ADR 0012): embedding, GEMV y GEMM simple (sin tiled).
    Q6_0,
}

/// Tipo de elemento de la KV cache (ADR 0009).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvType {
    F32,
    #[default]
    F16,
    /// Filas de 128 `int8` + 4 escalas f16 (136 bytes; ADR 0009).
    Q8_0,
}

impl KvType {
    /// Bytes por bloque de 32 elementos (unidad del planner).
    pub fn block_bytes(self) -> usize {
        match self {
            KvType::F32 => 128,
            KvType::F16 => 64,
            KvType::Q8_0 => 34,
        }
    }

    /// Bytes de `n` elementos de la caché (`n` múltiplo de 128 en Q8: filas completas).
    pub fn bytes(self, n: usize) -> usize {
        match self {
            KvType::Q8_0 => {
                assert_eq!(n % KV_Q8_DIM, 0, "KV Q8: filas de {KV_Q8_DIM} valores");
                n / KV_Q8_DIM * KV_Q8_ROW
            }
            _ => n * self.block_bytes() / 32,
        }
    }

    fn idx(self) -> usize {
        self as usize
    }

    pub fn name(self) -> &'static str {
        match self {
            KvType::F32 => "f32",
            KvType::F16 => "f16",
            KvType::Q8_0 => "q8_0",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "f32" => Some(KvType::F32),
            "f16" => Some(KvType::F16),
            "q8_0" | "q8" => Some(KvType::Q8_0),
            _ => None,
        }
    }
}

/// Matriz de pesos cuantizada `[rows, cols]` en un buffer de bytes (layout de ADR 0006).
#[derive(Debug, Clone, Copy)]
pub struct QMatrix<'a> {
    pub data: &'a Buffer<u8>,
    pub qtype: WeightType,
    pub rows: usize,
    pub cols: usize,
}

/// Pipelines compilados al iniciar; el camino caliente no compila nada.
#[derive(Debug)]
pub struct Kernels {
    add_f32: Pipeline,
    swiglu_f32: Pipeline,
    rms_norm_f32: Pipeline,
    softmax_f32: Pipeline,
    rope_neox_f32: Pipeline,
    embed_q8_0: Pipeline,
    embed_q6_0: Pipeline,
    gemv_q6_0: Pipeline,
    gemm_q6_0: Pipeline,
    gemv_fast_q6_0: Pipeline,
    gemv_scaled_q4_0: Pipeline,
    gemv_scaled_q6_0: Pipeline,
    add_norm_prep: Pipeline,
    gemv_scaled3_q4_0: Pipeline,
    gemv_scaled_swiglu_q4_0: Pipeline,
    gemv_q4_0: Pipeline,
    gemv_q8_0: Pipeline,
    gemm_q4_0: Pipeline,
    gemm_q8_0: Pipeline,
    attn_scores_f32: Pipeline,
    attn_pv_f32: Pipeline,
    gemv_fast_q4_0: Pipeline,
    gemv_fast_q8_0: Pipeline,
    gemm_tiled_q4_0: Pipeline,
    gemm_tiled_q8_0: Pipeline,
    /// Variantes por [`KvType`] (índice `KvType as usize`).
    flash_attn: [Pipeline; 3],
    /// `flash_attn_gqa` por tamaño de grupo GQA ([`GQA_GROUPS`]) y tipo de KV.
    flash_attn_gqa: [[Pipeline; 3]; 4],
    attn_decode_partial: [Pipeline; 3],
    /// `attn_decode_lanes` por tamaño de grupo GQA ([`GQA_GROUPS`]) y tipo de KV.
    attn_decode_lanes: [[Pipeline; 3]; 4],
    store_kv: [Pipeline; 3],
    /// `qk_norm_rope_store` por tipo de KV (T3.5).
    qk_norm_rope_store: [Pipeline; 3],
    attn_decode_reduce: Pipeline,
    /// Parámetros de lanzamiento de decode (ADR 0029); por defecto, los fijados a mano.
    launch: Launch,
    /// Variantes de los GEMV de decode con otros simdgroups por threadgroup (compiladas por
    /// `set_launch`).
    gemv_variants: Vec<(GemvOp, usize, Pipeline)>,
    /// Variantes de `attn_decode_lanes` (índice de grupo GQA, KV, simdgroups).
    lanes_variants: Vec<(usize, KvType, usize, Pipeline)>,
}

/// Compila `function` para cada tamaño de grupo de [`GQA_GROUPS`] (`#define GQA_G n`) y tipo de KV.
fn gqa_variants(
    ctx: &Context,
    source: &str,
    function: &str,
) -> Result<[[Pipeline; 3]; 4], MetalError> {
    let v = |g: usize| kv_variants(ctx, &format!("#define GQA_G {g}\n{source}"), function);
    Ok([v(1)?, v(2)?, v(4)?, v(8)?])
}

/// Compila `function` de `source` para cada tipo de KV (ver [`kv_source`]).
fn kv_variants(ctx: &Context, source: &str, function: &str) -> Result<[Pipeline; 3], MetalError> {
    let v = |kv| ctx.pipeline(&kv_source(kv, source), function);
    Ok([v(KvType::F32)?, v(KvType::F16)?, v(KvType::Q8_0)?])
}

fn groups(n: usize, per: usize) -> usize {
    n.div_ceil(per)
}

impl Kernels {
    pub fn new(ctx: &Context) -> Result<Self, MetalError> {
        use sources::*;
        Ok(Self {
            add_f32: ctx.pipeline(ELEMENTWISE, "add_f32")?,
            swiglu_f32: ctx.pipeline(ELEMENTWISE, "swiglu_f32")?,
            rms_norm_f32: ctx.pipeline(NORM, "rms_norm_f32")?,
            softmax_f32: ctx.pipeline(SOFTMAX, "softmax_f32")?,
            rope_neox_f32: ctx.pipeline(ROPE, "rope_neox_f32")?,
            embed_q8_0: ctx.pipeline(EMBED, "embed_q8_0")?,
            embed_q6_0: ctx.pipeline(EMBED, "embed_q6_0")?,
            gemv_q6_0: ctx.pipeline(MATMUL, "gemv_q6_0_f32")?,
            gemm_q6_0: ctx.pipeline(MATMUL, "gemm_q6_0_f32")?,
            gemv_fast_q6_0: ctx.pipeline(MATMUL, "gemv_fast_q6_0_f32")?,
            gemv_scaled_q4_0: ctx.pipeline(MATMUL, "gemv_scaled_q4_0_f32")?,
            gemv_scaled_q6_0: ctx.pipeline(MATMUL, "gemv_scaled_q6_0_f32")?,
            add_norm_prep: ctx.pipeline(NORM, "add_norm_prep")?,
            gemv_scaled3_q4_0: ctx.pipeline(MATMUL, "gemv_scaled3_q4_0_f32")?,
            gemv_scaled_swiglu_q4_0: ctx.pipeline(MATMUL, "gemv_scaled_swiglu_q4_0_f32")?,
            gemv_q4_0: ctx.pipeline(MATMUL, "gemv_q4_0_f32")?,
            gemv_q8_0: ctx.pipeline(MATMUL, "gemv_q8_0_f32")?,
            gemm_q4_0: ctx.pipeline(MATMUL, "gemm_q4_0_f32")?,
            gemm_q8_0: ctx.pipeline(MATMUL, "gemm_q8_0_f32")?,
            attn_scores_f32: ctx.pipeline(ATTENTION, "attn_scores_f32")?,
            attn_pv_f32: ctx.pipeline(ATTENTION, "attn_pv_f32")?,
            gemv_fast_q4_0: ctx.pipeline(MATMUL, "gemv_fast_q4_0_f32")?,
            gemv_fast_q8_0: ctx.pipeline(MATMUL, "gemv_fast_q8_0_f32")?,
            gemm_tiled_q4_0: ctx.pipeline(MATMUL_TILED, "gemm_tiled_q4_0_f32")?,
            gemm_tiled_q8_0: ctx.pipeline(MATMUL_TILED, "gemm_tiled_q8_0_f32")?,
            flash_attn: kv_variants(ctx, FLASH_ATTENTION, "flash_attn_f32")?,
            attn_decode_partial: kv_variants(ctx, DECODE_ATTENTION, "attn_decode_partial")?,
            attn_decode_lanes: gqa_variants(ctx, DECODE_ATTENTION, "attn_decode_lanes")?,
            flash_attn_gqa: gqa_variants(ctx, FLASH_ATTENTION, "flash_attn_gqa")?,
            qk_norm_rope_store: kv_variants(ctx, QKV, "qk_norm_rope_store")?,
            store_kv: [
                ctx.pipeline(KV, "store_kv_f32")?,
                ctx.pipeline(KV, "store_kv_f16")?,
                ctx.pipeline(KV, "store_kv_q8")?,
            ],
            attn_decode_reduce: ctx.pipeline(
                &kv_source(KvType::F32, DECODE_ATTENTION),
                "attn_decode_reduce",
            )?,
            launch: Launch::default(),
            gemv_variants: Vec::new(),
            lanes_variants: Vec::new(),
        })
    }

    /// Reemplaza los parámetros de lanzamiento de decode (ADR 0029). Se llama al cargar, nunca
    /// en el loop de decode: compila las variantes que falten (`#define GEMV_SG n` o
    /// `#define LANES_SG n`). Ningún parámetro cambia los resultados. Si una variante no compila o
    /// pide más hilos por threadgroup de los que admite su pipeline, devuelve error y deja los
    /// parámetros anteriores.
    pub fn set_launch(&mut self, ctx: &Context, launch: Launch) -> Result<(), LaunchError> {
        let err = |e: MetalError| LaunchError(e.to_string());
        for (op, sg) in launch.gemv_entries() {
            if sg == GEMV_SG_DEFAULT || self.gemv_variant(op, sg).is_some() {
                continue;
            }
            let src = format!("#define GEMV_SG {sg}\n{}", sources::MATMUL);
            let p = ctx.pipeline(&src, op.kernel_name()).map_err(err)?;
            if p.max_threads_per_threadgroup() < 32 * sg {
                return Err(LaunchError(format!(
                    "{}: {sg} simdgroups por threadgroup, el pipeline admite {}",
                    op.kernel_name(),
                    p.max_threads_per_threadgroup() / 32
                )));
            }
            self.gemv_variants.push((op, sg, p));
        }
        for (kv, group, sg) in launch.lanes_entries() {
            let gi = GQA_GROUPS.iter().position(|&g| g == group).ok_or_else(|| {
                LaunchError(format!("attn_decode_lanes: grupo GQA {group} no compilado"))
            })?;
            if sg == LANES_SG_DEFAULT || self.lanes_variant(gi, kv, sg).is_some() {
                continue;
            }
            let src = format!(
                "#define GQA_G {group}\n#define LANES_SG {sg}\n{}",
                sources::DECODE_ATTENTION
            );
            let p = ctx
                .pipeline(&kv_source(kv, &src), "attn_decode_lanes")
                .map_err(err)?;
            if p.max_threads_per_threadgroup() < 32 * sg {
                return Err(LaunchError(format!(
                    "attn_decode_lanes kv {} g {group}: {sg} simdgroups por threadgroup, el \
                     pipeline admite {}",
                    kv.name(),
                    p.max_threads_per_threadgroup() / 32
                )));
            }
            self.lanes_variants.push((gi, kv, sg, p));
        }
        self.launch = launch;
        Ok(())
    }

    fn gemv_variant(&self, op: GemvOp, sg: usize) -> Option<&Pipeline> {
        self.gemv_variants
            .iter()
            .find(|(o, s, _)| *o == op && *s == sg)
            .map(|v| &v.2)
    }

    fn lanes_variant(&self, gi: usize, kv: KvType, sg: usize) -> Option<&Pipeline> {
        self.lanes_variants
            .iter()
            .find(|(g, k, s, _)| *g == gi && *k == kv && *s == sg)
            .map(|v| &v.3)
    }

    /// Pipeline y simdgroups por threadgroup del GEMV `op` `[rows, cols]` según el `Launch`
    /// (`base` es la variante por defecto). Sin asignaciones: corre en cada paso de decode.
    fn gemv_pick<'p>(
        &'p self,
        op: GemvOp,
        rows: usize,
        cols: usize,
        base: &'p Pipeline,
    ) -> (&'p Pipeline, usize) {
        let sg = self.launch.gemv_sg(op, rows, cols);
        match self.gemv_variant(op, sg) {
            Some(p) if sg != GEMV_SG_DEFAULT => (p, sg),
            _ => (base, GEMV_SG_DEFAULT),
        }
    }

    pub fn launch(&self) -> &Launch {
        &self.launch
    }

    /// `out = a + b` sobre los primeros `n` elementos.
    pub fn add<'a>(
        &self,
        cmd: &mut Command<'a>,
        a: &'a Buffer<f32>,
        b: &'a Buffer<f32>,
        out: &'a Buffer<f32>,
        n: usize,
    ) {
        assert!(a.len() >= n && b.len() >= n && out.len() >= n);
        cmd.dispatch(
            &self.add_f32,
            &[Arg::buf(a), Arg::buf(b), Arg::buf(out), Arg::u32(n as u32)],
            [n, 1, 1],
            [ELEMENTWISE_TG, 1, 1],
        );
    }

    /// `out = silu(gate) · up` sobre los primeros `n` elementos.
    pub fn swiglu<'a>(
        &self,
        cmd: &mut Command<'a>,
        gate: &'a Buffer<f32>,
        up: &'a Buffer<f32>,
        out: &'a Buffer<f32>,
        n: usize,
    ) {
        assert!(gate.len() >= n && up.len() >= n && out.len() >= n);
        cmd.dispatch(
            &self.swiglu_f32,
            &[
                Arg::buf(gate),
                Arg::buf(up),
                Arg::buf(out),
                Arg::u32(n as u32),
            ],
            [n, 1, 1],
            [ELEMENTWISE_TG, 1, 1],
        );
    }

    /// RMSNorm de `rows` filas de largo `n`; `w` tiene `n` elementos.
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm<'a>(
        &self,
        cmd: &mut Command<'a>,
        x: Arg<'a>,
        w: Arg<'a>,
        out: Arg<'a>,
        rows: usize,
        n: usize,
        eps: f32,
    ) {
        cmd.dispatch_groups(
            &self.rms_norm_f32,
            &[x, w, out, Arg::u32(n as u32), Arg::f32(eps)],
            [rows, 1, 1],
            [ROW_TG, 1, 1],
        );
    }

    /// Softmax de `rows` filas de largo `n`.
    pub fn softmax<'a>(
        &self,
        cmd: &mut Command<'a>,
        x: &'a Buffer<f32>,
        out: &'a Buffer<f32>,
        rows: usize,
        n: usize,
    ) {
        assert!(x.len() >= rows * n && out.len() >= rows * n);
        cmd.dispatch_groups(
            &self.softmax_f32,
            &[Arg::buf(x), Arg::buf(out), Arg::u32(n as u32)],
            [rows, 1, 1],
            [ROW_TG, 1, 1],
        );
    }

    /// RoPE NeoX en el lugar sobre `x: [tokens, heads, dim]`, posiciones `pos0..pos0+tokens`.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_neox<'a>(
        &self,
        cmd: &mut Command<'a>,
        x: Arg<'a>,
        table: &'a RopeTable,
        tokens: usize,
        heads: usize,
        dim: usize,
        pos0: usize,
    ) {
        assert_eq!(dim, table.dim, "dimensión de la tabla RoPE");
        assert!(
            pos0 + tokens <= table.max_pos,
            "posición fuera de la tabla RoPE"
        );
        cmd.dispatch(
            &self.rope_neox_f32,
            &[
                x,
                Arg::buf(&table.cos),
                Arg::buf(&table.sin),
                Arg::u32(heads as u32),
                Arg::u32(dim as u32),
                Arg::u32(pos0 as u32),
            ],
            [dim / 2, heads, tokens],
            [(dim / 2).min(64), 1, 1],
        );
    }

    /// Embedding desde una tabla q8_0 o q6_0 `[vocab, h]`: `out[t, :] = tabla[ids[t], :]`.
    pub fn embed<'a>(
        &self,
        cmd: &mut Command<'a>,
        table: QMatrix<'a>,
        ids: &'a Buffer<u32>,
        out: Arg<'a>,
        tokens: usize,
    ) {
        let p = match table.qtype {
            WeightType::Q8_0 => &self.embed_q8_0,
            WeightType::Q6_0 => &self.embed_q6_0,
            WeightType::Q4_0 => panic!("embedding en q8_0 o q6_0"),
        };
        assert!(ids.len() >= tokens);
        cmd.dispatch(
            p,
            &[
                Arg::buf(table.data),
                Arg::buf(ids),
                out,
                Arg::u32(table.cols as u32),
            ],
            [table.cols, tokens, 1],
            [ELEMENTWISE_TG, 1, 1],
        );
    }

    /// GEMV para decode: `y[t, :] = W · x[t, :]` para `t < tokens` (T chico). Usa la versión de
    /// 4 filas por simdgroup si `rows % 8 == 0`; si no, la simple.
    /// Preparación de RMSNorm para decode (T3.5): `x += h` (si hay `h`), `xw = x · w` y sumas
    /// parciales de `x²` en `ss` (una por cada 256 elementos; ver [`norm_partials`]). Lo consume
    /// [`Kernels::gemv_scaled`].
    #[allow(clippy::too_many_arguments)]
    pub fn add_norm_prep<'a>(
        &self,
        cmd: &mut Command<'a>,
        x: Arg<'a>,
        h: Option<Arg<'a>>,
        w: Arg<'a>,
        xw: Arg<'a>,
        ss: Arg<'a>,
        n: usize,
    ) {
        let add = h.is_some();
        cmd.dispatch_groups(
            &self.add_norm_prep,
            &[
                x,
                h.unwrap_or(x),
                w,
                xw,
                ss,
                Arg::u32(n as u32),
                Arg::u32(add as u32),
            ],
            [norm_partials(n), 1, 1],
            [NORM_PREP_TG, 1, 1],
        );
    }

    /// `y = W · RMSNorm(x)` para un token a partir de [`Kernels::add_norm_prep`]: `xw = x · w` y
    /// las sumas parciales `ss` de `x²`. Requiere `rows % 8 == 0`; q4_0 o q6_0.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_scaled<'a>(
        &self,
        cmd: &mut Command<'a>,
        w: QMatrix<'a>,
        xw: Arg<'a>,
        ss: Arg<'a>,
        eps: f32,
        y: Arg<'a>,
    ) {
        assert_eq!(w.rows % 8, 0, "gemv_scaled: filas % 8");
        let p = match w.qtype {
            WeightType::Q4_0 => &self.gemv_scaled_q4_0,
            WeightType::Q6_0 => &self.gemv_scaled_q6_0,
            WeightType::Q8_0 => panic!("gemv_scaled: q4_0 o q6_0"),
        };
        let (p, sg) = self.gemv_pick(GemvOp::Scaled(w.qtype), w.rows, w.cols, p);
        cmd.dispatch_groups(
            p,
            &[
                Arg::buf(w.data),
                xw,
                y,
                Arg::u32(w.rows as u32),
                Arg::u32(w.cols as u32),
                ss,
                Arg::u32(norm_partials(w.cols) as u32),
                Arg::f32(eps),
            ],
            [w.rows / (GEMV_NR * sg), 1, 1],
            [32 * sg, 1, 1],
        );
    }

    /// Tres [`Kernels::gemv_scaled`] con la misma entrada en un dispatch (q, k y v de una capa).
    /// q4_0, filas de cada matriz múltiplo de 8. Mismos bits que las tres llamadas separadas.
    pub fn gemv_scaled3<'a>(
        &self,
        cmd: &mut Command<'a>,
        w: [QMatrix<'a>; 3],
        xw: Arg<'a>,
        ss: Arg<'a>,
        eps: f32,
        y: [Arg<'a>; 3],
    ) {
        let cols = w[0].cols;
        for m in &w {
            assert_eq!(m.qtype, WeightType::Q4_0, "gemv_scaled3: q4_0");
            assert_eq!(m.cols, cols, "gemv_scaled3: misma entrada");
            assert_eq!(m.rows % 8, 0, "gemv_scaled3: filas % 8");
        }
        let total = w[0].rows + w[1].rows + w[2].rows;
        // Cada threadgroup trabaja sobre una sola matriz: las filas de cada una, múltiplo del
        // bloque de filas del threadgroup.
        let (mut p, mut sg) = self.gemv_pick(GemvOp::Scaled3, total, cols, &self.gemv_scaled3_q4_0);
        if w.iter().any(|m| m.rows % (GEMV_NR * sg) != 0) {
            (p, sg) = (&self.gemv_scaled3_q4_0, GEMV_SG_DEFAULT);
        }
        let [y0, y1, y2] = y;
        cmd.dispatch_groups(
            p,
            &[
                Arg::buf(w[0].data),
                Arg::buf(w[1].data),
                Arg::buf(w[2].data),
                xw,
                y0,
                y1,
                y2,
                Arg::u32(w[0].rows as u32),
                Arg::u32(w[1].rows as u32),
                Arg::u32(cols as u32),
                ss,
                Arg::u32(norm_partials(cols) as u32),
                Arg::f32(eps),
            ],
            [total / (GEMV_NR * sg), 1, 1],
            [32 * sg, 1, 1],
        );
    }

    /// `y = silu(Wg · n) · (Wu · n)` con `n = RMSNorm(x)` desde [`Kernels::add_norm_prep`], en un
    /// dispatch. q4_0, filas múltiplo de 8. Mismos bits que dos `gemv_scaled` más `swiglu`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_scaled_swiglu<'a>(
        &self,
        cmd: &mut Command<'a>,
        gate: QMatrix<'a>,
        up: QMatrix<'a>,
        xw: Arg<'a>,
        ss: Arg<'a>,
        eps: f32,
        y: Arg<'a>,
    ) {
        assert!(gate.qtype == WeightType::Q4_0 && up.qtype == WeightType::Q4_0);
        assert!(gate.rows == up.rows && gate.cols == up.cols && gate.rows % 8 == 0);
        let (p, sg) = self.gemv_pick(
            GemvOp::ScaledSwiglu,
            gate.rows,
            gate.cols,
            &self.gemv_scaled_swiglu_q4_0,
        );
        cmd.dispatch_groups(
            p,
            &[
                Arg::buf(gate.data),
                Arg::buf(up.data),
                xw,
                y,
                Arg::u32(gate.cols as u32),
                ss,
                Arg::u32(norm_partials(gate.cols) as u32),
                Arg::f32(eps),
            ],
            [gate.rows / (GEMV_NR * sg), 1, 1],
            [32 * sg, 1, 1],
        );
    }

    pub fn gemv<'a>(
        &self,
        cmd: &mut Command<'a>,
        w: QMatrix<'a>,
        x: Arg<'a>,
        y: Arg<'a>,
        tokens: usize,
    ) {
        if w.rows % 8 != 0 {
            return self.gemv_simple(cmd, w, x, y, tokens);
        }
        let p = match w.qtype {
            WeightType::Q4_0 => &self.gemv_fast_q4_0,
            WeightType::Q8_0 => &self.gemv_fast_q8_0,
            WeightType::Q6_0 => &self.gemv_fast_q6_0,
        };
        let (p, sg) = self.gemv_pick(GemvOp::Fast(w.qtype), w.rows, w.cols, p);
        cmd.dispatch_groups(
            p,
            &[
                Arg::buf(w.data),
                x,
                y,
                Arg::u32(w.rows as u32),
                Arg::u32(w.cols as u32),
            ],
            [w.rows / (GEMV_NR * sg), tokens, 1],
            [32 * sg, 1, 1],
        );
    }

    /// GEMV simple: un simdgroup por fila (respaldo y referencia de rendimiento).
    pub fn gemv_simple<'a>(
        &self,
        cmd: &mut Command<'a>,
        w: QMatrix<'a>,
        x: Arg<'a>,
        y: Arg<'a>,
        tokens: usize,
    ) {
        let p = match w.qtype {
            WeightType::Q4_0 => &self.gemv_q4_0,
            WeightType::Q8_0 => &self.gemv_q8_0,
            WeightType::Q6_0 => &self.gemv_q6_0,
        };
        cmd.dispatch_groups(
            p,
            &[
                Arg::buf(w.data),
                x,
                y,
                Arg::u32(w.rows as u32),
                Arg::u32(w.cols as u32),
            ],
            [groups(w.rows, GEMV_ROWS_PER_TG), tokens, 1],
            [32 * GEMV_ROWS_PER_TG, 1, 1],
        );
    }

    /// GEMM para prefill: `y[t, :] = W · x[t, :]`. Usa el kernel tiled (simdgroup matrix) si
    /// `rows % 64 == 0` y `cols % 32 == 0`; si no, la versión simple.
    pub fn gemm<'a>(
        &self,
        cmd: &mut Command<'a>,
        w: QMatrix<'a>,
        x: Arg<'a>,
        y: Arg<'a>,
        tokens: usize,
    ) {
        if w.rows % 64 != 0 || w.cols % 32 != 0 || w.qtype == WeightType::Q6_0 {
            return self.gemm_naive(cmd, w, x, y, tokens);
        }
        let p = match w.qtype {
            WeightType::Q4_0 => &self.gemm_tiled_q4_0,
            WeightType::Q8_0 => &self.gemm_tiled_q8_0,
            WeightType::Q6_0 => unreachable!(),
        };
        cmd.dispatch_groups(
            p,
            &[
                Arg::buf(w.data),
                x,
                y,
                Arg::u32(w.rows as u32),
                Arg::u32(w.cols as u32),
                Arg::u32(tokens as u32),
            ],
            [w.rows / 64, groups(tokens, 32), 1],
            [128, 1, 1],
        );
    }

    /// GEMM simple: un hilo por salida, sin tiling (referencia de rendimiento y respaldo).
    pub fn gemm_naive<'a>(
        &self,
        cmd: &mut Command<'a>,
        w: QMatrix<'a>,
        x: Arg<'a>,
        y: Arg<'a>,
        tokens: usize,
    ) {
        let p = match w.qtype {
            WeightType::Q4_0 => &self.gemm_q4_0,
            WeightType::Q8_0 => &self.gemm_q8_0,
            WeightType::Q6_0 => &self.gemm_q6_0,
        };
        cmd.dispatch(
            p,
            &[
                Arg::buf(w.data),
                x,
                y,
                Arg::u32(w.rows as u32),
                Arg::u32(w.cols as u32),
            ],
            [w.rows, tokens, 1],
            [64, 1, 1],
        );
    }

    /// QK-norm, RoPE y escritura de K y V en la caché en un solo dispatch (T3.5). Mismos bits
    /// que `rms_norm` de q y k, `rope_neox` de q y k y `store_kv` de k y v. `q` y `k` se
    /// actualizan en el lugar; `k_dst`/`v_dst` apuntan a la caché desde `pos0`. head_dim 128.
    #[allow(clippy::too_many_arguments)]
    pub fn qk_norm_rope_store<'a>(
        &self,
        cmd: &mut Command<'a>,
        kv: KvType,
        qkv: [Arg<'a>; 3],
        norms: [Arg<'a>; 2],
        eps: f32,
        table: &'a RopeTable,
        dst: [Arg<'a>; 2],
        shape: AttnShape,
    ) {
        let AttnShape {
            tokens,
            hq,
            hkv,
            dim,
            pos0,
            ..
        } = shape;
        assert_eq!(dim, 128, "qk_norm_rope_store requiere head_dim 128");
        assert_eq!(dim, table.dim, "dimensión de la tabla RoPE");
        assert!(
            pos0 + tokens <= table.max_pos,
            "posición fuera de la tabla RoPE"
        );
        let [q, k, v] = qkv;
        let [qn, kn] = norms;
        let [kd, vd] = dst;
        cmd.dispatch_groups(
            &self.qk_norm_rope_store[kv.idx()],
            &[
                q,
                k,
                v,
                qn,
                kn,
                Arg::buf(&table.cos),
                Arg::buf(&table.sin),
                kd,
                vd,
                Arg::u32(hq as u32),
                Arg::u32(hkv as u32),
                Arg::u32(pos0 as u32),
                Arg::f32(eps),
            ],
            [tokens * (hq + 2 * hkv), 1, 1],
            [dim, 1, 1],
        );
    }

    /// Copia `n` floats de `src` a la KV cache `dst` (de tipo `kv`), convirtiendo si hace falta.
    /// En Q8 `n` es múltiplo de 128 (filas completas) y hay un hilo por bloque de 32.
    pub fn store_kv<'a>(
        &self,
        cmd: &mut Command<'a>,
        kv: KvType,
        src: Arg<'a>,
        dst: Arg<'a>,
        n: usize,
    ) {
        let threads = match kv {
            KvType::Q8_0 => {
                assert_eq!(n % KV_Q8_DIM, 0, "KV Q8: filas de {KV_Q8_DIM} valores");
                n / 32
            }
            _ => n,
        };
        cmd.dispatch(
            &self.store_kv[kv.idx()],
            &[src, dst, Arg::u32(n as u32)],
            [threads, 1, 1],
            [ELEMENTWISE_TG, 1, 1],
        );
    }

    /// Atención de decode (un token en la posición `pos0`) con las claves repartidas en tramos
    /// de `DECODE_CHUNK`. `partials` debe tener al menos `decode_partials_len(hq, pos0 + 1)`
    /// floats. `head_dim` debe ser 128.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_attention<'a>(
        &self,
        cmd: &mut Command<'a>,
        q: Arg<'a>,
        k: Arg<'a>,
        v: Arg<'a>,
        partials: &'a Buffer<f32>,
        o: Arg<'a>,
        shape: AttnShape,
    ) {
        let AttnShape {
            tokens,
            hq,
            hkv,
            dim,
            pos0,
            kv,
        } = shape;
        assert_eq!(tokens, 1, "decode_attention es para un token");
        assert_eq!(dim, 128, "decode_attention requiere head_dim 128");
        let lk = pos0 + 1;
        assert!(
            partials.len() >= decode_partials_len(hq, lk),
            "scratch de decode chico"
        );
        let scale = 1.0 / (dim as f32).sqrt();
        cmd.dispatch_groups(
            &self.attn_decode_partial[kv.idx()],
            &[
                q,
                k,
                v,
                Arg::buf(partials),
                Arg::u32(hq as u32),
                Arg::u32(hkv as u32),
                Arg::u32(lk as u32),
                Arg::f32(scale),
            ],
            [groups(lk, DECODE_CHUNK), hq, 1],
            [128, 1, 1],
        );
        cmd.dispatch_groups(
            &self.attn_decode_reduce,
            &[Arg::buf(partials), o, Arg::u32(lk as u32)],
            [hq, 1, 1],
            [dim, 1, 1],
        );
    }

    /// Atención de decode "una lane por clave": cada simdgroup atiende un tramo de claves para
    /// todas las cabezas de query de un grupo GQA, leyendo K y V una vez por grupo. El grupo
    /// (`hq / hkv`) tiene que estar en [`GQA_GROUPS`] (ver [`gqa_supported`]).
    /// Mismo scratch y reducción que `decode_attention`.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_attention_lanes<'a>(
        &self,
        cmd: &mut Command<'a>,
        q: Arg<'a>,
        k: Arg<'a>,
        v: Arg<'a>,
        partials: &'a Buffer<f32>,
        o: Arg<'a>,
        shape: AttnShape,
    ) {
        let AttnShape {
            tokens,
            hq,
            hkv,
            dim,
            pos0,
            kv,
        } = shape;
        assert_eq!(tokens, 1, "decode_attention es para un token");
        assert_eq!(dim, 128, "decode_attention requiere head_dim 128");
        let lk = pos0 + 1;
        assert!(
            partials.len() >= decode_partials_len(hq, lk),
            "scratch de decode chico"
        );
        let gi = gqa_index(hq, hkv).expect("decode_attention_lanes: grupo GQA no compilado");
        let base = &self.attn_decode_lanes[gi][kv.idx()];
        let (p, sg) = match self.launch.attn_lanes_sg(kv, GQA_GROUPS[gi], lk) {
            LANES_SG_DEFAULT => (base, LANES_SG_DEFAULT),
            sg => self
                .lanes_variant(gi, kv, sg)
                .map_or((base, LANES_SG_DEFAULT), |p| (p, sg)),
        };
        let scale = 1.0 / (dim as f32).sqrt();
        cmd.dispatch_groups(
            p,
            &[
                q,
                k,
                v,
                Arg::buf(partials),
                Arg::u32(hkv as u32),
                Arg::u32(lk as u32),
                Arg::f32(scale),
            ],
            [groups(groups(lk, DECODE_CHUNK), sg), hkv, 1],
            [32 * sg, 1, 1],
        );
        cmd.dispatch_groups(
            &self.attn_decode_reduce,
            &[Arg::buf(partials), o, Arg::u32(lk as u32)],
            [hq, 1, 1],
            [dim, 1, 1],
        );
    }

    /// Atención causal con GQA estilo FlashAttention (softmax online, sin scratch de puntajes).
    /// `head_dim` debe ser 128. La caché `k`/`v` debe poder leerse hasta `KV_ALIGN` posiciones
    /// más allá de `pos0 + tokens` (las posiciones fuera de rango se enmascaran).
    pub fn flash_attention<'a>(
        &self,
        cmd: &mut Command<'a>,
        q: Arg<'a>,
        k: Arg<'a>,
        v: Arg<'a>,
        o: Arg<'a>,
        shape: AttnShape,
    ) {
        let AttnShape {
            tokens,
            hq,
            hkv,
            dim,
            pos0,
            kv,
        } = shape;
        assert_eq!(dim, 128, "flash_attention requiere head_dim 128");
        assert!(hq % hkv == 0, "hq debe ser múltiplo de hkv");
        let scale = 1.0 / (dim as f32).sqrt();
        if let Some(gi) = gqa_index(hq, hkv) {
            // Variante GQA: FA_ROWS filas (FA_ROWS / grupo queries × grupo cabezas) por threadgroup.
            let qt = FA_ROWS / GQA_GROUPS[gi];
            cmd.dispatch_groups(
                &self.flash_attn_gqa[gi][kv.idx()],
                &[
                    q,
                    k,
                    v,
                    o,
                    Arg::u32(tokens as u32),
                    Arg::u32(hkv as u32),
                    Arg::u32(pos0 as u32),
                    Arg::f32(scale),
                ],
                [groups(tokens, qt), hkv, 1],
                [128, 1, 1],
            );
            return;
        }
        cmd.dispatch_groups(
            &self.flash_attn[kv.idx()],
            &[
                q,
                k,
                v,
                o,
                Arg::u32(tokens as u32),
                Arg::u32(hq as u32),
                Arg::u32(hkv as u32),
                Arg::u32(pos0 as u32),
                Arg::f32(scale),
            ],
            [groups(tokens, 32), hq, 1],
            [128, 1, 1],
        );
    }

    /// Atención causal con GQA (simple, sin tiling). `q: [tokens, hq, dim]`; `k`, `v`: caché de
    /// la capa `[max_pos, hkv, dim]` con las posiciones `0..pos0+tokens` ya escritas; `scores`:
    /// scratch de al menos `tokens · hq · (pos0 + tokens)` floats; `o: [tokens, hq, dim]`.
    #[allow(clippy::too_many_arguments)]
    pub fn attention<'a>(
        &self,
        cmd: &mut Command<'a>,
        q: Arg<'a>,
        k: Arg<'a>,
        v: Arg<'a>,
        scores: &'a Buffer<f32>,
        o: Arg<'a>,
        shape: AttnShape,
    ) {
        let AttnShape {
            tokens,
            hq,
            hkv,
            dim,
            pos0,
            kv,
        } = shape;
        assert!(hq % hkv == 0, "hq debe ser múltiplo de hkv");
        assert_eq!(kv, KvType::F32, "attention (simple) solo con KV f32");
        let lk = pos0 + tokens;
        assert!(
            scores.len() >= tokens * hq * lk,
            "scratch de atención chico"
        );
        let scale = 1.0 / (dim as f32).sqrt();
        // Arreglos fijos: este camino corre en cada paso de decode y no debe asignar.
        let [a, b, c, d, e] = [hq, hkv, dim, pos0, lk].map(|x| Arg::u32(x as u32));
        cmd.dispatch(
            &self.attn_scores_f32,
            &[q, k, Arg::buf(scores), a, b, c, d, e, Arg::f32(scale)],
            [lk, hq, tokens],
            [64, 1, 1],
        );
        cmd.dispatch_groups(
            &self.softmax_f32,
            &[Arg::buf(scores), Arg::buf(scores), Arg::u32(lk as u32)],
            [tokens * hq, 1, 1],
            [ROW_TG, 1, 1],
        );
        cmd.dispatch(
            &self.attn_pv_f32,
            &[Arg::buf(scores), v, o, a, b, c, d, e],
            [dim, hq, tokens],
            [dim.min(128), 1, 1],
        );
    }
}

/// Tamaños de grupo GQA (`hq / hkv`) para los que se compilan `decode_attention_lanes` y la
/// variante GQA de `flash_attention`.
pub const GQA_GROUPS: [usize; 4] = [1, 2, 4, 8];

/// Si hay kernels compilados para el grupo GQA de `hq` cabezas de query y `hkv` de KV.
pub fn gqa_supported(hq: usize, hkv: usize) -> bool {
    gqa_index(hq, hkv).is_some()
}

/// Índice en [`GQA_GROUPS`] del grupo de `hq / hkv`.
fn gqa_index(hq: usize, hkv: usize) -> Option<usize> {
    if hkv == 0 || hq % hkv != 0 {
        return None;
    }
    GQA_GROUPS.iter().position(|&g| g == hq / hkv)
}

/// Claves por tramo de `decode_attention` (`CHUNK` en MSL).
pub const DECODE_CHUNK: usize = 128;

/// Floats del scratch de parciales de `decode_attention` para `hq` cabezas y `lk` claves.
pub fn decode_partials_len(hq: usize, lk: usize) -> usize {
    hq * lk.div_ceil(DECODE_CHUNK) * 130
}

/// Filas (queries × cabezas del grupo) por threadgroup de la variante GQA de `flash_attention`
/// (`FA_ROWS` en MSL).
const FA_ROWS: usize = 16;

/// Hilos por threadgroup de `add_norm_prep` (cada uno escribe una suma parcial).
const NORM_PREP_TG: usize = 256;

/// Sumas parciales de `x²` que escribe `add_norm_prep` para un vector de `n` elementos.
pub fn norm_partials(n: usize) -> usize {
    n.div_ceil(NORM_PREP_TG)
}

/// Alineación (en posiciones) que necesita la KV cache para `flash_attention`.
pub const KV_ALIGN: usize = 64;

/// Forma de una llamada de atención.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttnShape {
    pub tokens: usize,
    pub hq: usize,
    pub hkv: usize,
    pub dim: usize,
    /// Posición absoluta del primer token de `q`.
    pub pos0: usize,
    /// Tipo de la KV cache que leen `k` y `v`.
    pub kv: KvType,
}

/// Tabla cos/sin de RoPE `[max_pos, dim/2]`, calculada en f64 y guardada en f32 (ADR 0004: la
/// precisión de `pos · inv_freq` en f32 no alcanza para posiciones grandes).
#[derive(Debug)]
pub struct RopeTable {
    pub cos: Buffer<f32>,
    pub sin: Buffer<f32>,
    pub dim: usize,
    pub max_pos: usize,
}

impl RopeTable {
    pub fn new(ctx: &Context, theta: f64, dim: usize, max_pos: usize) -> Result<Self, MetalError> {
        let (c, s) = rope_table(theta, dim, max_pos);
        Ok(Self {
            cos: ctx.buffer_from(&c)?,
            sin: ctx.buffer_from(&s)?,
            dim,
            max_pos,
        })
    }
}

/// `cos/sin(p · θ^(-2i/dim))` para `p < max_pos`, `i < dim/2`, en f64.
pub fn rope_table(theta: f64, dim: usize, max_pos: usize) -> (Vec<f32>, Vec<f32>) {
    let half = dim / 2;
    let inv: Vec<f64> = (0..half)
        .map(|i| 1.0 / theta.powf((2 * i) as f64 / dim as f64))
        .collect();
    let mut cos = Vec::with_capacity(max_pos * half);
    let mut sin = Vec::with_capacity(max_pos * half);
    for p in 0..max_pos {
        for f in &inv {
            let a = p as f64 * f;
            cos.push(a.cos() as f32);
            sin.push(a.sin() as f32);
        }
    }
    (cos, sin)
}
