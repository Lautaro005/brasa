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

use brasa_metal::{Arg, Buffer, Command, Context, MetalError, Pipeline};

pub mod reference;
pub mod testutil;

/// Fuentes MSL embebidas en el binario.
pub mod sources {
    pub const ELEMENTWISE: &str = include_str!("metal/elementwise.metal");
    pub const NORM: &str = include_str!("metal/norm.metal");
    pub const SOFTMAX: &str = include_str!("metal/softmax.metal");
    pub const ROPE: &str = include_str!("metal/rope.metal");
    pub const EMBED: &str = include_str!("metal/embed.metal");
    pub const MATMUL: &str = include_str!("metal/matmul.metal");
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
    gemv_q4_0: Pipeline,
    gemv_q8_0: Pipeline,
    gemm_q4_0: Pipeline,
    gemm_q8_0: Pipeline,
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
            gemv_q4_0: ctx.pipeline(MATMUL, "gemv_q4_0_f32")?,
            gemv_q8_0: ctx.pipeline(MATMUL, "gemv_q8_0_f32")?,
            gemm_q4_0: ctx.pipeline(MATMUL, "gemm_q4_0_f32")?,
            gemm_q8_0: ctx.pipeline(MATMUL, "gemm_q8_0_f32")?,
        })
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

    /// Embedding desde una tabla q8_0 `[vocab, h]`: `out[t, :] = tabla[ids[t], :]`.
    pub fn embed<'a>(
        &self,
        cmd: &mut Command<'a>,
        table: QMatrix<'a>,
        ids: &'a Buffer<u32>,
        out: Arg<'a>,
        tokens: usize,
    ) {
        assert_eq!(table.qtype, WeightType::Q8_0, "embedding solo en q8_0");
        assert!(ids.len() >= tokens);
        cmd.dispatch(
            &self.embed_q8_0,
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

    /// GEMV: `y[t, :] = W · x[t, :]` para `t < tokens`, un simdgroup por fila (decode, T chico).
    pub fn gemv<'a>(
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

    /// GEMM simple: `y[t, :] = W · x[t, :]`, un hilo por salida (prefill; sin tiling todavía).
    pub fn gemm<'a>(
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
