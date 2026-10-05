//! Implementaciones de referencia en CPU. Acumulan en f64 para que la referencia sea más precisa
//! que el kernel; la tolerancia de cada test mide la distancia del kernel a este valor.

use brasa_quant::{QType, dequantize};

/// `out[i] = a[i] + b[i]`
pub fn add(a: &[f32], b: &[f32], out: &mut [f32]) {
    for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
        *o = x + y;
    }
}

/// SwiGLU: `out[i] = silu(gate[i]) · up[i]`.
pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for ((o, g), u) in out.iter_mut().zip(gate).zip(up) {
        let g = *g as f64;
        *o = (g / (1.0 + (-g).exp()) * *u as f64) as f32;
    }
}

/// RMSNorm por filas de largo `n`.
pub fn rms_norm(x: &[f32], w: &[f32], n: usize, eps: f32, out: &mut [f32]) {
    for (xr, or) in x.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
        let ss: f64 = xr.iter().map(|v| (*v as f64).powi(2)).sum();
        let scale = 1.0 / (ss / n as f64 + eps as f64).sqrt();
        for ((o, v), wi) in or.iter_mut().zip(xr).zip(w) {
            *o = (*v as f64 * scale * *wi as f64) as f32;
        }
    }
}

/// Softmax por filas de largo `n`.
pub fn softmax(x: &[f32], n: usize, out: &mut [f32]) {
    for (xr, or) in x.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
        let m = xr.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b)) as f64;
        let s: f64 = xr.iter().map(|v| (*v as f64 - m).exp()).sum();
        for (o, v) in or.iter_mut().zip(xr) {
            *o = ((*v as f64 - m).exp() / s) as f32;
        }
    }
}

/// RoPE NeoX en el lugar sobre `x: [T, heads, dim]`, posiciones desde `pos0`, con la misma tabla
/// cos/sin `[max_pos, dim/2]` que usa el kernel.
pub fn rope_neox(x: &mut [f32], heads: usize, dim: usize, pos0: usize, cos: &[f32], sin: &[f32]) {
    let half = dim / 2;
    for (t, tok) in x.chunks_exact_mut(heads * dim).enumerate() {
        let p = pos0 + t;
        for v in tok.chunks_exact_mut(dim) {
            for i in 0..half {
                let (c, s) = (cos[p * half + i] as f64, sin[p * half + i] as f64);
                let (a, b) = (v[i] as f64, v[i + half] as f64);
                v[i] = (a * c - b * s) as f32;
                v[i + half] = (b * c + a * s) as f32;
            }
        }
    }
}

/// `y[t, r] = Σ_k W[r, k] · x[t, k]` con `W` cuantizado `[rows, cols]`.
/// Devuelve además `Σ_k |W[r, k] · x[t, k]|` por salida, la escala de la tolerancia.
pub fn matmul(q: QType, w: &[u8], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) -> Vec<f32> {
    let t_count = x.len() / cols;
    let mut abs_sum = vec![0f32; t_count * rows];
    let row_bytes = q.nbytes(cols);
    let mut wr = vec![0f32; cols];
    for r in 0..rows {
        dequantize(q, &w[r * row_bytes..(r + 1) * row_bytes], &mut wr);
        for t in 0..t_count {
            let xt = &x[t * cols..(t + 1) * cols];
            let (mut s, mut a) = (0f64, 0f64);
            for (wi, xi) in wr.iter().zip(xt) {
                let p = *wi as f64 * *xi as f64;
                s += p;
                a += p.abs();
            }
            y[t * rows + r] = s as f32;
            abs_sum[t * rows + r] = a as f32;
        }
    }
    abs_sum
}

/// Embedding: `out[t, :] = decuantizar(table[ids[t], :])`.
pub fn embed(q: QType, table: &[u8], h: usize, ids: &[u32], out: &mut [f32]) {
    let row_bytes = q.nbytes(h);
    for (t, id) in ids.iter().enumerate() {
        let r = *id as usize;
        dequantize(
            q,
            &table[r * row_bytes..(r + 1) * row_bytes],
            &mut out[t * h..(t + 1) * h],
        );
    }
}

/// Atención causal con GQA en f64. `q: [T, hq, dim]`; `k`, `v`: `[pos0 + T, hkv, dim]`.
/// Devuelve `o: [T, hq, dim]` y `max_j |v[j, kh, d]|` por componente (escala de la tolerancia).
#[allow(clippy::too_many_arguments)]
pub fn attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    tokens: usize,
    hq: usize,
    hkv: usize,
    dim: usize,
    pos0: usize,
) -> (Vec<f32>, Vec<f32>) {
    let group = hq / hkv;
    let scale = 1.0 / (dim as f64).sqrt();
    let mut o = vec![0f32; tokens * hq * dim];
    let mut vmax = vec![0f32; tokens * hq * dim];
    for t in 0..tokens {
        let last = pos0 + t;
        for h in 0..hq {
            let kh = h / group;
            let qv = &q[(t * hq + h) * dim..][..dim];
            let s: Vec<f64> = (0..=last)
                .map(|j| {
                    let kv = &k[(j * hkv + kh) * dim..][..dim];
                    qv.iter()
                        .zip(kv)
                        .map(|(a, b)| *a as f64 * *b as f64)
                        .sum::<f64>()
                        * scale
                })
                .collect();
            let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let z: f64 = s.iter().map(|x| (x - m).exp()).sum();
            for d in 0..dim {
                let (mut acc, mut mx) = (0f64, 0f32);
                for (j, sj) in s.iter().enumerate() {
                    let vj = v[(j * hkv + kh) * dim + d];
                    acc += (sj - m).exp() / z * vj as f64;
                    mx = mx.max(vj.abs());
                }
                o[(t * hq + h) * dim + d] = acc as f32;
                vmax[(t * hq + h) * dim + d] = mx;
            }
        }
    }
    (o, vmax)
}
