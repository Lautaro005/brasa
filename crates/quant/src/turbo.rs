//! KV cache TurboQuant (ADR 0033): especificación en CPU de la cuantización de una fila de
//! [`TQ_DIM`] valores a 4 bits, la que siguen `tq_store_tq4` y la referencia.
//!
//! Por fila (un token y una cabeza KV), en el orden de TurboQuant (Zandieh et al., 2025):
//! 1. `norm = ‖x‖` (guardada en f16);
//! 2. `y = R · x`, con `R` una rotación ortogonal fija (la misma para todas las capas y cabezas);
//! 3. cada coordenada de `y / norm` se cuantiza con un codebook de Lloyd-Max de 16 niveles,
//!    diseñado para la distribución de una coordenada de un vector unitario de 128 dimensiones
//!    (la Beta de TurboQuant), y se guarda el índice de 4 bits.
//!
//! Fila de 68 bytes: 64 bytes con los índices (dimensión `2b` en el nibble bajo del byte `b`, y
//! `2b + 1` en el alto), la norma f16 en los bytes 64..66 y 2 bytes de relleno en cero.
//!
//! La atención trabaja en el dominio rotado: `q·k = (R q)·(R k)`, así que la caché guarda `y`
//! y el decode rota `q` una vez por cabeza (y la salida de vuelta con `Rᵀ`). Por eso
//! [`dequantize_row`] devuelve `norm · codebook[c]` en el dominio rotado.
//!
//! Desviación del paper: solo la etapa MSE. La etapa QJL (1 bit sobre el residuo, para un
//! producto interno sin sesgo) no está; ver ADR 0033.

use std::sync::OnceLock;

use crate::qtype::{f16_to_f32, f32_to_f16};

/// Dimensión de cabeza de la fila cuantizada (head_dim de Qwen3).
pub const TQ_DIM: usize = 128;
/// Bits por coordenada.
pub const TQ_BITS: usize = 4;
/// Niveles del codebook.
pub const TQ_LEVELS: usize = 1 << TQ_BITS;
/// Bytes por fila: 64 de índices, norma f16 y 2 de relleno (4 bytes alineados).
pub const KV_TQ4_ROW: usize = 68;
/// Bytes de índices por fila.
const CODES_BYTES: usize = TQ_DIM / 2;
/// Semilla de la rotación: fija, para que la caché sea reproducible entre corridas.
const ROTATION_SEED: u64 = 0x7442_7261_7361_0033;
/// Puntos de la malla que integra el codebook.
const GRID: usize = 1 << 16;
/// Iteraciones máximas de Lloyd; converge mucho antes con esta densidad.
const LLOYD_ITERS: usize = 500;

/// Codebook de 16 niveles y sus 15 umbrales (puntos medios entre niveles consecutivos).
#[derive(Debug, Clone, PartialEq)]
pub struct Codebook {
    pub centroids: [f32; TQ_LEVELS],
    pub mids: [f32; TQ_LEVELS - 1],
}

/// Codebook de Lloyd-Max para la Beta de una coordenada de un vector unitario de [`TQ_DIM`]
/// dimensiones. Se calcula una vez (determinista, en f64).
pub fn codebook() -> &'static Codebook {
    static CB: OnceLock<Codebook> = OnceLock::new();
    CB.get_or_init(|| lloyd_max(TQ_DIM, TQ_LEVELS))
}

/// Lloyd-Max sobre la densidad `f(x) ∝ (1 − x²)^((d − 3) / 2)` en `[−1, 1]`, con la regla del
/// punto medio en una malla de [`GRID`] puntos. Como los centroides quedan ordenados, cada
/// iteración recorre la malla una sola vez.
fn lloyd_max(d: usize, levels: usize) -> Codebook {
    let expo = (d as f64 - 3.0) / 2.0;
    let xs: Vec<f64> = (0..GRID)
        .map(|i| -1.0 + (2.0 * i as f64 + 1.0) / GRID as f64)
        .collect();
    let ws: Vec<f64> = xs.iter().map(|x| (1.0 - x * x).powf(expo)).collect();
    let mut c: Vec<f64> = (0..levels)
        .map(|k| -1.0 + 2.0 * (k as f64 + 0.5) / levels as f64)
        .collect();
    for _ in 0..LLOYD_ITERS {
        let mut num = vec![0f64; levels];
        let mut den = vec![0f64; levels];
        let mut k = 0;
        for (x, w) in xs.iter().zip(&ws) {
            while k + 1 < levels && *x > 0.5 * (c[k] + c[k + 1]) {
                k += 1;
            }
            num[k] += w * x;
            den[k] += w;
        }
        let mut moved = 0f64;
        for k in 0..levels {
            if den[k] > 0.0 {
                let nc = num[k] / den[k];
                moved = moved.max((nc - c[k]).abs());
                c[k] = nc;
            }
        }
        if moved < 1e-13 {
            break;
        }
    }
    let mut cb = Codebook {
        centroids: [0.0; TQ_LEVELS],
        mids: [0.0; TQ_LEVELS - 1],
    };
    for (dst, src) in cb.centroids.iter_mut().zip(&c) {
        *dst = *src as f32;
    }
    for (k, m) in cb.mids.iter_mut().enumerate() {
        *m = (0.5 * (c[k] + c[k + 1])) as f32;
    }
    cb
}

/// Rotación ortogonal fija de [`TQ_DIM`] × [`TQ_DIM`] en f32, fila mayor (`R[i·D + j]`): la
/// base ortonormal de una matriz gaussiana de semilla fija (Gram-Schmidt en f64).
pub fn rotation() -> Vec<f32> {
    let d = TQ_DIM;
    let mut rng = SplitMix(ROTATION_SEED);
    // Columnas gaussianas (Box-Muller sobre SplitMix64).
    let mut cols: Vec<Vec<f64>> = (0..d)
        .map(|_| (0..d).map(|_| rng.gauss()).collect())
        .collect();
    for j in 0..d {
        let (done, rest) = cols.split_at_mut(j);
        let cj = &mut rest[0];
        for ck in done.iter() {
            let dot: f64 = cj.iter().zip(ck).map(|(a, b)| a * b).sum();
            for (a, b) in cj.iter_mut().zip(ck) {
                *a -= dot * b;
            }
        }
        let n: f64 = cj.iter().map(|v| v * v).sum::<f64>().sqrt();
        for v in cj.iter_mut() {
            *v /= n;
        }
    }
    let mut r = vec![0f32; d * d];
    for i in 0..d {
        for j in 0..d {
            r[i * d + j] = cols[j][i] as f32;
        }
    }
    r
}

struct SplitMix(u64);

impl SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniforme en (0, 1].
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    }

    fn gauss(&mut self) -> f64 {
        let (u1, u2) = (self.unit(), self.unit());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Cuantiza una fila de [`TQ_DIM`] valores en `out` ([`KV_TQ4_ROW`] bytes), con la rotación `rot`
/// (ver [`rotation`]). Es la especificación de `tq_store_tq4`.
pub fn quantize_row(x: &[f32], rot: &[f32], out: &mut [u8]) {
    assert_eq!(x.len(), TQ_DIM, "fila de {TQ_DIM} valores");
    assert_eq!(rot.len(), TQ_DIM * TQ_DIM, "rotación de {TQ_DIM}×{TQ_DIM}");
    assert_eq!(out.len(), KV_TQ4_ROW, "fila de {KV_TQ4_ROW} bytes");
    let cb = codebook();
    let norm2: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let nh = f32_to_f16(norm2.sqrt() as f32);
    let nf = f16_to_f32(nh);
    out.fill(0);
    for i in 0..TQ_DIM {
        let y: f64 = (0..TQ_DIM)
            .map(|j| rot[i * TQ_DIM + j] as f64 * x[j] as f64)
            .sum();
        let u = if nf == 0.0 {
            0.0
        } else {
            (y / nf as f64) as f32
        };
        let code = cb.mids.iter().filter(|&&m| u > m).count() as u8;
        out[i / 2] |= code << (4 * (i & 1));
    }
    out[CODES_BYTES..CODES_BYTES + 2].copy_from_slice(&nh.to_le_bytes());
}

/// Decuantiza una fila al dominio rotado: `out[i] = norm · codebook[c_i]` (ver el módulo).
pub fn dequantize_row(row: &[u8], out: &mut [f32]) {
    assert_eq!(row.len(), KV_TQ4_ROW, "fila de {KV_TQ4_ROW} bytes");
    assert_eq!(out.len(), TQ_DIM, "fila de {TQ_DIM} valores");
    let cb = codebook();
    let nf = f16_to_f32(u16::from_le_bytes([row[CODES_BYTES], row[CODES_BYTES + 1]]));
    for (i, o) in out.iter_mut().enumerate() {
        let b = row[i / 2];
        let c = (if i & 1 == 0 { b & 15 } else { b >> 4 }) as usize;
        *o = nf * cb.centroids[c];
    }
}

/// Decuantiza una fila al dominio original: `x̂ = Rᵀ · (norm · codebook[c])`.
pub fn dequantize_row_original(row: &[u8], rot: &[f32], out: &mut [f32]) {
    let mut y = vec![0f32; TQ_DIM];
    dequantize_row(row, &mut y);
    for (j, o) in out.iter_mut().enumerate() {
        *o = (0..TQ_DIM)
            .map(|i| rot[i * TQ_DIM + j] as f64 * y[i] as f64)
            .sum::<f64>() as f32;
    }
}

/// Bytes de `n` valores de la caché TQ4 (`n` múltiplo de [`TQ_DIM`]: filas completas).
pub fn tq4_bytes(n: usize) -> usize {
    assert_eq!(n % TQ_DIM, 0, "KV TQ4: filas de {TQ_DIM} valores");
    n / TQ_DIM * KV_TQ4_ROW
}

/// Declaraciones MSL de los constantes del codebook (`TQ_CB`, `TQ_MID`). El host las antepone a
/// las fuentes que las usan, así el kernel y la especificación comparten los mismos valores
/// (floats f32 con el número exacto, formato `{:e}`, que ida y vuelta conserva los bits).
pub fn metal_constants() -> String {
    let cb = codebook();
    let list = |v: &[f32]| {
        v.iter()
            .map(|x| format!("{x:e}f"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    format!(
        "constant float TQ_CB[{TQ_LEVELS}] = {{{}}};\nconstant float TQ_MID[{}] = {{{}}};\n",
        list(&cb.centroids),
        TQ_LEVELS - 1,
        list(&cb.mids),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_es_simetrico_y_creciente() {
        let cb = codebook();
        for k in 0..TQ_LEVELS - 1 {
            assert!(cb.centroids[k] < cb.centroids[k + 1]);
            assert!(cb.mids[k] > cb.centroids[k] && cb.mids[k] < cb.centroids[k + 1]);
        }
        for k in 0..TQ_LEVELS / 2 {
            let (a, b) = (cb.centroids[k], cb.centroids[TQ_LEVELS - 1 - k]);
            assert!((a + b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn codebook_coincide_con_lloyd_max_gaussiano() {
        // Lloyd-Max de 4 bits para N(0, 1) (Max, 1960): el nivel más alto es 2,7326σ. Con
        // σ = 1/√d la Beta de d = 128 es casi gaussiana, así que debe caer a menos de 2 %.
        let top = codebook().centroids[TQ_LEVELS - 1] as f64 * (TQ_DIM as f64).sqrt();
        assert!(
            (top - 2.7326).abs() / 2.7326 < 0.02,
            "nivel alto × √d = {top}"
        );
    }

    #[test]
    fn rotacion_es_ortogonal() {
        let r = rotation();
        let d = TQ_DIM;
        for i in 0..d {
            for j in 0..d {
                let dot: f64 = (0..d)
                    .map(|k| r[i * d + k] as f64 * r[j * d + k] as f64)
                    .sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-5, "R Rᵀ[{i}][{j}] = {dot}");
            }
        }
        assert_eq!(rotation(), r, "la rotación es determinista");
    }

    #[test]
    fn error_cuadratico_relativo_cerca_del_teorico() {
        // Distorsión de Lloyd-Max de 4 bits para la gaussiana: 0,009497 σ². Con la rotación,
        // el error relativo de un vector gaussiano debe quedar cerca de ese valor.
        let r = rotation();
        let mut rng = SplitMix(7);
        let (mut err, mut tot) = (0f64, 0f64);
        for _ in 0..200 {
            let x: Vec<f32> = (0..TQ_DIM).map(|_| rng.gauss() as f32).collect();
            let mut row = [0u8; KV_TQ4_ROW];
            quantize_row(&x, &r, &mut row);
            let mut xh = vec![0f32; TQ_DIM];
            dequantize_row_original(&row, &r, &mut xh);
            for (a, b) in x.iter().zip(&xh) {
                err += ((a - b) as f64).powi(2);
                tot += (*a as f64).powi(2);
            }
        }
        let rel = err / tot;
        assert!((rel - 0.009497).abs() < 0.001, "error relativo {rel}");
    }

    #[test]
    fn fila_ida_y_vuelta_en_el_dominio_rotado() {
        let r = rotation();
        let mut rng = SplitMix(11);
        let x: Vec<f32> = (0..TQ_DIM).map(|_| rng.gauss() as f32 * 3.0).collect();
        let mut row = [0u8; KV_TQ4_ROW];
        quantize_row(&x, &r, &mut row);
        // La norma se guarda en f16 y el resto es el codebook. Por vector, ‖ŷ‖/‖x‖ se desvía
        // ~2√(D/d) ≈ 1,7 % (la componente radial del error, que es el sesgo que corrige QJL):
        // la cota es de 5 %, unas 3 σ.
        let mut y = vec![0f32; TQ_DIM];
        dequantize_row(&row, &mut y);
        let nx: f64 = x.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        let ny: f64 = y.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        assert!((ny - nx).abs() / nx < 0.05, "‖ŷ‖ {ny} vs ‖x‖ {nx}");
        // Los bytes de relleno quedan en cero.
        assert_eq!(&row[66..68], &[0, 0]);
    }

    #[test]
    fn fila_nula_da_ceros() {
        let r = rotation();
        let mut row = [0xffu8; KV_TQ4_ROW];
        quantize_row(&vec![0f32; TQ_DIM], &r, &mut row);
        let mut y = vec![1f32; TQ_DIM];
        dequantize_row(&row, &mut y);
        assert!(y.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn constantes_metal_tienen_los_valores_exactos() {
        let src = metal_constants();
        let cb = codebook();
        assert!(src.contains("TQ_CB[16]") && src.contains("TQ_MID[15]"));
        // Ida y vuelta del texto: cada valor se parsea al mismo f32.
        let first = format!("{:e}f", cb.centroids[0]);
        assert!(src.contains(&first), "{first} no está en {src}");
    }

    #[test]
    fn bytes_de_la_cache() {
        assert_eq!(tq4_bytes(1024), 8 * KV_TQ4_ROW);
        assert_eq!(
            KV_TQ4_ROW * 8 * 2 * 36,
            39168,
            "KV por token de Qwen3-4B (K y V, 36 capas)"
        );
    }
}
