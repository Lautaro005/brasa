//! Esquemas de cuantización y decuantización de referencia en CPU (ADR 0006).

use serde::{Deserialize, Serialize};

pub const BLOCK: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QType {
    /// Escala f16 + 16 bytes de nibbles por bloque de 32: `w = d · (q − 8)`.
    Q4_0,
    /// Escala f16 + 32 × i8 por bloque de 32: `w = d · q`.
    Q8_0,
    /// Escala f16 + 16 bytes de bits bajos + 8 de bits altos por bloque de 32 (ADR 0012):
    /// `w = d · (q − 32)`, q de 6 bits.
    Q6_0,
    F32,
}

impl QType {
    /// Bytes que ocupan `n` elementos.
    pub fn nbytes(self, n: usize) -> usize {
        match self {
            QType::Q4_0 => n / BLOCK * 18,
            QType::Q8_0 => n / BLOCK * 34,
            QType::Q6_0 => n / BLOCK * 26,
            QType::F32 => n * 4,
        }
    }
}

/// f16 (bits IEEE 754 binary16) a f32, exacto.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if mant == 0 => sign,
        0 => {
            // Subnormal: normalizar.
            let shift = mant.leading_zeros() - 21;
            let m = (mant << shift) & 0x3ff;
            sign | ((113 - shift) << 23) | (m << 13)
        }
        0x1f => sign | 0x7f80_0000 | (mant << 13),
        e => sign | ((e + 112) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
}

/// f32 a f16 (bits) con redondeo al par más cercano, como la conversión de Metal y `torch.half`.
/// Desborde a ±inf; NaN se mantiene NaN.
pub fn f32_to_f16(f: f32) -> u16 {
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let a = x & 0x7fff_ffff;
    if a >= 0x7f80_0000 {
        let nan = if a > 0x7f80_0000 { 0x200 } else { 0 };
        return sign | 0x7c00 | nan;
    }
    if a >= 0x4780_0000 {
        return sign | 0x7c00; // ≥ 65536: siempre inf
    }
    let (h, rem, half) = if a >= 0x3880_0000 {
        // Normal en f16 (≥ 2^-14). Un acarreo de la mantisa sube el exponente (o llega a inf).
        let h = (((a >> 23) - 112) << 10) | ((a & 0x7f_ffff) >> 13);
        (h, a & 0x1fff, 0x1000)
    } else if a < 0x3300_0000 {
        return sign; // < 2^-25: redondea a cero
    } else {
        // Subnormal en f16: valor / 2^-24 = mantisa · 2^(exp - 126).
        let shift = 126 - (a >> 23);
        let mant = (a & 0x7f_ffff) | 0x80_0000;
        (mant >> shift, mant & ((1 << shift) - 1), 1 << (shift - 1))
    };
    let up = rem > half || (rem == half && h & 1 == 1);
    sign | (h + up as u32) as u16
}

/// Valores por fila de la KV cache Q8 (una cabeza KV de un token, head_dim de Qwen3).
pub const KV_Q8_DIM: usize = 128;
/// Bytes por fila de la KV cache Q8: 128 `int8` y 4 escalas f16 (ADR 0009).
pub const KV_Q8_ROW: usize = 136;

/// Cuantiza filas de [`KV_Q8_DIM`] valores al formato de la KV cache Q8 (ADR 0009). Es la
/// especificación que siguen `store_kv_q8` y la referencia Python: por bloque de 32,
/// `d = f16(amax / 127)` y `q = clamp(rint(x / d), -127, 127)` (0 si d = 0), con divisiones IEEE y
/// redondeo al par.
pub fn quantize_kv_q8(x: &[f32], out: &mut [u8]) {
    assert_eq!(x.len() % KV_Q8_DIM, 0, "filas de {KV_Q8_DIM} valores");
    assert_eq!(out.len(), x.len() / KV_Q8_DIM * KV_Q8_ROW);
    for (row, o) in x
        .chunks_exact(KV_Q8_DIM)
        .zip(out.chunks_exact_mut(KV_Q8_ROW))
    {
        for (b, blk) in row.chunks_exact(BLOCK).enumerate() {
            let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
            let dh = f32_to_f16(amax / 127.0);
            let d = f16_to_f32(dh);
            for (j, &v) in blk.iter().enumerate() {
                let q = if d == 0.0 {
                    0.0
                } else {
                    (v / d).round_ties_even().clamp(-127.0, 127.0)
                };
                o[b * BLOCK + j] = q as i8 as u8;
            }
            o[KV_Q8_DIM + 2 * b..KV_Q8_DIM + 2 * b + 2].copy_from_slice(&dh.to_le_bytes());
        }
    }
}

/// Decuantiza filas de la KV cache Q8 (inversa de [`quantize_kv_q8`]; `d · q` es exacto en f32).
pub fn dequantize_kv_q8(data: &[u8], out: &mut [f32]) {
    assert_eq!(data.len() % KV_Q8_ROW, 0);
    assert_eq!(out.len(), data.len() / KV_Q8_ROW * KV_Q8_DIM);
    for (r, o) in data
        .chunks_exact(KV_Q8_ROW)
        .zip(out.chunks_exact_mut(KV_Q8_DIM))
    {
        for (i, v) in o.iter_mut().enumerate() {
            let b = i / BLOCK;
            let d = f16_to_f32(u16::from_le_bytes([
                r[KV_Q8_DIM + 2 * b],
                r[KV_Q8_DIM + 2 * b + 1],
            ]));
            *v = d * (r[i] as i8) as f32;
        }
    }
}

/// Valor q (0..63) del elemento `i` de un bloque q6_0 de 26 bytes (ADR 0012).
pub fn q6_value(b: &[u8], i: usize) -> u8 {
    let ql = b[2 + i % 16];
    let lo = if i < 16 { ql & 0x0f } else { ql >> 4 };
    let hi = (b[18 + i % 8] >> (2 * (i / 8))) & 3;
    lo | (hi << 4)
}

/// Decuantiza `data` (tipo `q`) a f32. `out.len()` es la cantidad de elementos.
pub fn dequantize(q: QType, data: &[u8], out: &mut [f32]) {
    assert_eq!(
        data.len(),
        q.nbytes(out.len()),
        "tamaño de datos inconsistente"
    );
    match q {
        QType::F32 => {
            for (o, c) in out.iter_mut().zip(data.chunks_exact(4)) {
                *o = f32::from_le_bytes(c.try_into().unwrap());
            }
        }
        QType::Q4_0 => {
            for (o, b) in out.chunks_exact_mut(BLOCK).zip(data.chunks_exact(18)) {
                let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                for j in 0..16 {
                    let byte = b[2 + j];
                    o[j] = d * ((byte & 0x0f) as i32 - 8) as f32;
                    o[j + 16] = d * ((byte >> 4) as i32 - 8) as f32;
                }
            }
        }
        QType::Q6_0 => {
            for (o, b) in out.chunks_exact_mut(BLOCK).zip(data.chunks_exact(26)) {
                let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                for (i, v) in o.iter_mut().enumerate() {
                    *v = d * (q6_value(b, i) as i32 - 32) as f32;
                }
            }
        }
        QType::Q8_0 => {
            for (o, b) in out.chunks_exact_mut(BLOCK).zip(data.chunks_exact(34)) {
                let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                for j in 0..BLOCK {
                    o[j] = d * (b[2 + j] as i8) as f32;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_q8_ida_y_vuelta() {
        // Una fila con un bloque nulo, uno con amax exacto y valores en el medio de dos pasos.
        let mut x: Vec<f32> = (0..KV_Q8_DIM)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.173)
            .collect();
        x[32..64].fill(0.0);
        let mut q = vec![0u8; KV_Q8_ROW];
        quantize_kv_q8(&x, &mut q);
        let mut y = vec![0f32; KV_Q8_DIM];
        dequantize_kv_q8(&q, &mut y);
        for b in 0..4 {
            let blk = &x[b * 32..(b + 1) * 32];
            let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
            for j in 0..32 {
                let (a, r) = (blk[j], y[b * 32 + j]);
                // Medio paso de cuantización más el error relativo de la escala f16.
                assert!(
                    (a - r).abs() <= amax / 127.0 * 0.5 * (1.0 + 1e-3) + 1e-12,
                    "{a} {r}"
                );
            }
            // El máximo se representa con |q| = 127.
            if amax > 0.0 {
                assert!(
                    q[b * 32..(b + 1) * 32]
                        .iter()
                        .any(|&v| (v as i8).unsigned_abs() == 127)
                );
            }
        }
        assert!(y[32..64].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn f16_conocidos() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8); // subnormal mínimo
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert!(f16_to_f32(0x7c00).is_infinite());
        // Todos los finitos ida y vuelta por el mismo valor que f32 -> f16 de la CPU.
        for h in 0..=0x7bffu16 {
            let f = f16_to_f32(h);
            assert!(f.is_finite() && f >= 0.0);
        }
    }

    #[test]
    fn f32_a_f16_redondea_al_par() {
        // Ida y vuelta exacta para todo f16 no NaN.
        for h in 0..=u16::MAX {
            let f = f16_to_f32(h);
            if !f.is_nan() {
                assert_eq!(f32_to_f16(f), h, "{h:#06x}");
            }
        }
        // Puntos medios entre f16 consecutivos (exactos en f32): van al par; apenas por encima o
        // por debajo, al más cercano.
        for h in 0..0x7bffu16 {
            let (a, b) = (f16_to_f32(h), f16_to_f32(h + 1));
            let mid = (a + b) / 2.0;
            let even = if h % 2 == 0 { h } else { h + 1 };
            assert_eq!(f32_to_f16(mid), even, "medio de {h:#06x}");
            assert_eq!(f32_to_f16(f32::from_bits(mid.to_bits() + 1)), h + 1);
            assert_eq!(f32_to_f16(f32::from_bits(mid.to_bits() - 1)), h);
            assert_eq!(f32_to_f16(-mid), even | 0x8000);
        }
        assert_eq!(f32_to_f16(65520.0), 0x7c00); // medio entre 65504 e inf -> inf
        assert_eq!(f32_to_f16(1e10), 0x7c00);
        assert_eq!(f32_to_f16(2f32.powi(-25)), 0); // medio entre 0 y el subnormal mínimo
        assert_eq!(f32_to_f16(2f32.powi(-25) * 1.0001), 1);
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    #[test]
    fn q4_0_bloque_a_mano() {
        // d = 0.5 (f16 0x3800); byte j: nibble bajo = j % 16, alto = 15 - j % 16.
        let mut b = vec![0x00, 0x38];
        for j in 0..16u8 {
            b.push(j | ((15 - j) << 4));
        }
        let mut out = [0f32; 32];
        dequantize(QType::Q4_0, &b, &mut out);
        for j in 0..16 {
            assert_eq!(out[j], 0.5 * (j as f32 - 8.0));
            assert_eq!(out[j + 16], 0.5 * (7.0 - j as f32));
        }
    }

    #[test]
    fn q6_0_bloque_a_mano() {
        // d = 1 y q_j = 2j (0..62): ql y qh armados a mano según ADR 0012.
        let q: Vec<u8> = (0..32).map(|j| 2 * j as u8).collect();
        let mut b = vec![0x00, 0x3c];
        b.extend((0..16).map(|j| (q[j] & 0x0f) | ((q[j + 16] & 0x0f) << 4)));
        b.extend((0..8).map(|j| {
            (q[j] >> 4) | ((q[j + 8] >> 4) << 2) | ((q[j + 16] >> 4) << 4) | ((q[j + 24] >> 4) << 6)
        }));
        let mut out = [0f32; 32];
        dequantize(QType::Q6_0, &b, &mut out);
        for (j, o) in out.iter().enumerate() {
            assert_eq!(*o, 2.0 * j as f32 - 32.0);
        }
    }

    #[test]
    fn q8_0_bloque_a_mano() {
        let mut b = vec![0x00, 0x3c]; // d = 1
        b.extend((0..32).map(|j| (j as i8 - 16) as u8));
        let mut out = [0f32; 32];
        dequantize(QType::Q8_0, &b, &mut out);
        for (j, o) in out.iter().enumerate() {
            assert_eq!(*o, j as f32 - 16.0);
        }
    }
}
