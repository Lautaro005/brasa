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
    F32,
}

impl QType {
    /// Bytes que ocupan `n` elementos.
    pub fn nbytes(self, n: usize) -> usize {
        match self {
            QType::Q4_0 => n / BLOCK * 18,
            QType::Q8_0 => n / BLOCK * 34,
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
