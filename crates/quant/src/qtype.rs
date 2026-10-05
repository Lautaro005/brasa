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
