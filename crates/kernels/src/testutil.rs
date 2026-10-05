//! Utilidades deterministas para tests y microbenchmarks (sin dependencias externas).

/// Generador xorshift64* con semilla fija: mismos datos en cada corrida.
#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Uniforme en [-1, 1).
    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }

    /// Vector con valores uniformes en [-scale, scale).
    pub fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.uniform() * scale).collect()
    }
}

/// Distancia en ULPs entre dos f32 finitos.
pub fn ulps(a: f32, b: f32) -> u32 {
    let key = |x: f32| {
        let i = x.to_bits() as i32;
        if i < 0 { i32::MIN - i } else { i }
    };
    key(a).abs_diff(key(b))
}

/// Mediana de una lista de tiempos.
pub fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

impl Rng {
    /// f16 aleatorio en ±[2^-8, 2^-1) (bits directos, sin conversión).
    pub fn f16_scale(&mut self) -> [u8; 2] {
        let sign = (self.next_u64() & 1) as u16;
        let exp = 7 + (self.next_u64() % 7) as u16; // 2^-8 .. 2^-2
        let mant = (self.next_u64() & 0x3ff) as u16;
        ((sign << 15) | (exp << 10) | mant).to_le_bytes()
    }

    /// Matriz q4_0 `[rows, cols]` con escalas y nibbles aleatorios.
    pub fn q4_0(&mut self, rows: usize, cols: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(rows * cols / 32 * 18);
        for _ in 0..rows * cols / 32 {
            out.extend_from_slice(&self.f16_scale());
            out.extend((0..16).map(|_| self.next_u64() as u8));
        }
        out
    }

    /// Matriz q8_0 `[rows, cols]` con escalas y valores aleatorios en [-127, 127].
    pub fn q8_0(&mut self, rows: usize, cols: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(rows * cols / 32 * 34);
        for _ in 0..rows * cols / 32 {
            out.extend_from_slice(&self.f16_scale());
            out.extend((0..32).map(|_| ((self.next_u64() % 255) as i32 - 127) as i8 as u8));
        }
        out
    }
}

/// Peor error relativo `|g - r| / max(|r|, floor)`.
pub fn max_rel(got: &[f32], expected: &[f32], floor: f32) -> f32 {
    got.iter()
        .zip(expected)
        .map(|(g, r)| (g - r).abs() / r.abs().max(floor))
        .fold(0.0, f32::max)
}
