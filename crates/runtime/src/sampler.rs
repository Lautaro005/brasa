//! Sampling: greedy, temperatura, top-k, top-p y semilla. Sin asignaciones por paso: el buffer de
//! candidatos se reserva al crear el sampler.

/// Parámetros de muestreo. `temperature == 0` es greedy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    /// 0 = sin límite.
    pub top_k: usize,
    /// 1.0 = sin límite.
    pub top_p: f32,
    pub seed: u64,
}

impl SamplingParams {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
        }
    }

    /// Valores recomendados por Qwen3 en modo razonamiento (generation_config.json).
    pub fn qwen3_thinking(seed: u64) -> Self {
        Self {
            temperature: 0.6,
            top_k: 20,
            top_p: 0.95,
            seed,
        }
    }

    /// Valores recomendados por Qwen3 sin razonamiento (model card).
    pub fn qwen3_no_thinking(seed: u64) -> Self {
        Self {
            temperature: 0.7,
            top_k: 20,
            top_p: 0.8,
            seed,
        }
    }
}

/// PCG32 (O'Neill): determinista dada la semilla.
#[derive(Debug, Clone)]
struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    fn new(seed: u64) -> Self {
        let mut r = Self {
            state: 0,
            inc: (0xda3e_39cb_94b9_5bdb << 1) | 1,
        };
        r.next_u32();
        r.state = r.state.wrapping_add(seed);
        r.next_u32();
        r
    }

    fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniforme en [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
}

#[derive(Debug, Clone)]
pub struct Sampler {
    params: SamplingParams,
    rng: Pcg32,
    /// (token, logit) candidatos; capacidad = vocabulario.
    cand: Vec<(u32, f32)>,
}

impl Sampler {
    pub fn new(params: SamplingParams, vocab: usize) -> Self {
        Self {
            params,
            rng: Pcg32::new(params.seed),
            cand: Vec::with_capacity(vocab),
        }
    }

    pub fn params(&self) -> SamplingParams {
        self.params
    }

    /// Elige el próximo token a partir de los logits.
    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        let p = self.params;
        if p.temperature <= 0.0 {
            return argmax(logits);
        }
        self.cand.clear();
        self.cand
            .extend(logits.iter().enumerate().map(|(i, l)| (i as u32, *l)));
        // top-k: quedarse con los k mayores (sin ordenar todo el vocabulario).
        let by_logit = |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1);
        if p.top_k > 0 && p.top_k < self.cand.len() {
            self.cand.select_nth_unstable_by(p.top_k - 1, by_logit);
            self.cand.truncate(p.top_k);
        }
        self.cand.sort_unstable_by(by_logit);
        // Softmax con temperatura sobre los candidatos (ya ordenados de mayor a menor).
        let max = self.cand[0].1;
        let mut sum = 0f32;
        for c in self.cand.iter_mut() {
            c.1 = ((c.1 - max) / p.temperature).exp();
            sum += c.1;
        }
        // top-p: el prefijo más corto cuya masa alcanza top_p.
        let mut keep = self.cand.len();
        if p.top_p < 1.0 {
            let mut acc = 0f32;
            for (i, c) in self.cand.iter().enumerate() {
                acc += c.1 / sum;
                if acc >= p.top_p {
                    keep = i + 1;
                    break;
                }
            }
        }
        let total: f32 = self.cand[..keep].iter().map(|c| c.1).sum();
        let mut r = self.rng.next_f32() * total;
        for c in &self.cand[..keep] {
            r -= c.1;
            if r <= 0.0 {
                return c.0;
            }
        }
        self.cand[keep - 1].0
    }
}

/// Índice del primer máximo (los NaN no cuentan; 0 si no hay ningún valor mayor que -∞).
/// En dos pasadas sin saltos dependientes de los datos, para que se vectorice: con el
/// vocabulario de Qwen3 (151 936) baja de ~210 a ~30 µs por token en M1 Pro (T3.5).
pub fn argmax(v: &[f32]) -> u32 {
    const L: usize = 16;
    let mut lanes = [f32::NEG_INFINITY; L];
    let chunks = v.chunks_exact(L);
    let tail = chunks.remainder();
    for c in chunks {
        for (m, x) in lanes.iter_mut().zip(c) {
            *m = m.max(*x);
        }
    }
    let m = lanes
        .iter()
        .chain(tail)
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    if m == f32::NEG_INFINITY {
        return 0;
    }
    v.iter().position(|&x| x == m).unwrap_or(0) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_igual_a_la_version_escalar() {
        let escalar = |v: &[f32]| {
            let mut best = (0usize, f32::NEG_INFINITY);
            for (i, x) in v.iter().enumerate() {
                if *x > best.1 {
                    best = (i, *x);
                }
            }
            best.0 as u32
        };
        let (inf, nan) = (f32::NEG_INFINITY, f32::NAN);
        let casos: Vec<Vec<f32>> = vec![
            vec![],
            vec![inf, inf],
            vec![nan, inf, nan],
            vec![nan, 1.0, nan, 1.0],
            vec![-3.0, -1.0, -1.0, -2.0],
            (0..151_936).map(|i| ((i * 7919) % 1000) as f32).collect(),
            (0..37).map(|i| ((i * 13) % 5) as f32 - 2.0).collect(),
        ];
        for v in &casos {
            assert_eq!(argmax(v), escalar(v), "{:?}", &v[..v.len().min(8)]);
        }
    }

    #[test]
    fn greedy_es_argmax() {
        let mut s = Sampler::new(SamplingParams::greedy(), 4);
        assert_eq!(s.sample(&[0.1, 3.0, -1.0, 2.9]), 1);
    }

    #[test]
    fn semilla_reproducible() {
        let logits: Vec<f32> = (0..1000).map(|i| ((i * 37) % 101) as f32 / 10.0).collect();
        let run = |seed| {
            let mut s = Sampler::new(SamplingParams::qwen3_thinking(seed), logits.len());
            (0..50).map(|_| s.sample(&logits)).collect::<Vec<_>>()
        };
        assert_eq!(run(7), run(7));
        assert_ne!(run(7), run(8));
    }

    #[test]
    fn top_k_y_top_p_restringen() {
        // Logits 10, 9, 8, ... : con top_k = 3 solo pueden salir los tres primeros.
        let logits: Vec<f32> = (0..100).map(|i| 10.0 - i as f32).collect();
        let mut s = Sampler::new(
            SamplingParams {
                temperature: 1.0,
                top_k: 3,
                top_p: 1.0,
                seed: 1,
            },
            100,
        );
        for _ in 0..500 {
            assert!(s.sample(&logits) < 3);
        }
        // top_p chico: solo el más probable (masa ~0,63 > 0,5).
        let mut s = Sampler::new(
            SamplingParams {
                temperature: 1.0,
                top_k: 0,
                top_p: 0.5,
                seed: 1,
            },
            100,
        );
        for _ in 0..200 {
            assert_eq!(s.sample(&logits), 0);
        }
    }

    #[test]
    fn frecuencias_siguen_la_distribucion() {
        // Dos tokens con logits ln(3) y 0 a temperatura 1: probabilidades 0,75 y 0,25.
        let logits = [3f32.ln(), 0.0];
        let mut s = Sampler::new(
            SamplingParams {
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                seed: 3,
            },
            2,
        );
        let n = 20_000;
        let zeros = (0..n).filter(|_| s.sample(&logits) == 0).count();
        let f = zeros as f64 / n as f64;
        assert!((f - 0.75).abs() < 0.015, "frecuencia {f}");
    }
}
