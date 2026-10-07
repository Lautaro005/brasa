//! Autotuner de los parámetros de lanzamiento de decode (ADR 0029).
//!
//! Para cada kernel de la ruta de decode del modelo (con sus formas reales) mide en GPU cada
//! valor candidato de simdgroups por threadgroup, intercalando los candidatos en cada ronda para
//! que la deriva térmica o de reloj no favorezca a ninguno, y se queda con la mediana más baja.
//! Los pesos y la KV son aleatorios con las formas del modelo: el tiempo no depende de los valores.
//! Un candidato reemplaza al valor por defecto solo si es al menos [`MIN_GAIN`] más rápido; si no,
//! la entrada queda con el valor por defecto y su medición como evidencia.

use std::collections::BTreeMap;

use brasa_kernels::launch::SG_CANDIDATES;
use brasa_kernels::testutil::Rng;
use brasa_kernels::{
    AttnShape, GEMV_NR, GEMV_SG_DEFAULT, GemvOp, Kernels, KvType, LANES_SG_DEFAULT, Launch,
    QMatrix, WeightType, decode_partials_len, gqa_supported, norm_partials,
};
use brasa_metal::{Arg, Buffer, Command, Context, MetalError};

use crate::apply::{ATTN_LANES, SG, gemv_shape, lanes_shape, lanes_variant};
use crate::db::{TuningDb, TuningKey, TuningValue};
use crate::fingerprint::Fingerprint;

/// Mejora mínima (fracción del tiempo por defecto) para cambiar un valor por defecto.
pub const MIN_GAIN: f64 = 0.03;
/// Fracción mínima de rondas en que el candidato elegido le gana al valor por defecto.
pub const MIN_WINS: f64 = 0.8;
/// Dispersión máxima (rango intercuartil / mediana) del valor por defecto para aceptar un cambio.
pub const NOISE_MAX: f64 = 0.05;

/// Dimensiones del modelo que fijan las formas de decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelDims {
    pub layers: usize,
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub vocab: usize,
    /// Tipo de las matrices de las capas (q4_0 en Qwen3-4B Q4).
    pub layer_type: WeightType,
    /// Tipo de la tabla de embeddings, que hace de lm_head (q6_0, ADR 0012; q8_0 en archivos
    /// viejos).
    pub head_type: WeightType,
}

impl ModelDims {
    pub fn q_dim(&self) -> usize {
        self.heads * self.head_dim
    }

    pub fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// GEMV de un paso de decode, como los encola `Qwen3::encode_layer` y el lm_head:
    /// (operación, filas, columnas, veces por token).
    pub fn gemvs(&self) -> Vec<(GemvOp, usize, usize, usize)> {
        let (h, l) = (self.hidden, self.layers);
        let head = if self.head_type == WeightType::Q8_0 {
            GemvOp::Fast(WeightType::Q8_0)
        } else {
            GemvOp::Scaled(self.head_type)
        };
        vec![
            (GemvOp::Scaled3, self.q_dim() + 2 * self.kv_dim(), h, l),
            (GemvOp::Fast(self.layer_type), h, self.q_dim(), l),
            (GemvOp::ScaledSwiglu, self.ffn, h, l),
            (GemvOp::Fast(self.layer_type), h, self.ffn, l),
            (head, self.vocab, h, 1),
        ]
    }
}

/// Alcance del tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// GEMV de decode y atención a 2K y 16K con KV f16 y Q8; 15 rondas. Menos de un minuto.
    Quick,
    /// Además KV f32 y más longitudes de caché; 41 rondas.
    Full,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Quick => "quick",
            Mode::Full => "full",
        }
    }

    /// Rondas medidas (más una de calentamiento que se descarta).
    pub fn samples(self) -> usize {
        match self {
            Mode::Quick => 15,
            Mode::Full => 41,
        }
    }

    pub fn kv_types(self) -> &'static [KvType] {
        match self {
            Mode::Quick => &[KvType::F16, KvType::Q8_0],
            Mode::Full => &[KvType::F32, KvType::F16, KvType::Q8_0],
        }
    }

    /// Longitudes de caché medidas, hasta `max_ctx`.
    pub fn lengths(self, max_ctx: usize) -> Vec<usize> {
        let all: &[usize] = match self {
            Mode::Quick => &[2048, 16384],
            Mode::Full => &[512, 2048, 8192, 16384, 32768],
        };
        let mut v: Vec<usize> = all.iter().copied().filter(|&l| l <= max_ctx).collect();
        if v.is_empty() {
            v.push(max_ctx.max(1));
        }
        v
    }
}

/// Opciones del tuning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    pub mode: Mode,
    /// Contexto máximo a medir en la atención.
    pub max_ctx: usize,
    /// Límite de memoria para copias de pesos y cachés por kernel (bytes), para no medir con
    /// datos que entren en la caché del sistema (SLC) y no exigir más RAM de la que hay.
    pub copy_budget: usize,
}

impl Options {
    pub fn new(mode: Mode, max_ctx: usize) -> Self {
        Self {
            mode,
            max_ctx,
            copy_budget: 512 << 20,
        }
    }
}

/// Resultado de un kernel: clave, valor y los tiempos de todos los candidatos (µs por dispatch).
#[derive(Debug, Clone, PartialEq)]
pub struct Measured {
    pub key: TuningKey,
    pub value: TuningValue,
    pub candidates: Vec<(usize, f64)>,
    /// Dispersión del valor por defecto entre rondas (rango intercuartil / mediana). Por encima
    /// de [`NOISE_MAX`] la medición no sirve para cambiar nada (otra carga en la GPU).
    pub noise: f64,
}

impl Measured {
    pub fn noisy(&self) -> bool {
        self.noise > NOISE_MAX
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Rango intercuartil / mediana de las rondas del valor por defecto (0 si no se midió).
fn noise(rounds: &Rounds, default_sg: usize) -> f64 {
    let Some((_, t)) = rounds.iter().find(|(sg, _)| *sg == default_sg) else {
        return 0.0;
    };
    let mut v = t.clone();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n < 4 {
        return 0.0;
    }
    (v[3 * n / 4] - v[n / 4]) / v[n / 2]
}

/// Tiempos por ronda de cada candidato (µs por dispatch), en el orden de medición.
type Rounds = Vec<(usize, Vec<f64>)>;

/// Medianas por candidato.
fn medians(rounds: &Rounds) -> Vec<(usize, f64)> {
    rounds
        .iter()
        .map(|(sg, t)| (*sg, median(t.clone())))
        .collect()
}

/// Elige el candidato: el de mediana más baja, si mejora la mediana del valor por defecto en al
/// menos [`MIN_GAIN`] y además es más rápido que él en al menos [`MIN_WINS`] de las rondas
/// (comparación pareada: las rondas intercalan los candidatos). Si no, el valor por defecto.
/// Devuelve (elegido, µs elegido, µs por defecto).
fn choose(rounds: &Rounds, default_sg: usize) -> (usize, f64, f64) {
    let quiet = noise(rounds, default_sg) <= NOISE_MAX;
    let med = medians(rounds);
    let default_us = med
        .iter()
        .find(|(sg, _)| *sg == default_sg)
        .map_or(f64::INFINITY, |t| t.1);
    let (best_i, &(best_sg, best_us)) = med
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.1.total_cmp(&b.1.1))
        .expect("al menos un candidato");
    let wins = match rounds.iter().find(|(sg, _)| *sg == default_sg) {
        Some((_, d)) => {
            let b = &rounds[best_i].1;
            b.iter().zip(d).filter(|(x, y)| x < y).count() as f64 / b.len().max(1) as f64
        }
        None => 1.0,
    };
    if quiet && best_us < default_us * (1.0 - MIN_GAIN) && wins >= MIN_WINS {
        (best_sg, best_us, default_us)
    } else {
        (default_sg, default_us, default_us)
    }
}

/// Copias de un buffer de `bytes` que caben en el presupuesto (entre 1 y `max`).
fn copies(bytes: usize, budget: usize, max: usize) -> usize {
    (budget / bytes.max(1)).clamp(1, max.max(1))
}

fn weight_bytes(t: WeightType, rows: usize, cols: usize) -> Vec<u8> {
    let mut rng = Rng::new((rows * 31 + cols) as u64);
    match t {
        WeightType::Q4_0 => rng.q4_0(rows, cols),
        WeightType::Q8_0 => rng.q8_0(rows, cols),
        WeightType::Q6_0 => rng.q6_0(rows, cols),
    }
}

/// Bytes de una caché `[lk, hkv, head_dim]` en el tipo `kv`, con valores aleatorios válidos.
fn kv_bytes(kv: KvType, n: usize) -> Vec<u8> {
    let x = Rng::new(n as u64).vec(n, 2.0);
    match kv {
        KvType::F32 => x.iter().flat_map(|v| v.to_le_bytes()).collect(),
        KvType::F16 => x
            .iter()
            .flat_map(|&v| brasa_quant::f32_to_f16(v).to_le_bytes())
            .collect(),
        KvType::Q8_0 => {
            let mut b = vec![0u8; kv.bytes(n)];
            brasa_quant::quantize_kv_q8(&x, &mut b);
            b
        }
    }
}

/// Candidatos que compilan y entran en el límite de su pipeline (`set_launch` compila la
/// variante y la deja en caché para las mediciones).
fn usable(
    ctx: &Context,
    kernels: &mut Kernels,
    make: impl Fn(&mut Launch, usize) -> bool,
) -> Vec<(usize, Launch)> {
    let cands = SG_CANDIDATES
        .iter()
        .filter_map(|&sg| {
            let mut l = Launch::default();
            make(&mut l, sg).then_some((sg, l))
        })
        .filter(|(_, l)| kernels.set_launch(ctx, l.clone()).is_ok())
        .collect();
    let _ = kernels.set_launch(ctx, Launch::default());
    cands
}

/// Candidatos válidos para el GEMV `op` `[rows, cols]`.
fn gemv_candidates(
    ctx: &Context,
    kernels: &mut Kernels,
    op: GemvOp,
    rows: usize,
    cols: usize,
) -> Vec<(usize, Launch)> {
    usable(ctx, kernels, |l, sg| l.set_gemv(op, rows, cols, sg).is_ok())
}

#[allow(clippy::too_many_arguments)]
fn tune_gemv(
    ctx: &Context,
    kernels: &mut Kernels,
    dims: &ModelDims,
    op: GemvOp,
    rows: usize,
    cols: usize,
    per_token: usize,
    opts: &Options,
) -> Result<Option<Measured>, MetalError> {
    // gemv_scaled3: las tres matrices (q, k, v) por separado, como en el modelo.
    let parts: Vec<usize> = if op == GemvOp::Scaled3 {
        vec![dims.q_dim(), dims.kv_dim(), dims.kv_dim()]
    } else {
        vec![rows]
    };
    let qtype = match op {
        GemvOp::Fast(t) | GemvOp::Scaled(t) => t,
        GemvOp::Scaled3 | GemvOp::ScaledSwiglu => WeightType::Q4_0,
    };
    let mats_per_call = if op == GemvOp::ScaledSwiglu { 2 } else { 1 };
    let set_bytes: usize = parts
        .iter()
        .map(|&r| weight_bytes_len(qtype, r, cols))
        .sum::<usize>()
        * mats_per_call;
    let n_copies = copies(set_bytes, opts.copy_budget, per_token);
    let mut cands = gemv_candidates(ctx, kernels, op, rows, cols);
    if op == GemvOp::Scaled3 {
        // Cada matriz tiene que ser múltiplo de las filas por threadgroup.
        cands.retain(|(sg, _)| parts.iter().all(|r| r % (GEMV_NR * sg) == 0));
    }
    if cands.len() < 2 {
        return Ok(None);
    }
    // Pesos: un juego aleatorio por forma, copiado `n_copies` veces (direcciones distintas).
    let data: Vec<Vec<u8>> = parts
        .iter()
        .map(|&r| weight_bytes(qtype, r, cols))
        .collect();
    let mut sets: Vec<Vec<Buffer<u8>>> = Vec::with_capacity(n_copies);
    for _ in 0..n_copies {
        let mut set = Vec::with_capacity(parts.len() * mats_per_call);
        for _ in 0..mats_per_call {
            for d in &data {
                set.push(ctx.buffer_from(d)?);
            }
        }
        sets.push(set);
    }
    let mut rng = Rng::new(9);
    let x = ctx.buffer_from(&rng.vec(cols, 1.0))?;
    let ssv: Vec<f32> = rng
        .vec(norm_partials(cols), 1.0)
        .iter()
        .map(|v| v.abs() * 100.0 + 1.0)
        .collect();
    let ss = ctx.buffer_from(&ssv)?;
    let ys: Vec<Buffer<f32>> = parts
        .iter()
        .map(|&r| ctx.buffer::<f32>(r))
        .collect::<Result<_, _>>()?;
    let m = |data, rows| QMatrix {
        data,
        qtype,
        rows,
        cols,
    };
    let dispatches = per_token;
    let samples = opts.mode.samples();
    let times = measure(ctx, kernels, &cands, samples, dispatches, |k, cmd| {
        for i in 0..dispatches {
            let s = &sets[i % sets.len()];
            match op {
                GemvOp::Fast(_) => k.gemv(cmd, m(&s[0], rows), Arg::buf(&x), Arg::buf(&ys[0]), 1),
                GemvOp::Scaled(_) => k.gemv_scaled(
                    cmd,
                    m(&s[0], rows),
                    Arg::buf(&x),
                    Arg::buf(&ss),
                    1e-6,
                    Arg::buf(&ys[0]),
                ),
                GemvOp::Scaled3 => k.gemv_scaled3(
                    cmd,
                    [m(&s[0], parts[0]), m(&s[1], parts[1]), m(&s[2], parts[2])],
                    Arg::buf(&x),
                    Arg::buf(&ss),
                    1e-6,
                    [Arg::buf(&ys[0]), Arg::buf(&ys[1]), Arg::buf(&ys[2])],
                ),
                GemvOp::ScaledSwiglu => k.gemv_scaled_swiglu(
                    cmd,
                    m(&s[0], rows),
                    m(&s[1], rows),
                    Arg::buf(&x),
                    Arg::buf(&ss),
                    1e-6,
                    Arg::buf(&ys[0]),
                ),
            }
        }
    })?;
    let (sg, gpu_us, default_us) = choose(&times, GEMV_SG_DEFAULT);
    let noise = noise(&times, GEMV_SG_DEFAULT);
    Ok(Some(Measured {
        key: TuningKey::new(op.kernel_name(), gemv_shape(rows, cols), ""),
        value: TuningValue {
            params: BTreeMap::from([(SG.to_string(), sg as u32)]),
            gpu_us,
            default_us,
            samples: opts.mode.samples() as u32,
        },
        candidates: medians(&times),
        noise,
    }))
}

fn weight_bytes_len(t: WeightType, rows: usize, cols: usize) -> usize {
    let block = match t {
        WeightType::Q4_0 => 18,
        WeightType::Q8_0 => 34,
        WeightType::Q6_0 => 26,
    };
    rows * cols / 32 * block
}

/// Mide `encode` para cada candidato, intercalados por ronda (más una ronda de calentamiento que
/// se descarta); devuelve la mediana en µs por dispatch.
fn measure<'a, F>(
    ctx: &Context,
    kernels: &mut Kernels,
    candidates: &[(usize, Launch)],
    samples: usize,
    dispatches: usize,
    encode: F,
) -> Result<Rounds, MetalError>
where
    F: Fn(&Kernels, &mut Command<'a>),
{
    let mut times: Vec<Vec<f64>> = vec![Vec::with_capacity(samples); candidates.len()];
    for round in 0..=samples {
        for (i, (_, launch)) in candidates.iter().enumerate() {
            kernels
                .set_launch(ctx, launch.clone())
                .map_err(|e| MetalError(e.to_string()))?;
            let mut cmd = ctx.command()?;
            encode(kernels, &mut cmd);
            let t = cmd.commit_and_wait()?.gpu_seconds;
            if round > 0 {
                times[i].push(t * 1e6 / dispatches as f64);
            }
        }
    }
    Ok(candidates
        .iter()
        .zip(times)
        .map(|((sg, _), t)| (*sg, t))
        .collect())
}

fn tune_lanes(
    ctx: &Context,
    kernels: &mut Kernels,
    dims: &ModelDims,
    kv: KvType,
    lk: usize,
    opts: &Options,
) -> Result<Option<Measured>, MetalError> {
    let (hq, hkv, hd) = (dims.heads, dims.kv_heads, dims.head_dim);
    let group = hq / hkv;
    let cands = usable(ctx, kernels, |l, sg| {
        l.set_attn_lanes(kv, group, lk, sg).is_ok()
    });
    if cands.len() < 2 {
        return Ok(None);
    }
    let n = lk.next_multiple_of(128) * hkv * hd;
    let bytes = kv_bytes(kv, n);
    let n_copies = copies(2 * bytes.len(), opts.copy_budget, dims.layers.min(8));
    let caches: Vec<(Buffer<u8>, Buffer<u8>)> = (0..n_copies)
        .map(|_| Ok((ctx.buffer_from(&bytes)?, ctx.buffer_from(&bytes)?)))
        .collect::<Result<_, MetalError>>()?;
    let q = ctx.buffer_from(&Rng::new(11).vec(hq * hd, 1.0))?;
    let o = ctx.buffer::<f32>(hq * hd)?;
    let part = ctx.buffer::<f32>(decode_partials_len(hq, lk))?;
    let shape = AttnShape {
        tokens: 1,
        hq,
        hkv,
        dim: hd,
        pos0: lk - 1,
        kv,
    };
    let dispatches = dims.layers;
    let samples = opts.mode.samples();
    let times = measure(ctx, kernels, &cands, samples, dispatches, |k, cmd| {
        for l in 0..dispatches {
            let (kc, vc) = &caches[l % caches.len()];
            k.decode_attention_lanes(
                cmd,
                Arg::buf(&q),
                Arg::buf(kc),
                Arg::buf(vc),
                &part,
                Arg::buf(&o),
                shape,
            );
        }
    })?;
    let (sg, gpu_us, default_us) = choose(&times, LANES_SG_DEFAULT);
    let noise = noise(&times, LANES_SG_DEFAULT);
    Ok(Some(Measured {
        key: TuningKey::new(ATTN_LANES, lanes_shape(hkv, lk), lanes_variant(kv, group)),
        value: TuningValue {
            params: BTreeMap::from([(SG.to_string(), sg as u32)]),
            gpu_us,
            default_us,
            samples: opts.mode.samples() as u32,
        },
        candidates: medians(&times),
        noise,
    }))
}

/// Corre el tuning y devuelve la base para `fingerprint`. `report` recibe cada resultado apenas
/// se mide (para mostrar el avance).
pub fn tune(
    ctx: &Context,
    dims: &ModelDims,
    opts: &Options,
    fingerprint: Fingerprint,
    report: &mut dyn FnMut(&Measured),
) -> Result<TuningDb, MetalError> {
    let mut kernels = Kernels::new(ctx)?;
    let mut db = TuningDb::new(fingerprint);
    let mut add = |db: &mut TuningDb, m: Option<Measured>| {
        if let Some(m) = m {
            report(&m);
            db.insert(m.key, m.value);
        }
    };
    for (op, rows, cols, per_token) in dims.gemvs() {
        let m = tune_gemv(ctx, &mut kernels, dims, op, rows, cols, per_token, opts)?;
        add(&mut db, m);
    }
    if gqa_supported(dims.heads, dims.kv_heads) && dims.head_dim == 128 {
        for &kv in opts.mode.kv_types() {
            for lk in opts.mode.lengths(opts.max_ctx) {
                let m = tune_lanes(ctx, &mut kernels, dims, kv, lk, opts)?;
                add(&mut db, m);
            }
        }
    }
    Ok(db)
}

/// [`tune`] en el dispositivo Metal por defecto, con el fingerprint de esta máquina.
pub fn tune_this_machine(
    dims: &ModelDims,
    opts: &Options,
    report: &mut dyn FnMut(&Measured),
) -> Result<TuningDb, String> {
    let ctx = Context::new().map_err(|e| e.to_string())?;
    tune(&ctx, dims, opts, Fingerprint::current(), report).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elige_el_mas_rapido_solo_con_mejora_suficiente() {
        let r = |v: &[(usize, &[f64])]| -> Rounds {
            v.iter().map(|(sg, t)| (*sg, t.to_vec())).collect()
        };
        // 1 % más rápido que el valor por defecto: no alcanza.
        let a = r(&[(1, &[99.0; 5]), (2, &[100.0; 5]), (4, &[120.0; 5])]);
        assert_eq!(choose(&a, 2), (2, 100.0, 100.0));
        // 10 % más rápido en todas las rondas: se elige.
        let b = r(&[(1, &[90.0; 5]), (2, &[100.0; 5]), (4, &[120.0; 5])]);
        assert_eq!(choose(&b, 2), (1, 90.0, 100.0));
        // Mediana 10 % mejor pero gana solo 3 de 5 rondas: ruido, queda el valor por defecto.
        let c = r(&[
            (1, &[90.0, 90.0, 90.0, 130.0, 130.0]),
            (2, &[100.0, 100.0, 100.0, 100.0, 100.0]),
        ]);
        assert_eq!(choose(&c, 2), (2, 100.0, 100.0));
        // 10 % más rápido siempre, pero el valor por defecto varía 20 % entre rondas: ruido.
        let e = r(&[
            (1, &[90.0, 90.0, 90.0, 90.0, 90.0]),
            (2, &[100.0, 120.0, 100.0, 125.0, 100.0]),
        ]);
        assert!(noise(&e, 2) > NOISE_MAX);
        assert_eq!(choose(&e, 2).0, 2);
        // El valor por defecto es el más rápido.
        let d = r(&[(1, &[130.0; 3]), (4, &[100.0; 3]), (8, &[101.0; 3])]);
        assert_eq!(choose(&d, 4), (4, 100.0, 100.0));
    }

    #[test]
    fn formas_de_qwen3_4b() {
        let d = ModelDims {
            layers: 36,
            hidden: 2560,
            heads: 32,
            kv_heads: 8,
            head_dim: 128,
            ffn: 9728,
            vocab: 151_936,
            layer_type: WeightType::Q4_0,
            head_type: WeightType::Q6_0,
        };
        let g = d.gemvs();
        assert_eq!(g[0], (GemvOp::Scaled3, 6144, 2560, 36));
        assert_eq!(g[1], (GemvOp::Fast(WeightType::Q4_0), 2560, 4096, 36));
        assert_eq!(g[2], (GemvOp::ScaledSwiglu, 9728, 2560, 36));
        assert_eq!(g[3], (GemvOp::Fast(WeightType::Q4_0), 2560, 9728, 36));
        assert_eq!(g[4], (GemvOp::Scaled(WeightType::Q6_0), 151_936, 2560, 1));
        assert_eq!(Mode::Quick.lengths(16384), vec![2048, 16384]);
        assert_eq!(Mode::Quick.lengths(4096), vec![2048]);
        assert_eq!(Mode::Quick.lengths(1000), vec![1000]);
        assert_eq!(Mode::Full.lengths(8192), vec![512, 2048, 8192]);
        assert_eq!(weight_bytes_len(WeightType::Q4_0, 64, 64), 64 * 2 * 18);
        assert_eq!(weight_bytes(WeightType::Q4_0, 64, 64).len(), 64 * 2 * 18);
        assert_eq!(copies(100 << 20, 512 << 20, 36), 5);
        assert_eq!(copies(1 << 30, 512 << 20, 36), 1);
        assert_eq!(copies(1 << 20, 512 << 20, 36), 36);
    }
}
