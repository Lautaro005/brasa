//! Criterio de T1.6: forward completo con KV cache contra las fixtures (ADR 0006, dos
//! comparaciones).
//!
//! 1. Correctitud del engine, contra la referencia FP32 con los mismos pesos decuantizados
//!    (fixtures/qwen3-4b-q4), en todos los prompts:
//!    - logits de la última posición del prompt: error máximo relativo
//!      `max|Δ| / max|ref|` ≤ `LOGIT_TOL` y mismo top-1;
//!    - teacher forcing sobre prompt + greedy de la referencia, por los dos caminos del engine:
//!      prefill en bloques (GEMM) y decode de a un token (GEMV); el top-1 de Brasa debe ser el de
//!      la referencia en cada posición, salvo empates (diferencia top-1/top-2 de la referencia
//!      menor que `TIE`).
//! 2. Calidad de la cuantización, contra la referencia FP32 sin cuantizar (fixtures/qwen3-4b):
//!    coincidencia de top-1 en teacher forcing (informativo; mínimo `MIN_Q4_AGREE`).
//!
//! Se corre con cada tipo de KV cache (ADR 0009): f32 contra fixtures/qwen3-4b-q4, f16 contra
//! fixtures/qwen3-4b-q4-kvf16 y q8_0 contra fixtures/qwen3-4b-q4-kvq8 (referencias con K y V
//! redondeados al tipo de la caché). El teacher forcing se exige igual en los tres. Con KV
//! redondeada, una diferencia ínfima con la referencia cambia algunos elementos en un paso del
//! tipo, así que la tolerancia de logits sale del piso medido con la propia referencia perturbada
//! en 1e-7 (tools/kv_rounding_sensitivity.py): hasta 7e-4 en f16 (`LOGIT_TOL_KV16`) y hasta
//! 8,8e-3 en Q8 (`LOGIT_TOL_KVQ8`). En Q8 ese piso también cambia el top-1 en posiciones casi
//! empatadas: la referencia perturbada en 1e-6 cambia 13 de 1920 en long-context, con brecha
//! top-1/top-2 de hasta 8,9e-3 · |top-1|. Por eso, con KV Q8 cuenta como empate una brecha menor
//! que `TIE_REL_KVQ8` · |top-1|. Se mide además la pérdida por redondear la KV: coincidencia de
//! top-1 contra la referencia con KV sin redondear (mínimo `MIN_KV_AGREE`).
//!
//! Las pruebas `forward_kv_*` usan el GEMM de prefill exacto (`GemmInput::F32`); las
//! `forward_gemm_f16_*`, la ruta caliente, que redondea pesos y activaciones a f16 dentro del GEMM
//! (ADR 0030), con `LOGIT_TOL_GEMM16` y empates relativos `TIE_REL_GEMM16`.
//!
//! Necesita models/qwen3-4b-q4/model.brasa:
//!   cargo test --release -p brasa-models --test forward -- --ignored --nocapture

use std::path::{Path, PathBuf};

use brasa_metal::Context;
use brasa_models::qwen3::{GemmInput, KvType, Limits, Qwen3};
use serde_json::Value;

const LOGIT_TOL: f32 = 1e-4;
const LOGIT_TOL_KV16: f32 = 1e-3;
const LOGIT_TOL_KVQ8: f32 = 2e-2;
const TIE: f32 = 1e-3;
const TIE_REL_KVQ8: f32 = 1e-2;
const MIN_Q4_AGREE: f64 = 0.80;
const MIN_KV_AGREE: f64 = 0.98;
const CHUNK: usize = 64;
/// GEMM de prefill con entradas f16 (ADR 0030). Las fixtures no redondean pesos ni activaciones,
/// así que la tolerancia sale de lo medido en M1 Pro con KV f16: logits hasta 6,3e-4 (con GEMM
/// f32: 2,3e-4) y 2 de 2898 posiciones con otro top-1, con brecha top-1/top-2 de hasta
/// 3,5e-4 · |top-1|. Con KV Q8 manda su propia tolerancia (8,5e-3 medido).
const LOGIT_TOL_GEMM16: f32 = 2e-3;
const TIE_REL_GEMM16: f32 = 1e-3;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p)
        .unwrap()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn read_u32(p: &Path) -> Vec<u32> {
    read_f32(p).iter().map(|f| f.to_bits()).collect()
}

fn manifest(dir: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap()
}

fn argmax(v: &[f32]) -> u32 {
    v.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, x)| {
            if *x > bv { (i, *x) } else { (bi, bv) }
        })
        .0 as u32
}

/// Logits de todas las posiciones de `seq` con prefill en bloques de `CHUNK` (camino GEMM).
fn logits_prefill(model: &mut Qwen3, ctx: &Context, seq: &[u32]) -> Vec<f32> {
    let vocab = model.cfg.vocab;
    let mut out = vec![0f32; seq.len() * vocab];
    for (i, chunk) in seq.chunks(CHUNK).enumerate() {
        let pos0 = i * CHUNK;
        let rows = chunk.len();
        model
            .forward(
                ctx,
                chunk,
                pos0,
                rows,
                &mut out[pos0 * vocab..(pos0 + rows) * vocab],
            )
            .unwrap();
    }
    out
}

/// Fila `j` de top-k (ids, logits) de la referencia.
struct TopK {
    ids: Vec<u32>,
    logits: Vec<f32>,
    k: usize,
}

impl TopK {
    fn load(dir: &Path, k: usize) -> Self {
        Self {
            ids: read_u32(&dir.join("topk_ids.i32")),
            logits: read_f32(&dir.join("topk_logits.f32")),
            k,
        }
    }

    fn top1(&self, j: usize) -> u32 {
        self.ids[j * self.k]
    }

    fn gap(&self, j: usize) -> f32 {
        self.logits[j * self.k] - self.logits[j * self.k + 1]
    }

    /// Empate según el tipo de KV (ver la documentación del módulo).
    fn tie(&self, j: usize, kv: KvType, gemm: GemmInput) -> bool {
        let g = self.gap(j);
        let top = self.logits[j * self.k].abs();
        g < TIE
            || (kv == KvType::Q8_0 && g < TIE_REL_KVQ8 * top)
            || (gemm == GemmInput::F16 && g < TIE_REL_GEMM16 * top)
    }
}

/// Coincidencia de top-1 del engine sobre las secuencias de las fixtures en `dir`, para `id`.
fn top1_agree(
    model: &mut Qwen3,
    ctx: &Context,
    dir: &Path,
    greedy: usize,
    k: usize,
) -> (usize, usize) {
    let vocab = model.cfg.vocab;
    let prompt = read_u32(&dir.join("tokens.i32"));
    let g = read_u32(&dir.join("greedy.i32"));
    let seq: Vec<u32> = prompt.iter().chain(&g[..greedy - 1]).copied().collect();
    let logits = logits_prefill(model, ctx, &seq);
    let refk = TopK::load(dir, k);
    let agree = (0..seq.len())
        .filter(|&j| argmax(&logits[j * vocab..(j + 1) * vocab]) == refk.top1(j))
        .count();
    (agree, seq.len())
}

#[test]
#[ignore = "requiere models/qwen3-4b-q4/model.brasa"]
fn forward_kv_f32_igual_a_la_referencia() {
    forward_igual_a_la_referencia(KvType::F32, "fixtures/qwen3-4b-q4", GemmInput::F32);
}

/// Ruta caliente: GEMM de prefill con entradas f16 y KV f16 (ADR 0030).
#[test]
#[ignore = "requiere models/qwen3-4b-q4/model.brasa"]
fn forward_gemm_f16_kv_f16_igual_a_la_referencia() {
    forward_igual_a_la_referencia(KvType::F16, "fixtures/qwen3-4b-q4-kvf16", GemmInput::F16);
}

/// Perfil de agente: GEMM de prefill con entradas f16 y KV Q8 (ADR 0030).
#[test]
#[ignore = "requiere models/qwen3-4b-q4/model.brasa"]
fn forward_gemm_f16_kv_q8_igual_a_la_referencia() {
    forward_igual_a_la_referencia(KvType::Q8_0, "fixtures/qwen3-4b-q4-kvq8", GemmInput::F16);
}

#[test]
#[ignore = "requiere models/qwen3-4b-q4/model.brasa"]
fn forward_kv_f16_igual_a_la_referencia() {
    forward_igual_a_la_referencia(KvType::F16, "fixtures/qwen3-4b-q4-kvf16", GemmInput::F32);
}

#[test]
#[ignore = "requiere models/qwen3-4b-q4/model.brasa"]
fn forward_kv_q8_igual_a_la_referencia() {
    forward_igual_a_la_referencia(KvType::Q8_0, "fixtures/qwen3-4b-q4-kvq8", GemmInput::F32);
}

fn forward_igual_a_la_referencia(kv: KvType, fixtures: &str, gemm: GemmInput) {
    let fx_q4 = root().join(fixtures);
    let fx_fp = root().join("fixtures/qwen3-4b");
    let fx_kv32 = root().join("fixtures/qwen3-4b-q4");
    let logit_tol = match kv {
        KvType::F32 => LOGIT_TOL,
        KvType::F16 => LOGIT_TOL_KV16,
        KvType::Q8_0 => LOGIT_TOL_KVQ8,
    }
    .max(if gemm == GemmInput::F16 {
        LOGIT_TOL_GEMM16
    } else {
        0.0
    });
    let m_q4 = manifest(&fx_q4);
    let (greedy, k) = (
        m_q4["greedy_tokens"].as_u64().unwrap() as usize,
        m_q4["topk"].as_u64().unwrap() as usize,
    );
    let ctx = Context::new().unwrap();
    let limits = Limits {
        ctx: 2048,
        max_tokens: CHUNK,
        max_logit_rows: CHUNK,
        kv,
    };
    let mut model =
        Qwen3::load(&ctx, &root().join("models/qwen3-4b-q4/model.brasa"), limits).unwrap();
    model.prefill_gemm = gemm;
    let vocab = model.cfg.vocab;

    let mut worst_logit = 0f32;
    // Mayor brecha top-1/top-2 (relativa a |top-1|) entre las posiciones con otro top-1.
    let mut worst_gap = 0f32;
    let (mut tf_pos, mut tf_bad, mut tf_ties) = (0usize, 0usize, 0usize);
    let (mut dec_pos, mut dec_bad) = (0usize, 0usize);
    let (mut q_pos, mut q_agree) = (0usize, 0usize);
    let (mut kv_pos, mut kv_agree) = (0usize, 0usize);

    for p in m_q4["prompts"].as_array().unwrap() {
        let id = p["id"].as_str().unwrap();
        let dir = fx_q4.join(id);
        let prompt = read_u32(&dir.join("tokens.i32"));
        let g = read_u32(&dir.join("greedy.i32"));
        let refk = TopK::load(&dir, k);

        // 1a. Teacher forcing por prefill en bloques sobre prompt + greedy[:-1].
        let seq: Vec<u32> = prompt.iter().chain(&g[..greedy - 1]).copied().collect();
        let logits = logits_prefill(&mut model, &ctx, &seq);

        // Logits de la última posición del prompt contra la referencia completa.
        let last = read_f32(&dir.join("last_logits.f32"));
        let row = &logits[(prompt.len() - 1) * vocab..prompt.len() * vocab];
        let max_ref = last.iter().fold(0f32, |a, b| a.max(b.abs()));
        let max_diff = row
            .iter()
            .zip(&last)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        let rel = max_diff / max_ref;
        worst_logit = worst_logit.max(rel);
        assert_eq!(
            argmax(row),
            argmax(&last),
            "{id}: top-1 de la última posición"
        );

        let (mut bad, mut ties) = (0, 0);
        for j in 0..seq.len() {
            if argmax(&logits[j * vocab..(j + 1) * vocab]) != refk.top1(j) {
                worst_gap = worst_gap.max(refk.gap(j) / refk.logits[j * refk.k].abs());
                if refk.tie(j, kv, gemm) {
                    ties += 1;
                } else {
                    bad += 1;
                }
            }
        }

        // 1b. Teacher forcing por decode (GEMV): prefill del prompt y luego de a un token.
        let mut tail = vec![0f32; vocab];
        for (i, chunk) in prompt.chunks(CHUNK).enumerate() {
            let rows = 1;
            model
                .forward(&ctx, chunk, i * CHUNK, rows, &mut tail)
                .unwrap();
        }
        let mut dbad = 0;
        for (step, tok) in g[..greedy - 1].iter().enumerate() {
            let j = prompt.len() - 1 + step; // fila de la referencia que predice g[step]
            if argmax(&tail) != refk.top1(j) && !refk.tie(j, kv, gemm) {
                dbad += 1;
            }
            model
                .forward(&ctx, &[*tok], prompt.len() + step, 1, &mut tail)
                .unwrap();
        }

        // 2. Calidad: misma comparación de top-1 sobre la secuencia de la referencia FP32.
        let (agree, flen) = top1_agree(&mut model, &ctx, &fx_fp.join(id), greedy, k);
        // 3. Pérdida por redondear la KV: contra la referencia con KV sin redondear.
        if kv != KvType::F32 {
            let (a, n) = top1_agree(&mut model, &ctx, &fx_kv32.join(id), greedy, k);
            kv_agree += a;
            kv_pos += n;
        }

        println!(
            "{id:22} logits {rel:.1e} | TF prefill {bad} distintos, {ties} empates de {} | TF decode {dbad} distintos de {} | Q4 vs FP32 top-1 {agree}/{}",
            seq.len(),
            greedy - 1,
            flen
        );
        tf_pos += seq.len();
        tf_bad += bad;
        tf_ties += ties;
        dec_pos += greedy - 1;
        dec_bad += dbad;
        q_pos += flen;
        q_agree += agree;
    }
    println!(
        "KV {}, GEMM de prefill {:?}; mayor brecha relativa con otro top-1: {worst_gap:.2e}",
        kv.name(),
        gemm
    );
    if kv_pos > 0 {
        println!(
            "KV {} vs KV sin redondear: top-1 {kv_agree}/{kv_pos} ({:.2} %)",
            kv.name(),
            100.0 * kv_agree as f64 / kv_pos as f64
        );
    }
    println!(
        "TOTAL: logits peor {worst_logit:.2e} (tol {logit_tol:.0e}); TF prefill {tf_bad} distintos + {tf_ties} empates / {tf_pos}; TF decode {dec_bad} / {dec_pos}; Q4 vs FP32 top-1 {:.1} %",
        100.0 * q_agree as f64 / q_pos as f64
    );
    assert!(worst_logit <= logit_tol, "logits: {worst_logit}");
    assert_eq!(
        tf_bad, 0,
        "teacher forcing (prefill) distinto de la referencia"
    );
    assert_eq!(
        dec_bad, 0,
        "teacher forcing (decode) distinto de la referencia"
    );
    assert!(q_agree as f64 / q_pos as f64 >= MIN_Q4_AGREE, "calidad Q4");
    if kv_pos > 0 {
        assert!(
            kv_agree as f64 / kv_pos as f64 >= MIN_KV_AGREE,
            "pérdida por KV {}",
            kv.name()
        );
    }
}
