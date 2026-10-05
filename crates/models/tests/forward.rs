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
//! Necesita models/qwen3-4b-q4/model.brasa:
//!   cargo test --release -p brasa-models --test forward -- --ignored --nocapture

use std::path::{Path, PathBuf};

use brasa_metal::Context;
use brasa_models::qwen3::{Limits, Qwen3};
use serde_json::Value;

const LOGIT_TOL: f32 = 1e-4;
const TIE: f32 = 1e-3;
const MIN_Q4_AGREE: f64 = 0.80;
const CHUNK: usize = 64;

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
}

#[test]
#[ignore = "requiere models/qwen3-4b-q4/model.brasa"]
fn forward_completo_igual_a_la_referencia() {
    let fx_q4 = root().join("fixtures/qwen3-4b-q4");
    let fx_fp = root().join("fixtures/qwen3-4b");
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
    };
    let mut model =
        Qwen3::load(&ctx, &root().join("models/qwen3-4b-q4/model.brasa"), limits).unwrap();
    let vocab = model.cfg.vocab;

    let mut worst_logit = 0f32;
    let (mut tf_pos, mut tf_bad, mut tf_ties) = (0usize, 0usize, 0usize);
    let (mut dec_pos, mut dec_bad) = (0usize, 0usize);
    let (mut q_pos, mut q_agree) = (0usize, 0usize);

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
                if refk.gap(j) < TIE {
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
            if argmax(&tail) != refk.top1(j) && refk.gap(j) >= TIE {
                dbad += 1;
            }
            model
                .forward(&ctx, &[*tok], prompt.len() + step, 1, &mut tail)
                .unwrap();
        }

        // 2. Calidad: misma comparación de top-1 sobre la secuencia de la referencia FP32.
        let fdir = fx_fp.join(id);
        let fprompt = read_u32(&fdir.join("tokens.i32"));
        let fg = read_u32(&fdir.join("greedy.i32"));
        let fseq: Vec<u32> = fprompt.iter().chain(&fg[..greedy - 1]).copied().collect();
        let flog = logits_prefill(&mut model, &ctx, &fseq);
        let fk = TopK::load(&fdir, k);
        let agree = (0..fseq.len())
            .filter(|&j| argmax(&flog[j * vocab..(j + 1) * vocab]) == fk.top1(j))
            .count();

        println!(
            "{id:22} logits {rel:.1e} | TF prefill {bad} distintos, {ties} empates de {} | TF decode {dbad} distintos de {} | Q4 vs FP32 top-1 {agree}/{}",
            seq.len(),
            greedy - 1,
            fseq.len()
        );
        tf_pos += seq.len();
        tf_bad += bad;
        tf_ties += ties;
        dec_pos += greedy - 1;
        dec_bad += dbad;
        q_pos += fseq.len();
        q_agree += agree;
    }
    println!(
        "TOTAL: logits peor {worst_logit:.2e} (tol {LOGIT_TOL:.0e}); TF prefill {tf_bad} distintos + {tf_ties} empates / {tf_pos}; TF decode {dec_bad} / {dec_pos}; Q4 vs FP32 top-1 {:.1} %",
        100.0 * q_agree as f64 / q_pos as f64
    );
    assert!(worst_logit <= LOGIT_TOL, "logits: {worst_logit}");
    assert_eq!(
        tf_bad, 0,
        "teacher forcing (prefill) distinto de la referencia"
    );
    assert_eq!(
        dec_bad, 0,
        "teacher forcing (decode) distinto de la referencia"
    );
    assert!(q_agree as f64 / q_pos as f64 >= MIN_Q4_AGREE, "calidad Q4");
}
