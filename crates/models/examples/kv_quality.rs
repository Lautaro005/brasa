//! Calidad de la KV cache (ADR 0009 y 0033) frente a la referencia FP32 sin redondear
//! (fixtures/qwen3-4b-q4, mismos pesos decuantizados). Para cada tipo de KV pedido, en la misma
//! build y con el mismo modelo:
//! - generación greedy de 32 tokens por prompt (decode de a un token, como en el agente):
//!   longitud del prefijo que coincide con la referencia y coincidencias totales;
//! - error de logits de la última posición del prompt (prefill), relativo a max|ref|;
//! - coincidencia de top-1 en teacher forcing sobre prompt + greedy[:-1] (camino de prefill).
//!
//!   cargo run --release -p brasa-models --example kv_quality -- f16 q8_0 tq4
//!
//! Necesita models/qwen3-4b-q4/model.brasa y fixtures/qwen3-4b-q4. Los números van a
//! docs/bench/ con chip y commit: no son una prueba de regresión.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use brasa_metal::Context;
use brasa_models::qwen3::{KvType, Limits, Qwen3};
use serde_json::Value;

const CHUNK: usize = 64;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read_f32(p: &Path) -> Vec<f32> {
    fs::read(p)
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn read_u32(p: &Path) -> Vec<u32> {
    fs::read(p)
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, x) in v.iter().enumerate() {
        if *x > v[best] {
            best = i;
        }
    }
    best as u32
}

fn main() {
    let kvs: Vec<KvType> = {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let names = if args.is_empty() {
            vec!["f16".to_string(), "q8_0".to_string(), "tq4".to_string()]
        } else {
            args
        };
        names
            .iter()
            .map(|s| KvType::parse(s).unwrap_or_else(|| panic!("kv: {s}")))
            .collect()
    };
    let fx = root().join("fixtures/qwen3-4b-q4");
    let manifest: Value =
        serde_json::from_slice(&fs::read(fx.join("manifest.json")).unwrap()).unwrap();
    let greedy_n = manifest["greedy_tokens"].as_u64().unwrap() as usize;
    let topk = manifest["topk"].as_u64().unwrap() as usize;
    let prompts: Vec<String> = manifest["prompts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["id"].as_str().unwrap().to_string())
        .collect();
    let ctx = Context::new().unwrap();
    println!(
        "{:<6} {:>9} {:>11} {:>13} {:>13} {:>12}   prefijo por prompt",
        "kv", "KV@2K GiB", "greedy 32/32", "tokens iguales", "err. logits", "top-1 TF"
    );
    for kv in kvs {
        let limits = Limits {
            ctx: 2048,
            max_tokens: CHUNK,
            max_logit_rows: CHUNK,
            kv,
        };
        let mut model =
            Qwen3::load(&ctx, &root().join("models/qwen3-4b-q4/model.brasa"), limits).unwrap();
        let vocab = model.cfg.vocab;
        let kv_gib = model.allocated().kv as f64 / (1u64 << 30) as f64;
        let (mut full, mut same, mut total) = (0usize, 0usize, 0usize);
        let (mut worst_logit, mut top1_ok, mut top1_n) = (0f32, 0usize, 0usize);
        let mut prefixes = Vec::new();
        let t0 = Instant::now();
        for id in &prompts {
            let dir = fx.join(id);
            let prompt = read_u32(&dir.join("tokens.i32"));
            let g = read_u32(&dir.join("greedy.i32"));
            let last_ref = read_f32(&dir.join("last_logits.f32"));

            // Prefill del prompt; logits de la última posición.
            let mut logits = vec![0f32; vocab];
            for (i, chunk) in prompt.chunks(CHUNK).enumerate() {
                model
                    .forward(&ctx, chunk, i * CHUNK, 1, &mut logits)
                    .unwrap();
            }
            let max_ref = last_ref.iter().fold(0f32, |a, b| a.max(b.abs()));
            let diff = logits
                .iter()
                .zip(&last_ref)
                .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
            worst_logit = worst_logit.max(diff / max_ref);

            // Generación greedy de 32 tokens, de a uno.
            let mut tok = argmax(&logits);
            let mut generado = vec![tok];
            for s in 1..greedy_n {
                model
                    .forward(&ctx, &[tok], prompt.len() + s - 1, 1, &mut logits)
                    .unwrap();
                tok = argmax(&logits);
                generado.push(tok);
            }
            let prefix = generado.iter().zip(&g).take_while(|(a, b)| a == b).count();
            full += (prefix == greedy_n) as usize;
            same += generado.iter().zip(&g).filter(|(a, b)| a == b).count();
            total += greedy_n;
            prefixes.push(prefix);

            // Teacher forcing: prompt + greedy[:-1] por prefill, top-1 contra la referencia.
            let seq: Vec<u32> = prompt.iter().chain(&g[..greedy_n - 1]).copied().collect();
            let refk = read_u32(&dir.join("topk_ids.i32"));
            let mut out = vec![0f32; CHUNK * vocab];
            for (i, chunk) in seq.chunks(CHUNK).enumerate() {
                let rows = chunk.len();
                model
                    .forward(&ctx, chunk, i * CHUNK, rows, &mut out[..rows * vocab])
                    .unwrap();
                for j in 0..rows {
                    let pos = i * CHUNK + j;
                    let row = &out[j * vocab..(j + 1) * vocab];
                    top1_ok += (argmax(row) == refk[pos * topk]) as usize;
                    top1_n += 1;
                }
            }
        }
        let secs = t0.elapsed().as_secs_f64();
        let prefix_str: Vec<String> = prefixes.iter().map(|p| p.to_string()).collect();
        println!(
            "{:<6} {:>9.3} {:>8}/{:<2} {:>12.1}% {:>12.2e} {:>11.2}%   {}",
            kv.name(),
            kv_gib,
            full,
            prompts.len(),
            100.0 * same as f64 / total as f64,
            worst_logit,
            100.0 * top1_ok as f64 / top1_n as f64,
            prefix_str.join(" ")
        );
        eprintln!("  ({} en {secs:.1} s)", kv.name());
    }
}
