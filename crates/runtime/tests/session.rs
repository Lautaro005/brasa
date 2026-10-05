//! T1.7 de punta a punta: en greedy, `Session` reproduce la continuación de la referencia Q4
//! (fixtures/qwen3-4b-q4) y la reutilización del prefijo no cambia el resultado.
//!
//!   cargo test --release -p brasa-runtime --test session -- --ignored --nocapture

use std::path::{Path, PathBuf};

use brasa_runtime::{KvType, Limits, Sampler, SamplingParams, Session};
use serde_json::Value;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
#[ignore = "requiere models/qwen3-4b-q4"]
fn greedy_reproduce_la_referencia_y_reutiliza_prefijo() {
    let fx = root().join("fixtures/qwen3-4b-q4");
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(fx.join("manifest.json")).unwrap()).unwrap();
    // Bloques de prefill chicos para que los prompts se partan en varios.
    let limits = Limits {
        ctx: 1024,
        max_tokens: 16,
        max_logit_rows: 1,
        kv: KvType::F16,
    };
    let mut s = Session::load(&root().join("models/qwen3-4b-q4"), limits).unwrap();
    let greedy = manifest["greedy_tokens"].as_u64().unwrap() as usize;

    for p in manifest["prompts"].as_array().unwrap() {
        let id = p["id"].as_str().unwrap();
        let expected = p["greedy_text"].as_str().unwrap();
        // Solo prompts cuya continuación no pasa por un token de parada (la sesión se detiene ahí).
        if expected.contains("<|im_end|>") || p["prompt_tokens"].as_u64().unwrap() > 512 {
            continue;
        }
        let text = std::fs::read_to_string(fx.join(id).join("prompt.txt")).unwrap();
        let ids = s.tokenizer().encode(&text);
        let run = |s: &mut Session| {
            let mut out = String::new();
            let mut sampler = Sampler::new(SamplingParams::greedy(), s.vocab());
            let st = s
                .generate(&ids, greedy, &mut sampler, |t| {
                    out.push_str(t);
                    true
                })
                .unwrap();
            (out, st)
        };
        let (first, st1) = run(&mut s);
        assert_eq!(first, expected, "{id}: greedy distinto de la referencia");
        // Mismo prompt otra vez: reutiliza todo menos el último token y da lo mismo.
        let (second, st2) = run(&mut s);
        assert_eq!(
            st2.reused_tokens,
            ids.len() - 1,
            "{id}: no reutilizó el prefijo"
        );
        assert_eq!(second, first, "{id}: la reutilización cambió la salida");
        println!(
            "{id}: ok ({} tokens; prefill {:.0} ms, con prefijo en caché {:.0} ms)",
            ids.len(),
            st1.prefill_ms,
            st2.prefill_ms
        );
    }
}
