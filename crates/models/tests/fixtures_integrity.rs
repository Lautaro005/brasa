//! Integridad de fixtures/qwen3-4b (T0.6, ADR 0003): cada archivo coincide con el sha256 y la
//! forma registrados en el manifest, y el hash del conjunto es reproducible.

use std::path::{Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/qwen3-4b")
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[test]
fn hashes_y_formas_coinciden_con_el_manifest() {
    let root = fixtures();
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("manifest.json")).unwrap())
            .unwrap();
    let greedy = manifest["greedy_tokens"].as_u64().unwrap();
    let topk = manifest["topk"].as_u64().unwrap();
    let mut hashes = Vec::new();
    for p in manifest["prompts"].as_array().unwrap() {
        let id = p["id"].as_str().unwrap();
        let prompt_tokens = p["prompt_tokens"].as_u64().unwrap();
        let rows = prompt_tokens + greedy - 1;
        for (name, f) in p["files"].as_object().unwrap() {
            let bytes = std::fs::read(root.join(id).join(f["file"].as_str().unwrap())).unwrap();
            let h = hex(&Sha256::digest(&bytes));
            assert_eq!(h, f["sha256"].as_str().unwrap(), "{id}/{name}: sha256");
            hashes.push(h);
            if let Some(shape) = f["shape"].as_array() {
                let shape: Vec<u64> = shape.iter().map(|v| v.as_u64().unwrap()).collect();
                assert_eq!(
                    bytes.len() as u64,
                    shape.iter().product::<u64>() * 4,
                    "{id}/{name}: tamaño"
                );
                let expected: Vec<u64> = match name.as_str() {
                    "tokens" => vec![prompt_tokens],
                    "greedy" => vec![greedy],
                    "last_logits" => vec![151_936],
                    "topk_ids" | "topk_logits" => vec![rows, topk],
                    "logsumexp" => vec![rows],
                    other => panic!("archivo inesperado: {other}"),
                };
                assert_eq!(shape, expected, "{id}/{name}: forma");
            }
        }
    }
    hashes.sort();
    assert_eq!(
        hex(&Sha256::digest(hashes.concat().as_bytes())),
        manifest["fixtures_sha256"].as_str().unwrap(),
        "hash del conjunto"
    );
}

#[test]
fn referencia_coincide_con_transformers() {
    // Criterio de aceptación de la fixture (ADR 0003): top-1 casi siempre igual al oficial.
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(fixtures().join("manifest.json")).unwrap())
            .unwrap();
    let c = &manifest["crosscheck"];
    let agree = c["top1_agree"].as_f64().unwrap() / c["positions"].as_f64().unwrap();
    assert!(agree >= 0.98, "coincidencia top-1 {agree}");
}
