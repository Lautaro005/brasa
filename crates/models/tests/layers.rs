//! Criterio de T1.5: la salida de una capa (y del apilado completo) en GPU coincide con la
//! referencia FP32 de Python que usa los mismos pesos decuantizados (fixtures/qwen3-4b-q4).
//!
//! Necesita models/qwen3-4b-q4/model.brasa (no está en el repo):
//!   cargo test --release -p brasa-models --test layers -- --ignored --nocapture
//!
//! Tolerancia (error relativo L2 por token, ‖gpu − ref‖ / ‖ref‖): capa 0 ≤ 1e-5;
//! 36 capas ≤ 1e-4. Ambos lados calculan en f32 con distinto orden de suma; el error crece con
//! la profundidad.

use std::path::{Path, PathBuf};

use brasa_metal::Context;
use brasa_models::qwen3::{KvType, Limits, PrefillPrecision, Qwen3};
use serde_json::Value;

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

/// Peor error relativo L2 entre filas de largo `h`.
fn worst_row_rel(got: &[f32], expected: &[f32], h: usize) -> f64 {
    got.chunks(h)
        .zip(expected.chunks(h))
        .map(|(g, e)| {
            let num: f64 = g.iter().zip(e).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
            let den: f64 = e.iter().map(|b| (*b as f64).powi(2)).sum();
            (num / den).sqrt()
        })
        .fold(0.0, f64::max)
}

#[test]
#[ignore = "requiere models/qwen3-4b-q4/model.brasa"]
fn capas_iguales_a_la_referencia() {
    let fx = root().join("fixtures/qwen3-4b-q4");
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(fx.join("manifest.json")).unwrap()).unwrap();
    let ctx = Context::new().unwrap();
    let limits = Limits {
        ctx: 256,
        max_tokens: 64,
        max_logit_rows: 1,
        kv: KvType::F32,
    };
    let mut model =
        Qwen3::load(&ctx, &root().join("models/qwen3-4b-q4/model.brasa"), limits).unwrap();
    // Tolerancias de T1.5 (KV f32): con los kernels de prefill exactos. La ruta f16 se valida en
    // el forward (tests/forward.rs, ADR 0030).
    model.prefill_precision = PrefillPrecision::F32;
    let h = model.cfg.hidden;
    let layers = model.cfg.layers;

    let (mut worst0, mut worst_all) = (0f64, 0f64);
    for p in manifest["prompts"].as_array().unwrap() {
        if p["files"].get("embed").is_none() {
            continue;
        }
        let dir = fx.join(p["id"].as_str().unwrap());
        let x = read_f32(&dir.join("embed.f32"));
        let l0 = model.run_layers(&ctx, &x, 0, 0..1).unwrap();
        let e0 = worst_row_rel(&l0, &read_f32(&dir.join("layer_00.f32")), h);
        let all = model.run_layers(&ctx, &x, 0, 0..layers).unwrap();
        let e_all = worst_row_rel(
            &all,
            &read_f32(&dir.join(format!("layer_{:02}.f32", layers - 1))),
            h,
        );
        println!("{}: capa 0 {e0:.2e}, {layers} capas {e_all:.2e}", p["id"]);
        worst0 = worst0.max(e0);
        worst_all = worst_all.max(e_all);
    }
    assert!(worst0 <= 1e-5, "capa 0: {worst0}");
    assert!(worst_all <= 1e-4, "{layers} capas: {worst_all}");
}
