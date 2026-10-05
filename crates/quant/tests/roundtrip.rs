//! Criterio de T1.3 (ADR 0006): decuantizar cada tensor de `models/qwen3-4b-q4/model.brasa` y
//! compararlo con los pesos originales de `models/qwen3-4b-hf` dentro de la tolerancia documentada.
//!
//! Necesita los pesos (no están en el repo), por eso es `#[ignore]`. Correr con:
//!   tools/convert_brasa.py models/qwen3-4b-hf models/qwen3-4b-q4
//!   cargo test --release -p brasa-quant --test roundtrip -- --ignored --nocapture

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brasa_quant::safetensors::SafeTensors;
use brasa_quant::{BrasaFile, QType, dequantize, f16_to_f32};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Margen por el redondeo de la escala a f16 (ADR 0006).
const EPS: f32 = 1.0 / 1024.0;

#[test]
#[ignore = "requiere models/qwen3-4b-hf y models/qwen3-4b-q4"]
fn decuantizado_dentro_de_tolerancia() {
    let hf = root().join("models/qwen3-4b-hf");
    let brasa = BrasaFile::open(&root().join("models/qwen3-4b-q4/model.brasa")).unwrap();
    brasa.verify().unwrap();
    println!(
        "{} tensores, {:.2} GiB, data_sha256 {}",
        brasa.tensors().len(),
        brasa.weights_bytes() as f64 / (1u64 << 30) as f64,
        brasa.data_sha256()
    );

    let index: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(hf.join("model.safetensors.index.json")).unwrap(),
    )
    .unwrap();
    let mut files: HashMap<String, SafeTensors> = HashMap::new();
    let mut worst_rel: HashMap<&str, (f32, String)> = HashMap::new();
    let mut sum_rel: HashMap<&str, (f64, usize)> = HashMap::new();

    for t in brasa.tensors() {
        let file = index["weight_map"][&t.name].as_str().unwrap().to_string();
        let st = files
            .entry(file.clone())
            .or_insert_with(|| SafeTensors::open(&hf.join(&file)).unwrap());
        assert_eq!(st.shape(&t.name).unwrap(), t.shape.as_slice(), "{}", t.name);
        let orig = st.to_f32(&t.name).unwrap();
        let mut deq = vec![0f32; t.numel()];
        let data = brasa.data(t);
        dequantize(t.dtype, data, &mut deq);

        match t.dtype {
            QType::F32 => assert_eq!(orig, deq, "{}: f32 debe ser exacto", t.name),
            QType::Q4_0 | QType::Q8_0 => {
                let bb = if t.dtype == QType::Q4_0 { 18 } else { 34 };
                for (bi, (o, d)) in orig.chunks(32).zip(deq.chunks(32)).enumerate() {
                    let b = &data[bi * bb..];
                    let scale = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
                    for (w, wq) in o.iter().zip(d) {
                        let err = (w - wq).abs();
                        // q4_0: el extremo opuesto puede saturar en q = 15 (ŵ = 7·d).
                        let saturated = t.dtype == QType::Q4_0
                            && scale != 0.0
                            && (wq / scale - 7.0).abs() < 1e-3;
                        let bound = if saturated {
                            scale.abs()
                        } else {
                            scale.abs() / 2.0
                        } * (1.0 + EPS);
                        assert!(
                            err <= bound,
                            "{} bloque {bi}: |w - ŵ| = {err} > {bound} (w = {w}, ŵ = {wq}, d = {scale})",
                            t.name
                        );
                    }
                }
                let (mut se, mut ss) = (0f64, 0f64);
                for (w, wq) in orig.iter().zip(&deq) {
                    se += ((w - wq) as f64).powi(2);
                    ss += (*w as f64).powi(2);
                }
                let rel = (se / ss).sqrt() as f32;
                let key = if t.dtype == QType::Q4_0 {
                    "q4_0"
                } else {
                    "q8_0"
                };
                let w = worst_rel.entry(key).or_insert((0.0, String::new()));
                if rel > w.0 {
                    *w = (rel, t.name.clone());
                }
                let s = sum_rel.entry(key).or_insert((0.0, 0));
                s.0 += rel as f64;
                s.1 += 1;
            }
        }
    }
    for (k, (rel, name)) in &worst_rel {
        let (sum, n) = sum_rel[k];
        println!(
            "{k}: {n} tensores, error relativo (RMSE/RMS) medio {:.4}, peor {rel:.4} en {name}",
            sum / n as f64
        );
    }
}
