//! Aceptación de U3: `brasa models verify qwen3-4b-q4` pasa con los pesos actuales. No carga
//! GPU; solo lee el `.brasa` y recalcula sus sha256. Se omite si los pesos no están (CI).

use std::path::Path;

#[test]
#[ignore = "lee los pesos reales (~2 GiB); correr con --ignored"]
fn verifica_el_modelo_real() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen3-4b-q4");
    let weights = dir.join("model.brasa");
    if !weights.is_file() {
        eprintln!("sin pesos en {}, se omite", dir.display());
        return;
    }
    let r = brasa_catalog::verify::verify(&weights).unwrap();
    assert!(r.tensors > 0);
    assert_eq!(r.data_sha256.len(), 64);
    eprintln!(
        "ok: {} tensores, {} bytes, sha256 {}",
        r.tensors, r.bytes, r.data_sha256
    );
}
