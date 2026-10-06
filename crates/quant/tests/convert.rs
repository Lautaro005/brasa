//! Aceptación de U4: el `.brasa` generado por `brasa convert` tiene los mismos sha256 por tensor
//! que el de `tools/convert_brasa.py`. Convierte ~4 GB; se corre con `--ignored` y no toca GPU.

use std::path::Path;

use brasa_quant::BrasaFile;
use brasa_quant::convert::convert;

#[test]
#[ignore = "convierte ~4 GB y compara contra el .brasa de Python; correr con --ignored"]
fn conversion_coincide_con_el_conversor_python() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let hf = root.join("models/qwen3-4b-hf");
    let reference = root.join("models/qwen3-4b-q4/model.brasa");
    if !hf.join("config.json").is_file() || !reference.is_file() {
        eprintln!("sin pesos de referencia, se omite");
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    let r = convert(
        &hf,
        tmp.path(),
        "Qwen/Qwen3-4B",
        "1cfa9a7208912126459214e8b04321603b3df60c",
    )
    .unwrap();
    let ours = BrasaFile::open(&tmp.path().join("model.brasa")).unwrap();
    let py = BrasaFile::open(&reference).unwrap();
    assert_eq!(
        ours.tensors().len(),
        py.tensors().len(),
        "distinta cantidad"
    );
    let mut diffs = Vec::new();
    for t in py.tensors() {
        match ours.tensor(&t.name) {
            Some(o) if o.sha256 == t.sha256 => {}
            Some(_) => diffs.push(t.name.clone()),
            None => diffs.push(format!("{} (falta)", t.name)),
        }
    }
    assert!(diffs.is_empty(), "tensores distintos: {diffs:?}");
    assert_eq!(ours.data_sha256(), py.data_sha256(), "el conjunto difiere");
    eprintln!("ok: {} tensores, data_sha256 {}", r.tensors, r.data_sha256);
}
