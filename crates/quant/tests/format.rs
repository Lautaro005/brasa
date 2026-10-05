//! Formato `.brasa` con un archivo chico armado a mano (no necesita los pesos): lectura,
//! alineación, verificación de hashes y detección de alteraciones.

use brasa_quant::{BrasaFile, QType, dequantize};
use sha2::{Digest, Sha256};

const PAGE: usize = 16384;

fn hex(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

/// Un tensor f32 [2, 32] y uno q8_0 [1, 32] (d = 1, q = j - 16).
fn build() -> Vec<u8> {
    let f32_data: Vec<u8> = (0..64).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let mut q8 = vec![0x00u8, 0x3c];
    q8.extend((0..32).map(|j| (j as i8 - 16) as u8));
    let (h1, h2) = (hex(&f32_data), hex(&q8));
    let all = hex(format!("{h1}{h2}").as_bytes());
    let json = format!(
        r#"{{"format":"brasa","version":1,"converter":"test","model":{{"family":"qwen3","source_repo":"x","source_commit":"y","config":{{}}}},"quant":{{}},"page":{PAGE},"tensors":[{{"name":"a","dtype":"f32","shape":[2,32],"nbytes":256,"offset":{PAGE},"sha256":"{h1}"}},{{"name":"b","dtype":"q8_0","shape":[1,32],"nbytes":34,"offset":{},"sha256":"{h2}"}}],"data_sha256":"{all}"}}"#,
        2 * PAGE
    );
    let mut f = b"BRSA".to_vec();
    f.extend(1u32.to_le_bytes());
    f.extend((json.len() as u64).to_le_bytes());
    f.extend(json.as_bytes());
    f.resize(PAGE, 0);
    f.extend(&f32_data);
    f.resize(2 * PAGE, 0);
    f.extend(&q8);
    f.resize(3 * PAGE, 0);
    f
}

fn write(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("brasa-test-{}-{name}.brasa", std::process::id()));
    std::fs::write(&p, bytes).unwrap();
    p
}

#[test]
fn lee_y_verifica() {
    let p = write("ok", &build());
    let f = BrasaFile::open(&p).unwrap();
    f.verify().unwrap();
    let b = f.tensor("b").unwrap();
    assert_eq!(b.dtype, QType::Q8_0);
    assert_eq!(
        f.data(b).as_ptr() as usize % PAGE,
        0,
        "tensor no alineado a página"
    );
    let mut out = [0f32; 32];
    dequantize(b.dtype, f.data(b), &mut out);
    assert_eq!(out[0], -16.0);
    assert_eq!(out[31], 15.0);
    std::fs::remove_file(p).ok();
}

#[test]
fn detecta_alteracion_de_datos() {
    let mut bytes = build();
    bytes[PAGE + 5] ^= 1;
    let p = write("alterado", &bytes);
    let err = BrasaFile::open(&p).unwrap().verify().unwrap_err();
    assert!(err.0.contains("a: sha256"), "{err}");
    std::fs::remove_file(p).ok();
}

#[test]
fn rechaza_archivos_invalidos() {
    let mut bytes = build();
    bytes[0] = b'X';
    let p = write("magic", &bytes);
    assert!(BrasaFile::open(&p).is_err());
    std::fs::remove_file(p).ok();
    let p = write("truncado", &build()[..2 * PAGE]);
    assert!(
        BrasaFile::open(&p)
            .unwrap_err()
            .0
            .contains("fuera del archivo")
    );
    std::fs::remove_file(p).ok();
}
