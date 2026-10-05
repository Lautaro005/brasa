//! Modelos locales: carpetas con `model.brasa` bajo la carpeta de modelos.

use std::path::{Path, PathBuf};

use brasa_quant::BrasaFile;

use crate::{Error, Result};

/// Modelo `.brasa` encontrado en disco.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalModel {
    pub name: String,
    pub dir: PathBuf,
    pub weights: PathBuf,
    pub bytes: u64,
    /// sha256 agregado del encabezado (ADR 0006).
    pub data_sha256: String,
    /// Si el encabezado abre y es estructuralmente válido.
    pub ok: bool,
    pub error: Option<String>,
}

fn open(dir: &Path) -> LocalModel {
    let name = dir
        .file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    let weights = dir.join("model.brasa");
    let bytes = std::fs::metadata(&weights).map_or(0, |m| m.len());
    match BrasaFile::open(&weights) {
        Ok(f) => LocalModel {
            name,
            dir: dir.to_path_buf(),
            weights,
            bytes,
            data_sha256: f.data_sha256().to_string(),
            ok: true,
            error: None,
        },
        Err(e) => LocalModel {
            name,
            dir: dir.to_path_buf(),
            weights,
            bytes,
            data_sha256: String::new(),
            ok: false,
            error: Some(e.0),
        },
    }
}

/// Escanea `models_dir` en busca de carpetas con `model.brasa`.
pub fn scan(models_dir: &Path) -> Vec<LocalModel> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(models_dir) else {
        return out;
    };
    for e in entries.flatten() {
        let dir = e.path();
        if dir.join("model.brasa").is_file() {
            out.push(open(&dir));
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// Resuelve `name` como ruta directa o como carpeta dentro de `models_dir`.
pub fn resolve(models_dir: &Path, name: &str) -> Result<PathBuf> {
    let direct = Path::new(name);
    if direct.join("model.brasa").is_file() {
        return Ok(direct.to_path_buf());
    }
    let p = models_dir.join(name);
    if p.join("model.brasa").is_file() {
        return Ok(p);
    }
    Err(Error(format!(
        "no se encontró el modelo {name:?} (buscado en {} y como ruta)",
        models_dir.display()
    )))
}
