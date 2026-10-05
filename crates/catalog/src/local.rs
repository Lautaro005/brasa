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

/// Resuelve `name` como ruta directa o como carpeta dentro de `models_dir`. Pensado para lecturas
/// (por ejemplo `brasa models verify`): acepta rutas, pero nunca se usa para borrar.
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

/// Resuelve `name` como una **subcarpeta directa** de `models_dir`, para operaciones que borran
/// (`brasa rm`). Rechaza rutas absolutas, separadores, `.`/`..` y symlinks: canonicaliza y exige
/// que el destino sea exactamente `models_dir/<name>` dentro de la carpeta de modelos.
pub fn resolve_child(models_dir: &Path, name: &str) -> Result<PathBuf> {
    let comp = Path::new(name);
    if name.is_empty()
        || comp.is_absolute()
        || comp.components().count() != 1
        || matches!(name, "." | "..")
    {
        return Err(Error(format!(
            "el nombre {name:?} tiene que ser una carpeta dentro de {} (sin rutas ni `..`)",
            models_dir.display()
        )));
    }
    let base = std::fs::canonicalize(models_dir)
        .map_err(|e| Error(format!("{}: {e}", models_dir.display())))?;
    let want = base.join(name);
    let canon = std::fs::canonicalize(&want).map_err(|_| {
        Error(format!(
            "no se encontró el modelo {name:?} en {}",
            base.display()
        ))
    })?;
    if canon != want {
        return Err(Error(format!(
            "{name}: es un symlink; `brasa rm` solo borra carpetas reales dentro de {}",
            base.display()
        )));
    }
    if canon.parent() != Some(base.as_path()) || !canon.join("model.brasa").is_file() {
        return Err(Error(format!(
            "{name}: no es una carpeta de modelo dentro de {}",
            base.display()
        )));
    }
    Ok(canon)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn modelo(dir: &Path, name: &str) {
        let d = dir.join(name);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("model.brasa"), b"x").unwrap();
    }

    #[test]
    fn resolve_child_acepta_subcarpeta_directa() {
        let tmp = tempfile::tempdir().unwrap();
        modelo(tmp.path(), "qwen3-4b-q4");
        let p = resolve_child(tmp.path(), "qwen3-4b-q4").unwrap();
        assert_eq!(p, tmp.path().canonicalize().unwrap().join("qwen3-4b-q4"));
    }

    #[test]
    fn resolve_child_rechaza_escapes() {
        let tmp = tempfile::tempdir().unwrap();
        modelo(tmp.path(), "qwen3-4b-q4");
        for name in ["..", ".", "", "../qwen3-4b-q4", "qwen3-4b-q4/sub", "/etc"] {
            assert!(
                resolve_child(tmp.path(), name).is_err(),
                "aceptó {name:?} y no debería"
            );
        }
    }

    #[test]
    fn resolve_child_rechaza_symlinks() {
        let tmp = tempfile::tempdir().unwrap();
        modelo(tmp.path(), "qwen3-4b-q4");
        let fuera = tmp.path().join("otro");
        std::fs::create_dir_all(&fuera).unwrap();
        std::fs::write(fuera.join("model.brasa"), b"x").unwrap();
        std::os::unix::fs::symlink(&fuera, tmp.path().join("afuera")).unwrap();
        std::os::unix::fs::symlink(tmp.path().join("qwen3-4b-q4"), tmp.path().join("alias"))
            .unwrap();
        assert!(resolve_child(tmp.path(), "afuera").is_err());
        assert!(resolve_child(tmp.path(), "alias").is_err());
    }

    #[test]
    fn resolve_child_rechaza_carpeta_sin_pesos() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("vacia")).unwrap();
        assert!(resolve_child(tmp.path(), "vacia").is_err());
    }
}
