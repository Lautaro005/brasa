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
/// que el destino sea exactamente `models_dir/<name>` dentro de la carpeta de modelos. También
/// rechaza una carpeta de modelos alcanzada por un symlink (ver [`resolve_child_with`]).
pub fn resolve_child(models_dir: &Path, name: &str) -> Result<PathBuf> {
    resolve_child_with(models_dir, name, false)
}

/// Como [`resolve_child`], pero `seguir_base` permite que `models_dir` pase por un symlink. Por
/// defecto no se sigue: en un worktree `models -> ../brasa/models` un `rm` borraría los pesos
/// compartidos. Los firmlinks del sistema (`/var -> /private/var`, etc.) no cuentan.
pub fn resolve_child_with(models_dir: &Path, name: &str, seguir_base: bool) -> Result<PathBuf> {
    if !seguir_base {
        if let Some(link) = primer_symlink_del_usuario(models_dir) {
            let real = std::fs::canonicalize(models_dir).unwrap_or_else(|_| link.clone());
            return Err(Error(format!(
                "la carpeta de modelos {} pasa por el symlink {} -> {}; `brasa rm` no lo sigue \
                 (usá --seguir-symlink-base para borrar en la ruta real)",
                models_dir.display(),
                link.display(),
                real.display()
            )));
        }
    }
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

/// Firmlinks de macOS: no son symlinks que haya puesto el usuario.
fn es_firmlink_del_sistema(p: &Path) -> bool {
    matches!(
        p.to_str(),
        Some(
            "/var"
                | "/tmp"
                | "/etc"
                | "/private"
                | "/private/var"
                | "/private/tmp"
                | "/private/etc"
        )
    )
}

/// Primer componente del camino (resuelto contra el cwd si es relativo) que es un symlink y no es
/// un firmlink del sistema. `None` si la carpeta es real. Lo usan también los borrados de
/// `storage` (ADR 0034).
pub(crate) fn primer_symlink_del_usuario(path: &Path) -> Option<PathBuf> {
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().ok()?.join(path)
    };
    let mut prefix = PathBuf::new();
    for c in abs.components() {
        prefix.push(c);
        if es_firmlink_del_sistema(&prefix) {
            continue;
        }
        if std::fs::symlink_metadata(&prefix)
            .map(|m| m.file_type().is_symlink())
            .unwrap_or(false)
        {
            return Some(prefix);
        }
    }
    None
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

    #[test]
    fn resolve_child_rechaza_base_que_es_symlink() {
        let tmp = tempfile::tempdir().unwrap();
        let real = tmp.path().join("real");
        modelo(&real, "m");
        let ws = tmp.path().join("ws");
        std::fs::create_dir_all(&ws).unwrap();
        let base = ws.join("models");
        std::os::unix::fs::symlink(&real, &base).unwrap();

        // Sin el flag: se niega y no toca los pesos reales.
        let err = resolve_child(&base, "m").unwrap_err();
        assert!(err.0.contains("symlink"), "{}", err.0);
        assert!(real.join("m/model.brasa").is_file());

        // Con el flag: resuelve a la ruta real y se puede borrar.
        let p = resolve_child_with(&base, "m", true).unwrap();
        assert_eq!(p, real.canonicalize().unwrap().join("m"));
        std::fs::remove_dir_all(&p).unwrap();
        assert!(!real.join("m").exists());
    }

    #[test]
    fn resolve_child_rechaza_base_con_symlink_intermedio() {
        let tmp = tempfile::tempdir().unwrap();
        let real = tmp.path().join("real");
        modelo(&real, "m");
        let link = tmp.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        // `link` es un symlink intermedio: la base `link/sub` no existe como real.
        std::fs::create_dir_all(real.join("sub")).unwrap();
        assert!(resolve_child(&link.join("sub"), "m").is_err());
    }
}
