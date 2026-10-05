//! Manifiestos de modelo en TOML (ADR 0020): repo y revisión de Hugging Face, archivos con
//! sha256, cuantización destino, contexto máximo y licencia. Los de fábrica van embebidos en el
//! binario; además se pueden leer de una carpeta (`$BRASA_CATALOG`).

use std::path::{Component, Path};

use serde::Deserialize;

use crate::{Error, Result};

/// Comprueba que `path` sea relativa y sin `.` ni `..` (defensa contra manifiestos externos).
pub fn safe_relative(path: &str) -> std::result::Result<(), String> {
    if path.is_empty() {
        return Err("ruta vacía".into());
    }
    let p = Path::new(path);
    if p.is_absolute() {
        return Err(format!("ruta absoluta no permitida: {path:?}"));
    }
    for c in p.components() {
        if !matches!(c, Component::Normal(_)) {
            return Err(format!("ruta no permitida (`.` o `..`): {path:?}"));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct FileSpec {
    /// Ruta dentro del repo de Hugging Face.
    pub path: String,
    /// sha256 del contenido del archivo.
    pub sha256: String,
    #[serde(default)]
    pub size: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Manifest {
    pub name: String,
    pub family: String,
    pub source_repo: String,
    pub source_revision: String,
    /// Carpeta local donde `brasa pull` deja los safetensors.
    pub hf_dir: String,
    #[serde(default)]
    pub quant: String,
    pub max_context: usize,
    #[serde(default)]
    pub license: String,
    pub files: Vec<FileSpec>,
}

/// Manifiestos embebidos en el binario.
const BUILTIN: &[&str] = &[include_str!("../manifests/qwen3-4b-q4.toml")];

impl Manifest {
    pub fn parse(text: &str) -> Result<Self> {
        let m: Self =
            toml::from_str(text).map_err(|e| Error(format!("manifiesto inválido: {e}")))?;
        m.validate()?;
        Ok(m)
    }

    /// Valida los campos que el resto del código asume: sha256 de 64 hex y rutas seguras.
    fn validate(&self) -> Result<()> {
        for f in &self.files {
            if f.sha256.len() != 64 || !f.sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(Error(format!(
                    "{}: sha256 inválido en {:?} (se esperan 64 dígitos hex)",
                    self.name, f.path
                )));
            }
            safe_relative(&f.path).map_err(|e| Error(format!("{}: {e}", self.name)))?;
        }
        Ok(())
    }

    pub fn builtin() -> Result<Vec<Self>> {
        BUILTIN.iter().map(|t| Self::parse(t)).collect()
    }

    /// Manifiestos de una carpeta `*.toml` (además de los embebidos).
    pub fn load_dir(dir: &Path) -> Result<Vec<Self>> {
        let mut out = Vec::new();
        let entries =
            std::fs::read_dir(dir).map_err(|e| Error(format!("{}: {e}", dir.display())))?;
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().is_some_and(|x| x == "toml") {
                let text = std::fs::read_to_string(&p)
                    .map_err(|e| Error(format!("{}: {e}", p.display())))?;
                out.push(Self::parse(&text)?);
            }
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }

    /// Todos los manifiestos: embebidos más `$BRASA_CATALOG` si existe.
    pub fn all() -> Result<Vec<Self>> {
        let mut out = Self::builtin()?;
        if let Some(dir) = std::env::var_os("BRASA_CATALOG") {
            for m in Self::load_dir(Path::new(&dir))? {
                if !out.iter().any(|x| x.name == m.name) {
                    out.push(m);
                }
            }
        }
        Ok(out)
    }

    pub fn find(name: &str) -> Result<Self> {
        Self::all()?
            .into_iter()
            .find(|m| m.name == name)
            .ok_or_else(|| Error(format!("no hay un manifiesto para {name:?}")))
    }

    /// sha256 total de los archivos (para mostrar una identidad del conjunto).
    pub fn total_bytes(&self) -> u64 {
        self.files.iter().filter_map(|f| f.size).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn el_manifiesto_embebido_es_valido() {
        let all = Manifest::builtin().unwrap();
        let m = all.iter().find(|m| m.name == "qwen3-4b-q4").unwrap();
        assert_eq!(m.source_repo, "Qwen/Qwen3-4B");
        assert_eq!(m.hf_dir, "qwen3-4b-hf");
        assert!(m.files.iter().any(|f| f.path.ends_with(".safetensors")));
        assert!(m.files.iter().all(|f| f.sha256.len() == 64));
        assert!(m.total_bytes() > 7_000_000_000);
    }

    #[test]
    fn parsea_y_busca() {
        let m = Manifest::find("qwen3-4b-q4").unwrap();
        assert_eq!(m.family, "qwen3");
        assert!(Manifest::find("no-existe").is_err());
    }

    #[test]
    fn rechaza_sha256_corto_y_rutas_inseguras() {
        let base = r#"
name = "x"
family = "qwen3"
source_repo = "a/b"
source_revision = "r"
hf_dir = "x-hf"
max_context = 4096
files = [{ path = "%PATH%", sha256 = "%SHA%" , size = 1}]
"#;
        let malo = base
            .replace("%PATH%", "model.safetensors")
            .replace("%SHA%", "abc");
        assert!(Manifest::parse(&malo).is_err());
        let escape = base
            .replace("%PATH%", "../fuera.safetensors")
            .replace("%SHA%", &"0".repeat(64));
        assert!(Manifest::parse(&escape).is_err());
        let absoluto = base
            .replace("%PATH%", "/etc/passwd")
            .replace("%SHA%", &"0".repeat(64));
        assert!(Manifest::parse(&absoluto).is_err());
        let ok = base
            .replace("%PATH%", "sub/model.safetensors")
            .replace("%SHA%", &"a".repeat(64));
        assert!(Manifest::parse(&ok).is_ok());
    }
}
