//! Manifiestos de modelo en TOML (ADR 0020): repo y revisión de Hugging Face, archivos con
//! sha256, cuantización destino, contexto máximo y licencia. Los de fábrica van embebidos en el
//! binario; además se pueden leer de una carpeta (`$BRASA_CATALOG`).

use std::path::Path;

use serde::Deserialize;

use crate::{Error, Result};

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
        toml::from_str(text).map_err(|e| Error(format!("manifiesto inválido: {e}")))
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
}
