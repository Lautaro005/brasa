//! Lector del formato `.brasa` (ADR 0006): encabezado JSON + tensores alineados a página.

use std::path::Path;

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::mmap::Mmap;
use crate::qtype::QType;
use crate::{Error, Result};

const MAGIC: &[u8; 4] = b"BRSA";
const VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: QType,
    pub shape: Vec<usize>,
    pub offset: usize,
    pub nbytes: usize,
    pub sha256: String,
}

impl TensorInfo {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

#[derive(Debug, Deserialize)]
struct Meta {
    format: String,
    version: u32,
    converter: String,
    model: ModelMeta,
    page: usize,
    tensors: Vec<TensorInfo>,
    data_sha256: String,
}

#[derive(Debug, Deserialize)]
struct ModelMeta {
    family: String,
    source_repo: String,
    source_commit: String,
    config: serde_json::Value,
}

/// Archivo `.brasa` mapeado en memoria.
#[derive(Debug)]
pub struct BrasaFile {
    map: Mmap,
    meta: Meta,
}

fn bad(path: &Path, msg: impl std::fmt::Display) -> Error {
    Error(format!("{}: {msg}", path.display()))
}

impl BrasaFile {
    /// Abre y valida estructura (magic, versión, límites y alineación). No verifica hashes:
    /// eso es `verify`, que lee todo el archivo.
    pub fn open(path: &Path) -> Result<Self> {
        let map = Mmap::open(path)?;
        let bytes = map.as_slice();
        if bytes.len() < 16 || &bytes[..4] != MAGIC {
            return Err(bad(path, "no es un archivo .brasa"));
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != VERSION {
            return Err(bad(path, format!("versión {version} no soportada")));
        }
        let json_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let json = bytes
            .get(16..16 + json_len)
            .ok_or_else(|| bad(path, "encabezado truncado"))?;
        let meta: Meta =
            serde_json::from_slice(json).map_err(|e| bad(path, format!("encabezado: {e}")))?;
        if meta.format != "brasa" || meta.version != VERSION {
            return Err(bad(path, "encabezado inconsistente"));
        }
        for t in &meta.tensors {
            if t.offset % meta.page != 0 {
                return Err(bad(path, format!("{} no está alineado a página", t.name)));
            }
            if t.dtype.nbytes(t.numel()) != t.nbytes {
                return Err(bad(path, format!("{}: tamaño inconsistente", t.name)));
            }
            if t.offset + t.nbytes > bytes.len() {
                return Err(bad(path, format!("{}: fuera del archivo", t.name)));
            }
        }
        Ok(Self { map, meta })
    }

    /// Verifica el sha256 de cada tensor y el del conjunto.
    pub fn verify(&self) -> Result<()> {
        let mut all = String::with_capacity(64 * self.meta.tensors.len());
        for t in &self.meta.tensors {
            let h: String = Sha256::digest(self.data(t))
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            if h != t.sha256 {
                return Err(Error(format!("{}: sha256 no coincide", t.name)));
            }
            all.push_str(&h);
        }
        let h: String = Sha256::digest(all.as_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        if h != self.meta.data_sha256 {
            return Err(Error("data_sha256 no coincide".into()));
        }
        Ok(())
    }

    pub fn tensors(&self) -> &[TensorInfo] {
        &self.meta.tensors
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.meta.tensors.iter().find(|t| t.name == name)
    }

    /// Bytes del tensor (sin el relleno de página). Alineados a página.
    pub fn data(&self, t: &TensorInfo) -> &[u8] {
        &self.map.as_slice()[t.offset..t.offset + t.nbytes]
    }

    /// `config.json` original del modelo.
    pub fn config(&self) -> &serde_json::Value {
        &self.meta.model.config
    }

    pub fn family(&self) -> &str {
        &self.meta.model.family
    }

    /// (repo, commit) de los pesos de origen.
    pub fn source(&self) -> (&str, &str) {
        (&self.meta.model.source_repo, &self.meta.model.source_commit)
    }

    pub fn converter(&self) -> &str {
        &self.meta.converter
    }

    pub fn data_sha256(&self) -> &str {
        &self.meta.data_sha256
    }

    /// Bytes totales de tensores (sin relleno).
    pub fn weights_bytes(&self) -> usize {
        self.meta.tensors.iter().map(|t| t.nbytes).sum()
    }
}
