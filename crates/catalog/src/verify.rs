//! Verificación de un `.brasa`: recalcula el sha256 de cada tensor y el del conjunto (ADR 0006).

use std::path::Path;

use brasa_quant::BrasaFile;

use crate::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyReport {
    pub path: String,
    pub tensors: usize,
    pub bytes: u64,
    /// sha256 agregado del encabezado (coincide tras `verify`).
    pub data_sha256: String,
}

/// Abre `weights` y verifica todos los hashes. Lee el archivo entero.
pub fn verify(weights: &Path) -> Result<VerifyReport> {
    let f = BrasaFile::open(weights)?;
    f.verify()?;
    Ok(VerifyReport {
        path: weights.display().to_string(),
        tensors: f.tensors().len(),
        bytes: f.weights_bytes() as u64,
        data_sha256: f.data_sha256().to_string(),
    })
}
