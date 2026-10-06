//! Catálogo de modelos: manifiestos (ADR 0020), descubrimiento local, verificación por hash y
//! descarga desde Hugging Face.

pub mod local;
pub mod manifest;
pub mod pull;
pub mod verify;

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<brasa_quant::Error> for Error {
    fn from(e: brasa_quant::Error) -> Self {
        Error(e.0)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
