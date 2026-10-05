//! Adaptadores por familia de modelo y grafo de forward.

pub mod qwen3;

use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<brasa_metal::MetalError> for Error {
    fn from(e: brasa_metal::MetalError) -> Self {
        Error(e.to_string())
    }
}

impl From<brasa_quant::Error> for Error {
    fn from(e: brasa_quant::Error) -> Self {
        Error(e.0)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
