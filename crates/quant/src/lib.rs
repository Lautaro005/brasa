//! Empaquetado Q4/Q8 y formato nativo de pesos (ADR 0006).

pub mod brasa_file;
pub mod mmap;
pub mod qtype;
pub mod safetensors;

use std::fmt;

pub use brasa_file::{BrasaFile, TensorInfo};
pub use qtype::{QType, dequantize, f16_to_f32};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
