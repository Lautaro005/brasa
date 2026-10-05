//! Empaquetado Q4/Q8 y formato nativo de pesos (ADR 0006).

pub mod brasa_file;
pub mod mmap;
pub mod qtype;
pub mod safetensors;

use std::fmt;

pub use brasa_file::{BrasaFile, TensorInfo};
pub use qtype::{
    KV_Q8_DIM, KV_Q8_ROW, QType, dequantize, dequantize_kv_q8, f16_to_f32, f32_to_f16,
    quantize_kv_q8,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
