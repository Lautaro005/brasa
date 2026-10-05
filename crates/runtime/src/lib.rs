//! Sesión de inferencia: prefill, decode, sampling y prefix cache.

pub mod chat;
pub mod qwen_output;
pub mod sampler;
pub mod session;

use std::fmt;

pub use brasa_models::qwen3::Limits;
pub use sampler::{Sampler, SamplingParams};
pub use session::{GenStats, Session, StopReason};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<brasa_models::Error> for Error {
    fn from(e: brasa_models::Error) -> Self {
        Error(e.0)
    }
}

impl From<brasa_metal::MetalError> for Error {
    fn from(e: brasa_metal::MetalError) -> Self {
        Error(e.to_string())
    }
}

impl From<brasa_tokenizer::Error> for Error {
    fn from(e: brasa_tokenizer::Error) -> Self {
        Error(e.0)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
