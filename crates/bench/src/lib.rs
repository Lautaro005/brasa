//! Harness de benchmark y reportes comparables (ver ADR 0002).
//!
//! Los baselines (llama.cpp, MLX-LM) se ejecutan como procesos externos; nunca se enlazan.

pub mod brasa;
pub mod engine;
pub mod llama_cpp;
pub mod measure;
pub mod mlx;
pub mod models;
pub mod report;

use std::fmt;

/// Error del harness: siempre con un mensaje accionable.
#[derive(Debug)]
pub struct BenchError(pub String);

impl fmt::Display for BenchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for BenchError {}

pub type Result<T> = std::result::Result<T, BenchError>;

pub(crate) fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(BenchError(msg.into()))
}

pub use engine::{BenchConfig, Engine, Job, run_benchmark};
