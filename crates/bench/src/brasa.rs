//! Engine propio: ejecuta `brasa bench-once` (este mismo binario) como proceso externo, con la
//! misma medición que los baselines.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::engine::{Engine, Job};
use crate::measure::run_timed;
use crate::report::RunMetrics;
use crate::{BenchError, Result, err};

#[derive(Debug, Clone)]
pub struct Brasa {
    /// Binario `brasa` a ejecutar.
    pub bin: PathBuf,
    /// Carpeta del modelo (`model.brasa` + tokenizer).
    pub model_dir: PathBuf,
    pub weights: PathBuf,
    pub quant: String,
    /// Tokens por bloque de prefill.
    pub chunk: usize,
}

#[derive(Debug, Deserialize)]
struct Out {
    prompt_tokens: u32,
    gen_tokens: u32,
    ttft_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
}

impl Engine for Brasa {
    fn name(&self) -> &'static str {
        "brasa"
    }

    fn version(&self) -> Result<String> {
        Ok(format!(
            "{} ({})",
            env!("CARGO_PKG_VERSION"),
            crate::measure::brasa_commit()
        ))
    }

    fn settings(&self, _job: &Job) -> Vec<String> {
        vec![
            "--chunk".into(),
            self.chunk.to_string(),
            "greedy".into(),
            "stop-bloqueado".into(),
            "kv-f32".into(),
        ]
    }

    fn weights_path(&self) -> &Path {
        &self.weights
    }

    fn quant(&self) -> &str {
        &self.quant
    }

    fn run_once(&self, job: &Job) -> Result<RunMetrics> {
        let args: Vec<String> = vec![
            "bench-once".into(),
            "--model".into(),
            self.model_dir.display().to_string(),
            "--prompt".into(),
            job.prompt_path.display().to_string(),
            "--gen".into(),
            job.gen_tokens.to_string(),
            "--ctx".into(),
            job.ctx.to_string(),
            "--chunk".into(),
            self.chunk.to_string(),
        ];
        let out = run_timed(&self.bin.display().to_string(), &args)?;
        let line = out
            .stdout
            .lines()
            .rev()
            .find(|l| l.trim_start().starts_with('{'))
            .ok_or_else(|| BenchError("bench-once no imprimió JSON".into()))?;
        let d: Out = serde_json::from_str(line)
            .map_err(|e| BenchError(format!("JSON de bench-once inválido: {e}")))?;
        if d.prompt_tokens != job.prompt_tokens || d.gen_tokens != job.gen_tokens {
            return err(format!(
                "brasa procesó {}+{} tokens, se esperaban {}+{}",
                d.prompt_tokens, d.gen_tokens, job.prompt_tokens, job.gen_tokens
            ));
        }
        let decode_tokens = d.gen_tokens - 1;
        Ok(RunMetrics {
            ttft_ms: d.ttft_ms,
            prefill_ms: d.prefill_ms,
            prefill_tok_s: d.prompt_tokens as f64 / d.prefill_ms * 1000.0,
            decode_ms: d.decode_ms,
            decode_tokens,
            decode_tok_s: decode_tokens as f64 / d.decode_ms * 1000.0,
            peak_rss_bytes: out.memory.peak_rss,
            peak_footprint_bytes: out.memory.peak_footprint,
            extra: Default::default(),
        })
    }
}
