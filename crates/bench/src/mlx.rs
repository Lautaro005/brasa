//! Baseline MLX-LM: ejecuta `tools/bench_mlx.py` en el venv de `tools/` y lee su JSON.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;

use crate::engine::{Engine, Job};
use crate::measure::run_timed;
use crate::report::RunMetrics;
use crate::{BenchError, Result, err};

#[derive(Debug, Clone)]
pub struct MlxLm {
    /// Intérprete del venv, por defecto `.venv/bin/python`.
    pub python: String,
    pub model_dir: PathBuf,
    pub quant: String,
}

const DRIVER: &str = "tools/bench_mlx.py";

#[derive(Debug, Deserialize)]
struct DriverOutput {
    prompt_tokens: u32,
    gen_tokens: u32,
    ttft_ms: f64,
    decode_ms: f64,
    mlx_peak_memory_bytes: u64,
}

impl Engine for MlxLm {
    fn name(&self) -> &'static str {
        "mlx-lm"
    }

    fn version(&self) -> Result<String> {
        let out = Command::new(&self.python)
            .args([
                "-c",
                "import mlx.core, mlx_lm; print(f'mlx-lm {mlx_lm.__version__}, mlx {mlx.core.__version__}')",
            ])
            .output()
            .map_err(|e| BenchError(format!("no se pudo ejecutar {}: {e}", self.python)))?;
        if !out.status.success() {
            return err(format!(
                "mlx-lm no está instalado en {} (ver tools/requirements.txt)",
                self.python
            ));
        }
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    }

    fn settings(&self, job: &Job) -> Vec<String> {
        vec![
            DRIVER.into(),
            "--gen".into(),
            job.gen_tokens.to_string(),
            "greedy".into(),
            "eos-bloqueado".into(),
        ]
    }

    fn weights_path(&self) -> &Path {
        &self.model_dir
    }

    fn quant(&self) -> &str {
        &self.quant
    }

    fn run_once(&self, job: &Job) -> Result<RunMetrics> {
        let args = vec![
            DRIVER.to_string(),
            "--model".into(),
            self.model_dir.display().to_string(),
            "--prompt".into(),
            job.prompt_path.display().to_string(),
            "--gen".into(),
            job.gen_tokens.to_string(),
        ];
        let out = run_timed(&self.python, &args)?;
        let line = out
            .stdout
            .lines()
            .rev()
            .find(|l| l.trim_start().starts_with('{'))
            .ok_or_else(|| BenchError("bench_mlx.py no imprimió JSON".into()))?;
        let d: DriverOutput = serde_json::from_str(line)
            .map_err(|e| BenchError(format!("JSON de bench_mlx.py inválido: {e}")))?;
        if d.prompt_tokens != job.prompt_tokens || d.gen_tokens != job.gen_tokens {
            return err(format!(
                "mlx-lm procesó {}+{} tokens, se esperaban {}+{}",
                d.prompt_tokens, d.gen_tokens, job.prompt_tokens, job.gen_tokens
            ));
        }
        let decode_tokens = d.gen_tokens - 1;
        let mut extra = std::collections::BTreeMap::new();
        extra.insert(
            "mlx_peak_memory_bytes".to_string(),
            d.mlx_peak_memory_bytes as f64,
        );
        Ok(RunMetrics {
            ttft_ms: d.ttft_ms,
            prefill_ms: d.ttft_ms,
            prefill_tok_s: d.prompt_tokens as f64 / d.ttft_ms * 1000.0,
            decode_ms: d.decode_ms,
            decode_tokens,
            decode_tok_s: decode_tokens as f64 / d.decode_ms * 1000.0,
            peak_rss_bytes: out.memory.peak_rss,
            peak_footprint_bytes: out.memory.peak_footprint,
            extra,
        })
    }
}
