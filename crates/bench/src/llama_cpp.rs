//! Baseline llama.cpp: ejecuta `llama-completion` como proceso externo y lee sus tiempos.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::engine::{Engine, Job};
use crate::measure::run_timed;
use crate::report::RunMetrics;
use crate::{BenchError, Result, err};

#[derive(Debug, Clone)]
pub struct LlamaCpp {
    /// Binario `llama-completion` (por defecto, el del PATH).
    pub bin: String,
    pub gguf: PathBuf,
    pub quant: String,
    /// Argumentos adicionales, por ejemplo `-ctk q8_0 -ctv q8_0`.
    pub extra_args: Vec<String>,
}

impl LlamaCpp {
    fn base_args(&self, job: &Job) -> Vec<String> {
        let mut a: Vec<String> = [
            "-n",
            &job.gen_tokens.to_string(),
            "-c",
            &job.ctx.to_string(),
            "-ngl",
            "99",
            "-fa",
            "auto",
            "--ignore-eos",
            "--temp",
            "0",
            "-s",
            "0",
            "-no-cnv",
            "--simple-io",
            "--no-display-prompt",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        a.extend(self.extra_args.iter().cloned());
        a
    }
}

impl Engine for LlamaCpp {
    fn name(&self) -> &'static str {
        "llama.cpp"
    }

    fn version(&self) -> Result<String> {
        let out = Command::new(&self.bin)
            .arg("--version")
            .output()
            .map_err(|e| BenchError(format!("no se pudo ejecutar {}: {e}", self.bin)))?;
        let text = String::from_utf8_lossy(&out.stderr).into_owned()
            + &String::from_utf8_lossy(&out.stdout);
        text.lines()
            .find_map(|l| l.trim().strip_prefix("version: "))
            .map(str::to_string)
            .ok_or_else(|| BenchError(format!("{} --version no informó la versión", self.bin)))
    }

    fn settings(&self, job: &Job) -> Vec<String> {
        self.base_args(job)
    }

    fn weights_path(&self) -> &Path {
        &self.gguf
    }

    fn quant(&self) -> &str {
        &self.quant
    }

    fn run_once(&self, job: &Job) -> Result<RunMetrics> {
        let mut args = vec![
            "-m".to_string(),
            self.gguf.display().to_string(),
            "-f".to_string(),
            job.prompt_path.display().to_string(),
        ];
        args.extend(self.base_args(job));
        let out = run_timed(&self.bin, &args)?;
        let perf = parse_perf(&out.stderr)?;
        if perf.prompt_tokens != job.prompt_tokens {
            return err(format!(
                "llama.cpp procesó {} tokens de prompt, se esperaban {}",
                perf.prompt_tokens, job.prompt_tokens
            ));
        }
        if perf.decode_runs + 1 != job.gen_tokens {
            return err(format!(
                "llama.cpp generó {} tokens, se esperaban {}",
                perf.decode_runs + 1,
                job.gen_tokens
            ));
        }
        Ok(RunMetrics {
            // El primer token sale de los logits del prefill: TTFT = tiempo de prefill.
            ttft_ms: perf.prompt_ms,
            prefill_ms: perf.prompt_ms,
            prefill_tok_s: perf.prompt_tokens as f64 / perf.prompt_ms * 1000.0,
            decode_ms: perf.decode_ms,
            decode_tokens: perf.decode_runs,
            decode_tok_s: perf.decode_runs as f64 / perf.decode_ms * 1000.0,
            peak_rss_bytes: out.memory.peak_rss,
            peak_footprint_bytes: out.memory.peak_footprint,
            extra: Default::default(),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Perf {
    pub prompt_ms: f64,
    pub prompt_tokens: u32,
    pub decode_ms: f64,
    pub decode_runs: u32,
}

/// Lee las líneas `prompt eval time = X ms / N tokens` y `eval time = X ms / N runs`.
pub fn parse_perf(stderr: &str) -> Result<Perf> {
    let mut prompt = None;
    let mut decode = None;
    for line in stderr.lines() {
        let Some((_, rest)) = line.split_once("perf_print:") else {
            continue;
        };
        let rest = rest.trim();
        if let Some(v) = rest.strip_prefix("prompt eval time =") {
            prompt = parse_ms_count(v);
        } else if let Some(v) = rest.strip_prefix("eval time =") {
            decode = parse_ms_count(v);
        }
    }
    match (prompt, decode) {
        (Some((prompt_ms, prompt_tokens)), Some((decode_ms, decode_runs))) => Ok(Perf {
            prompt_ms,
            prompt_tokens,
            decode_ms,
            decode_runs,
        }),
        _ => err("no se encontraron los tiempos de llama.cpp (prompt eval / eval) en stderr"),
    }
}

/// `"  4062.19 ms /  1920 tokens (...)"` -> (4062.19, 1920)
fn parse_ms_count(s: &str) -> Option<(f64, u32)> {
    let (ms, rest) = s.split_once("ms /")?;
    let ms = ms.trim().parse().ok()?;
    let n = rest.split_whitespace().next()?.parse().ok()?;
    Some((ms, n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_llama_perf() {
        let s = "\
0.08.579.826 I common_perf_print:        load time =    1501.20 ms
0.08.579.828 I common_perf_print: prompt eval time =    4062.19 ms /  1920 tokens (    2.12 ms per token,   472.65 tokens per second)
0.08.579.829 I common_perf_print:        eval time =    2544.22 ms /   127 runs   (   20.03 ms per token,    49.92 tokens per second)
0.08.579.830 I common_perf_print:       total time =    6694.94 ms /  2047 tokens";
        assert_eq!(
            parse_perf(s).unwrap(),
            Perf {
                prompt_ms: 4062.19,
                prompt_tokens: 1920,
                decode_ms: 2544.22,
                decode_runs: 127
            }
        );
    }
}
