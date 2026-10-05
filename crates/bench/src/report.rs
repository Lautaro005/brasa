//! Formato del reporte JSON de benchmark. Versionado con `schema`.

use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BenchReport {
    pub schema: u32,
    /// UTC, ISO 8601.
    pub timestamp: String,
    /// Commit de Brasa que ejecutó el harness (con sufijo `-dirty` si había cambios).
    pub brasa_commit: String,
    pub machine: Machine,
    pub engine: EngineInfo,
    pub model: ModelInfo,
    pub ctx: u32,
    pub prompt_tokens: u32,
    pub gen_tokens: u32,
    pub prompt_sha256: String,
    pub warmup_runs: u32,
    pub runs: Vec<RunMetrics>,
    pub summary: Summary,
    pub system: SystemDuring,
    /// `false` si la corrida no es representativa (swap creciente, presión crítica).
    pub valid: bool,
    pub invalid_reasons: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Machine {
    pub chip: String,
    pub model: String,
    pub memory_bytes: u64,
    pub gpu_cores: Option<u32>,
    pub cpu_performance_cores: u32,
    pub cpu_efficiency_cores: u32,
    pub macos_version: String,
    pub macos_build: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineInfo {
    /// "brasa", "llama.cpp" o "mlx-lm".
    pub name: String,
    pub version: String,
    /// Argumentos efectivos con los que se ejecutó (sin rutas de pesos ni prompt).
    pub settings: Vec<String>,
    /// Etiqueta de variante, por ejemplo "kv-q8". Vacía para la configuración por defecto.
    pub label: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    /// Repo de origen de los pesos, por ejemplo "Qwen/Qwen3-4B".
    pub source_repo: String,
    /// Commit de los pesos de origen en Hugging Face.
    pub source_commit: String,
    pub quant: String,
    pub weights_path: String,
    pub weights_sha256: String,
    pub weights_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunMetrics {
    pub ttft_ms: f64,
    pub prefill_ms: f64,
    pub prefill_tok_s: f64,
    pub decode_ms: f64,
    /// Tokens generados después del primero (los que mide decode tok/s).
    pub decode_tokens: u32,
    pub decode_tok_s: f64,
    pub peak_rss_bytes: u64,
    pub peak_footprint_bytes: u64,
    /// Métricas propias del engine, por ejemplo el pico de GPU de MLX.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub extra: std::collections::BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Stat {
    pub median: f64,
    pub min: f64,
    pub max: f64,
}

impl Stat {
    pub fn of(values: &[f64]) -> Self {
        assert!(!values.is_empty(), "sin valores");
        let mut v = values.to_vec();
        v.sort_by(f64::total_cmp);
        let n = v.len();
        let median = if n % 2 == 1 {
            v[n / 2]
        } else {
            (v[n / 2 - 1] + v[n / 2]) / 2.0
        };
        Self {
            median,
            min: v[0],
            max: v[n - 1],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    pub ttft_ms: Stat,
    pub prefill_tok_s: Stat,
    pub decode_tok_s: Stat,
    pub peak_rss_bytes: Stat,
    pub peak_footprint_bytes: Stat,
}

impl Summary {
    pub fn of(runs: &[RunMetrics]) -> Self {
        let s = |f: fn(&RunMetrics) -> f64| Stat::of(&runs.iter().map(f).collect::<Vec<_>>());
        Self {
            ttft_ms: s(|r| r.ttft_ms),
            prefill_tok_s: s(|r| r.prefill_tok_s),
            decode_tok_s: s(|r| r.decode_tok_s),
            peak_rss_bytes: s(|r| r.peak_rss_bytes as f64),
            peak_footprint_bytes: s(|r| r.peak_footprint_bytes as f64),
        }
    }
}

/// Estado del sistema durante todo el benchmark (calentamiento incluido).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemDuring {
    pub start_pressure: String,
    pub worst_pressure: String,
    pub start_available_percent: Option<u8>,
    pub min_available_percent: Option<u8>,
    pub start_swap_used: u64,
    pub swap_growth: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stat_median() {
        let s = Stat::of(&[3.0, 1.0, 2.0]);
        assert_eq!((s.median, s.min, s.max), (2.0, 1.0, 3.0));
        assert_eq!(Stat::of(&[1.0, 2.0, 3.0, 10.0]).median, 2.5);
    }
}
