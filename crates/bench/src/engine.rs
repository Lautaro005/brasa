//! Orquestación común: calentamiento, corridas medidas, muestreo del sistema y reporte.

use std::path::{Path, PathBuf};
use std::time::Duration;

use brasa_memory::system::{PressureLevel, system_memory};
use brasa_memory::telemetry::MemorySampler;

use crate::measure::{brasa_commit, sha256_bytes, sha256_path, utc_now};
use crate::models::ModelSpec;
use crate::report::{
    BenchReport, EngineInfo, Machine, ModelInfo, RunMetrics, SCHEMA_VERSION, Summary, SystemDuring,
};
use crate::{BenchError, Result, err};

/// Una corrida a medir: el mismo prompt para todos los engines.
#[derive(Debug, Clone)]
pub struct Job {
    pub prompt_path: PathBuf,
    pub prompt_tokens: u32,
    pub gen_tokens: u32,
    pub ctx: u32,
}

/// Un engine ejecutable por el harness.
pub trait Engine {
    fn name(&self) -> &'static str;
    /// Versión exacta (incluye commit si el engine lo informa).
    fn version(&self) -> Result<String>;
    /// Argumentos efectivos, sin rutas de pesos ni prompt.
    fn settings(&self, job: &Job) -> Vec<String>;
    fn weights_path(&self) -> &Path;
    fn quant(&self) -> &str;
    /// Ejecuta una corrida completa y devuelve sus métricas. Debe verificar que procesó
    /// exactamente `job.prompt_tokens` y generó `job.gen_tokens`.
    fn run_once(&self, job: &Job) -> Result<RunMetrics>;
}

#[derive(Debug, Clone)]
pub struct BenchConfig {
    pub warmup: u32,
    pub runs: u32,
    pub label: String,
}

/// Crecimiento de swap a partir del cual la corrida se considera no representativa.
const MAX_SWAP_GROWTH: u64 = 256 << 20;

pub fn run_benchmark(
    engine: &dyn Engine,
    model: &ModelSpec,
    job: &Job,
    cfg: &BenchConfig,
) -> Result<BenchReport> {
    if cfg.runs == 0 {
        return err("se necesita al menos una corrida medida");
    }
    let prompt = std::fs::read(&job.prompt_path)
        .map_err(|e| BenchError(format!("{}: {e}", job.prompt_path.display())))?;
    let version = engine.version()?;
    eprintln!("calculando sha256 de {}", engine.weights_path().display());
    let (weights_sha256, weights_bytes) = sha256_path(engine.weights_path())?;

    let start = system_memory();
    let sampler = MemorySampler::start(Duration::from_millis(50));
    for i in 0..cfg.warmup {
        eprintln!(
            "{} ctx {}: calentamiento {}/{}",
            engine.name(),
            job.ctx,
            i + 1,
            cfg.warmup
        );
        engine.run_once(job)?;
    }
    let mut runs = Vec::with_capacity(cfg.runs as usize);
    for i in 0..cfg.runs {
        let r = engine.run_once(job)?;
        eprintln!(
            "{} ctx {}: corrida {}/{}: prefill {:.1} tok/s, decode {:.1} tok/s, TTFT {:.0} ms, RSS {:.2} GiB",
            engine.name(),
            job.ctx,
            i + 1,
            cfg.runs,
            r.prefill_tok_s,
            r.decode_tok_s,
            r.ttft_ms,
            r.peak_rss_bytes as f64 / (1u64 << 30) as f64
        );
        runs.push(r);
    }
    let mem = sampler.finish();

    let pressure = |p: PressureLevel| format!("{p:?}").to_lowercase();
    let mut invalid_reasons = Vec::new();
    let mut notes = Vec::new();
    if mem.swap_growth() > MAX_SWAP_GROWTH {
        invalid_reasons.push(format!(
            "el swap del sistema creció {} MiB durante el benchmark",
            mem.swap_growth() >> 20
        ));
    }
    if mem.worst_pressure == PressureLevel::Critical {
        invalid_reasons.push("presión de memoria crítica durante el benchmark".into());
    }
    if start.pressure != PressureLevel::Normal {
        notes.push(format!(
            "el sistema arrancó con presión {}; las cifras pueden estar por debajo de lo normal",
            pressure(start.pressure)
        ));
    }

    let hw = brasa_tuner::hardware::hardware_info();
    Ok(BenchReport {
        schema: SCHEMA_VERSION,
        timestamp: utc_now(),
        brasa_commit: brasa_commit(),
        machine: Machine {
            chip: hw.chip,
            model: hw.model,
            memory_bytes: hw.memory_bytes,
            gpu_cores: hw.gpu_cores,
            cpu_performance_cores: hw.cpu_performance_cores,
            cpu_efficiency_cores: hw.cpu_efficiency_cores,
            macos_version: hw.macos_version,
            macos_build: hw.macos_build,
        },
        engine: EngineInfo {
            name: engine.name().into(),
            version,
            settings: engine.settings(job),
            label: cfg.label.clone(),
        },
        model: ModelInfo {
            name: model.name.into(),
            source_repo: model.source_repo.into(),
            source_commit: model.source_commit.into(),
            quant: engine.quant().into(),
            weights_path: engine.weights_path().display().to_string(),
            weights_sha256,
            weights_bytes,
        },
        ctx: job.ctx,
        prompt_tokens: job.prompt_tokens,
        gen_tokens: job.gen_tokens,
        prompt_sha256: sha256_bytes(&prompt),
        warmup_runs: cfg.warmup,
        summary: Summary::of(&runs),
        runs,
        system: SystemDuring {
            start_pressure: pressure(start.pressure),
            worst_pressure: pressure(mem.worst_pressure),
            start_available_percent: start.available_percent,
            min_available_percent: mem.min_available_percent,
            start_swap_used: mem.start_swap_used,
            swap_growth: mem.swap_growth(),
        },
        valid: invalid_reasons.is_empty(),
        invalid_reasons,
        notes,
    })
}

/// Carpeta de reportes para esta máquina: `docs/bench/<chip>-<ram>gb`, p. ej. `m1pro-16gb`.
pub fn machine_dir() -> String {
    let hw = brasa_tuner::hardware::hardware_info();
    let chip: String = hw
        .chip
        .trim_start_matches("Apple ")
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_lowercase();
    format!("{chip}-{}gb", hw.memory_bytes >> 30)
}
