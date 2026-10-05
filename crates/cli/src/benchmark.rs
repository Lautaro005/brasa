//! `brasa benchmark`: corre un engine con los prompts fijos de `fixtures/bench/` y guarda un
//! reporte JSON en `docs/bench/<máquina>/`.

use std::path::{Path, PathBuf};

use brasa_bench::brasa::Brasa;
use brasa_bench::engine::machine_dir;
use brasa_bench::llama_cpp::LlamaCpp;
use brasa_bench::mlx::MlxLm;
use brasa_bench::report::BenchReport;
use brasa_bench::{BenchConfig, BenchError, Engine, Job, Result, models, run_benchmark};
use clap::{Args, ValueEnum};
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Baseline {
    #[value(name = "llama.cpp")]
    LlamaCpp,
    #[value(name = "mlx-lm")]
    MlxLm,
}

#[derive(Debug, Args)]
pub struct BenchmarkArgs {
    /// Engine de referencia a medir. Sin este flag se mide Brasa.
    #[arg(long, value_enum)]
    baseline: Option<Baseline>,
    /// Modelo del registro del harness.
    #[arg(long, default_value = "qwen3-4b-q4")]
    model: String,
    /// Contexto total (prompt + generación). Debe existir en fixtures/bench/manifest.json.
    #[arg(long)]
    ctx: u32,
    /// Corridas medidas (se reporta la mediana).
    #[arg(long, default_value_t = 3)]
    runs: u32,
    /// Corridas de calentamiento descartadas.
    #[arg(long, default_value_t = 1)]
    warmup: u32,
    /// Ruta de pesos alternativa (por defecto, la del registro).
    #[arg(long)]
    weights: Option<PathBuf>,
    /// Argumento extra para el engine; repetible. Ej.: --engine-arg=-ctk --engine-arg=q8_0
    #[arg(long = "engine-arg", allow_hyphen_values = true)]
    engine_args: Vec<String>,
    /// Etiqueta de la variante (por ejemplo "kv-q8"); va al reporte y al nombre del archivo.
    #[arg(long, default_value = "")]
    label: String,
    /// Binario de llama.cpp.
    #[arg(long, default_value = "llama-completion")]
    llama_bin: String,
    /// Tokens por bloque de prefill de Brasa.
    #[arg(long, default_value_t = 128)]
    chunk: usize,
    /// Tipo de KV cache de Brasa: f16 (por defecto), q8_0 o f32 (ADR 0009).
    #[arg(long, default_value = "f16", value_parser = crate::parse_kv)]
    kv: brasa_runtime::KvType,
    /// Intérprete de Python con mlx-lm (venv de tools/).
    #[arg(long, default_value = ".venv/bin/python")]
    python: String,
    /// Carpeta de salida (por defecto docs/bench/<chip>-<ram>gb).
    #[arg(long)]
    out_dir: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    gen_tokens: u32,
    prompts: std::collections::BTreeMap<String, PromptEntry>,
}

#[derive(Debug, Deserialize)]
struct PromptEntry {
    file: String,
    prompt_tokens: u32,
}

const FIXTURES: &str = "fixtures/bench";

pub fn run(args: BenchmarkArgs) -> Result<()> {
    // El reporte registra el commit del repo; el binario tiene que ser de ese mismo código.
    let built = env!("BRASA_BUILD_COMMIT");
    let head = brasa_bench::measure::brasa_commit();
    if built != head || head.ends_with("-dirty") {
        return Err(BenchError(format!(
            "el binario se compiló en {built} y el repo está en {head}; commitear los cambios y \
             correr `cargo build --release -p brasa-cli` antes de medir"
        )));
    }
    let model = models::find(&args.model)
        .ok_or_else(|| BenchError(format!("modelo desconocido: {}", args.model)))?;
    let manifest: Manifest = serde_json::from_str(
        &std::fs::read_to_string(Path::new(FIXTURES).join("manifest.json")).map_err(|e| {
            BenchError(format!(
                "{FIXTURES}/manifest.json: {e} (correr desde la raíz del repo)"
            ))
        })?,
    )
    .map_err(|e| BenchError(format!("manifest inválido: {e}")))?;
    let entry = manifest.prompts.get(&args.ctx.to_string()).ok_or_else(|| {
        BenchError(format!(
            "no hay prompt para ctx {}; disponibles: {:?}",
            args.ctx,
            manifest.prompts.keys().collect::<Vec<_>>()
        ))
    })?;
    let job = Job {
        prompt_path: Path::new(FIXTURES).join(&entry.file),
        prompt_tokens: entry.prompt_tokens,
        gen_tokens: manifest.gen_tokens,
        ctx: args.ctx,
    };

    let engine: Box<dyn Engine> = match args.baseline {
        Some(Baseline::LlamaCpp) => Box::new(LlamaCpp {
            bin: args.llama_bin.clone(),
            gguf: args
                .weights
                .clone()
                .unwrap_or_else(|| model.gguf_path.into()),
            quant: model.gguf_quant.into(),
            extra_args: args.engine_args.clone(),
        }),
        Some(Baseline::MlxLm) => {
            if !args.engine_args.is_empty() {
                return Err(BenchError("--engine-arg no se admite con mlx-lm".into()));
            }
            Box::new(MlxLm {
                python: args.python.clone(),
                model_dir: args
                    .weights
                    .clone()
                    .unwrap_or_else(|| model.mlx_path.into()),
                quant: model.mlx_quant.into(),
            })
        }
        None => {
            let dir: std::path::PathBuf = args
                .weights
                .clone()
                .unwrap_or_else(|| model.brasa_dir.into());
            Box::new(Brasa {
                bin: std::env::current_exe().map_err(|e| BenchError(e.to_string()))?,
                weights: dir.join("model.brasa"),
                model_dir: dir,
                quant: model.brasa_quant.into(),
                chunk: args.chunk,
                kv: args.kv.name().into(),
            })
        }
    };
    if !engine.weights_path().exists() {
        return Err(BenchError(format!(
            "no existen los pesos {} (ver la sección Comandos de CLAUDE.md)",
            engine.weights_path().display()
        )));
    }

    let cfg = BenchConfig {
        warmup: args.warmup,
        runs: args.runs,
        label: args.label.clone(),
    };
    let report = run_benchmark(engine.as_ref(), model, &job, &cfg)?;

    let out_dir = args
        .out_dir
        .unwrap_or_else(|| Path::new("docs/bench").join(machine_dir()));
    std::fs::create_dir_all(&out_dir).map_err(|e| BenchError(format!("{e}")))?;
    let path = out_dir.join(file_name(&report));
    let json = serde_json::to_string_pretty(&report).expect("reporte serializable") + "\n";
    std::fs::write(&path, json).map_err(|e| BenchError(format!("{}: {e}", path.display())))?;
    print_summary(&report);
    println!("reporte: {}", path.display());
    Ok(())
}

fn file_name(r: &BenchReport) -> String {
    let engine: String = r
        .engine
        .name
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect();
    let label = if r.engine.label.is_empty() {
        String::new()
    } else {
        format!("-{}", r.engine.label)
    };
    let ts: String = r
        .timestamp
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect();
    format!("{engine}{label}-{}-ctx{}-{ts}.json", r.model.name, r.ctx)
}

fn print_summary(r: &BenchReport) {
    let s = &r.summary;
    println!(
        "{} {} | {} | ctx {} ({} + {} tokens) | {} corridas",
        r.engine.name,
        r.engine.version,
        r.model.quant,
        r.ctx,
        r.prompt_tokens,
        r.gen_tokens,
        r.runs.len()
    );
    println!(
        "  TTFT      {:>9.0} ms     [{:.0} – {:.0}]",
        s.ttft_ms.median, s.ttft_ms.min, s.ttft_ms.max
    );
    println!(
        "  prefill   {:>9.1} tok/s  [{:.1} – {:.1}]",
        s.prefill_tok_s.median, s.prefill_tok_s.min, s.prefill_tok_s.max
    );
    println!(
        "  decode    {:>9.1} tok/s  [{:.1} – {:.1}]",
        s.decode_tok_s.median, s.decode_tok_s.min, s.decode_tok_s.max
    );
    let gib = |b: f64| b / (1u64 << 30) as f64;
    println!(
        "  memoria   {:>9.2} GiB    (pico de footprint; RSS {:.2} GiB)",
        gib(s.peak_footprint_bytes.median),
        gib(s.peak_rss_bytes.median)
    );
    println!(
        "  sistema   presión inicial {}, peor {}, swap +{} MiB",
        r.system.start_pressure,
        r.system.worst_pressure,
        r.system.swap_growth >> 20
    );
    if !r.valid {
        println!("  NO VÁLIDO: {}", r.invalid_reasons.join("; "));
    }
    for n in &r.notes {
        println!("  nota: {n}");
    }
}
