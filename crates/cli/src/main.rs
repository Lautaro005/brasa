//! Binario `brasa`.

mod bench_once;
mod benchmark;
mod connect;
mod doctor;
mod http;
mod models;
mod plan;
mod ps;
mod pull;
mod rm;
mod run;
mod serve;

use clap::{Parser, Subcommand};

/// Tipo de KV cache de la línea de comandos (`--kv f32|f16|q8_0`, ADR 0009).
pub fn parse_kv(s: &str) -> Result<brasa_runtime::KvType, String> {
    brasa_runtime::KvType::parse(s).ok_or_else(|| format!("--kv {s}: se espera f32, f16 o q8_0"))
}

#[derive(Debug, Parser)]
#[command(
    name = "brasa",
    version = concat!(env!("CARGO_PKG_VERSION"), " (", env!("BRASA_BUILD_COMMIT"), ")"),
    about = "Engine de inferencia local para Apple Silicon"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Muestra chip, núcleos, RAM, macOS, familia Metal y presión de memoria.
    Doctor {
        /// Salida en JSON.
        #[arg(long)]
        json: bool,
    },
    /// Mide TTFT, prefill, decode y memoria pico; guarda un reporte JSON en docs/bench/.
    Benchmark(benchmark::BenchmarkArgs),
    /// Chat con un modelo en la terminal.
    Run(run::RunArgs),
    /// Plan de memoria de un modelo y contexto, sin cargarlo.
    Plan(plan::PlanArgs),
    /// API local compatible con OpenAI y Anthropic para agentes.
    Serve(serve::ServeArgs),
    /// Estado y métricas de un `serve` corriendo (endpoints /api/status y /api/metrics).
    Ps(ps::PsArgs),
    /// Lista los modelos locales y verifica sus hashes.
    Models(models::ModelsArgs),
    /// Descarga un modelo de Hugging Face según su manifiesto.
    Pull(pull::PullArgs),
    /// Borra un modelo local (pide confirmación).
    Rm(rm::RmArgs),
    /// Imprime la configuración para Codex, Claude Code, Cline u OpenCode.
    Connect(connect::ConnectArgs),
    /// Una corrida medida para `brasa benchmark` (uso interno).
    #[command(hide = true)]
    BenchOnce(bench_once::BenchOnceArgs),
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Doctor { json } => doctor::run(json),
        Command::Serve(args) => {
            if let Err(e) = serve::run(args) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        Command::Ps(args) => {
            if let Err(e) = ps::run(args) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        Command::Models(args) => {
            if let Err(e) = models::run(args) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        Command::Pull(args) => {
            if let Err(e) = pull::run(args) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        Command::Rm(args) => {
            if let Err(e) = rm::run(args) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        Command::Connect(args) => connect::run(args),
        Command::BenchOnce(args) => {
            if let Err(e) = bench_once::run(args) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        Command::Plan(args) => match plan::run(args) {
            Ok(true) => {}
            Ok(false) => std::process::exit(2),
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        },
        Command::Run(args) => {
            if let Err(e) = run::run(args) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        Command::Benchmark(args) => {
            if let Err(e) = benchmark::run(args) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
    }
}
