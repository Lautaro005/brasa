//! Binario `brasa`.

mod bench_once;
mod benchmark;
mod doctor;
mod plan;
mod run;

use clap::{Parser, Subcommand};

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
    /// Una corrida medida para `brasa benchmark` (uso interno).
    #[command(hide = true)]
    BenchOnce(bench_once::BenchOnceArgs),
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Doctor { json } => doctor::run(json),
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
