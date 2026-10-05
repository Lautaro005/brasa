//! Binario `brasa`.

mod benchmark;
mod doctor;

use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(
    name = "brasa",
    version,
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
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Command::Doctor { json } => doctor::run(json),
        Command::Benchmark(args) => {
            if let Err(e) = benchmark::run(args) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
    }
}
