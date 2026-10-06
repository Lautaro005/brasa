//! Binario `brasa`.

mod bench_once;
mod benchmark;
mod completions;
mod config;
mod connect;
mod convert;
mod doctor;
mod http;
mod model;
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
    /// Model Manager de un `serve` corriendo: carga, libera, pausa, reanuda o apaga el modelo.
    Model(model::ModelArgs),
    /// Lista los modelos locales y verifica sus hashes.
    Models(models::ModelsArgs),
    /// Descarga un modelo de Hugging Face según su manifiesto.
    Pull(pull::PullArgs),
    /// Convierte safetensors de Hugging Face al formato nativo `.brasa` (sin Python).
    Convert(convert::ConvertArgs),
    /// Borra un modelo local (pide confirmación).
    Rm(rm::RmArgs),
    /// Configuración del usuario (~/.config/brasa/config.toml).
    Config(config::ConfigArgs),
    /// Scripts de autocompletado para zsh, bash o fish.
    Completions(completions::CompletionsArgs),
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
        Command::Model(args) => {
            if let Err(e) = model::run(args) {
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
        Command::Convert(args) => {
            if let Err(e) = convert::run(args) {
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
        Command::Config(args) => {
            if let Err(e) = config::run(args) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        Command::Completions(args) => completions::run(args),
        Command::Connect(args) => {
            if let Err(e) = connect::run(args) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
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

#[cfg(test)]
mod docs_tests {
    use super::Cli;
    use clap::CommandFactory;

    /// Aceptación de U6 (equivalente in-process de `scripts/check-docs.sh`): todo subcomando
    /// citado en README.md y docs/guia/ existe en el CLI. No ejecuta el binario.
    #[test]
    fn subcomandos_citados_en_la_guia_existen() {
        let root = Cli::command();
        let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut files = vec![repo.join("README.md")];
        for e in std::fs::read_dir(repo.join("docs/guia")).unwrap() {
            let p = e.unwrap().path();
            if p.extension().is_some_and(|x| x == "md") {
                files.push(p);
            }
        }
        let mut cited = std::collections::BTreeSet::new();
        for f in &files {
            let text = std::fs::read_to_string(f).unwrap();
            for chunk in text.split("brasa ").skip(1) {
                let word = chunk
                    .split(|c: char| !(c.is_ascii_alphanumeric() || c == '-'))
                    .next()
                    .unwrap_or("");
                if !word.is_empty() && !word.starts_with('-') {
                    cited.insert(word.to_string());
                }
            }
        }
        for sub in &cited {
            assert!(
                root.find_subcommand(sub).is_some(),
                "el subcomando {sub:?} citado en las guías no existe"
            );
        }
        assert!(
            cited.contains("serve") && cited.contains("doctor") && cited.contains("models"),
            "la guía no cita los comandos esperados: {cited:?}"
        );
    }
}

#[cfg(test)]
mod help_tests {
    use super::Cli;
    use clap::{Command, CommandFactory};

    /// Snapshot del `--help` de la raíz y de cada subcomando contra `tests/help/<nombre>.txt`.
    /// Para regenerarlos: `BRASA_BLESS=1 cargo test -p brasa-cli --bin brasa help_tests`.
    #[test]
    fn snapshot_de_help() {
        let dir = format!("{}/tests/help", env!("CARGO_MANIFEST_DIR"));
        let bless = std::env::var_os("BRASA_BLESS").is_some();
        let check = |name: &str, cmd: Command| {
            let mut cmd = cmd.term_width(100);
            let help = cmd.render_long_help().to_string();
            let path = format!("{dir}/{name}.txt");
            if bless {
                std::fs::create_dir_all(&dir).unwrap();
                std::fs::write(&path, &help).unwrap();
            } else {
                let expected = std::fs::read_to_string(&path).unwrap_or_else(|_| {
                    panic!("falta el snapshot {path}; regenerá con BRASA_BLESS=1 cargo test")
                });
                assert_eq!(help, expected, "el --help de {name} cambió");
            }
        };
        // La raíz incluye la versión con el commit del build; solo se snapshotean los subcomandos.
        let root = Cli::command();
        for sub in root.get_subcommands() {
            if sub.is_hide_set() || sub.get_name() == "help" {
                continue;
            }
            check(sub.get_name(), sub.clone());
        }
    }
}
