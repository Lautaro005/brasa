//! `brasa model <load|idle|pause|resume|stop>`: Model Manager por API (ADR 0025). Le manda la
//! orden a un `serve` corriendo y espera el resultado.

use clap::{Args, Subcommand};

use crate::config::{self, Config, DEFAULT_HOST, DEFAULT_PORT};
use crate::http;

#[derive(Debug, Args)]
pub struct ModelArgs {
    #[command(subcommand)]
    action: Action,
    /// Host del `serve` (por defecto el del archivo de configuración o 127.0.0.1).
    #[arg(long)]
    host: Option<String>,
    /// Puerto del `serve` (por defecto el del archivo de configuración o 8080).
    #[arg(long)]
    port: Option<u16>,
}

#[derive(Debug, Subcommand)]
enum Action {
    /// Carga el modelo con el plan de memoria.
    Load,
    /// Libera los pesos y la KV; el daemon sigue vivo.
    Idle,
    /// Frena la cola sin descargar el modelo.
    Pause,
    /// Vuelve a sacar trabajos de la cola.
    Resume,
    /// Apaga el daemon de forma ordenada.
    Stop,
}

impl Action {
    fn path(&self) -> &'static str {
        match self {
            Self::Load => "/api/model/load",
            Self::Idle => "/api/model/idle",
            Self::Pause => "/api/model/pause",
            Self::Resume => "/api/model/resume",
            Self::Stop => "/api/model/stop",
        }
    }
}

pub fn run(a: ModelArgs) -> Result<(), String> {
    let cfg = Config::load()?;
    let host = config::pick(a.host, cfg.host.clone(), DEFAULT_HOST.to_string());
    let port = config::pick(a.port, cfg.port, DEFAULT_PORT);
    let body = http::post(&host.value, port.value, a.action.path())?;
    let v: serde_json::Value = serde_json::from_str(&body).unwrap_or(serde_json::Value::Null);
    let state = v["state"].as_str().unwrap_or("?");
    println!("modelo: {state}");
    Ok(())
}
