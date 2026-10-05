//! `brasa connect <herramienta>`: imprime la configuración para usar el daemon local desde un
//! agente. No modifica archivos del usuario. El texto lo genera `brasa_daemon::connect`, la
//! misma fuente que usa la GUI en `/api/agents`.

use brasa_daemon::connect::{AgentTool, connect_text};
use clap::{Args, ValueEnum};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Tool {
    Codex,
    ClaudeCode,
    Cline,
    Opencode,
}

#[derive(Debug, Args)]
pub struct ConnectArgs {
    #[arg(value_enum)]
    tool: Tool,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Id del modelo servido.
    #[arg(long, default_value = "qwen3-4b-q4")]
    model: String,
    /// Contexto real del daemon (`brasa serve --ctx`).
    #[arg(long, default_value_t = 16384)]
    ctx: usize,
}

pub fn run(a: ConnectArgs) {
    let tool = match a.tool {
        Tool::Codex => AgentTool::Codex,
        Tool::ClaudeCode => AgentTool::ClaudeCode,
        Tool::Cline => AgentTool::Cline,
        Tool::Opencode => AgentTool::Opencode,
    };
    println!("{}", connect_text(tool, &a.host, a.port, &a.model, a.ctx));
}
