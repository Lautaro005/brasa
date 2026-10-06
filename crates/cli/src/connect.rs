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
    /// Host del `serve` (por defecto el del archivo de configuración o 127.0.0.1).
    #[arg(long)]
    host: Option<String>,
    /// Puerto del `serve` (por defecto el del archivo de configuración o 8080).
    #[arg(long)]
    port: Option<u16>,
    /// Id del modelo servido (por defecto el del archivo de configuración).
    #[arg(long)]
    model: Option<String>,
    /// Contexto real del daemon (`brasa serve --ctx`), por defecto `[serve] ctx`.
    #[arg(long)]
    ctx: Option<usize>,
}

pub fn run(a: ConnectArgs) -> Result<(), String> {
    let tool = match a.tool {
        Tool::Codex => AgentTool::Codex,
        Tool::ClaudeCode => AgentTool::ClaudeCode,
        Tool::Cline => AgentTool::Cline,
        Tool::Opencode => AgentTool::Opencode,
    };
    let cfg = crate::config::Config::load()?;
    let host = crate::config::pick(
        a.host,
        cfg.host.clone(),
        crate::config::DEFAULT_HOST.to_string(),
    );
    let port = crate::config::pick(a.port, cfg.port, crate::config::DEFAULT_PORT);
    let model = crate::config::pick(
        a.model,
        cfg.model.clone(),
        crate::config::DEFAULT_MODEL.to_string(),
    );
    let ctx = crate::config::serve_ctx(a.ctx, &cfg);
    println!(
        "{}",
        connect_text(tool, &host.value, port.value, &model.value, ctx.value)
    );
    Ok(())
}
