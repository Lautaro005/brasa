//! `brasa connect <herramienta>`: imprime la configuración para usar el daemon local desde un
//! agente. El texto lo genera `brasa_daemon::connect`, la misma fuente que usa la GUI en
//! `/api/agents`. Con `--apply` la escribe (ADR 0028): archivos propios de Brasa y, si hace
//! falta, lo mínimo en la config de la herramienta con respaldo previo.

use brasa_daemon::connect::{AgentTool, connect_text};
use brasa_daemon::connect_apply::{Paths, Target, apply};
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
    /// Escribe la configuración en vez de imprimirla (Codex, Claude Code y OpenCode; ADR 0028).
    #[arg(long)]
    apply: bool,
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
    if a.apply {
        let paths = Paths::from_env()?;
        let target = Target {
            host: &host.value,
            port: port.value,
            model: &model.value,
            ctx: ctx.value,
        };
        let done = apply(tool, &paths, &target).map_err(|e| e.to_string())?;
        for f in &done.written {
            println!("escrito      {f}");
        }
        for f in &done.backups {
            println!("respaldo     {f}");
        }
        for f in &done.unchanged {
            println!("sin cambios  {f}");
        }
        for n in &done.notes {
            println!("nota: {n}");
        }
        println!("uso: {}", done.usage);
        return Ok(());
    }
    println!(
        "{}",
        connect_text(tool, &host.value, port.value, &model.value, ctx.value)
    );
    Ok(())
}
