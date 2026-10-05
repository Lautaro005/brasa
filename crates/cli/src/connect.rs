//! `brasa connect <herramienta>`: imprime la configuración para usar el daemon local desde un
//! agente. No modifica archivos del usuario.

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
    let base = format!("http://{}:{}", a.host, a.port);
    let (model, ctx) = (&a.model, a.ctx);
    let compact = ctx * 3 / 4;
    match a.tool {
        Tool::Codex => println!(
            "# Agregar a ~/.codex/config.toml y usar `codex --profile brasa`.
# Codex habla la API Responses (wire_api = \"responses\").
[model_providers.brasa]
name = \"Brasa (local)\"
base_url = \"{base}/v1\"
wire_api = \"responses\"

[profiles.brasa]
model_provider = \"brasa\"
model = \"{model}\"
model_context_window = {ctx}
model_auto_compact_token_limit = {compact}"
        ),
        Tool::ClaudeCode => println!(
            "# Variables de entorno para Claude Code contra brasa (API Messages).
export ANTHROPIC_BASE_URL={base}
export ANTHROPIC_AUTH_TOKEN=brasa-local
export ANTHROPIC_MODEL={model}
export ANTHROPIC_DEFAULT_HAIKU_MODEL={model}
export CLAUDE_CODE_MAX_CONTEXT_TOKENS={ctx}
# El prompt completo de Claude Code (~30K tokens) no entra en {ctx}: limitar herramientas, p. ej.
#   claude --tools Bash,Read,Edit,Write,Glob,Grep"
        ),
        Tool::Cline | Tool::Opencode => println!(
            "# Proveedor compatible con OpenAI:
#   Base URL:   {base}/v1
#   API key:    brasa-local (cualquier valor)
#   Modelo:     {model}
#   Contexto:   {ctx} tokens"
        ),
    }
}
