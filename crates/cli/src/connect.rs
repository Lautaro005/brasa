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
            "# Codex (>= 0.15x) habla la API Responses. Dos archivos en ~/.codex/:
#
# 1) Agregar a ~/.codex/config.toml:
[model_providers.brasa]
name = \"Brasa (local)\"
base_url = \"{base}/v1\"
wire_api = \"responses\"

# 2) Crear ~/.codex/brasa.config.toml (perfil) con:
model_provider = \"brasa\"
model = \"{model}\"
model_context_window = {ctx}
model_auto_compact_token_limit = {compact}

# 3) Usar: codex --profile brasa"
        ),
        Tool::ClaudeCode => println!(
            "# Variables de entorno para Claude Code contra brasa (API Messages).
export ANTHROPIC_BASE_URL={base}
export ANTHROPIC_AUTH_TOKEN=brasa-local
export ANTHROPIC_MODEL={model}
export ANTHROPIC_DEFAULT_HAIKU_MODEL={model}
export CLAUDE_CODE_MAX_CONTEXT_TOKENS={ctx}
# Claude Code reserva la salida máxima dentro de la ventana: sin esto rechaza el prompt.
export CLAUDE_CODE_MAX_OUTPUT_TOKENS={out}
# Para un modelo que no conoce, Claude Code compacta con un margen fijo que en una ventana chica
# lo hace compactar en cada turno. Con esto compacta solo cuando brasa rechaza por contexto.
export CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT=1
# Con todas las herramientas el prompt ronda los 30K tokens; limitarlas, p. ej.:
#   claude --model {model} --tools Bash,Read,Edit,Write,Glob,Grep",
            out = (ctx / 8).min(2048)
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
