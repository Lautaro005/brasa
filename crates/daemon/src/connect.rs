//! Configuración de agentes que usa el daemon (una sola fuente para `brasa connect` y para la
//! GUI en `/api/agents`). No escribe archivos del usuario: solo genera el texto.

use axum::extract::{Query, State};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::Shared;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentTool {
    Codex,
    ClaudeCode,
    Cline,
    Opencode,
}

impl AgentTool {
    pub const ALL: [AgentTool; 4] = [
        AgentTool::Codex,
        AgentTool::ClaudeCode,
        AgentTool::Cline,
        AgentTool::Opencode,
    ];

    /// Clave estable para la GUI y las APIs (`codex`, `claude-code`, `cline`, `opencode`).
    pub fn key(self) -> &'static str {
        match self {
            AgentTool::Codex => "codex",
            AgentTool::ClaudeCode => "claude-code",
            AgentTool::Cline => "cline",
            AgentTool::Opencode => "opencode",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "codex" => Some(AgentTool::Codex),
            "claude-code" | "claude_code" => Some(AgentTool::ClaudeCode),
            "cline" => Some(AgentTool::Cline),
            "opencode" => Some(AgentTool::Opencode),
            _ => None,
        }
    }
}

/// Texto que imprime `brasa connect <tool>` y que muestra la GUI.
pub fn connect_text(tool: AgentTool, host: &str, port: u16, model: &str, ctx: usize) -> String {
    let base = format!("http://{host}:{port}");
    let compact = ctx * 3 / 4;
    match tool {
        AgentTool::Codex => format!(
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
        AgentTool::ClaudeCode => {
            let out = (ctx / 8).min(2048);
            format!(
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
#   claude --model {model} --tools Bash,Read,Edit,Write,Glob,Grep"
            )
        }
        AgentTool::Cline | AgentTool::Opencode => format!(
            "# Proveedor compatible con OpenAI:
#   Base URL:   {base}/v1
#   API key:    brasa-local (cualquier valor)
#   Modelo:     {model}
#   Contexto:   {ctx} tokens"
        ),
    }
}

#[derive(Debug, Deserialize)]
pub struct AgentsQuery {
    /// Contexto para el que se genera la config (por defecto, el del perfil en uso).
    pub ctx: Option<usize>,
}

/// `GET /api/agents?ctx=`: lo que imprime `brasa connect` para cada agente.
pub async fn agents(State(s): State<Shared>, Query(q): Query<AgentsQuery>) -> Response {
    let ctx = q.ctx.unwrap_or(s.ctx);
    let host = s.addr.ip().to_string();
    let port = s.addr.port();
    let mut tools = serde_json::Map::new();
    for t in AgentTool::ALL {
        tools.insert(
            t.key().to_string(),
            Value::String(connect_text(t, &host, port, &s.model_id, ctx)),
        );
    }
    axum::Json(json!({
        "host": host,
        "port": port,
        "ctx": ctx,
        "tools": Value::Object(tools),
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parsea_claves() {
        assert_eq!(AgentTool::parse("claude-code"), Some(AgentTool::ClaudeCode));
        assert_eq!(AgentTool::parse("codex"), Some(AgentTool::Codex));
        assert_eq!(AgentTool::parse("nope"), None);
    }

    #[test]
    fn el_texto_usa_el_host_y_el_contexto() {
        let t = connect_text(
            AgentTool::ClaudeCode,
            "127.0.0.1",
            8080,
            "qwen3-4b-q4",
            16384,
        );
        assert!(t.contains("http://127.0.0.1:8080"));
        assert!(t.contains("CLAUDE_CODE_MAX_CONTEXT_TOKENS=16384"));
    }
}
