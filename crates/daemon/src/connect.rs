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

/// Instrucciones del sistema para el modelo local (V6). Son propias y cortas a propósito: el
/// catálogo de OpenAI trae una plantilla larga y de ellos, y un 4B necesita instrucciones mínimas
/// y directas. Lo importante es que edite por shell y no espere un `apply_patch`.
const CODEX_INSTRUCTIONS: &str = "\
You are Codex, a coding agent running against a local model (Brasa) on the user's machine. You and the user share one workspace; collaborate until the task is genuinely done.

# Tools
Call tools only through the tool-calling interface. Never write a tool call as text.
Use the `exec_command` tool with a single `cmd` string.

# Editing files (there is no apply_patch)
You do not have an apply_patch tool. Edit files with the shell:
- see the file: `sed -n '1,200p' calc.py`
- change text: python3 -c \"import pathlib;p=pathlib.Path('calc.py');p.write_text(p.read_text().replace('return a - b','return a + b'))\"
- run the tests: `python3 -m unittest`
On macOS `sed -i` needs `-i ''`, so prefer the python3 one-liner above. Prefer one precise
replacement over rewriting the file.

# Working style
- Think briefly, then act.
- Verify with the shell: run the tests after changing code.
- If a command fails, read the error and fix it; do not repeat the same command.
- When done, reply with one short summary of what changed and the result.";

/// Catálogo de modelos de Codex para Brasa (`model_catalog_json`, V6).
///
/// La entrada **no** declara `apply_patch_tool_type`: sin eso Codex no le ofrece el parche
/// `apply_patch` al modelo y este edita los archivos con comandos de shell (`shell_type =
/// unified_exec`). Un modelo de 4B no genera parches `apply_patch` válidos; editar por shell sí le
/// sale. Los campos son los que Codex exige para parsear la entrada.
fn codex_catalog(model: &str, ctx: usize) -> String {
    let entry = json!({
        "slug": model,
        "display_name": model,
        "description": "Brasa (engine local)",
        "context_window": ctx,
        "effective_context_window_percent": 95,
        "default_reasoning_level": "none",
        "supported_reasoning_levels": [
            {"effort": "none", "description": "Sin razonamiento"}
        ],
        "priority": 1,
        "support_verbosity": false,
        "shell_type": "unified_exec",
        "supported_in_api": true,
        "visibility": "list",
        "input_modalities": ["text"],
        "truncation_policy": {"mode": "tokens", "limit": 10000},
        "supports_search_tool": false,
        "experimental_supported_tools": [],
        "base_instructions": CODEX_INSTRUCTIONS
    });
    serde_json::to_string_pretty(&json!({"models": [entry]})).expect("catálogo serializable")
}

/// Texto que imprime `brasa connect <tool>` y que muestra la GUI.
pub fn connect_text(tool: AgentTool, host: &str, port: u16, model: &str, ctx: usize) -> String {
    let base = format!("http://{host}:{port}");
    let compact = ctx * 3 / 4;
    match tool {
        AgentTool::Codex => {
            let home = std::env::var("HOME").unwrap_or_else(|_| "$HOME".to_string());
            let catalog = codex_catalog(model, ctx);
            format!(
                "# Codex (>= 0.153) habla la API Responses. Tres pasos en {home}/.codex/:
#
# 1) Agregar a {home}/.codex/config.toml:
[model_providers.brasa]
name = \"Brasa (local)\"
base_url = \"{base}/v1\"
wire_api = \"responses\"

# 2) Crear {home}/.codex/brasa-models.json con este catálogo. No declara apply_patch: así Codex no
#    le ofrece el parche al modelo y este edita los archivos con comandos de shell (un 4B no
#    genera parches apply_patch válidos; ver docs/demos/fase4-codex.md):
cat > {home}/.codex/brasa-models.json <<'JSON'
{catalog}
JSON

# 3) Crear {home}/.codex/brasa.config.toml (perfil) con:
model_provider = \"brasa\"
model = \"{model}\"
model_context_window = {ctx}
model_auto_compact_token_limit = {compact}
model_catalog_json = \"{home}/.codex/brasa-models.json\"

# 4) Usar: codex --profile brasa"
            )
        }
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
    pub ctx: Option<String>,
}

/// `GET /api/agents?ctx=`: lo que imprime `brasa connect` para cada agente.
pub async fn agents(State(s): State<Shared>, Query(q): Query<AgentsQuery>) -> Response {
    let ctx = match crate::parse_ctx(q.ctx.as_deref(), s.ctx) {
        Ok(v) => v,
        Err(e) => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                axum::Json(json!({"error": e})),
            )
                .into_response();
        }
    };
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

    #[test]
    fn codex_lleva_catalogo_sin_apply_patch() {
        let t = connect_text(AgentTool::Codex, "127.0.0.1", 8080, "qwen3-4b-q4", 16384);
        assert!(t.contains("model_catalog_json"));
        assert!(t.contains("\"shell_type\": \"unified_exec\""));
        assert!(t.contains("\"slug\": \"qwen3-4b-q4\""));
        // La clave del asunto: sin `apply_patch_tool_type` Codex no le da el parche al modelo.
        assert!(!t.contains("apply_patch_tool_type"), "{t}");
    }
}
