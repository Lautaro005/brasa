//! Tipos internos de conversación (regla 8 de CLAUDE.md: los formatos OpenAI y Anthropic se
//! traducen a estos en `brasa-daemon`; el resto del engine no conoce las APIs externas).

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// Llamada a herramienta pedida por el modelo (o devuelta por el cliente en el historial).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Argumentos como objeto JSON (si el cliente manda un string JSON, se parsea; si no es
    /// JSON válido, queda como `Value::String`).
    pub arguments: Value,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Message {
    pub role: Option<Role>,
    /// Texto del mensaje (partes de texto concatenadas).
    pub content: String,
    /// Razonamiento de un turno previo del asistente (el template de Qwen3 lo descarta).
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    /// Para `Role::Tool`: id de la llamada que responde.
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role: Some(role),
            content: content.into(),
            ..Default::default()
        }
    }
}

/// Definición de herramienta: nombre, descripción y esquema JSON de los parámetros.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ToolChoice {
    #[default]
    Auto,
    /// No ofrecer herramientas al modelo.
    None,
    /// El cliente exige una llamada (no se puede forzar en el muestreo; se trata como `Auto`).
    Required,
    /// El cliente exige esta herramienta (ídem).
    Named(String),
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    pub tools: Vec<Tool>,
    pub tool_choice: ToolChoice,
    /// Tokens máximos a generar (se recorta al contexto disponible).
    pub max_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub seed: Option<u64>,
    /// Razonamiento (`<think>`) activado.
    pub thinking: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    Cancelled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    /// Tokens del prompt reutilizados del prefix cache.
    pub cached_tokens: usize,
}

/// Evento de una generación, en orden.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChatEvent {
    /// Fragmento de texto de la respuesta.
    Text(String),
    /// Fragmento de razonamiento.
    Reasoning(String),
    /// Llamada a herramienta completa (JSON validado).
    ToolCall(ToolCall),
    Done {
        reason: FinishReason,
        usage: Usage,
    },
    /// Error del engine; termina la generación.
    Error(ErrorKind, String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorKind {
    /// El pedido no entra en el contexto del perfil de memoria.
    ContextLength,
    InvalidRequest,
    Internal,
}
