//! `POST /v1/messages` y `/v1/messages/count_tokens` (formato de la API Messages de Anthropic,
//! verificado contra la documentación de streaming y el SDK `anthropic` 1.11).

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::Event;
use axum::response::{IntoResponse, Response};
use brasa_core::chat::{
    ChatEvent, ChatRequest, ErrorKind, FinishReason, Message, Role, Tool, ToolCall, ToolChoice,
    Usage,
};
use serde_json::{Value, json};

use crate::Shared;
use crate::common::{
    SseEncoder, collect, content_text, named, new_id, opt_f32, opt_usize, sse, start,
};

fn error(status: StatusCode, kind: ErrorKind, msg: &str) -> Response {
    let typ = match kind {
        ErrorKind::Internal => "api_error",
        _ => "invalid_request_error",
    };
    // Claude Code reconoce el exceso de contexto por el texto de la API ("prompt is too long") y
    // solo entonces compacta cuando el modelo no es uno que conozca (ADR 0008).
    let msg = match kind {
        ErrorKind::ContextLength => format!("prompt is too long: {msg}"),
        _ => msg.to_string(),
    };
    (
        status,
        axum::Json(json!({"type": "error", "error": {"type": typ, "message": msg}})),
    )
        .into_response()
}

fn status_of(kind: ErrorKind) -> StatusCode {
    match kind {
        ErrorKind::Internal => StatusCode::INTERNAL_SERVER_ERROR,
        _ => StatusCode::BAD_REQUEST,
    }
}

/// Convierte los bloques de un mensaje de Anthropic en mensajes internos (un mensaje de usuario
/// con `tool_result` se parte en mensajes `Tool` más, si hay, el texto del usuario).
fn push_message(role: &str, content: &Value, out: &mut Vec<Message>) {
    let blocks: Vec<Value> = match content {
        Value::String(s) => vec![json!({"type": "text", "text": s})],
        Value::Array(a) => a.clone(),
        _ => vec![],
    };
    match role {
        "assistant" => {
            let mut m = Message::new(Role::Assistant, "");
            for b in &blocks {
                match b["type"].as_str() {
                    Some("text") => m.content.push_str(b["text"].as_str().unwrap_or_default()),
                    Some("thinking") => {
                        m.reasoning = Some(b["thinking"].as_str().unwrap_or_default().to_string())
                    }
                    Some("tool_use") => m.tool_calls.push(ToolCall {
                        id: b["id"].as_str().unwrap_or_default().to_string(),
                        name: b["name"].as_str().unwrap_or_default().to_string(),
                        arguments: b["input"].clone(),
                    }),
                    _ => {}
                }
            }
            out.push(m);
        }
        "system" => out.push(Message::new(Role::System, content_text(content))),
        _ => {
            let mut text = String::new();
            for b in &blocks {
                match b["type"].as_str() {
                    Some("tool_result") => {
                        let mut body = content_text(&b["content"]);
                        if b["is_error"].as_bool() == Some(true) {
                            body = format!("Error: {body}");
                        }
                        let mut m = Message::new(Role::Tool, body);
                        m.tool_call_id = b["tool_use_id"].as_str().map(str::to_string);
                        out.push(m);
                    }
                    Some("text") => text.push_str(b["text"].as_str().unwrap_or_default()),
                    _ => {}
                }
            }
            if !text.is_empty() {
                out.push(Message::new(Role::User, text));
            }
        }
    }
}

fn parse_request(b: &Value) -> Result<ChatRequest, String> {
    let mut messages = Vec::new();
    let system = content_text(&b["system"]);
    if !system.is_empty() {
        messages.push(Message::new(Role::System, system));
    }
    for m in b["messages"].as_array().ok_or("falta `messages`")? {
        push_message(
            m["role"].as_str().unwrap_or("user"),
            &m["content"],
            &mut messages,
        );
    }
    let tools = b["tools"]
        .as_array()
        .map(|ts| {
            ts.iter()
                // Herramientas de servidor (web_search_…, text_editor_…): sin equivalente local.
                .filter(|t| t.get("input_schema").is_some())
                .map(|t| Tool {
                    name: t["name"].as_str().unwrap_or_default().to_string(),
                    description: t["description"].as_str().map(str::to_string),
                    parameters: t["input_schema"].clone(),
                })
                .collect()
        })
        .unwrap_or_default();
    let tool_choice = match b["tool_choice"]["type"].as_str() {
        Some("none") => ToolChoice::None,
        Some("any") => ToolChoice::Required,
        Some("tool") => ToolChoice::Named(
            b["tool_choice"]["name"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
        ),
        _ => ToolChoice::Auto,
    };
    // `adaptive` deja que el modelo decida: brasa decide no pensar (ADR 0008).
    let thinking = b["thinking"]["type"] == "enabled";
    Ok(ChatRequest {
        messages,
        tools,
        tool_choice,
        max_tokens: opt_usize(b, "max_tokens"),
        temperature: opt_f32(b, "temperature"),
        top_p: opt_f32(b, "top_p"),
        top_k: opt_usize(b, "top_k"),
        seed: None,
        thinking,
    })
}

fn stop_reason(r: Option<FinishReason>) -> &'static str {
    match r {
        Some(FinishReason::ToolCalls) => "tool_use",
        Some(FinishReason::Length) => "max_tokens",
        _ => "end_turn",
    }
}

/// Uso con la convención de Anthropic: `input_tokens` excluye lo leído de caché.
fn usage_json(u: &Usage) -> Value {
    json!({
        "input_tokens": u.input_tokens - u.cached_tokens,
        "cache_read_input_tokens": u.cached_tokens,
        "cache_creation_input_tokens": 0,
        "output_tokens": u.output_tokens,
    })
}

fn parse_body(body: &[u8]) -> Result<(Value, ChatRequest), Box<Response>> {
    let b: Value = serde_json::from_slice(body).map_err(|e| {
        error(
            StatusCode::BAD_REQUEST,
            ErrorKind::InvalidRequest,
            &e.to_string(),
        )
    })?;
    let req = parse_request(&b).map_err(|e| {
        Box::new(error(
            StatusCode::BAD_REQUEST,
            ErrorKind::InvalidRequest,
            &e,
        ))
    })?;
    Ok((b, req))
}

pub async fn count_tokens(State(s): State<Shared>, body: axum::body::Bytes) -> Response {
    let (_, req) = match parse_body(&body) {
        Ok(v) => v,
        Err(r) => return *r,
    };
    match brasa_runtime::chat::render_prompt(&s.tok, &req) {
        Ok(text) => axum::Json(json!({"input_tokens": s.tok.encode(&text).len()})).into_response(),
        Err(e) => error(StatusCode::BAD_REQUEST, ErrorKind::InvalidRequest, &e.0),
    }
}

pub async fn messages(State(s): State<Shared>, body: axum::body::Bytes) -> Response {
    let (b, req) = match parse_body(&body) {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let model = b["model"].as_str().unwrap_or(&s.model_id).to_string();
    let stream = b["stream"].as_bool().unwrap_or(false);
    let run = match start(&s, "/v1/messages", req).await {
        Ok(r) => r,
        Err((k, m)) => return error(status_of(k), k, &m),
    };
    let id = new_id("msg_");
    if stream {
        return sse(
            s.clone(),
            run,
            MsgEncoder {
                id,
                model,
                index: 0,
                open: None,
            },
        )
        .into_response();
    }
    let c = collect(&s, run).await;
    if let Some((k, m)) = c.error {
        return error(status_of(k), k, &m);
    }
    let mut content = Vec::new();
    if !c.reasoning.is_empty() {
        content.push(json!({"type": "thinking", "thinking": c.reasoning, "signature": ""}));
    }
    if !c.text.is_empty() {
        content.push(json!({"type": "text", "text": c.text}));
    }
    for call in &c.tool_calls {
        content.push(
            json!({"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}),
        );
    }
    axum::Json(json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason(c.reason),
        "stop_sequence": null,
        "usage": usage_json(&c.usage),
    }))
    .into_response()
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum Block {
    Text,
    Thinking,
}

struct MsgEncoder {
    id: String,
    model: String,
    /// Índice del próximo bloque de contenido.
    index: usize,
    open: Option<Block>,
}

impl MsgEncoder {
    fn close(&mut self) -> Vec<Event> {
        let Some(kind) = self.open.take() else {
            return vec![];
        };
        let mut out = Vec::new();
        if kind == Block::Thinking {
            out.push(named(
                "content_block_delta",
                &json!({"type": "content_block_delta", "index": self.index,
                "delta": {"type": "signature_delta", "signature": ""}}),
            ));
        }
        out.push(named(
            "content_block_stop",
            &json!({"type": "content_block_stop", "index": self.index}),
        ));
        self.index += 1;
        out
    }

    fn delta(&mut self, kind: Block, t: String) -> Vec<Event> {
        let mut out = Vec::new();
        if self.open.is_some_and(|k| k != kind) {
            out.extend(self.close());
        }
        if self.open.is_none() {
            let block = match kind {
                Block::Text => json!({"type": "text", "text": ""}),
                Block::Thinking => json!({"type": "thinking", "thinking": "", "signature": ""}),
            };
            out.push(named("content_block_start", &json!({"type": "content_block_start", "index": self.index, "content_block": block})));
            self.open = Some(kind);
        }
        let delta = match kind {
            Block::Text => json!({"type": "text_delta", "text": t}),
            Block::Thinking => json!({"type": "thinking_delta", "thinking": t}),
        };
        out.push(named(
            "content_block_delta",
            &json!({"type": "content_block_delta", "index": self.index, "delta": delta}),
        ));
        out
    }
}

impl SseEncoder for MsgEncoder {
    fn start(&mut self) -> Vec<Event> {
        vec![
            named(
                "message_start",
                &json!({"type": "message_start", "message": {
                "id": self.id, "type": "message", "role": "assistant", "model": self.model,
                "content": [], "stop_reason": null, "stop_sequence": null,
                "usage": {"input_tokens": 0, "output_tokens": 0}}}),
            ),
            named("ping", &json!({"type": "ping"})),
        ]
    }

    fn event(&mut self, e: ChatEvent) -> Vec<Event> {
        match e {
            ChatEvent::Text(t) => self.delta(Block::Text, t),
            ChatEvent::Reasoning(t) => self.delta(Block::Thinking, t),
            ChatEvent::ToolCall(c) => {
                let mut out = self.close();
                out.push(named("content_block_start", &json!({"type": "content_block_start", "index": self.index,
                    "content_block": {"type": "tool_use", "id": c.id, "name": c.name, "input": {}}})));
                out.push(named("content_block_delta", &json!({"type": "content_block_delta", "index": self.index,
                    "delta": {"type": "input_json_delta", "partial_json": c.arguments.to_string()}})));
                out.push(named(
                    "content_block_stop",
                    &json!({"type": "content_block_stop", "index": self.index}),
                ));
                self.index += 1;
                out
            }
            ChatEvent::Done { reason, usage } => {
                let mut out = self.close();
                out.push(named(
                    "message_delta",
                    &json!({"type": "message_delta",
                    "delta": {"stop_reason": stop_reason(Some(reason)), "stop_sequence": null},
                    "usage": usage_json(&usage)}),
                ));
                out.push(named("message_stop", &json!({"type": "message_stop"})));
                out
            }
            ChatEvent::Error(_, m) => vec![named(
                "error",
                &json!({"type": "error", "error": {"type": "api_error", "message": m}}),
            )],
        }
    }

    fn end(&mut self) -> Vec<Event> {
        let mut out = self.close();
        out.push(named("message_stop", &json!({"type": "message_stop"})));
        out
    }
}
