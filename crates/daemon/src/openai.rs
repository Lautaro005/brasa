//! `POST /v1/chat/completions` (formato de `openai` 3.24: `ChatCompletion`/`ChatCompletionChunk`).

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
    SseEncoder, arguments_string, collect, content_text, data, new_id, opt_f32, opt_usize,
    parse_arguments, sse, start, unix_now,
};

pub fn error(status: StatusCode, kind: ErrorKind, msg: &str) -> Response {
    let (typ, code) = match kind {
        ErrorKind::ContextLength => ("invalid_request_error", json!("context_length_exceeded")),
        ErrorKind::InvalidRequest => ("invalid_request_error", Value::Null),
        ErrorKind::Internal => ("server_error", Value::Null),
    };
    (
        status,
        axum::Json(json!({"error": {"message": msg, "type": typ, "param": null, "code": code}})),
    )
        .into_response()
}

fn status_of(kind: ErrorKind) -> StatusCode {
    match kind {
        ErrorKind::Internal => StatusCode::INTERNAL_SERVER_ERROR,
        _ => StatusCode::BAD_REQUEST,
    }
}

fn parse_request(b: &Value) -> Result<ChatRequest, String> {
    let msgs = b["messages"].as_array().ok_or("falta `messages`")?;
    let mut messages = Vec::with_capacity(msgs.len());
    for m in msgs {
        let role = match m["role"].as_str().unwrap_or("user") {
            "system" | "developer" => Role::System,
            "assistant" => Role::Assistant,
            "tool" | "function" => Role::Tool,
            _ => Role::User,
        };
        let mut msg = Message::new(role, content_text(&m["content"]));
        if let Some(r) = m["reasoning_content"].as_str() {
            msg.reasoning = Some(r.to_string());
        }
        if let Some(calls) = m["tool_calls"].as_array() {
            for c in calls {
                msg.tool_calls.push(ToolCall {
                    id: c["id"].as_str().unwrap_or_default().to_string(),
                    name: c["function"]["name"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string(),
                    arguments: parse_arguments(&c["function"]["arguments"]),
                });
            }
        }
        msg.tool_call_id = m["tool_call_id"].as_str().map(str::to_string);
        messages.push(msg);
    }
    let tools = b["tools"]
        .as_array()
        .map(|ts| {
            ts.iter()
                .filter(|t| t["type"] == "function")
                .map(|t| Tool {
                    name: t["function"]["name"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string(),
                    description: t["function"]["description"].as_str().map(str::to_string),
                    parameters: t["function"]["parameters"].clone(),
                })
                .collect()
        })
        .unwrap_or_default();
    let tool_choice = match &b["tool_choice"] {
        Value::String(s) if s == "none" => ToolChoice::None,
        Value::String(s) if s == "required" => ToolChoice::Required,
        Value::Object(o) => o
            .get("function")
            .and_then(|f| f["name"].as_str())
            .map_or(ToolChoice::Auto, |n| ToolChoice::Named(n.to_string())),
        _ => ToolChoice::Auto,
    };
    let thinking = b["reasoning_effort"]
        .as_str()
        .is_some_and(|e| !matches!(e, "none" | "minimal"));
    Ok(ChatRequest {
        messages,
        tools,
        tool_choice,
        max_tokens: opt_usize(b, "max_completion_tokens").or(opt_usize(b, "max_tokens")),
        temperature: opt_f32(b, "temperature"),
        top_p: opt_f32(b, "top_p"),
        top_k: opt_usize(b, "top_k"),
        seed: b["seed"].as_u64(),
        thinking,
    })
}

fn finish_reason(r: Option<FinishReason>) -> &'static str {
    match r {
        Some(FinishReason::ToolCalls) => "tool_calls",
        Some(FinishReason::Length) => "length",
        _ => "stop",
    }
}

fn usage_json(u: &Usage) -> Value {
    json!({
        "prompt_tokens": u.input_tokens,
        "completion_tokens": u.output_tokens,
        "total_tokens": u.input_tokens + u.output_tokens,
        "prompt_tokens_details": {"cached_tokens": u.cached_tokens},
    })
}

fn call_json(c: &ToolCall) -> Value {
    json!({"id": c.id, "type": "function",
           "function": {"name": c.name, "arguments": arguments_string(&c.arguments)}})
}

pub async fn chat_completions(State(s): State<Shared>, body: axum::body::Bytes) -> Response {
    let b: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return error(
                StatusCode::BAD_REQUEST,
                ErrorKind::InvalidRequest,
                &e.to_string(),
            );
        }
    };
    let req = match parse_request(&b) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, ErrorKind::InvalidRequest, &e),
    };
    let model = b["model"].as_str().unwrap_or(&s.model_id).to_string();
    let stream = b["stream"].as_bool().unwrap_or(false);
    let include_usage = b["stream_options"]["include_usage"]
        .as_bool()
        .unwrap_or(false);
    let run = match start(&s, "/v1/chat/completions", req).await {
        Ok(r) => r,
        Err((k, m)) => return error(status_of(k), k, &m),
    };
    let id = new_id("chatcmpl-");
    if stream {
        return sse(
            s.clone(),
            run,
            ChunkEncoder {
                id,
                model,
                created: unix_now(),
                include_usage,
                calls: 0,
            },
        )
        .into_response();
    }
    let c = collect(&s, run).await;
    if let Some((k, m)) = c.error {
        return error(status_of(k), k, &m);
    }
    let mut message = json!({"role": "assistant",
                             "content": if c.text.is_empty() && !c.tool_calls.is_empty() { Value::Null } else { json!(c.text) }});
    if !c.tool_calls.is_empty() {
        message["tool_calls"] = c.tool_calls.iter().map(call_json).collect();
    }
    if !c.reasoning.is_empty() {
        message["reasoning_content"] = json!(c.reasoning);
    }
    axum::Json(json!({
        "id": id,
        "object": "chat.completion",
        "created": unix_now(),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason(c.reason), "logprobs": null}],
        "usage": usage_json(&c.usage),
    }))
    .into_response()
}

struct ChunkEncoder {
    id: String,
    model: String,
    created: u64,
    include_usage: bool,
    calls: usize,
}

impl ChunkEncoder {
    fn chunk(&self, delta: Value, finish: Option<&str>) -> Event {
        data(&json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish, "logprobs": null}],
        }))
    }
}

impl SseEncoder for ChunkEncoder {
    fn start(&mut self) -> Vec<Event> {
        vec![self.chunk(json!({"role": "assistant", "content": ""}), None)]
    }

    fn event(&mut self, e: ChatEvent) -> Vec<Event> {
        match e {
            ChatEvent::Text(t) => vec![self.chunk(json!({"content": t}), None)],
            ChatEvent::Reasoning(t) => vec![self.chunk(json!({"reasoning_content": t}), None)],
            ChatEvent::ToolCall(c) => {
                let index = self.calls;
                self.calls += 1;
                let mut call = call_json(&c);
                call["index"] = json!(index);
                vec![self.chunk(json!({"tool_calls": [call]}), None)]
            }
            ChatEvent::Done { reason, usage } => {
                let mut out = vec![self.chunk(json!({}), Some(finish_reason(Some(reason))))];
                if self.include_usage {
                    out.push(data(&json!({
                        "id": self.id, "object": "chat.completion.chunk", "created": self.created,
                        "model": self.model, "choices": [], "usage": usage_json(&usage),
                    })));
                }
                out.push(Event::default().data("[DONE]"));
                out
            }
            ChatEvent::Error(_, m) => vec![
                data(
                    &json!({"error": {"message": m, "type": "server_error", "param": null, "code": null}}),
                ),
                Event::default().data("[DONE]"),
            ],
        }
    }

    fn end(&mut self) -> Vec<Event> {
        vec![Event::default().data("[DONE]")]
    }
}
