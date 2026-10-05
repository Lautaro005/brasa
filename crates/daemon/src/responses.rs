//! `POST /v1/responses` (subconjunto de la API Responses que usa Codex; tipos de `openai` 3.24).

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
    SseEncoder, arguments_string, collect, content_text, named, new_id, opt_f32, opt_usize,
    parse_arguments, sse, start, unix_now,
};
use crate::openai::error;

/// Herramientas de un `namespace` se aplanan como `ns.nombre` (ADR 0008).
const NS_SEP: char = '.';

fn push_tools(tools: &[Value], prefix: Option<&str>, out: &mut Vec<Tool>) {
    for t in tools {
        match t["type"].as_str() {
            Some("function") => {
                let name = t["name"].as_str().unwrap_or_default();
                out.push(Tool {
                    name: match prefix {
                        Some(p) => format!("{p}{NS_SEP}{name}"),
                        None => name.to_string(),
                    },
                    description: t["description"].as_str().map(str::to_string),
                    parameters: t["parameters"].clone(),
                });
            }
            Some("namespace") => {
                if let Some(inner) = t["tools"].as_array() {
                    push_tools(inner, t["name"].as_str(), out);
                }
            }
            // web_search, custom, mcp, ...: sin equivalente local.
            _ => {}
        }
    }
}

fn parse_request(b: &Value) -> Result<ChatRequest, String> {
    let mut messages: Vec<Message> = Vec::new();
    if let Some(i) = b["instructions"].as_str() {
        messages.push(Message::new(Role::System, i));
    }
    match &b["input"] {
        Value::String(s) => messages.push(Message::new(Role::User, s.as_str())),
        Value::Array(items) => {
            for it in items {
                let typ = it["type"].as_str().unwrap_or("message");
                match typ {
                    "message" => {
                        let role = match it["role"].as_str().unwrap_or("user") {
                            "system" | "developer" => Role::System,
                            "assistant" => Role::Assistant,
                            _ => Role::User,
                        };
                        messages.push(Message::new(role, content_text(&it["content"])));
                    }
                    "function_call" => {
                        let mut name = it["name"].as_str().unwrap_or_default().to_string();
                        if let Some(ns) = it["namespace"].as_str() {
                            name = format!("{ns}{NS_SEP}{name}");
                        }
                        let call = ToolCall {
                            id: it["call_id"].as_str().unwrap_or_default().to_string(),
                            name,
                            arguments: parse_arguments(&it["arguments"]),
                        };
                        // Llamadas seguidas van en el mismo mensaje del asistente.
                        match messages.last_mut() {
                            Some(m) if m.role == Some(Role::Assistant) => m.tool_calls.push(call),
                            _ => {
                                let mut m = Message::new(Role::Assistant, "");
                                m.tool_calls.push(call);
                                messages.push(m);
                            }
                        }
                    }
                    "function_call_output" => {
                        let mut m = Message::new(Role::Tool, content_text(&it["output"]));
                        m.tool_call_id = it["call_id"].as_str().map(str::to_string);
                        messages.push(m);
                    }
                    // reasoning (cifrado del lado de OpenAI) y otros tipos: se omiten.
                    _ => {}
                }
            }
        }
        _ => return Err("falta `input`".into()),
    }
    let mut tools = Vec::new();
    if let Some(ts) = b["tools"].as_array() {
        push_tools(ts, None, &mut tools);
    }
    let tool_choice = match &b["tool_choice"] {
        Value::String(s) if s == "none" => ToolChoice::None,
        Value::String(s) if s == "required" => ToolChoice::Required,
        Value::Object(o) => o
            .get("name")
            .and_then(Value::as_str)
            .map_or(ToolChoice::Auto, |n| ToolChoice::Named(n.to_string())),
        _ => ToolChoice::Auto,
    };
    let thinking = b["reasoning"]["effort"]
        .as_str()
        .is_some_and(|e| matches!(e, "low" | "medium" | "high" | "xhigh"));
    Ok(ChatRequest {
        messages,
        tools,
        tool_choice,
        max_tokens: opt_usize(b, "max_output_tokens"),
        temperature: opt_f32(b, "temperature"),
        top_p: opt_f32(b, "top_p"),
        top_k: None,
        seed: None,
        thinking,
    })
}

fn call_item(c: &ToolCall, id: &str, status: &str, args: &str) -> Value {
    let (namespace, name) = match c.name.split_once(NS_SEP) {
        Some((ns, n)) => (Some(ns), n),
        None => (None, c.name.as_str()),
    };
    let mut v = json!({"type": "function_call", "id": id, "call_id": c.id, "name": name,
                       "arguments": args, "status": status});
    if let Some(ns) = namespace {
        v["namespace"] = json!(ns);
    }
    v
}

fn usage_json(u: &Usage) -> Value {
    json!({
        "input_tokens": u.input_tokens,
        "input_tokens_details": {"cached_tokens": u.cached_tokens},
        "output_tokens": u.output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": u.input_tokens + u.output_tokens,
    })
}

/// Objeto `Response` con los campos que el SDK exige.
fn response_obj(
    base: &Base,
    status: &str,
    output: Vec<Value>,
    usage: Option<&Usage>,
    reason: Option<FinishReason>,
) -> Value {
    json!({
        "id": base.id,
        "object": "response",
        "created_at": base.created,
        "status": status,
        "model": base.model,
        "output": output,
        "parallel_tool_calls": base.parallel,
        "tool_choice": base.tool_choice,
        "tools": [],
        "error": null,
        "incomplete_details": if reason == Some(FinishReason::Length) { json!({"reason": "max_output_tokens"}) } else { Value::Null },
        "instructions": null,
        "metadata": {},
        "text": {"format": {"type": "text"}},
        "usage": usage.map(usage_json),
    })
}

fn status_for(reason: Option<FinishReason>) -> &'static str {
    match reason {
        Some(FinishReason::Length) => "incomplete",
        _ => "completed",
    }
}

#[derive(Clone)]
struct Base {
    id: String,
    model: String,
    created: u64,
    parallel: bool,
    tool_choice: Value,
}

pub async fn create(State(s): State<Shared>, body: axum::body::Bytes) -> Response {
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
    if b["previous_response_id"].is_string() {
        return error(
            StatusCode::BAD_REQUEST,
            ErrorKind::InvalidRequest,
            "previous_response_id no está soportado (brasa no guarda respuestas); mandar el historial completo",
        );
    }
    let req = match parse_request(&b) {
        Ok(r) => r,
        Err(e) => return error(StatusCode::BAD_REQUEST, ErrorKind::InvalidRequest, &e),
    };
    let base = Base {
        id: new_id("resp_"),
        model: b["model"].as_str().unwrap_or(&s.model_id).to_string(),
        created: unix_now(),
        parallel: b["parallel_tool_calls"].as_bool().unwrap_or(true),
        tool_choice: if b["tool_choice"].is_null() {
            json!("auto")
        } else {
            b["tool_choice"].clone()
        },
    };
    let stream = b["stream"].as_bool().unwrap_or(false);
    let run = match start(&s, req).await {
        Ok(r) => r,
        Err((k, m)) => {
            let st = if k == ErrorKind::Internal {
                StatusCode::INTERNAL_SERVER_ERROR
            } else {
                StatusCode::BAD_REQUEST
            };
            return error(st, k, &m);
        }
    };
    if stream {
        return sse(
            run,
            RespEncoder {
                base,
                seq: 0,
                output: Vec::new(),
                open: None,
            },
        )
        .into_response();
    }
    let c = collect(run).await;
    if let Some((k, m)) = c.error {
        return error(StatusCode::INTERNAL_SERVER_ERROR, k, &m);
    }
    let mut output = Vec::new();
    if !c.reasoning.is_empty() {
        output.push(
            json!({"type": "reasoning", "id": new_id("rs_"), "summary": [],
                           "content": [{"type": "reasoning_text", "text": c.reasoning}]}),
        );
    }
    if !c.text.is_empty() {
        output.push(json!({"type": "message", "id": new_id("msg_"), "role": "assistant", "status": "completed",
                           "content": [{"type": "output_text", "text": c.text, "annotations": [], "logprobs": []}]}));
    }
    for call in &c.tool_calls {
        output.push(call_item(
            call,
            &new_id("fc_"),
            "completed",
            &arguments_string(&call.arguments),
        ));
    }
    axum::Json(response_obj(
        &base,
        status_for(c.reason),
        output,
        Some(&c.usage),
        c.reason,
    ))
    .into_response()
}

/// Item abierto en el stream (texto o razonamiento) con lo acumulado.
struct Open {
    kind: OpenKind,
    id: String,
    index: usize,
    text: String,
}

#[derive(PartialEq, Eq)]
enum OpenKind {
    Message,
    Reasoning,
}

struct RespEncoder {
    base: Base,
    seq: u64,
    output: Vec<Value>,
    open: Option<Open>,
}

impl RespEncoder {
    fn ev(&mut self, typ: &str, mut v: Value) -> Event {
        v["type"] = json!(typ);
        v["sequence_number"] = json!(self.seq);
        self.seq += 1;
        named(typ, &v)
    }

    fn close(&mut self) -> Vec<Event> {
        let Some(o) = self.open.take() else {
            return vec![];
        };
        let mut out = Vec::new();
        let item = match o.kind {
            OpenKind::Message => {
                out.push(self.ev("response.output_text.done", json!({"item_id": o.id, "output_index": o.index, "content_index": 0, "text": o.text, "logprobs": []})));
                let part = json!({"type": "output_text", "text": o.text, "annotations": [], "logprobs": []});
                out.push(self.ev("response.content_part.done", json!({"item_id": o.id, "output_index": o.index, "content_index": 0, "part": part.clone()})));
                json!({"type": "message", "id": o.id, "role": "assistant", "status": "completed", "content": [part]})
            }
            OpenKind::Reasoning => {
                out.push(self.ev("response.reasoning_text.done", json!({"item_id": o.id, "output_index": o.index, "content_index": 0, "text": o.text})));
                json!({"type": "reasoning", "id": o.id, "summary": [], "content": [{"type": "reasoning_text", "text": o.text}]})
            }
        };
        out.push(self.ev(
            "response.output_item.done",
            json!({"output_index": o.index, "item": item.clone()}),
        ));
        self.output.push(item);
        out
    }

    fn delta(&mut self, kind: OpenKind, t: String) -> Vec<Event> {
        let mut out = Vec::new();
        if self.open.as_ref().is_some_and(|o| o.kind != kind) {
            out.extend(self.close());
        }
        if self.open.is_none() {
            let index = self.output.len();
            let id = new_id(if kind == OpenKind::Message {
                "msg_"
            } else {
                "rs_"
            });
            match kind {
                OpenKind::Message => {
                    out.push(self.ev("response.output_item.added", json!({"output_index": index,
                        "item": {"type": "message", "id": id, "role": "assistant", "status": "in_progress", "content": []}})));
                    out.push(self.ev("response.content_part.added", json!({"item_id": id, "output_index": index, "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": [], "logprobs": []}})));
                }
                OpenKind::Reasoning => {
                    out.push(self.ev(
                        "response.output_item.added",
                        json!({"output_index": index,
                        "item": {"type": "reasoning", "id": id, "summary": [], "content": []}}),
                    ));
                }
            }
            self.open = Some(Open {
                kind,
                id,
                index,
                text: String::new(),
            });
        }
        let o = self.open.as_mut().unwrap();
        o.text.push_str(&t);
        let (id, index, is_msg) = (o.id.clone(), o.index, o.kind == OpenKind::Message);
        let typ = if is_msg {
            "response.output_text.delta"
        } else {
            "response.reasoning_text.delta"
        };
        let mut payload =
            json!({"item_id": id, "output_index": index, "content_index": 0, "delta": t});
        if is_msg {
            payload["logprobs"] = json!([]);
        }
        out.push(self.ev(typ, payload));
        out
    }
}

impl SseEncoder for RespEncoder {
    fn start(&mut self) -> Vec<Event> {
        let r = response_obj(&self.base, "in_progress", vec![], None, None);
        vec![
            self.ev("response.created", json!({"response": r.clone()})),
            self.ev("response.in_progress", json!({"response": r})),
        ]
    }

    fn event(&mut self, e: ChatEvent) -> Vec<Event> {
        match e {
            ChatEvent::Text(t) => self.delta(OpenKind::Message, t),
            ChatEvent::Reasoning(t) => self.delta(OpenKind::Reasoning, t),
            ChatEvent::ToolCall(c) => {
                let mut out = self.close();
                let index = self.output.len();
                let id = new_id("fc_");
                let args = arguments_string(&c.arguments);
                out.push(self.ev(
                    "response.output_item.added",
                    json!({"output_index": index, "item": call_item(&c, &id, "in_progress", "")}),
                ));
                out.push(self.ev(
                    "response.function_call_arguments.delta",
                    json!({"item_id": id, "output_index": index, "delta": args}),
                ));
                out.push(self.ev(
                    "response.function_call_arguments.done",
                    json!({"item_id": id, "output_index": index, "arguments": args}),
                ));
                let item = call_item(&c, &id, "completed", &args);
                out.push(self.ev(
                    "response.output_item.done",
                    json!({"output_index": index, "item": item.clone()}),
                ));
                self.output.push(item);
                out
            }
            ChatEvent::Done { reason, usage } => {
                let mut out = self.close();
                let status = status_for(Some(reason));
                let r = response_obj(
                    &self.base,
                    status,
                    std::mem::take(&mut self.output),
                    Some(&usage),
                    Some(reason),
                );
                let typ = if status == "incomplete" {
                    "response.incomplete"
                } else {
                    "response.completed"
                };
                out.push(self.ev(typ, json!({"response": r})));
                out
            }
            ChatEvent::Error(_, m) => {
                vec![self.ev(
                    "error",
                    json!({"code": "server_error", "message": m, "param": null}),
                )]
            }
        }
    }

    fn end(&mut self) -> Vec<Event> {
        let mut out = self.close();
        let r = response_obj(
            &self.base,
            "incomplete",
            std::mem::take(&mut self.output),
            None,
            None,
        );
        out.push(self.ev("response.incomplete", json!({"response": r})));
        out
    }
}
