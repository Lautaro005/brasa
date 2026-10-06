//! Lógica común a las tres APIs: encolar un trabajo, esperar el primer evento (para responder
//! errores con el código HTTP correcto), juntar eventos y armar streams SSE.

use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::response::sse::{Event, KeepAlive, Sse};
use brasa_core::chat::{ChatEvent, ChatRequest, ErrorKind, FinishReason, ToolCall, Usage};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::Shared;
use crate::engine::Job;
use crate::metrics::GenTimer;

static COUNTER: AtomicU64 = AtomicU64::new(1);

/// Id único con prefijo (`chatcmpl-…`, `resp_…`, `msg_…`).
pub fn new_id(prefix: &str) -> String {
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    format!("{prefix}{t:x}{n:04x}")
}

pub fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Generación en curso: primer evento ya recibido y el resto por canal.
pub struct Running {
    pub first: ChatEvent,
    pub rx: mpsc::UnboundedReceiver<ChatEvent>,
    timer: GenTimer,
}

/// Encola `req` y espera el primer evento. Si es un error, lo devuelve como `Err`.
/// `endpoint` es la clave de los contadores de `/api/metrics`.
pub async fn start(
    state: &Shared,
    endpoint: &'static str,
    req: ChatRequest,
) -> Result<Running, (ErrorKind, String)> {
    let timer = state.metrics.begin(endpoint);
    let (tx, mut rx) = mpsc::unbounded_channel();
    if let Err(e) = state.engine.submit(Job {
        req,
        events: tx,
        metrics: state.metrics.clone(),
    }) {
        state.metrics.finish(timer, None, Some(ErrorKind::Internal));
        return Err((ErrorKind::Internal, e));
    }
    match rx.recv().await {
        Some(ChatEvent::Error(k, m)) => {
            state.metrics.finish(timer, None, Some(k));
            Err((k, m))
        }
        Some(first) => Ok(Running { first, rx, timer }),
        None => {
            state.metrics.finish(timer, None, Some(ErrorKind::Internal));
            Err((
                ErrorKind::Internal,
                "el modelo terminó sin responder".into(),
            ))
        }
    }
}

/// Resultado completo de una generación sin streaming.
#[derive(Debug, Default)]
pub struct Collected {
    pub text: String,
    pub reasoning: String,
    pub tool_calls: Vec<ToolCall>,
    pub reason: Option<FinishReason>,
    pub usage: Usage,
    pub error: Option<(ErrorKind, String)>,
}

pub async fn collect(state: &Shared, mut run: Running) -> Collected {
    let mut c = Collected::default();
    let mut usage = None;
    let mut ev = Some(run.first);
    while let Some(e) = ev {
        run.timer.on_event(&e);
        match e {
            ChatEvent::Text(t) => c.text.push_str(&t),
            ChatEvent::Reasoning(t) => c.reasoning.push_str(&t),
            ChatEvent::ToolCall(call) => c.tool_calls.push(call),
            ChatEvent::Done { reason, usage: u } => {
                c.reason = Some(reason);
                c.usage = u;
                usage = Some(u);
                break;
            }
            ChatEvent::Error(k, m) => {
                c.error = Some((k, m));
                break;
            }
        }
        ev = run.rx.recv().await;
    }
    state
        .metrics
        .finish(run.timer, usage, c.error.as_ref().map(|(k, _)| *k));
    c
}

/// Traductor de eventos internos a eventos SSE de una API.
pub trait SseEncoder: Send + 'static {
    /// Eventos a emitir al empezar (antes del primer evento del modelo).
    fn start(&mut self) -> Vec<Event>;
    fn event(&mut self, e: ChatEvent) -> Vec<Event>;
    /// Eventos de cierre si el canal termina sin `Done` (cancelación del lado del modelo).
    fn end(&mut self) -> Vec<Event>;
}

/// Arma la respuesta SSE. Si el cliente se desconecta, el stream se suelta, se cierra el canal y
/// el hilo del modelo cancela la generación al fallar el próximo envío.
pub fn sse(state: Shared, run: Running, mut enc: impl SseEncoder) -> axum::response::Response {
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(64);
    tokio::spawn(async move {
        let Running {
            first,
            rx: mut events,
            mut timer,
        } = run;
        let send_all = async |evs: Vec<Event>| {
            for e in evs {
                if tx.send(Ok(e)).await.is_err() {
                    return false;
                }
            }
            true
        };
        if !send_all(enc.start()).await {
            state.metrics.finish(timer, None, None);
            return;
        }
        let mut next = Some(first);
        let mut finished = false;
        let mut usage = None;
        let mut error = None;
        while let Some(e) = next {
            timer.on_event(&e);
            let done = matches!(e, ChatEvent::Done { .. } | ChatEvent::Error(..));
            match &e {
                ChatEvent::Done { usage: u, .. } => usage = Some(*u),
                ChatEvent::Error(k, _) => error = Some(*k),
                _ => {}
            }
            if !send_all(enc.event(e)).await {
                break;
            }
            if done {
                finished = true;
                break;
            }
            next = events.recv().await;
        }
        if !finished {
            send_all(enc.end()).await;
            // Cancelado: el uso lo suma el hilo del modelo al no poder entregar el Done.
            state.metrics.cancelled(timer);
        } else {
            state.metrics.finish(timer, usage, error);
        }
    });
    use axum::response::IntoResponse;
    Sse::new(ReceiverStream::new(rx))
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Evento SSE con nombre y datos JSON.
pub fn named(name: &str, data: &Value) -> Event {
    Event::default().event(name).data(data.to_string())
}

/// Evento SSE sin nombre (`data: …`).
pub fn data(data: &Value) -> Event {
    Event::default().data(data.to_string())
}

/// Texto de un contenido que puede ser string o lista de partes `{type, text}`.
pub fn content_text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

/// Argumentos de una llamada: string JSON (OpenAI) u objeto (Anthropic) a `Value`.
pub fn parse_arguments(v: &Value) -> Value {
    match v {
        Value::String(s) => serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone())),
        Value::Null => Value::Object(Default::default()),
        other => other.clone(),
    }
}

/// Argumentos como string JSON (formato OpenAI).
pub fn arguments_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

pub fn opt_f32(v: &Value, key: &str) -> Option<f32> {
    v.get(key).and_then(Value::as_f64).map(|x| x as f32)
}

pub fn opt_usize(v: &Value, key: &str) -> Option<usize> {
    v.get(key).and_then(Value::as_u64).map(|x| x as usize)
}
