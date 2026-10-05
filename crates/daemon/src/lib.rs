//! API HTTP local compatible con OpenAI (Chat Completions, Responses) y Anthropic (Messages),
//! sobre los tipos internos de `brasa_core::chat` (ADR 0008).

mod anthropic;
mod common;
pub mod engine;
mod openai;
mod responses;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use brasa_runtime::Limits;
use brasa_tokenizer::Tokenizer;
use serde_json::json;

use crate::engine::Engine;

/// Configuración del servidor.
#[derive(Debug, Clone)]
pub struct ServeConfig {
    pub model_dir: PathBuf,
    /// Id del modelo que se informa en `/v1/models`.
    pub model_id: String,
    pub limits: Limits,
    pub addr: SocketAddr,
}

/// Estado compartido por los handlers.
#[derive(Debug)]
pub struct AppState {
    pub engine: Engine,
    /// Copia del tokenizer para contar tokens sin pasar por el hilo del modelo.
    pub tok: Tokenizer,
    pub model_id: String,
    /// Contexto real del perfil de memoria (no el nominal del modelo).
    pub ctx: usize,
}

pub type Shared = Arc<AppState>;

pub fn router(state: Shared) -> Router {
    Router::new()
        .route("/", get(health).head(health))
        .route("/api/hello", get(health).head(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/v1/responses", post(responses::create))
        .route("/v1/messages", post(anthropic::messages))
        .route("/v1/messages/count_tokens", post(anthropic::count_tokens))
        .with_state(state)
}

async fn health() -> impl IntoResponse {
    (StatusCode::OK, "brasa")
}

/// `GET /v1/models` en formato OpenAI, o Anthropic si el pedido trae `anthropic-version`.
async fn models(State(s): State<Shared>, headers: HeaderMap) -> Response {
    if headers.contains_key("anthropic-version") {
        axum::Json(json!({
            "data": [{
                "type": "model",
                "id": s.model_id,
                "display_name": s.model_id,
                "created_at": "2025-04-29T00:00:00Z",
                "max_input_tokens": s.ctx,
            }],
            "has_more": false,
            "first_id": s.model_id,
            "last_id": s.model_id,
        }))
        .into_response()
    } else {
        axum::Json(json!({
            "object": "list",
            "data": [{
                "id": s.model_id,
                "object": "model",
                "created": 0,
                "owned_by": "brasa",
                "context_length": s.ctx,
                "max_model_len": s.ctx,
            }],
        }))
        .into_response()
    }
}

/// Carga el modelo y sirve hasta que llegue Ctrl-C.
pub fn serve(cfg: ServeConfig) -> Result<(), String> {
    let tok = Tokenizer::from_dir(&cfg.model_dir).map_err(|e| e.0)?;
    let engine = Engine::start(cfg.model_dir.clone(), cfg.limits)?;
    let state = Arc::new(AppState {
        engine,
        tok,
        model_id: cfg.model_id.clone(),
        ctx: cfg.limits.ctx,
    });
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(cfg.addr)
            .await
            .map_err(|e| format!("no se pudo escuchar en {}: {e}", cfg.addr))?;
        eprintln!(
            "brasa sirviendo {} en http://{} (contexto {})",
            cfg.model_id, cfg.addr, cfg.limits.ctx
        );
        axum::serve(listener, router(state))
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await
            .map_err(|e| e.to_string())
    })
}
