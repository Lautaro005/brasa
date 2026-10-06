//! API HTTP local compatible con OpenAI (Chat Completions, Responses) y Anthropic (Messages),
//! sobre los tipos internos de `brasa_core::chat` (ADR 0008). Incluye estado, métricas (U1) y la
//! GUI web embebida (U2).

mod anthropic;
mod bench;
mod common;
pub mod connect;
pub mod engine;
mod metrics;
mod model;
mod openai;
mod plan;
mod responses;
mod status;
mod ui;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use axum::Router;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use brasa_memory::planner::Budget;
use brasa_runtime::Limits;
use brasa_tokenizer::Tokenizer;
use serde_json::json;

use crate::engine::{Engine, LoadedModel};
use crate::metrics::Metrics;

/// Configuración del servidor.
#[derive(Debug, Clone)]
pub struct ServeConfig {
    pub model_dir: PathBuf,
    /// Id del modelo que se informa en `/v1/models`.
    pub model_id: String,
    pub limits: Limits,
    pub addr: SocketAddr,
    /// Commit del binario (se informa en `/api/status`).
    pub commit: String,
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
    /// Carpeta del modelo (para `/api/plan`, que solo lee el encabezado).
    pub model_dir: PathBuf,
    /// Modelo cargado (plan de memoria, pesos, KV).
    pub model: LoadedModel,
    /// Presupuesto de memoria de esta máquina (para la barra de la GUI).
    pub budget: Budget,
    /// Carpeta de reportes de `docs/bench/`, resuelta una vez al arrancar (no por pedido).
    pub bench_dir: PathBuf,
    pub addr: SocketAddr,
    pub metrics: Arc<Metrics>,
    pub started: Instant,
    pub version: String,
    pub commit: String,
    /// Señal de apagado ordenado (la dispara `POST /api/model/stop`, ADR 0025).
    pub shutdown: Arc<tokio::sync::Notify>,
}

/// Datos de configuración del servidor que se fijan al arrancar.
#[derive(Debug, Clone)]
pub struct ServerMeta {
    pub model_id: String,
    pub model_dir: PathBuf,
    pub addr: SocketAddr,
    pub budget: Budget,
    pub bench_dir: PathBuf,
    pub commit: String,
}

impl AppState {
    pub fn new(engine: Engine, tok: Tokenizer, model: LoadedModel, meta: ServerMeta) -> Arc<Self> {
        let ctx = model.ctx;
        Arc::new(Self {
            engine,
            tok,
            model_id: meta.model_id,
            ctx,
            model_dir: meta.model_dir,
            model,
            budget: meta.budget,
            bench_dir: meta.bench_dir,
            addr: meta.addr,
            metrics: Arc::new(Metrics::new()),
            started: Instant::now(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            commit: meta.commit,
            shutdown: Arc::new(tokio::sync::Notify::new()),
        })
    }
}

pub type Shared = Arc<AppState>;

/// Contexto máximo aceptado en las entradas de la API (evita overflow y valores absurdos).
pub const MAX_CTX: usize = 262_144;

/// Parsea un `ctx` que llega por query string. `None` usa el `default`; vacío o fuera de rango
/// devuelven un mensaje claro para responder 400.
pub fn parse_ctx(raw: Option<&str>, default: usize) -> std::result::Result<usize, String> {
    match raw {
        None => Ok(default),
        Some("") => Err(format!("ctx vacío: pasá un número entre 1 y {MAX_CTX}")),
        Some(v) => match v.trim().parse::<usize>() {
            Ok(n) if (1..=MAX_CTX).contains(&n) => Ok(n),
            Ok(n) => Err(format!("ctx {n} fuera de rango (1..={MAX_CTX})")),
            Err(_) => Err(format!("ctx {v:?} no es un número")),
        },
    }
}

pub fn router(state: Shared) -> Router {
    Router::new()
        .route("/", get(health).head(health))
        .route("/api/hello", get(health).head(health))
        .route("/api/status", get(status::status))
        .route("/api/metrics", get(status::metrics))
        .route("/api/plan", get(plan::plan))
        .route("/api/bench", get(bench::bench))
        .route("/api/agents", get(connect::agents))
        .route("/api/model/load", post(model::load))
        .route("/api/model/idle", post(model::idle))
        .route("/api/model/pause", post(model::pause))
        .route("/api/model/resume", post(model::resume))
        .route("/api/model/stop", post(model::stop))
        .route("/ui", get(ui::index))
        .route("/ui/", get(ui::index))
        .route("/ui/app.css", get(ui::css))
        .route("/ui/app.js", get(ui::js))
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
    let (engine, model) = Engine::start(cfg.model_dir.clone(), cfg.limits)?;
    let budget = Budget::this_machine().unwrap_or_else(|| Budget::profile(16));
    let meta = ServerMeta {
        model_id: cfg.model_id.clone(),
        model_dir: cfg.model_dir.clone(),
        addr: cfg.addr,
        budget,
        bench_dir: bench::resolve_dir(),
        commit: cfg.commit.clone(),
    };
    let state = AppState::new(engine, tok, model, meta);
    let shutdown = state.shutdown.clone();
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
        eprintln!("GUI: http://{}/ui", cfg.addr);
        axum::serve(listener, router(state))
            .with_graceful_shutdown(async move {
                tokio::select! {
                    _ = tokio::signal::ctrl_c() => {}
                    _ = shutdown.notified() => {}
                }
            })
            .await
            .map_err(|e| e.to_string())
    })
}
