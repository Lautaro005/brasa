//! Model Manager por API (ADR 0025): `load`, `idle`, `pause`, `resume` y `stop`.

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::Shared;
use crate::engine::Engine;

fn ok(state: &str) -> Response {
    (StatusCode::OK, axum::Json(json!({ "state": state }))).into_response()
}

fn conflict(msg: String) -> Response {
    (StatusCode::CONFLICT, axum::Json(json!({ "error": msg }))).into_response()
}

/// Corre una operación que bloquea (cargar o descargar el modelo) fuera del runtime async.
async fn apply<F>(s: Shared, f: F) -> Response
where
    F: FnOnce(&Engine) -> Result<(), String> + Send + 'static,
{
    let engine = s.engine.clone();
    match tokio::task::spawn_blocking(move || f(&engine)).await {
        Ok(Ok(())) => ok(s.engine.state().name()),
        Ok(Err(e)) => conflict(e),
        Err(e) => conflict(e.to_string()),
    }
}

pub async fn load(State(s): State<Shared>) -> Response {
    apply(s, Engine::load).await
}

pub async fn idle(State(s): State<Shared>) -> Response {
    apply(s, Engine::idle).await
}

pub async fn pause(State(s): State<Shared>) -> Response {
    apply(s, Engine::pause).await
}

pub async fn resume(State(s): State<Shared>) -> Response {
    apply(s, Engine::resume).await
}

pub async fn stop(State(s): State<Shared>) -> Response {
    let engine = s.engine.clone();
    match tokio::task::spawn_blocking(move || engine.stop()).await {
        Ok(Ok(())) => {
            // Apagado ordenado: axum deja terminar las respuestas en vuelo (ADR 0025).
            s.shutdown.notify_one();
            ok("stopped")
        }
        Ok(Err(e)) => conflict(e),
        Err(e) => conflict(e.to_string()),
    }
}
