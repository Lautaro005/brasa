//! `GET /api/status` y `GET /api/metrics` (U1): estado del servidor y contadores desde el
//! arranque, en JSON.

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use brasa_memory::process::process_memory;
use brasa_memory::system::system_memory;
use serde_json::{Value, json};

use crate::Shared;

pub async fn status(State(s): State<Shared>) -> Response {
    let model = &s.model;
    let proc = process_memory();
    let sys = system_memory();
    let q = s.engine.queue();
    axum::Json(json!({
        "version": s.version,
        "commit": s.commit,
        "uptime_s": s.started.elapsed().as_secs_f64(),
        "model": {
            "id": s.model_id,
            "path": model.path,
            "family": model.family,
            "source_repo": model.source_repo,
            "source_commit": model.source_commit,
            // sha256 agregado de los pesos (ADR 0006, `data_sha256` del `.brasa`).
            "weights_sha256": model.weights_sha256,
            "weights_bytes": model.weights_bytes,
        },
        "context": {"ctx": model.ctx, "kv": model.kv, "chunk": model.chunk},
        "plan": serde_json::to_value(model.plan).unwrap_or(Value::Null),
        "process": proc,
        "system": sys,
        "queue": q,
    }))
    .into_response()
}

pub async fn metrics(State(s): State<Shared>) -> Response {
    axum::Json(s.metrics.snapshot()).into_response()
}
