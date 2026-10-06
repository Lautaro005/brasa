//! `GET /api/models`: modelos `.brasa` en la carpeta del modelo servido y entradas del catálogo
//! que todavía no están en disco. Solo lee encabezados; no carga pesos.

use std::path::Path;

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use brasa_catalog::local;
use brasa_catalog::manifest::Manifest;
use brasa_memory::planner::Fit;
use brasa_runtime::{KvType, Limits, Session};
use serde_json::{Value, json};

use crate::Shared;

fn same_dir(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(x), Ok(y)) => x == y,
        _ => a == b,
    }
}

/// ¿Entra con el contexto y la KV del servidor en el presupuesto de esta máquina?
fn fits(s: &Shared, dir: &Path) -> Value {
    let Some(kv) = KvType::parse(&s.model.kv) else {
        return Value::Null;
    };
    let limits = Limits {
        ctx: s.ctx,
        max_tokens: 128,
        max_logit_rows: 1,
        kv,
    };
    match Session::plan(dir, limits, &s.budget) {
        Ok((Fit::Fits(p), _)) => json!({"fits": true, "total_bytes": p.total}),
        Ok((Fit::TooBig { .. }, _)) => json!({"fits": false}),
        Err(_) => Value::Null,
    }
}

pub async fn models(State(s): State<Shared>) -> Response {
    let base = s.model_dir.parent().unwrap_or(Path::new("."));
    let found = local::scan(base);
    let installed: Vec<Value> = found
        .iter()
        .map(|m| {
            json!({
                "name": m.name,
                "bytes": m.bytes,
                "ok": m.ok,
                "error": m.error,
                "loaded": same_dir(&m.dir, &s.model_dir),
                "plan": if m.ok { fits(&s, &m.dir) } else { Value::Null },
            })
        })
        .collect();
    let catalog: Vec<Value> = Manifest::all()
        .unwrap_or_default()
        .into_iter()
        .filter(|m| !found.iter().any(|f| f.name == m.name))
        .map(|m| {
            json!({
                "name": m.name,
                "family": m.family,
                "quant": m.quant,
                "license": m.license,
                "max_context": m.max_context,
                "download_bytes": m.total_bytes(),
            })
        })
        .collect();
    axum::Json(json!({
        "dir": base.display().to_string(),
        "ctx": s.ctx,
        "kv": s.model.kv,
        "installed": installed,
        "catalog": catalog,
    }))
    .into_response()
}
