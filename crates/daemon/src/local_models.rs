//! `GET /api/models`: modelos `.brasa` en la carpeta de modelos (ADR 0031) y entradas del catálogo
//! que todavía no están en disco. Solo lee encabezados; no carga pesos.

use std::path::Path;

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use brasa_catalog::{dirs, local};
use brasa_memory::planner::Fit;
use brasa_runtime::{KvType, Limits, Session};
use serde_json::{Value, json};

use crate::Shared;
use crate::models_admin;

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
    let md = models_admin::current_dir(&s);
    let base = md.path.as_path();
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
    let catalog: Vec<Value> = s
        .catalog
        .all()
        .into_iter()
        .filter(|m| !found.iter().any(|f| f.name == m.name))
        .map(|m| {
            json!({
                "name": m.name,
                "family": m.family,
                "quant": m.quant,
                "license": m.license,
                "max_context": m.max_context,
                "download_bytes": m.download_bytes(),
                // Con pesos convertidos se puede bajar desde la GUI; si no, hace falta
                // `brasa pull --desde-fuente` y `brasa convert`.
                "prebuilt": m.prebuilt.is_some(),
                "pinned": m.prebuilt.as_ref().is_some_and(|p| p.is_pinned()),
                "partial_bytes": models_admin::partial_bytes(base, &m),
            })
        })
        .collect();
    axum::Json(json!({
        "dir": base.display().to_string(),
        "dir_source": md.source.as_str(),
        "dir_source_label": md.source.describe(),
        "free_bytes": dirs::free_bytes(base),
        "ctx": s.ctx,
        "kv": s.model.kv,
        "installed": installed,
        "catalog": catalog,
    }))
    .into_response()
}
