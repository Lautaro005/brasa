//! Almacenamiento desde la GUI (ADR 0034).
//!
//! - `GET /api/storage`: el inventario de `brasa storage --json` (volumen, reserva, modelos,
//!   descargas a medias y lo no reconocido), más qué modelo sirve este servidor y la descarga en
//!   curso.
//! - `POST /api/storage/clean {apply?, older_than_days?}`: limpia los `.part` viejos. Sin
//!   `"apply": true` es un dry-run. Nunca toca la carpeta de la descarga en curso.
//! - `DELETE /api/storage/models/{name} {"confirm": "<name>"}`: borra un modelo entero, con las
//!   reglas de `brasa rm` (`local::resolve_child`: subcarpeta real, sin symlinks). El cuerpo tiene
//!   que repetir el nombre exacto. No borra el modelo que sirve este servidor ni uno que se está
//!   descargando.
//!
//! Los `POST`/`DELETE` pasan por el middleware de origen (`origin.rs`).

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use axum::extract::{Path as UrlPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use brasa_catalog::{local, storage};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::Shared;
use crate::models_admin::current_dir;

fn err(code: StatusCode, msg: impl Into<String>) -> Response {
    (code, axum::Json(json!({ "error": msg.into() }))).into_response()
}

fn same_dir(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(x), Ok(y)) => x == y,
        _ => a == b,
    }
}

/// Nombre y destino de la descarga en curso, si hay una.
fn running_pull(s: &Shared) -> Option<(String, PathBuf)> {
    let st = s.pull.lock().expect("pull");
    (st.state == "running").then(|| {
        (
            st.name.clone().unwrap_or_default(),
            PathBuf::from(st.dest.clone().unwrap_or_default()),
        )
    })
}

/// El inventario como JSON, con lo propio del servidor: modelo servido y descarga en curso.
pub fn report_json(s: &Shared) -> Value {
    let md = current_dir(s);
    let sources: Vec<String> = s.catalog.all().into_iter().map(|m| m.hf_dir).collect();
    let r = storage::report(&md, &s.storage, &sources, SystemTime::now());
    let pulling = running_pull(s);
    let mut v = serde_json::to_value(&r).expect("report");
    if let Some(models) = v["models"].as_array_mut() {
        for m in models {
            let name = m["name"].as_str().unwrap_or_default().to_string();
            let loaded = same_dir(&md.path.join(&name), &s.model_dir);
            let downloading = pulling.as_ref().is_some_and(|(n, _)| *n == name);
            m["loaded"] = json!(loaded);
            m["deletable"] = json!(!loaded && !downloading);
        }
    }
    v["served_model"] = json!(s.model_id);
    v["pull"] = match &pulling {
        Some((name, dest)) => json!({"name": name, "dest": dest.display().to_string()}),
        None => Value::Null,
    };
    v["config"] = json!(s.config_path.display().to_string());
    v
}

pub async fn status(State(s): State<Shared>) -> Response {
    match tokio::task::spawn_blocking(move || report_json(&s)).await {
        Ok(v) => axum::Json(v).into_response(),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CleanReq {
    /// `true` borra; por defecto es un dry-run.
    #[serde(default)]
    pub apply: bool,
    /// Umbral en días; por defecto `[storage] partial_max_age_days`.
    pub older_than_days: Option<u64>,
}

pub async fn clean(State(s): State<Shared>, body: Option<axum::Json<CleanReq>>) -> Response {
    let req = body.map(|b| b.0).unwrap_or_default();
    let days = req
        .older_than_days
        .unwrap_or(s.storage.partial_max_age_days);
    if days == 0 {
        return err(
            StatusCode::BAD_REQUEST,
            "older_than_days tiene que ser al menos 1",
        );
    }
    let base = current_dir(&s).path;
    // La descarga en curso escribe su `.part` todo el tiempo: con el umbral de días no se
    // tocaría igual, pero se excluye explícitamente.
    let exclude: Vec<PathBuf> = running_pull(&s).map(|(_, d)| d).into_iter().collect();
    let res = tokio::task::spawn_blocking(move || {
        let opts = storage::CleanOptions {
            max_age_days: days,
            apply: req.apply,
            follow_base_symlink: false,
            exclude: &exclude,
            now: SystemTime::now(),
        };
        storage::clean_partials(&base, &opts)
    })
    .await;
    match res {
        Ok(Ok(r)) => axum::Json(r).into_response(),
        Ok(Err(e)) => err(StatusCode::CONFLICT, e.0),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

#[derive(Debug, Deserialize)]
pub struct DeleteReq {
    pub confirm: Option<String>,
}

pub async fn delete_model(
    State(s): State<Shared>,
    UrlPath(name): UrlPath<String>,
    body: Option<axum::Json<DeleteReq>>,
) -> Response {
    let confirm = body.and_then(|b| b.0.confirm);
    if confirm.as_deref() != Some(name.as_str()) {
        return err(
            StatusCode::BAD_REQUEST,
            format!(
                "para borrar {name:?} mandá en el cuerpo {{\"confirm\": {name:?}}} con el nombre exacto"
            ),
        );
    }
    if running_pull(&s).is_some_and(|(n, _)| n == name) {
        return err(
            StatusCode::CONFLICT,
            format!("{name} se está descargando: cancelá la descarga antes de borrarlo"),
        );
    }
    let base = current_dir(&s).path;
    let served = s.model_dir.clone();
    let res = tokio::task::spawn_blocking(move || {
        if std::fs::symlink_metadata(base.join(&name)).is_err() {
            return Err((
                StatusCode::NOT_FOUND,
                format!("no hay ningún modelo {name:?} en {}", base.display()),
            ));
        }
        // Las mismas reglas que `brasa rm`: subcarpeta directa, real, con `model.brasa`, y una
        // carpeta de modelos que no pase por un symlink.
        let dir = local::resolve_child(&base, &name).map_err(|e| (StatusCode::BAD_REQUEST, e.0))?;
        if same_dir(&dir, &served) {
            return Err((
                StatusCode::CONFLICT,
                format!(
                    "{name} es el modelo que sirve este servidor: detenelo y serví otro antes de \
                     borrarlo"
                ),
            ));
        }
        let bytes = storage::dir_bytes(&dir);
        std::fs::remove_dir_all(&dir).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("{}: {e}", dir.display()),
            )
        })?;
        Ok(json!({
            "deleted": name,
            "path": dir.display().to_string(),
            "bytes": bytes,
        }))
    })
    .await;
    match res {
        Ok(Ok(v)) => axum::Json(v).into_response(),
        Ok(Err((code, msg))) => err(code, msg),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}
