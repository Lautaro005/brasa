//! `GET /api/plan?ctx=&kv=&perfil=&chunk=`: reutiliza el planner como `brasa plan`, sin cargar
//! el modelo (solo lee el encabezado del `.brasa`).

use axum::extract::{Query, State};
use axum::response::{IntoResponse, Response};
use brasa_memory::planner::{Budget, Fit, describe, gib, rejection_message};
use brasa_runtime::{KvType, Limits, Session};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::Shared;

#[derive(Debug, Deserialize)]
pub struct PlanQuery {
    pub ctx: Option<usize>,
    pub kv: Option<String>,
    /// `8gb`, `16gb` o ausente (esta máquina).
    pub perfil: Option<String>,
    pub chunk: Option<usize>,
}

pub async fn plan(State(s): State<Shared>, Query(q): Query<PlanQuery>) -> Response {
    let ctx = q.ctx.unwrap_or(4096);
    let chunk = q.chunk.unwrap_or(128);
    let kv = match q.kv.as_deref() {
        None => KvType::F16,
        Some(name) => match KvType::parse(name) {
            Some(k) => k,
            None => {
                return (
                    axum::http::StatusCode::BAD_REQUEST,
                    axum::Json(json!({"error": format!("kv {name}: se espera f32, f16 o q8_0")})),
                )
                    .into_response();
            }
        },
    };
    let budget = match q.perfil.as_deref() {
        None | Some("esta") => match Budget::this_machine() {
            Some(b) => b,
            None => Budget::profile(16),
        },
        Some("8gb") => Budget::profile(8),
        Some("16gb") => Budget::profile(16),
        Some(other) => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                axum::Json(json!({"error": format!("perfil {other}: se espera 8gb, 16gb o esta")})),
            )
                .into_response();
        }
    };
    let limits = Limits {
        ctx,
        max_tokens: chunk,
        max_logit_rows: 1,
        kv,
    };
    let (fit, model_max) = match Session::plan(&s.model_dir, limits, &budget) {
        Ok(v) => v,
        Err(e) => {
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(json!({"error": e.0})),
            )
                .into_response();
        }
    };
    let (fits, plan, max_ctx, message) = match fit {
        Fit::Fits(p) => (
            true,
            serde_json::to_value(p).unwrap_or(Value::Null),
            Some(ctx),
            format!("contexto {ctx} -> entra: {}", describe(&p)),
        ),
        Fit::TooBig { plan, max_ctx } => (
            false,
            serde_json::to_value(plan).unwrap_or(Value::Null),
            max_ctx,
            rejection_message(ctx, &plan, &budget, max_ctx),
        ),
    };
    axum::Json(json!({
        "ctx": ctx,
        "chunk": chunk,
        "kv": kv.name(),
        "model_max_ctx": model_max,
        "budget": {"bytes": budget.bytes, "gib": gib(budget.bytes), "source": budget.source},
        "fits": fits,
        "plan": plan,
        "max_ctx": max_ctx,
        "message": message,
    }))
    .into_response()
}
