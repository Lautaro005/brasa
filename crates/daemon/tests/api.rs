//! U1: tests del daemon con un engine simulado (sin GPU ni pesos). Verifican los contadores de
//! `/api/metrics` tras un pedido streaming y uno no streaming, y la forma de `/api/status`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use brasa_core::chat::{ChatEvent, FinishReason, Usage};
use brasa_daemon::AppState;
use brasa_daemon::engine::{Engine, LoadedModel};
use brasa_memory::planner::MemoryPlan;
use brasa_tokenizer::Tokenizer;
use serde_json::{Value, json};
use tower::ServiceExt;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn state() -> Arc<AppState> {
    let engine = Engine::simulated(|req, emit| {
        if !emit(ChatEvent::Text("hola ".into())) {
            return;
        }
        if !emit(ChatEvent::Text("mundo".into())) {
            return;
        }
        emit(ChatEvent::Done {
            reason: FinishReason::Stop,
            usage: Usage {
                input_tokens: 7 + req.messages.len(),
                output_tokens: 3,
                cached_tokens: 2,
            },
        });
    });
    let tok = Tokenizer::from_dir(&root().join("fixtures/qwen3-4b/tokenizer")).unwrap();
    let model = LoadedModel {
        path: "/tmp/falso/model.brasa".into(),
        weights_sha256: "ab".repeat(32),
        weights_bytes: 2_000_000_000,
        family: "qwen3".into(),
        source_repo: "Qwen/Qwen3-4B".into(),
        source_commit: "1cfa9a72".into(),
        ctx: 2048,
        kv: "f16".into(),
        chunk: 512,
        plan: MemoryPlan {
            weights: 2_000_000_000,
            kv: 1_000_000,
            workspace: 500_000,
            overhead: 1_000_000,
            total: 2_002_500_000,
        },
    };
    AppState::new(engine, tok, "qwen3-4b-q4".into(), model, "test".into())
}

async fn json_body(resp: axum::response::Response) -> Value {
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

async fn post(app: Router, uri: &str, body: Value) -> axum::response::Response {
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    app.oneshot(req).await.unwrap()
}

async fn get(app: Router, uri: &str) -> axum::response::Response {
    app.oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap()
}

#[tokio::test]
async fn metrics_tras_pedido_no_streaming() {
    let s = state();
    let app = brasa_daemon::router(s.clone());
    let resp = post(
        app.clone(),
        "/v1/chat/completions",
        json!({"messages": [{"role": "user", "content": "hola"}], "stream": false}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_body(resp).await;
    assert_eq!(body["choices"][0]["message"]["content"], "hola mundo");

    let m = json_body(get(app, "/api/metrics").await).await;
    assert_eq!(m["endpoints"]["/v1/chat/completions"], 1);
    assert_eq!(m["prompt_tokens"], 8);
    assert_eq!(m["generated_tokens"], 3);
    assert_eq!(m["cached_tokens"], 2);
    assert_eq!(m["ttft_ms"]["samples"], 1);
    assert_eq!(m["decode_tok_s"]["samples"], 1);
    assert!(m["decode_tok_s"]["last"].as_f64().unwrap() > 0.0);
}

#[tokio::test]
async fn metrics_tras_pedido_streaming() {
    let s = state();
    let app = brasa_daemon::router(s.clone());
    let resp = post(
        app.clone(),
        "/v1/chat/completions",
        json!({"messages": [{"role": "user", "content": "hola"}], "stream": true,
               "stream_options": {"include_usage": true}}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    let text = String::from_utf8(
        axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    assert!(text.contains("hola"));
    assert!(text.contains("[DONE]"));

    let m = json_body(get(app, "/api/metrics").await).await;
    assert_eq!(m["endpoints"]["/v1/chat/completions"], 1);
    assert_eq!(m["prompt_tokens"], 8);
    assert_eq!(m["generated_tokens"], 3);
    assert_eq!(m["ttft_ms"]["samples"], 1);
    assert_eq!(m["decode_tok_s"]["samples"], 1);
}

#[tokio::test]
async fn status_informa_modelo_plan_y_cola() {
    let app = brasa_daemon::router(state());
    let resp = get(app, "/api/status").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let s = json_body(resp).await;
    assert_eq!(s["model"]["id"], "qwen3-4b-q4");
    assert_eq!(s["context"]["ctx"], 2048);
    assert_eq!(s["context"]["kv"], "f16");
    assert_eq!(s["plan"]["weights"], 2_000_000_000u64);
    assert_eq!(s["queue"], json!({"pending": 0, "running": 0}));
    assert!(s["uptime_s"].as_f64().unwrap() >= 0.0);
    assert_eq!(s["commit"], "test");
}
