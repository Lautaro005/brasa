//! U1: tests del daemon con un engine simulado (sin GPU ni pesos). Verifican los contadores de
//! `/api/metrics` tras un pedido streaming y uno no streaming, y la forma de `/api/status`.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use brasa_core::chat::{ChatEvent, FinishReason, Usage};
use brasa_daemon::engine::{Engine, LoadedModel};
use brasa_daemon::{AppState, ServerMeta};
use brasa_memory::planner::{Budget, MemoryPlan};
use brasa_tokenizer::Tokenizer;
use serde_json::{Value, json};
use tower::ServiceExt;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn state() -> Arc<AppState> {
    state_with(Engine::simulated(|req, emit| {
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
    }))
}

fn state_with(engine: Engine) -> Arc<AppState> {
    let tok = Tokenizer::from_dir(&root().join("fixtures/qwen3-4b/tokenizer")).unwrap();
    let model = LoadedModel {
        path: "/tmp/falso/model.brasa".into(),
        weights_sha256_declarado: "ab".repeat(32),
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
    let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
    let meta = ServerMeta {
        model_id: "qwen3-4b-q4".into(),
        model_dir: root().join("models/qwen3-4b-q4"),
        addr,
        budget: Budget::profile(16),
        bench_dir: root().join("docs/bench"),
        commit: "test".into(),
    };
    AppState::new(engine, tok, model, meta)
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
async fn gui_assets_se_sirven_con_su_content_type() {
    let app = brasa_daemon::router(state());
    let casos = [
        ("/ui", "text/html"),
        ("/ui/", "text/html"),
        ("/ui/app.css", "text/css"),
        ("/ui/app.js", "application/javascript"),
    ];
    for (uri, ct) in casos {
        let resp = get(app.clone(), uri).await;
        assert_eq!(resp.status(), StatusCode::OK, "{uri}");
        let got = resp
            .headers()
            .get("content-type")
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        assert!(got.starts_with(ct), "{uri}: {got}");
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(!text.contains("http://"), "{uri} referencia http");
        assert!(!text.contains("https://"), "{uri} referencia https");
    }
}

#[tokio::test]
async fn agents_devuelve_config_por_herramienta() {
    let app = brasa_daemon::router(state());
    let v = json_body(get(app, "/api/agents?ctx=4096").await).await;
    assert_eq!(v["ctx"], 4096);
    for key in ["codex", "claude-code", "cline", "opencode"] {
        assert!(
            v["tools"][key].as_str().unwrap().contains("127.0.0.1:8080"),
            "{key}"
        );
    }
    assert!(v["tools"]["claude-code"].as_str().unwrap().contains("4096"));
}

#[tokio::test]
async fn plan_y_agents_validan_el_ctx() {
    let app = brasa_daemon::router(state());
    // Vacío, cero y fuera de rango son 400 con mensaje, no un pánico en debug.
    for uri in [
        "/api/plan?ctx=",
        "/api/plan?ctx=0",
        "/api/plan?ctx=999999999",
        "/api/plan?ctx=abc",
        "/api/agents?ctx=",
        "/api/agents?ctx=0",
    ] {
        let resp = get(app.clone(), uri).await;
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST, "{uri}");
        let v = json_body(resp).await;
        assert!(v["error"].as_str().unwrap().contains("ctx"), "{uri}: {v}");
    }
    // Un ctx válido sigue funcionando.
    let resp = get(app, "/api/agents?ctx=2048").await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn bench_ignora_los_json_que_no_son_reportes() {
    // La carpeta real trae `doctor.json`, que no es un reporte.
    let app = brasa_daemon::router(state());
    let v = json_body(get(app, "/api/bench").await).await;
    assert!(v["reports"].is_array());
    let ignorados = v["ignored"].as_array().unwrap();
    assert!(
        ignorados
            .iter()
            .any(|f| f.as_str().unwrap().contains("doctor.json")),
        "{ignorados:?}"
    );
    for r in v["reports"].as_array().unwrap() {
        assert!(r["engine"].as_str().is_some(), "{r}");
    }
}

#[tokio::test]
async fn bench_responde_aunque_no_haya_carpeta() {
    let app = brasa_daemon::router(state());
    let resp = get(app, "/api/bench").await;
    assert_eq!(resp.status(), StatusCode::OK);
    let v = json_body(resp).await;
    assert!(v["reports"].is_array());
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

#[tokio::test]
async fn metrics_cuentan_un_stream_cancelado() {
    // El engine genera hasta que el envío falla (el cliente se fue) y entonces manda el Done
    // con el uso real, que ya no llega al handler: tiene que sumarse igual.
    let s = state_with(Engine::simulated(|_, emit| {
        let mut n = 0;
        while n < 100_000 && emit(ChatEvent::Text("x".into())) {
            n += 1;
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
        emit(ChatEvent::Done {
            reason: FinishReason::Cancelled,
            usage: Usage {
                input_tokens: 5,
                output_tokens: n,
                cached_tokens: 0,
            },
        });
    }));
    let app = brasa_daemon::router(s.clone());
    let resp = post(
        app.clone(),
        "/v1/chat/completions",
        json!({"messages": [{"role": "user", "content": "hola"}], "stream": true}),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    use tokio_stream::StreamExt;
    let mut body = resp.into_body().into_data_stream();
    let first = body.next().await.unwrap().unwrap();
    assert!(!first.is_empty());
    drop(body);

    let mut m = Value::Null;
    for _ in 0..200 {
        m = json_body(get(app.clone(), "/api/metrics").await).await;
        if m["cancelled"] == 1 && m["generated_tokens"].as_u64() > Some(0) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(m["cancelled"], 1, "{m}");
    assert_eq!(m["prompt_tokens"], 5, "{m}");
    assert!(m["generated_tokens"].as_u64().unwrap() > 0, "{m}");
    assert!(m["errors"].as_object().unwrap().is_empty(), "{m}");
}

// --- V5: Model Manager (ADR 0025) ---

fn pedido() -> Value {
    json!({"messages": [{"role": "user", "content": "hola"}], "stream": false})
}

async fn estado(app: Router) -> String {
    let v = json_body(get(app, "/api/status").await).await;
    v["state"].as_str().unwrap_or("?").to_string()
}

#[tokio::test]
async fn model_manager_transiciones() {
    let s = state();
    let app = brasa_daemon::router(s.clone());
    assert_eq!(estado(app.clone()).await, "loaded");

    // idle libera el modelo.
    let r = post(app.clone(), "/api/model/idle", json!({})).await;
    assert_eq!(r.status(), StatusCode::OK);
    assert_eq!(json_body(r).await["state"], "idle");
    assert_eq!(s.engine.state().name(), "idle");
    assert_eq!(estado(app.clone()).await, "idle");

    // pausar sin modelo es un conflicto, no un pánico.
    let r = post(app.clone(), "/api/model/pause", json!({})).await;
    assert_eq!(r.status(), StatusCode::CONFLICT);

    // load lo vuelve a cargar.
    let r = post(app.clone(), "/api/model/load", json!({})).await;
    assert_eq!(r.status(), StatusCode::OK);
    assert_eq!(s.engine.state().name(), "loaded");

    // pause deja la cola detenida; resume la vuelve a mover.
    let r = post(app.clone(), "/api/model/pause", json!({})).await;
    assert_eq!(r.status(), StatusCode::OK);
    assert_eq!(s.engine.state().name(), "paused");

    // load limpia la pausa y, sin pausa, resume es un conflicto.
    let r = post(app.clone(), "/api/model/load", json!({})).await;
    assert_eq!(r.status(), StatusCode::OK);
    let r = post(app.clone(), "/api/model/resume", json!({})).await;
    assert_eq!(r.status(), StatusCode::CONFLICT);

    // stop deja el modelo detenido.
    let r = post(app.clone(), "/api/model/stop", json!({})).await;
    assert_eq!(r.status(), StatusCode::OK);
    assert_eq!(s.engine.state().name(), "stopped");
}

#[tokio::test]
async fn pedido_en_idle_vuelve_a_cargar() {
    let s = state();
    let app = brasa_daemon::router(s.clone());
    post(app.clone(), "/api/model/idle", json!({})).await;

    let resp = post(app.clone(), "/v1/chat/completions", pedido()).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_body(resp).await;
    assert_eq!(body["choices"][0]["message"]["content"], "hola mundo");
    // El pedido dejó el modelo cargado otra vez.
    assert_eq!(s.engine.state().name(), "loaded");
}

#[tokio::test]
async fn pedido_en_pausa_espera_al_resume() {
    let s = state();
    let app = brasa_daemon::router(s.clone());
    post(app.clone(), "/api/model/pause", json!({})).await;

    // Con el modelo en pausa el pedido queda encolado: no responde dentro del plazo.
    let encolado = tokio::time::timeout(
        std::time::Duration::from_millis(400),
        post(app.clone(), "/v1/chat/completions", pedido()),
    )
    .await;
    assert!(encolado.is_err(), "respondió con el modelo en pausa");

    // Al reanudar, la cola vuelve a moverse.
    let r = post(app.clone(), "/api/model/resume", json!({})).await;
    assert_eq!(r.status(), StatusCode::OK);
    let resp = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        post(app.clone(), "/v1/chat/completions", pedido()),
    )
    .await
    .expect("el pedido no se sirvió tras resume");
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_body(resp).await;
    assert_eq!(body["choices"][0]["message"]["content"], "hola mundo");
}

#[tokio::test]
async fn status_informa_el_estado() {
    let app = brasa_daemon::router(state());
    let v = json_body(get(app.clone(), "/api/status").await).await;
    assert_eq!(v["state"], "loaded");
    // El resto de `/api/status` no cambió.
    assert_eq!(v["model"]["id"], "qwen3-4b-q4");
    assert_eq!(v["context"]["ctx"], 2048);
    assert_eq!(v["queue"]["pending"], 0);
}
