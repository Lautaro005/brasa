//! T1.8: el plan coincide con lo que se reserva de verdad, el overhead medido del proceso entra
//! en la constante del planner y un contexto que no entra se rechaza antes de cargar.
//! Un único test por binario (mide la huella del proceso).
//!
//!   cargo test --release -p brasa-runtime --test planner -- --ignored --nocapture

use std::path::{Path, PathBuf};

use brasa_memory::planner::{Budget, PROCESS_OVERHEAD, gib};
use brasa_memory::process::process_memory;
use brasa_runtime::{Limits, Session};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
#[ignore = "requiere models/qwen3-4b-q4"]
fn plan_igual_a_lo_reservado_y_rechazo() {
    let dir = root().join("models/qwen3-4b-q4");
    let budget = Budget::this_machine().unwrap();
    println!(
        "presupuesto: {:.2} GiB ({})",
        gib(budget.bytes),
        budget.source
    );

    // Contexto absurdo: se rechaza con mensaje claro, sin cargar pesos.
    let before = process_memory().unwrap().footprint;
    let big = Limits {
        ctx: 40_960,
        max_tokens: 512,
        max_logit_rows: 1,
    };
    let err = Session::load(&dir, big).unwrap_err();
    println!("rechazo: {err}");
    assert!(
        err.0.contains("no entra") && err.0.contains("--ctx"),
        "{err}"
    );
    let after_reject = process_memory().unwrap().footprint;
    assert!(
        after_reject < before + (64 << 20),
        "el rechazo cargó memoria"
    );

    // Contexto que entra: plan exacto en pesos, KV y workspace; overhead dentro de la constante.
    let limits = Limits {
        ctx: 4096,
        max_tokens: 128,
        max_logit_rows: 1,
    };
    let before = process_memory().unwrap().footprint;
    let (session, plan) = Session::load_with_budget(&dir, limits, &budget).unwrap();
    let after = process_memory().unwrap().footprint;
    let a = session.allocated();
    assert_eq!(a.weights, plan.weights, "pesos");
    assert_eq!(a.kv, plan.kv, "KV");
    assert_eq!(a.workspace, plan.workspace, "workspace");
    let buffers = a.weights + a.kv + a.workspace;
    let overhead = (after - before).saturating_sub(buffers);
    println!(
        "plan {:.3} GiB; huella medida {:.3} GiB; buffers {:.3} GiB; overhead medido {:.0} MiB (constante {} MiB)",
        gib(plan.total),
        gib(after - before),
        gib(buffers),
        overhead as f64 / (1 << 20) as f64,
        PROCESS_OVERHEAD >> 20
    );
    assert!(
        overhead <= PROCESS_OVERHEAD,
        "overhead {overhead} mayor que la constante"
    );
    assert!(
        after - before <= plan.total,
        "la huella real supera el plan"
    );
}
