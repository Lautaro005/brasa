//! Regla 4 de CLAUDE.md: el loop de decode no asigna memoria. Un allocator global cuenta las
//! asignaciones del heap de Rust durante pasos de decode ya en régimen (después del prefill).
//! Un único test por binario: el contador es global al proceso.
//!
//!   cargo test --release -p brasa-models --test decode_alloc -- --ignored --nocapture

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use brasa_metal::Context;
use brasa_models::qwen3::{KvType, Limits, Qwen3};

struct Counting;

static COUNT: AtomicUsize = AtomicUsize::new(0);
static ACTIVE: AtomicBool = AtomicBool::new(false);

// SAFETY: delega en el allocator del sistema; solo agrega un contador.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ACTIVE.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: mismo contrato que GlobalAlloc::alloc.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: mismo contrato que GlobalAlloc::dealloc.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new: usize) -> *mut u8 {
        if ACTIVE.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: mismo contrato que GlobalAlloc::realloc.
        unsafe { System.realloc(ptr, layout, new) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
#[ignore = "requiere models/qwen3-4b-q4/model.brasa"]
fn decode_sin_asignaciones() {
    let ctx = Context::new().unwrap();
    let limits = Limits {
        ctx: 512,
        max_tokens: 64,
        max_logit_rows: 1,
        kv: KvType::F16,
    };
    let mut model =
        Qwen3::load(&ctx, &root().join("models/qwen3-4b-q4/model.brasa"), limits).unwrap();
    let mut logits = vec![0f32; model.cfg.vocab];
    let prompt: Vec<u32> = (0..40).map(|i| 1000 + i).collect();
    model.forward(&ctx, &prompt, 0, 1, &mut logits).unwrap();
    // Un paso fuera de la medición, por si hay inicializaciones perezosas del sistema.
    model
        .forward(&ctx, &[42], prompt.len(), 1, &mut logits)
        .unwrap();

    let ids = [7u32];
    ACTIVE.store(true, Ordering::SeqCst);
    for step in 0..20 {
        model
            .forward(&ctx, &ids, prompt.len() + 1 + step, 1, &mut logits)
            .unwrap();
    }
    ACTIVE.store(false, Ordering::SeqCst);
    let n = COUNT.load(Ordering::SeqCst);
    println!("asignaciones en 20 pasos de decode: {n}");
    assert_eq!(n, 0, "el decode asignó memoria {n} veces");
}
