//! Perfil del decode: tiempo de pared vs tiempo de GPU por token, en una posición dada.
//!   cargo run --release -p brasa-models --example profile_decode -- [posición]

use std::path::Path;
use std::time::Instant;

use brasa_metal::Context;
use brasa_models::qwen3::{Limits, Qwen3};

fn main() {
    let pos: usize = std::env::args().nth(1).map_or(2000, |s| s.parse().unwrap());
    let ctx_len: usize = std::env::args()
        .nth(2)
        .map_or(pos + 64, |s| s.parse().unwrap());
    let ctx = Context::new().unwrap();
    let limits = Limits {
        ctx: ctx_len,
        max_tokens: 512,
        max_logit_rows: 1,
    };
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen3-4b-q4/model.brasa");
    let mut m = Qwen3::load(&ctx, &path, limits).unwrap();
    let mut logits = vec![0f32; m.cfg.vocab];
    let ids: Vec<u32> = (0..pos as u32).map(|i| 1000 + i % 5000).collect();
    for (i, c) in ids.chunks(512).enumerate() {
        m.forward(&ctx, c, i * 512, 1, &mut logits).unwrap();
    }
    let (mut wall, mut gpu) = (0f64, 0f64);
    let n = 32;
    for s in 0..n {
        let t = Instant::now();
        let g = m.forward(&ctx, &[42], pos + s, 1, &mut logits).unwrap();
        wall += t.elapsed().as_secs_f64();
        gpu += g.gpu_seconds;
    }
    println!(
        "posición {pos}: pared {:.2} ms/token, GPU {:.2} ms/token, fuera de GPU {:.2} ms ({:.1} tok/s)",
        wall / n as f64 * 1e3,
        gpu / n as f64 * 1e3,
        (wall - gpu) / n as f64 * 1e3,
        n as f64 / wall
    );
}
