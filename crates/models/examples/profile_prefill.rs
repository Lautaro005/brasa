//! Perfil del prefill con el modelo real: tiempo de pared y de GPU de un prompt de `n` tokens
//! procesado en bloques de `chunk`, más el desglose por tramos de posición.
//!   cargo run --release -p brasa-models --example profile_prefill -- [n] [chunk] [f16|q8_0|f32] [carpeta del modelo]
//! El contexto se fija en `n` redondeado a 128 (como `brasa benchmark` con ctx = n + 128).
//! Con PP_DEPTHS=0,1536,7680,15872 mide en cambio un solo bloque de `chunk` tokens en cada
//! posición (como `llama-bench -p 512 -d ...`); el tiempo no depende del contenido de la KV.

use std::path::Path;
use std::time::Instant;

use brasa_metal::Context;
use brasa_models::qwen3::{KvType, Limits, Qwen3};

fn main() {
    let arg = |i: usize| std::env::args().nth(i);
    let n: usize = arg(1).map_or(16256, |s| s.parse().unwrap());
    let chunk: usize = arg(2).map_or(512, |s| s.parse().unwrap());
    let kv = arg(3).map_or(KvType::F16, |s| {
        KvType::parse(&s).expect("kv: f32|f16|q8_0")
    });
    let path = arg(4).map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen3-4b-q4/model.brasa"),
        |d| Path::new(&d).join("model.brasa"),
    );
    let ctx = Context::new().unwrap();
    let limits = Limits {
        ctx: (n + 1).next_multiple_of(128),
        max_tokens: chunk,
        max_logit_rows: 1,
        kv,
    };
    if let Ok(depths) = std::env::var("PP_DEPTHS") {
        let depths: Vec<usize> = depths.split(',').map(|d| d.parse().unwrap()).collect();
        let cap = depths.iter().max().unwrap() + chunk + 1;
        let limits = Limits {
            ctx: cap.next_multiple_of(128),
            ..limits
        };
        let mut m = Qwen3::load(&ctx, &path, limits).unwrap();
        let mut logits = vec![0f32; m.cfg.vocab];
        let ids: Vec<u32> = (0..chunk as u32)
            .map(|i| 1000 + (i * 7919) % 50000)
            .collect();
        println!("{}; chunk = {chunk}, KV {}", ctx.device_name(), kv.name());
        for d in depths {
            let mut t: Vec<f64> = (0..5)
                .map(|_| {
                    m.forward(&ctx, &ids, d, 1, &mut logits)
                        .unwrap()
                        .gpu_seconds
                })
                .skip(1)
                .collect();
            t.sort_by(f64::total_cmp);
            let g = t[t.len() / 2];
            println!(
                "pp{chunk} @ d{d:<6} GPU {:>8.1} ms  {:>7.1} tok/s",
                g * 1e3,
                chunk as f64 / g
            );
        }
        return;
    }
    let mut m = Qwen3::load(&ctx, &path, limits).unwrap();
    let mut logits = vec![0f32; m.cfg.vocab];
    let ids: Vec<u32> = (0..n as u32).map(|i| 1000 + (i * 7919) % 50000).collect();
    // Calentamiento (compila pipelines, toca buffers).
    m.forward(&ctx, &ids[..chunk.min(n)], 0, 1, &mut logits)
        .unwrap();

    let quarters = 4;
    let mut q_gpu = vec![0f64; quarters];
    let (mut gpu, t0) = (0f64, Instant::now());
    for (i, c) in ids.chunks(chunk).enumerate() {
        let g = m
            .forward(&ctx, c, i * chunk, 1, &mut logits)
            .unwrap()
            .gpu_seconds;
        gpu += g;
        q_gpu[(i * chunk * quarters / n).min(quarters - 1)] += g;
    }
    let wall = t0.elapsed().as_secs_f64();
    println!(
        "{}; n = {n}, chunk = {chunk}, KV {}",
        ctx.device_name(),
        kv.name()
    );
    println!(
        "pared {:.2} s ({:.1} tok/s), GPU {:.2} s, fuera de GPU {:.2} s",
        wall,
        n as f64 / wall,
        gpu,
        wall - gpu
    );
    for (i, g) in q_gpu.iter().enumerate() {
        println!(
            "  posiciones {:>6}..{:<6} GPU {:>7.2} s",
            i * n / quarters,
            (i + 1) * n / quarters,
            g
        );
    }
}
