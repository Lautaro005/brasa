//! Atención de prefill (Qwen3-4B: hq 32, hkv 8, head_dim 128) con un bloque de 512 queries según
//! la posición: ms por capa y GFLOP/s de la parte causal. Mide la variante GQA de la ruta caliente.
//! Uso: cargo run --release -p brasa-kernels --example flash_sweep -- [f16|q8_0|f32]

use brasa_kernels::testutil::{KvPair, Rng, median};
use brasa_kernels::{AttnShape, Kernels, KvType};
use brasa_metal::{Arg, Command, Context};

fn time<'a>(ctx: &Context, mut encode: impl FnMut(&mut Command<'a>)) -> f64 {
    let t: Vec<f64> = (0..7)
        .map(|_| {
            let mut cmd = ctx.command().unwrap();
            encode(&mut cmd);
            cmd.commit_and_wait().unwrap().gpu_seconds
        })
        .skip(2)
        .collect();
    median(t)
}

fn main() {
    let kv = std::env::args().nth(1).map_or(KvType::F16, |s| {
        KvType::parse(&s).expect("kv: f32|f16|q8_0")
    });
    let ctx = Context::new().unwrap();
    let k = Kernels::new(&ctx).unwrap();
    let mut rng = Rng::new(9);
    let (hq, hkv, hd, tokens) = (32usize, 8usize, 128usize, 512usize);
    println!(
        "dispositivo: {}; KV {}; T = {tokens}",
        ctx.device_name(),
        kv.name()
    );
    println!("{:>6} {:>9} {:>9}", "pos0", "ms", "GFLOP/s");
    // FA_POS=15872 limita la corrida a una posición (para perfilar con xctrace).
    let only: Option<usize> = std::env::var("FA_POS").ok().and_then(|v| v.parse().ok());
    for pos0 in [0usize, 1536, 7680, 15872] {
        if only.is_some_and(|p| p != pos0) {
            continue;
        }
        let lk = pos0 + tokens;
        let cap = lk.next_multiple_of(brasa_kernels::KV_ALIGN);
        let n = cap * hkv * hd;
        let cache = KvPair::new(&ctx, kv, rng.vec(n, 1.0), rng.vec(n, 1.0));
        let (kc, vc) = cache.args();
        let q = ctx.buffer_from(&rng.vec(tokens * hq * hd, 1.0)).unwrap();
        let o = ctx.buffer::<f32>(tokens * hq * hd).unwrap();
        let shape = AttnShape {
            tokens,
            hq,
            hkv,
            dim: hd,
            pos0,
            kv,
        };
        let t = time(&ctx, |c| {
            k.flash_attention(c, Arg::buf(&q), kc, vc, Arg::buf(&o), shape)
        });
        let keys_seen = (tokens * pos0 + tokens * (tokens + 1) / 2) as f64;
        let flops = 4.0 * (hq * hd) as f64 * keys_seen;
        println!("{pos0:>6} {:>9.3} {:>9.1}", t * 1e3, flops / t / 1e9);
    }
}
