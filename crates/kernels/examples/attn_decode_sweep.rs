//! Atención de decode (Qwen3-4B: hq 32, hkv 8, head_dim 128) según la longitud de la caché:
//! µs por capa con 36 capas encoladas en un command buffer y una caché distinta por capa (como en
//! el modelo; una sola caché quedaría en la SLC). Sirve para separar costo fijo de costo por clave.
//! Compara la ruta caliente (`decode_attention_lanes`) con `decode_attention`.
//! Uso: cargo run --release -p brasa-kernels --example attn_decode_sweep -- [f16|q8_0|f32]

use brasa_kernels::testutil::{KvPair, Rng, median};
use brasa_kernels::{AttnShape, Kernels, KvType};
use brasa_metal::{Arg, Command, Context};

const LAYERS: usize = 36;

fn time<'a>(ctx: &Context, mut encode: impl FnMut(&mut Command<'a>)) -> f64 {
    let t: Vec<f64> = (0..9)
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
    let mut rng = Rng::new(5);
    let (hq, hkv, hd) = (32usize, 8usize, 128usize);
    println!("dispositivo: {}; KV {}", ctx.device_name(), kv.name());
    println!(
        "{:>6} {:>12} {:>9} {:>14}",
        "claves", "lanes µs/capa", "GB/s", "partial µs/capa"
    );
    for lk in [128usize, 512, 1024, 2048, 4096, 8192, 16384] {
        let n = lk.next_multiple_of(128) * hkv * hd;
        let caches: Vec<KvPair> = (0..LAYERS.min(12))
            .map(|_| KvPair::new(&ctx, kv, rng.vec(n, 1.0), rng.vec(n, 1.0)))
            .collect();
        let q = ctx.buffer_from(&rng.vec(hq * hd, 1.0)).unwrap();
        let o = ctx.buffer::<f32>(hq * hd).unwrap();
        let part = ctx
            .buffer::<f32>(brasa_kernels::decode_partials_len(hq, lk))
            .unwrap();
        let shape = AttnShape {
            tokens: 1,
            hq,
            hkv,
            dim: hd,
            pos0: lk - 1,
            kv,
        };
        let t = time(&ctx, |c| {
            for l in 0..LAYERS {
                let (kc, vc) = caches[l % caches.len()].args();
                k.decode_attention_lanes(c, Arg::buf(&q), kc, vc, &part, Arg::buf(&o), shape);
            }
        }) / LAYERS as f64;
        let t_old = time(&ctx, |c| {
            for l in 0..LAYERS {
                let (kc, vc) = caches[l % caches.len()].args();
                k.decode_attention(c, Arg::buf(&q), kc, vc, &part, Arg::buf(&o), shape);
            }
        }) / LAYERS as f64;
        let bytes = 2.0 * kv.bytes(lk * hkv * hd) as f64;
        println!(
            "{lk:>6} {:>12.1} {:>9.1} {:>14.1}",
            t * 1e6,
            bytes / t / 1e9,
            t_old * 1e6
        );
    }
}
