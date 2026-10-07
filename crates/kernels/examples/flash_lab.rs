//! Laboratorio de atención de prefill: compila un kernel candidato desde un .metal (con el acceso
//! a la KV de `kv_access.metal` antepuesto y `GQA_G 4`) y lo compara con `flash_attention`
//! (Qwen3-4B: hq 32, hkv 8, head_dim 128, T queries) en varias posiciones.
//!   cargo run --release -p brasa-kernels --example flash_lab -- <archivo.metal> <kernel> [f16|q8_0|f32] [T]
//! Grilla: [T / LAB_QT, hq / LAB_HG] threadgroups de LAB_NT hilos (por defecto 8, 1, 128).
use brasa_kernels::testutil::{KvPair, Rng, median};
use brasa_kernels::{AttnShape, Kernels, KvType, kv_source};
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

fn env(k: &str, d: usize) -> usize {
    std::env::var(k).ok().map_or(d, |v| v.parse().unwrap())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let kv = a.get(3).map_or(KvType::F16, |s| KvType::parse(s).unwrap());
    let tokens: usize = a.get(4).map_or(512, |s| s.parse().unwrap());
    let src = format!(
        "#define GQA_G 4\n{}",
        std::fs::read_to_string(&a[1]).unwrap()
    );
    let ctx = Context::new().unwrap();
    let k = Kernels::new(&ctx).unwrap();
    let p = ctx.pipeline(&kv_source(kv, &src), &a[2]).unwrap();
    let (qt, hg, nt) = (env("LAB_QT", 8), env("LAB_HG", 1), env("LAB_NT", 128));
    let (hq, hkv, hd) = (32usize, 8usize, 128usize);
    println!(
        "{}; {} max hilos/tg {}; KV {}; T = {tokens}",
        ctx.device_name(),
        a[2],
        p.max_threads_per_threadgroup(),
        kv.name()
    );
    let mut rng = Rng::new(9);
    let positions: Vec<usize> = std::env::var("LAB_POS").map_or(vec![0, 1536, 7680, 15872], |v| {
        v.split(',').map(|x| x.parse().unwrap()).collect()
    });
    for pos0 in positions {
        let lk = pos0 + tokens;
        let cap = lk.next_multiple_of(brasa_kernels::KV_ALIGN);
        let n = cap * hkv * hd;
        let cache = KvPair::new(&ctx, kv, rng.vec(n, 1.0), rng.vec(n, 1.0));
        let (kc, vc) = cache.args();
        // Q con escala ~3 para que el softmax no sea casi uniforme.
        let q = ctx.buffer_from(&rng.vec(tokens * hq * hd, 3.0)).unwrap();
        let o0 = ctx.buffer::<f32>(tokens * hq * hd).unwrap();
        let o1 = ctx.buffer::<f32>(tokens * hq * hd).unwrap();
        let shape = AttnShape {
            tokens,
            hq,
            hkv,
            dim: hd,
            pos0,
            kv,
        };
        let t0 = time(&ctx, |c| {
            k.flash_attention(c, Arg::buf(&q), kc, vc, Arg::buf(&o0), shape)
        });
        let scale = 1.0 / (hd as f32).sqrt();
        let t1 = time(&ctx, |c| {
            c.dispatch_groups(
                &p,
                &[
                    Arg::buf(&q),
                    kc,
                    vc,
                    Arg::buf(&o1),
                    Arg::u32(tokens as u32),
                    Arg::u32(hkv as u32),
                    Arg::u32(pos0 as u32),
                    Arg::f32(scale),
                ],
                [tokens.div_ceil(qt), hq / hg, 1],
                [nt, 1, 1],
            )
        });
        let (r0, r1) = (o0.as_slice(), o1.as_slice());
        let mx = r0.iter().fold(0f32, |m, v| m.max(v.abs()));
        let err = r0
            .iter()
            .zip(r1)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        let keys_seen = (tokens * pos0 + tokens * (tokens + 1) / 2) as f64;
        let fl = 4.0 * (hq * hd) as f64 * keys_seen;
        println!(
            "pos0 {pos0:>6}: actual {:>8.3} ms {:>7.1} GFLOP/s  candidato {:>8.3} ms {:>7.1} GFLOP/s  ({:+.1} %)  err {:.2e} / max|o|",
            t0 * 1e3,
            fl / t0 / 1e9,
            t1 * 1e3,
            fl / t1 / 1e9,
            (t0 / t1 - 1.0) * 100.0,
            err / mx
        );
    }
}
