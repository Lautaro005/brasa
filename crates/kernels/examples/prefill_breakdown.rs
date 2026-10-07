//! Desglose del prefill de Qwen3-4B por kernel: cada operación de una capa se encola 36 veces
//! seguidas en un command buffer, como en el forward, para un bloque de T tokens en la posición
//! `pos0`, y se mide el tiempo de GPU (mediana). Pesos aleatorios con las formas reales.
//!   cargo run --release -p brasa-kernels --example prefill_breakdown -- [T] [pos0] [f16|q8_0|f32]
use brasa_kernels::testutil::{KvPair, Rng, median};
use brasa_kernels::{AttnShape, Kernels, KvType, QMatrix, RopeTable, WeightType};
use brasa_metal::{Arg, Command, Context};

const LAYERS: usize = 36;

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
    let arg = |i: usize| std::env::args().nth(i);
    let t: usize = arg(1).map_or(512, |s| s.parse().unwrap());
    let pos: usize = arg(2).map_or(0, |s| s.parse().unwrap());
    let kv = arg(3).map_or(KvType::F16, |s| KvType::parse(&s).unwrap());
    let ctx = Context::new().unwrap();
    let k = Kernels::new(&ctx).unwrap();
    let mut rng = Rng::new(3);
    let (h, ffn, hq, hkv, hd) = (2560, 9728, 32, 8, 128);
    let buf = |rng: &mut Rng, n: usize| ctx.buffer_from(&rng.vec(n, 1.0)).unwrap();
    let (x, hb, q, kn, vn, attn) = (
        buf(&mut rng, t * h),
        buf(&mut rng, t * h),
        buf(&mut rng, t * hq * hd),
        buf(&mut rng, t * hkv * hd),
        buf(&mut rng, t * hkv * hd),
        buf(&mut rng, t * hq * hd),
    );
    let (gate, up) = (buf(&mut rng, t * ffn), buf(&mut rng, t * ffn));
    let (nw, hw) = (buf(&mut rng, h), buf(&mut rng, hd));
    let q4 = |rng: &mut Rng, r: usize, c: usize| ctx.buffer_from(&rng.q4_0(r, c)).unwrap();
    let (wq, wk, wv, wo) = (
        q4(&mut rng, hq * hd, h),
        q4(&mut rng, hkv * hd, h),
        q4(&mut rng, hkv * hd, h),
        q4(&mut rng, h, hq * hd),
    );
    let (wg, wu, wd) = (
        q4(&mut rng, ffn, h),
        q4(&mut rng, ffn, h),
        q4(&mut rng, h, ffn),
    );
    let m = |data, rows, cols| QMatrix {
        data,
        qtype: WeightType::Q4_0,
        rows,
        cols,
    };
    let lk = pos + t;
    let rope = RopeTable::new(&ctx, 1e6, hd, lk).unwrap();
    let cap = lk.next_multiple_of(brasa_kernels::KV_ALIGN);
    let cache = KvPair::new(
        &ctx,
        kv,
        rng.vec(cap * hkv * hd, 1.0),
        rng.vec(cap * hkv * hd, 1.0),
    );
    let (kc, vc) = cache.args();
    let shape = AttnShape {
        tokens: t,
        hq,
        hkv,
        dim: hd,
        pos0: pos,
        kv,
    };
    println!(
        "{}; prefill T = {t} en pos0 = {pos}, KV {}",
        ctx.device_name(),
        kv.name()
    );
    println!("{:<34} {:>9} {:>7}", "operación (×36 capas)", "ms", "%");
    let mut rows: Vec<(&str, f64)> = Vec::new();
    macro_rules! rep {
        ($name:expr, $c:ident => $body:block) => {
            rows.push(($name, time(&ctx, |$c| {
                for _ in 0..LAYERS $body
            })));
        };
    }
    rep!("rms_norm ×2", c => {
        k.rms_norm(c, Arg::buf(&x), Arg::buf(&nw), Arg::buf(&hb), t, h, 1e-6);
        k.rms_norm(c, Arg::buf(&x), Arg::buf(&nw), Arg::buf(&hb), t, h, 1e-6);
    });
    rep!("gemm q, k, v", c => {
        k.gemm(c, m(&wq, hq * hd, h), Arg::buf(&hb), Arg::buf(&q), t);
        k.gemm(c, m(&wk, hkv * hd, h), Arg::buf(&hb), Arg::buf(&kn), t);
        k.gemm(c, m(&wv, hkv * hd, h), Arg::buf(&hb), Arg::buf(&vn), t);
    });
    let (kd, vd) = cache.args();
    rep!("qk_norm_rope_store", c => {
        k.qk_norm_rope_store(
            c,
            kv,
            [Arg::buf(&q), Arg::buf(&kn), Arg::buf(&vn)],
            [Arg::buf(&hw), Arg::buf(&hw)],
            1e-6,
            &rope,
            [kd, vd],
            shape,
        );
    });
    rep!("flash_attention", c => {
        k.flash_attention(c, Arg::buf(&q), kc, vc, Arg::buf(&attn), shape);
    });
    rep!("gemm o", c => {
        k.gemm(c, m(&wo, h, hq * hd), Arg::buf(&attn), Arg::buf(&hb), t);
    });
    rep!("add residual ×2", c => {
        k.add(c, &x, &hb, &x, t * h);
        k.add(c, &x, &hb, &x, t * h);
    });
    rep!("gemm gate, up", c => {
        k.gemm(c, m(&wg, ffn, h), Arg::buf(&hb), Arg::buf(&gate), t);
        k.gemm(c, m(&wu, ffn, h), Arg::buf(&hb), Arg::buf(&up), t);
    });
    rep!("swiglu", c => {
        k.swiglu(c, &gate, &up, &gate, t * ffn);
    });
    rep!("gemm down", c => {
        k.gemm(c, m(&wd, h, ffn), Arg::buf(&gate), Arg::buf(&hb), t);
    });
    let total: f64 = rows.iter().map(|r| r.1).sum();
    for (name, v) in &rows {
        println!("{name:<34} {:>9.1} {:>6.1}%", v * 1e3, v / total * 100.0);
    }
    println!("{:<34} {:>9.1}", "suma", total * 1e3);
}
