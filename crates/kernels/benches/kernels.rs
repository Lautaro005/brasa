//! Microbenchmarks de kernels con las formas de Qwen3-4B: tiempo de GPU (mediana) y ancho de
//! banda o GFLOP/s efectivos. Uso: cargo bench -p brasa-kernels
//! Las cifras son de kernels sin optimizar (fase 1); sirven de línea base para la fase 3.

use brasa_kernels::testutil::KvPair;
use brasa_kernels::testutil::{Rng, median};
use brasa_kernels::{AttnShape, Kernels, KvType, QMatrix, RopeTable, WeightType};
use brasa_metal::{Arg, Command, Context};

const REPS: usize = 20;

fn time<'a>(ctx: &Context, mut encode: impl FnMut(&mut Command<'a>)) -> f64 {
    let times: Vec<f64> = (0..REPS + 2)
        .map(|_| {
            let mut cmd = ctx.command().unwrap();
            encode(&mut cmd);
            cmd.commit_and_wait().unwrap().gpu_seconds
        })
        .skip(2)
        .collect();
    median(times)
}

fn report(name: &str, t: f64, bytes: f64, flops: f64) {
    let mut s = format!(
        "{name:<36} {:>9.3} ms  {:>7.1} GB/s",
        t * 1e3,
        bytes / t / 1e9
    );
    if flops > 0.0 {
        s += &format!("  {:>7.1} GFLOP/s", flops / t / 1e9);
    }
    println!("{s}");
}

fn main() {
    let ctx = Context::new().expect("contexto Metal");
    let t0 = std::time::Instant::now();
    let k = Kernels::new(&ctx).expect("kernels");
    let compile_ms = t0.elapsed().as_secs_f64() * 1e3;
    let mut rng = Rng::new(1);
    println!("dispositivo: {}", ctx.device_name());
    println!("Kernels::new (compilar todos los pipelines): {compile_ms:.0} ms");

    let (h, ffn, heads, hd) = (2560usize, 9728usize, 32usize, 128usize);
    let big = 512 * ffn;
    let a = ctx.buffer_from(&rng.vec(big, 1.0)).unwrap();
    let b = ctx.buffer_from(&rng.vec(big, 1.0)).unwrap();
    let out = ctx.buffer::<f32>(big).unwrap();
    let w = ctx.buffer_from(&rng.vec(h, 1.0)).unwrap();

    let n = 1 << 24;
    let t = time(&ctx, |c| k.add(c, &a, &b, &out, n.min(big)));
    report("add_f32 n=4.98M", t, 3.0 * 4.0 * big.min(n) as f64, 0.0);
    for tokens in [1, 512] {
        let n = tokens * ffn;
        let t = time(&ctx, |c| k.swiglu(c, &a, &b, &out, n));
        report(
            &format!("swiglu_f32 T={tokens} n={ffn}"),
            t,
            12.0 * n as f64,
            0.0,
        );
        let t = time(&ctx, |c| {
            k.rms_norm(
                c,
                Arg::buf(&a),
                Arg::buf(&w),
                Arg::buf(&out),
                tokens,
                h,
                1e-6,
            )
        });
        report(
            &format!("rms_norm_f32 T={tokens} H={h}"),
            t,
            8.0 * (tokens * h) as f64,
            0.0,
        );
    }
    let t = time(&ctx, |c| k.softmax(c, &a, &out, heads, 2048));
    report("softmax_f32 32×2048", t, 8.0 * (heads * 2048) as f64, 0.0);
    let table = RopeTable::new(&ctx, 1e6, hd, 4096).unwrap();
    for tokens in [1, 512] {
        let t = time(&ctx, |c| {
            k.rope_neox(c, Arg::buf(&a), &table, tokens, heads, hd, 0)
        });
        report(
            &format!("rope_neox_f32 T={tokens} 32×128"),
            t,
            8.0 * (tokens * heads * hd) as f64,
            0.0,
        );
    }

    let vocab = 151_936;
    let emb = ctx.buffer_from(&rng.q8_0(vocab, h)).unwrap();
    let ids = ctx
        .buffer_from(
            &(0..512u32)
                .map(|i| i * 271 % vocab as u32)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let emb_m = QMatrix {
        data: &emb,
        qtype: WeightType::Q8_0,
        rows: vocab,
        cols: h,
    };
    let t = time(&ctx, |c| k.embed(c, emb_m, &ids, Arg::buf(&out), 512));
    report(
        "embed_q8_0 T=512",
        t,
        (512 * h) as f64 * (34.0 / 32.0 + 4.0),
        0.0,
    );

    let x = ctx.buffer_from(&rng.vec(512 * ffn, 1.0)).unwrap();
    let shapes = [(4096, h, "q_proj"), (ffn, h, "gate/up"), (h, ffn, "down")];
    for (rows, cols, name) in shapes {
        let wq = ctx.buffer_from(&rng.q4_0(rows, cols)).unwrap();
        let m = QMatrix {
            data: &wq,
            qtype: WeightType::Q4_0,
            rows,
            cols,
        };
        let wbytes = (rows * cols) as f64 * 18.0 / 32.0;
        let t = time(&ctx, |c| {
            k.gemv_simple(c, m, Arg::buf(&x), Arg::buf(&out), 1)
        });
        report(
            &format!("gemv_simple_q4_0 {name} {rows}×{cols} T=1"),
            t,
            wbytes,
            2.0 * (rows * cols) as f64,
        );
        let t = time(&ctx, |c| k.gemv(c, m, Arg::buf(&x), Arg::buf(&out), 1));
        report(
            &format!("gemv_q4_0 {name} {rows}×{cols} T=1"),
            t,
            wbytes,
            2.0 * (rows * cols) as f64,
        );
        let t = time(&ctx, |c| {
            k.gemm_naive(c, m, Arg::buf(&x), Arg::buf(&out), 512)
        });
        report(
            &format!("gemm_naive_q4_0 {name} {rows}×{cols} T=512"),
            t,
            wbytes,
            2.0 * 512.0 * (rows * cols) as f64,
        );
        let t = time(&ctx, |c| k.gemm(c, m, Arg::buf(&x), Arg::buf(&out), 512));
        report(
            &format!("gemm_tiled_q4_0 {name} {rows}×{cols} T=512"),
            t,
            wbytes,
            2.0 * 512.0 * (rows * cols) as f64,
        );
    }
    let t = time(&ctx, |c| {
        k.gemv_simple(c, emb_m, Arg::buf(&x), Arg::buf(&out), 1)
    });
    report(
        "gemv_simple_q8_0 lm_head 151936×2560 T=1",
        t,
        (vocab * h) as f64 * 34.0 / 32.0,
        2.0 * (vocab * h) as f64,
    );
    let t = time(&ctx, |c| k.gemv(c, emb_m, Arg::buf(&x), Arg::buf(&out), 1));
    report(
        "gemv_q8_0 lm_head 151936×2560 T=1",
        t,
        (vocab * h) as f64 * 34.0 / 32.0,
        2.0 * (vocab * h) as f64,
    );
    let emb6 = ctx.buffer_from(&rng.q6_0(vocab, h)).unwrap();
    let head6 = QMatrix {
        data: &emb6,
        qtype: WeightType::Q6_0,
        rows: vocab,
        cols: h,
    };
    let t = time(&ctx, |c| k.gemv(c, head6, Arg::buf(&x), Arg::buf(&out), 1));
    report(
        "gemv_q6_0 lm_head 151936×2560 T=1",
        t,
        (vocab * h) as f64 * 26.0 / 32.0,
        2.0 * (vocab * h) as f64,
    );

    // Atención: la simple (sin tiling, solo KV f32) y las de la ruta caliente con KV f32 y f16.
    // FLOPs de la parte causal exacta: cada query i ve pos0 + i + 1 claves.
    let (hq, hkv) = (32usize, 8usize);
    for (tokens, pos0) in [
        (1usize, 2047usize),
        (1, 8191),
        (1, 16383),
        (64, 448),
        (128, 1920),
        (512, 15872),
    ] {
        let lk = pos0 + tokens;
        let keys_seen = (tokens * pos0 + tokens * (tokens + 1) / 2) as f64;
        let flops = 4.0 * (hq * hd) as f64 * keys_seen;
        let q = ctx.buffer_from(&rng.vec(tokens * hq * hd, 1.0)).unwrap();
        let cap = lk.next_multiple_of(brasa_kernels::KV_ALIGN);
        let (kv_k, kv_v) = (rng.vec(cap * hkv * hd, 1.0), rng.vec(cap * hkv * hd, 1.0));
        let o = ctx.buffer::<f32>(tokens * hq * hd).unwrap();
        let shape = AttnShape {
            tokens,
            hq,
            hkv,
            dim: hd,
            pos0,
            kv: KvType::F32,
        };
        if tokens <= 128 {
            let kc = ctx.buffer_from(&kv_k).unwrap();
            let vc = ctx.buffer_from(&kv_v).unwrap();
            let scores = ctx.buffer::<f32>(tokens * hq * lk).unwrap();
            let t = time(&ctx, |c| {
                k.attention(
                    c,
                    Arg::buf(&q),
                    Arg::buf(&kc),
                    Arg::buf(&vc),
                    &scores,
                    Arg::buf(&o),
                    shape,
                )
            });
            report(
                &format!("attention_simple T={tokens} ctx={lk}"),
                t,
                2.0 * 4.0 * (lk * hkv * hd) as f64,
                flops,
            );
        }
        for kv in [KvType::F32, KvType::F16, KvType::Q8_0, KvType::Tq4] {
            // TQ4 se escribe con el kernel (la rotación cambia el dominio, ver `KvPair`).
            let tq = (kv == KvType::Tq4).then(|| {
                let rot = ctx.buffer_from(&brasa_quant::turbo::rotation()).unwrap();
                let (src_k, src_v) = (
                    ctx.buffer_from(&kv_k).unwrap(),
                    ctx.buffer_from(&kv_v).unwrap(),
                );
                let (ck, cv) = (
                    ctx.buffer::<u8>(kv.bytes(kv_k.len())).unwrap(),
                    ctx.buffer::<u8>(kv.bytes(kv_v.len())).unwrap(),
                );
                let mut cmd = ctx.command().unwrap();
                let rows = kv_k.len() / hd;
                k.tq_store(
                    &mut cmd,
                    Arg::buf(&src_k),
                    Arg::buf(&ck),
                    Arg::buf(&rot),
                    rows,
                );
                k.tq_store(
                    &mut cmd,
                    Arg::buf(&src_v),
                    Arg::buf(&cv),
                    Arg::buf(&rot),
                    rows,
                );
                cmd.commit_and_wait().unwrap();
                (ck, cv)
            });
            let cache =
                (kv != KvType::Tq4).then(|| KvPair::new(&ctx, kv, kv_k.clone(), kv_v.clone()));
            let (kc, vc) = match (&cache, &tq) {
                (Some(c), _) => c.args(),
                (None, Some((ck, cv))) => (Arg::buf(ck), Arg::buf(cv)),
                _ => unreachable!(),
            };
            let shape = AttnShape { kv, ..shape };
            let kv_bytes = 2.0 * (kv.block_bytes() as f64 / 32.0) * (lk * hkv * hd) as f64;
            let kvn = kv.name();
            let t = time(&ctx, |c| {
                k.flash_attention(c, Arg::buf(&q), kc, vc, Arg::buf(&o), shape)
            });
            report(
                &format!("flash_attention kv={kvn} T={tokens} ctx={lk}"),
                t,
                kv_bytes,
                flops,
            );
            if tokens == 1 {
                let part = ctx
                    .buffer::<f32>(brasa_kernels::decode_partials_len(hq, lk))
                    .unwrap();
                let t = time(&ctx, |c| {
                    k.decode_attention(c, Arg::buf(&q), kc, vc, &part, Arg::buf(&o), shape)
                });
                report(
                    &format!("decode_attention kv={kvn} T=1 ctx={lk}"),
                    t,
                    kv_bytes,
                    flops,
                );
                let t = time(&ctx, |c| {
                    k.decode_attention_lanes(c, Arg::buf(&q), kc, vc, &part, Arg::buf(&o), shape)
                });
                report(
                    &format!("decode_attention_lanes kv={kvn} T=1 ctx={lk}"),
                    t,
                    kv_bytes,
                    flops,
                );
            }
        }
    }
}
