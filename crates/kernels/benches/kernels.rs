//! Microbenchmarks de kernels con las formas de Qwen3-4B: tiempo de GPU (mediana) y ancho de
//! banda o GFLOP/s efectivos. Uso: cargo bench -p brasa-kernels
//! Las cifras son de kernels sin optimizar (fase 1); sirven de línea base para la fase 3.

use brasa_kernels::testutil::{Rng, median};
use brasa_kernels::{Kernels, QMatrix, RopeTable, WeightType};
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
    let k = Kernels::new(&ctx).expect("kernels");
    let mut rng = Rng::new(1);
    println!("dispositivo: {}", ctx.device_name());

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
        let t = time(&ctx, |c| k.gemv(c, m, Arg::buf(&x), Arg::buf(&out), 1));
        report(
            &format!("gemv_q4_0 {name} {rows}×{cols} T=1"),
            t,
            wbytes,
            2.0 * (rows * cols) as f64,
        );
        let t = time(&ctx, |c| k.gemm(c, m, Arg::buf(&x), Arg::buf(&out), 512));
        report(
            &format!("gemm_q4_0 {name} {rows}×{cols} T=512"),
            t,
            wbytes,
            2.0 * 512.0 * (rows * cols) as f64,
        );
    }
    let t = time(&ctx, |c| k.gemv(c, emb_m, Arg::buf(&x), Arg::buf(&out), 1));
    report(
        "gemv_q8_0 lm_head 151936×2560 T=1",
        t,
        (vocab * h) as f64 * 34.0 / 32.0,
        2.0 * (vocab * h) as f64,
    );
}
