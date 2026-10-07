//! Laboratorio de GEMM de prefill: compila un kernel candidato desde un archivo .metal y lo mide
//! contra `gemm_tiled` con las formas de Qwen3-4B (T tokens), verificando el resultado.
//!   cargo run --release -p brasa-kernels --example gemm_lab -- <archivo.metal> <kernel> [T]
//! Variables: LAB_BM / LAB_BN (filas y tokens por threadgroup, 64 / 32), LAB_NT (hilos, 128),
//! LAB_ORDER=t (eje x de la grilla = tokens en lugar de filas).
use brasa_kernels::testutil::{Rng, median};
use brasa_kernels::{Kernels, QMatrix, WeightType};
use brasa_metal::{Arg, Command, Context};

fn time<'a>(ctx: &Context, mut encode: impl FnMut(&mut Command<'a>)) -> f64 {
    let t: Vec<f64> = (0..12)
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
    let src = std::fs::read_to_string(&a[1]).unwrap();
    let ctx = Context::new().unwrap();
    let k = Kernels::new(&ctx).unwrap();
    let p = ctx.pipeline(&src, &a[2]).unwrap();
    let tokens: usize = a.get(3).map_or(512, |s| s.parse().unwrap());
    let (bm, bn, nt) = (env("LAB_BM", 64), env("LAB_BN", 32), env("LAB_NT", 128));
    let tok_x = std::env::var("LAB_ORDER").is_ok_and(|v| v == "t");
    println!(
        "{}; {} max hilos/tg {}; T = {tokens}",
        ctx.device_name(),
        a[2],
        p.max_threads_per_threadgroup()
    );
    let mut rng = Rng::new(5);
    let x = ctx.buffer_from(&rng.vec(tokens * 9728, 1.0)).unwrap();
    for (name, rows, cols) in [
        ("q_proj", 4096usize, 2560usize),
        ("k_proj", 1024, 2560),
        ("gate", 9728, 2560),
        ("down", 2560, 9728),
    ] {
        let w = ctx.buffer_from(&rng.q4_0(rows, cols)).unwrap();
        let m = QMatrix {
            data: &w,
            qtype: WeightType::Q4_0,
            rows,
            cols,
        };
        let y0 = ctx.buffer::<f32>(tokens * rows).unwrap();
        let y1 = ctx.buffer::<f32>(tokens * rows).unwrap();
        let t0 = time(&ctx, |c| k.gemm(c, m, Arg::buf(&x), Arg::buf(&y0), tokens));
        let (gr, gt) = (rows.div_ceil(bm), tokens.div_ceil(bn));
        let grid = if tok_x { [gt, gr, 1] } else { [gr, gt, 1] };
        let t1 = time(&ctx, |c| {
            c.dispatch_groups(
                &p,
                &[
                    Arg::buf(&w),
                    Arg::buf(&x),
                    Arg::buf(&y1),
                    Arg::u32(rows as u32),
                    Arg::u32(cols as u32),
                    Arg::u32(tokens as u32),
                ],
                grid,
                [nt, 1, 1],
            )
        });
        let (r0, r1) = (y0.as_slice(), y1.as_slice());
        let scale = r0.iter().fold(0f32, |m, v| m.max(v.abs()));
        let err = r0
            .iter()
            .zip(r1)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        let fl = 2.0 * (rows * cols * tokens) as f64;
        println!(
            "{name:<7} {rows:>5}×{cols:<5} actual {:>7.1} GFLOP/s  candidato {:>7.1} GFLOP/s  ({:+.1} %)  err {:.2e} rel",
            fl / t0 / 1e9,
            fl / t1 / 1e9,
            (t0 / t1 - 1.0) * 100.0,
            err / scale
        );
    }
}
