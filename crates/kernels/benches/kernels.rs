//! Microbenchmarks de kernels: tiempo de GPU (mediana) y ancho de banda efectivo.
//! Uso: cargo bench -p brasa-kernels

use brasa_kernels::Kernels;
use brasa_kernels::testutil::{Rng, median};
use brasa_metal::Context;

const REPS: usize = 20;

fn main() {
    let ctx = Context::new().expect("contexto Metal");
    let k = Kernels::new(&ctx).expect("kernels");
    println!("dispositivo: {}", ctx.device_name());
    bench_add(&ctx, &k, 1 << 24);
}

fn bench_add(ctx: &Context, k: &Kernels, n: usize) {
    let mut rng = Rng::new(1);
    let a = ctx.buffer_from(&rng.vec(n, 1.0)).unwrap();
    let b = ctx.buffer_from(&rng.vec(n, 1.0)).unwrap();
    let out = ctx.buffer::<f32>(n).unwrap();
    let len = n as u32;
    let times: Vec<f64> = (0..REPS + 2)
        .map(|_| {
            let mut cmd = ctx.command().unwrap();
            k.add(&mut cmd, &a, &b, &out, &len);
            cmd.commit_and_wait().unwrap().gpu_seconds
        })
        .skip(2) // calentamiento
        .collect();
    let t = median(times);
    let bytes = 3.0 * n as f64 * 4.0;
    println!(
        "add_f32      n={n:>9}  {:>8.3} ms  {:>6.1} GB/s",
        t * 1e3,
        bytes / t / 1e9
    );
}
