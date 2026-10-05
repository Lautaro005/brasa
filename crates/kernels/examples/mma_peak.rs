//! Techo de `simdgroup_multiply_accumulate` en el chip: un kernel que solo encadena productos
//! 8×8×8 en registros (sin memoria), en f32 y en f16 con acumulador f32. Sirve para saber cuánto
//! queda por ganar en un GEMM o en la atención antes de optimizarlos:
//!   cargo run --release -p brasa-kernels --example mma_peak
use brasa_metal::{Arg, Context};

const SRC: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;
kernel void peak(device float* out [[buffer(0)]], constant uint& iters [[buffer(1)]],
                 uint gid [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]]) {
    simdgroup_matrix<IN_T, 8, 8> a = make_filled_simdgroup_matrix<IN_T, 8, 8>(IN_T(1e-3));
    simdgroup_matrix<IN_T, 8, 8> b = make_filled_simdgroup_matrix<IN_T, 8, 8>(IN_T(1e-3));
    simdgroup_float8x8 c[NACC];
    for (uint i = 0; i < NACC; ++i) c[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint it = 0; it < iters; ++it)
        for (uint i = 0; i < NACC; ++i) simdgroup_multiply_accumulate(c[i], a, b, c[i]);
    float s = 0;
    for (uint i = 0; i < NACC; ++i) s += c[i].thread_elements()[0];
    if (s == 12345.0f) out[gid] = s;
}
"#;

fn main() {
    let ctx = Context::new().unwrap();
    let out = ctx.buffer::<f32>(1 << 16).unwrap();
    let (groups, iters) = (4096usize, 2048u32);
    println!("dispositivo: {}", ctx.device_name());
    for ty in ["float", "half"] {
        for nacc in [4usize, 8, 16] {
            let p = ctx
                .pipeline(
                    &format!("#define IN_T {ty}\n#define NACC {nacc}\n{SRC}"),
                    "peak",
                )
                .unwrap();
            let mut best = f64::MAX;
            for _ in 0..5 {
                let mut cmd = ctx.command().unwrap();
                cmd.dispatch_groups(
                    &p,
                    &[Arg::buf(&out), Arg::u32(iters)],
                    [groups, 1, 1],
                    [128, 1, 1],
                );
                best = best.min(cmd.commit_and_wait().unwrap().gpu_seconds);
            }
            let flops = 2.0 * 512.0 * (groups * 4 * nacc) as f64 * iters as f64;
            println!(
                "entrada {ty:<5} {nacc:>2} acumuladores: {:>7.0} GFLOP/s",
                flops / best / 1e9
            );
        }
    }
}
