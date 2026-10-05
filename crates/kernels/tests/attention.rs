//! Equivalencia GPU vs CPU de la atención causal con GQA (tolerancia en `brasa_kernels`).
//! Formas de Qwen3-4B: 32 cabezas de query, 8 de KV, head_dim 128.

use brasa_kernels::testutil::Rng;
use brasa_kernels::{AttnShape, Kernels, reference};
use brasa_metal::{Arg, Context};

#[test]
fn atencion_gqa_causal() {
    let ctx = Context::new().unwrap();
    let k = Kernels::new(&ctx).unwrap();
    let mut rng = Rng::new(21);
    let (hq, hkv, dim) = (32, 8, 128);
    let mut worst = 0f32;
    // Prefill desde 0, prefill continuado (prefijo en caché) y decode con contexto largo.
    for (tokens, pos0) in [(9, 0), (5, 11), (1, 0), (1, 1500)] {
        let lk = pos0 + tokens;
        let q = rng.vec(tokens * hq * dim, 2.0);
        let kc = rng.vec(lk * hkv * dim, 2.0);
        let vc = rng.vec(lk * hkv * dim, 3.0);
        let (expected, vmax) = reference::attention(&q, &kc, &vc, tokens, hq, hkv, dim, pos0);
        let gq = ctx.buffer_from(&q).unwrap();
        let gk = ctx.buffer_from(&kc).unwrap();
        let gv = ctx.buffer_from(&vc).unwrap();
        let scores = ctx.buffer::<f32>(tokens * hq * lk).unwrap();
        let mut o = ctx.buffer::<f32>(tokens * hq * dim).unwrap();
        let shape = AttnShape {
            tokens,
            hq,
            hkv,
            dim,
            pos0,
        };
        let mut cmd = ctx.command().unwrap();
        k.attention(
            &mut cmd,
            Arg::buf(&gq),
            Arg::buf(&gk),
            Arg::buf(&gv),
            &scores,
            Arg::buf(&o),
            shape,
        );
        cmd.commit_and_wait().unwrap();
        for ((g, e), m) in o.as_mut_slice().iter().zip(&expected).zip(&vmax) {
            worst = worst.max((g - e).abs() / m.max(1e-30));
        }
    }
    eprintln!("atención: error máximo / max|v| {worst:.2e}");
    assert!(worst <= 1e-5, "{worst}");
}
