//! Equivalencia GPU vs CPU de kernels elemento a elemento.
//!
//! Tolerancia de `add_f32`: 0 ULP. Una suma f32 es exacta y determinista en ambos lados
//! (math mode Safe, sin fusión).

use brasa_kernels::testutil::{Rng, ulps};
use brasa_kernels::{Kernels, reference};
use brasa_metal::Context;

#[test]
fn add_f32_igual_a_cpu() {
    let ctx = Context::new().unwrap();
    let k = Kernels::new(&ctx).unwrap();
    let mut rng = Rng::new(7);
    // Incluye tamaños que no son múltiplo del threadgroup y magnitudes muy distintas.
    for &n in &[1usize, 7, 255, 257, 1000, (1 << 20) + 3] {
        let a = rng.vec(n, 1e3);
        let b: Vec<f32> = rng.vec(n, 1e-3);
        let mut expected = vec![0.0; n];
        reference::add(&a, &b, &mut expected);

        let ga = ctx.buffer_from(&a).unwrap();
        let gb = ctx.buffer_from(&b).unwrap();
        let mut out = ctx.buffer::<f32>(n).unwrap();
        let len = n as u32;
        let mut cmd = ctx.command().unwrap();
        k.add(&mut cmd, &ga, &gb, &out, &len);
        cmd.commit_and_wait().unwrap();

        let got = out.as_mut_slice();
        let worst = got
            .iter()
            .zip(&expected)
            .map(|(g, e)| ulps(*g, *e))
            .max()
            .unwrap();
        assert_eq!(worst, 0, "n = {n}: {worst} ULP");
    }
}
