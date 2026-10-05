//! Equivalencia GPU vs CPU de kernels elemento a elemento (tolerancias en `brasa_kernels`).

use brasa_kernels::testutil::{Rng, max_rel, ulps};
use brasa_kernels::{Kernels, reference};
use brasa_metal::Context;

const SIZES: [usize; 6] = [1, 7, 255, 257, 1000, (1 << 20) + 3];

#[test]
fn add_f32_exacto() {
    let ctx = Context::new().unwrap();
    let k = Kernels::new(&ctx).unwrap();
    let mut rng = Rng::new(7);
    for n in SIZES {
        let a = rng.vec(n, 1e3);
        let b = rng.vec(n, 1e-3);
        let mut expected = vec![0.0; n];
        reference::add(&a, &b, &mut expected);
        let (ga, gb) = (ctx.buffer_from(&a).unwrap(), ctx.buffer_from(&b).unwrap());
        let mut out = ctx.buffer::<f32>(n).unwrap();
        let mut cmd = ctx.command().unwrap();
        k.add(&mut cmd, &ga, &gb, &out, n);
        cmd.commit_and_wait().unwrap();
        let worst = out
            .as_mut_slice()
            .iter()
            .zip(&expected)
            .map(|(g, e)| ulps(*g, *e))
            .max()
            .unwrap();
        assert_eq!(worst, 0, "n = {n}: {worst} ULP");
    }
}

#[test]
fn swiglu_f32() {
    let ctx = Context::new().unwrap();
    let k = Kernels::new(&ctx).unwrap();
    let mut rng = Rng::new(8);
    let mut worst = 0f32;
    for n in SIZES {
        let g = rng.vec(n, 20.0);
        let u = rng.vec(n, 4.0);
        let mut expected = vec![0.0; n];
        reference::swiglu(&g, &u, &mut expected);
        let (gg, gu) = (ctx.buffer_from(&g).unwrap(), ctx.buffer_from(&u).unwrap());
        let mut out = ctx.buffer::<f32>(n).unwrap();
        let mut cmd = ctx.command().unwrap();
        k.swiglu(&mut cmd, &gg, &gu, &out, n);
        cmd.commit_and_wait().unwrap();
        worst = worst.max(max_rel(out.as_mut_slice(), &expected, 1e-30));
    }
    eprintln!("swiglu: error relativo máximo {worst:.2e}");
    assert!(worst <= 1e-6, "{worst}");
}
