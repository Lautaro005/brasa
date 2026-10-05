//! Kernels fusionados de decode (T3.5): mismos bits que la secuencia de kernels que reemplazan
//! (cada uno de esos tiene su test contra la referencia CPU).

use brasa_kernels::testutil::Rng;
use brasa_kernels::{AttnShape, Kernels, KvType, RopeTable};
use brasa_metal::{Arg, Context};

#[test]
fn qk_norm_rope_store_igual_a_la_secuencia() {
    let ctx = Context::new().unwrap();
    let k = Kernels::new(&ctx).unwrap();
    let (hq, hkv, d, eps) = (32usize, 8usize, 128usize, 1e-6f32);
    let rope = RopeTable::new(&ctx, 1e6, d, 4096).unwrap();
    let mut rng = Rng::new(31);
    for kv in [KvType::F32, KvType::F16, KvType::Q8_0] {
        for (tokens, pos0) in [(1usize, 0usize), (1, 2999), (5, 37), (64, 128)] {
            let (q0, k0, v0) = (
                rng.vec(tokens * hq * d, 3.0),
                rng.vec(tokens * hkv * d, 3.0),
                rng.vec(tokens * hkv * d, 3.0),
            );
            let qn = ctx.buffer_from(&rng.vec(d, 1.5)).unwrap();
            let kn = ctx.buffer_from(&rng.vec(d, 1.5)).unwrap();
            let shape = AttnShape {
                tokens,
                hq,
                hkv,
                dim: d,
                pos0,
                kv,
            };
            let cache_bytes = kv.bytes(tokens * hkv * d);
            let run = |fused: bool| {
                let (mut q, mut kb) =
                    (ctx.buffer_from(&q0).unwrap(), ctx.buffer_from(&k0).unwrap());
                let v = ctx.buffer_from(&v0).unwrap();
                let mut kc = ctx.buffer::<u8>(cache_bytes).unwrap();
                let mut vc = ctx.buffer::<u8>(cache_bytes).unwrap();
                let mut cmd = ctx.command().unwrap();
                if fused {
                    k.qk_norm_rope_store(
                        &mut cmd,
                        kv,
                        [Arg::buf(&q), Arg::buf(&kb), Arg::buf(&v)],
                        [Arg::buf(&qn), Arg::buf(&kn)],
                        eps,
                        &rope,
                        [Arg::buf(&kc), Arg::buf(&vc)],
                        shape,
                    );
                } else {
                    let n = tokens * hkv * d;
                    k.rms_norm(
                        &mut cmd,
                        Arg::buf(&q),
                        Arg::buf(&qn),
                        Arg::buf(&q),
                        tokens * hq,
                        d,
                        eps,
                    );
                    k.rms_norm(
                        &mut cmd,
                        Arg::buf(&kb),
                        Arg::buf(&kn),
                        Arg::buf(&kb),
                        tokens * hkv,
                        d,
                        eps,
                    );
                    k.rope_neox(&mut cmd, Arg::buf(&q), &rope, tokens, hq, d, pos0);
                    k.rope_neox(&mut cmd, Arg::buf(&kb), &rope, tokens, hkv, d, pos0);
                    k.store_kv(&mut cmd, kv, Arg::buf(&kb), Arg::buf(&kc), n);
                    k.store_kv(&mut cmd, kv, Arg::buf(&v), Arg::buf(&vc), n);
                }
                cmd.commit_and_wait().unwrap();
                let bits = |x: &mut brasa_metal::Buffer<f32>| {
                    x.as_mut_slice()
                        .iter()
                        .map(|f| f.to_bits())
                        .collect::<Vec<u32>>()
                };
                (
                    bits(&mut q),
                    bits(&mut kb),
                    kc.as_mut_slice().to_vec(),
                    vc.as_mut_slice().to_vec(),
                )
            };
            let (a, b) = (run(false), run(true));
            let name = format!("KV {} T={tokens} pos0={pos0}", kv.name());
            assert!(a.0 == b.0, "{name}: q distinto");
            assert!(a.1 == b.1, "{name}: k distinto");
            assert!(a.2 == b.2, "{name}: caché K distinta");
            assert!(a.3 == b.3, "{name}: caché V distinta");
        }
    }
}

#[test]
fn add_norm_prep_y_gemv_scaled_igual_a_la_referencia() {
    // y = W · RMSNorm(x + h) con add_norm_prep + gemv_scaled, contra la referencia CPU de la suma,
    // RMSNorm y matmul (tolerancia de los GEMV). También verifica que x quede sumado.
    use brasa_kernels::{QMatrix, WeightType, reference};
    use brasa_quant::QType;
    let ctx = Context::new().unwrap();
    let k = Kernels::new(&ctx).unwrap();
    let eps = 1e-6f32;
    let mut worst = 0f32;
    for (qtype, rows, cols, add) in [
        (WeightType::Q4_0, 4096, 2560, true),
        (WeightType::Q4_0, 9728, 2560, false),
        (WeightType::Q4_0, 2560, 9728, true),
        (WeightType::Q6_0, 8192, 2560, true),
    ] {
        let mut rng = Rng::new((rows + cols) as u64);
        let (w, q) = match qtype {
            WeightType::Q4_0 => (rng.q4_0(rows, cols), QType::Q4_0),
            WeightType::Q6_0 => (rng.q6_0(rows, cols), QType::Q6_0),
            WeightType::Q8_0 => unreachable!(),
        };
        let x0 = rng.vec(cols, 40.0);
        let hv = rng.vec(cols, 10.0);
        let nw: Vec<f32> = rng.vec(cols, 1.0).iter().map(|v| 1.0 + v.abs()).collect();
        let xs: Vec<f32> = if add {
            x0.iter().zip(&hv).map(|(a, b)| a + b).collect()
        } else {
            x0.clone()
        };
        let mut hn = vec![0.0; cols];
        reference::rms_norm(&xs, &nw, cols, eps, &mut hn);
        let mut expected = vec![0.0; rows];
        let abs_sum = reference::matmul(q, &w, rows, cols, &hn, &mut expected);

        let gw = ctx.buffer_from(&w).unwrap();
        let mut gx = ctx.buffer_from(&x0).unwrap();
        let gh = ctx.buffer_from(&hv).unwrap();
        let gn = ctx.buffer_from(&nw).unwrap();
        let xw = ctx.buffer::<f32>(cols).unwrap();
        let ss = ctx
            .buffer::<f32>(brasa_kernels::norm_partials(cols))
            .unwrap();
        let mut y = ctx.buffer::<f32>(rows).unwrap();
        let m = QMatrix {
            data: &gw,
            qtype,
            rows,
            cols,
        };
        let mut cmd = ctx.command().unwrap();
        k.add_norm_prep(
            &mut cmd,
            Arg::buf(&gx),
            add.then(|| Arg::buf(&gh)),
            Arg::buf(&gn),
            Arg::buf(&xw),
            Arg::buf(&ss),
            cols,
        );
        k.gemv_scaled(&mut cmd, m, Arg::buf(&xw), Arg::buf(&ss), eps, Arg::buf(&y));
        cmd.commit_and_wait().unwrap();
        assert_eq!(gx.as_mut_slice(), &xs[..], "x += h exacto");
        let e = y
            .as_mut_slice()
            .iter()
            .zip(&expected)
            .zip(&abs_sum)
            .map(|((g, e), a)| (g - e).abs() / a.max(1e-30))
            .fold(0.0, f32::max);
        worst = worst.max(e);
    }
    eprintln!("add_norm_prep + gemv_scaled: error máximo / Σ|w·h| {worst:.2e}");
    assert!(worst <= 1e-5, "{worst}");
}

#[test]
fn gemv_scaled3_y_swiglu_igual_a_las_llamadas_separadas() {
    // Bit a bit contra gemv_scaled ×3 y gemv_scaled ×2 + swiglu (formas de Qwen3-4B).
    use brasa_kernels::{QMatrix, WeightType};
    let ctx = Context::new().unwrap();
    let k = Kernels::new(&ctx).unwrap();
    let mut rng = Rng::new(41);
    let (h, ffn, eps) = (2560usize, 9728usize, 1e-6f32);
    let mats: Vec<(usize, Vec<u8>)> = [4096, 1024, 1024, ffn, ffn]
        .iter()
        .map(|&r| (r, rng.q4_0(r, h)))
        .collect();
    let bufs: Vec<_> = mats
        .iter()
        .map(|(_, w)| ctx.buffer_from(w).unwrap())
        .collect();
    let m = |i: usize| QMatrix {
        data: &bufs[i],
        qtype: WeightType::Q4_0,
        rows: mats[i].0,
        cols: h,
    };
    let x = ctx.buffer_from(&rng.vec(h, 30.0)).unwrap();
    let nw = ctx.buffer_from(&rng.vec(h, 1.0)).unwrap();
    let xw = ctx.buffer::<f32>(h).unwrap();
    let ss = ctx.buffer::<f32>(brasa_kernels::norm_partials(h)).unwrap();
    let out = |n: usize| ctx.buffer::<f32>(n).unwrap();
    let (mut a, mut b): (Vec<_>, Vec<_>) = (
        [4096, 1024, 1024, ffn, ffn, ffn]
            .iter()
            .map(|&n| out(n))
            .collect(),
        [4096, 1024, 1024, ffn].iter().map(|&n| out(n)).collect(),
    );
    let mut cmd = ctx.command().unwrap();
    k.add_norm_prep(
        &mut cmd,
        Arg::buf(&x),
        None,
        Arg::buf(&nw),
        Arg::buf(&xw),
        Arg::buf(&ss),
        h,
    );
    for i in 0..3 {
        k.gemv_scaled(
            &mut cmd,
            m(i),
            Arg::buf(&xw),
            Arg::buf(&ss),
            eps,
            Arg::buf(&a[i]),
        );
    }
    k.gemv_scaled(
        &mut cmd,
        m(3),
        Arg::buf(&xw),
        Arg::buf(&ss),
        eps,
        Arg::buf(&a[3]),
    );
    k.gemv_scaled(
        &mut cmd,
        m(4),
        Arg::buf(&xw),
        Arg::buf(&ss),
        eps,
        Arg::buf(&a[4]),
    );
    k.swiglu(&mut cmd, &a[3], &a[4], &a[5], ffn);
    k.gemv_scaled3(
        &mut cmd,
        [m(0), m(1), m(2)],
        Arg::buf(&xw),
        Arg::buf(&ss),
        eps,
        [Arg::buf(&b[0]), Arg::buf(&b[1]), Arg::buf(&b[2])],
    );
    k.gemv_scaled_swiglu(
        &mut cmd,
        m(3),
        m(4),
        Arg::buf(&xw),
        Arg::buf(&ss),
        eps,
        Arg::buf(&b[3]),
    );
    cmd.commit_and_wait().unwrap();
    let bits = |v: &mut brasa_metal::Buffer<f32>| {
        v.as_mut_slice()
            .iter()
            .map(|f| f.to_bits())
            .collect::<Vec<_>>()
    };
    for i in 0..3 {
        assert!(
            bits(&mut a[i]) == bits(&mut b[i]),
            "gemv_scaled3: salida {i} distinta"
        );
    }
    assert!(
        bits(&mut a[5]) == bits(&mut b[3]),
        "gemv_scaled_swiglu distinto"
    );
}
