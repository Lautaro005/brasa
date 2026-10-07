//! Parámetros de lanzamiento de decode (ADR 0029): con cualquier cantidad de simdgroups por
//! threadgroup, los GEMV de decode y `attn_decode_lanes` dan los mismos bits que con los valores
//! por defecto (cada uno de esos kernels tiene además su test contra la referencia CPU).

use brasa_kernels::launch::SG_CANDIDATES;
use brasa_kernels::testutil::{KvPair, Rng};
use brasa_kernels::{AttnShape, GemvOp, Kernels, KvType, Launch, QMatrix, WeightType};
use brasa_metal::{Arg, Buffer, Context};

fn bits(b: &mut Buffer<f32>, n: usize) -> Vec<u32> {
    b.as_mut_slice()[..n].iter().map(|x| x.to_bits()).collect()
}

#[test]
fn gemv_de_decode_iguales_con_cualquier_lanzamiento() {
    let ctx = Context::new().unwrap();
    let mut k = Kernels::new(&ctx).unwrap();
    let mut rng = Rng::new(77);
    // Formas chicas con las mismas restricciones que las de Qwen3-4B (filas múltiplo de 32).
    let (rows, cols) = (96usize, 512usize);
    let (kr, vr) = (32usize, 32usize);
    let q4 = ctx.buffer_from(&rng.q4_0(rows, cols)).unwrap();
    let q4b = ctx.buffer_from(&rng.q4_0(rows, cols)).unwrap();
    let q4k = ctx.buffer_from(&rng.q4_0(kr, cols)).unwrap();
    let q4v = ctx.buffer_from(&rng.q4_0(vr, cols)).unwrap();
    let q8 = ctx.buffer_from(&rng.q8_0(rows, cols)).unwrap();
    let q6 = ctx.buffer_from(&rng.q6_0(rows, cols)).unwrap();
    let x = ctx.buffer_from(&rng.vec(2 * cols, 1.0)).unwrap();
    let ssv: Vec<f32> = rng
        .vec(cols.div_ceil(256), 1.0)
        .iter()
        .map(|v| v.abs() * 100.0)
        .collect();
    let ss = ctx.buffer_from(&ssv).unwrap();
    let m = |data, qtype, rows| QMatrix {
        data,
        qtype,
        rows,
        cols,
    };
    let ops: [(GemvOp, usize); 7] = [
        (GemvOp::Fast(WeightType::Q4_0), rows),
        (GemvOp::Fast(WeightType::Q8_0), rows),
        (GemvOp::Fast(WeightType::Q6_0), rows),
        (GemvOp::Scaled(WeightType::Q4_0), rows),
        (GemvOp::Scaled(WeightType::Q6_0), rows),
        (GemvOp::Scaled3, rows + kr + vr),
        (GemvOp::ScaledSwiglu, rows),
    ];
    for (op, op_rows) in ops {
        let run = |k: &Kernels| {
            let mut y = ctx.buffer::<f32>(2 * rows).unwrap();
            let mut yk = ctx.buffer::<f32>(kr).unwrap();
            let mut yv = ctx.buffer::<f32>(vr).unwrap();
            let mut cmd = ctx.command().unwrap();
            match op {
                GemvOp::Fast(t) => {
                    let w = match t {
                        WeightType::Q4_0 => &q4,
                        WeightType::Q8_0 => &q8,
                        WeightType::Q6_0 => &q6,
                    };
                    k.gemv(&mut cmd, m(w, t, rows), Arg::buf(&x), Arg::buf(&y), 2);
                }
                GemvOp::Scaled(t) => {
                    let w = if t == WeightType::Q4_0 { &q4 } else { &q6 };
                    k.gemv_scaled(
                        &mut cmd,
                        m(w, t, rows),
                        Arg::buf(&x),
                        Arg::buf(&ss),
                        1e-6,
                        Arg::buf(&y),
                    );
                }
                GemvOp::Scaled3 => k.gemv_scaled3(
                    &mut cmd,
                    [
                        m(&q4, WeightType::Q4_0, rows),
                        m(&q4k, WeightType::Q4_0, kr),
                        m(&q4v, WeightType::Q4_0, vr),
                    ],
                    Arg::buf(&x),
                    Arg::buf(&ss),
                    1e-6,
                    [Arg::buf(&y), Arg::buf(&yk), Arg::buf(&yv)],
                ),
                GemvOp::ScaledSwiglu => k.gemv_scaled_swiglu(
                    &mut cmd,
                    m(&q4, WeightType::Q4_0, rows),
                    m(&q4b, WeightType::Q4_0, rows),
                    Arg::buf(&x),
                    Arg::buf(&ss),
                    1e-6,
                    Arg::buf(&y),
                ),
            }
            cmd.commit_and_wait().unwrap();
            let n = if matches!(op, GemvOp::Fast(_)) {
                2 * rows
            } else {
                rows
            };
            let mut out = bits(&mut y, n);
            out.extend(bits(&mut yk, kr));
            out.extend(bits(&mut yv, vr));
            out
        };
        k.set_launch(&ctx, Launch::default()).unwrap();
        let want = run(&k);
        for sg in SG_CANDIDATES {
            let mut l = Launch::default();
            l.set_gemv(op, op_rows, cols, sg).unwrap();
            k.set_launch(&ctx, l).unwrap();
            assert_eq!(run(&k), want, "{} con sg {sg}", op.kernel_name());
        }
    }
}

#[test]
fn attn_decode_lanes_igual_con_cualquier_lanzamiento() {
    let ctx = Context::new().unwrap();
    let mut k = Kernels::new(&ctx).unwrap();
    let mut rng = Rng::new(78);
    let (hq, hkv, dim) = (32usize, 8usize, 128usize);
    for kv in [KvType::F32, KvType::F16, KvType::Q8_0] {
        for lk in [1usize, 37, 300, 1025] {
            let n = lk.next_multiple_of(128) * hkv * dim;
            let cache = KvPair::new(&ctx, kv, rng.vec(n, 2.0), rng.vec(n, 3.0));
            let (kc, vc) = cache.args();
            let q = ctx.buffer_from(&rng.vec(hq * dim, 2.0)).unwrap();
            let part = ctx
                .buffer::<f32>(brasa_kernels::decode_partials_len(hq, lk))
                .unwrap();
            let shape = AttnShape {
                tokens: 1,
                hq,
                hkv,
                dim,
                pos0: lk - 1,
                kv,
            };
            let run = |k: &Kernels| {
                let mut o = ctx.buffer::<f32>(hq * dim).unwrap();
                let mut cmd = ctx.command().unwrap();
                k.decode_attention_lanes(
                    &mut cmd,
                    Arg::buf(&q),
                    kc,
                    vc,
                    &part,
                    Arg::buf(&o),
                    shape,
                );
                cmd.commit_and_wait().unwrap();
                bits(&mut o, hq * dim)
            };
            k.set_launch(&ctx, Launch::default()).unwrap();
            let want = run(&k);
            for sg in SG_CANDIDATES {
                let mut l = Launch::default();
                l.set_attn_lanes(kv, hq / hkv, lk, sg).unwrap();
                k.set_launch(&ctx, l).unwrap();
                assert_eq!(run(&k), want, "KV {} lk {lk} sg {sg}", kv.name());
            }
        }
    }
}
