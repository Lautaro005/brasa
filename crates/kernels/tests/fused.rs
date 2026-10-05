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
