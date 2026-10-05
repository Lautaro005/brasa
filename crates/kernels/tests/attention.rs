//! Equivalencia GPU vs CPU de la atención causal con GQA (tolerancia en `brasa_kernels`).
//! Formas de Qwen3-4B: 32 cabezas de query, 8 de KV, head_dim 128. Cada kernel con KV f32 y f16
//! (ADR 0009: en f16 la referencia recibe K y V ya redondeados).

use brasa_kernels::testutil::{KvPair, Rng};
use brasa_kernels::{AttnShape, Kernels, KvType, reference};
use brasa_metal::{Arg, Context};
use brasa_quant::{f16_to_f32, f32_to_f16};

const KVS: [KvType; 2] = [KvType::F32, KvType::F16];

/// Error máximo `|gpu - ref| / max|v|` entre la salida de la GPU y la referencia.
fn worst(got: &[f32], expected: &[f32], vmax: &[f32]) -> f32 {
    got.iter()
        .zip(expected)
        .zip(vmax)
        .map(|((g, e), m)| (g - e).abs() / m.max(1e-30))
        .fold(0f32, f32::max)
}

#[test]
fn flash_attention_gqa_causal() {
    let ctx = Context::new().unwrap();
    let k = Kernels::new(&ctx).unwrap();
    let (hq, hkv, dim) = (32, 8, 128);
    for kv in KVS {
        let mut rng = Rng::new(22);
        let mut w = 0f32;
        // Prefill desde 0 (bloques parciales y completos), prefill continuado y decode.
        for (tokens, pos0) in [
            (9usize, 0usize),
            (5, 11),
            (1, 0),
            (1, 1500),
            (33, 40),
            (64, 448),
            (32, 0),
            (100, 7),
        ] {
            let lk = pos0 + tokens;
            // Capacidad redondeada a KV_ALIGN, con basura finita fuera de rango.
            let cap = lk.next_multiple_of(brasa_kernels::KV_ALIGN);
            let q = rng.vec(tokens * hq * dim, 2.0);
            let cache = KvPair::new(
                &ctx,
                kv,
                rng.vec(cap * hkv * dim, 2.0),
                rng.vec(cap * hkv * dim, 3.0),
            );
            let (expected, vmax) =
                reference::attention(&q, &cache.k, &cache.v, tokens, hq, hkv, dim, pos0);
            let gq = ctx.buffer_from(&q).unwrap();
            let mut o = ctx.buffer::<f32>(tokens * hq * dim).unwrap();
            let shape = AttnShape {
                tokens,
                hq,
                hkv,
                dim,
                pos0,
                kv,
            };
            let (gk, gv) = cache.args();
            let mut cmd = ctx.command().unwrap();
            k.flash_attention(&mut cmd, Arg::buf(&gq), gk, gv, Arg::buf(&o), shape);
            cmd.commit_and_wait().unwrap();
            w = w.max(worst(o.as_mut_slice(), &expected, &vmax));
        }
        eprintln!(
            "flash attention KV {}: error máximo / max|v| {w:.2e}",
            kv.name()
        );
        assert!(w <= 1e-5, "{}: {w}", kv.name());
    }
}

fn decode_case(variant: u8) {
    let ctx = Context::new().unwrap();
    let k = Kernels::new(&ctx).unwrap();
    let (hq, hkv, dim) = (32, 8, 128);
    for kv in KVS {
        let mut rng = Rng::new(23);
        let mut w = 0f32;
        // Contextos con tramos parciales, exactos y muchos tramos.
        for pos0 in [0usize, 1, 255, 256, 700, 4095, 16383] {
            let lk = pos0 + 1;
            let q = rng.vec(hq * dim, 2.0);
            let cache = KvPair::new(
                &ctx,
                kv,
                rng.vec(lk * hkv * dim, 2.0),
                rng.vec(lk * hkv * dim, 3.0),
            );
            let (expected, vmax) =
                reference::attention(&q, &cache.k, &cache.v, 1, hq, hkv, dim, pos0);
            let gq = ctx.buffer_from(&q).unwrap();
            let part = ctx
                .buffer::<f32>(brasa_kernels::decode_partials_len(hq, lk))
                .unwrap();
            let mut o = ctx.buffer::<f32>(hq * dim).unwrap();
            let shape = AttnShape {
                tokens: 1,
                hq,
                hkv,
                dim,
                pos0,
                kv,
            };
            let (gk, gv) = cache.args();
            let mut cmd = ctx.command().unwrap();
            match variant {
                0 => {
                    k.decode_attention(&mut cmd, Arg::buf(&gq), gk, gv, &part, Arg::buf(&o), shape)
                }
                _ => k.decode_attention_lanes(
                    &mut cmd,
                    Arg::buf(&gq),
                    gk,
                    gv,
                    &part,
                    Arg::buf(&o),
                    shape,
                ),
            }
            cmd.commit_and_wait().unwrap();
            w = w.max(worst(o.as_mut_slice(), &expected, &vmax));
        }
        let name = ["decode attention", "decode attention lanes"][variant as usize];
        eprintln!("{name} KV {}: error máximo / max|v| {w:.2e}", kv.name());
        assert!(w <= 1e-5, "{}: {w}", kv.name());
    }
}

#[test]
fn decode_attention_por_cabeza() {
    decode_case(0);
}

#[test]
fn decode_attention_lanes() {
    decode_case(1);
}

#[test]
fn atencion_gqa_causal() {
    let ctx = Context::new().unwrap();
    let k = Kernels::new(&ctx).unwrap();
    let mut rng = Rng::new(21);
    let (hq, hkv, dim) = (32, 8, 128);
    let mut w = 0f32;
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
            kv: KvType::F32,
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
        w = w.max(worst(o.as_mut_slice(), &expected, &vmax));
    }
    eprintln!("atención: error máximo / max|v| {w:.2e}");
    assert!(w <= 1e-5, "{w}");
}

#[test]
fn store_kv_convierte() {
    let ctx = Context::new().unwrap();
    let k = Kernels::new(&ctx).unwrap();
    let mut rng = Rng::new(24);
    let mut x = rng.vec(4099, 50.0);
    // Casos borde: medios exactos entre f16 (van al par), desborde, subnormales y ceros.
    let mid = (f16_to_f32(0x3c01) + f16_to_f32(0x3c02)) / 2.0;
    x[..8].copy_from_slice(&[mid, -mid, 65520.0, 1e-7, 3e-8, 0.0, -0.0, 70000.0]);
    let src = ctx.buffer_from(&x).unwrap();
    let mut d32 = ctx.buffer::<f32>(x.len()).unwrap();
    let mut d16 = ctx.buffer::<u16>(x.len()).unwrap();
    let mut cmd = ctx.command().unwrap();
    k.store_kv(
        &mut cmd,
        KvType::F32,
        Arg::buf(&src),
        Arg::buf(&d32),
        x.len(),
    );
    k.store_kv(
        &mut cmd,
        KvType::F16,
        Arg::buf(&src),
        Arg::buf(&d16),
        x.len(),
    );
    cmd.commit_and_wait().unwrap();
    assert_eq!(d32.as_mut_slice(), &x[..]);
    let want: Vec<u16> = x.iter().map(|&f| f32_to_f16(f)).collect();
    assert_eq!(d16.as_mut_slice(), &want[..]);
}

#[test]
fn decode_attention_lanes_otros_grupos() {
    // Grupos GQA 1, 2 y 8 (Qwen3-4B usa 4, cubierto arriba).
    let ctx = Context::new().unwrap();
    let k = Kernels::new(&ctx).unwrap();
    let mut rng = Rng::new(25);
    let (hkv, dim) = (8, 128);
    for hq in [8usize, 16, 64] {
        assert!(brasa_kernels::gqa_supported(hq, hkv));
        for kv in KVS {
            let mut w = 0f32;
            for pos0 in [0usize, 300, 4095] {
                let lk = pos0 + 1;
                let q = rng.vec(hq * dim, 2.0);
                let cache = KvPair::new(
                    &ctx,
                    kv,
                    rng.vec(lk * hkv * dim, 2.0),
                    rng.vec(lk * hkv * dim, 3.0),
                );
                let (expected, vmax) =
                    reference::attention(&q, &cache.k, &cache.v, 1, hq, hkv, dim, pos0);
                let gq = ctx.buffer_from(&q).unwrap();
                let part = ctx
                    .buffer::<f32>(brasa_kernels::decode_partials_len(hq, lk))
                    .unwrap();
                let mut o = ctx.buffer::<f32>(hq * dim).unwrap();
                let shape = AttnShape {
                    tokens: 1,
                    hq,
                    hkv,
                    dim,
                    pos0,
                    kv,
                };
                let (gk, gv) = cache.args();
                let mut cmd = ctx.command().unwrap();
                k.decode_attention_lanes(
                    &mut cmd,
                    Arg::buf(&gq),
                    gk,
                    gv,
                    &part,
                    Arg::buf(&o),
                    shape,
                );
                cmd.commit_and_wait().unwrap();
                w = w.max(worst(o.as_mut_slice(), &expected, &vmax));
            }
            eprintln!(
                "decode attention lanes grupo {} KV {}: {w:.2e}",
                hq / hkv,
                kv.name()
            );
            assert!(w <= 1e-5, "grupo {} {}: {w}", hq / hkv, kv.name());
        }
    }
    assert!(!brasa_kernels::gqa_supported(24, 8));
}
