//! Equivalencia GPU vs CPU de la atención causal con GQA (tolerancia en `brasa_kernels`).
//! Formas de Qwen3-4B: 32 cabezas de query, 8 de KV, head_dim 128. Cada kernel con KV f32, f16 y
//! Q8 (ADR 0009: la referencia recibe K y V ya redondeados al tipo de la caché).

use brasa_kernels::testutil::{KvPair, Rng};
use brasa_kernels::{AttnShape, Kernels, KvType, PrefillPrecision, reference};
use brasa_metal::{Arg, Context};
use brasa_quant::{f16_to_f32, f32_to_f16, quantize_kv_q8};

const KVS: [KvType; 3] = [KvType::F32, KvType::F16, KvType::Q8_0];

/// Como [`worst`], para una salida redondeada a f16: descuenta el redondeo final (a lo sumo
/// 2⁻¹¹ · |valor|) antes de comparar.
fn worst_f16(got: &[u16], expected: &[f32], vmax: &[f32]) -> f32 {
    got.iter()
        .zip(expected)
        .zip(vmax)
        .map(|((g, e), m)| {
            let d = (f16_to_f32(*g) - e).abs() - e.abs() * 2f32.powi(-11);
            d.max(0.0) / m.max(1e-30)
        })
        .fold(0f32, f32::max)
}

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
        let (mut w, mut w_raw, mut w32) = (0f32, 0f32, 0f32);
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
            // Q · escala y la salida se redondean a f16 en el kernel (ADR 0030): la tolerancia se
            // mide contra la referencia que redondea Q igual, descontando el redondeo de la
            // salida; contra la sin redondear solo se informa y se acota.
            let (expected, vmax) =
                reference::attention_q16(&q, &cache.k, &cache.v, tokens, hq, hkv, dim, pos0);
            let (raw, _) = reference::attention(&q, &cache.k, &cache.v, tokens, hq, hkv, dim, pos0);
            let gq = ctx.buffer_from(&q).unwrap();
            let mut o = ctx.buffer::<f32>(tokens * hq * dim).unwrap();
            let mut oh = ctx.buffer::<u16>(tokens * hq * dim).unwrap();
            let shape = AttnShape {
                tokens,
                hq,
                hkv,
                dim,
                pos0,
                kv,
            };
            let (gk, gv) = cache.args();
            // Ruta caliente: Q y la salida en f16.
            let mut cmd = ctx.command().unwrap();
            k.flash_attention(&mut cmd, Arg::buf(&gq), gk, gv, Arg::buf(&oh), shape);
            cmd.commit_and_wait().unwrap();
            w = w.max(worst_f16(oh.as_mut_slice(), &expected, &vmax));
            let oh32: Vec<f32> = oh.as_mut_slice().iter().map(|h| f16_to_f32(*h)).collect();
            w_raw = w_raw.max(worst(&oh32, &raw, &vmax));
            // Variante exacta (Q en f32) contra la referencia sin redondear.
            let mut cmd = ctx.command().unwrap();
            k.flash_attention_with(
                &mut cmd,
                Arg::buf(&gq),
                gk,
                gv,
                Arg::buf(&o),
                shape,
                PrefillPrecision::F32,
            );
            cmd.commit_and_wait().unwrap();
            w32 = w32.max(worst(o.as_mut_slice(), &raw, &vmax));
        }
        eprintln!(
            "flash attention KV {}: error máximo / max|v| {w:.2e} (f16), {w_raw:.2e} (f16 vs sin redondear), {w32:.2e} (f32)",
            kv.name()
        );
        assert!(w <= 1e-5, "{}: {w}", kv.name());
        assert!(w32 <= 1e-5, "{}: {w32}", kv.name());
        assert!(w_raw <= 2e-3, "{}: {w_raw}", kv.name());
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
fn store_kv_q8_igual_a_la_especificacion() {
    // Bit a bit contra brasa_quant::quantize_kv_q8 (ADR 0009), con bloques nulos, empates en
    // x / d (múltiplos impares de d / 2) y valores grandes.
    let ctx = Context::new().unwrap();
    let k = Kernels::new(&ctx).unwrap();
    let mut rng = Rng::new(26);
    let mut x = rng.vec(128 * 64, 20.0);
    x[32..64].fill(0.0);
    let d = f16_to_f32(f32_to_f16(1.0 / 127.0));
    for j in 0..32 {
        x[128 + j] = (j as f32 - 15.5) * d; // empates exactos (x / d = n + 0,5)
    }
    x[128 + 31] = 1.0; // amax = 1 → d = f16(1 / 127)
    x[256..288].fill(3000.0);
    let src = ctx.buffer_from(&x).unwrap();
    let mut dst = ctx.buffer::<u8>(KvType::Q8_0.bytes(x.len())).unwrap();
    let mut cmd = ctx.command().unwrap();
    k.store_kv(
        &mut cmd,
        KvType::Q8_0,
        Arg::buf(&src),
        Arg::buf(&dst),
        x.len(),
    );
    cmd.commit_and_wait().unwrap();
    let mut want = vec![0u8; KvType::Q8_0.bytes(x.len())];
    quantize_kv_q8(&x, &mut want);
    assert_eq!(dst.as_mut_slice(), &want[..]);
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
