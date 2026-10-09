//! TurboQuant TQ4 (ADR 0033): equivalencia GPU vs CPU de la rotación, la escritura de la caché y
//! la atención en el dominio rotado. Las referencias de la atención decuantizan los bytes que
//! escribió la propia GPU, así que el error medido es del kernel y no de la cuantización. Lo que
//! mide la cuantización (el error relativo de la caché) se reporta aparte.

use brasa_kernels::testutil::Rng;
use brasa_kernels::{AttnShape, Kernels, KvType, PrefillPrecision, decode_partials_len, reference};
use brasa_metal::{Arg, Context};
use brasa_quant::f32_to_f16;
use brasa_quant::turbo::{self, KV_TQ4_ROW, TQ_DIM};

const D: usize = TQ_DIM;
const HQ: usize = 32;
const HKV: usize = 8;

/// `y = R·x` (o `Rᵀ·x`) de cada fila de 128, en f64.
fn rotate_ref(x: &[f32], r: &[f32], transpose: bool) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for (xr, or) in x.chunks_exact(D).zip(out.chunks_exact_mut(D)) {
        for (i, o) in or.iter_mut().enumerate() {
            let s: f64 = (0..D)
                .map(|j| {
                    let m = if transpose {
                        r[j * D + i]
                    } else {
                        r[i * D + j]
                    };
                    m as f64 * xr[j] as f64
                })
                .sum();
            *o = s as f32;
        }
    }
    out
}

/// Código de la coordenada `i` de una fila TQ4.
fn nibble(row: &[u8], i: usize) -> u8 {
    if i & 1 == 0 {
        row[i / 2] & 15
    } else {
        row[i / 2] >> 4
    }
}

/// Dominio rotado de todas las filas de una caché TQ4 (`ŷ = norma · codebook[c]`).
fn dequant_rotated(bytes: &[u8]) -> Vec<f32> {
    let mut out = vec![0f32; bytes.len() / KV_TQ4_ROW * D];
    for (row, o) in bytes.chunks_exact(KV_TQ4_ROW).zip(out.chunks_exact_mut(D)) {
        turbo::dequantize_row(row, o);
    }
    out
}

fn max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0f32, |m, x| m.max(x.abs()))
}

#[test]
fn tq_rotate_igual_a_la_referencia() {
    let ctx = Context::new().unwrap();
    let k = Kernels::new(&ctx).unwrap();
    let r = turbo::rotation();
    let gr = ctx.buffer_from(&r).unwrap();
    let mut rng = Rng::new(31);
    let rows = 64;
    let x = rng.vec(rows * D, 2.0);
    for transpose in [false, true] {
        let gx = ctx.buffer_from(&x).unwrap();
        let mut cmd = ctx.command().unwrap();
        k.tq_rotate(&mut cmd, Arg::buf(&gx), Arg::buf(&gr), rows, transpose);
        cmd.commit_and_wait().unwrap();
        let want = rotate_ref(&x, &r, transpose);
        let err = gx
            .as_slice()
            .iter()
            .zip(&want)
            .fold(0f32, |m, (g, e)| m.max((g - e).abs()));
        let scale = max_abs(&want);
        eprintln!(
            "tq_rotate transpose={transpose}: error {:.2e} / max {:.2e}",
            err, scale
        );
        assert!(err <= 1e-5 * scale, "transpose={transpose}: {err}");
    }
}

#[test]
fn tq_rotate_f16_igual_a_la_referencia() {
    let ctx = Context::new().unwrap();
    let k = Kernels::new(&ctx).unwrap();
    let r = turbo::rotation();
    let gr = ctx.buffer_from(&r).unwrap();
    let mut rng = Rng::new(32);
    let rows = 64;
    let x = rng.vec(rows * D, 2.0);
    // La entrada ya es f16 (como la salida de la atención de prefill); la salida también.
    let bits: Vec<u16> = x.iter().map(|&v| f32_to_f16(v)).collect();
    let xr: Vec<f32> = bits.iter().map(|&b| brasa_quant::f16_to_f32(b)).collect();
    let gx = ctx.buffer_from(&bits).unwrap();
    let mut cmd = ctx.command().unwrap();
    k.tq_rotate_f16(&mut cmd, Arg::buf(&gx), Arg::buf(&gr), rows, true);
    cmd.commit_and_wait().unwrap();
    let want = rotate_ref(&xr, &r, true);
    let scale = max_abs(&want);
    let err = gx.as_slice().iter().zip(&want).fold(0f32, |m, (g, e)| {
        m.max((brasa_quant::f16_to_f32(*g) - e).abs())
    });
    eprintln!("tq_rotate_f16: error {err:.2e} / max {scale:.2e}");
    // Redondeo de la salida a f16: 2⁻¹¹ relativo sobre el máximo de la fila.
    assert!(err <= 2f32.powi(-10) * scale, "{err}");
}

#[test]
fn tq_store_igual_a_la_especificacion() {
    let ctx = Context::new().unwrap();
    let k = Kernels::new(&ctx).unwrap();
    let r = turbo::rotation();
    let gr = ctx.buffer_from(&r).unwrap();
    let mut rng = Rng::new(33);
    let rows = 512;
    let x = rng.vec(rows * D, 3.0);
    let gsrc = ctx.buffer_from(&x).unwrap();
    let mut gdst = ctx.buffer::<u8>(rows * KV_TQ4_ROW).unwrap();
    let mut cmd = ctx.command().unwrap();
    k.tq_store(
        &mut cmd,
        Arg::buf(&gsrc),
        Arg::buf(&gdst),
        Arg::buf(&gr),
        rows,
    );
    cmd.commit_and_wait().unwrap();

    let got = gdst.as_mut_slice().to_vec();
    let mut want = vec![0u8; rows * KV_TQ4_ROW];
    for (xr, o) in x.chunks_exact(D).zip(want.chunks_exact_mut(KV_TQ4_ROW)) {
        turbo::quantize_row(xr, &r, o);
    }
    // Códigos: la suma de f32 en GPU y f64 en CPU puede cambiar un código cuando la coordenada
    // cae justo en un umbral. Se exige que sea raro.
    let mut mism = 0usize;
    let mut norm_bad = 0usize;
    let mut pad_bad = 0usize;
    for (g, w) in got
        .chunks_exact(KV_TQ4_ROW)
        .zip(want.chunks_exact(KV_TQ4_ROW))
    {
        mism += (0..D).filter(|&i| nibble(g, i) != nibble(w, i)).count();
        norm_bad += (g[64..66] != w[64..66]) as usize;
        pad_bad += (g[66..68] != [0, 0]) as usize;
    }
    let mism_rate = mism as f64 / (rows * D) as f64;
    eprintln!("tq_store: códigos distintos {mism_rate:.2e}, normas distintas {norm_bad}/{rows}");
    assert!(mism_rate <= 2e-3, "códigos distintos: {mism_rate}");
    assert!(norm_bad <= rows / 50, "normas distintas: {norm_bad}");
    assert_eq!(pad_bad, 0, "relleno en cero");

    // Calidad de la caché escrita por la GPU: error relativo ‖x − x̂‖² / ‖x‖² en el dominio
    // original. La teoría de Lloyd-Max de 4 bits da 0,009497; con 512 filas el margen es ±0,0005.
    let (mut err, mut tot) = (0f64, 0f64);
    for (xr, g) in x.chunks_exact(D).zip(got.chunks_exact(KV_TQ4_ROW)) {
        let mut xh = vec![0f32; D];
        turbo::dequantize_row_original(g, &r, &mut xh);
        for (a, b) in xr.iter().zip(&xh) {
            err += ((a - b) as f64).powi(2);
            tot += (*a as f64).powi(2);
        }
    }
    let rel = err / tot;
    eprintln!("tq_store: error relativo de la caché {rel:.5} (teórico 0,00950)");
    assert!((rel - 0.009497).abs() < 0.0005, "{rel}");
}

/// Atención de decode con la caché TQ4 escrita por la GPU. La referencia está en el dominio
/// rotado: q' = R·q y K, V decuantizados de los bytes de la GPU.
fn decode_case(lanes: bool) {
    let ctx = Context::new().unwrap();
    let k = Kernels::new(&ctx).unwrap();
    let r = turbo::rotation();
    let gr = ctx.buffer_from(&r).unwrap();
    let mut rng = Rng::new(if lanes { 41 } else { 42 });
    let mut w_rot = 0f32;
    let mut w_orig = 0f32;
    for lk in [1usize, 37, 300, 1500] {
        let pos0 = lk - 1;
        let kx = rng.vec(lk * HKV * D, 2.0);
        let vx = rng.vec(lk * HKV * D, 3.0);
        let q = rng.vec(HQ * D, 2.0);
        let gk = ctx.buffer_from(&kx).unwrap();
        let gv = ctx.buffer_from(&vx).unwrap();
        let mut ck = ctx.buffer::<u8>(lk * HKV * KV_TQ4_ROW).unwrap();
        let mut cv = ctx.buffer::<u8>(lk * HKV * KV_TQ4_ROW).unwrap();
        let mut gq = ctx.buffer_from(&q).unwrap();
        let mut cmd = ctx.command().unwrap();
        k.tq_store(
            &mut cmd,
            Arg::buf(&gk),
            Arg::buf(&ck),
            Arg::buf(&gr),
            lk * HKV,
        );
        k.tq_store(
            &mut cmd,
            Arg::buf(&gv),
            Arg::buf(&cv),
            Arg::buf(&gr),
            lk * HKV,
        );
        k.tq_rotate(&mut cmd, Arg::buf(&gq), Arg::buf(&gr), HQ, false);
        cmd.commit_and_wait().unwrap();

        let kd = dequant_rotated(ck.as_mut_slice());
        let vd = dequant_rotated(cv.as_mut_slice());
        let qr = gq.as_mut_slice().to_vec();
        let (expected, vmax) = reference::attention(&qr, &kd, &vd, 1, HQ, HKV, D, pos0);

        let partials = ctx.buffer::<f32>(decode_partials_len(HQ, lk)).unwrap();
        let o = ctx.buffer::<f32>(HQ * D).unwrap();
        let shape = AttnShape {
            tokens: 1,
            hq: HQ,
            hkv: HKV,
            dim: D,
            pos0,
            kv: KvType::Tq4,
        };
        let mut cmd = ctx.command().unwrap();
        if lanes {
            k.decode_attention_lanes(
                &mut cmd,
                Arg::buf(&gq),
                Arg::buf(&ck),
                Arg::buf(&cv),
                &partials,
                Arg::buf(&o),
                shape,
            );
        } else {
            k.decode_attention(
                &mut cmd,
                Arg::buf(&gq),
                Arg::buf(&ck),
                Arg::buf(&cv),
                &partials,
                Arg::buf(&o),
                shape,
            );
        }
        cmd.commit_and_wait().unwrap();
        let err = o
            .as_slice()
            .iter()
            .zip(&expected)
            .zip(&vmax)
            .map(|((g, e), m)| (g - e).abs() / m.max(1e-30))
            .fold(0f32, f32::max);
        w_rot = w_rot.max(err);

        // Calidad frente a la atención exacta (sin cuantizar): salida del dominio rotado a
        // original con Rᵀ.
        // La escala es el máximo global de la salida exacta: con lk = 1, `vmax` por componente
        // es |v| de esa componente, que puede ser casi cero y dispara el cociente.
        let o_orig = rotate_ref(o.as_slice(), &r, true);
        let (exact, _) = reference::attention(&q, &kx, &vx, 1, HQ, HKV, D, pos0);
        let scale = max_abs(&exact);
        let e2 = o_orig
            .iter()
            .zip(&exact)
            .fold(0f32, |m, (g, e)| m.max((g - e).abs()))
            / scale;
        w_orig = w_orig.max(e2);
    }
    let name = if lanes {
        "decode_attention_lanes"
    } else {
        "decode_attention"
    };
    eprintln!(
        "{name} TQ4: error máx. vs referencia del dominio rotado {w_rot:.2e}; vs atención exacta {w_orig:.2e} (escala: máx. de la salida)"
    );
    assert!(w_rot <= 1e-5, "{name} TQ4: {w_rot}");
}

#[test]
fn decode_attention_tq4_por_cabeza() {
    decode_case(false);
}

#[test]
fn decode_attention_lanes_tq4() {
    decode_case(true);
}

#[test]
fn flash_attention_tq4_igual_a_la_referencia() {
    let ctx = Context::new().unwrap();
    let k = Kernels::new(&ctx).unwrap();
    let r = turbo::rotation();
    let gr = ctx.buffer_from(&r).unwrap();
    let mut rng = Rng::new(43);
    let mut w_f32 = 0f32;
    let mut w_f16 = 0f32;
    // Prefill desde 0 y continuado con prefijo en caché; con Q/salida en f32 y en f16 (la ruta
    // de prefill de Qwen3, ADR 0030, entrega la salida en f16 en bits `u16`).
    for (tokens, pos0) in [(40usize, 0usize), (16, 24)] {
        let lk = pos0 + tokens;
        let cap = lk.next_multiple_of(64);
        let kx = rng.vec(lk * HKV * D, 2.0);
        let vx = rng.vec(lk * HKV * D, 3.0);
        let q = rng.vec(tokens * HQ * D, 2.0);
        let gk = ctx.buffer_from(&kx).unwrap();
        let gv = ctx.buffer_from(&vx).unwrap();
        // La caché tiene la capacidad de KV_ALIGN; lo que no se escribe queda en cero.
        let mut ck = ctx.buffer::<u8>(cap * HKV * KV_TQ4_ROW).unwrap();
        let mut cv = ctx.buffer::<u8>(cap * HKV * KV_TQ4_ROW).unwrap();
        ck.as_mut_slice().fill(0);
        cv.as_mut_slice().fill(0);
        let mut gq = ctx.buffer_from(&q).unwrap();
        let of32 = ctx.buffer::<f32>(tokens * HQ * D).unwrap();
        let of16 = ctx.buffer::<u16>(tokens * HQ * D).unwrap();
        let shape = AttnShape {
            tokens,
            hq: HQ,
            hkv: HKV,
            dim: D,
            pos0,
            kv: KvType::Tq4,
        };
        let mut cmd = ctx.command().unwrap();
        k.tq_store(
            &mut cmd,
            Arg::buf(&gk),
            Arg::buf(&ck),
            Arg::buf(&gr),
            lk * HKV,
        );
        k.tq_store(
            &mut cmd,
            Arg::buf(&gv),
            Arg::buf(&cv),
            Arg::buf(&gr),
            lk * HKV,
        );
        k.tq_rotate(&mut cmd, Arg::buf(&gq), Arg::buf(&gr), tokens * HQ, false);
        cmd.commit_and_wait().unwrap();
        let kd = dequant_rotated(&ck.as_slice()[..lk * HKV * KV_TQ4_ROW]);
        let vd = dequant_rotated(&cv.as_slice()[..lk * HKV * KV_TQ4_ROW]);
        let qr = gq.as_mut_slice().to_vec();
        let (expected, vmax) = reference::attention(&qr, &kd, &vd, tokens, HQ, HKV, D, pos0);

        let mut cmd = ctx.command().unwrap();
        k.flash_attention_with(
            &mut cmd,
            Arg::buf(&gq),
            Arg::buf(&ck),
            Arg::buf(&cv),
            Arg::buf(&of32),
            shape,
            PrefillPrecision::F32,
        );
        k.flash_attention_with(
            &mut cmd,
            Arg::buf(&gq),
            Arg::buf(&ck),
            Arg::buf(&cv),
            Arg::buf(&of16),
            shape,
            PrefillPrecision::F16,
        );
        cmd.commit_and_wait().unwrap();
        let err32 = of32
            .as_slice()
            .iter()
            .zip(&expected)
            .zip(&vmax)
            .map(|((g, e), m)| (g - e).abs() / m.max(1e-30))
            .fold(0f32, f32::max);
        // Salida f16: se descuenta el redondeo final (2⁻¹¹ · |valor|), como en attention.rs.
        let err16 = of16
            .as_slice()
            .iter()
            .zip(&expected)
            .zip(&vmax)
            .map(|((g, e), m)| {
                let d = (brasa_quant::f16_to_f32(*g) - e).abs() - e.abs() * 2f32.powi(-11);
                d.max(0.0) / m.max(1e-30)
            })
            .fold(0f32, f32::max);
        w_f32 = w_f32.max(err32);
        w_f16 = w_f16.max(err16);
    }
    eprintln!(
        "flash_attention TQ4 f32: error {w_f32:.2e}; f16: error {w_f16:.2e} (ambos / max|v|)"
    );
    assert!(w_f32 <= 1e-5, "flash_attention TQ4 f32: {w_f32}");
    assert!(w_f16 <= 1e-3, "flash_attention TQ4 f16: {w_f16}");
}
