//! Desglose del decode de Qwen3-4B por kernel (T3.5): cada operación de una capa se encola 36
//! veces seguidas en un command buffer, como en el forward, y se mide el tiempo de GPU (mediana).
//! Al final, la capa completa ×36 + lm_head, para ver cuánto suman los huecos entre dispatches.
//! Pesos aleatorios con las formas reales; la KV en f16 con `pos` claves.
//!   cargo run --release -p brasa-kernels --example decode_breakdown -- [pos]
use brasa_kernels::testutil::{KvPair, Rng, median};
use brasa_kernels::{AttnShape, Kernels, KvType, QMatrix, RopeTable, WeightType};
use brasa_metal::{Arg, Command, Context};

const LAYERS: usize = 36;

fn time<'a>(ctx: &Context, mut encode: impl FnMut(&mut Command<'a>)) -> f64 {
    let t: Vec<f64> = (0..9)
        .map(|_| {
            let mut cmd = ctx.command().unwrap();
            encode(&mut cmd);
            cmd.commit_and_wait().unwrap().gpu_seconds
        })
        .skip(2)
        .collect();
    median(t)
}

fn main() {
    let pos: usize = std::env::args().nth(1).map_or(2000, |s| s.parse().unwrap());
    let ctx = Context::new().unwrap();
    let k = Kernels::new(&ctx).unwrap();
    let mut rng = Rng::new(3);
    let (h, ffn, hq, hkv, hd, vocab) = (2560, 9728, 32, 8, 128, 151_936);
    let buf = |rng: &mut Rng, n: usize| ctx.buffer_from(&rng.vec(n, 1.0)).unwrap();
    let (x, hb, q, kn, vn, attn) = (
        buf(&mut rng, h),
        buf(&mut rng, h),
        buf(&mut rng, hq * hd),
        buf(&mut rng, hkv * hd),
        buf(&mut rng, hkv * hd),
        buf(&mut rng, hq * hd),
    );
    let (gate, up, logits) = (buf(&mut rng, ffn), buf(&mut rng, ffn), buf(&mut rng, vocab));
    let (nw, hw) = (buf(&mut rng, h), buf(&mut rng, hd));
    let q4 = |rng: &mut Rng, r: usize, c: usize| ctx.buffer_from(&rng.q4_0(r, c)).unwrap();
    let (wq, wk, wv, wo) = (
        q4(&mut rng, hq * hd, h),
        q4(&mut rng, hkv * hd, h),
        q4(&mut rng, hkv * hd, h),
        q4(&mut rng, h, hq * hd),
    );
    let (wg, wu, wd) = (
        q4(&mut rng, ffn, h),
        q4(&mut rng, ffn, h),
        q4(&mut rng, h, ffn),
    );
    let emb = ctx.buffer_from(&rng.q8_0(vocab, h)).unwrap();
    let m = |data, rows, cols| QMatrix {
        data,
        qtype: WeightType::Q4_0,
        rows,
        cols,
    };
    let (mq, mk, mv, mo) = (
        m(&wq, hq * hd, h),
        m(&wk, hkv * hd, h),
        m(&wv, hkv * hd, h),
        m(&wo, h, hq * hd),
    );
    let (mg, mu, md) = (m(&wg, ffn, h), m(&wu, ffn, h), m(&wd, h, ffn));
    let head = QMatrix {
        data: &emb,
        qtype: WeightType::Q8_0,
        rows: vocab,
        cols: h,
    };
    let rope = RopeTable::new(&ctx, 1e6, hd, pos + 1).unwrap();
    let lk = pos + 1;
    let cache = KvPair::new(
        &ctx,
        KvType::F16,
        rng.vec(lk.next_multiple_of(128) * hkv * hd, 1.0),
        rng.vec(lk.next_multiple_of(128) * hkv * hd, 1.0),
    );
    let (kc, vc) = cache.args();
    let part = ctx
        .buffer::<f32>(brasa_kernels::decode_partials_len(hq, lk))
        .unwrap();
    let shape = AttnShape {
        tokens: 1,
        hq,
        hkv,
        dim: hd,
        pos0: pos,
        kv: KvType::F16,
    };
    let kvst = ctx.buffer::<u16>(hkv * hd).unwrap();

    println!(
        "dispositivo: {}; decode en la posición {pos}",
        ctx.device_name()
    );
    println!(
        "{:<34} {:>9} {:>9}",
        "operación (×36 capas)", "ms/token", "µs/disp."
    );
    let mut total = 0.0;
    let mut row = |name: &str, per_layer: usize, t: f64| {
        total += t;
        println!(
            "{name:<34} {:>9.3} {:>9.1}",
            t * 1e3,
            t * 1e6 / (LAYERS * per_layer) as f64
        );
    };
    // Encola el cuerpo 36 veces (una por capa) en un command buffer y mide.
    macro_rules! rep {
        ($c:ident => $body:block) => {
            time(&ctx, |$c| {
                for _ in 0..LAYERS $body
            })
        };
    }
    let (hn, rows_q, rows_k) = (h, hq, hkv);
    row(
        "rms_norm H (×2)",
        2,
        rep!(c => {
            k.rms_norm(c, Arg::buf(&x), Arg::buf(&nw), Arg::buf(&hb), 1, hn, 1e-6);
            k.rms_norm(c, Arg::buf(&x), Arg::buf(&nw), Arg::buf(&hb), 1, hn, 1e-6);
        }),
    );
    row(
        "rms_norm q y k por cabeza (×2)",
        2,
        rep!(c => {
            k.rms_norm(c, Arg::buf(&q), Arg::buf(&hw), Arg::buf(&q), rows_q, hd, 1e-6);
            k.rms_norm(c, Arg::buf(&kn), Arg::buf(&hw), Arg::buf(&kn), rows_k, hd, 1e-6);
        }),
    );
    row(
        "gemv q, k, v",
        3,
        rep!(c => {
            k.gemv(c, mq, Arg::buf(&hb), Arg::buf(&q), 1);
            k.gemv(c, mk, Arg::buf(&hb), Arg::buf(&kn), 1);
            k.gemv(c, mv, Arg::buf(&hb), Arg::buf(&vn), 1);
        }),
    );
    row(
        "rope q, k",
        2,
        rep!(c => {
            k.rope_neox(c, Arg::buf(&q), &rope, 1, hq, hd, pos);
            k.rope_neox(c, Arg::buf(&kn), &rope, 1, hkv, hd, pos);
        }),
    );
    row(
        "store_kv k, v",
        2,
        rep!(c => {
            k.store_kv(c, KvType::F16, Arg::buf(&kn), Arg::buf(&kvst), hkv * hd);
            k.store_kv(c, KvType::F16, Arg::buf(&vn), Arg::buf(&kvst), hkv * hd);
        }),
    );
    row(
        "decode_attention_lanes + reduce",
        2,
        rep!(c => {
            k.decode_attention_lanes(c, Arg::buf(&q), kc, vc, &part, Arg::buf(&attn), shape);
        }),
    );
    row(
        "gemv o",
        1,
        rep!(c => {
            k.gemv(c, mo, Arg::buf(&attn), Arg::buf(&hb), 1);
        }),
    );
    row(
        "add residual (×2)",
        2,
        rep!(c => {
            k.add(c, &x, &hb, &x, h);
            k.add(c, &x, &hb, &x, h);
        }),
    );
    row(
        "gemv gate, up",
        2,
        rep!(c => {
            k.gemv(c, mg, Arg::buf(&hb), Arg::buf(&gate), 1);
            k.gemv(c, mu, Arg::buf(&hb), Arg::buf(&up), 1);
        }),
    );
    row(
        "swiglu",
        1,
        rep!(c => {
            k.swiglu(c, &gate, &up, &gate, ffn);
        }),
    );
    row(
        "gemv down",
        1,
        rep!(c => {
            k.gemv(c, md, Arg::buf(&gate), Arg::buf(&hb), 1);
        }),
    );
    let qk_fused = rep!(c => {
        k.qk_norm_rope_store(
            c,
            KvType::F16,
            [Arg::buf(&q), Arg::buf(&kn), Arg::buf(&vn)],
            [Arg::buf(&hw), Arg::buf(&hw)],
            1e-6,
            &rope,
            [Arg::buf(&kvst), Arg::buf(&kvst)],
            shape,
        );
    });
    let ssb = ctx.buffer::<f32>(brasa_kernels::norm_partials(h)).unwrap();
    let prep2 = rep!(c => {
        k.add_norm_prep(c, Arg::buf(&x), Some(Arg::buf(&hb)), Arg::buf(&nw), Arg::buf(&attn), Arg::buf(&ssb), h);
        k.add_norm_prep(c, Arg::buf(&x), Some(Arg::buf(&hb)), Arg::buf(&nw), Arg::buf(&attn), Arg::buf(&ssb), h);
    });
    let scaled_qkv = rep!(c => {
        k.gemv_scaled(c, mq, Arg::buf(&attn), Arg::buf(&ssb), 1e-6, Arg::buf(&q));
        k.gemv_scaled(c, mk, Arg::buf(&attn), Arg::buf(&ssb), 1e-6, Arg::buf(&kn));
        k.gemv_scaled(c, mv, Arg::buf(&attn), Arg::buf(&ssb), 1e-6, Arg::buf(&vn));
    });
    let scaled_gu = rep!(c => {
        k.gemv_scaled(c, mg, Arg::buf(&attn), Arg::buf(&ssb), 1e-6, Arg::buf(&gate));
        k.gemv_scaled(c, mu, Arg::buf(&attn), Arg::buf(&ssb), 1e-6, Arg::buf(&up));
    });
    let merged_qkv = rep!(c => {
        k.gemv_scaled3(
            c,
            [mq, mk, mv],
            Arg::buf(&attn),
            Arg::buf(&ssb),
            1e-6,
            [Arg::buf(&q), Arg::buf(&kn), Arg::buf(&vn)],
        );
    });
    let merged_gu = rep!(c => {
        k.gemv_scaled_swiglu(c, mg, mu, Arg::buf(&attn), Arg::buf(&ssb), 1e-6, Arg::buf(&gate));
    });
    let t = time(&ctx, |c| {
        k.gemv(c, head, Arg::buf(&hb), Arg::buf(&logits), 1)
    });
    total += t;
    println!("{:<34} {:>9.3}", "lm_head (q8_0, una vez)", t * 1e3);
    println!("{:<34} {:>9.3}", "suma", total * 1e3);
    println!(
        "{:<34} {:>9.3}   (reemplaza rms_norm q/k + rope + store_kv)",
        "qk_norm_rope_store (fusionado)",
        qk_fused * 1e3
    );
    println!(
        "{:<34} {:>9.3}   (reemplaza rms_norm H ×2 + add ×2)",
        "add_norm_prep (×2)",
        prep2 * 1e3
    );
    println!("{:<34} {:>9.3}", "gemv_scaled q, k, v", scaled_qkv * 1e3);
    println!("{:<34} {:>9.3}", "gemv_scaled gate, up", scaled_gu * 1e3);
    println!(
        "{:<34} {:>9.3}",
        "gemv_scaled3 q, k, v (1 dispatch)",
        merged_qkv * 1e3
    );
    println!(
        "{:<34} {:>9.3}   (reemplaza gate + up + swiglu)",
        "gemv_scaled_swiglu",
        merged_gu * 1e3
    );
}
