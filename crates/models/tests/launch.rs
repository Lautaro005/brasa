//! ADR 0029: los parámetros de lanzamiento tuneados no cambian los logits. El forward completo
//! (prefill y decode) con simdgroups por threadgroup distintos de los valores por defecto en todos
//! los GEMV de decode y en la atención de decode da los mismos bits que con los valores por
//! defecto.
//!
//!   cargo test --release -p brasa-models --test launch -- --ignored --nocapture

use std::path::{Path, PathBuf};

use brasa_metal::Context;
use brasa_models::qwen3::{Config, KvType, Launch, Limits, Qwen3, WeightType, weight_types};
use brasa_tuner::tune::ModelDims;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Todas las entradas de decode del modelo con `sg` (o el mayor valor válido por debajo).
fn launch_with(cfg: &Config, path: &Path, kv: KvType, sg: usize) -> Launch {
    let (layer_type, head_type) = weight_types(path).unwrap();
    let dims = ModelDims {
        layers: cfg.layers,
        hidden: cfg.hidden,
        heads: cfg.heads,
        kv_heads: cfg.kv_heads,
        head_dim: cfg.head_dim,
        ffn: cfg.ffn,
        vocab: cfg.vocab,
        layer_type,
        head_type,
    };
    assert_eq!(layer_type, WeightType::Q4_0);
    let mut l = Launch::default();
    for (op, rows, cols, _) in dims.gemvs() {
        l.set_gemv(op, rows, cols, sg).unwrap();
    }
    for lk in [64, 4096] {
        l.set_attn_lanes(kv, cfg.heads / cfg.kv_heads, lk, sg)
            .unwrap();
    }
    l
}

/// Logits (bits) de un prefill de 300 tokens y 12 pasos de decode.
fn run(ctx: &Context, model: &mut Qwen3) -> Vec<Vec<u32>> {
    let vocab = model.cfg.vocab;
    let mut logits = vec![0f32; vocab];
    let prompt: Vec<u32> = (0..300).map(|i| 1000 + (i * 37) % 9000).collect();
    let mut out = Vec::new();
    for (i, c) in prompt.chunks(128).enumerate() {
        model.forward(ctx, c, i * 128, 1, &mut logits).unwrap();
    }
    out.push(logits.iter().map(|x| x.to_bits()).collect());
    for step in 0..12 {
        let id = 500 + step as u32 * 11;
        model
            .forward(ctx, &[id], prompt.len() + step, 1, &mut logits)
            .unwrap();
        out.push(logits.iter().map(|x| x.to_bits()).collect());
    }
    out
}

#[test]
#[ignore = "requiere models/qwen3-4b-q4/model.brasa"]
fn logits_iguales_con_parametros_tuneados() {
    let path = root().join("models/qwen3-4b-q4/model.brasa");
    for kv in [KvType::F16, KvType::Q8_0] {
        let ctx = Context::new().unwrap();
        let limits = Limits {
            ctx: 1024,
            max_tokens: 128,
            max_logit_rows: 1,
            kv,
        };
        let mut model = Qwen3::load(&ctx, &path, limits).unwrap();
        model.set_launch(&ctx, Launch::default()).unwrap();
        let want = run(&ctx, &mut model);
        for sg in [1usize, 4, 8] {
            let l = launch_with(&model.cfg, &path, kv, sg);
            model.set_launch(&ctx, l).unwrap();
            let got = run(&ctx, &mut model);
            let distintos = got.iter().zip(&want).filter(|(a, b)| a != b).count();
            println!(
                "KV {} sg {sg}: {distintos} de {} pasos con logits distintos",
                kv.name(),
                want.len()
            );
            assert_eq!(distintos, 0, "KV {} sg {sg}", kv.name());
        }
    }
}
