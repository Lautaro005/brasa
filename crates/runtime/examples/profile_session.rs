//! Perfil del decode a través de Session (comparar con brasa-models/examples/profile_decode).
//!   cargo run --release -p brasa-runtime --example profile_session -- [prompt] [ctx]

use std::path::Path;
use std::time::Instant;

use brasa_runtime::{Limits, Sampler, SamplingParams, Session};

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "fixtures/bench/prompt-2048.txt".into());
    let ctx: usize = std::env::args().nth(2).map_or(2048, |s| s.parse().unwrap());
    let limits = Limits {
        ctx,
        max_tokens: 512,
        max_logit_rows: 1,
    };
    let mut s = Session::load(&root.join("models/qwen3-4b-q4"), limits).unwrap();
    s.ignore_stop = true;
    let ids = s
        .tokenizer()
        .encode(&std::fs::read_to_string(root.join(prompt)).unwrap());
    let mut sampler = Sampler::new(SamplingParams::greedy(), s.vocab());
    if std::env::var("WARMUP").is_ok() {
        s.generate(&ids[..16], 1, &mut sampler, |_| true).unwrap();
        s.reset();
    }
    let mut steps = Vec::new();
    let mut last = Instant::now();
    let st = s
        .generate(&ids, 128, &mut sampler, |_| {
            steps.push(last.elapsed().as_secs_f64() * 1e3);
            last = Instant::now();
            true
        })
        .unwrap();
    steps.remove(0);
    steps.sort_by(f64::total_cmp);
    println!(
        "prefill {:.0} ms; decode {:.1} tok/s; por token: mediana {:.1} ms, p10 {:.1}, p90 {:.1}, máx {:.1}",
        st.prefill_ms,
        st.decode_tok_s(),
        steps[steps.len() / 2],
        steps[steps.len() / 10],
        steps[steps.len() * 9 / 10],
        steps[steps.len() - 1]
    );
}
