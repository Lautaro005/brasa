//! `brasa bench-once` (oculto): una corrida medida para `brasa benchmark` (ADR 0002).
//! Texto plano sin template, greedy, tokens de parada bloqueados; imprime una línea JSON.

use std::path::PathBuf;

use brasa_runtime::{Limits, Sampler, SamplingParams, Session};
use clap::Args;
use serde_json::json;

#[derive(Debug, Args)]
pub struct BenchOnceArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    prompt: PathBuf,
    #[arg(long = "gen")]
    gen_tokens: usize,
    #[arg(long)]
    ctx: usize,
    #[arg(long, default_value_t = 128)]
    chunk: usize,
}

pub fn run(a: BenchOnceArgs) -> Result<(), String> {
    let limits = Limits {
        ctx: a.ctx,
        max_tokens: a.chunk,
        max_logit_rows: 1,
    };
    let mut s = Session::load(&a.model, limits).map_err(|e| e.to_string())?;
    s.ignore_stop = true;
    let text = std::fs::read_to_string(&a.prompt).map_err(|e| e.to_string())?;
    let ids = s.tokenizer().encode(&text);
    let mut sampler = Sampler::new(SamplingParams::greedy(), s.vocab());
    // Calentamiento equivalente al de llama.cpp al cargar (excluido de los tiempos).
    s.generate(&ids[..16.min(ids.len())], 1, &mut sampler, |_| true)
        .map_err(|e| e.to_string())?;
    s.reset();
    let st = s
        .generate(&ids, a.gen_tokens, &mut sampler, |_| true)
        .map_err(|e| e.to_string())?;
    println!(
        "{}",
        json!({
            "prompt_tokens": st.prompt_tokens,
            "gen_tokens": st.generated,
            "ttft_ms": st.ttft_ms,
            "prefill_ms": st.prefill_ms,
            "decode_ms": st.decode_ms,
        })
    );
    Ok(())
}
