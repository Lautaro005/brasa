//! `brasa run <modelo>`: chat en la terminal con el chat template del modelo.
//! Con `--prompt` responde una vez y sale; sin él, lee mensajes de stdin (una línea por turno).

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use brasa_memory::telemetry::MemorySampler;
use brasa_runtime::{Limits, Sampler, SamplingParams, Session, StopReason};
use brasa_tokenizer::RenderOptions;
use clap::Args;
use serde_json::{Value, json};

#[derive(Debug, Args)]
pub struct RunArgs {
    /// Nombre del modelo (carpeta en ./models o $BRASA_MODELS) o ruta a su carpeta.
    model: String,
    /// Mensaje del usuario; sin esto, modo interactivo.
    #[arg(long, short)]
    prompt: Option<String>,
    /// Mensaje de sistema.
    #[arg(long)]
    system: Option<String>,
    /// Desactiva el razonamiento (`enable_thinking = false`).
    #[arg(long)]
    no_think: bool,
    /// Tokens máximos por respuesta.
    #[arg(long, default_value_t = 1024)]
    max_tokens: usize,
    /// Contexto (posiciones de la KV cache).
    #[arg(long, default_value_t = 4096)]
    ctx: usize,
    /// Tokens por bloque de prefill.
    #[arg(long, default_value_t = 128)]
    chunk: usize,
    /// Tipo de la KV cache: f16 (por defecto), q8_0 o f32 (ADR 0009).
    #[arg(long, default_value = "f16", value_parser = crate::parse_kv)]
    kv: brasa_runtime::KvType,
    /// Greedy (temperatura 0); ignora temp/top-k/top-p.
    #[arg(long)]
    greedy: bool,
    #[arg(long)]
    temp: Option<f32>,
    #[arg(long)]
    top_k: Option<usize>,
    #[arg(long)]
    top_p: Option<f32>,
    /// Semilla; por defecto, una distinta por ejecución (se informa al final).
    #[arg(long)]
    seed: Option<u64>,
}

pub fn resolve_model(name: &str) -> Result<PathBuf, String> {
    let direct = Path::new(name);
    if direct.join("model.brasa").exists() {
        return Ok(direct.to_path_buf());
    }
    let base = std::env::var("BRASA_MODELS").unwrap_or_else(|_| "models".into());
    let p = Path::new(&base).join(name);
    if p.join("model.brasa").exists() {
        return Ok(p);
    }
    Err(format!(
        "no se encontró el modelo {name:?} (buscado en {} y como ruta). Convertilo con tools/convert_brasa.py",
        p.display()
    ))
}

fn sampling(args: &RunArgs) -> SamplingParams {
    let seed = args.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos() as u64)
    });
    if args.greedy {
        return SamplingParams {
            seed,
            ..SamplingParams::greedy()
        };
    }
    let mut p = if args.no_think {
        SamplingParams::qwen3_no_thinking(seed)
    } else {
        SamplingParams::qwen3_thinking(seed)
    };
    if let Some(t) = args.temp {
        p.temperature = t;
    }
    if let Some(k) = args.top_k {
        p.top_k = k;
    }
    if let Some(v) = args.top_p {
        p.top_p = v;
    }
    p
}

pub fn run(args: RunArgs) -> Result<(), String> {
    let dir = resolve_model(&args.model)?;
    let mem = MemorySampler::start(Duration::from_millis(100));
    let limits = Limits {
        ctx: args.ctx,
        max_tokens: args.chunk,
        max_logit_rows: 1,
        kv: args.kv,
    };
    eprintln!("cargando {} (contexto {}) ...", dir.display(), args.ctx);
    let mut session = Session::load(&dir, limits).map_err(|e| e.to_string())?;
    let params = sampling(&args);
    let mut sampler = Sampler::new(params, session.vocab());
    let opts = RenderOptions {
        add_generation_prompt: true,
        enable_thinking: args.no_think.then_some(false),
    };

    let mut messages: Vec<Value> = Vec::new();
    if let Some(s) = &args.system {
        messages.push(json!({"role": "system", "content": s}));
    }
    let interactive = args.prompt.is_none();
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    loop {
        let user = match &args.prompt {
            Some(p) if messages.iter().all(|m| m["role"] != "user") => p.clone(),
            Some(_) => break,
            None => {
                eprint!("\n> ");
                match lines.next() {
                    Some(Ok(l)) if l.trim() == "/salir" => break,
                    Some(Ok(l)) if l.trim().is_empty() => continue,
                    Some(Ok(l)) => l,
                    _ => break,
                }
            }
        };
        messages.push(json!({"role": "user", "content": user}));
        let text = session
            .tokenizer()
            .template
            .render(&Value::Array(messages.clone()), None, &opts)
            .map_err(|e| e.to_string())?;
        let ids = session.tokenizer().encode(&text);

        let mut reply = String::new();
        let mut out = std::io::stdout();
        let stats = session
            .generate(&ids, args.max_tokens, &mut sampler, |t| {
                reply.push_str(t);
                let _ = out.write_all(t.as_bytes());
                let _ = out.flush();
                true
            })
            .map_err(|e| e.to_string())?;
        println!();
        eprintln!(
            "[prompt {} tokens ({} reutilizados) · prefill {:.1} tok/s · TTFT {:.0} ms · {} tokens · decode {:.1} tok/s · fin: {}]",
            stats.prompt_tokens,
            stats.reused_tokens,
            stats.prefill_tok_s(),
            stats.ttft_ms,
            stats.generated,
            stats.decode_tok_s(),
            match stats.stop {
                StopReason::Stop => "fin de turno",
                StopReason::MaxTokens => "max_tokens",
                StopReason::ContextFull => "contexto lleno",
                StopReason::Cancelled => "cancelado",
            }
        );
        messages.push(json!({"role": "assistant", "content": reply}));
        if !interactive {
            break;
        }
    }

    let r = mem.finish();
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    eprintln!(
        "[memoria: pico {:.2} GiB · presión peor {:?} · swap del sistema +{} MiB · semilla {}]",
        gib(r.peak_footprint),
        r.worst_pressure,
        r.swap_growth() >> 20,
        params.seed
    );
    Ok(())
}
