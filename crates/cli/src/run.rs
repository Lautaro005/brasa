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
    /// Nombre del modelo (carpeta en ./models o $BRASA_MODELS) o ruta a su carpeta. Sin esto, el
    /// del archivo de configuración.
    model: Option<String>,
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
    #[arg(long)]
    ctx: Option<usize>,
    /// Tokens por bloque de prefill.
    #[arg(long, default_value_t = brasa_runtime::DEFAULT_CHUNK)]
    chunk: usize,
    /// Tipo de la KV cache: f16 (por defecto), q8_0 o f32 (ADR 0009).
    #[arg(long, value_parser = crate::parse_kv)]
    kv: Option<brasa_runtime::KvType>,
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

/// Carpeta base de modelos: `$BRASA_MODELS` o `./models`.
pub fn models_dir() -> PathBuf {
    PathBuf::from(std::env::var("BRASA_MODELS").unwrap_or_else(|_| "models".into()))
}

pub fn resolve_model(name: &str) -> Result<PathBuf, String> {
    let direct = Path::new(name);
    if direct.join("model.brasa").exists() {
        return Ok(direct.to_path_buf());
    }
    let base = models_dir();
    let p = base.join(name);
    if p.join("model.brasa").exists() {
        return Ok(p);
    }
    // La carpeta de safetensors de Hugging Face sale del manifiesto si existe.
    let hf = brasa_catalog::manifest::Manifest::find(name)
        .map_or_else(|_| format!("{name}-hf"), |m| m.hf_dir);
    Err(format!(
        "no se encontró el modelo {name:?} (buscado en {} y como ruta).\n\
         Sugerencia: corré `brasa models` para ver los locales, o `brasa pull {name}` y \
         `brasa convert {} {}` para bajarlo y convertirlo.",
        p.display(),
        base.join(&hf).display(),
        base.join(name).display()
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

/// Valores efectivos de `run`: flag > archivo > defecto, con su origen.
#[derive(Debug, Clone)]
pub struct Effective {
    pub model: crate::config::Value<String>,
    pub ctx: crate::config::Value<usize>,
    pub kv: crate::config::Value<brasa_runtime::KvType>,
}

/// Resuelve la configuración efectiva de `run` sin cargar nada (función pura).
pub fn effective(args: &RunArgs, cfg: &crate::config::Config) -> Result<Effective, String> {
    let model = crate::config::pick(
        args.model.clone(),
        cfg.model.clone(),
        crate::config::DEFAULT_MODEL.to_string(),
    );
    let ctx = crate::config::run_ctx(args.ctx, cfg);
    let cfg_kv = cfg.kv.as_deref().map(crate::parse_kv).transpose()?;
    let kv = crate::config::pick(
        args.kv,
        cfg_kv,
        brasa_runtime::KvType::parse(crate::config::DEFAULT_KV).expect("kv por defecto"),
    );
    Ok(Effective { model, ctx, kv })
}

pub fn run(args: RunArgs) -> Result<(), String> {
    let cfg = crate::config::Config::load()?;
    let e = effective(&args, &cfg)?;
    let (model, ctx, kv) = (e.model, e.ctx, e.kv);
    let dir = resolve_model(&model.value)?;
    let mem = MemorySampler::start(Duration::from_millis(100));
    let limits = Limits {
        ctx: ctx.value,
        max_tokens: args.chunk,
        max_logit_rows: 1,
        kv: kv.value,
    };
    eprintln!("cargando {} (contexto {}) ...", dir.display(), ctx.value);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, Section, Source};

    fn args() -> RunArgs {
        RunArgs {
            model: None,
            prompt: None,
            system: None,
            no_think: false,
            max_tokens: 1024,
            chunk: brasa_runtime::DEFAULT_CHUNK,
            ctx: None,
            kv: None,
            greedy: false,
            temp: None,
            top_k: None,
            top_p: None,
            seed: None,
        }
    }

    #[test]
    fn effective_usa_archivo_y_flags() {
        let cfg = Config {
            kv: Some("q8_0".into()),
            run: Section { ctx: Some(2048) },
            ..Config::default()
        };
        // Sin flags: mandan el archivo y su [run] ctx.
        let e = effective(&args(), &cfg).unwrap();
        assert_eq!((e.ctx.value, e.ctx.source), (2048, Source::File));
        assert_eq!(e.kv.value.name(), "q8_0");
        assert_eq!(
            (e.model.value.as_str(), e.model.source),
            ("qwen3-4b-q4", Source::Default)
        );

        // Con flag: manda el flag.
        let mut a = args();
        a.ctx = Some(4096);
        let e = effective(&a, &cfg).unwrap();
        assert_eq!((e.ctx.value, e.ctx.source), (4096, Source::Flag));
    }
}
