//! `brasa serve <modelo>`: API local compatible con OpenAI y Anthropic (ADR 0008).

use std::net::SocketAddr;

use brasa_daemon::ServeConfig;
use brasa_runtime::Limits;
use clap::Args;

use crate::run::resolve_model;

#[derive(Debug, Args)]
pub struct ServeArgs {
    /// Nombre del modelo (carpeta en ./models o $BRASA_MODELS) o ruta.
    model: String,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Contexto (los agentes mandan prompts de 10K–30K tokens). El planner lo verifica.
    #[arg(long, default_value_t = 16384)]
    ctx: usize,
    /// Tokens por bloque de prefill.
    #[arg(long, default_value_t = 512)]
    chunk: usize,
    /// Tipo de la KV cache: f16 (por defecto) o f32 (ADR 0009).
    #[arg(long, default_value = "f16", value_parser = crate::parse_kv)]
    kv: brasa_runtime::KvType,
}

pub fn run(a: ServeArgs) -> Result<(), String> {
    let dir = resolve_model(&a.model)?;
    let model_id = dir
        .file_name()
        .map_or_else(|| a.model.clone(), |n| n.to_string_lossy().into_owned());
    let addr: SocketAddr = format!("{}:{}", a.host, a.port)
        .parse()
        .map_err(|e| format!("dirección inválida: {e}"))?;
    eprintln!("cargando {} (contexto {}) ...", dir.display(), a.ctx);
    brasa_daemon::serve(ServeConfig {
        model_dir: dir,
        model_id,
        limits: Limits {
            ctx: a.ctx,
            max_tokens: a.chunk,
            max_logit_rows: 1,
            kv: a.kv,
        },
        addr,
    })
}
