//! `brasa serve [modelo]`: API local compatible con OpenAI y Anthropic (ADR 0008).

use std::net::SocketAddr;

use brasa_daemon::ServeConfig;
use brasa_runtime::{KvType, Limits};
use clap::Args;

use crate::config::{
    self, Config, DEFAULT_CTX, DEFAULT_HOST, DEFAULT_KV, DEFAULT_MODEL, DEFAULT_PORT,
};
use crate::run::resolve_model;

#[derive(Debug, Args)]
pub struct ServeArgs {
    /// Nombre del modelo (carpeta en ./models o $BRASA_MODELS) o ruta. Sin esto, el del archivo
    /// de configuración.
    model: Option<String>,
    /// Dirección a escuchar (por defecto 127.0.0.1).
    #[arg(long)]
    host: Option<String>,
    /// Puerto (por defecto 8080).
    #[arg(long)]
    port: Option<u16>,
    /// Contexto (los agentes mandan prompts de 10K–30K tokens). El planner lo verifica.
    #[arg(long)]
    ctx: Option<usize>,
    /// Tokens por bloque de prefill.
    #[arg(long, default_value_t = 512)]
    chunk: usize,
    /// Tipo de la KV cache: f16 (por defecto), q8_0 o f32 (ADR 0009).
    #[arg(long, value_parser = crate::parse_kv)]
    kv: Option<KvType>,
}

pub fn run(a: ServeArgs) -> Result<(), String> {
    let cfg = Config::load()?;
    let model = config::pick(a.model, cfg.model.clone(), DEFAULT_MODEL.to_string());
    let host = config::pick(a.host, cfg.host.clone(), DEFAULT_HOST.to_string());
    let port = config::pick(a.port, cfg.port, DEFAULT_PORT);
    let ctx = config::pick(a.ctx, cfg.ctx, DEFAULT_CTX);
    let cfg_kv = cfg
        .kv
        .as_deref()
        .map(|s| {
            KvType::parse(s).ok_or_else(|| format!("config.kv {s}: se espera f32, f16 o q8_0"))
        })
        .transpose()?;
    let kv = config::pick(
        a.kv,
        cfg_kv,
        KvType::parse(DEFAULT_KV).expect("kv por defecto"),
    );

    let dir = resolve_model(&model.value)?;
    let model_id = dir
        .file_name()
        .map_or_else(|| model.value.clone(), |n| n.to_string_lossy().into_owned());
    let addr: SocketAddr = format!("{}:{}", host.value, port.value)
        .parse()
        .map_err(|e| format!("dirección inválida: {e}"))?;
    eprintln!("cargando {} (contexto {}) ...", dir.display(), ctx.value);
    brasa_daemon::serve(ServeConfig {
        model_dir: dir,
        model_id,
        limits: Limits {
            ctx: ctx.value,
            max_tokens: a.chunk,
            max_logit_rows: 1,
            kv: kv.value,
        },
        addr,
        commit: env!("BRASA_BUILD_COMMIT").to_string(),
    })
}
