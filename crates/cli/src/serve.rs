//! `brasa serve [modelo]`: API local compatible con OpenAI y Anthropic (ADR 0008).

use std::net::SocketAddr;

use brasa_daemon::ServeConfig;
use brasa_runtime::{KvType, Limits};
use clap::Args;

use crate::config::{self, Config, DEFAULT_HOST, DEFAULT_KV, DEFAULT_MODEL, DEFAULT_PORT};
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
    #[arg(long, default_value_t = brasa_runtime::DEFAULT_CHUNK)]
    chunk: usize,
    /// Tipo de la KV cache: f16 (por defecto), q8_0 o f32 (ADR 0009).
    #[arg(long, value_parser = crate::parse_kv)]
    kv: Option<KvType>,
}

/// Valores efectivos de `serve`: flag > archivo > defecto, con su origen.
#[derive(Debug, Clone)]
pub struct Effective {
    pub model: config::Value<String>,
    pub host: config::Value<String>,
    pub port: config::Value<u16>,
    pub ctx: config::Value<usize>,
    pub kv: config::Value<KvType>,
}

/// Resuelve la configuración efectiva de `serve` sin cargar nada (función pura).
pub fn effective(a: &ServeArgs, cfg: &Config) -> Result<Effective, String> {
    let model = config::pick(
        a.model.clone(),
        cfg.model.clone(),
        DEFAULT_MODEL.to_string(),
    );
    let host = config::pick(a.host.clone(), cfg.host.clone(), DEFAULT_HOST.to_string());
    let port = config::pick(a.port, cfg.port, DEFAULT_PORT);
    let ctx = config::serve_ctx(a.ctx, cfg);
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
    Ok(Effective {
        model,
        host,
        port,
        ctx,
        kv,
    })
}

pub fn run(a: ServeArgs) -> Result<(), String> {
    let cfg = Config::load()?;
    let e = effective(&a, &cfg)?;
    let (model, host, port, ctx, kv) = (e.model, e.host, e.port, e.ctx, e.kv);

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Section, Source};

    fn args(ctx: Option<usize>) -> ServeArgs {
        ServeArgs {
            model: None,
            host: None,
            port: None,
            ctx,
            chunk: brasa_runtime::DEFAULT_CHUNK,
            kv: None,
        }
    }

    #[test]
    fn effective_usa_archivo_y_flags() {
        let cfg = Config {
            port: Some(9123),
            kv: Some("q8_0".into()),
            serve: Section { ctx: Some(8192) },
            ..Config::default()
        };
        // Sin flags: mandan el archivo y su [serve] ctx.
        let e = effective(&args(None), &cfg).unwrap();
        assert_eq!((e.port.value, e.port.source), (9123, Source::File));
        assert_eq!((e.ctx.value, e.ctx.source), (8192, Source::File));
        assert_eq!(e.kv.value.name(), "q8_0");
        assert_eq!(
            (e.model.value.as_str(), e.model.source),
            ("qwen3-4b-q4", Source::Default)
        );

        // Con flags: mandan los flags.
        let mut a = args(Some(4096));
        a.port = Some(8080);
        let e = effective(&a, &cfg).unwrap();
        assert_eq!((e.port.value, e.port.source), (8080, Source::Flag));
        assert_eq!((e.ctx.value, e.ctx.source), (4096, Source::Flag));
    }
}
