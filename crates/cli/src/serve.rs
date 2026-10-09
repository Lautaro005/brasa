//! `brasa serve [modelo]`: API local compatible con OpenAI y Anthropic (ADR 0008).

use std::net::SocketAddr;

use brasa_daemon::ServeConfig;
use brasa_memory::planner::{Budget, Profile, simulated_budget};
use brasa_runtime::{KvType, Limits};
use clap::Args;

use crate::config::{self, Config, DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_PORT, Perfil};
use crate::run::resolve_model;

#[derive(Debug, Args)]
pub struct ServeArgs {
    /// Nombre del modelo (carpeta dentro de la carpeta de modelos; ver `brasa config show`) o
    /// ruta. Sin esto, el del archivo de configuración.
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
    /// Tipo de la KV cache: f32, f16 o q8_0 (ADR 0009). Por defecto, el del perfil: q8_0 en
    /// 8 GB, f16 en 16 GB.
    #[arg(long, value_parser = crate::parse_kv)]
    kv: Option<KvType>,
    /// Perfil de memoria: fija la KV por defecto y simula el presupuesto de esa Mac (el menor
    /// entre el del perfil y el de esta). Sin esto, el perfil de esta Mac según su RAM.
    #[arg(long, value_enum)]
    perfil: Option<Perfil>,
}

/// Valores efectivos de `serve`: flag > archivo > defecto, con su origen.
#[derive(Debug, Clone)]
pub struct Effective {
    pub model: config::Value<String>,
    pub host: config::Value<String>,
    pub port: config::Value<u16>,
    pub ctx: config::Value<usize>,
    pub kv: config::Value<KvType>,
    /// Perfil de memoria efectivo y si vino de `--perfil` (si no, de la RAM de esta Mac).
    pub profile: config::Value<Profile>,
}

/// Resuelve la configuración efectiva de `serve` sin cargar nada. `machine` es el perfil de esta
/// Mac (se pasa para que la función sea pura).
pub fn effective(a: &ServeArgs, cfg: &Config, machine: Profile) -> Result<Effective, String> {
    let profile = config::pick(a.perfil.map(Perfil::profile), None, machine);
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
        KvType::parse(profile.value.agent_kv()).expect("kv del perfil"),
    );
    Ok(Effective {
        model,
        host,
        port,
        ctx,
        kv,
        profile,
    })
}

pub fn run(a: ServeArgs) -> Result<(), String> {
    let cfg = Config::load()?;
    let e = effective(&a, &cfg, Profile::this_machine())?;
    let (model, host, port, ctx, kv) = (e.model, e.host, e.port, e.ctx, e.kv);
    // Con --perfil, el planner usa el presupuesto simulado; sin él, el de esta máquina.
    let budget = (e.profile.source == config::Source::Flag)
        .then(|| simulated_budget(e.profile.value, Budget::this_machine()));

    let dir = resolve_model(&model.value)?;
    let model_id = dir
        .file_name()
        .map_or_else(|| model.value.clone(), |n| n.to_string_lossy().into_owned());
    let addr: SocketAddr = format!("{}:{}", host.value, port.value)
        .parse()
        .map_err(|e| format!("dirección inválida: {e}"))?;
    eprintln!(
        "cargando {} (contexto {}, KV {}, perfil {}{}) ...",
        dir.display(),
        ctx.value,
        kv.value.name(),
        e.profile.value.name(),
        match &budget {
            Some(b) => format!("; presupuesto {}", b.source),
            None => String::new(),
        }
    );
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
        models_dir: brasa_catalog::dirs::models_dir(None).map_err(|e| e.0)?,
        config_path: brasa_catalog::dirs::config_path(),
        // Un espejo o un servidor de prueba (ADR 0031); por defecto, Hugging Face.
        hf_endpoint: std::env::var("BRASA_HF_ENDPOINT")
            .ok()
            .filter(|e| !e.is_empty())
            .unwrap_or_else(|| brasa_catalog::pull::HF_ENDPOINT.to_string()),
        budget,
        storage: cfg.storage()?,
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
            perfil: None,
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
        let e = effective(&args(None), &cfg, Profile::G16).unwrap();
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
        let e = effective(&a, &cfg, Profile::G16).unwrap();
        assert_eq!((e.port.value, e.port.source), (8080, Source::Flag));
        assert_eq!((e.ctx.value, e.ctx.source), (4096, Source::Flag));
    }

    #[test]
    fn perfil_fija_la_kv_por_defecto_del_modo_agente() {
        let vacio = Config::default();
        // 8 GB (por RAM o por --perfil): 16K con KV Q8 (CLAUDE.md, perfil de agente).
        let e = effective(&args(None), &vacio, Profile::G8).unwrap();
        assert_eq!((e.ctx.value, e.kv.value.name()), (16384, "q8_0"));
        assert_eq!(
            (e.profile.value, e.profile.source),
            (Profile::G8, Source::Default)
        );
        let mut a = args(None);
        a.perfil = Some(Perfil::G8);
        let e = effective(&a, &vacio, Profile::G16).unwrap();
        assert_eq!((e.ctx.value, e.kv.value.name()), (16384, "q8_0"));
        assert_eq!(
            (e.profile.value, e.profile.source),
            (Profile::G8, Source::Flag)
        );
        // 16 GB: 16K con KV f16.
        let e = effective(&args(None), &vacio, Profile::G16).unwrap();
        assert_eq!((e.ctx.value, e.kv.value.name()), (16384, "f16"));
        // --kv y el archivo mandan sobre el perfil.
        a.kv = KvType::parse("f16");
        let e = effective(&a, &vacio, Profile::G16).unwrap();
        assert_eq!((e.kv.value.name(), e.kv.source), ("f16", Source::Flag));
        let cfg = Config {
            kv: Some("f32".into()),
            ..Config::default()
        };
        let e = effective(&args(None), &cfg, Profile::G8).unwrap();
        assert_eq!((e.kv.value.name(), e.kv.source), ("f32", Source::File));
    }
}
