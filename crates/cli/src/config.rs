//! Configuración del usuario en `~/.config/brasa/config.toml` (U5). Los flags mandan sobre el
//! archivo, y el archivo sobre los valores por defecto. `brasa config show` muestra la
//! configuración efectiva y de dónde salió cada valor.

use std::path::{Path, PathBuf};

use clap::{Args, Subcommand};
use serde::Deserialize;

/// Valores por defecto.
pub const DEFAULT_MODEL: &str = "qwen3-4b-q4";
pub const DEFAULT_HOST: &str = "127.0.0.1";
pub const DEFAULT_PORT: u16 = 8080;
/// Contexto por defecto de `serve` (agentes con prompts largos).
pub const DEFAULT_SERVE_CTX: usize = 16384;
/// Contexto por defecto de `run` (chat interactivo).
pub const DEFAULT_RUN_CTX: usize = 4096;
pub const DEFAULT_KV: &str = "f16";

/// Perfil de memoria de la línea de comandos (`--perfil 8gb|16gb`, ADR 0007 y 0029).
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Perfil {
    #[value(name = "8gb")]
    G8,
    #[value(name = "16gb")]
    G16,
}

impl Perfil {
    pub fn profile(self) -> brasa_memory::planner::Profile {
        match self {
            Perfil::G8 => brasa_memory::planner::Profile::G8,
            Perfil::G16 => brasa_memory::planner::Profile::G16,
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub model: Option<String>,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub kv: Option<String>,
    /// Carpeta de modelos (ADR 0031). Ruta absoluta o con `~/`.
    pub models_dir: Option<String>,
    #[serde(default)]
    pub run: Section,
    #[serde(default)]
    pub serve: Section,
    #[serde(default)]
    pub storage: StorageSection,
}

/// Sección `[storage]` (ADR 0034). Sin valores, los de `brasa_catalog::storage::Settings`.
///
/// ```toml
/// [storage]
/// reserve_gib = 2            # GiB del volumen que una descarga no puede usar
/// partial_max_age_days = 7   # una descarga a medias con más días es vieja (`brasa storage clean`)
/// ```
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StorageSection {
    pub reserve_gib: Option<f64>,
    pub partial_max_age_days: Option<u64>,
}

/// Sección por subcomando (`[run]`, `[serve]`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Section {
    pub ctx: Option<usize>,
}

/// De dónde salió un valor efectivo.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Flag,
    File,
    Default,
}

impl Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Source::Flag => "flag",
            Source::File => "archivo",
            Source::Default => "defecto",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Value<T> {
    pub value: T,
    pub source: Source,
}

/// Precedencia: flag > archivo > defecto.
pub fn pick<T>(flag: Option<T>, file: Option<T>, default: T) -> Value<T> {
    match (flag, file) {
        (Some(value), _) => Value {
            value,
            source: Source::Flag,
        },
        (None, Some(value)) => Value {
            value,
            source: Source::File,
        },
        (None, None) => Value {
            value: default,
            source: Source::Default,
        },
    }
}

/// Contexto efectivo de `serve`: flag > `[serve] ctx` > defecto.
pub fn serve_ctx(flag: Option<usize>, cfg: &Config) -> Value<usize> {
    pick(flag, cfg.serve.ctx, DEFAULT_SERVE_CTX)
}

/// Contexto efectivo de `run`: flag > `[run] ctx` > defecto.
pub fn run_ctx(flag: Option<usize>, cfg: &Config) -> Value<usize> {
    pick(flag, cfg.run.ctx, DEFAULT_RUN_CTX)
}

impl Config {
    /// Ruta del archivo: `$BRASA_CONFIG`, si no `$XDG_CONFIG_HOME/brasa/config.toml`, si no
    /// `~/.config/brasa/config.toml`.
    pub fn path() -> PathBuf {
        brasa_catalog::dirs::config_path()
    }

    pub fn load() -> Result<Self, String> {
        Self::load_from(&Self::path())
    }

    /// Ajustes de almacenamiento validados (`[storage]`, ADR 0034).
    pub fn storage(&self) -> Result<brasa_catalog::storage::Settings, String> {
        brasa_catalog::storage::Settings::new(
            self.storage.reserve_gib,
            self.storage.partial_max_age_days,
        )
        .map_err(|e| format!("{}: {e}", Self::path().display()))
    }

    /// Carga el archivo; su ausencia no es error (queda todo por defecto).
    pub fn load_from(path: &Path) -> Result<Self, String> {
        match std::fs::read_to_string(path) {
            Ok(text) => toml::from_str(&text).map_err(|e| format!("{}: {e}", path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(format!("{}: {e}", path.display())),
        }
    }
}

#[derive(Debug, Args)]
pub struct ConfigArgs {
    #[command(subcommand)]
    cmd: Option<ConfigCmd>,
}

#[derive(Debug, Subcommand)]
enum ConfigCmd {
    /// Muestra la configuración efectiva y de dónde sale cada valor.
    Show {
        #[arg(long)]
        json: bool,
    },
}

pub fn run(a: ConfigArgs) -> Result<(), String> {
    match a.cmd {
        None | Some(ConfigCmd::Show { json: false }) => show(false),
        Some(ConfigCmd::Show { json: true }) => show(true),
    }
}

fn show(json: bool) -> Result<(), String> {
    let path = Config::path();
    let cfg = Config::load_from(&path)?;
    let exists = path.is_file();
    let v = |flag: Option<String>, file: Option<String>, default: &str| {
        let r = pick(flag, file, default.to_string());
        (r.value, r.source)
    };
    let model = v(None, cfg.model.clone(), DEFAULT_MODEL);
    let host = v(None, cfg.host.clone(), DEFAULT_HOST);
    let port = pick(None, cfg.port, DEFAULT_PORT);
    let ctx_run = run_ctx(None, &cfg);
    let ctx_serve = serve_ctx(None, &cfg);
    let kv = v(None, cfg.kv.clone(), DEFAULT_KV);
    let md = brasa_catalog::dirs::models_dir(None).map_err(|e| e.0)?;
    let md_path = md.path.display().to_string();
    // `serve` toma la KV por defecto del perfil de memoria de esta Mac (ADR 0029).
    let profile = brasa_memory::planner::Profile::this_machine();
    let kv_serve = v(None, cfg.kv.clone(), profile.agent_kv());
    let st = cfg.storage()?;
    let src = |set: bool| if set { Source::File } else { Source::Default };
    let reserve_src = src(cfg.storage.reserve_gib.is_some());
    let age_src = src(cfg.storage.partial_max_age_days.is_some());
    if json {
        let item = |value: String, source: Source| serde_json::json!({"value": value, "source": source.as_str()});
        let n = |value: usize, source: Source| serde_json::json!({"value": value, "source": source.as_str()});
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "path": path.display().to_string(),
                "exists": exists,
                "model": item(model.0, model.1),
                "host": item(host.0, host.1),
                "port": serde_json::json!({"value": port.value, "source": port.source.as_str()}),
                "run": {"ctx": n(ctx_run.value, ctx_run.source)},
                "serve": {
                    "ctx": n(ctx_serve.value, ctx_serve.source),
                    "kv": item(kv_serve.0, kv_serve.1),
                    "perfil": profile.name(),
                },
                "kv": item(kv.0, kv.1),
                "models_dir": serde_json::json!({
                    "value": md_path,
                    "source": md.source.as_str(),
                }),
                "storage": {
                    "reserve_bytes": {"value": st.reserve_bytes, "source": reserve_src.as_str()},
                    "partial_max_age_days": {
                        "value": st.partial_max_age_days,
                        "source": age_src.as_str(),
                    },
                },
            }))
            .unwrap()
        );
    } else {
        println!(
            "archivo   {} ({})",
            path.display(),
            if exists { "existe" } else { "no existe" }
        );
        println!("model     {:>10}   ({})", model.0, model.1.as_str());
        println!("host      {:>10}   ({})", host.0, host.1.as_str());
        println!("port      {:>10}   ({})", port.value, port.source.as_str());
        println!(
            "run ctx   {:>10}   ({})",
            ctx_run.value,
            ctx_run.source.as_str()
        );
        println!(
            "serve ctx {:>10}   ({})",
            ctx_serve.value,
            ctx_serve.source.as_str()
        );
        println!("kv        {:>10}   ({})", kv.0, kv.1.as_str());
        println!("models_dir {md_path}   ({})", md.source.as_str());
        println!(
            "serve kv  {:>10}   ({}; perfil {})",
            kv_serve.0,
            kv_serve.1.as_str(),
            profile.name()
        );
        // `[storage]` (ADR 0034): reserva que una descarga no usa y umbral de `storage clean`.
        println!(
            "reserva   {:>10}   ({}; [storage] reserve_gib)",
            format!("{:.2} GiB", st.reserve_gib()),
            reserve_src.as_str()
        );
        println!(
            "parciales {:>10}   ({}; [storage] partial_max_age_days)",
            format!("{} días", st.partial_max_age_days),
            age_src.as_str()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precedencia_flag_archivo_defecto() {
        // flag manda
        let r = pick(Some(9u16), Some(1u16), 8080);
        assert_eq!((r.value, r.source), (9, Source::Flag));
        // sin flag, manda el archivo
        let r = pick(None, Some(1u16), 8080);
        assert_eq!((r.value, r.source), (1, Source::File));
        // sin flag ni archivo, el defecto
        let r = pick(None, None, 8080);
        assert_eq!((r.value, r.source), (8080, Source::Default));
    }

    #[test]
    fn carga_archivo_y_ausencia_es_defecto() {
        let dir = std::env::temp_dir().join(format!("brasa-config-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("config.toml");
        std::fs::write(
            &p,
            "port = 9999\nkv = \"q8_0\"\nmodel = \"otro\"\n[run]\nctx = 2048\n[serve]\nctx = 8192\n",
        )
        .unwrap();
        let c = Config::load_from(&p).unwrap();
        assert_eq!(c.port, Some(9999));
        assert_eq!(c.run.ctx, Some(2048));
        assert_eq!(c.serve.ctx, Some(8192));
        assert_eq!(c.kv.as_deref(), Some("q8_0"));
        assert_eq!(c.model.as_deref(), Some("otro"));
        let missing = Config::load_from(&dir.join("nope.toml")).unwrap();
        assert_eq!(missing, Config::default());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn las_claves_coinciden_con_las_del_catalogo() {
        // `brasa_catalog::dirs::save_models_dir` se niega a tocar archivos con claves que no
        // conoce: su lista tiene que ser exactamente la de `Config`.
        let text = "model = \"x\"\nhost = \"x\"\nport = 1\nkv = \"x\"\nmodels_dir = \"x\"\n\
                    [run]\nctx = 1\n[serve]\nctx = 1\n\
                    [storage]\nreserve_gib = 1.5\npartial_max_age_days = 3\n";
        let c: Config = toml::from_str(text).unwrap();
        assert_eq!(c.models_dir.as_deref(), Some("x"));
        assert_eq!(c.storage.reserve_gib, Some(1.5));
        assert_eq!(c.storage.partial_max_age_days, Some(3));
        let table: toml::Table = toml::from_str(text).unwrap();
        let mut keys: Vec<&str> = table.keys().map(String::as_str).collect();
        let mut want = brasa_catalog::dirs::CONFIG_KEYS.to_vec();
        keys.sort_unstable();
        want.sort_unstable();
        assert_eq!(keys, want);
    }

    #[test]
    fn storage_tiene_defectos_y_valida() {
        let c = Config::default();
        let s = c.storage().unwrap();
        assert_eq!(s, brasa_catalog::storage::Settings::default());
        assert_eq!(s.reserve_bytes, 2 << 30);
        assert_eq!(s.partial_max_age_days, 7);
        let c: Config = toml::from_str("[storage]\nreserve_gib = 0\n").unwrap();
        assert_eq!(c.storage().unwrap().reserve_bytes, 0);
        // Un entero también vale como GiB.
        let c: Config = toml::from_str("[storage]\nreserve_gib = 4\n").unwrap();
        assert_eq!(c.storage().unwrap().reserve_bytes, 4 << 30);
        for malo in [
            "[storage]\nreserve_gib = -1\n",
            "[storage]\npartial_max_age_days = 0\n",
        ] {
            let c: Config = toml::from_str(malo).unwrap();
            assert!(c.storage().is_err(), "{malo}");
        }
        // Un error de tipeo en la sección falla, no se ignora.
        assert!(toml::from_str::<Config>("[storage]\nreserva = 1\n").is_err());
    }

    #[test]
    fn un_campo_desconocido_falla_con_mensaje() {
        let dir = std::env::temp_dir().join(format!("brasa-config-unknown-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("config.toml");
        // `contx` es un error de tipeo de `ctx`: tiene que fallar, no ignorarse.
        std::fs::write(&p, "[serve]\ncontx = 8192\n").unwrap();
        let e = Config::load_from(&p).unwrap_err();
        assert!(e.contains("contx"), "{e}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn serve_y_run_usan_la_precedencia() {
        let dir =
            std::env::temp_dir().join(format!("brasa-config-precedencia-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("config.toml");
        std::fs::write(&p, "[run]\nctx = 2048\n[serve]\nctx = 8192\n").unwrap();
        let cfg = Config::load_from(&p).unwrap();
        // Sin archivo, cada subcomando tiene su propio defecto real.
        let vacio = Config::default();
        assert_eq!(
            (run_ctx(None, &vacio).value, run_ctx(None, &vacio).source),
            (DEFAULT_RUN_CTX, Source::Default)
        );
        assert_eq!(
            (
                serve_ctx(None, &vacio).value,
                serve_ctx(None, &vacio).source
            ),
            (DEFAULT_SERVE_CTX, Source::Default)
        );
        assert_ne!(DEFAULT_RUN_CTX, DEFAULT_SERVE_CTX);
        // El archivo manda sobre el defecto, y el flag sobre el archivo.
        assert_eq!(
            (run_ctx(None, &cfg).value, run_ctx(None, &cfg).source),
            (2048, Source::File)
        );
        assert_eq!(
            (
                serve_ctx(Some(4096), &cfg).value,
                serve_ctx(Some(4096), &cfg).source
            ),
            (4096, Source::Flag)
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
