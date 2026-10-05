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
pub const DEFAULT_CTX: usize = 16384;
pub const DEFAULT_KV: &str = "f16";

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
pub struct Config {
    pub model: Option<String>,
    pub host: Option<String>,
    pub port: Option<u16>,
    pub ctx: Option<usize>,
    pub kv: Option<String>,
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

fn home() -> PathBuf {
    std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from)
}

impl Config {
    /// Ruta del archivo: `$BRASA_CONFIG`, si no `$XDG_CONFIG_HOME/brasa/config.toml`, si no
    /// `~/.config/brasa/config.toml`.
    pub fn path() -> PathBuf {
        if let Some(p) = std::env::var_os("BRASA_CONFIG") {
            return PathBuf::from(p);
        }
        let base = std::env::var_os("XDG_CONFIG_HOME")
            .map_or_else(|| home().join(".config"), PathBuf::from);
        base.join("brasa").join("config.toml")
    }

    pub fn load() -> Result<Self, String> {
        Self::load_from(&Self::path())
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
    let ctx = pick(None, cfg.ctx, DEFAULT_CTX);
    let kv = v(None, cfg.kv.clone(), DEFAULT_KV);
    if json {
        let item = |value: String, source: Source| serde_json::json!({"value": value, "source": source.as_str()});
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "path": path.display().to_string(),
                "exists": exists,
                "model": item(model.0, model.1),
                "host": item(host.0, host.1),
                "port": serde_json::json!({"value": port.value, "source": port.source.as_str()}),
                "ctx": serde_json::json!({"value": ctx.value, "source": ctx.source.as_str()}),
                "kv": item(kv.0, kv.1),
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
        println!("ctx       {:>10}   ({})", ctx.value, ctx.source.as_str());
        println!("kv        {:>10}   ({})", kv.0, kv.1.as_str());
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
            "port = 9999\nctx = 2048\nkv = \"q8_0\"\nmodel = \"otro\"\n",
        )
        .unwrap();
        let c = Config::load_from(&p).unwrap();
        assert_eq!(c.port, Some(9999));
        assert_eq!(c.ctx, Some(2048));
        assert_eq!(c.kv.as_deref(), Some("q8_0"));
        assert_eq!(c.model.as_deref(), Some("otro"));
        let missing = Config::load_from(&dir.join("nope.toml")).unwrap();
        assert_eq!(missing, Config::default());
        std::fs::remove_dir_all(&dir).ok();
    }
}
