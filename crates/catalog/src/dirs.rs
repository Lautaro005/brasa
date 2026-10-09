//! Carpeta de modelos y archivo de configuración (ADR 0031).
//!
//! Precedencia de la carpeta de modelos: flag `--dir` > `$BRASA_MODELS` > `models_dir` del archivo
//! de configuración > `./models` si existe (checkout de desarrollo) > predeterminado
//! `~/Library/Application Support/brasa/models`. La resuelven igual `serve`, `run`, `pull`, `rm`,
//! `models` y `/api/models`.
//!
//! El archivo de configuración (ADR 0022) lo lee `brasa-cli`; acá solo se lee la clave
//! `models_dir` y se reescribe esa línea ([`save_models_dir`]) sin tocar el resto.

use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

use crate::{Error, Result};

/// De dónde sale la carpeta de modelos.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirSource {
    Flag,
    Env,
    File,
    Checkout,
    Default,
}

impl DirSource {
    pub fn as_str(self) -> &'static str {
        match self {
            DirSource::Flag => "flag",
            DirSource::Env => "env",
            DirSource::File => "archivo",
            DirSource::Checkout => "checkout",
            DirSource::Default => "defecto",
        }
    }

    /// Explicación corta para la GUI y `brasa config show`.
    pub fn describe(self) -> &'static str {
        match self {
            DirSource::Flag => "flag --dir",
            DirSource::Env => "variable BRASA_MODELS",
            DirSource::File => "models_dir del archivo de configuración",
            DirSource::Checkout => "./models del checkout de desarrollo",
            DirSource::Default => "carpeta predeterminada",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelsDir {
    pub path: PathBuf,
    pub source: DirSource,
}

/// Claves de primer nivel que entiende el archivo de configuración (ADR 0022 + `models_dir`).
/// `brasa-cli` tiene un test que verifica que coincidan con su `Config`.
pub const CONFIG_KEYS: &[&str] = &["model", "host", "port", "kv", "models_dir", "run", "serve"];

fn home() -> PathBuf {
    std::env::var_os("HOME").map_or_else(|| PathBuf::from("."), PathBuf::from)
}

/// Ruta del archivo de configuración: `$BRASA_CONFIG`, si no `$XDG_CONFIG_HOME/brasa/config.toml`,
/// si no `~/.config/brasa/config.toml`.
pub fn config_path() -> PathBuf {
    if let Some(p) = std::env::var_os("BRASA_CONFIG") {
        return PathBuf::from(p);
    }
    let base =
        std::env::var_os("XDG_CONFIG_HOME").map_or_else(|| home().join(".config"), PathBuf::from);
    base.join("brasa").join("config.toml")
}

/// Carpeta predeterminada: `~/Library/Application Support/brasa/models`.
pub fn default_models_dir(home: &Path) -> PathBuf {
    home.join("Library/Application Support/brasa/models")
}

/// Expande `~/` y exige una ruta absoluta (un `models_dir` relativo dependería del cwd).
pub fn expand(value: &str, home: &Path) -> Result<PathBuf> {
    let p = if value == "~" {
        home.to_path_buf()
    } else if let Some(rest) = value.strip_prefix("~/") {
        home.join(rest)
    } else {
        PathBuf::from(value)
    };
    if !p.is_absolute() {
        return Err(Error(format!(
            "models_dir {value:?} tiene que ser una ruta absoluta (o empezar con ~/)"
        )));
    }
    Ok(p)
}

/// Resolución pura (sin leer el entorno): ver el comentario del módulo.
pub fn resolve(
    flag: Option<PathBuf>,
    env: Option<OsString>,
    file: Option<&str>,
    checkout_models: Option<PathBuf>,
    home: &Path,
) -> Result<ModelsDir> {
    let (path, source) = if let Some(p) = flag {
        (p, DirSource::Flag)
    } else if let Some(e) = env.filter(|e| !e.is_empty()) {
        (PathBuf::from(e), DirSource::Env)
    } else if let Some(f) = file {
        (expand(f, home)?, DirSource::File)
    } else if let Some(c) = checkout_models {
        (c, DirSource::Checkout)
    } else {
        (default_models_dir(home), DirSource::Default)
    };
    Ok(ModelsDir { path, source })
}

/// Lee `models_dir` del archivo de configuración (`None` si el archivo o la clave no están).
pub fn read_models_dir(config: &Path) -> Result<Option<String>> {
    let text = match std::fs::read_to_string(config) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(Error(format!("{}: {e}", config.display()))),
    };
    let table: toml::Table =
        toml::from_str(&text).map_err(|e| Error(format!("{}: {e}", config.display())))?;
    match table.get("models_dir") {
        None => Ok(None),
        Some(toml::Value::String(s)) => Ok(Some(s.clone())),
        Some(_) => Err(Error(format!(
            "{}: models_dir tiene que ser un texto",
            config.display()
        ))),
    }
}

/// Carpeta efectiva para este proceso: `flag` > entorno > archivo > `./models` > defecto.
pub fn models_dir(flag: Option<PathBuf>) -> Result<ModelsDir> {
    let file = read_models_dir(&config_path())?;
    let checkout = Path::new("models");
    let checkout = checkout.is_dir().then(|| checkout.to_path_buf());
    resolve(
        flag,
        std::env::var_os("BRASA_MODELS"),
        file.as_deref(),
        checkout,
        &home(),
    )
}

/// Bytes libres para el usuario en el volumen de `path` (`statvfs`), o `None` si no se puede leer.
pub fn free_bytes(path: &Path) -> Option<u64> {
    volume_bytes(path).map(|(_, free)| free)
}

/// Tamaño total y bytes libres para el usuario del volumen de `path` (`statvfs`), o `None` si no
/// se puede leer (por ejemplo, si `path` no existe).
pub fn volume_bytes(path: &Path) -> Option<(u64, u64)> {
    use std::os::unix::ffi::OsStrExt;
    let c = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    // SAFETY: `st` es un struct plano que `statvfs` llena; `c` es una cadena C válida.
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statvfs(c.as_ptr(), &raw mut st) };
    (rc == 0).then(|| {
        (
            u64::from(st.f_blocks) * st.f_frsize,
            u64::from(st.f_bavail) * st.f_frsize,
        )
    })
}

/// Valida una carpeta de modelos nueva: absoluta, sin `..`; la crea si falta y comprueba que se
/// pueda escribir. Devuelve la ruta normalizada (sin `.` ni barras finales).
pub fn prepare_models_dir(raw: &str) -> Result<PathBuf> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err(Error("la ruta está vacía".into()));
    }
    if raw.contains('\0') {
        return Err(Error("la ruta tiene un carácter nulo".into()));
    }
    let p = Path::new(raw);
    if !p.is_absolute() {
        return Err(Error(format!("{raw:?}: tiene que ser una ruta absoluta")));
    }
    let mut norm = PathBuf::new();
    for c in p.components() {
        match c {
            Component::ParentDir => {
                return Err(Error(format!("{raw:?}: no se permite `..`")));
            }
            Component::CurDir => {}
            c => norm.push(c),
        }
    }
    std::fs::create_dir_all(&norm).map_err(|e| Error(format!("{}: {e}", norm.display())))?;
    if !norm.is_dir() {
        return Err(Error(format!("{}: no es una carpeta", norm.display())));
    }
    let probe = norm.join(format!(".brasa-escritura-{}", std::process::id()));
    std::fs::write(&probe, b"")
        .map_err(|e| Error(format!("{}: no se puede escribir ({e})", norm.display())))?;
    std::fs::remove_file(&probe).ok();
    Ok(norm)
}

/// Guarda `models_dir = "<dir>"` en el archivo de configuración sin tocar el resto: reemplaza la
/// línea si existe o la agrega antes de la primera sección. Comentarios y orden se conservan.
///
/// No reescribe a ciegas: si el archivo no es TOML válido, tiene claves que no conoce
/// ([`CONFIG_KEYS`]) o `models_dir` no está en una sola línea simple, devuelve un error y no lo
/// toca. Después de editar, vuelve a parsear y exige que todo lo demás quede igual.
pub fn save_models_dir(config: &Path, dir: &Path) -> Result<()> {
    let value = dir
        .to_str()
        .ok_or_else(|| Error(format!("{}: la ruta no es UTF-8", dir.display())))?;
    let line = format!("models_dir = {}", toml::Value::String(value.to_string()));
    let err = |msg: String| {
        Error(format!(
            "{}: {msg}; no se modificó el archivo",
            config.display()
        ))
    };
    let text = match std::fs::read_to_string(config) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(err(e.to_string())),
    };
    let old: toml::Table =
        toml::from_str(&text).map_err(|e| err(format!("no es TOML válido ({e})")))?;
    if let Some(k) = old.keys().find(|k| !CONFIG_KEYS.contains(&k.as_str())) {
        return Err(err(format!(
            "tiene la clave {k:?}, que brasa no conoce: editalo a mano"
        )));
    }

    let lines: Vec<&str> = text.lines().collect();
    // Primer nivel: hasta el primer encabezado de sección.
    let top_end = lines
        .iter()
        .position(|l| l.trim_start().starts_with('['))
        .unwrap_or(lines.len());
    let is_key = |l: &str| {
        l.trim_start()
            .strip_prefix("models_dir")
            .is_some_and(|r| r.trim_start().starts_with('='))
    };
    let hits: Vec<usize> = (0..top_end).filter(|&i| is_key(lines[i])).collect();
    let mut out: Vec<String> = lines.iter().map(|l| (*l).to_string()).collect();
    match hits.as_slice() {
        [] if old.contains_key("models_dir") => {
            return Err(err(
                "models_dir no está en una línea simple: editalo a mano".into(),
            ));
        }
        [] => {
            // Antes de la primera sección, sin dejarla pegada a líneas en blanco finales.
            let mut at = top_end;
            while at > 0 && out[at - 1].trim().is_empty() {
                at -= 1;
            }
            out.insert(at, line);
        }
        [i] => out[*i] = line,
        _ => return Err(err("models_dir aparece más de una vez".into())),
    }
    let mut new_text = out.join("\n");
    new_text.push('\n');

    // Verificación: solo cambió models_dir.
    let new: toml::Table = toml::from_str(&new_text)
        .map_err(|e| err(format!("la edición no daría TOML válido ({e})")))?;
    let mut a = old.clone();
    let mut b = new.clone();
    a.remove("models_dir");
    let got = b.remove("models_dir");
    if a != b || got != Some(toml::Value::String(value.to_string())) {
        return Err(err(
            "no se pudo cambiar solo models_dir (formato inesperado): editalo a mano".into(),
        ));
    }

    if let Some(parent) = config.parent() {
        std::fs::create_dir_all(parent).map_err(|e| err(e.to_string()))?;
    }
    let tmp = config.with_extension("toml.tmp");
    std::fs::write(&tmp, new_text).map_err(|e| err(e.to_string()))?;
    std::fs::rename(&tmp, config).map_err(|e| {
        std::fs::remove_file(&tmp).ok();
        err(e.to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precedencia() {
        let home = Path::new("/Users/x");
        let r = |flag: Option<&str>, env: Option<&str>, file: Option<&str>, co: Option<&str>| {
            let m = resolve(
                flag.map(PathBuf::from),
                env.map(OsString::from),
                file,
                co.map(PathBuf::from),
                home,
            )
            .unwrap();
            (m.path.display().to_string(), m.source)
        };
        let all = r(Some("/f"), Some("/e"), Some("/a"), Some("models"));
        assert_eq!(all, ("/f".into(), DirSource::Flag));
        let sin_flag = r(None, Some("/e"), Some("/a"), Some("models"));
        assert_eq!(sin_flag, ("/e".into(), DirSource::Env));
        let archivo = r(None, None, Some("~/pesos"), Some("models"));
        assert_eq!(archivo, ("/Users/x/pesos".into(), DirSource::File));
        let checkout = r(None, None, None, Some("models"));
        assert_eq!(checkout, ("models".into(), DirSource::Checkout));
        let defecto = r(None, None, None, None);
        assert_eq!(
            defecto,
            (
                "/Users/x/Library/Application Support/brasa/models".into(),
                DirSource::Default
            )
        );
        // BRASA_MODELS vacío no cuenta.
        assert_eq!(r(None, Some(""), None, None).1, DirSource::Default);
        // models_dir relativo en el archivo: error claro.
        assert!(resolve(None, None, Some("relativa"), None, home).is_err());
    }

    #[test]
    fn lee_el_espacio_libre() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(free_bytes(tmp.path()).is_some_and(|b| b > 0));
        assert!(free_bytes(Path::new("/no/existe/brasa")).is_none());
    }

    #[test]
    fn prepara_la_carpeta_y_rechaza_rutas_malas() {
        let tmp = tempfile::tempdir().unwrap();
        let nueva = tmp.path().join("a/b/./modelos/");
        let p = prepare_models_dir(nueva.to_str().unwrap()).unwrap();
        assert!(p.is_dir());
        assert_eq!(p, tmp.path().join("a/b/modelos"));
        for malo in ["", "  ", "relativa/x", "./x", "/tmp/../etc", "/tmp/a\0b"] {
            assert!(prepare_models_dir(malo).is_err(), "aceptó {malo:?}");
        }
        // Un archivo no es una carpeta.
        let f = tmp.path().join("archivo");
        std::fs::write(&f, b"x").unwrap();
        assert!(prepare_models_dir(f.to_str().unwrap()).is_err());
    }

    #[test]
    fn guarda_models_dir_sin_tocar_lo_demas() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = tmp.path().join("brasa/config.toml");
        // Archivo ausente: se crea con la clave.
        save_models_dir(&cfg, Path::new("/m/uno")).unwrap();
        assert_eq!(
            std::fs::read_to_string(&cfg).unwrap(),
            "models_dir = \"/m/uno\"\n"
        );
        assert_eq!(read_models_dir(&cfg).unwrap().as_deref(), Some("/m/uno"));

        // Con comentarios, otras claves y secciones: se agrega antes de la primera sección.
        let texto = "# mi config\nport = 9000 # puerto\n\n[serve]\nctx = 8192\n";
        std::fs::write(&cfg, texto).unwrap();
        save_models_dir(&cfg, Path::new("/m/dos con espacio")).unwrap();
        let t = std::fs::read_to_string(&cfg).unwrap();
        assert_eq!(
            t,
            "# mi config\nport = 9000 # puerto\nmodels_dir = \"/m/dos con espacio\"\n\n[serve]\nctx = 8192\n"
        );
        // Y si ya está, se reemplaza esa línea.
        save_models_dir(&cfg, Path::new("/m/\"tres\"")).unwrap();
        let t = std::fs::read_to_string(&cfg).unwrap();
        assert!(t.contains("port = 9000 # puerto"), "{t}");
        assert_eq!(
            read_models_dir(&cfg).unwrap().as_deref(),
            Some("/m/\"tres\"")
        );
        assert_eq!(t.matches("models_dir").count(), 1);
    }

    #[test]
    fn no_reescribe_a_ciegas() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = tmp.path().join("config.toml");
        for texto in [
            "clave_rara = 1\n",
            "port = \n",
            "models_dir = \"/a\"\nmodels_dir = \"/b\"\n",
            "models_dir = \"\"\"\n/a\"\"\"\n",
        ] {
            std::fs::write(&cfg, texto).unwrap();
            let e = save_models_dir(&cfg, Path::new("/m")).unwrap_err();
            assert!(e.0.contains("no se modificó"), "{texto:?}: {}", e.0);
            assert_eq!(std::fs::read_to_string(&cfg).unwrap(), texto);
        }
    }
}
