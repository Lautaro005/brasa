//! Manifiestos de modelo en TOML (ADR 0020): repo y revisión de Hugging Face, archivos con
//! sha256, cuantización destino, contexto máximo y licencia. Los de fábrica van embebidos en el
//! binario; además se pueden leer de una carpeta (`$BRASA_CATALOG`).
//!
//! Desde ADR 0031 un manifiesto puede traer además `[prebuilt]`: los pesos ya convertidos
//! (`model.brasa` y tokenizers) en un repo de Hugging Face, que `brasa pull` baja directo a la
//! carpeta del modelo.

use std::path::{Component, Path};

use serde::Deserialize;

use crate::{Error, Result};

/// Comprueba que `path` sea relativa y sin `.` ni `..` (defensa contra manifiestos externos).
pub fn safe_relative(path: &str) -> std::result::Result<(), String> {
    if path.is_empty() {
        return Err("ruta vacía".into());
    }
    let p = Path::new(path);
    if p.is_absolute() {
        return Err(format!("ruta absoluta no permitida: {path:?}"));
    }
    for c in p.components() {
        if !matches!(c, Component::Normal(_)) {
            return Err(format!("ruta no permitida (`.` o `..`): {path:?}"));
        }
    }
    Ok(())
}

/// Un valor que se une a `models_dir` (como `name` o `hf_dir`) tiene que ser un solo componente
/// normal: sin `/`, `.`, `..` ni rutas absolutas.
fn validate_component(field: &str, value: &str) -> Result<()> {
    let p = Path::new(value);
    if value.is_empty()
        || p.is_absolute()
        || p.components().count() != 1
        || matches!(value, "." | "..")
    {
        return Err(Error(format!(
            "{field} {value:?}: tiene que ser un solo componente, sin `/`, `.` ni `..`"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct FileSpec {
    /// Ruta dentro del repo de Hugging Face.
    pub path: String,
    /// sha256 del contenido del archivo.
    pub sha256: String,
    #[serde(default)]
    pub size: Option<u64>,
}

/// Pesos ya convertidos al formato nativo (ADR 0031): repo, revisión y subcarpeta en Hugging
/// Face, y los archivos que van a `<carpeta de modelos>/<name>/`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Prebuilt {
    pub repo: String,
    /// Commit de 40 hex (fijado) o una rama mientras el repo no tiene revisión fija. El sha256 de
    /// cada archivo se verifica igual.
    pub revision: String,
    /// Subcarpeta del repo donde están los archivos (vacía: la raíz).
    #[serde(default)]
    pub subdir: String,
    pub files: Vec<FileSpec>,
}

impl Prebuilt {
    /// La revisión es un commit fijo (40 hex), no una rama.
    pub fn is_pinned(&self) -> bool {
        is_commit(&self.revision)
    }

    pub fn total_bytes(&self) -> u64 {
        self.files.iter().filter_map(|f| f.size).sum()
    }
}

fn is_commit(s: &str) -> bool {
    s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Caracteres admitidos en un segmento de repo o revisión (forman parte de la URL).
fn url_segment_ok(s: &str) -> bool {
    !s.is_empty()
        && !matches!(s, "." | "..")
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
}

fn validate_files(name: &str, files: &[FileSpec]) -> Result<()> {
    for f in files {
        if f.sha256.len() != 64 || !f.sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(Error(format!(
                "{name}: sha256 inválido en {:?} (se esperan 64 dígitos hex)",
                f.path
            )));
        }
        safe_relative(&f.path).map_err(|e| Error(format!("{name}: {e}")))?;
    }
    Ok(())
}

fn yes() -> bool {
    true
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Manifest {
    pub name: String,
    pub family: String,
    pub source_repo: String,
    pub source_revision: String,
    /// Carpeta local donde `brasa pull` deja los safetensors.
    pub hf_dir: String,
    #[serde(default)]
    pub quant: String,
    pub max_context: usize,
    #[serde(default)]
    pub license: String,
    pub files: Vec<FileSpec>,
    /// Si `brasa convert` produce este modelo desde los safetensors (`pull --desde-fuente`).
    #[serde(default = "yes")]
    pub convertible: bool,
    /// Pesos ya convertidos (ADR 0031). Sin esto, `pull` baja los safetensors.
    #[serde(default)]
    pub prebuilt: Option<Prebuilt>,
}

/// Manifiestos embebidos en el binario.
const BUILTIN: &[&str] = &[
    include_str!("../manifests/qwen3-4b-q4.toml"),
    include_str!("../manifests/qwen3-4b-q4-e8.toml"),
];

impl Manifest {
    pub fn parse(text: &str) -> Result<Self> {
        let mut m: Self =
            toml::from_str(text).map_err(|e| Error(format!("manifiesto inválido: {e}")))?;
        m.validate()?;
        // El sha256 calculado siempre va en minúsculas: se normaliza el del manifiesto.
        for f in &mut m.files {
            f.sha256 = f.sha256.to_ascii_lowercase();
        }
        if let Some(p) = &mut m.prebuilt {
            for f in &mut p.files {
                f.sha256 = f.sha256.to_ascii_lowercase();
            }
        }
        Ok(m)
    }

    /// Valida los campos que el resto del código asume: `name`/`hf_dir` de un solo componente,
    /// sha256 de 64 hex y rutas de archivo seguras.
    fn validate(&self) -> Result<()> {
        validate_component("name", &self.name)?;
        validate_component("hf_dir", &self.hf_dir)?;
        validate_files(&self.name, &self.files)?;
        if let Some(p) = &self.prebuilt {
            let n = &self.name;
            let mut parts = p.repo.split('/');
            let repo_ok = matches!(
                (parts.next(), parts.next(), parts.next()),
                (Some(a), Some(b), None) if url_segment_ok(a) && url_segment_ok(b)
            );
            if !repo_ok {
                return Err(Error(format!(
                    "{n}: prebuilt.repo {:?} tiene que ser `dueño/nombre`",
                    p.repo
                )));
            }
            if !url_segment_ok(&p.revision) {
                return Err(Error(format!(
                    "{n}: prebuilt.revision {:?} inválida (commit de 40 hex o nombre de rama)",
                    p.revision
                )));
            }
            if !p.subdir.is_empty() {
                safe_relative(&p.subdir)
                    .map_err(|e| Error(format!("{n}: prebuilt.subdir: {e}")))?;
            }
            validate_files(n, &p.files)?;
            if !p.files.iter().any(|f| f.path == "model.brasa") {
                return Err(Error(format!("{n}: prebuilt no incluye model.brasa")));
            }
        }
        Ok(())
    }

    pub fn builtin() -> Result<Vec<Self>> {
        BUILTIN.iter().map(|t| Self::parse(t)).collect()
    }

    /// Manifiestos de una carpeta `*.toml` (además de los embebidos).
    pub fn load_dir(dir: &Path) -> Result<Vec<Self>> {
        let mut out = Vec::new();
        let entries =
            std::fs::read_dir(dir).map_err(|e| Error(format!("{}: {e}", dir.display())))?;
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().is_some_and(|x| x == "toml") {
                let text = std::fs::read_to_string(&p)
                    .map_err(|e| Error(format!("{}: {e}", p.display())))?;
                out.push(Self::parse(&text)?);
            }
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }

    /// Todos los manifiestos: embebidos más `$BRASA_CATALOG` si existe.
    pub fn all() -> Result<Vec<Self>> {
        let mut out = Self::builtin()?;
        if let Some(dir) = std::env::var_os("BRASA_CATALOG") {
            for m in Self::load_dir(Path::new(&dir))? {
                if !out.iter().any(|x| x.name == m.name) {
                    out.push(m);
                }
            }
        }
        Ok(out)
    }

    pub fn find(name: &str) -> Result<Self> {
        Self::all()?
            .into_iter()
            .find(|m| m.name == name)
            .ok_or_else(|| Error(format!("no hay un manifiesto para {name:?}")))
    }

    /// Bytes de los safetensors de origen.
    pub fn total_bytes(&self) -> u64 {
        self.files.iter().filter_map(|f| f.size).sum()
    }

    /// Bytes que baja `brasa pull` por defecto: los pesos convertidos si hay, si no el origen.
    pub fn download_bytes(&self) -> u64 {
        self.prebuilt
            .as_ref()
            .map_or_else(|| self.total_bytes(), Prebuilt::total_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn el_manifiesto_embebido_es_valido() {
        let all = Manifest::builtin().unwrap();
        let m = all.iter().find(|m| m.name == "qwen3-4b-q4").unwrap();
        assert_eq!(m.source_repo, "Qwen/Qwen3-4B");
        assert_eq!(m.hf_dir, "qwen3-4b-hf");
        assert!(m.files.iter().any(|f| f.path.ends_with(".safetensors")));
        assert!(m.files.iter().all(|f| f.sha256.len() == 64));
        assert!(m.total_bytes() > 7_000_000_000);
    }

    #[test]
    fn los_embebidos_traen_pesos_convertidos() {
        for name in ["qwen3-4b-q4", "qwen3-4b-q4-e8"] {
            let m = Manifest::find(name).unwrap();
            let p = m.prebuilt.as_ref().expect("prebuilt");
            assert_eq!(p.repo, "lautiss/brasa-v0.01-base");
            assert_eq!(p.subdir, name);
            let paths: Vec<_> = p.files.iter().map(|f| f.path.as_str()).collect();
            assert_eq!(
                paths,
                ["tokenizer.json", "tokenizer_config.json", "model.brasa"],
                "model.brasa va último: la carpeta cuenta como instalada recién al final"
            );
            assert!(p.files.iter().all(|f| f.size.is_some()));
            assert!(m.download_bytes() > 2_000_000_000 && m.download_bytes() < 3_000_000_000);
        }
        assert!(Manifest::find("qwen3-4b-q4").unwrap().convertible);
        assert!(!Manifest::find("qwen3-4b-q4-e8").unwrap().convertible);
    }

    #[test]
    fn valida_el_prebuilt() {
        let plantilla = r#"
name = "x"
family = "qwen3"
source_repo = "a/b"
source_revision = "r"
hf_dir = "x-hf"
max_context = 4096
files = []
[prebuilt]
repo = "%REPO%"
revision = "%REV%"
subdir = "%SUB%"
files = [{ path = "%PATH%", sha256 = "%SHA%" }]
"#;
        let sha = "a".repeat(64);
        let t = |repo: &str, rev: &str, sub: &str, path: &str| {
            plantilla
                .replace("%REPO%", repo)
                .replace("%REV%", rev)
                .replace("%SUB%", sub)
                .replace("%PATH%", path)
                .replace("%SHA%", &sha)
        };
        let ok = Manifest::parse(&t("u/r", "main", "x", "model.brasa")).unwrap();
        assert!(!ok.prebuilt.unwrap().is_pinned());
        let fijo = Manifest::parse(&t("u/r", &"0".repeat(40), "", "model.brasa")).unwrap();
        assert!(fijo.prebuilt.unwrap().is_pinned());
        for (repo, rev, sub, path) in [
            ("u", "main", "x", "model.brasa"),
            ("u/r/s", "main", "x", "model.brasa"),
            ("u/..", "main", "x", "model.brasa"),
            ("u/r", "ma/in", "x", "model.brasa"),
            ("u/r", "..", "x", "model.brasa"),
            ("u/r", "main?x=1", "x", "model.brasa"),
            ("u/r", "main", "../x", "model.brasa"),
            ("u/r", "main", "/x", "model.brasa"),
            ("u/r", "main", "x", "../model.brasa"),
            ("u/r", "main", "x", "tokenizer.json"),
        ] {
            assert!(
                Manifest::parse(&t(repo, rev, sub, path)).is_err(),
                "aceptó repo={repo:?} rev={rev:?} sub={sub:?} path={path:?}"
            );
        }
    }

    #[test]
    fn parsea_y_busca() {
        let m = Manifest::find("qwen3-4b-q4").unwrap();
        assert_eq!(m.family, "qwen3");
        assert!(Manifest::find("no-existe").is_err());
    }

    #[test]
    fn rechaza_sha256_corto_y_rutas_inseguras() {
        let base = r#"
name = "x"
family = "qwen3"
source_repo = "a/b"
source_revision = "r"
hf_dir = "x-hf"
max_context = 4096
files = [{ path = "%PATH%", sha256 = "%SHA%" , size = 1}]
"#;
        let malo = base
            .replace("%PATH%", "model.safetensors")
            .replace("%SHA%", "abc");
        assert!(Manifest::parse(&malo).is_err());
        let escape = base
            .replace("%PATH%", "../fuera.safetensors")
            .replace("%SHA%", &"0".repeat(64));
        assert!(Manifest::parse(&escape).is_err());
        let absoluto = base
            .replace("%PATH%", "/etc/passwd")
            .replace("%SHA%", &"0".repeat(64));
        assert!(Manifest::parse(&absoluto).is_err());
        let ok = base
            .replace("%PATH%", "sub/model.safetensors")
            .replace("%SHA%", &"a".repeat(64));
        assert!(Manifest::parse(&ok).is_ok());
    }

    #[test]
    fn rechaza_name_y_hf_dir_inseguros() {
        let plantilla = r#"
name = "%NAME%"
family = "qwen3"
source_repo = "a/b"
source_revision = "r"
hf_dir = "%HF%"
max_context = 4096
files = [{ path = "model.safetensors", sha256 = "%SHA%" }]
"#;
        let sha = "a".repeat(64);
        for (name, hf) in [
            ("ok", "../x"),
            ("ok", "/tmp/x"),
            ("ok", "a/b"),
            ("..", "ok"),
            (".", "ok"),
            ("a/b", "ok"),
            ("/abs", "ok"),
        ] {
            let t = plantilla
                .replace("%NAME%", name)
                .replace("%HF%", hf)
                .replace("%SHA%", &sha);
            assert!(
                Manifest::parse(&t).is_err(),
                "aceptó name={name:?} hf_dir={hf:?}"
            );
        }
    }

    #[test]
    fn normaliza_el_sha256_a_minusculas() {
        let t = r#"
name = "x"
family = "qwen3"
source_repo = "a/b"
source_revision = "r"
hf_dir = "x-hf"
max_context = 4096
files = [{ path = "model.safetensors", sha256 = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" }]
"#;
        let m = Manifest::parse(t).unwrap();
        assert_eq!(m.files[0].sha256, "a".repeat(64));
    }
}
