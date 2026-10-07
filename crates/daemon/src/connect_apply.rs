//! Conectar un agente escribiendo su configuración (ADR 0028). Lo usan el botón **Conectar** de la
//! GUI (`POST /api/agents/<herramienta>/connect`) y `brasa connect <herramienta> --apply`.
//!
//! Nunca cambia la herramienta que el usuario usa por defecto: escribe archivos propios de Brasa y,
//! cuando hace falta tocar una config ajena, agrega lo mínimo con respaldo previo. Si la config
//! existente no se entiende o choca con lo que habría que escribir, devuelve error sin escribir.

use std::io::Write;
use std::path::{Path, PathBuf};

use axum::extract::{Path as UrlPath, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use serde_json::{Value, json};

use crate::Shared;
use crate::connect::{AgentTool, AgentsQuery, codex_catalog};

/// Primera línea de comentario de los lanzadores que escribe Brasa (para no pisar archivos ajenos).
const MARK: &str = "# brasa: lanzador generado por brasa connect";

/// Carpetas donde se escribe. Salen del entorno del proceso; los tests usan un `HOME` temporal.
#[derive(Debug, Clone)]
pub struct Paths {
    pub home: PathBuf,
    /// `$CODEX_HOME` o `~/.codex`.
    pub codex_home: PathBuf,
    /// `$XDG_CONFIG_HOME` o `~/.config`.
    pub config_home: PathBuf,
}

impl Paths {
    pub fn from_env() -> Result<Self, String> {
        let home = std::env::var_os("HOME")
            .filter(|h| !h.is_empty())
            .map(PathBuf::from)
            .ok_or("no está definida la variable HOME")?;
        Ok(Self::with_home(
            &home,
            std::env::var_os("CODEX_HOME").map(PathBuf::from),
            std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from),
        ))
    }

    pub fn with_home(home: &Path, codex: Option<PathBuf>, config: Option<PathBuf>) -> Self {
        Self {
            home: home.to_path_buf(),
            codex_home: codex
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or_else(|| home.join(".codex")),
            config_home: config
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or_else(|| home.join(".config")),
        }
    }
}

/// Destino del daemon al que se conecta la herramienta.
#[derive(Debug, Clone)]
pub struct Target<'a> {
    pub host: &'a str,
    pub port: u16,
    pub model: &'a str,
    pub ctx: usize,
}

impl Target<'_> {
    fn base(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }
}

/// Qué se hizo, para mostrarlo en la GUI y en la terminal.
#[derive(Debug, Default, Clone, PartialEq, Serialize)]
pub struct Applied {
    pub written: Vec<String>,
    pub unchanged: Vec<String>,
    pub backups: Vec<String>,
    /// Cómo usar la herramienta conectada.
    pub usage: String,
    pub notes: Vec<String>,
}

/// Error con el código HTTP que corresponde.
#[derive(Debug, Clone, PartialEq)]
pub enum ApplyError {
    /// La herramienta no se puede conectar escribiendo archivos (Cline).
    Unsupported(String),
    /// La config existente choca o no se entiende: no se escribió nada.
    Conflict(String),
    Io(String),
}

impl std::fmt::Display for ApplyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ApplyError::Unsupported(m) | ApplyError::Conflict(m) | ApplyError::Io(m) => {
                f.write_str(m)
            }
        }
    }
}

fn io(path: &Path, e: std::io::Error) -> ApplyError {
    ApplyError::Io(format!("{}: {e}", path.display()))
}

fn show(p: &Path) -> String {
    p.display().to_string()
}

/// Escritura atómica: archivo temporal en la misma carpeta y `rename`.
fn write_atomic(path: &Path, body: &str, mode: Option<u32>) -> Result<(), ApplyError> {
    let dir = path.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir).map_err(|e| io(dir, e))?;
    let tmp = dir.join(format!(
        ".{}.brasa-tmp",
        path.file_name()
            .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
    ));
    let mut f = std::fs::File::create(&tmp).map_err(|e| io(&tmp, e))?;
    f.write_all(body.as_bytes()).map_err(|e| io(&tmp, e))?;
    f.sync_all().map_err(|e| io(&tmp, e))?;
    drop(f);
    if let Some(m) = mode {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(m))
            .map_err(|e| io(&tmp, e))?;
    }
    std::fs::rename(&tmp, path).map_err(|e| io(path, e))
}

/// Escribe un archivo propio de Brasa si cambió. Lo anota en `written` o en `unchanged`.
fn own_file(
    path: &Path,
    body: &str,
    mode: Option<u32>,
    out: &mut Applied,
) -> Result<(), ApplyError> {
    if std::fs::read_to_string(path).is_ok_and(|old| old == body) {
        out.unchanged.push(show(path));
        return Ok(());
    }
    write_atomic(path, body, mode)?;
    out.written.push(show(path));
    Ok(())
}

/// Copia `path` a `<path>.brasa-bak-<segundos unix>` antes de modificarlo.
fn backup(path: &Path, out: &mut Applied) -> Result<(), ApplyError> {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let mut name = path.as_os_str().to_owned();
    name.push(format!(".brasa-bak-{secs}"));
    let bak = PathBuf::from(name);
    std::fs::copy(path, &bak).map_err(|e| io(&bak, e))?;
    out.backups.push(show(&bak));
    Ok(())
}

pub fn apply(tool: AgentTool, paths: &Paths, t: &Target) -> Result<Applied, ApplyError> {
    match tool {
        AgentTool::Codex => codex(paths, t),
        AgentTool::ClaudeCode => claude_code(paths, t),
        AgentTool::Opencode => opencode(paths, t),
        AgentTool::Cline => Err(ApplyError::Unsupported(
            "Cline guarda su configuración dentro de VS Code: no se puede conectar desde acá. \
             Copiá los datos en Cline (Ajustes → API Provider → OpenAI Compatible)."
                .into(),
        )),
    }
}

fn codex(paths: &Paths, t: &Target) -> Result<Applied, ApplyError> {
    let mut out = Applied {
        usage: "codex --profile brasa".into(),
        ..Applied::default()
    };
    let dir = &paths.codex_home;
    let config = dir.join("config.toml");
    let base_url = format!("{}/v1", t.base());

    // Primero se valida la config ajena: si choca, no se escribe nada.
    let existing = match std::fs::read_to_string(&config) {
        Ok(s) => Some(s),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(io(&config, e)),
    };
    let mut add_provider = true;
    if let Some(text) = &existing {
        let table: toml::Table = toml::from_str(text).map_err(|e| {
            ApplyError::Conflict(format!(
                "{} no es TOML válido ({e}); no se modificó. Agregá la tabla [model_providers.brasa] a mano.",
                show(&config)
            ))
        })?;
        if let Some(p) = table.get("model_providers").and_then(|m| m.get("brasa")) {
            match p.get("base_url").and_then(|u| u.as_str()) {
                Some(u) if u == base_url => add_provider = false,
                other => {
                    return Err(ApplyError::Conflict(format!(
                        "{} ya tiene [model_providers.brasa] con base_url {:?}, distinta de {base_url:?}; \
                         no se modificó. Cambiala a mano.",
                        show(&config),
                        other.unwrap_or("(sin base_url)")
                    )));
                }
            }
        }
    }

    let catalog_path = dir.join("brasa-models.json");
    own_file(
        &catalog_path,
        &(codex_catalog(t.model, t.ctx) + "\n"),
        None,
        &mut out,
    )?;
    let profile = format!(
        "# Perfil de Codex para brasa (generado por brasa connect; se reescribe al reconectar).\n\
         model_provider = \"brasa\"\n\
         model = \"{model}\"\n\
         model_context_window = {ctx}\n\
         model_auto_compact_token_limit = {compact}\n\
         model_catalog_json = {catalog}\n",
        model = t.model,
        ctx = t.ctx,
        compact = t.ctx * 3 / 4,
        catalog = toml::Value::String(show(&catalog_path)),
    );
    own_file(&dir.join("brasa.config.toml"), &profile, None, &mut out)?;

    if add_provider {
        let block = format!(
            "\n# Agregado por brasa connect (Brasa, engine local). Se usa con: codex --profile brasa\n\
             [model_providers.brasa]\n\
             name = \"Brasa (local)\"\n\
             base_url = \"{base_url}\"\n\
             wire_api = \"responses\"\n"
        );
        let body = match &existing {
            Some(text) => {
                backup(&config, &mut out)?;
                let sep = if text.is_empty() || text.ends_with('\n') {
                    ""
                } else {
                    "\n"
                };
                format!("{text}{sep}{block}")
            }
            None => block.trim_start().to_string(),
        };
        write_atomic(&config, &body, None)?;
        out.written.push(show(&config));
    } else {
        out.unchanged.push(show(&config));
    }
    Ok(out)
}

fn claude_code(paths: &Paths, t: &Target) -> Result<Applied, ApplyError> {
    let bin = paths.home.join(".local/bin");
    let path = bin.join("claude-brasa");
    if let Ok(old) = std::fs::read_to_string(&path)
        && !old.lines().nth(1).is_some_and(|l| l.starts_with(MARK))
    {
        return Err(ApplyError::Conflict(format!(
            "{} ya existe y no lo creó Brasa; no se modificó.",
            show(&path)
        )));
    }
    let out_tokens = (t.ctx / 8).min(2048);
    let script = format!(
        "#!/bin/sh\n\
         {MARK} claude-code (se reescribe al reconectar; se puede borrar).\n\
         # Arranca Claude Code contra brasa sin tocar ~/.claude.\n\
         export ANTHROPIC_BASE_URL={base}\n\
         export ANTHROPIC_AUTH_TOKEN=brasa-local\n\
         export ANTHROPIC_MODEL={model}\n\
         export ANTHROPIC_DEFAULT_HAIKU_MODEL={model}\n\
         export CLAUDE_CODE_MAX_CONTEXT_TOKENS={ctx}\n\
         export CLAUDE_CODE_MAX_OUTPUT_TOKENS={out_tokens}\n\
         export CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT=1\n\
         exec claude \"$@\"\n",
        base = t.base(),
        model = t.model,
        ctx = t.ctx,
    );
    let mut out = Applied {
        usage: "claude-brasa --tools Bash,Read,Edit,Write,Glob,Grep".into(),
        ..Applied::default()
    };
    own_file(&path, &script, Some(0o755), &mut out)?;
    let in_path =
        std::env::var_os("PATH").is_some_and(|p| std::env::split_paths(&p).any(|d| d == bin));
    if !in_path {
        out.notes.push(format!(
            "{} no está en el PATH: corré el lanzador con la ruta completa o agregá la carpeta al PATH.",
            show(&bin)
        ));
    }
    Ok(out)
}

fn opencode(paths: &Paths, t: &Target) -> Result<Applied, ApplyError> {
    let dir = paths.config_home.join("opencode");
    let jsonc = dir.join("opencode.jsonc");
    let path = dir.join("opencode.json");
    if jsonc.exists() {
        return Err(ApplyError::Conflict(format!(
            "{} usa JSONC (con comentarios): no se reescribe para no perderlos. Agregá el proveedor `brasa` a mano.",
            show(&jsonc)
        )));
    }
    let provider = json!({
        "npm": "@ai-sdk/openai-compatible",
        "name": "Brasa (local)",
        "options": {"baseURL": format!("{}/v1", t.base())},
        "models": {t.model: {"name": format!("{} (Brasa)", t.model)}}
    });
    let mut out = Applied {
        usage: format!("en OpenCode, elegí el modelo brasa/{}", t.model),
        ..Applied::default()
    };
    let mut doc = match std::fs::read_to_string(&path) {
        Ok(text) => serde_json::from_str::<Value>(&text).map_err(|e| {
            ApplyError::Conflict(format!(
                "{} no es JSON estricto ({e}); puede tener comentarios. No se modificó: agregá el proveedor `brasa` a mano.",
                show(&path)
            ))
        })?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            json!({"$schema": "https://opencode.ai/config.json"})
        }
        Err(e) => return Err(io(&path, e)),
    };
    let Some(root) = doc.as_object_mut() else {
        return Err(ApplyError::Conflict(format!(
            "{} no es un objeto JSON; no se modificó.",
            show(&path)
        )));
    };
    let providers = root.entry("provider").or_insert_with(|| json!({}));
    let Some(providers) = providers.as_object_mut() else {
        return Err(ApplyError::Conflict(format!(
            "{}: `provider` no es un objeto; no se modificó.",
            show(&path)
        )));
    };
    if providers.get("brasa") == Some(&provider) {
        out.unchanged.push(show(&path));
        return Ok(out);
    }
    providers.insert("brasa".into(), provider);
    if path.exists() {
        backup(&path, &mut out)?;
    }
    let body =
        serde_json::to_string_pretty(&doc).map_err(|e| ApplyError::Io(e.to_string()))? + "\n";
    write_atomic(&path, &body, None)?;
    out.written.push(show(&path));
    Ok(out)
}

/// `POST /api/agents/<herramienta>/connect?ctx=`: botón **Conectar** de la GUI.
pub async fn connect(
    State(s): State<Shared>,
    UrlPath(tool): UrlPath<String>,
    Query(q): Query<AgentsQuery>,
) -> Response {
    let err =
        |code: StatusCode, msg: String| (code, axum::Json(json!({"error": msg}))).into_response();
    let Some(tool) = AgentTool::parse(&tool) else {
        return err(
            StatusCode::NOT_FOUND,
            format!("herramienta desconocida: {tool}"),
        );
    };
    let ctx = match crate::parse_ctx(q.ctx.as_deref(), s.ctx) {
        Ok(v) => v,
        Err(e) => return err(StatusCode::BAD_REQUEST, e),
    };
    let paths = match Paths::from_env() {
        Ok(p) => p,
        Err(e) => return err(StatusCode::INTERNAL_SERVER_ERROR, e),
    };
    let host = crate::connect::client_host(s.addr);
    let model = s.model_id.clone();
    let port = s.addr.port();
    let res = tokio::task::spawn_blocking(move || {
        apply(
            tool,
            &paths,
            &Target {
                host: &host,
                port,
                model: &model,
                ctx,
            },
        )
    })
    .await;
    match res {
        Ok(Ok(a)) => {
            axum::Json(json!({"tool": tool.key(), "ctx": ctx, "applied": a})).into_response()
        }
        Ok(Err(ApplyError::Unsupported(m))) => err(StatusCode::BAD_REQUEST, m),
        Ok(Err(ApplyError::Conflict(m))) => err(StatusCode::CONFLICT, m),
        Ok(Err(ApplyError::Io(m))) => err(StatusCode::INTERNAL_SERVER_ERROR, m),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target() -> Target<'static> {
        Target {
            host: "127.0.0.1",
            port: 8080,
            model: "qwen3-4b-q4",
            ctx: 16384,
        }
    }

    fn paths(home: &Path) -> Paths {
        Paths::with_home(home, None, None)
    }

    #[test]
    fn codex_crea_todo_y_es_idempotente() {
        let h = tempfile::tempdir().unwrap();
        let a = apply(AgentTool::Codex, &paths(h.path()), &target()).unwrap();
        assert_eq!(a.written.len(), 3, "{a:?}");
        let cfg = std::fs::read_to_string(h.path().join(".codex/config.toml")).unwrap();
        let t: toml::Table = toml::from_str(&cfg).unwrap();
        assert_eq!(
            t["model_providers"]["brasa"]["base_url"].as_str(),
            Some("http://127.0.0.1:8080/v1")
        );
        let prof: toml::Table = toml::from_str(
            &std::fs::read_to_string(h.path().join(".codex/brasa.config.toml")).unwrap(),
        )
        .unwrap();
        assert_eq!(prof["model_context_window"].as_integer(), Some(16384));
        let b = apply(AgentTool::Codex, &paths(h.path()), &target()).unwrap();
        assert!(b.written.is_empty(), "{b:?}");
        assert_eq!(b.unchanged.len(), 3);
    }

    #[test]
    fn codex_agrega_al_final_con_respaldo_y_conserva_comentarios() {
        let h = tempfile::tempdir().unwrap();
        let cfg = h.path().join(".codex/config.toml");
        std::fs::create_dir_all(cfg.parent().unwrap()).unwrap();
        let mine =
            "# mi config\nmodel = \"gpt-5\"\n\n[model_providers.otro]\nbase_url = \"http://x\"";
        std::fs::write(&cfg, mine).unwrap();
        let a = apply(AgentTool::Codex, &paths(h.path()), &target()).unwrap();
        assert_eq!(a.backups.len(), 1);
        assert_eq!(std::fs::read_to_string(&a.backups[0]).unwrap(), mine);
        let new = std::fs::read_to_string(&cfg).unwrap();
        assert!(new.starts_with(mine), "{new}");
        let t: toml::Table = toml::from_str(&new).unwrap();
        assert_eq!(t["model"].as_str(), Some("gpt-5"));
        assert!(t["model_providers"].get("otro").is_some());
        assert!(t["model_providers"].get("brasa").is_some());
    }

    #[test]
    fn codex_no_toca_una_config_que_choca() {
        let h = tempfile::tempdir().unwrap();
        let cfg = h.path().join(".codex/config.toml");
        std::fs::create_dir_all(cfg.parent().unwrap()).unwrap();
        let other = "[model_providers.brasa]\nbase_url = \"http://127.0.0.1:9999/v1\"\n";
        std::fs::write(&cfg, other).unwrap();
        let e = apply(AgentTool::Codex, &paths(h.path()), &target()).unwrap_err();
        assert!(matches!(e, ApplyError::Conflict(_)), "{e:?}");
        assert_eq!(std::fs::read_to_string(&cfg).unwrap(), other);
        assert!(!h.path().join(".codex/brasa.config.toml").exists());
        std::fs::write(&cfg, "esto no es [toml").unwrap();
        let e = apply(AgentTool::Codex, &paths(h.path()), &target()).unwrap_err();
        assert!(matches!(e, ApplyError::Conflict(_)), "{e:?}");
    }

    #[test]
    fn codex_respeta_codex_home() {
        let h = tempfile::tempdir().unwrap();
        let p = Paths::with_home(h.path(), Some(h.path().join("otro-codex")), None);
        apply(AgentTool::Codex, &p, &target()).unwrap();
        assert!(h.path().join("otro-codex/brasa.config.toml").exists());
        assert!(!h.path().join(".codex").exists());
    }

    #[test]
    fn claude_code_escribe_un_lanzador_y_no_pisa_ajenos() {
        let h = tempfile::tempdir().unwrap();
        let a = apply(AgentTool::ClaudeCode, &paths(h.path()), &target()).unwrap();
        let path = h.path().join(".local/bin/claude-brasa");
        let s = std::fs::read_to_string(&path).unwrap();
        assert!(s.contains("export ANTHROPIC_BASE_URL=http://127.0.0.1:8080"));
        assert!(s.contains("exec claude \"$@\""));
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o755
        );
        assert_eq!(a.written, vec![show(&path)]);
        assert!(!h.path().join(".claude").exists());
        // Otra vez: sin cambios. Un archivo ajeno con el mismo nombre: error sin tocarlo.
        let b = apply(AgentTool::ClaudeCode, &paths(h.path()), &target()).unwrap();
        assert_eq!(b.unchanged.len(), 1);
        std::fs::write(&path, "#!/bin/sh\necho mío\n").unwrap();
        let e = apply(AgentTool::ClaudeCode, &paths(h.path()), &target()).unwrap_err();
        assert!(matches!(e, ApplyError::Conflict(_)));
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "#!/bin/sh\necho mío\n"
        );
    }

    #[test]
    fn opencode_crea_agrega_y_respeta_jsonc() {
        let h = tempfile::tempdir().unwrap();
        let p = paths(h.path());
        let path = h.path().join(".config/opencode/opencode.json");
        apply(AgentTool::Opencode, &p, &target()).unwrap();
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(
            v["provider"]["brasa"]["options"]["baseURL"],
            "http://127.0.0.1:8080/v1"
        );
        assert!(v.get("model").is_none(), "no cambia el modelo por defecto");
        // Con config previa: conserva el resto, respalda, y la segunda vez no cambia nada.
        std::fs::write(
            &path,
            r#"{"theme": "x", "provider": {"otro": {"name": "o"}}}"#,
        )
        .unwrap();
        let a = apply(AgentTool::Opencode, &p, &target()).unwrap();
        assert_eq!(a.backups.len(), 1);
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v["theme"], "x");
        assert!(v["provider"].get("otro").is_some() && v["provider"].get("brasa").is_some());
        assert_eq!(
            apply(AgentTool::Opencode, &p, &target())
                .unwrap()
                .unchanged
                .len(),
            1
        );
        // Con comentarios: error y archivo intacto.
        let commented = "{\n  // mío\n  \"theme\": \"x\"\n}\n";
        std::fs::write(&path, commented).unwrap();
        let e = apply(AgentTool::Opencode, &p, &target()).unwrap_err();
        assert!(matches!(e, ApplyError::Conflict(_)));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), commented);
    }

    #[test]
    fn cline_no_se_conecta() {
        let h = tempfile::tempdir().unwrap();
        let e = apply(AgentTool::Cline, &paths(h.path()), &target()).unwrap_err();
        assert!(matches!(e, ApplyError::Unsupported(_)));
        assert_eq!(std::fs::read_dir(h.path()).unwrap().count(), 0);
    }
}
