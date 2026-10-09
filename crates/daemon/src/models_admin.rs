//! Descarga de modelos y carpeta de modelos desde la GUI (ADR 0031).
//!
//! - `POST /api/models/pull {name}`: baja en segundo plano los pesos ya convertidos de un modelo
//!   del catálogo (una descarga a la vez). Las URLs salen solo del manifiesto; del pedido se usa
//!   únicamente el nombre, que tiene que estar en el catálogo.
//! - `GET /api/models/pull`: progreso por archivo, estado y error.
//! - `POST /api/models/pull/cancel` (o `DELETE /api/models/pull`): cancela; el `.part` queda para
//!   reanudar.
//! - `POST /api/models/dir {path}`: cambia la carpeta de modelos y la guarda en el archivo de
//!   configuración (`models_dir`).
//! - `POST /api/models/dir/choose`: abre el selector de carpetas de macOS y devuelve la ruta
//!   elegida, sin aplicarla.
//! - `POST /api/models/dir/open`: abre la carpeta de modelos en Finder.
//!
//! Todos los `POST`/`DELETE` pasan por el middleware de origen (`origin.rs`).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use brasa_catalog::dirs::{self, DirSource, ModelsDir};
use brasa_catalog::manifest::Manifest;
use brasa_catalog::storage;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::Shared;

/// De dónde salen los manifiestos del catálogo.
#[derive(Debug, Clone)]
pub enum Catalog {
    /// Los embebidos más `$BRASA_CATALOG` (`Manifest::all`).
    System,
    /// Una lista fija (tests).
    Fixed(Vec<Manifest>),
}

impl Catalog {
    pub fn all(&self) -> Vec<Manifest> {
        match self {
            Catalog::System => Manifest::all().unwrap_or_default(),
            Catalog::Fixed(v) => v.clone(),
        }
    }
}

/// Acciones de escritorio que el daemon delega en macOS. Se inyectan para que los tests no abran
/// Finder ni diálogos.
#[derive(Debug, Clone, Copy)]
pub struct Desktop {
    /// Selector de carpetas: `Ok(Some(ruta))`, `Ok(None)` si el usuario canceló.
    pub choose_folder: fn(Option<&Path>) -> Result<Option<String>, String>,
    /// Abre una carpeta en Finder.
    pub open: fn(&Path) -> Result<(), String>,
}

impl Desktop {
    pub fn system() -> Self {
        Self {
            choose_folder: osascript_choose_folder,
            open: finder_open,
        }
    }
}

/// `osascript -e 'POSIX path of (choose folder …)'`. La carpeta actual va como argumento (no se
/// interpola en el script).
fn osascript_choose_folder(current: Option<&Path>) -> Result<Option<String>, String> {
    let prompt = "Elegí la carpeta de modelos de Brasa";
    let mut cmd = std::process::Command::new("/usr/bin/osascript");
    match current.filter(|p| p.is_dir()) {
        Some(dir) => {
            cmd.args([
                "-e",
                "on run argv",
                "-e",
                &format!(
                    "POSIX path of (choose folder with prompt \"{prompt}\" default location \
                     (POSIX file (item 1 of argv)))"
                ),
                "-e",
                "end run",
            ])
            .arg(dir);
        }
        None => {
            cmd.args([
                "-e",
                &format!("POSIX path of (choose folder with prompt \"{prompt}\")"),
            ]);
        }
    }
    let out = cmd
        .output()
        .map_err(|e| format!("no se pudo abrir el selector: {e}"))?;
    if out.status.success() {
        let s = String::from_utf8_lossy(&out.stdout)
            .trim_end_matches('\n')
            .to_string();
        let s = if s.len() > 1 {
            s.trim_end_matches('/').to_string()
        } else {
            s
        };
        return Ok(Some(s));
    }
    let err = String::from_utf8_lossy(&out.stderr);
    // -128: el usuario canceló.
    if err.contains("-128") {
        return Ok(None);
    }
    Err(format!("el selector de carpetas falló: {}", err.trim()))
}

fn finder_open(dir: &Path) -> Result<(), String> {
    let st = std::process::Command::new("/usr/bin/open")
        .arg(dir)
        .status()
        .map_err(|e| format!("no se pudo abrir Finder: {e}"))?;
    if st.success() {
        Ok(())
    } else {
        Err(format!("`open {}` terminó con {st}", dir.display()))
    }
}

/// Progreso de un archivo de la descarga.
#[derive(Debug, Clone, Serialize)]
pub struct PullFile {
    pub path: String,
    pub size: Option<u64>,
    pub done: u64,
}

/// Estado de la descarga (una a la vez).
#[derive(Debug, Clone, Serialize)]
pub struct PullStatus {
    /// `idle`, `running`, `done`, `error` o `cancelled`.
    pub state: &'static str,
    pub name: Option<String>,
    pub dest: Option<String>,
    pub repo: Option<String>,
    pub revision: Option<String>,
    pub files: Vec<PullFile>,
    pub done_bytes: u64,
    pub total_bytes: u64,
    pub error: Option<String>,
    /// Inicio y fin en ms desde la época Unix.
    pub started_ms: Option<u64>,
    pub finished_ms: Option<u64>,
    #[serde(skip)]
    cancel: Arc<AtomicBool>,
}

impl Default for PullStatus {
    fn default() -> Self {
        Self {
            state: "idle",
            name: None,
            dest: None,
            repo: None,
            revision: None,
            files: Vec::new(),
            done_bytes: 0,
            total_bytes: 0,
            error: None,
            started_ms: None,
            finished_ms: None,
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

fn err(code: StatusCode, msg: impl Into<String>) -> Response {
    (code, axum::Json(json!({ "error": msg.into() }))).into_response()
}

fn status_json(s: &Shared) -> Response {
    let st = s.pull.lock().expect("pull").clone();
    axum::Json(st).into_response()
}

pub fn current_dir(s: &Shared) -> ModelsDir {
    s.models_dir.read().expect("models_dir").clone()
}

/// Bytes ya en disco de una descarga a medias (archivos completos y `.part`).
pub fn partial_bytes(dir: &Path, m: &Manifest) -> u64 {
    let Some(p) = &m.prebuilt else { return 0 };
    storage::bytes_on_disk(&dir.join(&m.name), &p.files)
}

#[derive(Debug, Deserialize)]
pub struct PullReq {
    pub name: String,
}

pub async fn pull_status(State(s): State<Shared>) -> Response {
    status_json(&s)
}

pub async fn pull_start(State(s): State<Shared>, body: Option<axum::Json<PullReq>>) -> Response {
    let Some(axum::Json(req)) = body else {
        return err(
            StatusCode::BAD_REQUEST,
            "falta el cuerpo {\"name\": \"<modelo>\"}",
        );
    };
    let Some(m) = s.catalog.all().into_iter().find(|m| m.name == req.name) else {
        return err(
            StatusCode::NOT_FOUND,
            format!("{:?} no está en el catálogo", req.name),
        );
    };
    let Some(pre) = m.prebuilt.clone() else {
        return err(
            StatusCode::BAD_REQUEST,
            format!(
                "{}: no hay pesos convertidos para descargar; usá `brasa pull --desde-fuente {}` \
                 y `brasa convert`",
                m.name, m.name
            ),
        );
    };
    let base = current_dir(&s).path;
    let dest = base.join(&m.name);
    if dest.join("model.brasa").is_file() {
        return err(
            StatusCode::CONFLICT,
            format!("{} ya está en disco ({})", m.name, dest.display()),
        );
    }
    if let Err(e) = std::fs::create_dir_all(&base) {
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("{}: {e}", base.display()),
        );
    }
    let total = pre.total_bytes();
    // Chequeo previo (ADR 0034): lo que falta bajar tiene que entrar sin usar la reserva. Lo ya
    // bajado (archivos completos y `.part`) no se vuelve a pedir.
    let need = storage::remaining_bytes(&dest, &pre.files);
    if let Err(e) = storage::check_space(&base, need, &s.storage) {
        return err(StatusCode::INSUFFICIENT_STORAGE, e.0);
    }
    let cancel = Arc::new(AtomicBool::new(false));
    {
        let mut st = s.pull.lock().expect("pull");
        if st.state == "running" {
            return err(
                StatusCode::CONFLICT,
                format!(
                    "ya hay una descarga en curso ({})",
                    st.name.as_deref().unwrap_or("?")
                ),
            );
        }
        *st = PullStatus {
            state: "running",
            name: Some(m.name.clone()),
            dest: Some(dest.display().to_string()),
            repo: Some(pre.repo.clone()),
            revision: Some(pre.revision.clone()),
            files: pre
                .files
                .iter()
                .map(|f| PullFile {
                    path: f.path.clone(),
                    size: f.size,
                    done: 0,
                })
                .collect(),
            done_bytes: 0,
            total_bytes: total,
            error: None,
            started_ms: Some(now_ms()),
            finished_ms: None,
            cancel: cancel.clone(),
        };
    }
    let shared = s.pull.clone();
    let endpoint = s.hf_endpoint.clone();
    std::thread::Builder::new()
        .name("brasa-pull".into())
        .spawn(move || {
            let progress = |path: &str, got: u64, _total: Option<u64>| {
                let mut st = shared.lock().expect("pull");
                if let Some(f) = st.files.iter_mut().find(|f| f.path == path) {
                    f.done = got;
                }
                st.done_bytes = st.files.iter().map(|f| f.done).sum();
            };
            let r = brasa_catalog::pull::download_prebuilt(
                &m,
                &dest,
                &endpoint,
                progress,
                Some(&cancel),
            );
            let mut st = shared.lock().expect("pull");
            st.finished_ms = Some(now_ms());
            match r {
                Ok(_) => st.state = "done",
                Err(e) if e.0 == brasa_catalog::pull::CANCELLED => st.state = "cancelled",
                Err(e) => {
                    st.state = "error";
                    st.error = Some(e.0);
                }
            }
        })
        .expect("hilo de descarga");
    (StatusCode::ACCEPTED, status_json(&s)).into_response()
}

pub async fn pull_cancel(State(s): State<Shared>) -> Response {
    {
        let st = s.pull.lock().expect("pull");
        if st.state != "running" {
            return err(StatusCode::CONFLICT, "no hay ninguna descarga en curso");
        }
        st.cancel.store(true, Ordering::Relaxed);
    }
    status_json(&s)
}

#[derive(Debug, Deserialize)]
pub struct DirReq {
    pub path: String,
}

fn dir_json(s: &Shared) -> serde_json::Value {
    let d = current_dir(s);
    json!({
        "dir": d.path.display().to_string(),
        "source": d.source.as_str(),
        "source_label": d.source.describe(),
        "config": s.config_path.display().to_string(),
    })
}

pub async fn set_dir(State(s): State<Shared>, body: Option<axum::Json<DirReq>>) -> Response {
    let Some(axum::Json(req)) = body else {
        return err(
            StatusCode::BAD_REQUEST,
            "falta el cuerpo {\"path\": \"/ruta\"}",
        );
    };
    if s.pull.lock().expect("pull").state == "running" {
        return err(
            StatusCode::CONFLICT,
            "hay una descarga en curso: cancelala o esperá a que termine",
        );
    }
    let config = s.config_path.clone();
    let res = tokio::task::spawn_blocking(move || {
        let dir =
            dirs::prepare_models_dir(&req.path).map_err(|e| (StatusCode::BAD_REQUEST, e.0))?;
        dirs::save_models_dir(&config, &dir).map_err(|e| (StatusCode::CONFLICT, e.0))?;
        Ok::<PathBuf, (StatusCode, String)>(dir)
    })
    .await;
    let prev = current_dir(&s).source;
    match res {
        Ok(Ok(dir)) => {
            *s.models_dir.write().expect("models_dir") = ModelsDir {
                path: dir,
                source: DirSource::File,
            };
            let mut body = dir_json(&s);
            if matches!(prev, DirSource::Env | DirSource::Flag) {
                body["note"] = json!(
                    "Guardada en el archivo de configuración. Este servidor ya la usa, pero \
                     mientras BRASA_MODELS esté definida, los próximos `brasa` van a usar esa."
                );
            }
            axum::Json(body).into_response()
        }
        Ok(Err((code, msg))) => err(code, msg),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

pub async fn choose_dir(State(s): State<Shared>) -> Response {
    let current = current_dir(&s).path;
    let choose = s.desktop.choose_folder;
    match tokio::task::spawn_blocking(move || choose(Some(&current))).await {
        Ok(Ok(Some(path))) => {
            axum::Json(json!({ "path": path, "cancelled": false })).into_response()
        }
        Ok(Ok(None)) => axum::Json(json!({ "path": null, "cancelled": true })).into_response(),
        Ok(Err(e)) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

pub async fn open_dir(State(s): State<Shared>) -> Response {
    let dir = current_dir(&s).path;
    let open = s.desktop.open;
    let res = tokio::task::spawn_blocking(move || {
        std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        open(&dir)
    })
    .await;
    match res {
        Ok(Ok(())) => axum::Json(dir_json(&s)).into_response(),
        Ok(Err(e)) => err(StatusCode::INTERNAL_SERVER_ERROR, e),
        Err(e) => err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}
