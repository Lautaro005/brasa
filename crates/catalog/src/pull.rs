//! Descarga de los archivos de un manifiesto desde Hugging Face (ADR 0020), con reanudación por
//! `Range` y verificación de sha256. Si un archivo no coincide, se borra el parcial y se falla
//! con un mensaje claro.
//!
//! Dos fuentes (ADR 0031): los pesos ya convertidos de `[prebuilt]` ([`download_prebuilt`]) o los
//! safetensors de origen ([`download`]). Las dos aceptan una bandera de cancelación: al
//! cancelar, el `.part` queda en disco para reanudar.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::manifest::{FileSpec, Manifest};
use crate::{Error, Result};

/// Endpoint por defecto de Hugging Face.
pub const HF_ENDPOINT: &str = "https://huggingface.co";

/// Agente `ureq` con timeouts de conexión y de espera de la respuesta (evita colgarse en una red
/// lenta). En ureq 3, `timeout_recv_body` es un presupuesto total para el cuerpo y no se reinicia
/// en cada lectura: no se usa, porque cortaría la descarga de archivos de varios GB.
fn agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(10)))
        .timeout_recv_response(Some(Duration::from_secs(60)))
        .build()
        .into()
}

/// URL de un archivo en una revisión fija (layout de HF).
pub fn file_url(endpoint: &str, repo: &str, revision: &str, path: &str) -> String {
    format!(
        "{}/{repo}/resolve/{revision}/{path}",
        endpoint.trim_end_matches('/')
    )
}

fn io_err(path: &Path, e: std::io::Error) -> Error {
    Error(format!("{}: {e}", path.display()))
}

/// Lee todo `r` actualizando `hasher`; devuelve los bytes leídos.
fn hash_into(r: &mut impl Read, hasher: &mut Sha256) -> Result<u64> {
    let mut buf = vec![0u8; 1 << 20];
    let mut total = 0u64;
    loop {
        let n = r
            .read(&mut buf)
            .map_err(|e| Error(format!("error leyendo: {e}")))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total += n as u64;
    }
    Ok(total)
}

fn hash_file(path: &Path) -> Result<(String, u64)> {
    let mut f = File::open(path).map_err(|e| io_err(path, e))?;
    let mut hasher = Sha256::new();
    let total = hash_into(&mut f, &mut hasher)?;
    Ok((hex(&hasher.finalize()), total))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn net_err(e: ureq::Error) -> Error {
    match e {
        ureq::Error::StatusCode(code) => Error(format!("HTTP {code}")),
        e => Error(format!("error de red: {e}")),
    }
}

type Response = ureq::http::Response<ureq::Body>;

/// Pide el archivo; si `have > 0`, con `Range`. Si el servidor responde 416 (no acepta
/// reanudar), se vuelve a pedir completo.
fn call(agent: &ureq::Agent, url: &str, have: u64) -> Result<Response> {
    let req = if have > 0 {
        agent.get(url).header("Range", &format!("bytes={have}-"))
    } else {
        agent.get(url)
    };
    match req.call() {
        Ok(r) => Ok(r),
        Err(ureq::Error::StatusCode(416)) if have > 0 => agent.get(url).call().map_err(net_err),
        Err(e) => Err(net_err(e)),
    }
}

/// Mensaje del error que devuelve una descarga cancelada.
pub const CANCELLED: &str = "descarga cancelada";

/// Origen de un conjunto de archivos en Hugging Face.
#[derive(Debug, Clone, Copy)]
pub struct Remote<'a> {
    pub repo: &'a str,
    pub revision: &'a str,
    /// Subcarpeta del repo (vacía: la raíz).
    pub subdir: &'a str,
}

/// Descarga los safetensors de origen del manifiesto en `dest`. `progress(path, recibido, total)`.
pub fn download(
    manifest: &Manifest,
    dest: &Path,
    endpoint: &str,
    progress: impl FnMut(&str, u64, Option<u64>),
) -> Result<Vec<PathBuf>> {
    let remote = Remote {
        repo: &manifest.source_repo,
        revision: &manifest.source_revision,
        subdir: "",
    };
    download_set(endpoint, remote, &manifest.files, dest, progress, None)
}

/// Descarga los pesos ya convertidos (`[prebuilt]`) en `dest` (la carpeta del modelo), en el orden
/// del manifiesto. Falla si el manifiesto no los tiene.
pub fn download_prebuilt(
    manifest: &Manifest,
    dest: &Path,
    endpoint: &str,
    progress: impl FnMut(&str, u64, Option<u64>),
    cancel: Option<&AtomicBool>,
) -> Result<Vec<PathBuf>> {
    let p = manifest.prebuilt.as_ref().ok_or_else(|| {
        Error(format!(
            "{}: el manifiesto no tiene pesos convertidos ([prebuilt])",
            manifest.name
        ))
    })?;
    let remote = Remote {
        repo: &p.repo,
        revision: &p.revision,
        subdir: &p.subdir,
    };
    download_set(endpoint, remote, &p.files, dest, progress, cancel)
}

/// Descarga `files` de `remote` en `dest`, uno por vez y en orden.
pub fn download_set(
    endpoint: &str,
    remote: Remote<'_>,
    files: &[FileSpec],
    dest: &Path,
    mut progress: impl FnMut(&str, u64, Option<u64>),
    cancel: Option<&AtomicBool>,
) -> Result<Vec<PathBuf>> {
    let mut done = Vec::new();
    for spec in files {
        done.push(download_file(
            endpoint,
            remote,
            spec,
            dest,
            &mut progress,
            cancel,
        )?);
    }
    Ok(done)
}

fn cancelled(cancel: Option<&AtomicBool>) -> bool {
    cancel.is_some_and(|c| c.load(Ordering::Relaxed))
}

/// Descarga un archivo con reanudación y verifica su sha256. Si `cancel` se activa, corta y deja
/// el `.part` para reanudar.
pub fn download_file(
    endpoint: &str,
    remote: Remote<'_>,
    spec: &FileSpec,
    dest: &Path,
    progress: &mut impl FnMut(&str, u64, Option<u64>),
    cancel: Option<&AtomicBool>,
) -> Result<PathBuf> {
    let repo = remote.repo;
    crate::manifest::safe_relative(&spec.path).map_err(|e| Error(format!("{repo}: {e}")))?;
    if !remote.subdir.is_empty() {
        crate::manifest::safe_relative(remote.subdir).map_err(|e| Error(format!("{repo}: {e}")))?;
    }
    if cancelled(cancel) {
        return Err(Error(CANCELLED.into()));
    }
    let final_path = dest.join(&spec.path);
    if let Some(parent) = final_path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| io_err(parent, e))?;
    }
    // Ya está y coincide: no se vuelve a bajar.
    if final_path.is_file() {
        let (h, n) = hash_file(&final_path)?;
        if h == spec.sha256 {
            progress(&spec.path, n, spec.size);
            return Ok(final_path);
        }
    }
    let part = dest.join(format!("{}.part", spec.path));
    let mut hasher = Sha256::new();
    let mut have = 0u64;
    if part.is_file() {
        // El estado de sha2 no se persiste: se rehashea el parcial y se sigue desde ahí.
        let mut f = File::open(&part).map_err(|e| io_err(&part, e))?;
        have = hash_into(&mut f, &mut hasher)?;
    }
    progress(&spec.path, have, spec.size);
    let repo_path = if remote.subdir.is_empty() {
        spec.path.clone()
    } else {
        format!("{}/{}", remote.subdir, spec.path)
    };
    let url = file_url(endpoint, repo, remote.revision, &repo_path);
    let agent = agent();
    let mut resp = call(&agent, &url, have)?;
    let status = resp.status().as_u16();
    let mut file;
    if status == 206 {
        // Con 206, el servidor tiene que empezar en lo que ya tenemos; si no, se descarta.
        let expect = format!("bytes {have}-");
        let cont = resp
            .headers()
            .get("Content-Range")
            .and_then(|v| v.to_str().ok())
            .is_some_and(|cr| cr.starts_with(&expect));
        if !cont {
            hasher = Sha256::new();
            have = 0;
            std::fs::remove_file(&part).ok();
            let fresh = agent.get(&url).call().map_err(net_err)?;
            if fresh.status().as_u16() == 206 {
                return Err(Error(format!(
                    "{url}: el servidor no respondió el archivo completo"
                )));
            }
            file = File::create(&part).map_err(|e| io_err(&part, e))?;
            resp = fresh;
        } else {
            file = OpenOptions::new()
                .append(true)
                .open(&part)
                .map_err(|e| io_err(&part, e))?;
        }
    } else {
        // 200: contenido completo; si había parcial, se descarta.
        hasher = Sha256::new();
        have = 0;
        file = File::create(&part).map_err(|e| io_err(&part, e))?;
    }
    let mut reader = resp.into_body().into_reader();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        if cancelled(cancel) {
            // El `.part` queda: la próxima descarga sigue desde acá.
            file.flush().map_err(|e| io_err(&part, e))?;
            return Err(Error(CANCELLED.into()));
        }
        let n = reader
            .read(&mut buf)
            .map_err(|e| Error(format!("{url}: {e}")))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        file.write_all(&buf[..n]).map_err(|e| io_err(&part, e))?;
        have += n as u64;
        progress(&spec.path, have, spec.size);
    }
    file.flush().map_err(|e| io_err(&part, e))?;
    drop(file);
    let got = hex(&hasher.finalize());
    if got != spec.sha256 {
        std::fs::remove_file(&part).ok();
        return Err(Error(format!(
            "{}: sha256 no coincide (esperado {}, obtenido {got})",
            spec.path, spec.sha256
        )));
    }
    std::fs::rename(&part, &final_path).map_err(|e| io_err(&final_path, e))?;
    Ok(final_path)
}
