//! Descarga de los archivos de un manifiesto desde Hugging Face (ADR 0020), con reanudación por
//! `Range` y verificación de sha256. Si un archivo no coincide, se borra el parcial y se falla
//! con un mensaje claro.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::manifest::{FileSpec, Manifest};
use crate::{Error, Result};

/// Endpoint por defecto de Hugging Face.
pub const HF_ENDPOINT: &str = "https://huggingface.co";

/// Agente `ureq` con timeouts de conexión y de lectura (evita colgarse en una red lenta).
fn agent() -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(10))
        .timeout_read(Duration::from_secs(60))
        .build()
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
        ureq::Error::Status(code, r) => {
            let body = r.into_string().unwrap_or_default();
            Error(format!("HTTP {code}: {}", body.trim()))
        }
        ureq::Error::Transport(t) => Error(format!("error de red: {t}")),
    }
}

/// Pide el archivo; si `have > 0`, con `Range`. Si el servidor responde 416 (no acepta
/// reanudar), se vuelve a pedir completo.
fn call(agent: &ureq::Agent, url: &str, have: u64) -> Result<ureq::Response> {
    let req = if have > 0 {
        agent.get(url).set("Range", &format!("bytes={have}-"))
    } else {
        agent.get(url)
    };
    match req.call() {
        Ok(r) => Ok(r),
        Err(ureq::Error::Status(416, _)) if have > 0 => agent.get(url).call().map_err(net_err),
        Err(e) => Err(net_err(e)),
    }
}

/// Descarga todos los archivos del manifiesto en `dest`. `progress(path, recibido, total)`.
pub fn download(
    manifest: &Manifest,
    dest: &Path,
    endpoint: &str,
    mut progress: impl FnMut(&str, u64, Option<u64>),
) -> Result<Vec<PathBuf>> {
    let mut done = Vec::new();
    for spec in &manifest.files {
        let p = download_file(
            endpoint,
            &manifest.source_repo,
            &manifest.source_revision,
            spec,
            dest,
            &mut progress,
        )?;
        done.push(p);
    }
    Ok(done)
}

/// Descarga un archivo con reanudación y verifica su sha256.
pub fn download_file(
    endpoint: &str,
    repo: &str,
    revision: &str,
    spec: &FileSpec,
    dest: &Path,
    progress: &mut impl FnMut(&str, u64, Option<u64>),
) -> Result<PathBuf> {
    crate::manifest::safe_relative(&spec.path).map_err(|e| Error(format!("{repo}: {e}")))?;
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
    let url = file_url(endpoint, repo, revision, &spec.path);
    let agent = agent();
    let mut resp = call(&agent, &url, have)?;
    let status = resp.status();
    let mut file;
    if status == 206 {
        // Con 206, el servidor tiene que empezar en lo que ya tenemos; si no, se descarta.
        let expect = format!("bytes {have}-");
        let cont = resp
            .header("Content-Range")
            .is_some_and(|cr| cr.starts_with(&expect));
        if !cont {
            hasher = Sha256::new();
            have = 0;
            std::fs::remove_file(&part).ok();
            let fresh = agent.get(&url).call().map_err(net_err)?;
            if fresh.status() == 206 {
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
    let mut reader = resp.into_reader();
    let mut buf = vec![0u8; 1 << 20];
    loop {
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
