//! Descarga de los archivos de un manifiesto desde Hugging Face (ADR 0020), con reanudación por
//! `Range` y verificación de sha256. Si un archivo no coincide, se borra el parcial y se falla
//! con un mensaje claro.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::manifest::{FileSpec, Manifest};
use crate::{Error, Result};

/// Endpoint por defecto de Hugging Face.
pub const HF_ENDPOINT: &str = "https://huggingface.co";

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
    let mut req = ureq::get(&url);
    if have > 0 {
        req = req.set("Range", &format!("bytes={have}-"));
    }
    let resp = match req.call() {
        Ok(r) => r,
        Err(ureq::Error::Status(416, _)) if have > 0 => {
            // El servidor no acepta reanudar: se empieza de cero.
            std::fs::remove_file(&part).ok();
            hasher = Sha256::new();
            have = 0;
            ureq::get(&url).call().map_err(net_err)?
        }
        Err(e) => return Err(net_err(e)),
    };
    let status = resp.status();
    let mut file = if status == 206 {
        OpenOptions::new()
            .append(true)
            .open(&part)
            .map_err(|e| io_err(&part, e))?
    } else {
        // 200: contenido completo; si había parcial, se descarta.
        hasher = Sha256::new();
        have = 0;
        File::create(&part).map_err(|e| io_err(&part, e))?
    };
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
