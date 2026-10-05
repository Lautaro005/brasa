//! Utilidades de medición: ejecutar bajo `/usr/bin/time -l`, hashes, commit y fecha.

use std::io::Read;
use std::path::Path;
use std::process::Command;

use sha2::{Digest, Sha256};

use crate::{Result, err};

/// Memoria pico de un proceso hijo según `/usr/bin/time -l`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChildMemory {
    pub peak_rss: u64,
    pub peak_footprint: u64,
}

/// Salida de un proceso medido.
#[derive(Debug)]
pub struct TimedOutput {
    pub stdout: String,
    /// stderr del proceso, sin el bloque que agrega `time`.
    pub stderr: String,
    pub memory: ChildMemory,
}

/// Ejecuta `program args` bajo `/usr/bin/time -l` y devuelve salida y memoria pico.
pub fn run_timed(program: &str, args: &[String]) -> Result<TimedOutput> {
    let out = Command::new("/usr/bin/time")
        .arg("-l")
        .arg(program)
        .args(args)
        .output()
        .map_err(|e| crate::BenchError(format!("no se pudo ejecutar {program}: {e}")))?;
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    if !out.status.success() {
        let tail: Vec<&str> = stderr.lines().rev().take(20).collect();
        let tail: Vec<&str> = tail.into_iter().rev().collect();
        return err(format!(
            "{program} terminó con {}:\n{}",
            out.status,
            tail.join("\n")
        ));
    }
    let memory = parse_time_l(&stderr)?;
    Ok(TimedOutput {
        stdout,
        stderr,
        memory,
    })
}

/// Lee "maximum resident set size" y "peak memory footprint" de la salida de `time -l`.
pub fn parse_time_l(stderr: &str) -> Result<ChildMemory> {
    let field = |name: &str| {
        stderr.lines().rev().find_map(|l| {
            let l = l.trim();
            l.strip_suffix(name)
                .and_then(|n| n.trim().parse::<u64>().ok())
        })
    };
    match (
        field("maximum resident set size"),
        field("peak memory footprint"),
    ) {
        (Some(peak_rss), Some(peak_footprint)) => Ok(ChildMemory {
            peak_rss,
            peak_footprint,
        }),
        _ => err("no se encontró la memoria pico en la salida de /usr/bin/time -l"),
    }
}

/// SHA-256 de un archivo o, si es un directorio, de sus archivos de pesos y configuración
/// (`*.safetensors`, `*.json`) en orden por nombre: sha256(nombre || sha256(archivo) ...).
pub fn sha256_path(path: &Path) -> Result<(String, u64)> {
    if path.is_file() {
        return sha256_file(path);
    }
    let mut entries: Vec<_> = std::fs::read_dir(path)
        .map_err(|e| crate::BenchError(format!("{}: {e}", path.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.is_file()
                && matches!(
                    p.extension().and_then(|x| x.to_str()),
                    Some("safetensors" | "json")
                )
        })
        .collect();
    entries.sort();
    if entries.is_empty() {
        return err(format!("{}: no hay pesos", path.display()));
    }
    let mut outer = Sha256::new();
    let mut total = 0;
    for p in entries {
        let (h, n) = sha256_file(&p)?;
        outer.update(p.file_name().unwrap().as_encoded_bytes());
        outer.update(h.as_bytes());
        total += n;
    }
    Ok((hex(&outer.finalize()), total))
}

fn sha256_file(path: &Path) -> Result<(String, u64)> {
    let mut f = std::fs::File::open(path)
        .map_err(|e| crate::BenchError(format!("{}: {e}", path.display())))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    let mut total = 0u64;
    loop {
        let n = f
            .read(&mut buf)
            .map_err(|e| crate::BenchError(format!("{}: {e}", path.display())))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
        total += n as u64;
    }
    Ok((hex(&h.finalize()), total))
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Commit actual de Brasa (`git rev-parse`), con `-dirty` si hay cambios sin commitear.
pub fn brasa_commit() -> String {
    let git = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    };
    let Some(commit) = git(&["rev-parse", "--short=12", "HEAD"]) else {
        return "desconocido".into();
    };
    let dirty =
        git(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty());
    if dirty {
        format!("{commit}-dirty")
    } else {
        commit
    }
}

/// Fecha y hora UTC actual en ISO 8601 (`2026-10-04T23:55:00Z`).
pub fn utc_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let (days, rem) = (secs / 86_400, secs % 86_400);
    let (y, m, d) = civil_from_days(days as i64);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem / 60 % 60,
        rem % 60
    )
}

/// Días desde 1970-01-01 a fecha civil (algoritmo de H. Hinnant).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_time_output() {
        let s = "algo\n          2787213312  maximum resident set size\n                   0  average shared memory size\n           392137512  peak memory footprint\n";
        assert_eq!(
            parse_time_l(s).unwrap(),
            ChildMemory {
                peak_rss: 2787213312,
                peak_footprint: 392137512
            }
        );
        assert!(parse_time_l("nada").is_err());
    }

    #[test]
    fn civil_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(20_730), (2026, 10, 4));
        assert_eq!(civil_from_days(11_016), (2000, 2, 29));
    }

    #[test]
    fn sha256_known() {
        assert_eq!(
            sha256_bytes(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
