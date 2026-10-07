//! `brasa models`: lista los modelos locales y verifica sus hashes (U3).

use std::path::{Path, PathBuf};

use brasa_catalog::{local, verify};
use brasa_memory::planner::{Budget, Fit, gib};
use brasa_runtime::{KvType, Limits, Session};
use clap::{Args, Subcommand};
use serde_json::json;

use crate::run::models_dir;

#[derive(Debug, Args)]
pub struct ModelsArgs {
    #[command(subcommand)]
    cmd: Option<ModelsCmd>,
    /// Salida en JSON.
    #[arg(long)]
    json: bool,
    /// Carpeta base de modelos (por defecto la efectiva; ver `brasa config show`).
    #[arg(long)]
    dir: Option<PathBuf>,
    /// Verifica el sha256 de cada modelo listado (lee los archivos enteros).
    #[arg(long)]
    verify: bool,
}

#[derive(Debug, Subcommand)]
enum ModelsCmd {
    /// Recalcula los sha256 de un modelo contra su `.brasa`.
    Verify {
        model: String,
        #[arg(long)]
        json: bool,
    },
}

pub fn run(a: ModelsArgs) -> Result<(), String> {
    let dir = match a.dir {
        Some(d) => d,
        None => models_dir()?,
    };
    match a.cmd {
        Some(ModelsCmd::Verify { model, json }) => verify_model(&dir, &model, json),
        None => list(&dir, a.json, a.verify),
    }
}

/// Contexto por defecto para la comprobación del planner.
const DEFAULT_CTX: usize = 4096;

fn fits(dir: &Path) -> Option<bool> {
    let budget = Budget::this_machine()?;
    let limits = Limits {
        ctx: DEFAULT_CTX,
        max_tokens: 128,
        max_logit_rows: 1,
        kv: KvType::F16,
    };
    match Session::plan(dir, limits, &budget) {
        Ok((Fit::Fits(_), _)) => Some(true),
        Ok((Fit::TooBig { .. }, _)) => Some(false),
        Err(_) => None,
    }
}

fn list(dir: &Path, json: bool, full_verify: bool) -> Result<(), String> {
    let models = local::scan(dir);
    if json {
        let arr: Vec<_> = models
            .iter()
            .map(|m| {
                let verified = if full_verify {
                    verify::verify(&m.weights).map(|_| true).ok()
                } else {
                    None
                };
                json!({
                    "name": m.name,
                    "path": m.weights.display().to_string(),
                    "bytes": m.bytes,
                    "data_sha256": m.data_sha256,
                    "structure_ok": m.ok,
                    "error": m.error,
                    "verified": verified,
                    "fits_default_ctx": fits(&m.dir),
                    "default_ctx": DEFAULT_CTX,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&arr).unwrap());
        return Ok(());
    }
    if models.is_empty() {
        println!("no hay modelos en {}", dir.display());
        return Ok(());
    }
    println!(
        "{:<22} {:>9}  {:<18} {:<10} {:<10} EN MEMORIA",
        "MODELO", "TAMAÑO", "SHA256(encabezado)", "ESTRUCTURA", "VERIFICADO"
    );
    for m in &models {
        let verified = if full_verify {
            match verify::verify(&m.weights) {
                Ok(_) => "sí".to_string(),
                Err(e) => format!("no ({})", e.0),
            }
        } else {
            "no".to_string()
        };
        let mem = match fits(&m.dir) {
            Some(true) => format!("sí (ctx {DEFAULT_CTX})"),
            Some(false) => "no".to_string(),
            None => "—".to_string(),
        };
        let sha: String = m.data_sha256.chars().take(16).collect();
        println!(
            "{:<22} {:>9}  {:<18} {:<10} {:<10} {}",
            m.name,
            format!("{:.2} GiB", m.bytes as f64 / (1u64 << 30) as f64),
            format!("{sha}…"),
            if m.ok { "ok" } else { "error" },
            verified,
            mem
        );
    }
    if !full_verify {
        println!("\n(verificado: no; corré `brasa models --verify` para recalcular los sha256)");
    }
    Ok(())
}

fn verify_model(dir: &Path, model: &str, json: bool) -> Result<(), String> {
    let m = local::resolve(dir, model).map_err(|e| e.0)?;
    let weights = m.join("model.brasa");
    let r = verify::verify(&weights).map_err(|e| e.0)?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "model": model,
                "path": r.path,
                "tensors": r.tensors,
                "weights_bytes": r.bytes,
                "data_sha256": r.data_sha256,
                "ok": true,
            }))
            .unwrap()
        );
    } else {
        println!(
            "{}: ok — {} tensores, {:.2} GiB, sha256 {}",
            model,
            r.tensors,
            gib(r.bytes),
            r.data_sha256
        );
    }
    Ok(())
}
