//! `brasa rm <modelo>`: borra la carpeta de un modelo local, pidiendo confirmación (U3).

use std::io::Write;
use std::path::PathBuf;

use brasa_catalog::local;
use clap::Args;

use crate::run::models_dir;

#[derive(Debug, Args)]
pub struct RmArgs {
    model: String,
    /// Carpeta base de modelos.
    #[arg(long)]
    dir: Option<PathBuf>,
    /// No pedir confirmación.
    #[arg(short, long)]
    yes: bool,
    /// Permitir que la carpeta de modelos pase por un symlink (por defecto `rm` se niega).
    #[arg(long)]
    seguir_symlink_base: bool,
}

pub fn run(a: RmArgs) -> Result<(), String> {
    let base = a.dir.unwrap_or_else(models_dir);
    // Solo subcarpetas directas de `models_dir`, sin rutas, `..` ni symlinks: `rm` borra.
    // Si la base es (o pasa por) un symlink, se niega salvo --seguir-symlink-base.
    let dir = local::resolve_child_with(&base, &a.model, a.seguir_symlink_base).map_err(|e| e.0)?;
    let bytes = std::fs::metadata(dir.join("model.brasa")).map_or(0, |m| m.len());
    if !a.yes {
        eprint!(
            "¿Borrar {} ({:.2} GiB)? [s/N] ",
            dir.display(),
            bytes as f64 / (1u64 << 30) as f64
        );
        std::io::stderr().flush().ok();
        let mut line = String::new();
        std::io::stdin()
            .read_line(&mut line)
            .map_err(|e| e.to_string())?;
        let t = line.trim().to_lowercase();
        if !matches!(t.as_str(), "s" | "si" | "sí" | "y" | "yes") {
            println!("cancelado");
            return Ok(());
        }
    }
    std::fs::remove_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    println!("borrado {}", dir.display());
    Ok(())
}
