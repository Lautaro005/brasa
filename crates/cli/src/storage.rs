//! `brasa storage`: espacio de la carpeta de modelos y del volumen (ADR 0034): modelos, descargas
//! a medias con su antigüedad, lo que no se reconoce (solo se informa) y la reserva que una
//! descarga no puede usar. `brasa storage clean` borra las descargas a medias viejas: sin
//! `--apply` solo muestra qué borraría. Borrar un modelo entero sigue siendo `brasa rm`.

use std::path::PathBuf;
use std::time::SystemTime;

use brasa_catalog::dirs;
use brasa_catalog::manifest::Manifest;
use brasa_catalog::storage::{self, CleanOptions, human, human_age};
use clap::{Args, Subcommand};

use crate::config::Config;

#[derive(Debug, Args)]
pub struct StorageArgs {
    #[command(subcommand)]
    cmd: Option<StorageCmd>,
    /// Carpeta base de modelos (por defecto la efectiva; ver `brasa config show`).
    #[arg(long)]
    dir: Option<PathBuf>,
    /// Salida en JSON (la misma forma que `GET /api/storage`).
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Subcommand)]
enum StorageCmd {
    /// Borra las descargas a medias (`.part`) viejas. Sin --apply solo muestra qué borraría.
    Clean {
        /// Borra de verdad (sin esto es un dry-run).
        #[arg(long)]
        apply: bool,
        /// Antigüedad mínima en días (por defecto `partial_max_age_days` de `[storage]`).
        #[arg(long)]
        dias: Option<u64>,
        /// Carpeta base de modelos (por defecto la efectiva; ver `brasa config show`).
        #[arg(long)]
        dir: Option<PathBuf>,
        /// Salida en JSON.
        #[arg(long)]
        json: bool,
        /// Permitir que la carpeta de modelos pase por un symlink (por defecto se niega).
        #[arg(long)]
        seguir_symlink_base: bool,
    },
}

/// Carpetas de safetensors de origen del catálogo (`hf_dir`), que el inventario reconoce.
pub fn source_dirs(catalog: &[Manifest]) -> Vec<String> {
    catalog.iter().map(|m| m.hf_dir.clone()).collect()
}

pub fn run(a: StorageArgs) -> Result<(), String> {
    let settings = Config::load()?.storage()?;
    match a.cmd {
        None => status(a.dir, a.json, &settings),
        Some(StorageCmd::Clean {
            apply,
            dias,
            dir,
            json,
            seguir_symlink_base,
        }) => clean(
            dir.or(a.dir),
            json || a.json,
            apply,
            dias.unwrap_or(settings.partial_max_age_days),
            seguir_symlink_base,
        ),
    }
}

fn status(dir: Option<PathBuf>, json: bool, settings: &storage::Settings) -> Result<(), String> {
    let md = dirs::models_dir(dir).map_err(|e| e.0)?;
    let catalog = Manifest::all().unwrap_or_default();
    let r = storage::report(&md, settings, &source_dirs(&catalog), SystemTime::now());
    if json {
        println!("{}", serde_json::to_string_pretty(&r).unwrap());
        return Ok(());
    }
    println!(
        "carpeta    {} ({}{})",
        r.dir,
        r.dir_source,
        if r.exists { "" } else { "; todavía no existe" }
    );
    match r.volume {
        Some(v) => println!(
            "volumen    {} libres de {} ({} usados)",
            human(v.free_bytes),
            human(v.total_bytes),
            human(v.total_bytes.saturating_sub(v.free_bytes))
        ),
        None => println!("volumen    no se pudo leer"),
    }
    println!(
        "reserva    {} (las descargas no la usan; [storage] reserve_gib)",
        human(r.reserve_bytes)
    );
    if let Some(a) = r.available_bytes {
        println!("disponible {} para descargas", human(a));
    }
    println!();
    println!("modelos    {} · {}", r.models.len(), human(r.models_bytes));
    for m in &r.models {
        println!("  {:<36} {:>11}", m.name, human(m.bytes));
    }
    println!(
        "parciales  {} · {} ({} con {} días o más: {})",
        r.partials.len(),
        human(r.partial_bytes),
        r.partials.iter().filter(|p| p.old).count(),
        r.partial_max_age_days,
        human(r.old_partial_bytes)
    );
    for p in &r.partials {
        println!(
            "  {:<36} {:>11}   hace {}{}",
            p.path,
            human(p.bytes),
            human_age(p.age_secs),
            if p.old { "   vieja" } else { "" }
        );
    }
    if !r.other.is_empty() {
        println!(
            "otros      {} · {} (no se tocan)",
            r.other.len(),
            human(r.other_bytes)
        );
        for o in &r.other {
            let size = if o.target.is_some() {
                String::new()
            } else {
                human(o.bytes)
            };
            println!(
                "  {:<36} {:>11}   {}{}",
                o.name,
                size,
                o.label,
                o.target
                    .as_deref()
                    .map(|t| format!(" -> {t}"))
                    .unwrap_or_default()
            );
        }
    }
    if r.old_partial_bytes > 0 {
        println!();
        println!("para ver qué se borraría: brasa storage clean (con --apply borra)");
    }
    Ok(())
}

fn clean(
    dir: Option<PathBuf>,
    json: bool,
    apply: bool,
    days: u64,
    follow: bool,
) -> Result<(), String> {
    let md = dirs::models_dir(dir).map_err(|e| e.0)?;
    let opts = CleanOptions {
        max_age_days: days,
        apply,
        follow_base_symlink: follow,
        exclude: &[],
        now: SystemTime::now(),
    };
    let r = storage::clean_partials(&md.path, &opts).map_err(|e| e.0)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&r).unwrap());
        return Ok(());
    }
    let verb = if apply { "borrado" } else { "se borraría" };
    for i in &r.items {
        println!(
            "{verb}  {:<36} {:>11}   hace {}",
            i.path,
            human(i.bytes),
            human_age(i.age_secs)
        );
    }
    for d in &r.removed_dirs {
        println!("borrada carpeta vacía {d}");
    }
    for s in &r.skipped {
        println!("sin tocar  {}: {}", s.path, s.reason);
    }
    if r.items.is_empty() {
        println!(
            "no hay descargas a medias con {} días o más en {}",
            r.older_than_days, r.dir
        );
    } else if apply {
        println!(
            "liberado: {} en {}",
            human(r.bytes),
            archivos(r.items.len())
        );
    } else {
        println!(
            "total: {} en {} (dry-run: nada se borró; con --apply se borran)",
            human(r.bytes),
            archivos(r.items.len())
        );
    }
    if r.kept > 0 {
        println!(
            "sin tocar por ser más nuevos que el umbral: {} ({}), para reanudar",
            archivos(r.kept),
            human(r.kept_bytes)
        );
    }
    Ok(())
}

fn archivos(n: usize) -> String {
    if n == 1 {
        "1 archivo".into()
    } else {
        format!("{n} archivos")
    }
}
