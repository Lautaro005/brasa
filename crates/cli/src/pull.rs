//! `brasa pull <modelo>`: descarga los archivos del manifiesto desde Hugging Face, con
//! reanudación y verificación de sha256 (U3).

use std::path::PathBuf;

use brasa_catalog::manifest::Manifest;
use brasa_catalog::pull;
use clap::Args;

use crate::run::models_dir;

#[derive(Debug, Args)]
pub struct PullArgs {
    /// Nombre del manifiesto (por ejemplo `qwen3-4b-q4`).
    model: String,
    /// Carpeta de destino (por defecto `models/<hf_dir>`).
    #[arg(long)]
    dir: Option<PathBuf>,
    /// Endpoint de Hugging Face (se puede apuntar a un espejo o servidor de prueba).
    #[arg(long, default_value = "https://huggingface.co")]
    endpoint: String,
    /// Solo muestra los archivos del manifiesto, sin descargar.
    #[arg(long)]
    dry_run: bool,
    /// Salida en JSON.
    #[arg(long)]
    json: bool,
}

pub fn run(a: PullArgs) -> Result<(), String> {
    let m = Manifest::find(&a.model).map_err(|e| e.0)?;
    let dest = a.dir.unwrap_or_else(|| models_dir().join(&m.hf_dir));
    if a.dry_run {
        let total: u64 = m.files.iter().filter_map(|f| f.size).sum();
        if a.json {
            let files: Vec<_> = m
                .files
                .iter()
                .map(|f| serde_json::json!({"path": f.path, "size": f.size, "sha256": f.sha256}))
                .collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "model": m.name,
                    "repo": m.source_repo,
                    "revision": m.source_revision,
                    "dest": dest.display().to_string(),
                    "files": files,
                    "total_bytes": total,
                }))
                .unwrap()
            );
        } else {
            println!(
                "{} @ {} -> {}",
                m.source_repo,
                short(&m.source_revision),
                dest.display()
            );
            for f in &m.files {
                println!(
                    "  {:<40} {:>10}  {}…",
                    f.path,
                    human(f.size.unwrap_or(0)),
                    // El manifiesto valida 64 hex al leerse, pero no se indexa a ciegas.
                    f.sha256.get(..12).unwrap_or(&f.sha256)
                );
            }
            println!("  total: {}", human(total));
        }
        return Ok(());
    }

    eprintln!(
        "descargando {} @ {} en {} …",
        m.source_repo,
        short(&m.source_revision),
        dest.display()
    );
    let mut current = String::new();
    let mut pct = u64::MAX;
    let done = pull::download(&m, &dest, &a.endpoint, |path, recv, total| {
        if path != current {
            if !current.is_empty() {
                eprintln!();
            }
            eprint!("  {path} ");
            current = path.to_string();
            pct = u64::MAX;
        }
        if let Some(t) = total.filter(|t| *t > 0) {
            let p = recv * 100 / t;
            if p != pct && (p == 100 || p / 5 != pct / 5) {
                eprint!("{p}% ");
                pct = p;
            }
        }
    })
    .map_err(|e| e.0)?;
    eprintln!();
    let total: u64 = done
        .iter()
        .map(|p| std::fs::metadata(p).map_or(0, |m| m.len()))
        .sum();
    if a.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "model": m.name,
                "dest": dest.display().to_string(),
                "files": done.len(),
                "bytes": total,
            }))
            .unwrap()
        );
    } else {
        println!(
            "listo: {} archivos, {} en {}",
            done.len(),
            human(total),
            dest.display()
        );
        println!(
            "siguiente: brasa convert {} {}",
            dest.display(),
            models_dir().join(&m.name).display()
        );
    }
    Ok(())
}

fn short(s: &str) -> String {
    s.chars().take(12).collect()
}

fn human(b: u64) -> String {
    const GIB: u64 = 1 << 30;
    if b >= GIB {
        format!("{:.2} GiB", b as f64 / GIB as f64)
    } else {
        format!("{:.1} MiB", b as f64 / (1 << 20) as f64)
    }
}
