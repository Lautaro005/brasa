//! `brasa pull <modelo>`: descarga un modelo del catálogo con reanudación y verificación de sha256
//! (U3). Por defecto baja los pesos ya convertidos (`model.brasa` y tokenizers) a la carpeta del
//! modelo (ADR 0031); con `--desde-fuente`, los safetensors de origen para `brasa convert`.

use std::path::PathBuf;

use brasa_catalog::manifest::{FileSpec, Manifest};
use brasa_catalog::{pull, storage};
use clap::Args;

use crate::config::Config;
use crate::run::models_dir;

#[derive(Debug, Args)]
pub struct PullArgs {
    /// Nombre del manifiesto (por ejemplo `qwen3-4b-q4`).
    model: String,
    /// Carpeta de destino (por defecto `<carpeta de modelos>/<modelo>`, o `<carpeta de
    /// modelos>/<hf_dir>` con --desde-fuente).
    #[arg(long)]
    dir: Option<PathBuf>,
    /// Endpoint de Hugging Face (se puede apuntar a un espejo o servidor de prueba).
    #[arg(long, default_value = pull::HF_ENDPOINT)]
    endpoint: String,
    /// Baja los safetensors de origen en vez de los pesos convertidos (después, `brasa convert`).
    #[arg(long)]
    desde_fuente: bool,
    /// Solo muestra los archivos del manifiesto, sin descargar.
    #[arg(long)]
    dry_run: bool,
    /// Salida en JSON.
    #[arg(long)]
    json: bool,
}

/// Qué baja `pull` para un manifiesto.
struct Plan<'a> {
    repo: &'a str,
    revision: &'a str,
    subdir: &'a str,
    files: &'a [FileSpec],
    dest: PathBuf,
}

fn plan<'a>(m: &'a Manifest, a: &PullArgs) -> Result<Plan<'a>, String> {
    if a.desde_fuente || m.prebuilt.is_none() {
        if a.desde_fuente && !m.convertible {
            return Err(format!(
                "{}: `brasa convert` no produce este modelo (su cuantización es {}); bajalo ya \
                 convertido con `brasa pull {}`",
                m.name, m.quant, m.name
            ));
        }
        let dest = match &a.dir {
            Some(d) => d.clone(),
            None => models_dir()?.join(&m.hf_dir),
        };
        return Ok(Plan {
            repo: &m.source_repo,
            revision: &m.source_revision,
            subdir: "",
            files: &m.files,
            dest,
        });
    }
    let p = m.prebuilt.as_ref().expect("prebuilt");
    let dest = match &a.dir {
        Some(d) => d.clone(),
        None => models_dir()?.join(&m.name),
    };
    Ok(Plan {
        repo: &p.repo,
        revision: &p.revision,
        subdir: &p.subdir,
        files: &p.files,
        dest,
    })
}

pub fn run(a: PullArgs) -> Result<(), String> {
    let m = Manifest::find(&a.model).map_err(|e| e.0)?;
    let p = plan(&m, &a)?;
    let from_source = a.desde_fuente || m.prebuilt.is_none();
    let origin = if p.subdir.is_empty() {
        format!("{} @ {}", p.repo, short(p.revision))
    } else {
        format!("{} @ {} ({})", p.repo, short(p.revision), p.subdir)
    };
    let total: u64 = p.files.iter().filter_map(|f| f.size).sum();
    // Chequeo previo de espacio (ADR 0034): lo que falta bajar tiene que entrar sin usar la
    // reserva. Lo que ya está en disco (archivos completos y `.part`) no se vuelve a pedir.
    let settings = Config::load()?.storage()?;
    let remaining = storage::remaining_bytes(&p.dest, p.files);
    let volume = storage::volume(&p.dest);
    let available = volume.map(|v| v.free_bytes.saturating_sub(settings.reserve_bytes));
    if a.dry_run {
        if a.json {
            let files: Vec<_> = p
                .files
                .iter()
                .map(|f| serde_json::json!({"path": f.path, "size": f.size, "sha256": f.sha256}))
                .collect();
            println!(
                "{}",
                serde_json::to_string_pretty(&serde_json::json!({
                    "model": m.name,
                    "source": if from_source { "fuente" } else { "convertido" },
                    "repo": p.repo,
                    "revision": p.revision,
                    "subdir": p.subdir,
                    "dest": p.dest.display().to_string(),
                    "files": files,
                    "total_bytes": total,
                    "remaining_bytes": remaining,
                    "free_bytes": volume.map(|v| v.free_bytes),
                    "reserve_bytes": settings.reserve_bytes,
                    "available_bytes": available,
                }))
                .unwrap()
            );
        } else {
            println!("{origin} -> {}", p.dest.display());
            for f in p.files {
                println!(
                    "  {:<40} {:>10}  {}…",
                    f.path,
                    human(f.size.unwrap_or(0)),
                    // El manifiesto valida 64 hex al leerse, pero no se indexa a ciegas.
                    f.sha256.get(..12).unwrap_or(&f.sha256)
                );
            }
            println!("  total: {}", human(total));
            println!(
                "  falta bajar: {}; disponible: {}",
                human(remaining),
                available.map_or_else(
                    || "no se pudo leer".into(),
                    |b| format!(
                        "{} (libre menos la reserva de {})",
                        human(b),
                        human(settings.reserve_bytes)
                    )
                )
            );
        }
        return Ok(());
    }
    storage::check_space(&p.dest, remaining, &settings).map_err(|e| e.0)?;

    if !from_source && m.prebuilt.as_ref().is_some_and(|p| !p.is_pinned()) {
        eprintln!(
            "aviso: la revisión {:?} no es un commit fijo; cada archivo se verifica igual por sha256",
            p.revision
        );
    }
    eprintln!("descargando {origin} en {} …", p.dest.display());
    let mut current = String::new();
    let mut pct = u64::MAX;
    let progress = |path: &str, recv: u64, total: Option<u64>| {
        if path != current {
            if !current.is_empty() {
                eprintln!();
            }
            eprint!("  {path} ");
            current = path.to_string();
            pct = u64::MAX;
        }
        if let Some(t) = total.filter(|t| *t > 0) {
            let q = recv * 100 / t;
            if q != pct && (q == 100 || q / 5 != pct / 5) {
                eprint!("{q}% ");
                pct = q;
            }
        }
    };
    let remote = pull::Remote {
        repo: p.repo,
        revision: p.revision,
        subdir: p.subdir,
    };
    let done = pull::download_set(&a.endpoint, remote, p.files, &p.dest, progress, None)
        .map_err(|e| e.0)?;
    eprintln!();
    let bytes: u64 = done
        .iter()
        .map(|f| std::fs::metadata(f).map_or(0, |m| m.len()))
        .sum();
    if a.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "model": m.name,
                "source": if from_source { "fuente" } else { "convertido" },
                "dest": p.dest.display().to_string(),
                "files": done.len(),
                "bytes": bytes,
            }))
            .unwrap()
        );
    } else {
        println!(
            "listo: {} archivos, {} en {}",
            done.len(),
            human(bytes),
            p.dest.display()
        );
        if from_source {
            println!(
                "siguiente: brasa convert {} {}",
                p.dest.display(),
                models_dir()?.join(&m.name).display()
            );
        } else {
            println!("siguiente: brasa serve {}", m.name);
        }
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
