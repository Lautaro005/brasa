//! `brasa convert <dir-hf> <dir-salida>`: conversión nativa a `.brasa` sin Python (U4).

use std::path::{Path, PathBuf};

use brasa_catalog::manifest::Manifest;
use brasa_quant::convert::convert;
use clap::Args;

#[derive(Debug, Args)]
pub struct ConvertArgs {
    /// Carpeta con los safetensors de Hugging Face (config.json, index.json y shards).
    src: PathBuf,
    /// Carpeta de salida (`model.brasa`, `tokenizer.json`, `tokenizer_config.json`).
    dst: PathBuf,
    /// Repo de origen (por defecto, el del manifiesto del modelo).
    #[arg(long)]
    source_repo: Option<String>,
    /// Revisión de origen (por defecto, la del manifiesto del modelo).
    #[arg(long)]
    source_commit: Option<String>,
    /// Salida en JSON.
    #[arg(long)]
    json: bool,
}

/// Procedencia para el encabezado: los flags mandan; si faltan, sale del manifiesto cuyo nombre
/// coincide con la carpeta de salida (o cuyo `hf_dir` coincide con la de entrada).
fn provenance(
    src: &Path,
    dst: &Path,
    repo_flag: Option<String>,
    commit_flag: Option<String>,
) -> Result<(String, String), String> {
    let m = Manifest::all().ok().and_then(|all| {
        let dst_name = dst.file_name().map(|s| s.to_string_lossy().into_owned());
        let src_name = src.file_name().map(|s| s.to_string_lossy().into_owned());
        all.into_iter()
            .find(|m| Some(&m.name) == dst_name.as_ref() || Some(&m.hf_dir) == src_name.as_ref())
    });
    let repo = repo_flag
        .or_else(|| m.as_ref().map(|m| m.source_repo.clone()))
        .ok_or("no sé el repo de origen: pasá --source-repo o usá una carpeta con manifiesto")?;
    let commit = commit_flag
        .or_else(|| m.as_ref().map(|m| m.source_revision.clone()))
        .ok_or(
            "no sé la revisión de origen: pasá --source-commit o usá una carpeta con manifiesto",
        )?;
    Ok((repo, commit))
}

pub fn run(a: ConvertArgs) -> Result<(), String> {
    let (repo, commit) = provenance(&a.src, &a.dst, a.source_repo, a.source_commit)?;
    eprintln!(
        "convirtiendo {} -> {} ({repo} @ {}) …",
        a.src.display(),
        a.dst.display(),
        commit.chars().take(12).collect::<String>()
    );
    let r = convert(&a.src, &a.dst, &repo, &commit).map_err(|e| e.0)?;
    if a.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "tensors": r.tensors,
                "weights_bytes": r.bytes,
                "data_sha256": r.data_sha256,
                "model": a.dst.join("model.brasa").display().to_string(),
            }))
            .unwrap()
        );
    } else {
        println!(
            "listo: {} tensores, {:.2} GiB, data_sha256 {}",
            r.tensors,
            r.bytes as f64 / (1u64 << 30) as f64,
            r.data_sha256
        );
        println!("siguiente: brasa plan {} --ctx 16384", a.dst.display());
    }
    Ok(())
}
