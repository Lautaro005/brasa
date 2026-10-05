//! `brasa convert <dir-hf> <dir-salida>`: conversión nativa a `.brasa` sin Python (U4).

use std::path::PathBuf;

use brasa_quant::convert::convert;
use clap::Args;

#[derive(Debug, Args)]
pub struct ConvertArgs {
    /// Carpeta con los safetensors de Hugging Face (config.json, index.json y shards).
    src: PathBuf,
    /// Carpeta de salida (`model.brasa`, `tokenizer.json`, `tokenizer_config.json`).
    dst: PathBuf,
    /// Repo de origen (va al encabezado del `.brasa`).
    #[arg(long, default_value = "Qwen/Qwen3-4B")]
    source_repo: String,
    /// Revisión de origen (va al encabezado del `.brasa`).
    #[arg(long, default_value = "1cfa9a7208912126459214e8b04321603b3df60c")]
    source_commit: String,
    /// Salida en JSON.
    #[arg(long)]
    json: bool,
}

pub fn run(a: ConvertArgs) -> Result<(), String> {
    eprintln!(
        "convirtiendo {} -> {} (sin Python) …",
        a.src.display(),
        a.dst.display()
    );
    let r = convert(&a.src, &a.dst, &a.source_repo, &a.source_commit).map_err(|e| e.0)?;
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
