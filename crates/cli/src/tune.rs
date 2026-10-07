//! `brasa tune [modelo]`: autotuner de los parámetros de lanzamiento de decode (ADR 0029). Mide
//! en la GPU de esta Mac, con las formas del modelo, y guarda la base de tuning del fingerprint
//! actual (ADR 0026). `run`, `serve` y `benchmark` la usan al cargar.

use std::time::Instant;

use brasa_tuner::db::default_dir;
use brasa_tuner::tune::{Measured, Mode, Options, tune_this_machine};
use clap::Args;

use crate::config::{self, Config, DEFAULT_MODEL, DEFAULT_SERVE_CTX};
use crate::run::resolve_model;

#[derive(Debug, Args)]
pub struct TuneArgs {
    /// Modelo cuyas formas se miden (carpeta en ./models o $BRASA_MODELS, o ruta). Sin esto, el
    /// del archivo de configuración.
    model: Option<String>,
    /// Tuning completo: además KV f32, más longitudes de caché y 15 rondas (quick: 5).
    #[arg(long)]
    full: bool,
    /// Contexto máximo para medir la atención de decode.
    #[arg(long, default_value_t = DEFAULT_SERVE_CTX)]
    ctx: usize,
    /// Mide e imprime sin guardar la base.
    #[arg(long)]
    dry_run: bool,
    /// Salida en JSON.
    #[arg(long)]
    json: bool,
}

fn candidates(m: &Measured) -> String {
    m.candidates
        .iter()
        .map(|(sg, us)| format!("{sg}:{us:.1}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn gain(m: &Measured) -> f64 {
    (1.0 - m.value.gpu_us / m.value.default_us) * 100.0
}

pub fn run(a: TuneArgs) -> Result<(), String> {
    let cfg = Config::load()?;
    let model = config::pick(
        a.model.clone(),
        cfg.model.clone(),
        DEFAULT_MODEL.to_string(),
    );
    let dir = resolve_model(&model.value)?;
    let dims = brasa_runtime::tune_dims(&dir).map_err(|e| e.to_string())?;
    let mode = if a.full { Mode::Full } else { Mode::Quick };
    let opts = Options::new(mode, a.ctx);
    let fp = brasa_tuner::Fingerprint::current();
    if !a.json {
        println!(
            "fingerprint  {} ({}, {} núcleos de GPU, macOS {} {})",
            fp.id(),
            fp.chip,
            fp.gpu_cores.map_or("?".to_string(), |n| n.to_string()),
            fp.macos_version,
            fp.macos_build
        );
        println!(
            "modo         {} ({} rondas; KV {}; cachés {:?})",
            mode.name(),
            mode.samples(),
            mode.kv_types()
                .iter()
                .map(|k| k.name())
                .collect::<Vec<_>>()
                .join(", "),
            mode.lengths(a.ctx)
        );
        println!("modelo       {}", dir.display());
        println!();
        println!(
            "{:<28} {:<22} {:<12} {:<44} elegido",
            "kernel", "forma", "variante", "SG:µs por dispatch (mediana)"
        );
    }
    let t0 = Instant::now();
    let json = a.json;
    let mut noisy = 0;
    let mut report = |m: &Measured| {
        noisy += usize::from(m.noisy());
        if !json {
            println!(
                "{:<28} {:<22} {:<12} {:<44} SG={} ({:+.1} %){}",
                m.key.kernel,
                m.key.shape,
                m.key.variant,
                candidates(m),
                m.value
                    .params
                    .get(brasa_tuner::apply::SG)
                    .copied()
                    .unwrap_or(0),
                gain(m),
                if m.noisy() {
                    format!("  ruido {:.0} %", m.noise * 100.0)
                } else {
                    String::new()
                }
            );
        }
    };
    let db = tune_this_machine(&dims, &opts, &mut report)?;
    let elapsed = t0.elapsed().as_secs_f64();
    if noisy > 0 && !json {
        eprintln!(
            "aviso: {noisy} mediciones con más de {:.0} % de dispersión (otra carga en la GPU); \
             quedaron en el valor por defecto. Cerrá otras aplicaciones y repetí `brasa tune`.",
            brasa_tuner::tune::NOISE_MAX * 100.0
        );
    }
    let saved = if a.dry_run {
        None
    } else {
        let d = default_dir().ok_or("no hay carpeta para la base (falta HOME)")?;
        Some(db.save_in(&d).map_err(|e| e.to_string())?)
    };
    if json {
        let entries: Vec<_> = db
            .iter()
            .map(|(k, v)| {
                serde_json::json!({
                    "kernel": k.kernel, "shape": k.shape, "variant": k.variant,
                    "params": v.params, "gpu_us": v.gpu_us, "default_us": v.default_us,
                    "samples": v.samples,
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "fingerprint_id": fp.id(),
                "mode": mode.name(),
                "elapsed_s": elapsed,
                "path": saved.as_ref().map(|p| p.display().to_string()),
                "entries": entries,
            }))
            .expect("JSON serializable")
        );
        return Ok(());
    }
    println!();
    let changed = db.iter().filter(|(_, v)| v.gpu_us < v.default_us).count();
    println!(
        "{} entradas ({changed} distintas de los valores por defecto); {} tune en {elapsed:.1} s \
         (medido en {})",
        db.len(),
        mode.name(),
        fp.chip
    );
    match saved {
        Some(p) => println!("base guardada en {}", p.display()),
        None => println!("--dry-run: no se guardó la base"),
    }
    Ok(())
}
