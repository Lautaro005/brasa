//! `brasa doctor`: diagnóstico del hardware y del estado de memoria.

use brasa_memory::system::{PressureLevel, SystemMemory, system_memory};
use brasa_tuner::apply::disabled_by_env;
use brasa_tuner::db::default_dir;
use brasa_tuner::fingerprint::{Fingerprint, kernels_version};
use brasa_tuner::hardware::{HardwareInfo, hardware_info};
use brasa_tuner::{TuningDb, TuningStatus, resolve};
use serde::Serialize;

#[derive(Debug, Serialize)]
struct Report {
    brasa_version: &'static str,
    hardware: HardwareInfo,
    memory: SystemMemory,
    /// Fingerprint de la base de tuning (ADR 0026): local, sin identificadores de la unidad.
    tuning: Tuning,
}

#[derive(Debug, Serialize)]
struct Tuning {
    fingerprint_id: String,
    fingerprint: Fingerprint,
    /// Base de tuning de este fingerprint (ADR 0029).
    db: TuningDbInfo,
}

#[derive(Debug, Serialize)]
struct TuningDbInfo {
    /// `tuned`, `missing`, `invalid`, `disabled` o `nodir`.
    status: &'static str,
    /// Ruta del archivo de la base (exista o no).
    path: Option<String>,
    /// Entradas de la base (0 si no hay o es inválida).
    entries: usize,
    message: String,
}

fn db_info(fingerprint: &Fingerprint) -> TuningDbInfo {
    let dir = default_dir();
    let resolved = if disabled_by_env() {
        brasa_tuner::Resolved {
            launch: Default::default(),
            status: TuningStatus::Disabled,
        }
    } else {
        resolve(dir.as_deref(), fingerprint)
    };
    let (status, entries) = match &resolved.status {
        TuningStatus::Tuned { entries, .. } => ("tuned", *entries),
        TuningStatus::Missing { .. } => ("missing", 0),
        TuningStatus::Invalid { .. } => ("invalid", 0),
        TuningStatus::Disabled => ("disabled", 0),
        TuningStatus::NoDir => ("nodir", 0),
    };
    TuningDbInfo {
        status,
        path: dir.map(|d| {
            d.join(TuningDb::file_name(fingerprint))
                .display()
                .to_string()
        }),
        entries,
        message: resolved.status.to_string(),
    }
}

pub fn run(json: bool) {
    let hardware = hardware_info();
    let fingerprint = Fingerprint::from_hardware(&hardware, &kernels_version());
    let report = Report {
        brasa_version: env!("CARGO_PKG_VERSION"),
        hardware,
        memory: system_memory(),
        tuning: Tuning {
            fingerprint_id: fingerprint.id(),
            db: db_info(&fingerprint),
            fingerprint,
        },
    };
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("reporte serializable")
        );
    } else {
        print_text(&report);
    }
}

fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1u64 << 30) as f64)
}

fn print_text(r: &Report) {
    let hw = &r.hardware;
    let mem = &r.memory;
    println!("brasa {}", r.brasa_version);
    println!();
    println!("Hardware");
    println!("  chip            {}", hw.chip);
    println!("  modelo          {}", hw.model);
    println!(
        "  CPU             {} núcleos ({} rendimiento + {} eficiencia)",
        hw.cpu_performance_cores + hw.cpu_efficiency_cores,
        hw.cpu_performance_cores,
        hw.cpu_efficiency_cores
    );
    match hw.gpu_cores {
        Some(n) => println!("  GPU             {n} núcleos"),
        None => println!("  GPU             núcleos desconocidos"),
    }
    println!("  RAM             {}", gib(hw.memory_bytes));
    println!(
        "  macOS           {} ({})",
        hw.macos_version, hw.macos_build
    );
    println!("  arquitectura    {}", hw.arch);
    println!();
    println!("Metal");
    match &hw.metal {
        Some(m) => {
            println!("  dispositivo     {}", m.name);
            println!(
                "  familia         {}{}{}",
                m.apple_family.as_deref().unwrap_or("desconocida"),
                if m.metal3 { ", metal3" } else { "" },
                if m.metal4 { ", metal4" } else { "" }
            );
            println!(
                "  memoria unif.   {}",
                if m.unified_memory { "sí" } else { "no" }
            );
            println!("  working set rec {}", gib(m.recommended_max_working_set));
            println!("  buffer máximo   {}", gib(m.max_buffer_length));
            println!("  threadgroup mem {} KiB", m.max_threadgroup_memory / 1024);
        }
        None => println!("  no se encontró un dispositivo Metal"),
    }
    println!();
    println!("Tuning");
    println!("  fingerprint     {}", r.tuning.fingerprint_id);
    println!("  kernels         {}", r.tuning.fingerprint.kernels_version);
    println!("  base            {}", r.tuning.db.message);
    if let Some(p) = &r.tuning.db.path {
        println!("  archivo         {p}");
    }
    println!();
    println!("Memoria del sistema");
    let pressure = match mem.pressure {
        PressureLevel::Normal => "normal",
        PressureLevel::Warning => "advertencia",
        PressureLevel::Critical => "crítica",
        PressureLevel::Unknown => "desconocida",
    };
    println!("  presión         {pressure}");
    if let Some(p) = mem.available_percent {
        println!("  disponible      {p}% (según el kernel)");
    }
    println!("  libre           {}", gib(mem.free));
    println!("  activa          {}", gib(mem.active));
    println!("  inactiva        {}", gib(mem.inactive));
    println!("  wired           {}", gib(mem.wired));
    println!("  comprimida      {}", gib(mem.compressed));
    println!(
        "  swap            {} usados de {}",
        gib(mem.swap_used),
        gib(mem.swap_total)
    );
    if mem.pressure != PressureLevel::Normal || mem.swap_used > 1 << 30 {
        println!();
        println!(
            "Aviso: el sistema ya está bajo presión de memoria o usando swap. Los benchmarks no \
             son representativos en este estado; cerrá aplicaciones antes de medir."
        );
    }
}
