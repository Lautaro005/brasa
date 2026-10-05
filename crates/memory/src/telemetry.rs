//! Muestreo de memoria durante una ejecución: huella del proceso y presión del sistema.
//!
//! Corre en un hilo aparte; no toca el loop de decode.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::process::{ProcessMemory, process_memory};
use crate::system::{PressureLevel, SystemMemory, system_memory};

/// Resumen de memoria de una ejecución. Tamaños en bytes.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MemoryReport {
    pub duration_ms: u64,
    pub samples: usize,
    pub start_footprint: u64,
    pub end_footprint: u64,
    /// Máximo muestreado de la huella del proceso durante la ejecución.
    pub peak_footprint: u64,
    pub start_swap_used: u64,
    pub peak_swap_used: u64,
    /// Peor nivel de presión observado.
    pub worst_pressure: PressureLevel,
    pub min_available_percent: Option<u8>,
    /// Páginas leídas desde disco por el proceso durante la ejecución.
    pub pageins: u64,
}

impl MemoryReport {
    /// Crecimiento del swap del sistema durante la ejecución. Un valor que crece entre
    /// ejecuciones indica que el perfil no entra en memoria.
    pub fn swap_growth(&self) -> u64 {
        self.peak_swap_used.saturating_sub(self.start_swap_used)
    }
}

/// Muestreador en segundo plano. `start` arranca, `finish` detiene y devuelve el reporte.
#[derive(Debug)]
pub struct MemorySampler {
    stop: Arc<AtomicBool>,
    handle: JoinHandle<Acc>,
    started: Instant,
    start_proc: ProcessMemory,
    start_sys: SystemMemory,
}

#[derive(Debug)]
struct Acc {
    samples: usize,
    peak_footprint: u64,
    peak_swap_used: u64,
    worst_pressure: PressureLevel,
    min_available_percent: Option<u8>,
}

fn severity(p: PressureLevel) -> u8 {
    match p {
        PressureLevel::Unknown => 0,
        PressureLevel::Normal => 1,
        PressureLevel::Warning => 2,
        PressureLevel::Critical => 3,
    }
}

impl Acc {
    fn add(&mut self, proc_mem: Option<ProcessMemory>, sys: &SystemMemory) {
        self.samples += 1;
        if let Some(p) = proc_mem {
            self.peak_footprint = self.peak_footprint.max(p.footprint);
        }
        self.peak_swap_used = self.peak_swap_used.max(sys.swap_used);
        if severity(sys.pressure) > severity(self.worst_pressure) {
            self.worst_pressure = sys.pressure;
        }
        self.min_available_percent = match (self.min_available_percent, sys.available_percent) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
    }
}

impl MemorySampler {
    /// Arranca el muestreo con el intervalo dado (10–100 ms es razonable).
    pub fn start(interval: Duration) -> Self {
        let start_proc = process_memory().expect("proc_pid_rusage");
        let start_sys = system_memory();
        let mut acc = Acc {
            samples: 0,
            peak_footprint: 0,
            peak_swap_used: 0,
            worst_pressure: PressureLevel::Unknown,
            min_available_percent: None,
        };
        acc.add(Some(start_proc), &start_sys);
        let stop = Arc::new(AtomicBool::new(false));
        let stop_thread = Arc::clone(&stop);
        let handle = std::thread::Builder::new()
            .name("brasa-mem-sampler".into())
            .spawn(move || {
                while !stop_thread.load(Ordering::Relaxed) {
                    std::thread::sleep(interval);
                    acc.add(process_memory(), &system_memory());
                }
                acc
            })
            .expect("hilo de muestreo");
        Self {
            stop,
            handle,
            started: Instant::now(),
            start_proc,
            start_sys,
        }
    }

    /// Detiene el muestreo, toma una última muestra y devuelve el reporte.
    pub fn finish(self) -> MemoryReport {
        self.stop.store(true, Ordering::Relaxed);
        let mut acc = self.handle.join().expect("hilo de muestreo");
        let end_proc = process_memory().expect("proc_pid_rusage");
        acc.add(Some(end_proc), &system_memory());
        MemoryReport {
            duration_ms: self.started.elapsed().as_millis() as u64,
            samples: acc.samples,
            start_footprint: self.start_proc.footprint,
            end_footprint: end_proc.footprint,
            peak_footprint: acc.peak_footprint,
            start_swap_used: self.start_sys.swap_used,
            peak_swap_used: acc.peak_swap_used,
            worst_pressure: acc.worst_pressure,
            min_available_percent: acc.min_available_percent,
            pageins: end_proc.pageins.saturating_sub(self.start_proc.pageins),
        }
    }
}
