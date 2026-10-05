//! Memoria del proceso actual según el kernel (`proc_pid_rusage`).
//!
//! La métrica principal es `phys_footprint`: es la que usa macOS para presión y jetsam e
//! incluye páginas comprimidas y memoria de GPU asignada por el proceso. `resident` sola
//! subestima cuando el sistema comprime.

use serde::Serialize;

/// Foto de memoria del proceso. Tamaños en bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ProcessMemory {
    /// Huella física (`ri_phys_footprint`).
    pub footprint: u64,
    /// Memoria residente (`ri_resident_size`).
    pub resident: u64,
    /// Pico de huella física en toda la vida del proceso.
    pub lifetime_peak_footprint: u64,
    /// Páginas leídas desde disco (incluye swap-in y mapeos de archivos).
    pub pageins: u64,
}

/// Lee la memoria del proceso actual. `None` solo si el kernel rechaza la consulta.
pub fn process_memory() -> Option<ProcessMemory> {
    let mut info = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    // SAFETY: `info` tiene el tamaño de rusage_info_v4, que es lo que pide RUSAGE_INFO_V4.
    let rc = unsafe {
        libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V4,
            info.as_mut_ptr().cast(),
        )
    };
    if rc != 0 {
        return None;
    }
    // SAFETY: el kernel completó la estructura (rc == 0); además partió de ceros.
    let info = unsafe { info.assume_init() };
    Some(ProcessMemory {
        footprint: info.ri_phys_footprint,
        resident: info.ri_resident_size,
        lifetime_peak_footprint: info.ri_lifetime_max_phys_footprint,
        pageins: info.ri_pageins,
    })
}
