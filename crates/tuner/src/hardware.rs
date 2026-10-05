//! Detección del hardware: chip, núcleos, RAM, macOS y propiedades Metal.
//!
//! No se leen número de serie, UUID de plataforma ni ningún identificador personal: solo datos
//! que comparten todas las unidades del mismo modelo.

use brasa_core::sys::{sysctl_string, sysctl_u64};
use brasa_metal::DeviceInfo;
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct HardwareInfo {
    /// Nombre comercial del chip, por ejemplo "Apple M1 Pro".
    pub chip: String,
    /// Identificador de modelo de Mac, por ejemplo "MacBookPro18,1" (no identifica la unidad).
    pub model: String,
    pub cpu_performance_cores: u32,
    pub cpu_efficiency_cores: u32,
    /// Núcleos de GPU según IOKit (`gpu-core-count`). `None` si no se pudo leer.
    pub gpu_cores: Option<u32>,
    /// RAM física en bytes.
    pub memory_bytes: u64,
    pub macos_version: String,
    pub macos_build: String,
    pub arch: String,
    /// `None` si no hay GPU Metal.
    pub metal: Option<DeviceInfo>,
}

/// Detecta el hardware de la máquina actual.
pub fn hardware_info() -> HardwareInfo {
    // En Apple Silicon perflevel0 es el cluster de rendimiento y perflevel1 el de eficiencia.
    let perf = sysctl_u64("hw.perflevel0.physicalcpu");
    let eff = sysctl_u64("hw.perflevel1.physicalcpu");
    let (perf, eff) = match (perf, eff) {
        (Some(p), e) => (p, e.unwrap_or(0)),
        (None, _) => (sysctl_u64("hw.physicalcpu").unwrap_or(0), 0),
    };
    HardwareInfo {
        chip: sysctl_string("machdep.cpu.brand_string").unwrap_or_default(),
        model: sysctl_string("hw.model").unwrap_or_default(),
        cpu_performance_cores: perf as u32,
        cpu_efficiency_cores: eff as u32,
        gpu_cores: crate::iokit::gpu_core_count(),
        memory_bytes: sysctl_u64("hw.memsize").unwrap_or(0),
        macos_version: sysctl_string("kern.osproductversion").unwrap_or_default(),
        macos_build: sysctl_string("kern.osversion").unwrap_or_default(),
        arch: std::env::consts::ARCH.to_string(),
        metal: brasa_metal::device_info(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_apple_silicon() {
        let hw = hardware_info();
        assert!(hw.chip.starts_with("Apple M"), "{hw:?}");
        assert_eq!(hw.arch, "aarch64");
        assert!(hw.cpu_performance_cores > 0);
        assert!(hw.gpu_cores.is_some_and(|n| n >= 7), "{hw:?}");
        assert!(hw.memory_bytes >= 8 << 30);
        assert!(!hw.macos_version.is_empty());
        let metal = hw.metal.expect("metal");
        assert_eq!(metal.name, hw.chip);
    }
}
