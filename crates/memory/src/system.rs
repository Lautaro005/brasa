//! Estado de memoria del sistema: uso de páginas, swap y nivel de presión del kernel.

use brasa_core::sys::{sysctl_u64, sysctl_value};
use serde::Serialize;

/// Nivel de presión de memoria según `kern.memorystatus_vm_pressure_level`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PressureLevel {
    Normal,
    Warning,
    Critical,
    Unknown,
}

impl PressureLevel {
    fn from_kernel(level: u64) -> Self {
        // Valores de <sys/kern_memorystatus.h>: 1 normal, 2 warn, 4 critical.
        match level {
            1 => Self::Normal,
            2 => Self::Warning,
            4 => Self::Critical,
            _ => Self::Unknown,
        }
    }
}

/// Foto de la memoria del sistema. Todos los tamaños en bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SystemMemory {
    pub total: u64,
    /// Libre de verdad (sin las páginas especulativas, que se cuentan aparte).
    pub free: u64,
    pub active: u64,
    pub inactive: u64,
    pub speculative: u64,
    pub wired: u64,
    /// Memoria ocupada por el compresor (páginas físicas que usa, no lo que representa).
    pub compressed: u64,
    pub swap_total: u64,
    pub swap_used: u64,
    pub pressure: PressureLevel,
    /// Porcentaje de memoria disponible según el kernel (`kern.memorystatus_level`).
    pub available_percent: Option<u8>,
}

/// Lee el estado actual de la memoria del sistema.
pub fn system_memory() -> SystemMemory {
    let page = sysctl_u64("hw.pagesize").unwrap_or(16384);
    let vm = vm_statistics().unwrap_or_default();
    let swap = sysctl_value::<libc::xsw_usage>("vm.swapusage");
    SystemMemory {
        total: sysctl_u64("hw.memsize").unwrap_or(0),
        // En XNU `free_count` incluye las páginas especulativas; se reportan por separado.
        free: vm.free_count.saturating_sub(vm.speculative_count) as u64 * page,
        active: vm.active_count as u64 * page,
        inactive: vm.inactive_count as u64 * page,
        speculative: vm.speculative_count as u64 * page,
        wired: vm.wire_count as u64 * page,
        compressed: vm.compressor_page_count as u64 * page,
        swap_total: swap.map_or(0, |s| s.xsu_total),
        swap_used: swap.map_or(0, |s| s.xsu_used),
        pressure: sysctl_u64("kern.memorystatus_vm_pressure_level")
            .map_or(PressureLevel::Unknown, PressureLevel::from_kernel),
        available_percent: sysctl_u64("kern.memorystatus_level").map(|v| v.min(100) as u8),
    }
}

unsafe extern "C" {
    // libc la marca deprecada en favor de `mach2`; es una función estable de libSystem.
    fn mach_host_self() -> libc::mach_port_t;
}

fn vm_statistics() -> Option<libc::vm_statistics64> {
    let mut stats = libc::vm_statistics64::default();
    let mut count = libc::HOST_VM_INFO64_COUNT;
    // SAFETY: `stats` tiene exactamente HOST_VM_INFO64_COUNT enteros; el kernel escribe como
    // mucho `count` enteros.
    let rc = unsafe {
        libc::host_statistics64(
            mach_host_self(),
            libc::HOST_VM_INFO64,
            (&raw mut stats).cast(),
            &mut count,
        )
    };
    (rc == libc::KERN_SUCCESS).then_some(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_is_consistent() {
        let m = system_memory();
        assert!(m.total >= 8 << 30);
        let counted = m.free + m.active + m.inactive + m.speculative + m.wired + m.compressed;
        assert!(counted > 0 && counted <= m.total, "{m:?}");
        assert!(m.swap_used <= m.swap_total);
        assert_ne!(m.pressure, PressureLevel::Unknown);
    }
}
