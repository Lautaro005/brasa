//! Dispositivo Metal, buffers, command queues y cache de pipelines.

pub mod runtime;

pub use runtime::{
    Arg, Buffer, Command, Context, Element, GpuTiming, MetalError, Pending, Pipeline,
};

use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily};
use serde::Serialize;

/// Propiedades del dispositivo Metal por defecto relevantes para planificar memoria y kernels.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceInfo {
    /// Nombre que reporta Metal, por ejemplo "Apple M1 Pro".
    pub name: String,
    /// Familia Apple más alta soportada, por ejemplo "apple7" (M1) o "apple8" (M2).
    pub apple_family: Option<String>,
    /// Soporta la familia Metal 3.
    pub metal3: bool,
    /// Soporta la familia Metal 4.
    pub metal4: bool,
    /// Memoria unificada CPU/GPU.
    pub unified_memory: bool,
    /// Tamaño de working set que Metal recomienda no superar, en bytes.
    pub recommended_max_working_set: u64,
    /// Tamaño máximo de un `MTLBuffer`, en bytes.
    pub max_buffer_length: u64,
    /// Memoria threadgroup máxima por grupo, en bytes.
    pub max_threadgroup_memory: u64,
}

/// Familias Apple de mayor a menor; la primera soportada es la que se reporta.
const APPLE_FAMILIES: [(MTLGPUFamily, &str); 10] = [
    (MTLGPUFamily::Apple10, "apple10"),
    (MTLGPUFamily::Apple9, "apple9"),
    (MTLGPUFamily::Apple8, "apple8"),
    (MTLGPUFamily::Apple7, "apple7"),
    (MTLGPUFamily::Apple6, "apple6"),
    (MTLGPUFamily::Apple5, "apple5"),
    (MTLGPUFamily::Apple4, "apple4"),
    (MTLGPUFamily::Apple3, "apple3"),
    (MTLGPUFamily::Apple2, "apple2"),
    (MTLGPUFamily::Apple1, "apple1"),
];

/// Consulta el dispositivo Metal por defecto. `None` si el sistema no tiene GPU Metal.
pub fn device_info() -> Option<DeviceInfo> {
    let device = MTLCreateSystemDefaultDevice()?;
    let apple_family = APPLE_FAMILIES
        .iter()
        .find(|(family, _)| device.supportsFamily(*family))
        .map(|(_, name)| (*name).to_string());
    Some(DeviceInfo {
        name: device.name().to_string(),
        apple_family,
        metal3: device.supportsFamily(MTLGPUFamily::Metal3),
        metal4: device.supportsFamily(MTLGPUFamily::Metal4),
        unified_memory: device.hasUnifiedMemory(),
        recommended_max_working_set: device.recommendedMaxWorkingSetSize(),
        max_buffer_length: device.maxBufferLength() as u64,
        max_threadgroup_memory: device.maxThreadgroupMemoryLength() as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_device_is_apple_silicon() {
        let info = device_info().expect("se espera una GPU Metal");
        assert!(info.name.starts_with("Apple"), "nombre: {}", info.name);
        assert!(info.unified_memory);
        // Brasa apunta a M1 o posterior: familia Apple7 como mínimo.
        let family = info.apple_family.expect("familia Apple");
        let n: u32 = family.trim_start_matches("apple").parse().unwrap();
        assert!(n >= 7, "familia {family}");
        assert!(info.recommended_max_working_set > 0);
    }
}
