//! Criterio de T0.3: un buffer de tamaño conocido se refleja en el reporte dentro de un margen.
//!
//! Un único test en este binario: los tests de un mismo binario corren en hilos del mismo
//! proceso y sus asignaciones contaminarían la medición.

use std::time::Duration;

use brasa_memory::process::process_memory;
use brasa_memory::telemetry::MemorySampler;

const MIB: u64 = 1 << 20;
const SIZE: u64 = 256 * MIB;
/// Margen: el runtime de tests puede mover algunos MiB. 8 MiB por debajo o 16 MiB por encima del
/// tamaño pedido.
const BELOW: u64 = 8 * MIB;
const ABOVE: u64 = 16 * MIB;

/// Buffer anónimo con `mmap`/`munmap` directos: tamaño exacto y sin el cache de regiones
/// grandes de libmalloc, que retiene memoria liberada y la reutiliza.
struct Buffer {
    ptr: *mut u8,
    len: usize,
}

impl Buffer {
    fn new_touched(len: usize) -> Self {
        // SAFETY: mapeo anónimo privado, sin archivo.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert_ne!(ptr, libc::MAP_FAILED);
        let ptr = ptr.cast::<u8>();
        // Escribir una vez por página fuerza a que cada página quede residente.
        for i in (0..len).step_by(4096) {
            // SAFETY: `i < len`, dentro del mapeo.
            unsafe { std::ptr::write_volatile(ptr.add(i), 1) };
        }
        Self { ptr, len }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: `ptr`/`len` vienen de un mmap exitoso que no se liberó antes.
        unsafe { libc::munmap(self.ptr.cast(), self.len) };
    }
}

fn assert_within(label: &str, delta: u64) {
    assert!(
        delta + BELOW >= SIZE && delta <= SIZE + ABOVE,
        "{label}: delta {} MiB, esperado {} MiB (-{} / +{})",
        delta / MIB,
        SIZE / MIB,
        BELOW / MIB,
        ABOVE / MIB
    );
}

#[test]
fn buffer_conocido_se_refleja_en_el_reporte() {
    // 1) Foto directa antes y después de asignar.
    let before = process_memory().unwrap().footprint;
    let buf = Buffer::new_touched(SIZE as usize);
    let after = process_memory().unwrap().footprint;
    assert_within("process_memory", after.saturating_sub(before));
    drop(buf);

    // 2) Muestreador durante una "ejecución" que asigna, sostiene y libera.
    let sampler = MemorySampler::start(Duration::from_millis(10));
    std::thread::sleep(Duration::from_millis(30));
    let buf = Buffer::new_touched(SIZE as usize);
    std::thread::sleep(Duration::from_millis(100));
    drop(buf);
    std::thread::sleep(Duration::from_millis(30));
    let report = sampler.finish();
    eprintln!("{report:#?}");

    assert!(report.samples >= 5, "pocas muestras: {}", report.samples);
    assert_within(
        "pico del muestreador",
        report.peak_footprint.saturating_sub(report.start_footprint),
    );
    // Al liberar, la huella vuelve cerca del inicio.
    assert!(
        report.end_footprint < report.start_footprint + ABOVE,
        "no volvió: inicio {} MiB, fin {} MiB",
        report.start_footprint / MIB,
        report.end_footprint / MIB
    );
}
