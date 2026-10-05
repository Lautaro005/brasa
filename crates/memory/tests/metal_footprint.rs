//! Los buffers Metal compartidos cuentan en la huella del proceso (pendiente de T0.3, ADR 0002):
//! si no contaran, el planner y los benchmarks de Brasa subestimarían la memoria.
//! Un único test por binario para que no se mezclen asignaciones de otros tests.

use brasa_memory::process::process_memory;
use brasa_metal::Context;

const MIB: u64 = 1 << 20;
const SIZE: u64 = 256 * MIB;

#[test]
fn buffer_metal_cuenta_en_la_huella() {
    let ctx = Context::new().unwrap();
    // Calentar el contexto (cola, estructuras internas) antes de medir.
    drop(ctx.buffer::<u8>(1).unwrap());
    let before = process_memory().unwrap().footprint;
    // `buffer` llena con ceros: todas las páginas quedan escritas.
    let buf = ctx.buffer::<u8>(SIZE as usize).unwrap();
    let after = process_memory().unwrap().footprint;
    let delta = after.saturating_sub(before);
    eprintln!("delta de huella: {:.2} MiB", delta as f64 / MIB as f64);
    assert!(
        delta + 8 * MIB >= SIZE && delta <= SIZE + 16 * MIB,
        "delta {} MiB para un buffer de {} MiB",
        delta / MIB,
        SIZE / MIB
    );
    drop(buf);
    let released = process_memory().unwrap().footprint;
    assert!(
        released < before + 16 * MIB,
        "no se liberó: {} MiB",
        (released - before) / MIB
    );
}
