//! Implementaciones de referencia en CPU, en el mismo orden de operaciones que los kernels.

/// `out[i] = a[i] + b[i]`
pub fn add(a: &[f32], b: &[f32], out: &mut [f32]) {
    for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
        *o = x + y;
    }
}
