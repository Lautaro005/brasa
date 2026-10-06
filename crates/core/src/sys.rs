//! Lectura de `sysctl` por nombre. Base para fingerprint de hardware y telemetría de memoria.

use std::ffi::CString;
use std::mem::size_of;

/// Lee un sysctl de tamaño fijo (`T` debe ser un tipo plano, sin punteros).
///
/// Devuelve `None` si el nombre no existe o el tamaño no coincide.
pub fn sysctl_value<T: Copy + Default>(name: &str) -> Option<T> {
    let cname = CString::new(name).ok()?;
    let mut value = T::default();
    let mut len = size_of::<T>();
    // SAFETY: `value` es un `T` válido de `len` bytes; sysctlbyname escribe como mucho `len`
    // bytes y actualiza `len` con lo escrito.
    let rc = unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            (&raw mut value).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    (rc == 0 && len == size_of::<T>()).then_some(value)
}

/// Lee un sysctl entero (acepta 4 u 8 bytes).
pub fn sysctl_u64(name: &str) -> Option<u64> {
    let bytes = sysctl_bytes(name)?;
    match bytes.len() {
        4 => Some(u32::from_ne_bytes(bytes.try_into().ok()?) as u64),
        8 => Some(u64::from_ne_bytes(bytes.try_into().ok()?)),
        _ => None,
    }
}

/// Lee un sysctl de texto, sin el terminador nulo.
pub fn sysctl_string(name: &str) -> Option<String> {
    let mut bytes = sysctl_bytes(name)?;
    while bytes.last() == Some(&0) {
        bytes.pop();
    }
    String::from_utf8(bytes).ok()
}

fn sysctl_bytes(name: &str) -> Option<Vec<u8>> {
    let cname = CString::new(name).ok()?;
    let mut len = 0usize;
    // SAFETY: consulta de tamaño con buffer nulo.
    let rc = unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            std::ptr::null_mut(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return None;
    }
    let mut buf = vec![0u8; len];
    // SAFETY: `buf` tiene `len` bytes.
    let rc = unsafe {
        libc::sysctlbyname(
            cname.as_ptr(),
            buf.as_mut_ptr().cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 {
        return None;
    }
    buf.truncate(len);
    Some(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_known_sysctls() {
        // Valor plausible, no el de las Macs del proyecto: el runner de CI de GitHub tiene 7 GB.
        assert!(sysctl_u64("hw.memsize").unwrap() >= 1 << 30);
        assert!(sysctl_string("kern.osproductversion").is_some());
        assert_eq!(sysctl_u64("hw.pagesize"), Some(16384));
        assert!(sysctl_u64("no.existe.esto").is_none());
    }
}
