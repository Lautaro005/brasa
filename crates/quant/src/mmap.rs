//! Mapeo de archivos de solo lectura (`mmap`), alineado a página.

use std::fs::File;
use std::os::fd::AsRawFd;
use std::path::Path;

use crate::{Error, Result};

/// Archivo mapeado en memoria, solo lectura. Se desmapea al soltarlo.
#[derive(Debug)]
pub struct Mmap {
    ptr: *mut libc::c_void,
    len: usize,
}

// SAFETY: el mapeo es de solo lectura y no se modifica después de crearse.
unsafe impl Send for Mmap {}
// SAFETY: ídem; acceso concurrente de solo lectura.
unsafe impl Sync for Mmap {}

impl Mmap {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let len = file
            .metadata()
            .map_err(|e| Error(format!("{}: {e}", path.display())))?
            .len() as usize;
        if len == 0 {
            return Err(Error(format!("{}: archivo vacío", path.display())));
        }
        // SAFETY: mapeo privado de solo lectura de un descriptor válido, de largo `len`.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(Error(format!(
                "{}: mmap falló: {}",
                path.display(),
                std::io::Error::last_os_error()
            )));
        }
        Ok(Self { ptr, len })
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: región mapeada de `len` bytes, válida mientras viva `self`.
        unsafe { std::slice::from_raw_parts(self.ptr.cast(), self.len) }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        // SAFETY: `ptr`/`len` vienen de un mmap exitoso.
        unsafe { libc::munmap(self.ptr, self.len) };
    }
}
