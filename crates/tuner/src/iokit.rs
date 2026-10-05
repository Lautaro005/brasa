//! FFI mínima a IOKit/CoreFoundation para leer `gpu-core-count` del acelerador AGX.

use std::ffi::{c_char, c_void};

type CFTypeRef = *const c_void;
type CFStringRef = *const c_void;
type CFAllocatorRef = *const c_void;
type IoObject = u32;

const K_CF_STRING_ENCODING_UTF8: u32 = 0x0800_0100;
const K_CF_NUMBER_SINT64_TYPE: i32 = 4;
const K_IO_REGISTRY_ITERATE_RECURSIVELY: u32 = 1;
const K_IO_MAIN_PORT_DEFAULT: u32 = 0;

#[link(name = "IOKit", kind = "framework")]
unsafe extern "C" {
    fn IOServiceMatching(name: *const c_char) -> *mut c_void;
    fn IOServiceGetMatchingService(main_port: u32, matching: *mut c_void) -> IoObject;
    fn IORegistryEntrySearchCFProperty(
        entry: IoObject,
        plane: *const c_char,
        key: CFStringRef,
        allocator: CFAllocatorRef,
        options: u32,
    ) -> CFTypeRef;
    fn IOObjectRelease(object: IoObject) -> i32;
}

#[link(name = "CoreFoundation", kind = "framework")]
unsafe extern "C" {
    fn CFStringCreateWithCString(
        alloc: CFAllocatorRef,
        c_str: *const c_char,
        encoding: u32,
    ) -> CFStringRef;
    fn CFNumberGetValue(number: CFTypeRef, the_type: i32, value_ptr: *mut c_void) -> bool;
    fn CFGetTypeID(cf: CFTypeRef) -> usize;
    fn CFNumberGetTypeID() -> usize;
    fn CFRelease(cf: CFTypeRef);
}

/// Núcleos de GPU del acelerador AGX (Apple Silicon). `None` si no se encuentra.
pub(crate) fn gpu_core_count() -> Option<u32> {
    // SAFETY: llamadas IOKit/CF estándar; cada objeto creado o copiado se libera una sola vez.
    unsafe {
        // IOServiceGetMatchingService consume la referencia del diccionario.
        let matching = IOServiceMatching(c"AGXAccelerator".as_ptr());
        if matching.is_null() {
            return None;
        }
        let service = IOServiceGetMatchingService(K_IO_MAIN_PORT_DEFAULT, matching);
        if service == 0 {
            return None;
        }
        let key = CFStringCreateWithCString(
            std::ptr::null(),
            c"gpu-core-count".as_ptr(),
            K_CF_STRING_ENCODING_UTF8,
        );
        let value = IORegistryEntrySearchCFProperty(
            service,
            c"IOService".as_ptr(),
            key,
            std::ptr::null(),
            K_IO_REGISTRY_ITERATE_RECURSIVELY,
        );
        CFRelease(key);
        IOObjectRelease(service);
        if value.is_null() {
            return None;
        }
        let mut n: i64 = 0;
        let ok = CFGetTypeID(value) == CFNumberGetTypeID()
            && CFNumberGetValue(value, K_CF_NUMBER_SINT64_TYPE, (&raw mut n).cast());
        CFRelease(value);
        (ok && n > 0).then_some(n as u32)
    }
}
