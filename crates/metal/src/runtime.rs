//! Runtime Metal mínimo: contexto, buffers compartidos, compilación de MSL, cache de pipelines y
//! comandos (ADR 0004).

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
};

/// Error del runtime Metal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalError(pub String);

impl fmt::Display for MetalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "metal: {}", self.0)
    }
}

impl std::error::Error for MetalError {}

pub type Result<T> = std::result::Result<T, MetalError>;

type Device = ProtocolObject<dyn MTLDevice>;
type Library = ProtocolObject<dyn MTLLibrary>;
type PipelineState = ProtocolObject<dyn MTLComputePipelineState>;
type MtlBuffer = ProtocolObject<dyn MTLBuffer>;

/// Tipos de elemento que pueden vivir en un `Buffer`.
///
/// # Safety
/// Debe ser un tipo plano (`Copy`, sin punteros ni padding) válido para cualquier patrón de bits,
/// porque la GPU escribe bytes arbitrarios.
pub unsafe trait Element: Copy + Default + 'static {}

// SAFETY: tipos numéricos primitivos, válidos para cualquier patrón de bits.
unsafe impl Element for f32 {}
unsafe impl Element for u8 {}
unsafe impl Element for i8 {}
unsafe impl Element for u16 {}
unsafe impl Element for i16 {}
unsafe impl Element for u32 {}
unsafe impl Element for i32 {}

/// Buffer de memoria unificada (`MTLStorageModeShared`) con `len` elementos de tipo `T`.
pub struct Buffer<T: Element> {
    raw: Retained<MtlBuffer>,
    len: usize,
    _t: PhantomData<T>,
}

impl<T: Element> fmt::Debug for Buffer<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Buffer")
            .field("type", &std::any::type_name::<T>())
            .field("len", &self.len)
            .finish()
    }
}

impl<T: Element> Buffer<T> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn byte_len(&self) -> usize {
        self.len * size_of::<T>()
    }

    /// Contenido visto desde CPU. Seguro porque un `Command` que usa el buffer lo mantiene
    /// prestado hasta terminar (`commit_and_wait`).
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: memoria Shared de al menos len * size_of::<T>() bytes, alineada a página; T es
        // Element (cualquier patrón de bits es válido).
        unsafe { std::slice::from_raw_parts(self.raw.contents().as_ptr().cast(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: igual que `as_slice`; `&mut self` garantiza que no hay otros préstamos, incluido
        // un `Command` en curso.
        unsafe { std::slice::from_raw_parts_mut(self.raw.contents().as_ptr().cast(), self.len) }
    }

    fn raw(&self) -> &MtlBuffer {
        &self.raw
    }
}

/// Pipeline de cómputo compilado.
#[derive(Clone)]
pub struct Pipeline {
    state: Retained<PipelineState>,
    name: String,
}

impl fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pipeline")
            .field("name", &self.name)
            .field("max_threads", &self.max_threads_per_threadgroup())
            .finish()
    }
}

impl Pipeline {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn max_threads_per_threadgroup(&self) -> usize {
        self.state.maxTotalThreadsPerThreadgroup()
    }

    pub fn thread_execution_width(&self) -> usize {
        self.state.threadExecutionWidth()
    }
}

/// Dispositivo, cola y caches. Uno por proceso; no es `Sync` (los caches usan `RefCell`).
pub struct Context {
    device: Retained<Device>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    libraries: RefCell<HashMap<u64, Retained<Library>>>,
    pipelines: RefCell<HashMap<(u64, String), Pipeline>>,
}

impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Context")
            .field("device", &self.device.name().to_string())
            .field("pipelines", &self.pipelines.borrow().len())
            .finish()
    }
}

fn source_hash(source: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    source.hash(&mut h);
    h.finish()
}

impl Context {
    /// Crea el contexto sobre el dispositivo Metal por defecto.
    pub fn new() -> Result<Self> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| MetalError("no hay dispositivo Metal".into()))?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| MetalError("no se pudo crear la command queue".into()))?;
        Ok(Self {
            device,
            queue,
            libraries: RefCell::default(),
            pipelines: RefCell::default(),
        })
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// Buffer compartido de `len` elementos inicializado en cero.
    pub fn buffer<T: Element>(&self, len: usize) -> Result<Buffer<T>> {
        // Metal no acepta buffers de 0 bytes.
        let bytes = (len * size_of::<T>()).max(1);
        let raw = self
            .device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| MetalError(format!("no se pudo asignar un buffer de {bytes} bytes")))?;
        let mut buf = Buffer {
            raw,
            len,
            _t: PhantomData,
        };
        buf.as_mut_slice().fill(T::default());
        Ok(buf)
    }

    /// Buffer compartido con una copia de `data`.
    pub fn buffer_from<T: Element>(&self, data: &[T]) -> Result<Buffer<T>> {
        let mut buf = self.buffer(data.len())?;
        buf.as_mut_slice().copy_from_slice(data);
        Ok(buf)
    }

    /// Compila (o toma del cache) la función `function` de la fuente MSL `source`.
    pub fn pipeline(&self, source: &str, function: &str) -> Result<Pipeline> {
        let key = (source_hash(source), function.to_string());
        if let Some(p) = self.pipelines.borrow().get(&key) {
            return Ok(p.clone());
        }
        let library = self.library(source, key.0)?;
        let func = library
            .newFunctionWithName(&NSString::from_str(function))
            .ok_or_else(|| MetalError(format!("la función {function} no existe en la fuente")))?;
        let state = self
            .device
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| MetalError(format!("pipeline {function}: {e}")))?;
        let pipeline = Pipeline {
            state,
            name: function.to_string(),
        };
        self.pipelines.borrow_mut().insert(key, pipeline.clone());
        Ok(pipeline)
    }

    fn library(&self, source: &str, hash: u64) -> Result<Retained<Library>> {
        if let Some(l) = self.libraries.borrow().get(&hash) {
            return Ok(l.clone());
        }
        let options = MTLCompileOptions::new();
        options.setMathMode(MTLMathMode::Safe);
        let library = self
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
            .map_err(|e| MetalError(format!("error de compilación MSL: {e}")))?;
        self.libraries.borrow_mut().insert(hash, library.clone());
        Ok(library)
    }

    /// Nuevo comando con un compute encoder abierto.
    pub fn command<'a>(&self) -> Result<Command<'a>> {
        let cb = self
            .queue
            .commandBuffer()
            .ok_or_else(|| MetalError("no se pudo crear el command buffer".into()))?;
        let encoder = cb
            .computeCommandEncoder()
            .ok_or_else(|| MetalError("no se pudo crear el compute encoder".into()))?;
        Ok(Command {
            cb,
            encoder,
            _borrows: PhantomData,
        })
    }
}

/// Argumento de un dispatch, en el índice dado por su posición.
#[derive(Debug, Clone, Copy)]
pub enum Arg<'a> {
    /// Buffer desde un offset en bytes.
    Buf(&'a MtlBuffer, usize),
    /// Constante pequeña copiada en el argumento (`setBytes`), sin asignar memoria.
    Inline([u8; INLINE_MAX], usize),
}

/// Tamaño máximo de una constante `Inline`.
pub const INLINE_MAX: usize = 32;

impl<'a> Arg<'a> {
    pub fn buf<T: Element>(b: &'a Buffer<T>) -> Self {
        Arg::Buf(b.raw(), 0)
    }

    /// Buffer a partir del elemento `elem` (vista sin copia, por ejemplo una capa del KV).
    pub fn buf_at<T: Element>(b: &'a Buffer<T>, elem: usize) -> Self {
        assert!(elem <= b.len(), "offset fuera del buffer");
        Arg::Buf(b.raw(), elem * size_of::<T>())
    }

    /// Valor plano copiado en el argumento (por ejemplo, un `u32` con un tamaño).
    pub fn value<T: Element>(v: T) -> Self {
        let n = size_of::<T>();
        assert!(n <= INLINE_MAX, "constante demasiado grande para Inline");
        let mut bytes = [0u8; INLINE_MAX];
        // SAFETY: T es un tipo plano de `n` bytes; se copia a un arreglo de al menos `n` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping((&raw const v).cast::<u8>(), bytes.as_mut_ptr(), n)
        };
        Arg::Inline(bytes, n)
    }

    pub fn u32(v: u32) -> Self {
        Self::value(v)
    }

    pub fn f32(v: f32) -> Self {
        Self::value(v)
    }
}

/// Tiempos de un comando ejecutado.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GpuTiming {
    /// Tiempo de GPU entre el inicio y el fin del command buffer, en segundos.
    pub gpu_seconds: f64,
}

/// Command buffer con un compute encoder serial. Los buffers pasados a `dispatch` quedan
/// prestados durante `'a`, hasta `commit_and_wait`.
pub struct Command<'a> {
    cb: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    _borrows: PhantomData<&'a ()>,
}

impl fmt::Debug for Command<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Command").finish_non_exhaustive()
    }
}

fn size(v: [usize; 3]) -> MTLSize {
    MTLSize {
        width: v[0],
        height: v[1],
        depth: v[2],
    }
}

impl<'a> Command<'a> {
    /// Encola `pipeline` sobre una grilla de `grid` hilos en grupos de `threadgroup`.
    /// La grilla no necesita ser múltiplo del grupo (`dispatchThreads`).
    pub fn dispatch(
        &mut self,
        pipeline: &Pipeline,
        args: &[Arg<'a>],
        grid: [usize; 3],
        threadgroup: [usize; 3],
    ) {
        self.encoder.setComputePipelineState(&pipeline.state);
        self.bind(args);
        self.encoder
            .dispatchThreads_threadsPerThreadgroup(size(grid), size(threadgroup));
    }

    /// Encola `pipeline` con `groups` threadgroups de `threadgroup` hilos cada uno.
    pub fn dispatch_groups(
        &mut self,
        pipeline: &Pipeline,
        args: &[Arg<'a>],
        groups: [usize; 3],
        threadgroup: [usize; 3],
    ) {
        self.encoder.setComputePipelineState(&pipeline.state);
        self.bind(args);
        self.encoder
            .dispatchThreadgroups_threadsPerThreadgroup(size(groups), size(threadgroup));
    }

    fn bind(&mut self, args: &[Arg<'a>]) {
        for (i, arg) in args.iter().enumerate() {
            match arg {
                // SAFETY: buffer vivo durante 'a, offset dentro del buffer, índice dentro de los
                // 31 slots de Metal.
                Arg::Buf(b, off) => unsafe {
                    self.encoder.setBuffer_offset_atIndex(Some(b), *off, i)
                },
                Arg::Inline(bytes, n) => {
                    // SAFETY: Metal copia los `n` bytes durante la llamada.
                    unsafe {
                        self.encoder.setBytes_length_atIndex(
                            NonNull::new(bytes.as_ptr().cast_mut().cast()).unwrap(),
                            *n,
                            i,
                        )
                    }
                }
            }
        }
    }

    /// Cierra el encoder, envía el comando y espera a que la GPU termine.
    pub fn commit_and_wait(self) -> Result<GpuTiming> {
        self.commit().wait()
    }

    /// Cierra el encoder y envía el comando sin esperar. Los comandos de una misma cola corren
    /// en orden, así que la CPU puede ir codificando el siguiente mientras la GPU ejecuta este.
    /// Los buffers siguen prestados hasta [`Pending::wait`].
    pub fn commit(self) -> Pending<'a> {
        self.encoder.endEncoding();
        self.cb.commit();
        Pending {
            cb: self.cb,
            _borrows: PhantomData,
        }
    }
}

/// Comando enviado a la GPU, todavía sin esperar ([`Command::commit`]).
#[must_use = "hay que esperar el comando (wait) antes de leer sus resultados"]
pub struct Pending<'a> {
    cb: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    _borrows: PhantomData<&'a ()>,
}

impl fmt::Debug for Pending<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pending").finish_non_exhaustive()
    }
}

impl Pending<'_> {
    /// Espera a que la GPU termine este comando.
    pub fn wait(self) -> Result<GpuTiming> {
        self.cb.waitUntilCompleted();
        if self.cb.status() != MTLCommandBufferStatus::Completed {
            let detail = self
                .cb
                .error()
                .map(|e| e.to_string())
                .unwrap_or_else(|| format!("estado {:?}", self.cb.status()));
            return Err(MetalError(format!("el comando falló: {detail}")));
        }
        Ok(GpuTiming {
            gpu_seconds: self.cb.GPUEndTime() - self.cb.GPUStartTime(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: &str = "
#include <metal_stdlib>
using namespace metal;
kernel void fill(device float* out [[buffer(0)]], constant float& v [[buffer(1)]],
                 uint i [[thread_position_in_grid]]) { out[i] = v + float(i); }
";

    #[test]
    fn compile_cache_dispatch() {
        let ctx = Context::new().unwrap();
        let p1 = ctx.pipeline(SRC, "fill").unwrap();
        let p2 = ctx.pipeline(SRC, "fill").unwrap();
        assert!(
            std::ptr::eq(&*p1.state, &*p2.state),
            "el pipeline no se cacheó"
        );
        let mut out = ctx.buffer::<f32>(1000).unwrap();
        let v = 0.5f32;
        let mut cmd = ctx.command().unwrap();
        cmd.dispatch(
            &p1,
            &[Arg::buf(&out), Arg::f32(v)],
            [1000, 1, 1],
            [256, 1, 1],
        );
        let t = cmd.commit_and_wait().unwrap();
        assert!(t.gpu_seconds >= 0.0);
        for (i, x) in out.as_mut_slice().iter().enumerate() {
            assert_eq!(*x, 0.5 + i as f32);
        }
    }

    #[test]
    fn compile_error_is_reported() {
        let ctx = Context::new().unwrap();
        let e = ctx.pipeline("kernel void x( {", "x").unwrap_err();
        assert!(e.0.contains("compilación"), "{e}");
        let e = ctx.pipeline(SRC, "no_existe").unwrap_err();
        assert!(e.0.contains("no_existe"), "{e}");
    }
}
