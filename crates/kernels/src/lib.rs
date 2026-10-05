//! Fuentes .metal y registro de variantes por chip.
//!
//! Cada kernel tiene: fuente en `src/metal/`, referencia CPU en [`reference`], test de
//! equivalencia en `tests/` y microbenchmark en `benches/` (regla 2 de CLAUDE.md, ADR 0004).

use brasa_metal::{Arg, Buffer, Command, Context, MetalError, Pipeline};

pub mod reference;
pub mod testutil;

/// Fuentes MSL embebidas en el binario.
pub mod sources {
    pub const ELEMENTWISE: &str = include_str!("metal/elementwise.metal");
}

/// Hilos por threadgroup para kernels elemento a elemento.
const ELEMENTWISE_TG: usize = 256;

/// Pipelines compilados al iniciar; el camino caliente no compila nada.
#[derive(Debug)]
pub struct Kernels {
    add_f32: Pipeline,
}

impl Kernels {
    pub fn new(ctx: &Context) -> Result<Self, MetalError> {
        Ok(Self {
            add_f32: ctx.pipeline(sources::ELEMENTWISE, "add_f32")?,
        })
    }

    /// `out = a + b`. Los tres buffers deben tener el mismo largo.
    pub fn add<'a>(
        &self,
        cmd: &mut Command<'a>,
        a: &'a Buffer<f32>,
        b: &'a Buffer<f32>,
        out: &'a Buffer<f32>,
        n: &'a u32,
    ) {
        let len = *n as usize;
        assert!(a.len() >= len && b.len() >= len && out.len() >= len);
        cmd.dispatch(
            &self.add_f32,
            &[Arg::buf(a), Arg::buf(b), Arg::buf(out), Arg::value(n)],
            [len, 1, 1],
            [ELEMENTWISE_TG, 1, 1],
        );
    }
}
