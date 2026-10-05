# ADR 0001 — Bindings de Metal y consultas al sistema

Estado: aceptada (T0.2)

## Contexto

Brasa necesita hablar con Metal (dispositivo, buffers, pipelines, command queues) y consultar el
sistema (sysctl, IOKit, Mach) desde Rust. Opciones evaluadas:

- `metal` (metal-rs, gfx-rs): mantenimiento detenido; sus propios autores recomiendan migrar a objc2.
- `objc2-metal` 0.3: generado desde los headers del SDK, cubre Metal 3/4, tipado, mantenido.
- FFI propio a Objective-C: control total, mucho código repetitivo y sin seguridad de tipos.

## Decisión

- Metal: `objc2` + `objc2-metal` + `objc2-foundation`, encapsulados en `crates/metal`. Ningún otro
  crate importa objc2 directamente; el resto ve tipos propios de `brasa-metal`.
- sysctl y estadísticas de VM: `libc` 0.2 (sysctlbyname, host_statistics64, xsw_usage).
- IOKit (núcleos de GPU): FFI mínima propia (5 funciones) en `crates/tuner`; no justifica una
  dependencia más.
- CLI: `clap` (derive). JSON: `serde` + `serde_json`.

## Consecuencias

- Las llamadas a objc2 son `unsafe` en varios puntos; se concentran en `brasa-metal` con comentarios
  `SAFETY`.
- Si objc2-metal cambia de API entre versiones menores (0.x), el impacto queda acotado a un crate.
- Nada de esto agrega dependencias de runtime fuera de los frameworks del sistema.
