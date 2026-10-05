# ADR 0004 — Runtime Metal mínimo

Estado: aceptada (T1.1)

## Decisión

- **`Context`** (en `brasa-metal`): dispositivo por defecto, una command queue y caches de
  bibliotecas y pipelines. Los shaders se compilan **desde fuente MSL en tiempo de ejecución**
  (`newLibraryWithSource`), sin `.metallib` precompilado: no hace falta Xcode para compilar
  Brasa y el costo (decenas de ms) se paga una vez por proceso al cargar el modelo. Precompilar
  queda como optimización de arranque para más adelante.
- **Math mode `Safe` por defecto** (sin fast-math): la equivalencia contra la referencia CPU es
  el criterio de entrada de cada kernel. Un kernel puede pedir fast-math solo si su test de
  equivalencia sigue pasando con la tolerancia documentada.
- **Funciones trascendentes con `precise::`** (`precise::exp`, `precise::sqrt`, ...). Medido en
  M1 Pro durante T1.4: aun con math mode `Safe`, `exp` da 1,07e-6 de error relativo (~18 ULP) en
  SwiGLU; `precise::exp` da 2,4e-7 (~4 ULP).
- **Buffers `Shared`** (memoria unificada, accesibles desde CPU sin copias), tipados por
  elemento (`Buffer<T>`), con seguimiento de hazards de Metal activado. Se revisa en la fase 3.
- **`Command<'a>`**: un command buffer con un único compute encoder serial. Toma prestados los
  buffers que usa durante la vida del comando, así el borrow checker impide escribirlos desde CPU
  mientras la GPU puede estar usándolos. `commit_and_wait` devuelve el tiempo de GPU
  (`GPUEndTime - GPUStartTime`), que usan los microbenchmarks.
- **Kernels** en `crates/kernels/src/metal/*.metal`, cada uno con su referencia CPU en
  `brasa_kernels::reference`, test de equivalencia en `crates/kernels/tests/` y microbenchmark en
  `crates/kernels/benches/` (regla 2 de CLAUDE.md).

## Consecuencias

- Las asignaciones (`Context::buffer`) ocurren al cargar; el camino de decode solo encola
  dispatches sobre buffers existentes.
- Pipelines y bibliotecas se cachean por (hash de la fuente, nombre de función).
