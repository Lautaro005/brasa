# ADR 0011 — GEMM de prefill (T3.3)

Estado: aceptada en f32 (2,82 TFLOPS); el criterio de 3,5 TFLOPS queda abierto

## Contexto

A 2K el prefill de Brasa da 280 tok/s contra 490 de llama.cpp (docs/bench/baseline.md). Casi todo
el TTFT a 2K es GEMM: 1920 tokens × ~7,2 GFLOP por token a 2,6 TFLOPS son ~5,3 s de 6,85 s. T3.3
pide ≥ 3,5 TFLOPS, lo que llama.cpp logra en este equipo según su TTFT.

Medido en M1 Pro 16 GB, macOS 27.0, base 6cd264a/b93d550, con `cargo bench -p brasa-kernels`
(`gemm_tiled_q4_0` con T=512) y `cargo run --release -p brasa-kernels --example mma_peak`.

## Techo del chip

`mma_peak` encadena `simdgroup_multiply_accumulate` en registros, sin memoria:

| Entrada | 4 acumuladores | 8 | 16 |
|---|---:|---:|---:|
| f32 | 4770 GFLOP/s | 4845 | 4950 |
| f16 (acumulador f32) | 4678 | 4785 | 4971 |

En el M1 Pro el MMA en f16 no es más rápido que en f32. Esto explica también que Q/P en f16 no
acelerara la atención (ADR 0010).

## Decisión

`gemm_tiled` se queda en f32 con estos cambios:

- **Memoria threadgroup por bloques 8×8 contiguos.** Antes cada fragmento cargaba 8 filas con
  stride de 32 floats, que caen en los mismos bancos.
- **Prefetch a registros.** Las lecturas de W y x del paso siguiente de K se emiten antes de
  calcular el actual.
- **Sub-bloque de 32 filas × 16 tokens por simdgroup**, en lugar de 16 × 32.

Medido:

| Variante | q_proj 4096×2560 | gate/up 9728×2560 | down 2560×9728 |
|---|---:|---:|---:|
| antes | 2616 GFLOP/s | 2575 | 2563 |
| ahora | 2814 | 2834 | 2816 |

Lo que se probó y no entra:

| Cambio | q_proj |
|---|---:|
| 32×32 por simdgroup (16 acumuladores), tiles 64×64 o 128×32 | 245–355 |
| ídem con `#pragma unroll` o cargando A de a un fragmento | 277–281 |
| x leído directo de memoria del dispositivo (sin Bs, más ocupación) | 2678 |
| 8 simdgroups (tiles 128×32 o 64×64, 8 acumuladores) | 2430–2555 |
| sin cargar nada (solo productos desde memoria threadgroup) | 3577 |

Notas sobre estas mediciones:

- Los 16 acumuladores no hunden el kernel de `mma_peak`, pero sí el GEMM. No es presión de
  registros visible (`pipeline_limits` da 1024 hilos en ambos). Queda sin explicar.
- El techo sin cargas, 3,58 TFLOPS, es lo máximo que da este bucle aunque la carga se solapara
  por completo. Así, 3,5 TFLOPS en f32 está prácticamente fuera de alcance.

## Lo que hace llama.cpp

Su GEMM (`ggml/src/ggml-metal/kernels/mul_mm.metal` en el commit del baseline, 7fe450e19) tiene
las mismas dimensiones que el nuestro: 64×32 por threadgroup, 4 simdgroups, `mc[8]` y 32 de K. Pero
guarda en memoria threadgroup **los pesos y las activaciones en f16** (`S0 = S1 = half`, también
en `kernel_mul_mm_f32_f32`). Media memoria threadgroup y medias lecturas por fragmento: de ahí sale
su ventaja, no del MMA.

## Consecuencias

- +8–10 % en el GEMM. T3.3 **no se cumple** en f32.
- Igualar a llama.cpp pide redondear las activaciones a f16 antes del GEMM. Es una decisión de
  precisión (regla 3) que queda para el usuario. Si se toma, se verifica como en ADR 0009:
  - referencia que redondea x en el mismo punto;
  - pérdida de top-1 medida contra la referencia sin redondear.

  El riesgo está en las entradas con valores atípicos, sobre todo la salida de SwiGLU que entra
  a `down`. Los pesos q4 decuantizados no son exactos en f16 (d · q necesita hasta 14 bits de
  mantisa). La alternativa exacta es guardar q como entero en f16 y aplicar d aparte, a costa de
  más productos.
