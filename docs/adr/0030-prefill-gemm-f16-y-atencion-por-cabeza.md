# ADR 0030 — Prefill: GEMM con entradas f16 y atención por cabeza

Estado: aceptada (2026-10-07)

## Contexto

La compuerta de la fase 3 está bloqueada por el prefill. Medido en M1 Pro 16 GB (commit 8dc9fa3,
`brasa benchmark`):

| Ctx | Prefill Brasa | llama.cpp |
|---:|---:|---:|
| 2048 | 299 tok/s | 490 |
| 8192 | 195 | 356 |
| 16384 | 138 | 260 |

El usuario pidió volver a intentar operandos f16 si hacía falta para alcanzar el objetivo, con
ADR, pérdida medida contra la referencia y tolerancia documentada (ver ADR 0010 y 0011, donde los
intentos anteriores no aceleraron).

## Perfil antes de tocar kernels

Herramientas nuevas:

- `cargo run --release -p brasa-models --example profile_prefill -- <n> <chunk>`: prompt completo
  con el modelo real (tiempo de pared y de GPU). Con `PP_DEPTHS=0,1536,7680,15872` mide un
  bloque en cada posición, como `llama-bench -p 512 -d ...`.
- `cargo run --release -p brasa-kernels --example prefill_breakdown -- [T] [pos0] [kv]`: tiempo
  de GPU por operación (×36 capas).
- `gemm_lab` y `flash_lab` (ejemplos de `brasa-kernels`): compilan un kernel candidato desde un
  `.metal` y lo comparan con el actual (velocidad y error).

Bloque de 512 tokens según la posición, GPU, M1 Pro 16 GB (macOS 27.0), KV f16:

| Posición | Brasa antes | llama.cpp (`llama-bench`, build 7fe450e19) |
|---:|---:|---:|
| 0 | 1442 ms (355 tok/s) | 960 ms (533 tok/s) |
| 1536 | 1833 ms (279) | 1160 ms (441) |
| 7680 | 3393 ms (151) | 1963 ms (261) |
| 15872 | 5511 ms (93) | 3106 ms (165) |

En la posición 0 casi todo es GEMM (~2,6 TFLOPS efectivos contra ~3,9 de llama.cpp). A 16K, la
atención de cada bloque cuesta ~113 ms por capa contra ~60 de llama.cpp. Las dos partes había
que acelerarlas ~1,5–1,9×. El resto (RMSNorm, QK-norm + RoPE, sumas, SwiGLU) es < 3 %.

## Decisión

### GEMM (`matmul_tiled.metal`)

Mismas dimensiones de tile que antes (64 filas × 32 tokens, 4 simdgroups de 32 × 16, BK = 32),
con la estructura de `kernel_mul_mm` de llama.cpp (solo la idea; el kernel es propio):

- se calcula Cᵀ = X·Wᵀ, así cada fragmento del acumulador se escribe directo en `y`;
- índices `ushort`, todos los fragmentos de un paso de K cargados antes de los productos;
- sin prefetch a registros del paso siguiente;
- q4_0 se decuantiza con máscaras sobre pares de bytes: `d1·(q & m) + md` = `d·(q − 8)` exacto en
  f32 (sin desplazamientos);
- pesos y activaciones en memoria threadgroup como **half** (`GemmInput::F16`, la ruta caliente);
  acumulación en f32. La variante `GemmInput::F32` (mismo kernel con float) queda para verificar.

`gemm_lab`, T = 512, M1 Pro (GFLOP/s):

| Variante | q_proj | gate/up | down |
|---|---:|---:|---:|
| antes (f32, ADR 0011) | 2813 | 2835 | 2815 |
| estructura nueva, entradas f32 (exacto) | 3369 | 3387 | 3353 |
| ídem con prefetch a registros, f16 | 3232 | 3251 | 3211 |
| estructura nueva, entradas f16 | **3828** | **3852** | **3809** |
| ídem con decuantización por desplazamientos | 3528 | 3550 | 3514 |
| 64 × 64 por threadgroup (16 acumuladores por simdgroup), f16 | 478 | 399 | 425 |

- El orden de la grilla (filas o tokens en x) no cambia nada (±0,5 %).
- Lo que rinde es bajar registros por hilo (`max_total_threads_per_threadgroup`: 704 → 832):
  sacar el prefetch y los desplazamientos.
- 16 acumuladores por simdgroup siguen hundiendo el kernel (~10×), como en ADR 0011, también
  con fragmentos half.

+36 % con entradas f16, por encima del umbral de 15 % que fijó ADR 0011 para aceptar la
pérdida. La variante exacta en f32 gana +20 %.

### Atención de prefill (`flash_attn_gqa`)

Reescrita con el reparto de `kernel_flash_attn_ext` de llama.cpp:

- threadgroup = **8 queries de una cabeza** (antes: 16 filas = 4 queries × 4 cabezas del grupo
  GQA); 4 simdgroups, bloques de 64 claves;
- S = Q·Kᵀ: cada simdgroup, 2 tiles de 8 claves; K se lee directo de la caché como fragmento
  (`simdgroup_half8x8` sin convertir con KV f16; producto mixto f32 × f16 con acumulación f32);
- softmax con 2 filas por simdgroup y 2 claves por lane (`simd_max`/`simd_sum`);
- **O en memoria threadgroup** (f32), reescalada por fila en el softmax y cargada en 4
  fragmentos por simdgroup para P·V;
- **Q y P en f32**: no hay cambio de precisión. Los únicos redondeos son los de la caché.

`flash_lab`, T = 512, KV f16, M1 Pro:

| Posición | antes | nueva (Q f32) | Q en half | O en registros |
|---:|---:|---:|---:|---:|
| 0 | 2,28 ms | 1,25 ms | 1,25 | — |
| 1536 | 12,9 ms (1161 GFLOP/s) | 7,47 ms (2013) | 7,47 | 9,76 |
| 15872 | 111–119 ms (~1200) | 66,2–66,9 ms (2023–2044) | 66,2 | 84,4 |

- Error contra el kernel anterior: 2e-6–7e-6 · max|o| (Q f32) y 2e-4 con Q en half, sin
  ganancia de velocidad: Q queda en f32.
- Con O en registros (8 floats por hilo) rinde ~25 % menos que en memoria threadgroup.
- Reescalado perezoso (FlashAttention-3) y máscara solo en los bloques de la diagonal: sin
  ganancia medible (1917 GFLOP/s a 16K, ruido).

Tests de `brasa-kernels` con la misma tolerancia de antes (1e-5 · max|v|): 9,9e-7 (f32), 9,4e-7
(f16), 8,2e-7 (q8_0).

## Verificación de calidad del GEMM f16

Kernels (`cargo test -p brasa-kernels --test ops`):

- contra una referencia que redondea igual (pesos decuantizados y x a f16, producto en f64):
  ≤ 1e-5 · Σ|w·x| (medido 5,6e-7);
- contra la referencia sin redondear: 5,0e-5 · Σ|w·x| (cota del test: 2⁻¹⁰).

Forward (T1.6, `cargo test --release -p brasa-models --test forward -- --ignored`), 2898
posiciones de teacher forcing en bloques de 64 tokens:

| GEMM | KV | logits peor | TF prefill distintos + empates | top-1 vs FP32 | top-1 vs KV sin redondear |
|---|---|---:|---:|---:|---:|
| f32 | f32 | 4,4e-6 | 0 + 0 | 88,3 % | — |
| f32 | f16 | 2,3e-4 | 0 + 0 | 88,3 % | 100 % |
| f16 | f16 | 6,3e-4 | 0 + 2 | 88,3 % | 99,93 % |
| f32 | q8_0 | 7,4e-3 | 0 + 12 | 88,1 % | 98,45 % |
| f16 | q8_0 | 8,5e-3 | 0 + 17 | 88,1 % | 98,34 % |

- Las 2 posiciones que cambian de top-1 con GEMM f16 y KV f16 tienen brecha top-1/top-2 ≤
  3,5e-4 · |top-1|.
- La calidad contra el modelo sin cuantizar no cambia (88,3 %), como predijo ADR 0011.

Tolerancias de las pruebas `forward_gemm_f16_*`:

- logits ≤ 2e-3 (`LOGIT_TOL_GEMM16`; con KV Q8 manda su 2e-2);
- empate si la brecha es < 1e-3 · |top-1| (`TIE_REL_GEMM16`).

Las pruebas `forward_kv_*` siguen con el GEMM exacto y las tolerancias de antes.

## Consecuencias

- `Kernels::gemm` usa entradas f16; `Kernels::gemm_with(.., GemmInput::F32)` y
  `Qwen3::prefill_gemm` permiten el camino exacto.
- El decode no cambia (GEMV en f32).
- Resultados de punta a punta y estado de la compuerta: ver la sección siguiente.
