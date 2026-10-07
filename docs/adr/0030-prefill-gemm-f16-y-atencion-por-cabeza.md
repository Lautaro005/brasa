# ADR 0030 — Prefill: entradas f16, GEMM de 64 × 64 y atención por pares de cabezas

Estado: aceptada (2026-10-07)

## Contexto

La compuerta de la fase 3 está bloqueada por el prefill. Medido en M1 Pro 16 GB (commit 8dc9fa3,
`brasa benchmark`, bloques de 128 tokens):

| Ctx | Prefill Brasa | llama.cpp (T0.5) |
|---:|---:|---:|
| 2048 | 299 tok/s | 490 |
| 8192 | 195 | 356 |
| 16384 | 138 | 260 |

El usuario pidió volver a intentar operandos f16 si hacía falta para alcanzar el objetivo, con
ADR, pérdida medida contra la referencia y tolerancia documentada (ADR 0010 y 0011: los intentos
anteriores no aceleraron).

## Perfil antes de tocar kernels

Herramientas nuevas (ejemplos):

- `brasa-models` `profile_prefill -- <n> <chunk>`: prompt completo con el modelo real (pared y
  GPU). Con `PP_DEPTHS=0,1536,7680,15872`, un bloque en cada posición, como
  `llama-bench -p 512 -d ...`; con `PP_VERBOSE=1`, el tiempo de cada bloque.
- `brasa-kernels` `prefill_breakdown -- [T] [pos0] [kv]`: tiempo de GPU por operación (×36 capas).
- `brasa-kernels` `gemm_lab` y `flash_lab`: compilan un kernel candidato desde un `.metal` y lo
  comparan con el actual (velocidad y error).

Bloque de 512 tokens según la posición, GPU, M1 Pro 16 GB, macOS 27.0, KV f16:

| Posición | Brasa antes | llama.cpp (`llama-bench`, build 7fe450e19) |
|---:|---:|---:|
| 0 | 1442 ms (355 tok/s) | 960 ms (533 tok/s) |
| 1536 | 1833 ms (279) | 1161 ms (441) |
| 7680 | 3393 ms (151) | 1963 ms (261) |
| 15872 | 5511 ms (93) | 3106 ms (165) |

Desglose (`prefill_breakdown`, T = 512, ms por bloque, 36 capas) — detalle en
`docs/bench/m1pro-16gb/prefill-desglose.md`:

- en la posición 0, los GEMM son 1024 de 1098 ms (~2,6 TFLOPS efectivos contra ~3,9 de llama.cpp);
- en 15872, la atención es 2536 de 3605 ms (~113 ms por capa contra ~60 de llama.cpp);
- el resto (RMSNorm, QK-norm + RoPE, sumas, SwiGLU) es < 3 %.

Había que acelerar las dos partes ~1,5–1,9×.

## Decisión

### 1. GEMM (`matmul_tiled.metal`)

Estructura de `kernel_mul_mm` de llama.cpp (solo la idea; el kernel es propio):

- se calcula Cᵀ = X·Wᵀ, así cada fragmento del acumulador se escribe directo en `y`;
- índices `ushort`; los fragmentos de un paso de K se cargan todos antes de los productos;
- sin prefetch a registros del paso siguiente;
- q4_0 se decuantiza con máscaras sobre pares de bytes: `d1·(q & m) + md` = `d·(q − 8)`,
  exacto en f32;
- **pesos y activaciones en f16** en memoria threadgroup (`PrefillPrecision::F16`, la ruta
  caliente), acumulación en f32. `PrefillPrecision::F32` (mismo kernel con float) queda para
  verificar.

Después:

- **bloques de 64 filas × 64 tokens**: 16 acumuladores por simdgroup;
- los tokens que no completan un bloque de 64 van a la variante de 32 tokens (8 acumuladores).

`gemm_lab`, T = 512, M1 Pro (GFLOP/s; las mediciones de cada fila son A/B en la misma corrida):

| Variante | q_proj | gate/up | down |
|---|---:|---:|---:|
| antes (f32, ADR 0011) | 2813 | 2835 | 2815 |
| estructura nueva, entradas f32 (exacto) | 3369 | 3387 | 3353 |
| ídem con prefetch a registros, f16 | 3232 | 3251 | 3211 |
| estructura nueva, entradas f16 | 3828 | 3852 | 3809 |
| ídem con decuantización por desplazamientos | 3528 | 3550 | 3514 |
| pasos de 64 en K | +2,3 % | +2,3 % | +2,4 % |
| 64 × 64 por threadgroup, 16 acumuladores, con código de borde en la salida | 478 | 399 | 425 |
| ídem sin código de borde (bloques completos) | 4000 | 4043 | 3980 |
| ídem con x en f16 en memoria del dispositivo | +2,8 % | +2,8 % | +2,9 % |

Lo que se aprendió:

- El derrumbe con 16 acumuladores (~10×, también en ADR 0011) no es presión de registros. Lo
  causa cualquier código de borde en la salida, aun con índices constantes: el compilador saca
  los acumuladores de registros. Por eso los bloques incompletos van a la variante de 32.
- El orden de la grilla (filas o tokens en x) no cambia nada (±0,5 %).
- Sin efecto o peor:
  - 128 × 64 con 8 simdgroups (−9 %);
  - pasos de 64 en K con 16 acumuladores (−1 %);
  - A guardada como [fila][k] con cargas transpuestas (−3,5 %);
  - otro orden de cargas en el bucle interno (0 %).

### 2. Entradas del GEMM en f16 desde el kernel anterior

RMSNorm (`rms_norm_f16`), la atención y SwiGLU escriben su salida en f16 (`ws.xh`), que es lo
único que lee el GEMM siguiente. Son los mismos bits que antes, porque el GEMM igual redondeaba x
a f16. Se ahorra la conversión en cada threadgroup y la mitad de las lecturas de x (+2,8 % en el
GEMM).

### 3. Menos pasadas

- **`gemm_multi`**: q, k y v (y gate, up en la ruta exacta) en un dispatch (+0,4 %).
- **`gemm_swiglu`**: gate, up y SwiGLU en un kernel. Cada threadgroup calcula 32 filas de gate y
  las mismas 32 de up, y escribe silu(gate)·up en f16 sin pasar gate ni up por memoria (mismos
  bits; 439 ms contra 445 + 10 por bloque).
- **`qk_norm_rope_store`**: un simdgroup por fila (4 por threadgroup), sin barreras. Mismos bits;
  2,5× más rápido en prefill, algo más rápido en decode.

Probado y descartado: la suma residual dentro del GEMM (y += W·x). Cargar el fragmento de y en
la salida hunde el kernel ~100×, el mismo efecto que el código de borde.

### 4. Atención de prefill (`flash_attn_gqa`)

Reescrita con el reparto de `kernel_flash_attn_ext` de llama.cpp, más K y V compartidos entre
cabezas:

- threadgroup = 8 queries × **2 cabezas del mismo grupo GQA** (antes: 4 queries × 4 cabezas,
  16 filas); 4 simdgroups, bloques de 64 claves;
- S = Q·Kᵀ: cada simdgroup, 2 tiles de 8 claves para las 2 cabezas. K se lee directo de la caché
  como fragmento (half con KV f16), y cada fragmento sirve a las 2 cabezas;
- softmax en f32: 4 filas por simdgroup y 2 claves por lane;
- **O en memoria threadgroup** (f32), reescalada por fila en el softmax;
- P·V con cada fragmento de V leído una vez para las 2 cabezas;
- **Q en f16** en memoria threadgroup (variante exacta con `FA_Q_F32`); salida en f16 para el
  GEMM de o_proj;
- los bloques de queries con más claves se despachan primero.

`flash_lab` / `flash_sweep`, T = 512, KV f16, M1 Pro (GFLOP/s):

| Variante | pos 1536 | pos 15872 |
|---|---:|---:|
| antes (ADR 0010) | 1161 | ~1200 |
| 1 cabeza, 8 queries, Q f32 | 2013 | 2044 |
| ídem con O en registros | 1541 | 1602 |
| ídem con bloques de 128 claves | 1773 | 1846 |
| 2 cabezas, Q f32 | 2123 | 2205 |
| **2 cabezas, Q f16** (la aceptada) | **2384** | **2482** |
| 4 cabezas, Q f16 (32 KiB de memoria threadgroup) | 1856 | 1980 |
| 2 cabezas, 8 simdgroups | +1,4 % | +0,3 % |

- Con KV Q8: 2612 GFLOP/s a 16K (fragmentos float armados por lane; la caché ocupa la mitad).
- Reescalado perezoso y máscara solo en la diagonal: sin ganancia.

## Verificación de calidad

Kernels (`cargo test -p brasa-kernels`):

- **GEMM f16**:
  - contra una referencia que redondea pesos y x a f16 (producto en f64): ≤ 1e-5 · Σ|w·x|
    (medido 5,6e-7);
  - contra la referencia sin redondear: 5,0e-5 · Σ|w·x| (cota del test: 2⁻¹⁰).
- **Atención f16**:
  - contra la referencia que redondea Q·escala a f16, descontando el redondeo final de la salida
    (2⁻¹¹·|o|): ≤ 1e-5 · max|v| (medido 3,8e-7);
  - contra la referencia sin redondear: 5,3e-4 · max|v| (cota del test: 2e-3);
  - la variante exacta (Q y salida f32): 9,9e-7 contra la sin redondear (tolerancia de antes,
    1e-5).
- **`rms_norm_f16`, `swiglu_f16`, `gemm_multi`, `gemm_swiglu` y `qk_norm_rope_store`**: mismos
  bits que la versión f32 redondeada o que la secuencia sin fusionar (tests de igualdad exacta).

Forward (T1.6, `cargo test --release -p brasa-models --test forward -- --ignored`): 2898
posiciones de teacher forcing en bloques de 64 tokens, contra las fixtures existentes (que no
redondean pesos, activaciones ni Q):

| Prefill | KV | logits peor | TF prefill distintos + empates | top-1 vs FP32 | top-1 vs KV sin redondear |
|---|---|---:|---:|---:|---:|
| exacto (f32) | f32 | 4,4e-6 | 0 + 0 | 88,3 % | — |
| exacto (f32) | f16 | 2,3e-4 | 0 + 0 | 88,3 % | 100 % |
| **f16** | f16 | 8,1e-4 | 0 + 0 | 88,3 % | 100 % |
| exacto (f32) | q8_0 | 7,4e-3 | 0 + 12 | 88,1 % | 98,45 % |
| **f16** | q8_0 | 8,1e-3 | 0 + 12 | 88,3 % | 98,45 % |

- La calidad contra el modelo sin cuantizar no cambia (88,3 %), como predijo ADR 0011.
- Con KV Q8 y prefill f16, la posición casi empatada con mayor brecha tiene brecha top-1/top-2
  de 1,26e-2 · |top-1|. Es apenas más que el empate de Q8 solo (1e-2): las dos perturbaciones se
  suman.

Tolerancias de las pruebas `forward_prefill_f16_*`:

| Constante | Valor | Qué es |
|---|---:|---|
| `LOGIT_TOL_F16` | 2e-3 | logits (con KV Q8 manda su 2e-2) |
| `TIE_REL_F16` | 1e-3 · \|top-1\| | brecha que cuenta como empate |
| `TIE_REL_F16_KVQ8` | 2e-2 · \|top-1\| | brecha de empate con KV Q8 |

Las pruebas `forward_kv_*` y las de capas (T1.5) usan los kernels exactos con las tolerancias de
antes.

## Bloque de prefill de 1024 tokens

`--chunk` pasa a 1024 por defecto en `serve`, `run`, `plan`, `benchmark` y la GUI. Antes era 512
en `serve` y 128 en el resto.

Medido en M1 Pro, con `profile_prefill` y `brasa benchmark` alternados en la misma sesión:

- con 1024 el prefill rinde +0,9 % (2K) y +0,7 % (8K) más que con 512;
- con carga de escritorio, +3 % a 2K (471 contra 457 tok/s);
- 2048 suma otro +0,5 % a 2K.

El workspace crece de ~0,09 a ~0,17 GiB (`brasa plan`). En el perfil de 8 GB, 16K con KV Q8
sigue entrando: 3,82 de 4,83 GiB.

## Consecuencias

- `Kernels::gemm`, `gemm_multi`, `gemm_swiglu` y `flash_attention` usan entradas f16. Con
  `PrefillPrecision::F32` (`*_with`, `Qwen3::prefill_precision`) se usa el camino exacto. Con 8
  tokens o menos por forward, el modelo usa GEMV y la atención en f32.
- El decode no cambia de precisión (GEMV en f32).
- Resultados de punta a punta y estado de la compuerta: `docs/bench/baseline.md` y la sección
  "Fase 3" del plan.
