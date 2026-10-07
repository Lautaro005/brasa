# ADR 0029 — Autotuner de decode y perfiles de memoria

Estado: aceptada (fase 4)

## Contexto

PLAN.md, fase 4: autotuner con fingerprint y base de tuning (ADR 0026), perfiles de 8 y 16 GB, y
"quick tune menor a 1 minuto". ADR 0026 fijó el fingerprint, el formato de la base y la lista de
constantes candidatas, y dejó para el tuner qué se tunea y con qué mecanismo. En paralelo se
reescriben los GEMM de prefill y `flash_attention` (rama `fase-3-prefill`): tunear esos kernels
ahora chocaría con ese trabajo y mediría código que va a cambiar.

## Decisión

### Qué se tunea

Solo parámetros de decode que se pueden elegir al lanzar, sin recompilar, y que **no cambian los
resultados**:

| Kernel | Parámetro | Por defecto | Candidatos |
|---|---|---|---|
| `gemv_scaled3_q4_0` (q, k, v) | simdgroups por threadgroup (`SG`) | 2 | 1, 2, 4, 8 |
| `gemv_fast_q4_0` (o, down), `gemv_fast_q8_0` | `SG` | 2 | 1, 2, 4, 8 |
| `gemv_scaled_swiglu_q4_0` (gate, up) | `SG` | 2 | 1, 2, 4, 8 |
| `gemv_scaled_q6_0` / `gemv_fast_q8_0` (lm_head) | `SG` | 2 | 1, 2, 4, 8 |
| `attn_decode_lanes` | tramos (simdgroups) por threadgroup | 4 | 1, 2, 4, 8 |

- **Por qué estos.** Cuántos simdgroups van en un threadgroup cambia la ocupación y el reparto de
  threadgroups entre núcleos de GPU, que es justo lo que cambia entre chips (14/16 núcleos en la
  M1 Pro, 8/10 en la M2). Cada fila de un GEMV y cada tramo de claves de la atención los calcula un
  simdgroup con el mismo código, en el mismo orden de sumas, cualquiera sea `SG`: la salida es la
  misma bit a bit. Lo verifican `crates/kernels/tests/launch.rs` (cada kernel con cada candidato)
  y `crates/models/tests/launch.rs` (logits del modelo completo, prefill y decode, KV f16 y Q8).
- **Mecanismo.** El valor sigue siendo constante de compilación (`GSG` en `matmul.metal`,
  `SG_PER_TG` en `decode_attention.metal`), con los valores de siempre por defecto. Para otro
  valor, `Kernels::set_launch` compila al cargar una variante con `#define GEMV_SG n` o
  `#define LANES_SG n` antepuesto (como `GQA_G`), solo de los kernels que la base pide, y la
  valida contra `maxTotalThreadsPerThreadgroup`. Sin base no se compila nada extra. Se probó
  primero leer `[[simdgroups_per_threadgroup]]` en tiempo de ejecución (sin variantes): mismos
  bits, pero el decode quedó ~1 % más lento a 2K (`gemv_scaled_swiglu` +1,2 %,
  `attn_decode_lanes` +1 % en `decode_breakdown`, 4 corridas alternadas; con la constante
  volvió al tiempo de main). Por eso, variantes compiladas.
- **Qué no se tunea, y por qué.**
  - `DECODE_CHUNK` (claves por tramo): cambia cómo se reparte el softmax entre tramos (otros bits)
    y el tamaño del scratch de parciales que reservan el planner y `Qwen3::load`.
  - `NR` (filas por simdgroup de los GEMV): constante de compilación que dimensiona arreglos en
    registros; tunearla exige variantes compiladas y medir presión de registros.
  - GEMM de prefill y `flash_attention`: en reescritura en otra rama. Cuando se integre, sus
    parámetros (ADR 0026) entran con este mismo esquema de claves.
  - Hilos de los kernels elemento a elemento y de `add_norm_prep`: suman menos del 5 % del decode
    (`decode_breakdown`) y `NORM_PREP_TG` dimensiona un buffer.
  - Tamaño del bloque de prefill (host): ya es `--chunk`, y su efecto depende del GEMM en
    reescritura.

### Cómo se mide

`brasa tune [modelo]` (`brasa_tuner::tune`): formas reales del modelo (del encabezado del
`.brasa`), pesos y KV aleatorios (el tiempo no depende de los valores). Cada GEMV se encola tantas
veces como capas tiene el modelo, rotando copias de los pesos en direcciones distintas hasta
512 MiB (para no medir con datos en la caché del sistema); la atención, 36 veces rotando hasta 8
cachés. Tiempo de GPU del command buffer (`GPUStartTime`/`GPUEndTime`) dividido por la cantidad de
dispatches. Los candidatos se **intercalan** en cada ronda (la deriva de reloj o temperatura no
favorece a ninguno), con una ronda de calentamiento descartada.

- **quick** (por defecto): 15 rondas; GEMV de decode y atención a 2048 y 16 384 claves con KV f16
  y Q8.
- **full** (`--full`): 41 rondas; además KV f32 y 512, 8192 y 32 768 claves (hasta `--ctx`).

**Regla de elección.** Gana el candidato de mediana más baja solo si mejora la mediana del valor
por defecto en al menos 3 % **y** le gana en al menos 80 % de las rondas pareadas, y siempre que
la dispersión del valor por defecto (rango intercuartil / mediana) no pase de 5 %: con otra carga
en la GPU (medido: el escritorio grabando pantalla durante una corrida) los tiempos se dispersan
20–40 % y una corrida eligió `SG=8` con "+20,7 %" en la atención; con el control de dispersión esa
medición queda en el valor por defecto y `brasa tune` avisa. Si no, la entrada guarda el valor
por defecto. La primera versión (5 rondas, solo 2 % de mediana) elegía
ruido: dos corridas seguidas en la M1 Pro dieron 4 y 1 cambios distintos, todos de 2–9 % en
kernels que corridas largas muestran empatados.

**Base.** Clave ADR 0026: `kernel` = función MSL, `shape` = `rows=…,cols=…` (GEMV) o
`hkv=…,lk=…` (atención), `variant` = `""` o `kv=…,g=…`; `params` = `{"SG": n}`; `gpu_us` y
`default_us` = medianas en µs por dispatch de la misma corrida; `samples` = rondas. Se escriben
todas las entradas medidas, también las que quedan en el valor por defecto: son la evidencia.

### Cómo se usa

- `Session::load` (y por lo tanto `run`, `serve`, `benchmark`) llama a
  `brasa_tuner::resolve_current()`: abre la base del fingerprint actual y la traduce a un
  `Launch`. Sin base, base de otro fingerprint, dañada o de otro esquema: valores por defecto y un
  estado que lo explica (`TuningStatus`), nunca pánico. `BRASA_TUNING=off` fuerza los valores por
  defecto (para A/B con el mismo binario). Entradas que este binario no sabe usar se ignoran y se
  cuentan.
- La atención usa la entrada de la longitud más cercana en escala logarítmica a la actual.
- **Sin asignaciones en decode (regla 4):** la consulta recorre dos `Vec` armados al cargar.
  `tests/decode_alloc.rs` corre ahora con un `Launch` no trivial.
- `run` y `serve` imprimen el estado (`tuning: …`); `/api/status` lo informa en `tuning`;
  `brasa doctor` (y `--json`, `tuning.db`) muestra si hay base, cuántas entradas y la ruta.

### Perfiles de 8 y 16 GB

`brasa_memory::planner::Profile` reúne lo que antes estaba disperso:

| Perfil | Se elige | Presupuesto simulado | `serve` por defecto |
|---|---|---|---|
| 8gb | RAM < 12 GiB, o `--perfil 8gb` | estimado: 2/3 de la RAM − 512 MiB (ADR 0007) | 16K, KV Q8 |
| 16gb | RAM ≥ 12 GiB, o `--perfil 16gb` | estimado: 74 % de la RAM − 512 MiB (medido en M1 Pro) | 16K, KV f16 |

- 16K porque los agentes mandan prompts de 15K a 30K tokens (CLAUDE.md). KV Q8 en 8 GB porque es
  el perfil de agente de CLAUDE.md y porque con f16 el plan de 16K (4,78 GiB con bloque de 512)
  entra por menos de 0,1 GiB en un presupuesto que es una estimación (4,83 GiB); con Q8 el plan es
  3,73 GiB y deja 1,1 GiB de margen (`brasa plan --perfil 8gb`). En 16 GB queda f16:
  entra con margen, es la que mide 100 % de top-1 contra la referencia (Q8: 98,4 %, ADR 0009) y era
  el valor por defecto de `serve`. `--kv` y `kv` del archivo mandan sobre el perfil.
- `serve --perfil 8gb|16gb` además carga con el presupuesto del perfil (el menor entre el del
  perfil y el de esta Mac): en la M1 Pro simula lo que entraría en la M2. `plan --perfil` ya
  existía; ahora usa el mismo tipo.
- `brasa config show` muestra la KV efectiva de `serve` y el perfil.
- Las cifras del perfil de 8 GB son **estimadas** hasta que corra `scripts/validate-8gb.sh` en la
  M2 (que ahora incluye `brasa tune` y el A/B del tuning).

## Resultados (M1 Pro 16 GB, 16 núcleos de GPU, macOS 27.0 26A428, 2026-10-06)

- **Quick tune:** 5,5 / 5,5 / 5,7 s en tres corridas (`brasa tune`, tiempo informado por el
  comando; 6 s de pared cada una). Aceptación de fase 4 (< 60 s): cumplida en la M1 Pro.
- **Elección:** las tres corridas dejaron las 9 entradas en el valor por defecto. Las diferencias
  entre candidatos son de 0–2 % salvo `SG=1`, que es 3–7 % más lento en GEMV de o y down y en la
  atención a 2K. En la M1 Pro los valores fijados a mano ya son los mejores (o empatan); el
  tuner lo confirma con evidencia en lugar de suponerlo. Si cambia algo en la M2 está por medir.
- **A/B de decode** (`profile_decode`, mediana de 4 corridas alternadas, tiempo de GPU por token):
  ver la tabla en `docs/bench/m1pro-16gb/tuning-ab.md`. Leer `[[simdgroups_per_threadgroup]]` en
  lugar de una constante no cambia el decode más allá del ruido.

## Consecuencias

- Toda máquina nueva o actualización de macOS arranca sin base y con los valores de siempre;
  `brasa tune` la crea en segundos.
- Los kernels de decode tuneables quedan con el tamaño de threadgroup como parámetro del host:
  un kernel nuevo de decode con la misma estructura entra sumando una `GemvOp` o una clave.
- Los parámetros que cambian bits o buffers (`DECODE_CHUNK`, `NORM_PREP_TG`) y los de prefill
  siguen fuera; entrar exige el test de equivalencia con tolerancia (reglas 2 y 3) y, para los
  que dimensionan buffers, que el planner use el valor de la base o el máximo de los candidatos.
