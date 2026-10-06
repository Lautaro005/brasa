# ADR 0026 — Fingerprint y base de tuning

Estado: aceptada (fase 4, primer paso: fingerprint y base; el tuner va aparte)

## Contexto

PLAN.md, fase 4: autotuner con fingerprint y base de tuning, con aceptación "quick tune menor a
1 minuto". Hoy los parámetros de lanzamiento de los kernels (filas por threadgroup, tamaños de
tile, claves por bloque) son constantes fijadas a mano en la M1 Pro. Antes de escribir el tuner
hay que decidir qué identifica a una máquina para que un resultado siga siendo válido, dónde se
guardan los resultados y cuándo se descartan. Regla 9 de CLAUDE.md: el fingerprint es local y no
incluye número de serie ni identificadores personales.

## Decisión

**Fingerprint** (`brasa_tuner::Fingerprint`). Campos, todos de `hardware_info()` salvo el último:

| Campo | Fuente | Por qué |
|---|---|---|
| `chip` | `machdep.cpu.brand_string` | define la GPU y el ancho de banda |
| `gpu_cores` | IOKit `gpu-core-count` | el mismo chip viene con distinta cantidad de núcleos |
| `cpu_performance_cores`, `cpu_efficiency_cores` | `hw.perflevel*.physicalcpu` | distingue binnings |
| `memory_bytes` | `hw.memsize` | los perfiles de 8 y 16 GB eligen formas distintas |
| `metal_family` | Metal (`apple7`, `apple8`, …) | features de la GPU |
| `macos_version`, `macos_build` | `kern.osproductversion`, `kern.osversion` | ver abajo |
| `kernels_version` | hash de las fuentes MSL | ver abajo |

- **Compilador y driver Metal.** `MTLDevice` no expone la versión del compilador MSL ni del
  driver AGX; los dos vienen con el sistema. Además el runtime no fija `languageVersion` (usa la
  última del sistema). Por eso el build de macOS (`24F74`, incluso con sufijo de respuesta rápida
  de seguridad) hace de versión del compilador y del driver: cualquier actualización del sistema
  invalida la base. Con un quick tune de menos de un minuto, el costo es aceptable.
- **Versión de los kernels.** Primeros 16 hex del SHA-256 de `brasa_kernels::sources::ALL`
  (nombre y contenido de cada `.metal`, con largo delante). Un test verifica que la lista cubra
  todos los archivos de `crates/kernels/src/metal/`. Los parámetros de lanzamiento del host que
  pasen a tunearse quedan en la clave de la base, no en este hash.
- **Excluidos a propósito:** número de serie, `IOPlatformUUID`, hostname, nombre de usuario,
  direcciones MAC y cualquier otro dato propio de la unidad. Tampoco entra `hw.model`
  (`MacBookPro18,1`): no cambia la GPU, y sacarlo hace que dos Macs iguales compartan la base. El
  id no identifica una unidad: lo comparten todas las Macs con el mismo chip, la misma RAM, el
  mismo macOS y el mismo binario. No sale de la máquina (sin telemetría, regla 9).
- **Hash.** Texto canónico versionado (`brasa-fingerprint v1`, luego una línea
  `campo=largo:valor` por campo en orden fijo; el largo impide que un valor imite otro campo) y
  SHA-256. El id son los primeros 16 hex (64 bits); alcanza porque el archivo guarda el
  fingerprint completo y la carga compara campo por campo. Un test fija el hash de un fingerprint
  de ejemplo: si cambia el formato hay que subir `FINGERPRINT_VERSION`.
- `brasa doctor` y `brasa doctor --json` (sección `tuning`) muestran el id y el fingerprint.

**Base de tuning** (`brasa_tuner::TuningDb`).

- **Ubicación:** `~/Library/Application Support/brasa/tuning/<fingerprint id>.json`, o
  `$BRASA_TUNING_DIR/<id>.json` (tests, CI, comparar bases). No va en `~/.config/brasa`
  (ADR 0022): eso es configuración que escribe el usuario, y esto lo genera la máquina. Tampoco en
  `~/Library/Caches`: macOS puede vaciarla, y entonces el rendimiento cambiaría entre corridas sin
  aviso. Tampoco en el repo: depende de la máquina.
- **Formato:** JSON con indentación (serde_json ya es dependencia; se lee a mano al revisar un
  reporte; son decenas de entradas). Salida determinista (entradas ordenadas por clave):

  ```json
  {
    "schema": 1,
    "fingerprint_id": "ad822116840bfa03",
    "fingerprint": { "chip": "Apple M1 Pro", "gpu_cores": 16, "...": "..." },
    "entries": [
      { "kernel": "flash_attn_gqa", "shape": "t=512,hq=32,hkv=8,d=128",
        "variant": "kv=f16,g=4", "params": { "BC": 64, "FA_ROWS": 16 },
        "gpu_us": 0.0, "default_us": 0.0, "samples": 20 }
    ]
  }
  ```

- **Clave** = `(kernel, shape, variant)`: función MSL, forma canónica de la llamada, variante de
  compilación que no se tunea (tipo de KV, grupo GQA). **Valor** = parámetros elegidos, mediana
  del tiempo de GPU con ellos y con los valores por defecto medidos en la misma corrida, y cantidad
  de repeticiones. Así cada entrada lleva su propia evidencia (regla 6).
- **Invalidación.** Un cambio de fingerprint (otra máquina, otro macOS, otros kernels) cambia el
  nombre del archivo: la base nueva arranca vacía y el engine usa los valores por defecto, que son
  los de hoy. Un archivo con el nombre correcto pero de otro `schema`, de otro fingerprint o dañado
  da un error claro (`TuningError::{Schema, Fingerprint, Corrupt}`, nunca pánico) y no se usa a
  medias; el llamador decide volver a tunear. Las bases viejas no se borran solas (pesan pocos KB).
- **Escritura atómica:** `<archivo>.tmp.<pid>`, `fsync` y `rename`. Un corte a mitad deja la
  versión anterior.
- **Sin entrada, valores por defecto.** El engine tiene que comportarse igual que hoy sin base.

**Qué se tunea primero** (solo la lista; el tuner es la tarea siguiente). Constantes actuales en
`crates/kernels`:

| Kernel | Parámetros (valor actual) | Dónde |
|---|---|---|
| `flash_attn_gqa` (prefill) | `FA_ROWS` = 16 (16 o 32), `BC` = 64 claves por bloque, `NSGF` = 4 simdgroups | `flash_attention.metal`, `FA_ROWS` en `lib.rs` |
| `flash_attn_f32` (sin GQA) | `BQ` = 32, `BKEYS` = 32 | `flash_attention.metal` |
| `gemm_tiled_q4_0/q8_0_f32` (prefill) | `BM` = 64, `BN` = 32, `BK` = 32, `NT` = 128, sub-bloque `SGM×SGN` = 32×16 | `matmul_tiled.metal` |
| GEMV de decode | `ROWS_PER_TG` = 4 (`GEMV_ROWS_PER_TG`) en `gemv_*`; `NR` = 4 filas por simdgroup en `gemv_fast_*`/`gemv_scaled_*` | `matmul.metal`, `lib.rs` |
| `attn_decode_lanes` / `attn_decode_partial` | `SG_PER_TG` = 4 (`DECODE_LANES_SG`), `CHUNK` = 128 (`DECODE_CHUNK`) | `decode_attention.metal`, `lib.rs` |
| elementwise, `rms_norm`, `softmax`, `add_norm_prep` | hilos por threadgroup = 256 | `ELEMENTWISE_TG`, `ROW_TG`, `NORM_PREP_TG`, `TG` en MSL |

Prioridad: prefill (GEMM y flash attention), que es el bloqueo medido en la fase 3; después
decode. Dos parámetros cambian tamaños de buffers preasignados: `DECODE_CHUNK` (parciales de
`decode_attention`) y `NORM_PREP_TG` (sumas parciales). Si se tunean, el planner de memoria
(ADR 0007) y la preasignación (regla 4) tienen que usar el valor de la base o el máximo de los
candidatos.

## Consecuencias

- `brasa-tuner` depende de `brasa-kernels` (para el hash de las fuentes) y de `sha2`/`serde_json`.
  `brasa-kernels` no depende del tuner, así que no hay ciclo.
- Toda fuente `.metal` nueva tiene que sumarse a `sources::ALL` (lo exige un test).
- Una actualización de macOS o un cambio en cualquier kernel obliga a re-tunear. Es deliberado.
- Una variante tuneada no se acepta solo por ser rápida: tiene que pasar el test de equivalencia
  del kernel con su tolerancia (reglas 2 y 3). El mecanismo para compilar variantes (prefijo
  `#define`, como ya se hace con `GQA_G`, o function constants) se decide con el tuner.
