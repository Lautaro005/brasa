# ADR 0006 — Formato nativo de pesos `.brasa` y cuantización Q4/Q8

Estado: aceptada (T1.3)

## Decisión

**Esquemas** (bloques de 32 a lo largo de la dimensión de entrada, escala junto al bloque):

| dtype | Bloque | Bytes | bits/peso | Cuantización | Decuantización |
|---|---|---:|---:|---|---|
| `q4_0` | escala f16 + 16 B de nibbles | 18 | 4,5 | `d = m / -8` (m = valor de mayor módulo, con signo); `q = clamp(round(w/d) + 8, 0, 15)` | `w = d · (q − 8)` |
| `q8_0` | escala f16 + 32 × i8 | 34 | 8,5 | `d = max|w| / 127`; `q = round(w/d)` | `w = d · q` |
| `f32` | — | 4 | 32 | sin pérdida (BF16 → F32 es exacto) | — |

- Orden de nibbles de `q4_0` (igual que llama.cpp): el byte `j` guarda el elemento `j` en el
  nibble bajo y el `j + 16` en el alto. Permite decuantizar 16 + 16 elementos con dos máscaras.
- `q` se calcula con la escala **ya redondeada a f16**, para que la decuantización sea consistente.
- **Qué va en cada esquema** (Qwen3): proyecciones de atención y MLP en `q4_0`; la tabla de
  embeddings, que también es el `lm_head` (pesos atados), en `q8_0`, porque el error en la
  proyección de salida pega directo en los logits; normas (RMSNorm, QK-norm) en `f32`.
- Mejoras de calidad (búsqueda de escala por RMSE, Q4 con mínimo, grupos de 64) quedan para
  después y entran solo si mejoran una medición de calidad.

**Archivo `.brasa`:**

```
0       "BRSA"  u32 versión (=1)  u64 largo del JSON
16      JSON (UTF-8) con config del modelo, origen, esquema y directorio de tensores
...     relleno hasta 16 KiB
data    cada tensor empieza alineado a 16 KiB (página de Apple Silicon) y su largo
        se rellena a múltiplo de 16 KiB
```

Los offsets del directorio son absolutos. La alineación a página permite `mmap` del archivo y
envolver cada tensor en un `MTLBuffer` sin copia (`newBufferWithBytesNoCopy`). Cada tensor tiene
su `sha256`; el directorio tiene `data_sha256` = sha256 de la concatenación de los sha256 de los
tensores en orden. El JSON incluye repo y commit de origen y la versión del conversor.

Un modelo es una carpeta: `model.brasa` + `tokenizer.json` + `tokenizer_config.json`.

**Conversor:** `tools/convert_brasa.py` (Python offline, regla 7). **Lector:** `crates/quant`
(Rust): parseo, `mmap`, verificación de hashes y decuantización de referencia en CPU.

## Tolerancia documentada (criterio de T1.3)

Comparando cada peso decuantizado `ŵ` contra el original `w` (BF16 → F32), por bloque con escala
`d`:

- `q4_0`: `|w − ŵ| ≤ |d|/2 · (1 + 2⁻¹⁰)` salvo los elementos del extremo opuesto al de mayor
  módulo, que pueden saturar en `q = 15`: para esos `|w − ŵ| ≤ |d| · (1 + 2⁻¹⁰)`.
  (El factor `2⁻¹⁰` cubre el redondeo de la escala a f16.)
- `q8_0`: `|w − ŵ| ≤ |d|/2 · (1 + 2⁻¹⁰)`.
- `f32`: igualdad exacta.

Además se reporta el error relativo (RMSE / RMS de los pesos) por tensor.

## Consecuencia para T1.6

Los logits de Brasa (pesos Q4) no pueden compararse con tolerancia estrecha contra la referencia
FP32 sin cuantizar. T1.6 usa dos comparaciones: (1) **correctitud del engine**: GPU contra una
referencia CPU que usa los mismos pesos decuantizados, con tolerancia estrecha; (2) **calidad de
la cuantización**: contra las fixtures FP32, con métricas de coincidencia de top-1 y KL.
