# ADR 0012 — Tabla de embeddings (= lm_head) en q6_0

Estado: aceptada (T3.5, 2026-10-05; decisión delegada por el usuario, tomada con datos)

## Contexto

En decode, el lm_head lee toda la tabla de embeddings atada (151 936 × 2560) en cada token. En
q8_0 (ADR 0006) son 413 MB y 2,4 ms por token a ~174 GB/s: después de los GEMV de las capas, es el
término más grande que queda (`decode_breakdown`, T3.5). En el baseline, llama.cpp
(`models/qwen3-4b-q4_0.gguf`) guarda esa tabla en Q6_K: 6,56 bits/peso, 319 MB.

## Medición

`tools/eval_embed_quant.py` usa la referencia FP32 con los lineales q4_0 de `model.brasa` y cambia
solo la tabla, cuantizada desde los pesos BF16. Teacher forcing sobre las 2898 posiciones de
`fixtures/qwen3-4b`, contra la referencia FP32 sin cuantizar:

| Tabla | bits/peso | MB | top-1 vs FP32 | KL top-20 media | p99 |
|---|---:|---:|---:|---:|---:|
| q8_0 (actual) | 8,5 | 413 | 88,58 % | 9,42e-2 | 1,11 |
| q6_0 | 6,5 | 316 | 88,34 % | 9,49e-2 | 1,06 |
| q4_0 | 4,5 | 219 | 86,82 % | 1,08e-1 | 1,29 |

La fila q8_0 reproduce el 88,6 % que mide T1.6 en el engine. q6_0 pierde 0,24 puntos de top-1 y
0,7 % de KL. q4_0 pierde 1,8 puntos y 15 % de KL: más que todo lo que costó cuantizar las capas
en relación con la tabla, y quedaría por debajo de la precisión del baseline.

## Decisión

La tabla de embeddings pasa a **q6_0**, un formato propio con la misma idea que q4_0:

- Bloque de 32 valores en 26 bytes: `d` f16, `ql[16]` y `qh[8]`.
- `ql[j]`: bits 0–3 del elemento j (nibble bajo) y del elemento j + 16 (nibble alto).
- `qh[j]`: bits 4–5 de los elementos j, j + 8, j + 16 y j + 24, en los desplazamientos 0, 2, 4
  y 6.
- Valor: `d · (q − 32)`, con q ∈ [0, 63].
- Cuantización como q4_0: m es el valor de mayor módulo con signo, `d = f16(m / −32)` y
  `q = clamp(rint(x / d) + 32, 0, 63)`. El extremo queda exacto.

Los lineales siguen en q4_0 y las normas en f32. Lectura en el engine:

- embedding: kernel `embed_q6_0`;
- lm_head: `gemv_fast_q6_0`, con el mismo reparto que q4_0 (cada lane, media fila de bloque y los
  8 bytes de `qh`).

## Consecuencias

- −97 MB por token leído en decode y −97 MB de pesos, que dejan margen en el perfil de 8 GB.
- Cambian los pesos (`model.brasa`, nuevo `data_sha256`). Se regeneran las fixtures que dependen de
  ellos (`qwen3-4b-q4`, `-kvf16`, `-kvq8`). T1.6 se vuelve a correr y la métrica "Q4 vs FP32"
  pasa a medir el modelo nuevo.
- Los reportes de benchmark anteriores quedan asociados a los pesos viejos por su sha256.
- La conversión nativa de la rama `ui` (U4) tiene que sumar q6_0 para seguir dando los mismos
  sha256 que `tools/convert_brasa.py`.
