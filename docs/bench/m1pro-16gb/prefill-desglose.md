# Prefill: desglose y comparación por posición (fase 3, ADR 0030)

Apple M1 Pro 16 GB (16 núcleos GPU), macOS 27.0. Qwen3-4B q4_0 (`models/qwen3-4b-q4`), KV f16.
Tiempos de GPU (mediana). Los tiempos "después" se midieron con la máquina liviana, antes de las
corridas finales de `brasa benchmark` (ver al final).

## Desglose de un bloque de 512 tokens (×36 capas)

`cargo run --release -p brasa-kernels --example prefill_breakdown -- 512 <pos0>`, en ms.

Antes: commit 0a178aa. Después: commits 74948f2 (gate + up + SwiGLU) y e8077ce
(`qk_norm_rope_store`).

| Operación | pos 0 antes | pos 0 después | pos 15872 antes | pos 15872 después |
|---|---:|---:|---:|---:|
| RMSNorm ×2 | 3,7 | 3,1 (salida f16) | 3,9 | ~3,1 |
| GEMM q, k, v | 160,7 | 145,0 | 162,4 | ~145 |
| QK-norm + RoPE + KV | 8,4 | 3,1 | 8,0 | ~3,1 |
| Atención | 46,5 | 40,8 | 2535,7 | ~2038¹ |
| GEMM o | 110,5 | 95,1 | 108,6 | ~95 |
| suma residual ×2 | 3,0 | 3,0 | 2,9 | ~3,0 |
| GEMM gate, up + SwiGLU | 501,6 + 12,4 | 439,1 (un kernel) | 516,3 + 12,3 | ~439 |
| GEMM down | 251,1 | 225,8 | 254,9 | ~226 |
| **suma** | **1097,8** | **~955** | **3604,9** | **~2952** |

¹ `flash_sweep f16`, pos 15872: 56,6 ms por capa (2390 GFLOP/s).

Notas:

- La columna "pos 15872 después" sale de las mediciones por kernel en pos 0 (los GEMM y las
  operaciones por elemento no dependen de la posición) más la atención de `flash_sweep`.
- Con el modelo real, `profile_prefill` midió 2954 ms en esa posición.

## Bloque de 512 tokens según la posición, contra llama.cpp

| Posición | Brasa antes (0a178aa) | Brasa después | llama.cpp |
|---:|---:|---:|---:|
| 0 | 1442 ms (355 tok/s) | 962 ms (532) | 960 ms (533) |
| 1536 | 1833 ms (279) | 1154 ms (444) | 1161 ms (441) |
| 7680 | 3393 ms (151) | 1924 ms (266) | 1963 ms (261) |
| 15872 | 5511 ms (93) | 2954 ms (173) | 3106 ms (165) |

Comandos:

- Brasa: `PP_DEPTHS=0,1536,7680,15872 cargo run --release -p brasa-models --example
  profile_prefill -- 0 512`.
- llama.cpp: `llama-bench -m models/qwen3-4b-q4_0.gguf -p 512 -n 0 -fa 1 -r 2 -d
  0,1536,7680,15872` (build 7fe450e19, misma sesión que "antes").

## Prompt completo (`profile_prefill <n> 512`, tiempo de pared)

| Prompt | antes (bloques de 128) | después (bloques de 512) | llama.cpp (T0.5) |
|---:|---:|---:|---:|
| 1920 | 300,8 tok/s | 486,7 | 490,0 |
| 8064 | 200,1 | 356,4 | 355,6 |
| 16256 | (138 en `brasa benchmark`) | 262,4 | 259,7 |

Con bloques de 1024 (el nuevo valor por defecto), 1920 y 8064 tokens rinden +0,9 % y +0,7 % más
que con 512.

## Corridas de `brasa benchmark` (2026-10-07, con carga de escritorio)

Commits c69ad8e / 536c491, bloques de 1024. Durante estas corridas el escritorio estaba en uso:
WindowServer y WebKit entre 50 y 100 % de CPU, video en la GPU y ~8–10 GiB de swap. Los números
absolutos salen 3–10 % por debajo de los de la máquina liviana. Por eso **no se guardaron como
reportes** de esta carpeta (no entran en `baseline.md`). Brasa y llama.cpp se corrieron uno
detrás del otro, en las mismas condiciones:

| Ctx | Variante | Prefill Brasa | Prefill llama.cpp | Decode Brasa | Decode llama.cpp |
|---:|---|---:|---:|---:|---:|
| 2048 | KV f16 | 466,9 | 470,4 | 46,7 | 48,8 |
| 8192 | KV f16 | 342,8 | 342,4 | 35,4 | 34,4 |
| 16384 | KV f16 | 233,4 (242 → 223, cayendo) | 245,5 | 24,4 | 24,0 |
| 16384 | KV Q8 | 247,5 | 243,7 | 24,5 | 18,0 |

El reporte que cierra la compuerta (`brasa benchmark` a 2K, 8K, 16K y 16K KV Q8 en
`docs/bench/m1pro-16gb/`) queda para correr con la máquina liviana.
