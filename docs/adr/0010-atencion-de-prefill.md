# ADR 0010 — Atención de prefill (T3.2)

Estado: aceptada en f32. Media precisión (Q y P en f16) probada y descartada (2026-10-05)

## Contexto

`flash_attn_f32` rendía ~0,78 TFLOPS en el caso de T3.2 (`flash_attention` T=512 a 16K). A 16K la
atención son ~79 TFLOP, la mayor parte del TTFT. Medido en M1 Pro con `cargo bench -p brasa-kernels`
y `cargo run --release -p brasa-kernels --example pipeline_limits`.

## Decisión

`flash_attn_gqa` (variante para grupos GQA 1, 2, 4 y 8; la anterior queda para el resto):

- Threadgroup = 16 filas (queries × cabezas del grupo GQA) de una cabeza KV, 4 simdgroups.
- El trabajo se reparte por claves en S = Q·Kᵀ y por dimensiones en O = P·V, como el kernel de
  llama.cpp. Así cada tile de K y V se lee una vez por threadgroup y cada simdgroup guarda solo
  un cuarto de O.
- Bloques de 64 claves (`KV_ALIGN` pasa a 64).
- Reescalado perezoso del softmax (FlashAttention-3): m solo cambia si el máximo nuevo lo supera
  en más de 8, y O no se reescala cuando todas las filas tienen α = 1.
- Toda la aritmética en f32.

Lo que se midió en el camino, para no repetirlo:

| Cambio | T=512 a 16K, KV f16 |
|---|---:|
| `flash_attn_f32` (antes) | 784 GFLOP/s |
| K/V en memoria threadgroup, 8 simdgroups, softmax por scratch | 627 |
| ídem con softmax en registros | 483 |
| reparto por claves/dimensiones, 32 filas (20 KiB de memoria threadgroup) | 499 |
| ídem con 16 filas (10 KiB) | 860 |
| + reescalado perezoso | 1151 |
| + bloques de 64 claves (f32: 955 → 1075) | 1147 |

Conclusiones:

- La ocupación (memoria threadgroup por threadgroup) pesó más que el reparto de lecturas.
- Escalar O por elemento en cada bloque costaba ~25 % del kernel.
- Sacar las lecturas de K o de V cambia poco, así que el límite está en el ritmo de MMA en f32.
- Rellenar filas contra conflictos de banco y declarar `max_total_threads_per_threadgroup` no
  cambiaron nada.

## Consecuencias

- 1,47× sobre el kernel anterior: el criterio de T3.2 (≥ 3×) **no se cumple**.
- Hacia ~2,3 TFLOPS (lo que logra llama.cpp en este equipo, estimado desde su TTFT a 16K), el
  camino conocido es hacer Q·K y P·V con Q, K, V y P en f16 y acumular en f32, como llama.cpp.
  Redondear Q y P cambia los puntajes ~5e-4 relativo, así que hace falta una referencia que
  redondee igual y otra tolerancia, igual que en ADR 0009. Es una decisión de precisión
  (regla 3) y queda para el usuario.

## Prueba de Q y P en f16 (2026-10-05)

El usuario aprobó probarlo con una condición: entra solo si el microbenchmark mejora ≥ 1,5×. Si no
acelera, no vale la pérdida de precisión (regla 3).

Variante probada sobre `flash_attn_gqa` con KV f16:

- Q escalada y P en memoria threadgroup como `half`.
- K y V cargados como `simdgroup_half8x8` sin convertir.
- S, O, m y l en f32 (`simdgroup_multiply_accumulate` mixto).
- Salida escrita directo desde los fragmentos.

El error contra la referencia sin redondear fue 5e-4 / max|v|, y el layout de los fragmentos se
verificó correcto.

Medido en M1 Pro 16 GB, macOS 27.0, commit 6cd264a, con `cargo bench -p brasa-kernels`
(`flash_attention` T=512 a 16K):

| Variante | GFLOP/s |
|---|---:|
| f32 (la aceptada) | 1167 |
| Q/K/V/P en f16, 16 filas | 1180 |
| ídem con 32 filas (usa la memoria threadgroup liberada) | 627 |

No llega a la condición (+1 %), así que se revirtió. En el M1 el MMA en f16 no es más rápido que
en f32, y achicar la memoria threadgroup tampoco ayuda. Lo que le da ventaja a llama.cpp en este
equipo no es la precisión. Queda para un intento futuro, con perfilado (Xcode GPU counters) antes
de tocar el kernel. Con esto no hay ninguna decisión de precisión pendiente en la atención.

