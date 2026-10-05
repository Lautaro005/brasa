# ADR 0009 — KV cache en f16 y Q8, y cómo se verifica

Estado: aceptada (fase 3, T3.1; Q8 en T3.4)

## Contexto

Hasta la fase 2 la KV cache es f32: 288 KiB por token en Qwen3-4B, 4,5 GiB a 16K. Medido en
M1 Pro el 2026-10-05 (`cargo bench -p brasa-kernels`):

- `decode_attention_gqa` a 16K tarda 1,65 ms por capa. Son ~59 ms por token solo en atención,
  frente a ~24 ms de los GEMV, y por eso el decode a 16K da 11,8 tok/s.
- `flash_attention` de prefill rinde ~0,7 TFLOPS contra 2,65 del GEMM tiled. A 16K la atención
  son ~79 TFLOP, que explican la mayor parte del TTFT de 153 s (corrida inválida por swap, pero
  la proporción se sostiene en los microbenchmarks).

llama.cpp usa KV f16 por defecto. El perfil de agente apunta a KV Q8 (PLAN.md).

## Decisión

**Tipos de KV.** `KvType::{F32, F16}` ahora y `Q8_0` en T3.4. Se elige al cargar (`Limits.kv`), el
planner usa su tamaño y los kernels de atención tienen una variante por tipo, generada con
templates de MSL. f32 sigue disponible como camino de verificación. f16 pasa a ser el defecto de
`run`/`serve`/`benchmark`.

**Escritura en la caché.** Las proyecciones K y V se escriben en un scratch f32, donde se aplican
QK-norm y RoPE. Después `store_kv` las convierte al tipo de la caché con redondeo al par más
cercano (el de Metal y el de `torch.half`). En f32 es una copia exacta.

**Verificación (regla 3).** Redondear la KV cambia los logits, así que la comparación contra la
referencia FP32 pura ya no alcanza 1e-4. Se separan dos preguntas:

1. *¿El engine calcula bien?* La referencia (`tools/qwen3_ref.py --kv f16`) redondea K y V
   al tipo de la caché en el mismo punto que el engine, después de RoPE. Las fixtures nuevas
   (`fixtures/qwen3-4b-q4-kvf16/`) se comparan con el mismo criterio de T1.6: error relativo de
   logits ≤ 1e-4, mismo top-1, y teacher forcing por los caminos de prefill y decode.
   **Tolerancia de logits con KV redondeada: 1e-3** (el teacher forcing se sigue exigiendo
   exacto). Medido el 2026-10-05:
   - El engine con KV f16 dio 2,05e-4 contra su referencia, con teacher forcing perfecto en
     2898 + 310 posiciones.
   - No es un error del engine. La misma referencia, con K y V perturbados en 1e-7 relativo
     antes de redondear, se mueve entre 2,3e-4 y 7,1e-4 (`tools/kv_rounding_sensitivity.py`,
     prompts chat-no-think, tools-roundtrip y long-context). Una diferencia por debajo del
     ruido de una operación f32 alcanza para que algunos elementos redondeen al ULP de f16
     vecino.
   - 1e-4 no se puede pedir aunque el engine fuera perfecto; 1e-3 deja margen sobre el piso
     medido.
2. *¿Cuánto se pierde?* Contra la referencia sin redondear (`fixtures/qwen3-4b-q4`) se mide la
   coincidencia de top-1 en teacher forcing. Es informativa: se reporta, y un piso evita
   regresiones groseras.

Para cada kernel nuevo valen las reglas de siempre: referencia CPU (con el mismo redondeo de
entrada), test de equivalencia y microbenchmark.

**Kernel de decode.** Con KV f16 el `decode_attention_gqa` original no mejoraba (1,68 → 1,63 ms a
16K): lo limitaban la latencia y las barreras del staging en memoria threadgroup, no el ancho de
banda. Lo reemplaza `decode_attention_lanes`:

- cada lane toma una clave y calcula su producto con todas las queries del grupo GQA;
- el softmax online se reduce una vez cada 32 claves;
- V se lee coalescido;
- el tamaño del grupo es constante de compilación (se compila para 1, 2, 4 y 8). Con un valor en
  tiempo de ejecución, los arreglos por cabeza salían de registros y el kernel era ~1,5× más lento.

Medido en M1 Pro a 16K por capa:

| KV | Antes | `decode_attention_lanes` |
|---|---:|---:|
| f32 | 1,68 ms | 1,06 ms |
| f16 | 1,63 ms | 0,71 ms |

A 2K da 0,13 ms y es más rápido en todos los contextos. Para grupos no compilados queda
`decode_attention` (una cabeza por threadgroup).

## Consecuencias

- KV a la mitad (2,25 GiB a 16K) y la mitad de bytes leídos por token en decode.
- Las fixtures crecen (6,5 MB por variante de KV).
- Q8 (T3.4) sigue el mismo esquema: `--kv q8_0` en la referencia, fixtures propias y medición de
  pérdida contra la referencia sin redondear.
