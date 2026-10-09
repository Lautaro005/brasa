# ADR 0033 — KV cache TurboQuant (TQ4), opcional y medido

Estado: propuesta, con implementación (2026-10-09). No es el defecto de ningún perfil.

## Contexto

La KV cache de Qwen3-4B cuesta 144 KiB por token en f16 y 76,5 KiB en Q8 (ADR 0009). A 16K,
f16 ocupa 2,25 GiB y Q8 1,20 GiB. Se pidió probar TurboQuant (Zandieh et al., 2025,
arXiv:2504.19874) para ver si reduce más la memoria y el costo de la atención de decode sin
perder calidad.

TurboQuant, en la parte que se implementa aquí (la etapa MSE): por fila de un token y una
cabeza KV, se normaliza, se rota con una matriz ortogonal fija y cada coordenada se cuantiza con
un codebook de Lloyd-Max diseñado para la distribución de una coordenada de un vector unitario de
128 dimensiones. La etapa QJL del paper (1 bit sobre el residuo, para un producto interno sin
sesgo) **no** se implementa.

## Decisión

Se agrega `KvType::Tq4` (`--kv tq4`), con estos elementos:

- **Fila de 68 bytes** por (token, cabeza KV): 64 bytes de índices de 4 bits (dimensión `2b` en el
  nibble bajo del byte `b`), norma f16 en los bytes 64..66 y 2 bytes de relleno. Es 2× menor que
  Q8 y 3,8× menor que f16.
- **Rotación fija R** de 128 × 128, generada con Gram-Schmidt sobre una gaussiana de semilla
  constante (`brasa_quant::turbo::rotation`). Es la misma para todas las capas y cabezas, y ocupa
  64 KiB en la GPU.
- **Codebook** de 16 niveles calculado por Lloyd-Max sobre la densidad
  `(1 − x²)^((d−3)/2)` con d = 128 (`brasa_quant::turbo::codebook`). Sus constantes entran al
  código MSL como `TQ_CB` y `TQ_MID`, así que la GPU y la especificación en CPU usan los mismos
  valores.
- **Atención en el dominio rotado.** `q·k = (Rq)·(Rk)`, así que la caché guarda `R·k` (decodificado
  como `norma · codebook[c]`), el decode rota `q` una vez por cabeza con `R`, y la salida de la
  atención vuelve al dominio original con `Rᵀ` antes de `o_proj`. Así no hay que rotar cada
  token de la caché.
- **Escritura.** `qk_norm_rope_store` deja K y V en el scratch f32 (con RoPE aplicado), y después
  `tq_store` rota y cuantiza a la caché. `tq_rotate` rota Q y la salida, en f32 o f16 según la
  ruta de prefill (ADR 0030).

Archivos: `crates/quant/src/turbo.rs` (especificación), `crates/kernels/src/metal/tq.metal`
(rotación y escritura), `kv_access.metal` (acceso a la caché por tipo), `crates/models/src/qwen3.rs`
(forward). Pruebas: `crates/quant` (unitarias), `crates/kernels/tests/tq.rs` (GPU contra
especificación y referencia). Medición: `crates/models/examples/kv_quality.rs`.

## Verificación

Tolerancias, todas medidas en la M1 Pro 16 GB (macOS 27.0):

| Qué | Criterio | Resultado |
|---|---|---|
| Rotación GPU vs CPU (f32) | error ≤ 1e-5 · max \|y\| | 1,7e-6 (max 4,2) |
| Rotación GPU vs CPU (f16 de salida) | error ≤ 2⁻¹⁰ · max \|y\| | 9,8e-4 (max 5,0) |
| Escritura GPU vs especificación | códigos distintos ≤ 0,2 % | 0 de 65 536 |
| Normas f16 GPU vs especificación | ≤ 2 % de filas distintas | 0 de 512 |
| Error relativo de la caché (teórico Lloyd-Max 4 bits: 0,009497) | ±0,0005 | 0,00937 |
| Atención de decode GPU vs referencia del dominio rotado | error ≤ 1e-5 · max \|v\| | 2,1e-7 (lanes), 1,3e-7 (por cabeza) |
| Atención de prefill f32 vs referencia | ≤ 1e-5 | 7,6e-7 |
| Atención de prefill f16 (salida en f16) | ≤ 1e-3 | 4,6e-4 |

## Calidad (medida, no es una prueba de regresión)

`kv_quality` corre el forward de Qwen3-4B con la misma build para cada tipo de KV, contra la
referencia FP32 sin redondear (`fixtures/qwen3-4b-q4`, 10 prompts, 32 tokens greedy cada uno):

| KV | KV @2K | Greedy 32/32 | Tokens iguales | Error de logits | Top-1 teacher forcing |
|---|---:|---:|---:|---:|---:|
| f16 | 0,281 GiB | 10/10 | 100,0 % | 9,4e-4 | 100,00 % |
| Q8 | 0,149 GiB | 8/10 | 97,8 % | 2,3e-2 | 98,45 % |
| TQ4 | 0,075 GiB | 2/10 | 44,1 % | 8,2e-1 | 78,71 % |

En 16K, la KV es 2,25 GiB (f16), 1,20 GiB (Q8) y 0,60 GiB (TQ4).

**Conclusión de calidad: TQ4 tal como está implementado pierde mucha precisión.** No es un bug
de la rotación ni de la escritura, por dos motivos. Primero, la escritura coincide con la
especificación y el error relativo medido (0,0094) es el teórico de Lloyd-Max de 4 bits.
Segundo, un error de convención daría resultados casi aleatorios, no 79 % de top-1. El problema
es de magnitud: con 4 bits, el error relativo por elemento es ~0,95 % del cuadrado de la norma,
unas 200 veces el de Q8 por bloque de 32 (4,6e-5). En los logits eso es ~10 % de ruido sobre la
dispersión de los scores, y se acumula en 36 capas.

El paper reporta calidad neutra con ~3,5 bits, pero con la etapa QJL para las claves y con
canales atípicos a más bits. Ninguna de las dos cosas está aquí.

## Velocidad

Decode con `profile_decode` (M1 Pro 16 GB, macOS 27.0, mismo build, `BRASA_TUNING=off` para
las tres variantes, GPU sin otras cargas medibles). Tiempo de GPU por token:

| KV | 2K: GPU ms/token | 2K: tok/s | 16K: GPU ms/token | 16K: tok/s |
|---|---:|---:|---:|---:|
| f16 | 18,94 | 48,7 | 36,08 | 27,1 |
| Q8 | 19,19 | 49,4 | 35,64 | 27,6 |
| TQ4 | 20,60 | 46,3 | 36,38 | 26,4 |

**TQ4 no es más rápido.** A 2K es un 7 % más lento que Q8 y a 16K queda a la par (+2 %). La caché
se lee a la mitad de bytes, pero la lectura de la caché no es el cuello de botella del decode
(los GEMV de pesos ocupan la mayor parte), y la decodificación por nibble con la tabla
`TQ_CB` más los dos kernels de rotación por capa cuestan más de lo que se ahorra. Las
optimizaciones posibles (fusionar la rotación en otros kernels, decodificar con menos
accesos) no se probaron.

Prefill de 16K con `profile_prefill -- 16000 1024` (chunk 1024, BRASA_TUNING=off, GPU):

| KV | pared | tok/s | GPU total | GPU 12K..16K |
|---|---:|---:|---:|---:|
| Q8 | 61,47 s | 260,3 | 60,54 s | 20,15 s |
| TQ4 | 70,87 s | 225,8 | 70,73 s | 22,29 s |

El prefill con TQ4 es un 17 % más lento. La atención de prefill decodifica cada elemento de K y V
en `load_kv` (la misma ruta por elemento de Q8), y la rotación de Q y de la salida suma dos
pasadas por capa.

## Consecuencias

- `tq4` queda como opción explícita. Ningún perfil lo usa por defecto: 8 GB sigue con Q8 y 16 GB
  con f16.
- Cambiar `ALL` de fuentes de `kernels` invalida la base de tuning de ADR 0026: hay que volver a
  correr `brasa tune` en la M1 Pro antes de usar los valores tuneados de Q8 o f16 otra vez.
- Un codebook por tamaño de dimensión: el código asume `head_dim = 128` (Qwen3). Otra dimensión
  requiere otro codebook y otra rotación.
- La etapa QJL y los canales atípicos quedan fuera. Si se quieren, es una ADR nueva con su
  medición.

## Alternativas consideradas

- **TQ5 o TQ6** (más bits): el error relativo baja a ~0,25 % y ~0,06 %, pero el ahorro frente a Q8
  se reduce al 40 % y 28 %. No se implementó hasta medir si la calidad compensa.
- **Mezcla K en Q8 y V en TQ4:** el ruido de K es el que se amplifica en el softmax, así que
  quizá conserve calidad. El ahorro sería solo 25 % frente a Q8. No se probó.
- **Bloques con escala de 32 valores** (como Q8, pero de 4 bits): mejor error por bit que
  Lloyd-Max con una norma por fila, y es lo que ya hace Q4_0 de llama.cpp.
