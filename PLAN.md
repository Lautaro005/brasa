# PLAN.md — Brasa

Estado: fases 0 y 1 desglosadas; el resto está a nivel de objetivo. Cada tarea termina con un comando que demuestra el criterio.

## Qué cambia por usar agentes con herramientas

- **Prefix cache y tool calling suben de prioridad:** pasan de la fase 3 a la fase 2, porque sin ellos Codex y Claude Code no son usables.
- **Contexto:** el perfil de agente apunta a 16K con KV Q8 (en Qwen3-4B, 1,20 GiB de KV; ver tabla); 2K queda solo para tests. A confirmar con mediciones.
- **Calidad de herramientas:** Qwen3-4B es chico para agentes complejos. Sirve para validar el engine; el valor se mide en latencia, memoria y estabilidad, no en que reemplace a un modelo grande.

## KV cache de Qwen3-4B

Calculado desde el `config.json` real (`Qwen/Qwen3-4B` @ `1cfa9a72`): 36 capas, 8 cabezas KV,
`head_dim` 128, 32 cabezas de query (GQA 4:1). Por token se guardan K y V:
2 × 36 × 8 × 128 = 73 728 elementos.

- FP16: 147 456 B/token (144 KiB).
- Q8 por bloques de 32 con escala FP16 (34 B cada 32 elementos): 78 336 B/token (76,5 KiB).

| Contexto | KV FP16 | KV Q8 (g32) |
|---:|---:|---:|
| 2 048 | 288 MiB | 153 MiB |
| 8 192 | 1,13 GiB | 612 MiB |
| 16 384 | 2,25 GiB | 1,20 GiB |
| 32 768 | 4,50 GiB | 2,39 GiB |
| 40 960 (máx. nativo) | 5,63 GiB | 2,99 GiB |

Cálculo, no medición: no incluye workspace de atención ni padding de alineación. El planner (T1.8)
usa estas cifras como piso.

## Fase 0 — Base y contrato

**T0.1 Workspace.** Crear el workspace de Rust con los crates vacíos de `CLAUDE.md`, CI local (`fmt`, `clippy`, `test`).
Aceptación: `cargo test --workspace` pasa en limpio.

**T0.2 `doctor`.** Detectar chip, GPU cores, RAM física, versión de macOS, familia Metal y presión de memoria actual.
Aceptación: `brasa doctor` imprime los datos correctos en la M1 Pro 16 GB y en la M2 8 GB; salida también en `--json`.

**T0.3 Telemetría de memoria.** Medir memoria residente del proceso y presión del sistema durante una ejecución.
Aceptación: un test arranca un buffer de tamaño conocido y el reporte lo refleja dentro de un margen.

**T0.4 Harness de benchmark.** Formato de reporte (chip, RAM, macOS, commit, modelo, cuantización, contexto, TTFT, prefill tok/s, decode tok/s, pico de memoria).
Aceptación: `brasa benchmark --baseline llama.cpp` ejecuta Qwen3-4B Q4 con el mismo prompt y guarda un reporte JSON en `docs/bench/`.

**T0.5 Baselines.** Medir llama.cpp y MLX-LM en las dos Macs con contextos 2K, 8K y 16K.
Aceptación: tabla de referencia en `docs/bench/baseline.md`. Este es el número a superar.

**T0.6 Fixtures.** Script en `tools/` que, con un runtime de referencia FP16, guarda tokens y logits de Qwen3-4B para un set fijo de prompts.
Aceptación: `fixtures/qwen3-4b/` contiene prompts, tokens y logits con hash registrado.

## Fase 1 — Núcleo correcto

**T1.1 Runtime Metal mínimo.** Dispositivo, buffers compartidos, command queue, compilación de MSL desde fuente y cache de pipelines.
Aceptación: un kernel de suma de vectores corre y se compara con CPU.

**T1.2 Tokenizer y chat template de Qwen3.** BPE y plantilla, incluyendo formato de herramientas.
Aceptación: tokens idénticos a la referencia en todos los prompts de `fixtures/`.

**T1.3 Formato nativo Q4.** Empaquetado Q4 por grupos de 32, escalas junto al bloque, alineado a página, hash verificable. Conversor en `tools/` desde safetensors.
Aceptación: desempaquetar y comparar contra los pesos originales dentro de la tolerancia de cuantización documentada.

**T1.4 Kernels básicos con referencia CPU.** RMSNorm, matmul Q4 (GEMV y GEMM simples), RoPE, softmax, activación SwiGLU, embedding.
Aceptación: cada kernel con test de equivalencia y microbenchmark; sin optimizar todavía.

**T1.5 Atención con GQA y QK-norm.** Versión simple correcta, sin tiling.
Aceptación: salida de una capa igual a la referencia dentro de tolerancia.

**T1.6 Forward pass completo y KV cache.** Preasignada, contexto fijo por perfil.
Aceptación: logits finales dentro de tolerancia en todos los fixtures; teacher forcing coincide.

**T1.7 Sampling y `run`.** Greedy, temperatura, top-p, seed.
Aceptación: `brasa run qwen3-4b-q4` genera texto coherente en la M1 Pro y en la M2 8 GB sin swap creciente.

**T1.8 Planner de memoria v0.** Calcula pesos + KV + workspace + margen y rechaza lo que no entra.
Aceptación: en la M2 8 GB rechaza un contexto que no cabe con un mensaje claro y acepta uno que sí.

Compuerta de la fase 1: calidad correcta en las dos Macs. Todavía no se exige velocidad.

## Adelanto de la fase 3 (aprobado el 2026-10-05)

Con los kernels de la fase 1 el prefill rinde ~20 tok/s: un prompt de agente de 15K–30K tokens
tardaría minutos en el primer turno. Antes de la fase 2 se adelantan de la fase 3 el **GEMM tiled
de prefill**, la **atención tiled** (prefill y decode) y la medición de Brasa en el harness de
benchmark (`brasa benchmark` sin `--baseline`). Mismas reglas: referencia CPU, equivalencia,
microbenchmark, y el test de punta a punta de T1.6 debe seguir pasando.

## Fase 2 — API para agentes

- `serve` con `/v1/chat/completions`, `/v1/responses` y `/v1/messages`, streaming, cancelación y `/v1/models` con contexto real.
- Tool calling con parseo del template de Qwen3 y JSON válido.
- Prefix cache de KV entre turnos.
- `connect <herramienta>` genera la configuración para Codex, Claude Code, Cline y OpenCode.
- Suite `conformance`.
- Aceptación: Codex y Claude Code completan una tarea real contra el daemon local; el segundo turno con el mismo prefijo reduce el TTFT de forma medible.

## Fase 3 — Velocidad

GEMV de decode y GEMM de prefill especializados, atención tiled estilo FlashAttention, fusión validada, KV Q8, prefill por chunks.
Aceptación: igualar o superar los baselines de T0.5 en cada contexto, sin salir de tolerancia de calidad.

Desglose (aprobado el 2026-10-05). Punto de partida medido en M1 Pro con `cargo bench -p brasa-kernels`:
GEMM tiled 2,65 TFLOPS, `flash_attention` ~0,7 TFLOPS, `decode_attention_gqa` a 16K 1,65 ms/capa
(81 GB/s). Prioridad de uso: Claude Code (Codex queda para después).

- **T3.1 KV cache f16** (ADR 0009). `KvType` en `Limits`, `store_kv` y variantes f16 de las tres
  atenciones. Aceptación: T1.6 pasa con KV f32 (sin cambios) y con KV f16 contra
  `fixtures/qwen3-4b-q4-kvf16` (teacher forcing exacto; logits ≤ 1e-3, ver ADR 0009); coincidencia de top-1 contra la referencia sin
  redondear ≥ 98 %; atención de decode con KV f16 a 16K ≤ 0,9 ms; el planner coincide con lo reservado.
- **T3.2 Atención de prefill.** FlashAttention que comparte K/V entre las cabezas de un grupo GQA y
  lee la caché f16. Aceptación: equivalencia contra la referencia CPU, `flash_attention` T=512 a 16K
  ≥ 3× la variante actual, T1.6 sigue pasando.
  Estado (2026-10-05): 1,47× en f32 (ADR 0010). Q/P en f16 se probó y no acelera (+1 %), revertido.
  Criterio no cumplido; se sigue con T3.3 y se vuelve con perfilado de GPU.
- **T3.3 GEMM de prefill.** Aceptación: ≥ 3,5 TFLOPS en las tres formas de Qwen3-4B con T=512 y
  T1.6 sigue pasando.
  Estado (2026-10-05): 2,82 TFLOPS en f32 (+8–10 %, ADR 0011). El techo del bucle sin cargas es
  3,58; llama.cpp llega a ~3,5 con pesos y activaciones en f16. Criterio no cumplido: pasar las
  activaciones a f16 es decisión del usuario.
- **T3.4 KV Q8** (perfil de agente 16K). Aceptación: fixtures `kvq8`, T1.6 con su referencia,
  pérdida de top-1 medida y documentada, KV de 16K ≤ 1,25 GiB.
  Estado (2026-10-05): hecha en M1 Pro (ADR 0009). T1.6 pasa con tolerancia 2e-2 según el piso
  medido; pérdida de top-1 98,38 %; KV de 16K 1,195 GiB. Falta medirla en la M2 8 GB.
- **T3.5 Decode.** Fusiones y GEMV según perfil (`profile_decode`). Aceptación: decode a 2K ≥ 50,8 tok/s
  (llama.cpp).
  Estado (2026-10-05), en curso. 42,8 tok/s a 2K: 22,7 ms de GPU + 0,66 ms de CPU por token.
  Desglose medido con `decode_breakdown`:
  - GEMV de pesos: 15,0 ms, a 145–174 GB/s, cerca del ancho de banda.
  - Atención: 4,5–4,8 ms, con ~15 µs fijos por capa.
  - `rms_norm` de H: 0,9–1,5 ms.
  - Ops chicas: ~1 ms.

  Probado sin éxito:
  - tramos de atención de 32 o 64 (peor a 2K y a 16K);
  - RMSNorm de 1024 hilos (peor);
  - RMSNorm fusionada al GEMV: gate/up 5,8 → 8,0 ms, por las lecturas extra;
  - GEMV con lecturas float4/ushort: 5,8 → 6,4 ms.

  Sin cambiar precisión, lo que queda suma ~1,5 ms: fusiones de ops chicas, codificar el token
  siguiente durante la GPU y el costo fijo de la atención. No alcanza; 50,8 pide además bajar el
  lm_head de q8_0 (413 MB por token, 2,4 ms) a menos bits. Es decisión del usuario (ADR 0006).

  Hecho después, con resultados idénticos a los de antes:
  - la GPU arranca con las dos primeras capas mientras se codifican las demás (CPU por token
    ~0,68 → ~0,50 ms);
  - `qk_norm_rope_store` reemplaza seis dispatches por capa y da los mismos bits (A/B en 8 corridas
    alternadas: GPU mínima 22,08 → 21,57 ms).

  Probado sin éxito: RMSNorm con los valores en registros (T=1 igual; T=512, 0,059 → 0,142 ms).
  Las cifras absolutas se vuelven a medir con la máquina liviana en T3.6: con la máquina ocupada
  (otros procesos usando la GPU) el decode a 2K oscila entre 40,6 y 43,7 tok/s.

  Tabla de embeddings en q6_0 (ADR 0012), decisión delegada por el usuario y tomada con datos:
  - top-1 contra FP32: 88,34 % (q8_0: 88,58 %);
  - lm_head: 2,35 → 1,94 ms;
  - A/B del modelo, GPU por token a 2K: 21,44–21,91 → 21,13–21,57 ms.

  Se regeneraron las fixtures q4, kvf16 y kvq8, y T1.5, T1.6, T1.7 y T1.8 pasan. Con lo seguro ya
  hecho, el decode a 2K queda en ~21,1 ms de GPU + ~0,5 ms de CPU (≈ 46 tok/s en la máquina
  liviana, a confirmar en T3.6).

  Después, con los mismos bits salvo donde se indica:
  - RMSNorm de decode sin dispatch propio (`add_norm_prep` + `gemv_scaled`; numérica f32
    equivalente, T1.6 igual): GPU 21,15 → 20,79 ms;
  - q/k/v en un dispatch y gate/up/SwiGLU en otro: 20,97 → 20,48 ms.

  Mejor corrida A/B a 2K: 47,2 tok/s (pared 21,2 ms). Faltan ~1,5 ms para 50,8.

  Atención de decode (`attn_decode_lanes`): tenía ~60 µs fijos por capa porque cada lane leía K
  y V de a una fila y pagaba la latencia de memoria en serie (`attn_decode_sweep`: 93 µs con 128
  claves). Con las lecturas de K y de V agrupadas de a 4 antes de los productos, mismos bits
  (T1.6 idéntico en f32, f16 y q8_0):
  - por capa, f16: 2K 148 → 103 µs, 16K 777 → 564 µs; q8_0: 2K 165 → 110 µs, 16K 783 → 590 µs;
  - A/B del modelo, 4 rondas alternadas: a 2K (f16) 47,1 → 50,5 tok/s (pared 21,24 → 19,82 ms);
    a 16K (q8_0) 22,9 → 26,1 tok/s.
  Quedan ~0,1 ms para 50,8 a 2K; la cifra final se mide en T3.6 con la máquina liviana.
- **T3.6 Cierre.** `brasa benchmark` válido a 2K, 8K y 16K contra los baselines de T0.5, demo de
  Claude Code repetida y tabla en docs/bench/baseline.md.

## Fase 4 — Autotuning y memoria

Autotuner con fingerprint, base de tuning, perfiles 8 y 16 GB, Model Manager por API (`load`, `idle`, `pause`, `resume`, `stop`).
Aceptación: quick tune menor a 1 minuto; en la M2 8 GB presión estable con el contexto declarado.

## Fase 5 en adelante

GUI web, segunda familia (Llama 3.2 3B), luego speculative decoding, Qwen3.5 (Gated DeltaNet) y MoE. Cada una entra solo si mejora una medición frente a la ruta base.

## Decisiones abiertas

- ~~Confirmar `config.json` real de Qwen3-4B y recalcular la tabla de KV.~~ Hecho (ver tabla de KV).
- Verificar los formatos de API de Codex y Claude Code vigentes antes de la fase 2.
- Licencia del repo: Apache-2.0 propuesta.
