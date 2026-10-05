# ADR 0002 — Metodología de benchmark y baselines

Estado: aceptada (T0.4)

## Contexto

T0.4 y T0.5 fijan el número a superar. Para que la comparación sea justa hay que fijar
cuantización, prompt, definición de cada métrica y forma de medir la memoria, y que todo sea igual
para llama.cpp, MLX-LM y Brasa.

## Decisión

**Cuantización comparable.** Brasa usa Q4 por grupos de 32 con escala por bloque. Equivalentes:

- llama.cpp: `Q4_0` (bloques de 32, escala FP16). Se genera desde los safetensors oficiales con
  `convert_hf_to_gguf.py` + `llama-quantize`, porque el repo GGUF oficial de Qwen solo trae
  `Q4_K_M`. `Q4_K_M` se puede medir como referencia extra, marcada como tal.
- MLX-LM: `mlx_lm.convert -q --q-bits 4 --q-group-size 32` (el default de MLX es 64).

**Mismo prompt.** `tools/make_bench_prompts.py` genera, con el tokenizer oficial, un texto de
exactamente `ctx - gen` tokens para cada contexto (2K, 8K, 16K). Es texto plano sin chat
template, para que todos los engines procesen la misma secuencia. `gen` = 128.

**Métricas.**

- Prefill tok/s = tokens del prompt / tiempo de procesar el prompt.
- Decode tok/s = (tokens generados − 1) / tiempo de generarlos, sin el primero.
- TTFT = tiempo desde el prompt ya tokenizado hasta el primer token generado. No incluye carga del
  modelo.
- Pico de memoria: se guardan dos métricas de `/usr/bin/time -l` del proceso del engine.
  - `peak_rss` ("maximum resident set size"): incluye páginas de archivos mapeados. Es la métrica
    principal, porque llama.cpp mapea el GGUF con `mmap` y sus pesos no aparecen en el footprint.
  - `peak_footprint` ("peak memory footprint"): memoria anónima, comprimida y de GPU; es lo que
    mira jetsam. Medido en M1 Pro: llama.cpp 2K da 392 MB de footprint y 2,79 GB de RSS.
  MLX además reporta su pico de GPU (`mx.get_peak_memory`).
- Durante cada corrida se muestrea la memoria del sistema (`MemorySampler`): crecimiento de swap y
  peor nivel de presión. Una corrida con swap creciente se marca como no válida.

**Ejecución.** Decodificación greedy, EOS ignorado (para generar siempre `gen` tokens),
1 corrida de calentamiento descartada + 3 medidas; se reporta la mediana y el rango.
llama.cpp con todas las capas en GPU (`-ngl 99`), flash attention `auto` y KV FP16 por defecto;
las variantes (KV Q8) se reportan como filas aparte.

**Reporte.** JSON en `docs/bench/<chip>-<ram>/`, con chip, RAM, macOS, commit de Brasa, engine y
versión, commit de pesos (HF), sha256 del archivo de pesos, cuantización, contexto y métricas.

## Consecuencias

- Las cifras de baseline dependen del estado de la máquina; el reporte guarda la presión inicial y
  el swap, y el harness avisa si el sistema arranca bajo presión.
- La comparación es por proceso completo (incluye runtime de Python en MLX), documentado en el
  reporte.
