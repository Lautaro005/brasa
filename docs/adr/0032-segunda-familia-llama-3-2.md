# ADR 0032 — Segunda familia de modelos: Llama 3.2 3B

Estado: propuesta (2026-10-09). Solo investigación y diseño: no hay código, ni conversión, ni
medición en GPU. Las decisiones abiertas están marcadas como "Decisión abierta" y no se cierran
acá.

## Contexto

Brasa soporta una sola familia, Qwen3 (`family = "qwen3"`; `Config::from_json`, `model_shape` y
`tools/convert_brasa.py` la exigen). Esta propuesta evalúa Llama 3.2 3B como segunda: más chico que
Qwen3-4B, GQA, pesos atados y `head_dim` 128, que es lo que asumen los kernels de atención. El
criterio de éxito sería el mismo de la primera familia: equivalencia numérica contra una referencia
(regla 2 y 3 de CLAUDE.md), planner correcto (regla 5) y servir agentes con herramientas.

## Fuentes y revisiones

Todo número de este ADR sale de una de estas fuentes. Lo que no, está marcado "derivado" (cuenta
propia, con la fórmula) o "no verificado".

| Fuente | Revisión | Cómo se leyó |
|---|---|---|
| `meta-llama/Llama-3.2-3B` (`config.json`, `generation_config.json`, `special_tokens_map.json`, `tokenizer_config.json`, `model.safetensors.index.json`, `LICENSE.txt`, `README.md`) | `13afe5124825b4f3751f836b40dafda64c1ed062` (última modificación 2024-10-24) | conector de Hugging Face autenticado (cuenta con la licencia aceptada), rama `main`, 2026-10-09 |
| `meta-llama/Llama-3.2-3B-Instruct` (`config.json`, `generation_config.json`, `tokenizer_config.json` con el `chat_template`, `tokenizer.json`, `README.md`) | `0cb88a4f764b7a12671c53f0838cd831a0843b95` (2024-10-24) | idem |
| `huggingface/transformers`, `src/transformers/modeling_rope_utils.py` | tag `v4.45.0` = commit `2ef31dec1676249d26044a8aa8abe33dbecf0d10` (es la versión que figura en el `config.json`: `4.45.0.dev0`) | `raw.githubusercontent.com`, público |
| `Qwen/Qwen3-4B` (`config.json`, para comparar) | `1cfa9a7208912126459214e8b04321603b3df60c` | copia local en `models/qwen3-4b-hf/` |
| `lautiss/brasa-v0.01-base` (`README.md`, `LICENSE`) | rama `main`, 2026-10-09 | conector de Hugging Face |

Limitaciones de la verificación:

- Los dos repos de Meta son gated (acceso manual: la API pública informa `gated: manual`). Un
  `WebFetch` anónimo a `https://huggingface.co/meta-llama/Llama-3.2-3B/blob/main/config.json`
  devolvió HTTP 401. Los archivos se leyeron con el conector autenticado de Hugging Face. Los hashes
  son los de la cabeza de `main` que informa la API pública en el mismo momento; el conector lee
  `main` y no permite fijar la revisión, así que la coincidencia entre ese hash y los bytes leídos
  se apoya en que ambos tienen la misma fecha de última modificación (2024-10-24).
- La documentación de formatos de prompt de Meta (`llama.com/docs/model-cards-and-prompt-formats/
  llama3_2/`) redirige dos veces a hosts distintos (`developer.meta.com`, luego `dev.meta.ai`); no
  se siguió la cadena. El formato de herramientas de este ADR sale del `chat_template` del repo
  oficial, no de esa página. La model card (`README.md`) no describe el formato de herramientas.
- La model card del repo base dice que los modelos cuantizados (QLoRA, SpinQuant) tienen contexto
  de 8k; no se evalúan acá, solo los pesos BF16.

## Qué cambia frente a Qwen3

Valores del `config.json` oficial (idénticos en base e instruct salvo `eos_token_id`), contra
Qwen3-4B:

| | Qwen3-4B | Llama 3.2 3B | Fuente (Llama) |
|---|---:|---:|---|
| `model_type` | `qwen3` | `llama` | `config.json` |
| `num_hidden_layers` | 36 | 28 | `config.json` |
| `hidden_size` | 2560 | 3072 | `config.json` |
| `num_attention_heads` | 32 | 24 | `config.json` |
| `num_key_value_heads` | 8 | 8 | `config.json` |
| `head_dim` | 128 | 128 | `config.json` (explícito) |
| grupo GQA (`heads / kv_heads`) | 4 | 3 | derivado |
| `q_dim` (`heads · head_dim`) | 4096 | 3072 | derivado |
| `intermediate_size` | 9728 | 8192 | `config.json` |
| `vocab_size` | 151 936 | 128 256 | `config.json` |
| `rms_norm_eps` | 1e-6 | 1e-5 | `config.json` |
| `rope_theta` | 1 000 000 | 500 000 | `config.json` |
| `rope_scaling` | `null` | tipo `llama3` (abajo) | `config.json` |
| `max_position_embeddings` | 40 960 | 131 072 | `config.json` |
| `tie_word_embeddings` | `true` | `true` | `config.json` |
| `hidden_act` | `silu` | `silu` | `config.json` |
| `attention_bias`, `mlp_bias` | `false` | `false` | `config.json` |
| QK-norm | sí | no | índice de tensores (abajo) |

La model card confirma GQA y embeddings compartidos ("GQA: Yes", "Shared Embeddings: Yes"), 3,21 B
de parámetros y contexto de 128k para los modelos sin cuantizar.

**Cuenta de parámetros (derivada, coincide con la fuente).** Con las dimensiones de arriba: por
capa `2·q_dim·h + 2·kv_dim·h + 3·ffn·h` = 100 663 296; ×28 capas = 2 818 572 288; más la tabla de
embeddings `128 256 · 3072` = 394 002 432; más normas (`28·2·3072 + 3072` = 175 104) da
3 212 749 824 parámetros. `model.safetensors.index.json` informa `total_size` = 6 425 499 648
bytes, exactamente 2 bytes por parámetro en BF16, y el Hub informa 3212,7 M. Esto confirma las
dimensiones y que no hay otros tensores.

### RoPE con escalado `llama3`

`config.json` (ambos repos):

```json
"rope_theta": 500000.0,
"rope_scaling": {"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
                 "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
```

El algoritmo no está en el repo de Meta; se tomó de `_compute_llama3_parameters` en transformers
v4.45.0 (revisión de la tabla de fuentes). Sobre las frecuencias de RoPE por defecto
`inv_freq[i] = theta^(-2i/dim)`, con `dim = head_dim = 128`, `i = 0..63`:

```text
wavelen          = 2π / inv_freq[i]
low_freq_wavelen = original_max / low_freq_factor     = 8192
high_freq_wavelen= original_max / high_freq_factor    = 2048
wavelen < high_freq_wavelen          -> inv_freq[i]                (sin cambio)
wavelen > low_freq_wavelen           -> inv_freq[i] / factor
si no (intermedio), con
  s = (original_max / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor):
                                     -> (1 - s) · inv_freq[i] / factor + s · inv_freq[i]
```

`attention_factor` es 1,0 (`_compute_default_rope_parameters`), o sea que cos/sin no se reescalan.
Contando con `theta = 500000` y `dim = 128` (cuenta propia, no de la fuente): de los 64 pares, 29
quedan sin cambio, 29 se dividen por 32 y 6 se interpolan.

El escalado es estático: cambia las frecuencias para toda posición, no solo para contextos largos.
Por eso un modelo con RoPE sin escalar no reproduce los logits de la referencia ni a contextos
cortos (relevante para la Decisión abierta 2).

Convención de rotación. `rope_neox_f32` de Brasa (`rope.metal`) es `rotate_half` (mitades
`[0, D/2)` y `[D/2, D)`). Los safetensors de Hugging Face de Llama están pensados para esa misma
convención (transformers usa `rotate_half`); no se verificó contra el repo de Meta y lo tiene que
confirmar la prueba de equivalencia (no es una fuente leída acá).

### Sin QK-norm

`model.safetensors.index.json` lista por capa solo `input_layernorm`, `post_attention_layernorm`,
`self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`, más `model.embed_tokens.weight` y
`model.norm.weight`. No hay `q_norm`/`k_norm` ni `lm_head.weight` (el `lm_head` es la tabla de
embeddings atada, como en Qwen3). Los nombres son los mismos que en Qwen3 menos las dos normas.

### Vocabulario, tokens especiales y tokenizer

- `vocab_size` 128 256: 128 000 tokens de BPE más 256 especiales (`added_tokens` del
  `tokenizer.json`: ids 128000 a 128255).
- Ids relevantes (`tokenizer_config.json` del instruct): 128000 `<|begin_of_text|>`, 128001
  `<|end_of_text|>`, 128004 `<|finetune_right_pad_id|>`, 128006 `<|start_header_id|>`, 128007
  `<|end_header_id|>`, 128008 `<|eom_id|>`, 128009 `<|eot_id|>`, 128010 `<|python_tag|>`.
- `tokenizer.json`: `normalizer: null`; pre-tokenizer `Sequence[Split(regex, Isolated), ByteLevel
  (add_prefix_space=false, use_regex=false)]`; la regex (dígitos de a 1 a 3 con `\p{N}{1,3}`) es
  `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|
  \s*[\r\n]+|\s+(?!\S)|\s+`; el `post_processor` antepone `<|begin_of_text|>`; el modelo BPE trae
  `ignore_merges: true`.
- Fin de generación. Base: `eos_token` = `<|end_of_text|>` (`tokenizer_config.json`, `eos_token_id`
  128001). Instruct: `eos_token` = `<|eot_id|>` en `tokenizer_config.json` y
  `eos_token_id: [128001, 128008, 128009]` en `config.json` y `generation_config.json`.
- `generation_config.json`: `temperature` 0,6 y `top_p` 0,9 (con `do_sample: true`) en los dos.
- El repo base no tiene `chat_template` en `tokenizer_config.json`; el instruct sí.

Frente al código actual (`crates/tokenizer`, solo lectura):

1. `Tokenizer::from_dir` construye `stop_ids` con `eos_token` más `<|endoftext|>`. Para el instruct
   daría solo `<|eot_id|>`; faltan `<|eom_id|>` y `<|end_of_text|>`. El cierre de turno tiene que
   salir de `eos_token_id` de `generation_config.json` o de una lista del manifiesto.
2. `Bpe::from_tokenizer_json` acepta el `Sequence` con `Split` (busca la regex con
   `find_split_regex`), `normalizer: null` y no aplica `post_processor`. El BOS lo emite el chat
   template literalmente (`{{- bos_token }}`) y `next_added` lo reconoce; para `encode` de texto
   plano (completions, base) hay que anteponerlo.
3. `ignore_merges: true` no se implementa (no aparece en `bpe.rs`): en HF una pieza que ya es un
   token del vocabulario se emite directa sin aplicar merges. Puede no cambiar nada, pero no está
   demostrado: lo decide `tools/make_tokenizer_cases.py` contra HF `tokenizers`.
4. `ChatTemplate::from_tokenizer_config` falla si no hay `chat_template` (el caso del base).

### Plantilla de chat y llamadas a herramientas (instruct)

Del `chat_template` de `Llama-3.2-3B-Instruct` (revisión de la tabla de fuentes):

- Formato: `<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n` + `Cutting Knowledge
  Date: December 2023\nToday Date: <fecha>\n\n` + mensaje de sistema + `<|eot_id|>`, y cada turno
  `<|start_header_id|>{rol}<|end_header_id|>\n\n{contenido}<|eot_id|>`. El prompt de generación
  termina en la cabecera de `assistant`.
- La fecha sale de `strftime_now("%d %b %Y")` si el entorno la define; si no, el template cae a
  `"26 Jul 2024"`. El `Environment` de Brasa no registra `strftime_now`, así que hoy usaría la fecha
  fija. Relevante para el prefix cache: una fecha real cambia el prefijo todos los días.
- Con `tools`, el system lleva `Environment: ipython` y, por defecto (`tools_in_user_message`
  verdadero), las definiciones van en el **primer mensaje de usuario** con la instrucción de
  responder con JSON `{"name": nombre, "parameters": diccionario}` ("Do not use variables.").
  Cada herramienta se serializa con `tojson(indent=4)`: Brasa implementa `tojson` sin `indent`
  (`write_py_json`), hay que agregarlo.
- Llamada del asistente: `{"name": "...", "parameters": <arguments|tojson>}` seguida de `<|eot_id|>`.
  El template exige **una sola llamada por mensaje** (`raise_exception("This model only supports
  single tool-calls at once!")`). Un historial de agente con llamadas paralelas haría fallar el
  render.
- Resultado: rol `tool` (o `ipython`) se renderiza con cabecera `ipython`.
- No hay `<think>`, ni `enable_thinking`, ni `reasoning_content`.
- `message['content']|trim` supone contenido string: el daemon tiene que aplanar los arreglos de
  partes de OpenAI/Anthropic antes del render.
- Posibles diferencias minijinja/jinja2 que hay que probar: `message.content is iterable` (en
  Jinja2 un string es iterable, así que el contenido de una herramienta pasaría por `tojson` y
  quedaría entre comillas; en minijinja hay que comprobarlo), `tool_call.arguments | tojson` (con
  `arguments` ya como objeto, no como string JSON de OpenAI), `trim_blocks`/`lstrip_blocks`.
- Salida del modelo: el template no dice cómo la emite; en `eos_token_id` del instruct está
  `<|eom_id|>` (128008), cuyo rol en el protocolo de herramientas de Meta no se verificó acá (ver
  "No verificado"). `crates/runtime/src/qwen_output.rs` entiende `<think>` y
  `<tool_call>{...}</tool_call>`; para Llama hace falta otro parser (JSON `name`/`parameters`,
  posiblemente precedido por `<|python_tag|>`).

La model card declara los modelos de texto del instruct "optimizados para diálogo multilingüe,
incluida la recuperación agéntica y el resumen". No afirma nada sobre calidad en agentes de código;
**no hay ninguna medición de Brasa sobre eso** (regla 6) y es el principal riesgo de producto de
esta propuesta.

## Qué kernels sirven y cuáles faltan

Lectura de `crates/kernels/src/`, `crates/models/src/qwen3.rs` y `tools/convert_brasa.py`, sin
modificar nada.

| Pieza | Estado con Llama 3.2 3B |
|---|---|
| GEMV/GEMM Q4 (`matmul`, `matmul_tiled`), tabla de embeddings q6_0/q8_0, `embed`, `swiglu`, `rms_norm`, `add_norm` | Sirven tal cual. Las dimensiones (3072, 1024, 8192, 128 256) son múltiplos de 32 (bloque de cuantización) y de 128. |
| KV f32/f16/Q8, `store_kv`, formato de fila de 128 valores | Sirven tal cual: `head_dim` = 128 (los kernels fijan 128) y `kv_heads` = 8 igual que Qwen3. |
| `rope_neox_f32` (RoPE `rotate_half`) | Sirve. El escalado `llama3` solo cambia la tabla de frecuencias, que se arma en CPU. |
| Tabla RoPE (`RopeTable::new(ctx, theta, dim, max_pos)`, `rope_table`) | Falta: hoy recibe solo `theta`. Hace falta una variante que aplique el escalado de arriba (cálculo en f64 en CPU, como la actual). Es la referencia CPU. |
| `qk_norm_rope_store` (QK-norm + RoPE + `store_kv` fusionados, T3.5) | No sirve: Llama no tiene QK-norm y normalizar con pesos 1 no es la identidad (RMSNorm normaliza). Opciones en la Decisión abierta 4. |
| Atención de decode (`decode_attention_lanes`, ADR 0009) | **No cubre el grupo GQA 3.** `GQA_GROUPS = [1, 2, 4, 8]` y 24/8 = 3. `gqa_supported` da falso y el modelo cae a `decode_attention` (una cabeza por threadgroup), el kernel que ADR 0009 dejó "para grupos no compilados". Su costo con grupo 3 no está medido; el único dato cercano es que el kernel por grupo anterior a `decode_attention_lanes` tardaba 1,63 ms contra 0,71 ms por capa a 16K con KV f16 en Qwen3 (ADR 0009, M1 Pro). |
| Atención de prefill (`flash_attention`, ADR 0030) | Mismo problema: sin GQA 3 usa la variante sin GQA, solo f32 (`precision()` fuerza `PrefillPrecision::F32` y el GEMM de prefill queda con entrada f32). Pierde las mejoras del ADR 0030. |
| `Config::from_json`, `model_shape`, `weight_types`, `Layer` | Rechazan `model_type != "qwen3"` y exigen `q_norm`/`k_norm`. Hay que abstraer la familia (RoPE, QK-norm opcional). |
| `tools/convert_brasa.py` | Exige `model_type == "qwen3"`; `dtype_for` (1-D en f32, embeddings q6_0, resto q4_0) sirve sin cambios; el repo de origen y el commit hay que parametrizarlos. |
| Referencia de logits | `tools/qwen3_ref.py` y `make_fixtures.py` son de Qwen3: hace falta una referencia Llama (transformers, solo en `tools/`) con el mismo esquema de fixtures (FP32, `q4`, `q4-kvf16`, `q4-kvq8`). |

Sobre el grupo 3: agregar `3` a `GQA_GROUPS` no alcanza sin revisar los kernels. En
`flash_attention.metal` la variante GQA procesa 2 cabezas de query por threadgroup (`FH = 2` si
`GQA_G >= 2`) y calcula `kh = h0 / GQA_G` para el par, lo que con grupo 3 mezclaría cabezas de dos
cabezas KV distintas (la cabeza 2 y la 3 son de las KV 0 y 1). Habría que usar `FH = 1` o `FH = 3`
para G = 3. Hay que verificarlo en el kernel real antes de afirmar nada más. Cualquier kernel nuevo
cumple la regla 2: referencia CPU, test de equivalencia con tolerancia documentada y microbenchmark.

## Impacto en el planner

### KV por token y por contexto

Fórmula del planner (ADR 0007, `kv_bytes`):
`2 · capas · ctx · kv_heads · head_dim · bytes_por_elemento`, con `ctx` alineado a 64 posiciones, el
bloque contado en bytes por 32 elementos (128 en f32, 64 en f16, 34 en Q8, ADR 0009) y cada buffer
(K y V) redondeado a páginas de 16 KiB.

Por token (K y V, todas las capas): Qwen3-4B = 2·36·8·128 = 73 728 elementos; Llama 3.2 3B =
2·28·8·128 = **57 344 elementos**, un 22,2 % menos.

| Por token | Qwen3-4B | Llama 3.2 3B |
|---|---:|---:|
| f32 | 288 KiB | 224 KiB |
| f16 | 144 KiB | 112 KiB |
| Q8 (34 B / 32 elementos) | 76,5 KiB | 59,5 KiB |

KV total con la fórmula exacta del planner (GiB):

| Contexto | Qwen3-4B f32 / f16 / Q8 | Llama 3.2 3B f32 / f16 / Q8 |
|---:|---|---|
| 2 048 | 0,562 / 0,281 / 0,149 | 0,438 / 0,219 / 0,116 |
| 8 192 | 2,250 / 1,125 / 0,598 | 1,750 / 0,875 / 0,465 |
| 16 384 | 4,500 / 2,250 / 1,195 | 3,500 / 1,750 / 0,930 |
| 32 768 | 9,000 / 4,500 / 2,391 | 7,000 / 3,500 / 1,859 |
| 131 072 | 36,000 / 18,000 / 9,562 | 28,000 / 14,000 / 7,438 |

Los números de Qwen3-4B reproducen los de ADR 0009 (288 KiB por token en f32, 4,5 GiB a 16K, 2,25 en
f16, 1,195 en Q8). ADR 0007 no trae una tabla propia: la única tabla de referencia es la de PLAN.md
que cita el test `kv_coincide_con_la_tabla_de_plan_md` del planner (4,5 GiB a 16K en f32), y
PLAN.md no está en este árbol; por eso la comparación se hace contra esos valores y contra ADR 0009.

### Plan completo a 16K (estimado)

Estimación con una réplica en Python de `plan` (pesos + KV + workspace + 256 MiB de overhead), con
`max_tokens = 512` y una fila de logits como en los tests del planner. Con los pesos de Qwen3-4B de
los tests (2 362 232 013 B) la réplica da 3,74 GiB a 16K con Q8; ADR 0009 informa 3,76 GiB, así que
la réplica está dentro de ~1 % y las cifras de Llama valen como estimación, no como medición.

Pesos de Llama (derivados, `q4_0` a 18 B/32 en las lineales, `q6_0` a 26 B/32 en la tabla, normas
f32, sin el relleno de página por tensor): lineales 1 585 446 912 B + tabla 320 126 976 B + normas
700 416 B = 1 906 274 304 B = **1,775 GiB** (con tabla `q8_0`: 1,867 GiB). Para comparar, el
`model.brasa` real de Qwen3-4B-q4 mide 2 362 228 736 B (2,20 GiB, ADR 0031).

| A 16 384 de contexto | Qwen3-4B | Llama 3.2 3B |
|---|---:|---:|
| Pesos | 2,20 GiB | 1,78 GiB (est.) |
| KV Q8 / f16 | 1,195 / 2,250 GiB | 0,930 / 1,750 GiB |
| Workspace | 93 MiB | 84 MiB |
| Plan total, KV Q8 | 3,74 GiB | 3,04 GiB |
| Plan total, KV f16 | 4,79 GiB | 3,86 GiB |

Contexto máximo que entraría (búsqueda en pasos de 256 como `max_context`, presupuestos estimados
de `Profile`: 8 GB = 4,83 GiB, 16 GB = 11,34 GiB; ambos marcados "estimado" en el planner):

| | Qwen3-4B | Llama 3.2 3B |
|---|---:|---:|
| 8 GB, KV Q8 | 31 232 | 47 616 |
| 8 GB, KV f16 | 16 640 | 25 344 |
| 16 GB, KV f16 | 63 744 | 86 016 |
| 16 GB, KV Q8 | 119 552 | ≥ 131 072 (el tope de la búsqueda) |

Son capacidad de memoria, no calidad ni velocidad. El nominal de Llama es 131 072 y el de Qwen3
40 960; `GET /v1/models` seguiría informando el contexto del perfil de memoria, no el nominal.
La pérdida de top-1 de la KV Q8 (98,38 % en ADR 0009) se midió solo en Qwen3; para Llama hay que
volver a medirla antes de usarla como defecto del perfil de 8 GB.

### Cambios en el planner

- `ModelShape` ya sirve (capas, hidden, heads, kv_heads, head_dim, ffn, vocab). El workspace de
  `Qwen3::load` y `workspace_bytes` usan `q_dim = heads · head_dim`; con Llama coincide con `hidden`
  (3072) y con Qwen3 no, así que el test que compara el plan con lo reservado hay que repetirlo con
  Llama.
- `plan` cuenta cada tensor por separado: sin `q_norm`/`k_norm` hay menos tensores.
- La tabla RoPE es `2 · ctx · head_dim/2 · 4 B`, igual que ahora.

## Licencia

Llama 3.2 se distribuye bajo la **Llama 3.2 Community License** (`LICENSE.txt` y metadato
`license: llama3.2` en los dos repos), un acuerdo propio de Meta, "a custom, commercial license
agreement" en sus palabras, no una licencia de software libre/OSI. Qwen3-4B es Apache-2.0
(manifiesto `qwen3-4b-q4.toml`). Este resumen es una lectura del texto del repo, no asesoría legal.

Qué dice el texto (revisión de la tabla de fuentes) y qué implica para Brasa:

| Cláusula | Resumen | Implicación |
|---|---|---|
| 1.a Concesión | licencia no exclusiva, mundial, intransferible, sin regalías, para usar, reproducir, distribuir, copiar, crear obras derivadas y modificar | Convertir y cuantizar está permitido. |
| 1.b.i Redistribución | quien distribuye los Llama Materials, obras derivadas o un producto o servicio que los contenga debe entregar copia del acuerdo y mostrar "Built with Llama" en un sitio, interfaz, blog, página "about" o documentación del producto; si usa los materiales o sus salidas para crear, entrenar o mejorar un modelo de IA que se distribuye, el nombre del modelo debe empezar con "Llama" | Un `.brasa` convertido es, razonablemente, una obra derivada o los materiales mismos: publicarlo en Hugging Face exige el acuerdo junto a los pesos y el "Built with Llama" en el README. El nombre `llama-3.2-3b-...` ya empieza con "Llama". Si la interfaz de Brasa ofrece el modelo, también corresponde mostrarlo ahí. |
| 1.b.iii Aviso | conservar en un archivo "Notice" el texto: "Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved." | Un `NOTICE` por carpeta de modelo publicada. |
| 1.b.iv Uso aceptable | cumplir las leyes y la Política de Uso Aceptable, incorporada por referencia | Lo hereda quien use los pesos. |
| 2 Términos comerciales | quien, al 2024-09-25, tenga más de 700 millones de usuarios activos mensuales debe pedir licencia a Meta | No aplica a Brasa; sí a terceros muy grandes. |
| 5.a Marcas | no hay licencia de marcas salvo el uso de "Llama" exigido por 1.b.i y las guías de marca de Meta | El nombre de modelo y la marca "Brasa" quedan separados. |
| 5.b Obras derivadas | entre Meta y el licenciatario, las obras derivadas hechas por el licenciatario son suyas, salvo lo que Meta ya posee | Los pesos convertidos siguen sujetos al acuerdo. |
| 5.c y 6 Terminación | demandar a Meta por infracción de PI sobre Llama termina las licencias; Meta puede terminar el acuerdo por incumplimiento, y entonces hay que borrar los materiales | Riesgo de continuidad que Apache-2.0 no tiene. |

Implicaciones prácticas para publicar:

- El repo `lautiss/brasa-v0.01-base` declara `license: apache-2.0` en el metadato del README y trae
  un `LICENSE` Apache-2.0 en la raíz. Mezclar ahí pesos de Llama haría falso el metadato y dejaría
  el acuerdo de Meta fuera de la raíz. Ver Decisión abierta 6.
- El acceso a los repos de Meta es manual (gated): quien quiera convertir desde la fuente
  (`brasa pull --desde-fuente`) necesita una cuenta de Hugging Face con la licencia aceptada y un
  token. El camino de `brasa pull` de pesos ya convertidos (ADR 0031) no lo necesitaría, pero
  entonces Brasa pasa a redistribuir los pesos, con las obligaciones de arriba.
- La Política de Uso Aceptable tiene una cláusula adicional sobre modelos multimodales y la Unión
  Europea. La model card describe a 1B y 3B como "text only", por lo que no aplicaría a este modelo;
  no se evaluó el resto de la política.
- `license` del manifiesto es hoy una cadena (`"Apache-2.0"`); la GUI y `brasa models` tendrían que
  mostrar la licencia de cada modelo y, para Llama, la marca "Built with Llama".

## Decisiones abiertas

**Decisión abierta 1: base o instruct.**
- *Instruct* (`Llama-3.2-3B-Instruct`): tiene `chat_template` y formato de herramientas; es lo que
  usa un agente. Cierra en `<|eot_id|>`.
- *Base* (`Llama-3.2-3B`): sin `chat_template` (`Tokenizer::from_dir` falla hoy) y sin protocolo de
  herramientas; sirve para pruebas de logits y para completions.
- Las dos a la vez: el `config.json` es igual salvo `eos_token_id`; los pesos son distintos.

**Decisión abierta 2: escalado `llama3` en el primer corte.**
- *Implementarlo desde el principio*: un cambio solo en la tabla RoPE en CPU (el kernel no cambia);
  permite la comparación con la referencia dentro de tolerancia a cualquier contexto.
- *Diferirlo y limitar a contextos ≤ 8192*: no reproduce los logits de la referencia (el escalado
  cambia las frecuencias en todas las posiciones), así que el criterio de equivalencia de la regla 3
  no se podría verificar contra transformers; haría falta una referencia propia sin escalado.

**Decisión abierta 3: atención con grupo GQA 3.**
- *Usar los caminos genéricos* (`decode_attention`, `flash_attention` sin GQA): no hay kernels
  nuevos, pero el prefill queda en f32 y el decode con una cabeza por threadgroup. No está medido
  cuánto cuesta en Llama.
- *Agregar un grupo 3* a `GQA_GROUPS` y a las variantes de los kernels: con referencia CPU, test de
  equivalencia y microbenchmark (regla 2), y resolviendo el emparejamiento de cabezas descrito
  arriba.

**Decisión abierta 4: RoPE y escritura en caché sin QK-norm.**
- *Camino sin fusionar*: `rope_neox_f32` más `store_kv_*`, que ya existen; sin kernels nuevos, con
  más dispatches por capa que `qk_norm_rope_store` (RoPE de q, RoPE de k y guardado de k y v;
  costo sin medir).
- *Variante fusionada* de `qk_norm_rope_store` sin normas (RoPE + `store_kv`), con las tres cosas de
  la regla 2.

**Decisión abierta 5: herramientas.**
- Alcance del formato: una sola llamada por mensaje del asistente (lo que el template admite),
  contra reescribir el template para admitir llamadas paralelas (se aleja del formato de
  entrenamiento y no hay medición).
- Ubicación de las definiciones: primer mensaje de usuario (defecto del template) contra el system
  (`tools_in_user_message = false`). El segundo mantiene el system como único prefijo estable.
- Cómo se da la fecha del system: `date_string` fija, real, o el valor por defecto del template. Una
  fecha real rompe el prefix cache cada día.
- Parser de salida propio en `crates/runtime` (hoy solo `qwen_output.rs`).

**Decisión abierta 6: distribución de pesos.**
- *Publicar pesos convertidos* en un repo propio para Llama con `license: llama3.2`, `LICENSE`,
  `NOTICE` y "Built with Llama" (y decidir si se deja público o gated).
- *Publicarlos en el repo existente* con una carpeta por modelo y su propia licencia: el metadato
  `license: apache-2.0` del repo pasaría a ser engañoso.
- *No publicar*: solo `brasa pull --desde-fuente` + `brasa convert`, de modo que cada usuario acepta
  la licencia con su cuenta y Brasa no redistribuye.

**Decisión abierta 7: cuantización de la tabla de embeddings.** ADR 0012 eligió `q6_0` midiendo
Qwen3-4B (88,34 % de coincidencia de top-1 contra FP32 frente a 88,58 % con `q8_0`). Para Llama
(vocabulario menor, mismas dimensiones de bloque) hay que repetir `tools/eval_embed_quant.py`
antes de fijar `q6_0` o `q8_0`.

**Decisión abierta 8: validación en la M2 de 8 GB.** Las cifras de arriba son del planner con
presupuesto estimado. Se puede dejar para después de la validación del perfil de 8 GB de Qwen3
(`scripts/validate-8gb.sh`) o exigirla antes de aceptar este ADR.

## Plan de verificación (si se acepta)

Cada paso con criterio verificable por comando, como en el resto del plan:

1. Referencia: `tools/llama_ref.py` y fixtures (`fixtures/llama-3.2-3b*/`) con transformers, FP32 y
   variantes Q4/KV, mismo esquema que ADR 0003 y 0009.
2. Tokenizer: casos borde contra HF `tokenizers` (`ignore_merges`, dígitos `\p{N}{1,3}`, BOS, tokens
   especiales) y render del template con y sin herramientas contra `apply_chat_template`.
3. Tabla RoPE con escalado: prueba contra los `inv_freq` de transformers v4.45.0.
4. Forward: logits contra la referencia con la tolerancia que fije el ADR (error relativo ≤ 1e-4
   con KV f32; con KV redondeada, según ADR 0009), teacher forcing por prefill y decode.
5. Planner: test que compare el plan con lo que el modelo reserva de verdad, para Llama.
6. `decode_alloc`: cero asignaciones en decode (regla 4).
7. Conformance (`tools/conformance/run.py`) con el parser de herramientas de Llama.
8. `brasa benchmark` en M1 Pro 16 GB y, si se exige, en la M2 8 GB: recién ahí se escribe una cifra
   de velocidad o memoria como "medido en <chip> <RAM>".

## No verificado

- El rol de `<|eom_id|>` y de `<|python_tag|>` en el formato de herramientas del 3B: están en
  `eos_token_id` y en la lista de tokens especiales, pero no se leyó la documentación de Meta (la
  página redirige) ni se probó el modelo.
- Que los safetensors de Hugging Face usen la convención `rotate_half` sin permutar (se apoya en
  transformers; lo cubre la prueba de equivalencia).
- Todo el comportamiento de agentes de código con Llama 3.2 3B (calidad, tasa de llamadas válidas,
  velocidad, memoria real): sin mediciones.
- Los tamaños de pesos de Llama son derivados de las dimensiones, sin conversión real ni relleno por
  tensor; los totales del planner son una réplica en Python validada contra Qwen3-4B dentro de ~1 %.
- La interpretación legal de que un `.brasa` es una obra derivada a los efectos del acuerdo de Meta,
  y las obligaciones de gating en Hugging Face.

## Consecuencias si se acepta

- `Config`, `model_shape` y el cargador dejan de asumir Qwen3: familia y RoPE pasan a ser parte de
  la configuración del modelo.
- KV por token 22 % menor y pesos estimados ~19 % menores que Qwen3-4B (1,78 contra 2,20 GiB, ambos
  Q4 con tabla q6_0): más contexto en 8 GB según el planner, sin ninguna afirmación de velocidad.
- Un acuerdo de licencia por modelo que Brasa tiene que mostrar y respetar (manifiesto, GUI,
  README), y un repo o carpeta de pesos con licencia distinta de la del engine.
- Costo de mantenimiento: otro parser de salida de herramientas, otro template y otro juego de
  fixtures, ya que hoy todo el camino de agentes está probado solo con Qwen3.
