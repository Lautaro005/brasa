# ADR 0005 — Tokenizer BPE y chat template

Estado: aceptada (T1.2)

## Contexto

Qwen3 usa el tokenizer de Qwen2 (`tokenizer.json`, verificado en `Qwen/Qwen3-4B@1cfa9a72`):
normalización NFC, pre-tokenización por regex estilo GPT-4 con lookahead (`\s+(?!\S)`),
BPE byte-level sin `byte_fallback` (151 643 entradas de vocabulario, 151 387 merges) y 26 tokens
agregados (`<|im_start|>`, `<tool_call>`, `<think>`, ...) que se reconocen literalmente antes de
pre-tokenizar. El chat template es Jinja (en `tokenizer_config.json`) y define el formato de
herramientas: definiciones en `<tools>` como JSON por línea, llamadas en `<tool_call>` y
resultados en `<tool_response>` dentro de un turno `user`.

## Decisión

- **BPE propio** en `crates/tokenizer`, cargado desde `tokenizer.json`. Sin depender del crate
  `tokenizers` de HF: el algoritmo es corto y queda bajo nuestro control (y testeado contra las
  fixtures).
- **Pre-tokenización** con `fancy-regex` y el patrón exacto de `tokenizer.json` (el crate `regex`
  no soporta lookahead). NFC con `unicode-normalization`.
- **Chat template** ejecutado con `minijinja` 2.x a partir del template original, no reescrito a
  mano: así un cambio de template o una segunda familia de modelos no requiere código nuevo.
  Compatibilidad con Python vía `minijinja-contrib` (métodos `startswith`, `split`, `strip`...) y
  un filtro `tojson` propio que reproduce `json.dumps(x, ensure_ascii=False)` como hace
  transformers (separadores `", "` y `": "`, orden de claves preservado). El `tojson` de minijinja
  es compacto y cambiaría los tokens del bloque `<tools>`.
- **Decodificación incremental**: `StreamDecoder` retiene bytes UTF-8 incompletos entre tokens,
  para el streaming de la API.
- `tokenizer.json`, `tokenizer_config.json` y la licencia de Qwen se copian a
  `fixtures/qwen3-4b/tokenizer/` (11 MB) para que los tests no dependan de `models/`.

## Criterio

Para los 10 prompts de `fixtures/qwen3-4b/`: el template en Rust renderiza exactamente
`prompt.txt` a partir de `tools/fixture_prompts.json`, y la tokenización produce exactamente
`tokens.i32`. Además, decodificar los tokens devuelve el texto original.
