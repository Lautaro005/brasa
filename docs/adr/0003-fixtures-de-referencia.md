# ADR 0003 — Fixtures de referencia de Qwen3-4B

Estado: aceptada (T0.6)

## Contexto

T1.2 (tokenizer), T1.5 (atención) y T1.6 (forward completo) se validan contra tokens y logits de
referencia. PLAN.md pide "un runtime de referencia FP16". En esta máquina (16 GB):

- transformers en FP32 necesita 16 GB solo de pesos: no entra.
- FP16 puro no es una referencia limpia: los pesos de Qwen3 son BF16, convertirlos a FP16 cambia
  valores, y FP16 agrega su propio redondeo en activaciones grandes.
- BF16 en transformers entra (8 GB), pero su error es del mismo orden que el que queremos medir.

## Decisión

- **Referencia = FP32 exacto sobre los pesos BF16 originales.** `tools/qwen3_ref.py` implementa el
  forward de Qwen3 en PyTorch FP32 (RMSNorm, QK-norm, RoPE, GQA, SwiGLU, lm_head atado), leyendo
  cada tensor BF16 por mmap y subiéndolo a FP32 al usarlo, capa por capa. Pico de memoria bajo,
  sin redondeo intermedio de 16 bits.
- **Validación cruzada** contra la implementación oficial de transformers en BF16: para cada
  prompt se registra el error máximo y medio de logits y la coincidencia de top-1. Esto (a) valida
  que la referencia propia es Qwen3 y no otra cosa, y (b) mide el ruido esperable de un runtime de
  16 bits, que fija el piso de tolerancia de Brasa (T1.6).
- **Qué se guarda** en `fixtures/qwen3-4b/`, por prompt:
  - texto renderizado con el chat template oficial (fuente de verdad de T1.2) y tokens;
  - logits FP32 completos de la última posición;
  - top-20 (índices y logits) y logsumexp de todas las posiciones (teacher forcing);
  - continuación greedy de 32 tokens.
  Formato: binarios little-endian (`.f32`, `.i32`) + `manifest.json` con formas y sha256 de cada
  archivo y del conjunto. Sin pickle ni npz: Rust los lee sin dependencias.
- El set de prompts (`tools/fixture_prompts.json`) cubre texto en inglés y español, código,
  Unicode, chat con system, multi-turno con razonamiento previo, `enable_thinking=false`,
  definición de herramientas, llamada a herramienta y respuesta de herramienta, y un prompt largo.

## Consecuencias

- La referencia propia podría tener un error compartido con Brasa si ambos interpretan mal el
  modelo; la validación cruzada con transformers es lo que lo descarta. Si el top-1 no coincide
  casi siempre, la fixture no se acepta.
- Los logits completos de todas las posiciones no se commitean (608 KB por posición); el top-20
  alcanza para teacher forcing y el script puede regenerar todo.
