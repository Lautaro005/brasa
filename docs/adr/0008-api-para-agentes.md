# ADR 0008 — API local para agentes (fase 2)

Estado: aceptada (fase 2)

## Contexto

Formatos verificados el 2026-10-05 contra fuentes vigentes, no de memoria:

- **Anthropic Messages**: documentación de streaming (platform.claude.com) y SDK oficial
  `anthropic` 1.11.0. Eventos `message_start`, `content_block_start/delta/stop` (`text_delta`,
  `input_json_delta`, `thinking_delta`, `signature_delta`), `message_delta` (con `stop_reason` y
  `usage` acumulado), `message_stop`, `ping`, `error`.
- **OpenAI Chat Completions y Responses**: tipos del SDK oficial `openai` 3.24.0
  (`ChatCompletionChunk`, `ResponseStreamEvent`). La documentación web de OpenAI rechaza el
  acceso automatizado; los tipos del SDK son la definición que usan los clientes.
- **Pedidos reales capturados** con un servidor que solo registra:
  - Codex CLI 0.153.4 usa **solo** `POST /v1/responses` (`wire_api = "responses"`; el valor
    `chat` ya no figura en la configuración de ejemplo vigente). `stream: true`,
    `instructions` de ~17 KB, `input` con mensajes `developer`/`user` (`input_text`), y más
    adelante `function_call` / `function_call_output` / `reasoning`. `tools` mezcla `function`,
    `namespace` (herramientas anidadas de subagentes) y `web_search`. ~10K tokens en total.
  - Claude Code 2.1.274 usa `POST /v1/messages?beta=true` con `stream: true`,
    `thinking: {type: "adaptive"}`, `context_management`, `output_config`, mensajes `system`
    intercalados y 27 herramientas (~110 KB): ~30K tokens. `HEAD /api/hello` al iniciar.

## Decisión

**Arquitectura.** `crates/daemon` con axum + tokio. El modelo vive en un hilo propio (el
contexto Metal no es `Sync`); los handlers HTTP le mandan trabajos por un canal y reciben
eventos por otro. Un trabajo a la vez (cola FIFO); el prefix cache de la sesión reutiliza el
prefijo común con el pedido anterior, que es el caso de un agente en una conversación.
Cancelación: si el cliente corta, el envío de eventos falla y la generación se detiene.

**Tipos propios** (regla 8, `crates/core::chat`): `ChatRequest` (mensajes con texto, llamadas y
resultados de herramientas; herramientas con nombre, descripción y esquema JSON; límites;
sampling; razonamiento sí/no) y `ChatEvent` (texto, razonamiento, llamada a herramienta, fin con
motivo y uso de tokens). Cada API se traduce a estos tipos en `crates/daemon` y vuelta.

**Salida de Qwen3.** Un parser incremental separa `<think>…</think>`, texto y bloques
`<tool_call>{json}</tool_call>`. Una llamada se emite cuando su bloque se cierra y su JSON es
válido; si no lo es, se devuelve como texto (el agente ve el error en lugar de una llamada
corrupta).

**Razonamiento.** Desactivado salvo que el cliente lo pida (Anthropic `thinking` distinto de
`disabled`; OpenAI `reasoning_effort` o Responses `reasoning.effort` distinto de `none`/ausente
en chat). Para agentes, pensar en cada turno multiplica la latencia de un modelo de 4B. Cuando
está activo: Anthropic lo emite como bloque `thinking` (firma vacía), Responses como item
`reasoning`, Chat Completions en `reasoning_content` (convención de vLLM).

**Herramientas.** Se traducen `function` (OpenAI/Responses) y las de Anthropic con
`input_schema`. `namespace` se aplana (`namespace.tool`); `web_search` y otros tipos sin
equivalente se omiten y se registran.

**Contexto real.** `GET /v1/models` (formato OpenAI, o Anthropic si llega `anthropic-version`)
informa el contexto del planner de memoria, no el nominal. Un pedido que no entra se rechaza con
error del cliente (`context_length_exceeded` / `invalid_request_error`) en vez de truncar.

**Endpoints.** `/v1/chat/completions`, `/v1/responses`, `/v1/messages`,
`/v1/messages/count_tokens`, `/v1/models`, `HEAD|GET /` y `/api/hello` (salud). Con y sin
streaming. Errores en el formato de cada API.

**Integración.** `brasa connect <codex|claude-code|cline|opencode>` imprime la configuración
(no la escribe en archivos del usuario sin `--escribir`). Para Claude Code el prompt completo
supera el contexto de este modelo: la configuración sugerida limita las herramientas
(`--tools`) y fija `CLAUDE_CODE_MAX_CONTEXT_TOKENS` al contexto real.

**Conformance.** `tools/conformance/` (Python offline, SDKs oficiales) contra el daemon local:
texto, streaming, herramientas, resultados de herramientas, errores y cancelación, en las tres
APIs.

## Consecuencias

- Un solo trabajo a la vez: suficiente para un agente; varios clientes concurrentes esperan.
- Si Codex o Claude Code cambian de formato, la suite de conformance y los pedidos capturados
  son el punto de partida para actualizar la traducción.
