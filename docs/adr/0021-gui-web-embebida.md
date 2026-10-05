# ADR 0021 — GUI web embebida

Estado: aceptada (U2)

## Contexto

`brasa serve` carga el modelo y expone la API para agentes. Falta una interfaz para mirar el
estado, el catálogo de benchmarks, planificar la memoria y chatear sin herramientas externas.
La regla 3 de CLAUDE.md prohíbe telemetría saliente: la GUI no puede cargar nada de internet
(CDN, fuentes remotas, analytics).

## Decisión

**Assets embebidos.** `crates/daemon/assets/{index.html,app.css,app.js}` se incluyen con
`include_str!` y se sirven en `/ui` (y `/ui/app.css`, `/ui/app.js`). No hay framework, ni `npm`,
ni paso de build: un archivo HTML, uno CSS y uno JS vanilla. Un test del daemon pide cada asset y
verifica 200 con su `content-type` y que el cuerpo no contenga `http://` ni `https://` (el único
namespace SVG usado se evita armando el markup del gráfico por `innerHTML`, que el parser crea en
el namespace correcto).

**Endpoints de datos** (todos locales, JSON):

- `/api/status` y `/api/metrics` (U1) para la pantalla Estado, refrescada cada 2 s.
- `/api/plan?ctx=&kv=&perfil=&chunk=` reutiliza el planner como `brasa plan`, sin cargar el
  modelo (solo lee el encabezado del `.brasa`).
- `/api/bench` lista y resume los reportes JSON de `docs/bench/` (o `$BRASA_BENCH_DIR`); los
  archivos ilegibles o marcados `valid: false` se informan con su motivo.
- `/api/agents?ctx=` devuelve el mismo texto que `brasa connect` para cada herramienta. La
  generación vive en `brasa_daemon::connect`, una sola fuente para el CLI y la GUI (antes estaba
  duplicada en `crates/cli`).

**Chat.** Streaming por `/v1/chat/completions` con `fetch` + lectura manual del cuerpo SSE
(`EventSource` no admite POST). El razonamiento llega en `reasoning_content` y se muestra
plegable; se desactiva mandando `reasoning_effort: none` (ADR 0008). El botón de cancelar aborta
la conexión y el daemon cancela al fallar el envío. El historial se guarda en `localStorage`.

**Tema.** Claro y oscuro con `prefers-color-scheme`; la disposición es fluida y usable a 1280 px
y en ancho de teléfono.

## Consecuencias

- El binario crece con los assets (unas decenas de KB), sin dependencias nuevas.
- La GUI comparte exactamente los mismos endpoints que los agentes; no hay un camino privilegiado.
- Sin capturas verificadas en esta sesión: la prueba contra un `serve` real a 2K queda pendiente
  (ver `docs/gui/README.md` y "Estado" de `QWEN-DEEPSEEK.md`).
