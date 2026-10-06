# GUI web embebida (U2)

`brasa serve` sirve un dashboard en `/ui`. HTML, CSS y JS viven en `crates/daemon/assets/` y se
embeben con `include_str!`: no hay `npm`, paso de build ni pedidos a internet (regla 3 de
CLAUDE.md). Un test del daemon verifica que cada asset responda 200 con su content-type y que
ningún asset contenga `http://` ni `https://`.

## Cómo abrirla

```bash
brasa serve qwen3-4b-q4 --ctx 16384
# en el navegador:
#   http://127.0.0.1:8080/ui
```

## Pantallas

1. **Chat.** Streaming por `/v1/chat/completions`; el razonamiento (`<think>`) se muestra plegable
   y se activa/desactiva con el switch (manda `reasoning_effort: none` para desactivarlo). Hay
   parámetros de muestreo (`max_tokens`, `temperature`, `top_p`, `top_k`, `seed`), botón de
   cancelar que corta la conexión (el daemon cancela al desconectar) e historial en `localStorage`.
2. **Estado.** Lo de U1 (`/api/status` y `/api/metrics`) refrescado cada 2 s: barra de memoria
   (plan contra presupuesto), prefix cache, TTFT, tok/s de decode y cola.
3. **Benchmarks.** Tabla y gráficos SVG propios (sin librerías) desde `GET /api/bench`, que lista
   los reportes de `docs/bench/`. Los inválidos se marcan en rojo y con `invalid_reasons`.
4. **Plan.** Formulario de contexto, tipo de KV y perfil (esta máquina / 16 GB / 8 GB). Llama a
   `GET /api/plan`, que reutiliza el planner como `brasa plan`, sin cargar el modelo.
5. **Agentes.** Muestra lo que imprime `brasa connect` para cada agente (misma fuente en
   `brasa_daemon::connect`), con botón de copiar. `GET /api/agents`.

Tema claro y oscuro vía `prefers-color-scheme`, usable a 1280 px y en ancho de teléfono.

## Capturas

Pendientes: se sacan con un `serve` real a 2K (dos turnos de chat y una cancelación). En esta
sesión no se pudieron tomar porque la verificación con `brasa` quedó fuera de alcance por pedido
del usuario; quedan anotadas en "Estado" de `QWEN-DEEPSEEK.md`.

## Revisión 2 (R2.5): qué mirar a mano

Estos tres arreglos son de `app.js` y no tienen test automatizado (el daemon cubre las respuestas
de `/api/plan`, `/api/agents` y el filtrado de `/api/bench`). Con un `serve` real a 2K:

- **Errores con mensaje del daemon.** En Plan, pedí un `ctx` inválido (por ejemplo `0` o vacío):
  la salida muestra el mensaje del daemon (`ctx 0 fuera de rango…`), no un genérico `HTTP 400`.
  Igual en Agentes con el campo de contexto vacío.
- **Agentes arranca con el ctx real.** Al abrir la pestaña, el campo de contexto ya trae el
  `context.ctx` del servidor (no 16384 fijo). Si se borra el campo y se recarga, no se manda
  `ctx=` y la petición no da 400.
- **Un error a mitad del stream.** Con una respuesta cortada, el error se muestra una sola vez
  (nota en el mensaje del asistente), no repetido como mensaje de error aparte.

