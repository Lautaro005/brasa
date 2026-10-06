# GUI web embebida

`brasa serve` sirve la GUI en `/ui`. HTML, CSS y JS viven en `crates/daemon/assets/` y se embeben
con `include_str!`: no hay `npm`, paso de build ni pedidos a internet (regla 3 de CLAUDE.md). Un
test del daemon verifica que cada asset responda 200 con su content-type y que ningún asset
contenga `http://` ni `https://`. Los assets se sirven con `Cache-Control: no-cache` porque cambian
con el binario.

## Cómo abrirla

```bash
brasa serve qwen3-4b-q4 --ctx 16384
# en el navegador:
#   http://127.0.0.1:8080/ui
```

## Diseño

Rediseño de 2026-10-06 con el flujo de `/impeccable` (modo Operate). La dirección elegida por el
usuario es **Ceniza y brasa**: la interfaz es ceniza neutra y solo "arde" lo que está trabajando.
La rampa de incandescencia se usa exclusivamente para datos (tokens por segundo) y para el estado
activo (LED y pedidos en curso). Tema claro y oscuro según `prefers-color-scheme`; en el claro la
rampa va de ámbar pálido a rojo oscuro y en el oscuro de rojo oscuro a amarillo, para que más
actividad siempre sea más contraste contra el fondo. Fuentes del sistema; sin imágenes.

Ninguna cifra sale de la GUI: todo viene de `/api/*` o de `/v1/*`.

## Pantallas

1. **Monitor** (inicio). La tira de actividad muestra los últimos 120 s, un casillero por segundo,
   con un carril de prefill y otro de decode coloreado por tokens generados en ese segundo, en una
   escala fija (0, <10, 10–20, 20–30, 30–40, 40–50, ≥50). Debajo:
   - la tabla de pedidos en curso y los 12 últimos, con cliente (producto del `User-Agent`:
     `claude-cli/…`, `codex_cli_rs/…`, `navegador`), endpoint, TTFT, tok/s, tokens de prompt,
     del prefix cache y de salida;
   - la memoria: huella del proceso contra el presupuesto, el plan (pesos, KV, workspace,
     overhead) y la presión del sistema;
   - el modelo y los botones del Model Manager (cargar, descargar, pausar, reanudar, detener;
     detener pide confirmación);
   - los contadores desde el arranque.
2. **Chat.** Streaming por `/v1/chat/completions`, razonamiento plegable, cancelar (corta la
   conexión y el daemon cancela) e historial en `localStorage`. La barra lateral muestra el último
   pedido (TTFT, decode, prompt, prefix cache, salida) con su tira de actividad, y los parámetros
   de muestreo.
3. **Modelos.** Los `.brasa` de la carpeta del modelo servido, con tamaño en disco, si entran con
   el contexto y la KV de este servidor (planner), y la huella y memoria residente del cargado. Las
   entradas del catálogo que no están en disco aparecen con su comando `brasa pull`.
4. **Benchmarks.** Gráficos de barras por contexto (decode y prefill, último reporte de cada engine
   y variante) y la tabla completa de `docs/bench/`, desde `GET /api/bench`.
5. **Plan de memoria.** El planner de `brasa plan` (`GET /api/plan`), con el veredicto, la barra del
   plan contra el presupuesto y el desglose.
6. **Agentes.** Lo que imprime `brasa connect` para cada herramienta (`GET /api/agents`), con
   botón de copiar.

## API que usa

- `GET /api/status`, `GET /api/metrics`: estado, plan, memoria y contadores (cada 2 s).
- `GET /api/activity`: pedidos en curso, los 12 últimos terminados y 120 s de actividad por segundo
  (cada 1 s). Los tokens de un pedido se reparten de forma uniforme sobre su tramo de decode al
  terminar; el prefill se ve en vivo.
- `GET /api/models`: modelos en disco y catálogo.
- `POST /api/model/{load,idle,pause,resume,stop}`: Model Manager (ADR 0025).

## Capturas

Tomadas el 2026-10-06 en M1 Pro 16 GB, macOS 27.0, con
`brasa serve qwen3-4b-q4 --ctx 16384 --kv q8_0` y pedidos reales (dos de 1654 tokens con
`User-Agent` de Claude Code y de Codex, el segundo servido desde el prefix cache, y uno desde el
chat de la GUI), a 390 px de ancho:

- [Monitor en un teléfono](monitor-movil.jpg): la tira de actividad con prefill y decode, los
  pedidos por cliente y los contadores.
- [Chat en un teléfono](chat-movil.jpg): una respuesta con lista y negrita renderizadas, el turno
  anterior cancelado a mitad y la barra con el último pedido de este chat.

Pendiente: capturas de escritorio (claro y oscuro) a 1440 px.

Las capturas de la primera versión (`chat-dos-turnos.jpg`, `chat-cancelado.jpg`, `estado.jpg`)
quedan como referencia de la GUI anterior.
