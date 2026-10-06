# ADR 0025 — Model Manager (load, idle, pause, resume, stop)

Estado: aceptada (V5)

## Contexto

PLAN.md, fase 4: el Model Manager. Hoy el daemon carga el modelo al arrancar y lo tiene siempre en
memoria; `brasa serve` no lo suelta hasta que se corta el proceso. En una Mac de 8 GB, o compartida
con otras apps, hace falta poder liberar pesos y KV sin cerrar el daemon, y frenar el trabajo sin
descargar.

Restricciones de la tarea: solo `crates/daemon/` y `crates/cli/`. El contexto Metal no es `Sync`, así
que la sesión vive en el hilo del engine (ADR 0008); soltar la `Session` alcanza para liberar los
pesos, sin tocar `brasa-runtime`.

## Decisión

**Estados** (los informa `/api/status` en `state`): `loaded`, `idle`, `paused`, `loading`,
`stopped`. `loading` es transitorio (mientras se carga).

**Operaciones** (endpoint `POST /api/model/<op>` y `brasa model <op>`, que lo llama por HTTP y espera
el resultado):

- **`load`**: crea la `Session` con el plan de memoria. Si falla, queda en `idle` y devuelve el
  error. Limpia la pausa.
- **`idle`**: suelta la `Session` (pesos y KV). El daemon sigue vivo atendiendo `/api/*`. Es la
  única operación que baja la huella. Limpia la pausa.
- **`pause`**: deja de sacar trabajos de la cola, sin descargar. La cola es FIFO: lo que ya estaba
  encolado antes del `pause` se sirve primero. Error si el modelo está en `idle`.
- **`resume`**: vuelve a sacar trabajos. Error si no estaba en `pause`.
- **`stop`**: suelta la `Session`, termina el hilo del engine y apaga el servidor con el apagado
  ordenado de axum (las respuestas en vuelo terminan). Idempotente.

**`/v1/*` según el estado:**

- **Cargado o en pausa sin descargar**: igual que siempre.
- **`idle`**: el próximo pedido **vuelve a cargar** el modelo y se sirve; el cliente solo ve más
  latencia en ese primer pedido. Si la recarga falla, el pedido termina con
  `ChatEvent::Error(Internal, …)`, traducido al formato de cada API. Se eligió recargar en vez de
  devolver un error para no romper a un agente que ya está andando; `pause` es la herramienta para
  frenar, no `idle`.
- **`paused`**: los pedidos quedan encolados y no se procesan hasta `resume`.

**`brasa ps`** muestra el estado y, en `idle`, no informa huella de modelo.

**Hilo del engine.** Dos canales: trabajos y control. El control tiene prioridad y se atiende con
`try_recv`; cuando no hay trabajos se espera con `recv_timeout` de 25 ms para no bloquear el control.
En pausa, el hilo espera solo en el canal de control, así los trabajos encolados quedan detenidos de
verdad. `load`/`idle`/`stop` bloquean, así que los handlers los corren en `spawn_blocking`.

## Consecuencias

- `idle` baja la huella; se mide con `brasa ps` (una corrida corta con el marcador de GPU).
- El auto-load desde `idle` cuesta latencia en el primer pedido tras descargar; a cambio, un agente
  no recibe un error por un modelo que el usuario descargó para hacer lugar.
- `pause` puede dejar a un cliente esperando indefinidamente: es su semántica y queda documentada.
- `drop(session)` libera los buffers de Metal; no hace falta tocar `brasa-runtime`.
- Se habilitó la feature `time` de tokio en el workspace, solo para el test de `pause` (necesita un
  timeout). Medición con el modelo real: `brasa ps` pasa de 2,70 GiB de huella a 0,14 GiB con
  `idle`, y vuelve a 2,72 GiB con `load`.
