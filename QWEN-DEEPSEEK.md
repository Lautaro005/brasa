# Trabajo delegado a Qwen Code (DeepSeek)

Este archivo es el encargo completo. Leelo entero antes de empezar, y después leé `CLAUDE.md`,
`PLAN.md` y `docs/adr/0008-api-para-agentes.md`. En paralelo, Claude Code sigue con la fase 3
(kernels, velocidad, memoria) en la rama `fase-3`. Vos hacés lo que rodea al engine: catálogo de
modelos, estado del servidor, GUI web, CLI y documentación.

## Dónde trabajar

- **Worktree propio:** `/Users/lauti/Desktop/test/brasa-ui`, rama `ui`, que sale de `fase-3`. Si
  no existe, crealo con `git worktree add ../brasa-ui -b ui fase-3` desde
  `/Users/lauti/Desktop/test/brasa`. No trabajes nunca en la carpeta `brasa/`: ahí trabaja Claude.
- **Pesos:** en `brasa/models/` (gitignored). No los copies. Usá la ruta absoluta
  `/Users/lauti/Desktop/test/brasa/models/qwen3-4b-q4`, o un symlink `models -> ../brasa/models`
  dentro del worktree.
- **Integración:** no hagas merge ni rebase con `fase-3`. Claude integra la rama `ui` cuando una
  tarea esté cerrada.

## Qué podés tocar y qué no

| Podés crear o modificar | No toques (es del camino caliente o de la fase 3) |
|---|---|
| `crates/catalog/` (hoy vacío) | `crates/kernels/`, `crates/metal/`, `crates/models/` |
| `crates/daemon/` (rutas nuevas, GUI; no cambies el comportamiento de `/v1/*`) | `crates/memory/`, `crates/runtime/`, `crates/tuner/` |
| `crates/cli/` (subcomandos nuevos y UX) | `crates/quant/`: módulos **nuevos** sí; los existentes, no |
| `docs/` (guías nuevas; ADR desde **0020**) | `docs/adr/0001`–`0019`, `docs/bench/`, `fixtures/` |
| `README.md`, `scripts/` nuevos | `tools/` existentes, `PLAN.md` (fase 3), `CLAUDE.md` |

Si para una tarea necesitás cambiar algo de la columna derecha (por ejemplo, exponer un dato desde
`brasa-runtime`), no lo cambies: anotalo en "Pedidos a Claude", al final de este archivo, y seguí
con otra cosa.

## Reglas (de CLAUDE.md, resumidas; manda CLAUDE.md)

1. Nada de Ollama, llama.cpp ni MLX en el binario. Python solo en `tools/` y con `.venv`. El
   binario final no depende de Python.
2. Nada se afirma sin medir. No escribas cifras de velocidad en docs o README sin un reporte de
   `brasa benchmark` en `docs/bench/`; para eso, citá los reportes que ya existen.
3. Sin telemetría saliente. La GUI no carga nada de internet: sin CDN, sin fuentes remotas, sin
   analytics. Todo se embebe en el binario.
4. La API compatible es transporte: los tipos internos son de `brasa-core`.
5. Antes de una decisión de diseño no trivial (dependencia nueva, formato de manifiesto, estructura
   de la GUI), escribí un ADR corto en `docs/adr/` numerado desde 0020.
6. Un commit por tarea, en castellano, chico y descriptivo. No incluyas la línea
   `Claude-Session: ...`. No hagas push ni abras PR.
7. Antes de cada commit, `./scripts/ci.sh` tiene que salir con código 0 (fmt, clippy
   `-D warnings`, tests). Verificá el código de salida, no solo el texto.
8. Verificá los formatos contra la documentación vigente (API de Hugging Face, specs), no de
   memoria.

## Coordinación de GPU y memoria (importante)

Claude mide velocidad en esta misma Mac. Un `brasa serve` tuyo con el modelo cargado ocupa GPU y
~3 GB de RAM y arruina sus mediciones.

- **No corras** `cargo bench`, `brasa benchmark` ni los tests `--ignored` que cargan el modelo
  (forward, layers, session, planner).
- Para probar con el modelo real, usá `brasa serve qwen3-4b-q4 --ctx 2048`, el menor tiempo
  posible. Mientras corre, creá el archivo `/Users/lauti/Desktop/test/.brasa-gpu-en-uso` con una
  línea: qué estás corriendo y desde cuándo. Borralo al parar el servidor. Claude no mide mientras
  ese archivo exista.
- Para la mayoría del trabajo de daemon y GUI, usá un backend falso en los tests: un `Engine`
  simulado que emita eventos fijos. No hace falta el modelo.

## Tareas, en orden

Cada tarea cierra con un criterio verificable por comando. Si un criterio no se cumple, documentá
el bloqueo en "Estado" y seguí con la siguiente.

### U1. Estado y métricas del servidor

- `GET /api/status` (JSON):
  - modelo cargado, ruta y sha256 de los pesos;
  - contexto del perfil y tipo de KV;
  - plan de memoria (pesos, KV, workspace, total) y huella actual del proceso;
  - presión del sistema;
  - estado de la cola;
  - uptime y versión con commit.
- `GET /api/metrics` (JSON): contadores desde el arranque.
  - pedidos por endpoint, tokens de prompt y generados, tokens de prompt reutilizados por el
    prefix cache;
  - TTFT y tok/s de decode (último, media y p50 de los últimos 100);
  - errores por tipo.

  Se miden en el daemon alrededor de los eventos del engine.
- `brasa ps`: pide `/api/status` y `/api/metrics` a un `serve` corriendo (`--port`, por defecto
  8080) y los muestra legibles; con `--json`, crudos.
- **Aceptación:** tests del daemon con engine simulado que verifican los contadores tras un pedido
  streaming y uno no streaming; `brasa ps` contra un `serve` real a 2K muestra datos coherentes.

### U2. GUI web embebida (dashboard)

`brasa serve` sirve una GUI en `/ui`: HTML, CSS y JS sin framework ni paso de build (sin npm),
embebidos con `include_str!`/`include_bytes!`. Pantallas:

1. **Chat:**
   - streaming por `/v1/chat/completions`;
   - razonamiento `<think>` plegable y switch para desactivarlo (`/no_think` del template de
     Qwen3; verificá cómo lo expone el daemon);
   - parámetros de muestreo;
   - botón para cancelar, que corta la conexión (el daemon ya cancela al desconectar);
   - historial en `localStorage`.
2. **Estado:** lo de U1, refrescado cada 2 s. Memoria con barra (plan contra presupuesto),
   prefix cache, tok/s, cola.
3. **Benchmarks:** tabla y gráficos (SVG propio, sin librerías) de los reportes JSON de
   `docs/bench/`. El daemon expone `GET /api/bench`, que lista y lee esa carpeta si existe; marcá
   los inválidos.
4. **Plan:** formulario de contexto, tipo de KV y perfil (16gb/8gb). Llama a un endpoint
   `GET /api/plan?ctx=&kv=&perfil=`, que reutiliza el planner como `brasa plan`, sin cargar nada.
5. **Agentes:** muestra lo que imprime `brasa connect` para cada agente, con botón de copiar.

Modo claro y oscuro (`prefers-color-scheme`), usable a 1280 px y en ancho de teléfono.

**Aceptación:**
- test del daemon que pide `/ui` y cada asset (200, content-type correcto);
- test de que ningún asset referencia `http://` ni `https://` externos;
- con `serve` real a 2K, una conversación de dos turnos y una cancelación funcionan. Dejá capturas
  en `docs/gui/`.

### U3. Catálogo de modelos (`crates/catalog`)

- **Manifiesto** por modelo (`catalog/qwen3-4b.toml` o embebido):
  - repo y revisión de Hugging Face;
  - archivos con sha256;
  - cuantización destino;
  - contexto máximo;
  - licencia.

  ADR 0020 con el formato.
- **Comandos:**
  - `brasa models`: lista los modelos locales con tamaño, hash verificado sí/no y si entran en
    memoria con el contexto por defecto (planner);
  - `brasa models verify <modelo>`: recalcula los sha256 contra el `.brasa`;
  - `brasa pull <modelo>`: descarga los safetensors de Hugging Face a la revisión fijada, con
    reanudación y verificación de sha256;
  - `brasa rm <modelo>`: pide confirmación.
- **Aceptación:** `brasa models verify qwen3-4b-q4` pasa con los pesos actuales. `brasa pull` de un
  repo chico de prueba verifica los hashes y falla limpio con un hash alterado.

### U4. Conversión nativa a `.brasa` (sin Python)

Portá `tools/convert_brasa.py` a Rust, como módulo **nuevo** en `crates/quant` (por ejemplo
`convert.rs`) usando el lector de safetensors y el escritor existentes, y exponelo como
`brasa convert <dir-hf> <dir-salida>`. Así `brasa pull` puede terminar en un modelo usable.

**Aceptación:** sobre `models/qwen3-4b-hf`, el `.brasa` generado tiene **los mismos sha256 por
tensor** que el de `tools/convert_brasa.py` (`models/qwen3-4b-q4/model.brasa`). Comparalo con un
test `--ignored` que no cargue la GPU. Si algún tensor difiere (por ejemplo, por el redondeo de
bf16 → f32 → q4), documentá exactamente por qué y no lo des por cerrado.

### U5. CLI y configuración

- `~/.config/brasa/config.toml`: puerto, contexto, `--kv` y modelo por defecto. Los flags mandan
  sobre el archivo, y `brasa config show` muestra la configuración efectiva y de dónde salió cada
  valor.
- `brasa completions zsh|bash|fish` (clap_complete).
- Errores con sugerencia accionable (modelo inexistente → `brasa models`; contexto que no entra →
  el `--ctx` máximo, que el planner ya calcula).
- `--json` en `doctor` (ya existe), `plan`, `models` y `ps`.
- **Aceptación:** tests de precedencia de configuración (flag > archivo > defecto) y snapshot del
  `--help` de cada subcomando.

### U6. Documentación de usuario

- `README.md`: qué es, instalación desde fuente, primeros pasos (`doctor` → `pull`/`convert` →
  `run` → `serve` → `connect`) y cifras **solo** citando `docs/bench/baseline.md`.
- `docs/guia/`: uso con Claude Code, Codex, Cline y OpenCode; GUI; perfiles de memoria y `--kv`;
  solución de problemas.
- **Aceptación:** cada comando citado en la guía existe (un script que extraiga los bloques
  `bash` y corra `--help` de cada subcomando).

## Estado (actualizalo al cerrar cada tarea)

| Tarea | Estado | Commit | Notas |
|---|---|---|---|
| U1 | hecha | 42c37e2 | Tests del daemon con engine simulado (streaming y no streaming) y `/api/status`. Falta correr `brasa ps` contra un `serve` real a 2K: el usuario pidió no ejecutar `brasa`. |
| U2 | pendiente | | |
| U3 | pendiente | | |
| U4 | pendiente | | |
| U5 | pendiente | | |
| U6 | pendiente | | |

## Pedidos a Claude

(Cambios que necesitás fuera de tu zona. Uno por línea, con qué y para qué.)
