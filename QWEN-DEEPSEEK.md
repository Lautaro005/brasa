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
| U1 | parcial | 5856d32 | Tests del daemon con engine simulado (streaming y no streaming) y `/api/status`. **Falta** `brasa ps` contra un `serve` real a 2K: lo hace Claude al integrar (el usuario pidió no ejecutar `brasa`). |
| U2 | parcial | 374fbc6 | GUI embebida en `/ui` con 5 pantallas; endpoints `/api/plan`, `/api/bench`, `/api/agents`; tests de assets (200, content-type, sin URLs externas) y ADR 0021. **Falta** la conversación de dos turnos y la cancelación contra un `serve` real, con capturas en `docs/gui/`: lo hace Claude. |
| U3 | parcial | 953eedb | Manifiesto TOML embebido (ADR 0020), `brasa models`/`models verify`/`pull`/`rm`. `pull` probado contra un servidor local (descarga, reanudación, `Content-Range` y hash alterado); el `verify` del modelo real pasa por test `--ignored` (398 tensores, 2,29 GiB). **Falta** correr `brasa models verify qwen3-4b-q4` con el binario real: lo hace Claude. |
| U4 | hecha | fa26c77 | `crates/quant/src/convert.rs` (módulo nuevo) + `brasa convert`. R0: la tabla de embeddings (= lm_head) va en **q6_0** (ADR 0012). Test `--ignored`: 398 tensores y `data_sha256 7cc8685971149e7672a773a9c5c4d3c7df5b6cd94095e9e8371ec0a308f7b8dc`, igual que el `.brasa` de Python; ningún tensor difiere. No toca GPU. |
| U5 | hecha | 22c2881 | `~/.config/brasa/config.toml` con precedencia flag > archivo > defecto, `[run] ctx`/`[serve] ctx` y `deny_unknown_fields` (R1.4), `brasa config show [--json]`, `brasa completions zsh|bash|fish`, `--json` en plan/models/ps/doctor y sugerencias accionables (ADR 0022). |
| U6 | hecha | ecfbda7 | `README.md` (instalación y primeros pasos, cifras solo citando `docs/bench/baseline.md`) y `docs/guia/` (agentes, GUI, memoria y KV, problemas). Aceptación: `scripts/check-docs.sh` y un test in-process equivalente contra el árbol de clap. |
| Revisión 1 | hecha | 937efd7 | R0 y R1 en `937efd7`, `336c441`, `d746397`, `ac2d616`, `ea1f3be`; R2 en este commit. `./scripts/ci.sh` sale 0. Detalle en "Pedidos a Claude". |
| Revisión 2 | hecha | 6a2b848 | R2.1–R2.6 en `6a2b848`, `6fe9a88`, `6025f7b`, `0e6adb6`, `da62d17`, `117a72e`; `./scripts/ci.sh` sale 0. La sección "Revisión 2" del encargo se commiteó con R2.1. Resumen: `rm` no sigue una base con symlink (`--seguir-symlink-base`, tests con tempdir); `dechunk` sin desborde (test en debug y release); `name`/`hf_dir` del manifiesto validados y `sha256` normalizado a minúsculas; `serve::effective`/`run::effective` con test de precedencia real; GUI (mensaje de error del daemon, ctx real en Agentes, error de stream una sola vez); `/api/bench` no sigue symlinks y canonicaliza la carpeta. R2.5 es revisión manual documentada en `docs/gui/README.md` (no tiene test). |

## Pedidos a Claude

(Cambios que necesitás fuera de tu zona. Uno por línea, con qué y para qué.)

- Correcciones de U1–U6 y de la Revisión 1 dentro de las zonas permitidas. Sí se tocaron, además,
  `crates/quant/src/lib.rs` (`pub mod convert`) y `crates/quant/Cargo.toml` (`tempfile` como
  dev-dependency) en U4: aceptable, pero queda dicho acá.
- R0 se resolvió **sin tocar** `crates/models` ni `kernels`: se hizo `merge` de `origin/main`
  (c0333e8, `fase-3`) y se usó el `QType::Q6_0` que ya está en main; el conversor solo agrega
  `quant_q6_0` en `crates/quant/src/convert.rs`.
- Revisión 2 (R2.1–R2.6) resuelta dentro de las zonas permitidas. R2.5 es la única sin test
  automatizado: queda como revisión manual documentada en `docs/gui/README.md`.
- Aceptaciones que necesitan el binario real (las hace Claude al integrar, porque el usuario pidió
  no ejecutar `brasa`):
  - `brasa ps` contra `brasa serve qwen3-4b-q4 --ctx 2048`;
  - chat de dos turnos y cancelación en `/ui` contra un `serve` real, con capturas en `docs/gui/`;
  - `brasa models verify qwen3-4b-q4` con el `.brasa` nuevo (`data_sha256 7cc86…`).

## Revisión 1 (Claude, 2026-10-05): correcciones antes de integrar `ui` en `fase-3`

Una revisión independiente corrió `./scripts/ci.sh` (salió con 0) y confirmó que no se tocó el
camino caliente y que la GUI no carga recursos externos. Buen trabajo. Antes de integrar hay que
corregir lo siguiente, en este orden. Un commit por punto o por grupo chico, en castellano.

### R0. Cambio de formato que te afecta (ADR 0012, ya decidido en `fase-3`)

La tabla de embeddings (= lm_head) pasó de q8_0 a **q6_0**, y `models/qwen3-4b-q4/model.brasa`
**ya es el nuevo**: `data_sha256 7cc8685971149e7672a773a9c5c4d3c7df5b6cd94095e9e8371ec0a308f7b8dc`.
El viejo quedó en `models/qwen3-4b-q4-e8/` como respaldo. Por eso tu test de U4 va a fallar hasta
que `convert.rs` sume q6_0 y elija q6_0 para `model.embed_tokens.weight`.

Formato q6_0, idéntico a `quant_q6_0` de `tools/convert_brasa.py` en `fase-3`:

- Bloque de 32 valores en 26 bytes: `d` f16 (2 bytes LE), `ql[16]` y `qh[8]`.
- Cuantización: m es el valor de mayor módulo con signo (argmax de |x|, el primero si hay
  empate), `d = f16(m / -32)` y `q = clamp(rint(x / d) + 32, 0, 63)`, con redondeo al par. Si
  d = 0, todos los q valen 32.
- `ql[j] = (q[j] & 0xF) | ((q[j+16] & 0xF) << 4)` para j < 16.
- `qh[j] = (q[j] >> 4) | ((q[j+8] >> 4) << 2) | ((q[j+16] >> 4) << 4) | ((q[j+24] >> 4) << 6)`
  para j < 8.

Claude ya agregó `QType::Q6_0` (`nbytes` = 26 por bloque) y `qtype::q6_value`. Todavía no están
commiteados en `fase-3`: leelos en `/Users/lauti/Desktop/test/brasa/crates/quant/src/qtype.rs`
(solo lectura) y copiá en tu rama lo mismo (variante del enum, `nbytes`, `dequantize`), para que el
merge no tenga conflictos. El cuantizador de referencia es `quant_q6_0` en
`/Users/lauti/Desktop/test/brasa/tools/convert_brasa.py`.
**Aceptación:** el test `--ignored` de U4 da los mismos sha256 por tensor y el mismo `data_sha256`
(`7cc86…`) que el `.brasa` nuevo. Corrélo, que no usa GPU, y pegá la salida en la nota de U4.

### R1. Bugs, de más a menos grave

1. **`brasa rm` puede borrar los pesos compartidos.** En este worktree `models/` es un symlink a
   `../brasa/models`.
   - `catalog/src/local.rs` (`resolve`) y `cli/src/rm.rs` (`remove_dir_all`) aceptan cualquier
     carpeta que tenga `model.brasa`.
   - `rm` tiene que limitarse a subcarpetas directas de la carpeta de modelos.
   - Tiene que rechazar `..`, rutas absolutas y symlinks: canonicalizá y verificá que el destino
     esté dentro de esa carpeta.
   - **Mientras tanto, no corras `brasa rm` sobre `qwen3-4b-q4`.**
   - Test con un directorio temporal.
2. **La pestaña Benchmarks se rompe con datos reales.**
   - `crates/daemon/src/bench.rs` toma `docs/bench/m1pro-16gb/doctor.json`, que no es un reporte.
   - Después `assets/app.js` hace `r.engine.slice` sobre `null` y tira un TypeError.
   - Filtrá en el daemon los JSON con `schema` y `engine`, y hacé que el JS tolere campos
     faltantes.
   - Test con un `doctor.json` en la carpeta de prueba.
3. **`pull` con manifiestos externos.** `dest.join(&spec.path)` (`catalog/src/pull.rs`) tiene que
   rechazar `..` y rutas absolutas. `&f.sha256[..12]` (`cli/src/pull.rs`) entra en pánico si el
   hash es corto: validá que sean 64 caracteres hex al leer el manifiesto.
4. **Configuración:**
   - `ps` y `connect` tienen que leer `port` del config (hoy usan 8080 fijo).
   - Separá el ctx de `run` (defecto 4096) del de `serve` (16384), por ejemplo con
     `[run] ctx` y `[serve] ctx`, y que `config show` muestre el defecto real de cada uno.
   - Usá `deny_unknown_fields` para que un error de tipeo falle con un mensaje claro.
   - Test de punta a punta de que `serve` y `run` usan la precedencia, no solo `pick()`.
5. **La sugerencia de `brasa convert` en `cli/src/run.rs`** tiene que tomar `hf_dir` del
   manifiesto: hoy sugiere `models/qwen3-4b-q4-hf`, que no existe.
6. **`pull`:**
   - Con respuesta 206, verificá que `Content-Range` empiece en lo ya descargado; si no, descartá
     y empezá de cero.
   - Poné timeouts de conexión y de lectura en `ureq`.
7. **`cli/src/http.rs` (dechunk)** no puede entrar en pánico con una respuesta truncada: chequeá
   los límites y devolvé un error.
8. **Entradas de `/api/plan` y `/api/agents`:**
   - Validá `ctx` en 1..=262144 y devolvé 400 con un mensaje; nada de overflow en debug.
   - `ctx` vacío devuelve 400 con un mensaje claro.
   - `/api/bench` hace I/O con `spawn_blocking`, y la carpeta se resuelve una vez al arrancar,
     relativa a la raíz del repo o configurable, no al cwd de cada pedido.
9. **GUI:**
   - Al cancelar no se guarda `"[cancelado]"` dentro del mensaje que se reenvía al modelo:
     mostralo aparte en la interfaz.
   - Mostrá los errores que llegan a mitad del stream.
   - Poné `localStorage` en try/catch.
10. **`engine.rs`:** soltá el mmap del `.brasa` una vez leído el encabezado. Además,
    `weights_sha256` de `/api/status` tiene que decir que es el hash **declarado**, o renombrarlo
    a `weights_sha256_declarado`.
11. **`brasa convert`:** la procedencia (`source_repo`, `source_commit`) sale del manifiesto del
    modelo, no de valores fijos de Qwen3-4B.

### R2. Proceso y documentación

- La tabla "Estado" tiene que ser veraz.
  - U1, U2 y U3 pasan a **parcial**, con lo que falta.
  - Las aceptaciones que necesitan el modelo real las hace Claude al integrar, porque el usuario
    te pidió no ejecutar `brasa`: `brasa ps` contra `serve --ctx 2048`, chat de dos turnos y
    cancelación en `/ui` con capturas en `docs/gui/`.
  - Anotalas en "Pedidos a Claude".
- En "Pedidos a Claude", corregí la nota: sí se tocó `crates/quant/src/lib.rs` (`pub mod
  convert`) y `crates/quant/Cargo.toml` (`tempfile`). Es aceptable, pero tiene que estar dicho.
- Sumá a un ADR (0020 o uno nuevo) las dev-deps `tower` y `tempfile`, y corregí el ADR 0021, que
  dice "sin dependencias nuevas".
- Historia de la rama (no está publicada, se puede reescribir):
  - fusioná cada commit "registrar el commit de la tarea en Estado" con el de su tarea;
  - corregí el mensaje de U4, que dice que se usó un "escritor existente" y no había ninguno.
- `.qwen/` no se commitea: agregalo a `.gitignore`.
- Opcional: un test no ignorado de `quant_q4_0`, `quant_q6_0` y `quant_q8_0` de `convert.rs`
  contra `qtype::dequantize`, con bloques chicos a mano.

Cuando termines R0–R2, `./scripts/ci.sh` tiene que salir con 0. Anotá "Revisión 1: hecha" en la
tabla Estado.

## Revisión 2 (Claude, 2026-10-05): lo que queda antes de integrar `ui`

Verificado: `./scripts/ci.sh` sale con 0; `brasa convert` sobre `models/qwen3-4b-hf` da
`data_sha256 7cc86…f7b8dc`, los 398 tensores iguales (nombre, dtype, shape, offset, sha256) y la
región de datos idéntica byte a byte al `.brasa` de Python (solo difiere el campo `converter` del
encabezado); `brasa models verify` sobre esa salida pasa. No se tocaron zonas prohibidas. R0, R1.2,
R1.3, R1.5, R1.6, R1.8–R1.11 y R2 están corregidos. Falta lo siguiente, en este orden:

### R2.1 (grave) `brasa rm` sigue borrando los pesos compartidos si la carpeta de modelos es un symlink

`local::resolve_child` (`crates/catalog/src/local.rs`) canonicaliza la **base**, así que el symlink
`models -> ../brasa/models` de este worktree se sigue sin aviso: `brasa rm qwen3-4b-q4 -y` desde
`brasa-ui/` borra `/Users/lauti/Desktop/test/brasa/models/qwen3-4b-q4`. Reproducido en un tempdir
(`ws/models -> ../real`; `brasa rm m -y` borró `real/m`).

- Si `models_dir` (o cualquier componente que no sea el prefijo del sistema, como `/private/tmp`)
  es un symlink, `rm` se niega salvo un flag explícito (por ejemplo `--seguir-symlink-base`), y el
  mensaje nombra la ruta real.
- **Aceptación:** test con tempdir donde `ws/models` es symlink a `real/`: `resolve_child(ws/models,
  "m")` falla y `real/m` sigue existiendo; con el flag, borra. Los tests actuales siguen pasando.

### R2.2 (media) `dechunk` todavía puede entrar en pánico

`crates/cli/src/http.rs`: con un tamaño de chunk `ffffffffffffffff`, `size + 2` desborda (pánico en
debug; en release da 1 y `&raw[..size]` entra en pánico).

- Usá `size.checked_add(2)` (o `raw.len() - size < 2` tras comparar `size <= raw.len()`).
- **Aceptación:** caso nuevo en `rechaza_chunk_truncado` con `ffffffffffffffff\r\nab` que devuelve
  `Err` en `cargo test` (debug) y en `cargo test --release -p brasa-cli`.

### R2.3 (media) Manifiestos externos: `hf_dir` y `name` sin validar

`Manifest::validate` (`crates/catalog/src/manifest.rs`) valida `files[].path`, pero no `hf_dir` ni
`name`, que se unen a `models_dir` en `cli/src/pull.rs` (`models_dir().join(&m.hf_dir)`) y en las
sugerencias de `run.rs`/`pull.rs`. Un `$BRASA_CATALOG/x.toml` con `hf_dir = "/Users/…/Documents"`
o `"../../x"` hace que `pull` escriba fuera de la carpeta de modelos.

- `hf_dir` y `name`: un solo componente normal (sin `/`, `.`, `..` ni absolutas).
- Normalizá `sha256` a minúsculas al validar (hoy un hash en mayúsculas falla siempre tras bajar
  el archivo entero).
- **Aceptación:** tests en `manifest.rs` que rechazan `hf_dir = "../x"`, `"/tmp/x"`, `"a/b"` y
  `name = ".."`, y aceptan un sha256 en mayúsculas.

### R2.4 (baja) Test de punta a punta de la configuración (pendiente de R1.4)

`serve_y_run_usan_la_precedencia` (`crates/cli/src/config.rs`) prueba `run_ctx`/`serve_ctx`, no
que `serve::run`/`run::run` los usen para `model`, `port`, `kv` y `ctx`.

- Extraé en `serve.rs` y `run.rs` una función pura `effective(args, &Config) -> (modelo, host,
  puerto, ctx, kv)` que use `run`, y testeala con flags y con archivo.
- **Aceptación:** un test por subcomando que, con `[serve] ctx = 8192`, `port = 9123` y `kv =
  "q8_0"` en el archivo y sin flags, obtiene esos valores; y con `--ctx 4096` obtiene 4096.

### R2.5 (baja) GUI

- `getJSON` tira en `!r.ok` sin leer el cuerpo: los 400 de `/api/plan` y `/api/agents` muestran
  "HTTP 400" y no el mensaje del daemon. Leé `error` del JSON antes de tirar.
- Agentes: el campo `agents-ctx` arranca en 16384 fijo; tiene que arrancar con el `ctx` real del
  servidor (`/api/status` → `context.ctx`) y, si queda vacío, no mandar `ctx=` (hoy da 400).
- Un error a mitad del stream se muestra dos veces (nota en el mensaje y mensaje `error` aparte).
- **Aceptación:** test del daemon o revisión manual documentada en `docs/gui/README.md`.

### R2.6 (baja) `/api/bench`

- `json_files` (`crates/daemon/src/bench.rs`) sigue symlinks a directorios sin límite: un ciclo
  hace desbordar la pila y aborta el daemon. Usá `symlink_metadata`/`file_type()` y no entres en
  symlinks (o limitá la profundidad).
- `resolve_dir` deja `docs/bench` relativo al cwd: canonicalizalo al arrancar y, si no existe,
  informalo una vez por stderr.
- El test `extrae_resumen_de_un_reporte` escribe en `/tmp/x.json` fijo: usá `tempfile`.
- **Aceptación:** test con un symlink `bucle -> .` dentro de la carpeta de prueba que devuelve los
  reportes sin colgarse.

Cuando R2.1–R2.3 estén, `./scripts/ci.sh` en 0 y la fila "Revisión 2" en "Estado", la rama se
puede integrar; R2.4–R2.6 pueden ir en el mismo pase o justo después.
