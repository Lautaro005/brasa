# Política de seguridad

> **English (short).** Brasa is pre-1.0 software under active development: only the `main` branch
> gets security fixes. Please report vulnerabilities through GitHub's **private vulnerability
> reporting** ("Report a vulnerability" in the repository's **Security** tab), not in a public
> issue. Below: what to include, the response timelines (good-faith goals, not an SLA) and the
> project's threat model.

## Versiones soportadas

Brasa está en desarrollo, antes de 1.0. Se corrigen vulnerabilidades **solo en `main`**; no hay
versiones publicadas ni ramas de mantenimiento con parches.

| Versión | Soporte |
|---|---|
| `main` | Sí |
| Cualquier otra rama, etiqueta o commit | No |

## Cómo reportar

Usá el **reporte privado de vulnerabilidades de GitHub**: en la pestaña **Security** del
repositorio, botón **"Report a vulnerability"**. El reporte queda privado entre quien lo envía y los
mantenedores. No abras un issue público con detalles ni con una prueba de concepto.

> El reporte privado tiene que estar habilitado por la administración del repositorio. Si el botón
> no aparece, ver "Pedidos al usuario". No se publica ningún email personal como contacto.

### Qué incluir

- **Versión y entorno:** `brasa --version`, `brasa doctor --json`, commit del repo, versión de macOS
  y chip (por ejemplo, M1 Pro 16 GB).
- **Componente:** `brasa serve` y su API, la GUI (`/ui`), el CLI, el catálogo
  (`brasa pull`/`brasa rm`) o el formato de pesos `.brasa`.
- **Pasos para reproducir** con el pedido o el archivo de entrada mínimo.
- **Impacto** esperado: por ejemplo, leer o escribir fuera de la carpeta de modelos, tumbar el
  daemon, o ejecutar código.
- Si lo tenés, una **prueba de concepto**.

### Plazos (compromiso de buena fe, no un SLA)

Son objetivos de buena fe, no una obligación contractual:

- acuse de recibo: dentro de **5 días hábiles**;
- primera evaluación (si está en alcance y su gravedad): dentro de **14 días**;
- corrección o plan de mitigación: lo antes posible según la gravedad. `main` es lo único que se
  parchea.

Si el reporte queda fuera de alcance, se explica por qué.

## Modelo de amenazas

Brasa es un engine **local**: el usuario corre `brasa` en su máquina y habla con el modelo por un
socket de loopback. No hay servicio remoto, cuentas ni datos de usuario.

### En alcance

- **Servidor local.** `brasa serve` escucha en `127.0.0.1` por defecto y **no tiene
  autenticación**: cualquiera con acceso a la máquina y al puerto puede hablar con la API. Cambiar
  la interfaz (`--host`) es decisión del usuario; el default no expone el puerto a la red. Es
  vulnerabilidad, por ejemplo, que un pedido HTTP bien formado lea o escriba fuera de la carpeta de
  modelos, de la carpeta de reportes y del archivo de configuración, o que tumbe el daemon.
- **Pedidos desde el navegador.** Sin autenticación, una página web abierta en el navegador del
  usuario podría hacer que este envíe pedidos al daemon. Por eso:
  - los pedidos que modifican estado (cualquier método que no sea `GET`, `HEAD` u `OPTIONS`: `/v1/*`,
    `/api/model/*`, `/api/agents/<herramienta>/connect`, `/api/models/pull` (y su cancelación) y
    `/api/models/dir*`) que traen un `Origin` distinto del propio daemon se rechazan con 403 (CSRF).
    Los SDKs y agentes no mandan `Origin`; la GUI de `/ui` es del mismo origen.
    `/api/agents/<herramienta>/connect` escribe archivos de configuración en la carpeta del usuario
    (ADR 0028): solo rutas fijas debajo de `HOME`, sin caminos que vengan del pedido, y con respaldo
    de lo que modifica;
  - `/api/models/pull` descarga solo modelos del catálogo: del pedido se usa el nombre, y repo,
    revisión, rutas y sha256 salen del manifiesto (ADR 0031). `/api/models/dir` cambia la carpeta de
    modelos a una ruta absoluta sin `..` que se pueda escribir y la guarda en `models_dir` del
    archivo de configuración editando solo esa línea; `/api/models/dir/choose` y `/dir/open` no
    reciben rutas del pedido (abren el selector de macOS y la carpeta efectiva en Finder);
  - escuchando en loopback, el `Host` tiene que ser `127.0.0.1`, `localhost` o `[::1]` (DNS
    rebinding).

  Saltear cualquiera de las dos reglas desde una página web es una vulnerabilidad en alcance.
- **Integridad de los pesos.** El formato nativo `.brasa` guarda un sha256 de cada tensor y uno del
  bloque de datos completo, y `brasa models verify` los recalcula. Un `.brasa` o un safetensors con
  encabezados manipulados (longitudes, offsets, `dtype` o `shape` mentidos) que provoque un pánico,
  un desborde o una lectura fuera del mmap es una vulnerabilidad en alcance.
- **Catálogo y archivos.** Los manifiestos se validan (`name`, `hf_dir` y `path` no pueden salir de
  la carpeta de modelos; en `[prebuilt]`, `repo` es `dueño/nombre`, `revision` un segmento sin `/`
  y `subdir` una ruta relativa sin `..`), `brasa pull` rechaza rutas inseguras del manifiesto y
  verifica el sha256 de cada archivo, y `brasa rm` se limita a subcarpetas reales de la carpeta de
  modelos (rechaza `..`, rutas absolutas y symlinks). Un escape de esas reglas es una
  vulnerabilidad.
- **GUI embebida.** La GUI se sirve desde el binario (`include_str!`), sin recursos remotos, sin
  CDN y sin analytics; tampoco manda telemetría. Una fuga de datos a internet desde la GUI o el
  daemon es una vulnerabilidad.
- **Denegación de servicio del daemon.** Un pedido o un archivo de entrada que haga abortar el
  proceso (por ejemplo, un `ctx` fuera de rango o un encabezado con longitudes absurdas) cuenta como
  vulnerabilidad.

### Fuera de alcance (qué no es una vulnerabilidad)

- **La calidad de las respuestas del modelo.** Que Qwen3-4B invente, se equivoque o genere código
  inseguro no es una vulnerabilidad de Brasa.
- **Prompt injection hacia un agente que conectaste.** Si le das herramientas a un agente (Claude
  Code, Codex, Cline, OpenCode) y el modelo ejecuta algo indebido por el contenido del prompt, es el
  modelo y el agente, no el engine: Brasa no decide qué herramientas corre el agente.
- **Consumo de memoria dentro de lo que el planner declara.** Si el perfil de contexto entra en el
  presupuesto que el planner calcula e imprime, el uso de memoria es el esperado; que la máquina se
  quede sin RAM por otros procesos abiertos no es una vulnerabilidad del engine.

## Cómo se corrigen

Sin SLA contractual. La corrección va a `main` y se anota en el aviso; si el reporte lo pide, se
coordina la divulgación. No hay versiones publicadas que retro-portar.

## Licencia

Apache-2.0; ver [LICENSE](LICENSE).

<!--
Referencias del modelo de amenazas (archivo:línea):
- Pedidos desde el navegador (Origin y Host): crates/daemon/src/origin.rs (check, local_only)
  y crates/daemon/src/lib.rs (router, capa from_fn_with_state).
- Conectar agentes (rutas fijas, respaldo, sin pisar archivos ajenos): crates/daemon/src/connect_apply.rs
  (Paths, codex, claude_code, opencode) y sus tests con un HOME temporal; test de API
  `connect_rechaza_cline_y_herramientas_desconocidas` (incluye el 403 por origen).
- Descarga y carpeta de modelos (ADR 0031): rutas en crates/daemon/src/lib.rs:164-174, detrás de la
  misma capa de origen; crates/daemon/src/models_admin.rs:226 (pull_start: el nombre se busca en el
  catálogo, nada más sale del pedido), :338 (pull_cancel), :364 (set_dir), :406 (choose_dir) y :419
  (open_dir, sin rutas del pedido); :71 (osascript con la carpeta como argumento, no interpolada).
  Validación de la carpeta: crates/catalog/src/dirs.rs:163 (prepare_models_dir: absoluta, sin `..`,
  se prueba la escritura) y :202 (save_models_dir: edita una sola línea y verifica que el resto quede
  igual). URL armada solo del manifiesto: crates/catalog/src/pull.rs:219. Tests de API
  `descarga_y_carpeta_rechazan_otro_origen` (403 para los seis POST/DELETE, con un escritorio que
  entra en pánico si se lo llama), `pull_baja_los_pesos_convertidos_del_catalogo` (un `url` en el
  pedido se ignora) y `dir_cambia_la_carpeta_y_la_guarda` (rutas relativas y `..` dan 400; un
  archivo con claves desconocidas no se reescribe).
- `brasa serve` en 127.0.0.1 sin autenticación: crates/cli/src/config.rs:12 (DEFAULT_HOST),
  crates/cli/src/serve.rs:19 (flag --host), crates/daemon/src/lib.rs:157 (router(), sin middleware
  de autenticación) y crates/daemon/src/lib.rs:261 (TcpListener::bind(cfg.addr)).
- sha256 por tensor y del bloque de datos: crates/quant/src/brasa_file.rs:98 (verify),
  crates/quant/src/brasa_file.rs:101 (sha256 por tensor), crates/quant/src/brasa_file.rs:110
  (sha256 del conjunto); crates/catalog/src/verify.rs:19 (verify) y crates/cli/src/models.rs:133
  (`brasa models verify`).
- Manifiestos validados y sin rutas fuera de la carpeta de modelos: crates/catalog/src/manifest.rs:16
  (safe_relative), crates/catalog/src/manifest.rs:34 (validate_component),
  crates/catalog/src/manifest.rs:159 (validate, incluido `[prebuilt]`); crates/catalog/src/pull.rs:186
  (safe_relative del path antes de descargar).
- `rm` acotado: crates/catalog/src/local.rs:88 (resolve_child), crates/catalog/src/local.rs:95
  (resolve_child_with, que rechaza una base con symlink) y crates/cli/src/rm.rs:32/51
  (resolve_child_with antes de remove_dir_all).
- GUI embebida sin recursos remotos ni telemetría: crates/daemon/src/ui.rs:7-9 (include_str!),
  crates/daemon/tests/api.rs:211-212 (test: ningún asset referencia http:// ni https://);
  CLAUDE.md:23 (regla 9, sin telemetría saliente).
- Archivos de pesos maliciosos: crates/quant/src/brasa_file.rs:64 (open valida magic, versión,
  límites y alineación), crates/quant/src/brasa_file.rs:67 (magic BRSA) y
  crates/quant/src/safetensors.rs:18 (open parsea el encabezado JSON con el tamaño declarado).
- Presupuesto de memoria del planner: crates/memory/src/planner.rs:82 (plan).
- Prompt injection fuera de alcance: crates/daemon/src/openai.rs:131 (el daemon traduce el pedido a
  los tipos de brasa-core y sirve al modelo; no ejecuta herramientas).
- `ctx` fuera de rango acotado: crates/daemon/src/lib.rs:141 (MAX_CTX).
-->
