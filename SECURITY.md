# Security policy

## Supported versions

Brasa is under development, before 1.0. Vulnerabilities are fixed **only on `main`**; there are no
published releases or maintenance branches with patches.

| Version | Supported |
|---|---|
| `main` | Yes |
| Any other branch, tag or commit | No |

## Reporting a vulnerability

Use **GitHub private vulnerability reporting**: in the repository's **Security** tab, click
**"Report a vulnerability"**. The report stays private between you and the maintainers. Don't
open a public issue with details or a proof of concept.

> Private reporting has to be enabled by the repository's administrators. If the button is
> missing, open a public issue that only asks for a private contact, without any details. No
> personal email address is published as a contact.

### What to include

- **Version and environment:** `brasa --version`, `brasa doctor --json`, the repository commit,
  the macOS version and the chip (for example, M1 Pro 16 GB).
- **Component:** `brasa serve` and its API, the GUI (`/ui`), the CLI, the catalog and downloads
  (`brasa pull`, `brasa rm`, the Models screen) or the `.brasa` weights format.
- **Steps to reproduce,** with the minimal request or input file.
- The expected **impact**: for example, reading or writing outside the models folder, crashing the
  daemon, or running code.
- A **proof of concept,** if you have one.

### Timelines (good-faith goals, not an SLA)

These are good-faith goals, not a contractual obligation:

- acknowledgement: within **5 business days**;
- first assessment (whether it is in scope, and its severity): within **14 days**;
- fix or mitigation plan: as soon as possible depending on severity. Only `main` is patched.

If a report is out of scope, we explain why.

## Threat model

Brasa is a **local** engine: you run `brasa` on your machine and talk to the model over a loopback
socket. There is no remote service, no accounts and no user data. The only outbound network
traffic is the model download from Hugging Face that you start (`brasa pull` or the Models
screen).

### In scope

- **Local server.** `brasa serve` listens on `127.0.0.1` by default and **has no
  authentication**: anyone with access to the machine and the port can talk to the API. Changing
  the interface (`--host`) is the user's decision; the default does not expose the port to the
  network. It is a vulnerability, for example, if a well-formed HTTP request reads or writes outside
  the models folder, the reports folder and the configuration file, or crashes the daemon.
- **Requests from the browser.** Without authentication, a web page open in the user's browser
  could make it send requests to the daemon. Therefore:
  - state-changing requests (any method other than `GET`, `HEAD` or `OPTIONS`: `/v1/*`,
    `/api/model/*`, `/api/agents/<tool>/connect`, `/api/models/pull` (and its cancellation) and
    `/api/models/dir*`) that carry an `Origin` different from the daemon's own are rejected with
    403 (CSRF). SDKs and agents don't send `Origin`; the GUI at `/ui` is same-origin.
    `/api/agents/<tool>/connect` writes configuration files in the user's home folder (ADR 0028):
    only fixed paths under `HOME`, no paths taken from the request, and a backup of anything it
    modifies;
  - `/api/models/pull` only downloads catalog models: only the name is taken from the request,
    and the repository, revision, paths and sha256 come from the manifest (ADR 0031).
    `/api/models/dir` changes the models folder to a writable absolute path without `..` and saves
    it as `models_dir` in the configuration file, editing only that line; `/api/models/dir/choose`
    and `/dir/open` take no paths from the request (they open the macOS folder picker and the
    current folder in Finder);
  - when listening on loopback, `Host` must be `127.0.0.1`, `localhost` or `[::1]` (DNS
    rebinding).

  Bypassing either rule from a web page is an in-scope vulnerability.
- **Weights integrity.** The native `.brasa` format stores a sha256 of each tensor and one of the
  whole data block, and `brasa models verify` recomputes them. A `.brasa` or safetensors file with
  tampered headers (lengths, offsets, `dtype` or `shape` that lie) that causes a panic, an overflow
  or a read outside the mmap is an in-scope vulnerability.
- **Catalog and files.** Manifests are validated (`name`, `hf_dir` and `path` cannot leave the
  models folder; in `[prebuilt]`, `repo` is `owner/name`, `revision` a segment without `/` and
  `subdir` a relative path without `..`), `brasa pull` rejects unsafe manifest paths and checks the
  sha256 of every file, and `brasa rm` is limited to real subfolders of the models folder (it
  rejects `..`, absolute paths and symlinks). Escaping those rules is a vulnerability.
- **Embedded GUI.** The GUI is served from the binary (`include_str!`), with no remote resources,
  no CDN and no analytics, and it sends no telemetry. Data leaking to the internet from the GUI or
  the daemon is a vulnerability.
- **Daemon denial of service.** A request or input file that makes the process abort (for example,
  an out-of-range `ctx` or a header with absurd lengths) counts as a vulnerability.

### Out of scope (what is not a vulnerability)

- **The quality of the model's answers.** Qwen3-4B making things up, being wrong or writing
  insecure code is not a vulnerability in Brasa.
- **Prompt injection against an agent you connected.** If you give tools to an agent (Claude
  Code, Codex, Cline, OpenCode) and the model runs something it shouldn't because of the prompt's
  content, that is the model and the agent, not the engine: Brasa does not decide which tools the
  agent runs.
- **Memory use within what the planner declares.** If the context profile fits the budget that
  the planner computes and prints, memory use is as expected; the machine running out of RAM
  because of other open processes is not an engine vulnerability.

## How fixes are made

No contractual SLA. The fix goes to `main` and is noted in the advisory; if the reporter asks,
disclosure is coordinated. There are no published releases to backport to.

## License

Apache-2.0; see [LICENSE](LICENSE).

<!--
Threat model references (file:line):
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
