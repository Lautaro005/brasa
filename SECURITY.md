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
- Requests from the browser (Origin and Host): crates/daemon/src/origin.rs (check, local_only)
  and crates/daemon/src/lib.rs (router, from_fn_with_state layer).
- Connecting agents (fixed paths, backup, never overwriting files that aren't Brasa's):
  crates/daemon/src/connect_apply.rs (Paths, codex, claude_code, opencode) and its tests with a
  temporary HOME; API test `connect_rechaza_cline_y_herramientas_desconocidas` (includes the 403
  by origin).
- Download and models folder (ADR 0031): routes in crates/daemon/src/lib.rs:168-177, behind the
  same origin layer; crates/daemon/src/models_admin.rs:219 (pull_start: the name is looked up in
  the catalog, nothing else is taken from the request), :338 (pull_cancel), :364 (set_dir), :406
  (choose_dir) and :419 (open_dir, no paths from the request); :71 (osascript with the folder as
  an argument, not interpolated). Folder validation: crates/catalog/src/dirs.rs:163
  (prepare_models_dir: absolute, no `..`, the write is tested) and :202 (save_models_dir: edits a
  single line and checks that the rest stays the same). URL built only from the manifest:
  crates/catalog/src/pull.rs:219. API tests `descarga_y_carpeta_rechazan_otro_origen` (403 for the
  six POST/DELETE routes, with a desktop that panics if it is called),
  `pull_baja_los_pesos_convertidos_del_catalogo` (a `url` in the request is ignored) and
  `dir_cambia_la_carpeta_y_la_guarda` (relative paths and `..` give 400; a file with unknown keys
  is not rewritten).
- `brasa serve` on 127.0.0.1 without authentication: crates/cli/src/config.rs:12 (DEFAULT_HOST),
  crates/cli/src/serve.rs:19-20 (--host flag), crates/daemon/src/lib.rs:160 (router(), no
  authentication middleware) and crates/daemon/src/lib.rs:269 (TcpListener::bind(cfg.addr)).
- sha256 per tensor and of the data block: crates/quant/src/brasa_file.rs:98 (verify),
  crates/quant/src/brasa_file.rs:101 (sha256 per tensor), crates/quant/src/brasa_file.rs:110
  (sha256 of the whole); crates/catalog/src/verify.rs:19 (verify) and crates/cli/src/models.rs:133
  (`brasa models verify`).
- Manifests validated, with no paths outside the models folder: crates/catalog/src/manifest.rs:16
  (safe_relative), crates/catalog/src/manifest.rs:34 (validate_component),
  crates/catalog/src/manifest.rs:159 (validate, including `[prebuilt]`); crates/catalog/src/pull.rs:186
  (safe_relative of the path before downloading).
- Bounded `rm`: crates/catalog/src/local.rs:88 (resolve_child), crates/catalog/src/local.rs:95
  (resolve_child_with, which rejects a base with a symlink) and crates/cli/src/rm.rs:32/51
  (resolve_child_with before remove_dir_all).
- Embedded GUI with no remote resources or telemetry: crates/daemon/src/ui.rs:7-9 (include_str!),
  crates/daemon/tests/api.rs:212-213 (test: no asset references http:// or https://);
  CLAUDE.md:23 (rule 9, no outbound telemetry).
- Malicious weights files: crates/quant/src/brasa_file.rs:64 (open validates magic, version,
  bounds and alignment), crates/quant/src/brasa_file.rs:67 (BRSA magic) and
  crates/quant/src/safetensors.rs:18 (open parses the JSON header with the declared size).
- Planner memory budget: crates/memory/src/planner.rs:83 (plan).
- Prompt injection out of scope: crates/daemon/src/openai.rs:131 (the daemon translates the request
  to brasa-core types and serves the model; it does not run tools).
- Bounded out-of-range `ctx`: crates/daemon/src/lib.rs:144 (MAX_CTX).
-->
