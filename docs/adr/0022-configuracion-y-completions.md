# ADR 0022 — Configuración de usuario y autocompletado

Estado: aceptada (U5)

## Contexto

`brasa serve` y `brasa run` repiten los mismos valores (modelo, puerto, contexto, tipo de KV) en
cada invocación. Hace falta un archivo de configuración con precedencia clara y scripts de
autocompletado para los shells.

## Decisión

**Archivo.** `~/.config/brasa/config.toml` (o `$BRASA_CONFIG`, o `$XDG_CONFIG_HOME/brasa/config.toml`):

```toml
model = "qwen3-4b-q4"
host  = "127.0.0.1"
port  = 8080
kv    = "f16"

[run]
ctx = 4096

[serve]
ctx = 16384
```

Todos los campos son opcionales: un archivo ausente no es error y deja los valores por defecto.
`run` y `serve` tienen contextos por defecto distintos (4096 y 16384), así que van en secciones
propias: `[run] ctx` y `[serve] ctx`. El resto de los valores es común. Se usa
`#[serde(deny_unknown_fields)]`, así que un error de tipeo (`contx`) falla con un mensaje que
nombra el campo en vez de ignorarse. `brasa config show` muestra el defecto real de cada
subcomando.

**Precedencia.** `flag > archivo > defecto`, resuelta por `config::pick` (y por `run_ctx`/
`serve_ctx`, que envuelven la sección correspondiente). Para que un flag ausente no pise el
archivo, los argumentos afectados (`[modelo]`, `--host`, `--port`, `--ctx`, `--kv`) son `Option`;
si ninguno aporta, se usa el defecto. `brasa ps` y `brasa connect` también leen `host` y `port`
del archivo (antes fijaban 8080), y `connect` toma el modelo y el `[serve] ctx`.
`brasa config show [--json]` imprime la configuración efectiva y, por cada valor, de dónde sale
(`flag`, `archivo` o `defecto`).

**Completions.** `brasa completions zsh|bash|fish` con `clap_complete` (dependencia nueva). El
script se genera del mismo `Command` de clap, así que no puede desincronizarse.

**Errores accionables.** Un modelo inexistente sugiere `brasa models`, `brasa pull` y
`brasa convert`; un contexto que no entra ya informa el ctx máximo que calcula el planner
(`Contexto máximo que entra: N (--ctx N)`).

**JSON.** `doctor` (ya estaba), `plan`, `models` y `ps` aceptan `--json`.

**Snapshots de ayuda.** Un test unitario compara el `--help` de cada subcomando contra
`crates/cli/tests/help/<nombre>.txt`; se regeneran con `BRASA_BLESS=1 cargo test -p brasa-cli`.
No se snapshotea el `--help` raíz porque incluye el commit del build.

## Consecuencias

- Cambiar un valor por defecto o un flag obliga a regenerar los snapshots (el test lo avisa).
- `clap_complete` es la única dependencia nueva; no arrastra red ni runtime.
