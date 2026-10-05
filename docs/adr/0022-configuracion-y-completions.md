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
ctx   = 16384
kv    = "f16"
```

Todos los campos son opcionales: un archivo ausente no es error y deja los valores por defecto.
Variables: modelo `qwen3-4b-q4`, host `127.0.0.1`, puerto `8080`, ctx `16384`, kv `f16` (con
`brasa run`, el ctx por defecto es 4096).

**Precedencia.** `flag > archivo > defecto`, resuelta por `config::pick`. Para que un flag ausente
no pise el archivo, los argumentos afectados (`[modelo]`, `--host`, `--port`, `--ctx`, `--kv`) son
`Option`; si ninguno aporta, se usa el defecto. `brasa config show [--json]` imprime la
configuración efectiva y, por cada valor, de dónde sale (`flag`, `archivo` o `defecto`).

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
