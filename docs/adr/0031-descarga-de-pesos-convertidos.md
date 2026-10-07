# ADR 0031 — Descarga de pesos convertidos y carpeta de modelos elegible

Estado: aceptada (2026-10-06)

## Contexto

Hasta ahora (ADR 0020) el manifiesto apunta a los safetensors de `Qwen/Qwen3-4B` y `brasa pull`
los baja (≈7,6 GiB) para que el usuario corra `brasa convert`. Los pesos ya convertidos se van a
publicar en un repo público de Hugging Face con todas las cuantizaciones:

```text
lautiss/brasa-v0.01-base
  README.md, LICENSE
  qwen3-4b-q4/{model.brasa, tokenizer.json, tokenizer_config.json}      (q6_0 embeddings, ADR 0012)
  qwen3-4b-q4-e8/{model.brasa, tokenizer.json, tokenizer_config.json}   (q8_0 embeddings)
```

Además los modelos salían solo de `./models` o `$BRASA_MODELS`: un binario instalado fuera del
checkout no tenía una carpeta propia, y la GUI no podía bajar modelos ni elegir dónde guardarlos.

## Decisión

**Manifiesto.** Se agrega una sección opcional `[prebuilt]` (repo, revisión, subcarpeta y archivos
con `size` y `sha256`) y el campo `convertible` (por defecto `true`):

```toml
[prebuilt]
repo = "lautiss/brasa-v0.01-base"
revision = "c8b159a2ab1ad1163689cad5931add471cc3d882"   # commit del repo publicado
subdir = "qwen3-4b-q4"
[[prebuilt.files]]         # tokenizers primero, model.brasa último
path = "model.brasa"
size = 2362228736
sha256 = "98a9…"
```

- La URL es `<endpoint>/<repo>/resolve/<revision>/<subdir>/<path>`. `repo` tiene que ser
  `dueño/nombre`, `revision` un segmento sin `/` (commit o rama), `subdir` y `path` rutas relativas
  sin `..` (validado al parsear).
- Los sha256 y tamaños se calcularon sobre `models/qwen3-4b-q4/` y `models/qwen3-4b-q4-e8/` el
  2026-10-06. El repo se publicó el 2026-10-07 y la revisión quedó fijada en los dos manifiestos al
  commit `c8b159a2`; los sha256 coinciden con los `oid` LFS de la API de Hugging Face. Con una
  revisión sin fijar (una rama), `brasa pull` avisa (`Prebuilt::is_pinned`).
- `model.brasa` va último: `local::scan` reconoce un modelo por `model.brasa`, que aparece (por el
  `rename` del `.part`) solo cuando todo lo demás ya está.
- Nuevo manifiesto `qwen3-4b-q4-e8` (q8_0 embeddings). `brasa convert` solo produce q6_0, así que
  lleva `convertible = false`: se baja ya convertido.
- El `quant` del manifiesto `qwen3-4b-q4` decía q8_0 embeddings; se corrige a q6_0 (lo que produce
  el conversor desde ADR 0012).

**`brasa pull <modelo>`** baja los pesos convertidos a `<carpeta de modelos>/<modelo>/`, con la
misma reanudación por `Range` y verificación de sha256 de ADR 0020. El camino viejo queda detrás de
`--desde-fuente` (safetensors a `<carpeta de modelos>/<hf_dir>` + `brasa convert`); para un
manifiesto no convertible falla con un mensaje claro. Las descargas aceptan una bandera de
cancelación que corta entre lecturas y deja el `.part` para reanudar.

**Carpeta de modelos.** Precedencia: flag `--dir` (en `models`, `rm`, `pull`) > `$BRASA_MODELS` >
`models_dir` del archivo de configuración (ADR 0022) > `./models` si existe (checkout de desarrollo)
> `~/Library/Application Support/brasa/models`. La resuelve `brasa_catalog::dirs` y la usan `serve`,
`run`, `pull`, `rm`, `models` y `/api/models`; `brasa config show` la muestra con su origen
(`flag`, `env`, `archivo`, `checkout`, `defecto`). `models_dir` en el archivo tiene que ser absoluta
(o empezar con `~/`): una relativa dependería del cwd de cada comando.

Predeterminado en `Application Support` y no en `~/.brasa` ni `~/.cache`: es la carpeta de datos de
aplicación de macOS (la que usaría una app empaquetada) y los pesos no son un caché regenerable
barato (2,2–2,3 GiB cada uno). Consecuencia: entra en los respaldos de Time Machine; quien no lo
quiera elige otra carpeta.

**Guardar `models_dir` sin romper el archivo.** `toml` no conserva comentarios al reescribir, y el
archivo es del usuario. `dirs::save_models_dir` edita una sola línea: reemplaza
`models_dir = …` si está en el primer nivel o la agrega antes de la primera sección. Se niega (sin
tocar nada) si el archivo no es TOML válido, si tiene claves que Brasa no conoce, si `models_dir`
aparece más de una vez o no está en una línea simple. Después de editar vuelve a parsear y exige
que la tabla resultante sea la original con solo `models_dir` cambiado; escribe a un `.tmp` y hace
`rename`. Comentarios y orden se conservan.

**API del daemon.**

| Método y ruta | Qué hace |
|---|---|
| `GET /api/models` | suma `dir`, `dir_source`, `dir_source_label`, `free_bytes` (statvfs) y, por entrada del catálogo, `prebuilt`, `pinned` y `partial_bytes` |
| `POST /api/models/pull {name}` | 202 y arranca la descarga en un hilo; 404 si no está en el catálogo, 400 sin `[prebuilt]`, 409 si hay otra en curso o ya está en disco, 507 si no hay espacio |
| `GET /api/models/pull` | `state` (`idle`, `running`, `done`, `error`, `cancelled`), bytes hechos y totales, por archivo, `error` |
| `POST /api/models/pull/cancel`, `DELETE /api/models/pull` | cancela; 409 si no hay descarga |
| `POST /api/models/dir {path}` | valida (absoluta, sin `..`), crea, prueba escritura, guarda en el archivo y aplica; 400/409 con el motivo; 409 durante una descarga |
| `POST /api/models/dir/choose` | `osascript` con `choose folder`; devuelve `{path}` o `{cancelled: true}` sin aplicar |
| `POST /api/models/dir/open` | `open <carpeta de modelos>` en Finder |

- **Solo URLs del manifiesto.** Del pedido de `pull` se usa solo `name`, que tiene que coincidir con
  un manifiesto del catálogo; repo, revisión, rutas y hashes salen del manifiesto, y el endpoint de
  la configuración del servidor (Hugging Face; `BRASA_HF_ENDPOINT` permite un espejo o un servidor
  de prueba, como `brasa pull --endpoint`).
- **Una descarga a la vez**, con el estado en memoria del daemon. El destino se fija al empezar; por
  eso no se puede cambiar la carpeta mientras baja.
- `choose` y `open` no reciben rutas del pedido: el selector parte de la carpeta actual (pasada como
  argumento de `osascript`, no interpolada en el script) y `open` abre la carpeta efectiva. Las dos
  acciones se inyectan (`models_admin::Desktop`) para que los tests no abran Finder ni diálogos.
- Todos los `POST`/`DELETE` nuevos pasan por el middleware de origen (`origin.rs`): una página web
  ajena recibe 403. Los tests lo verifican para cada ruta con un escritorio que entra en pánico si
  se lo llama.

**GUI (pantalla Modelos).** Fila "Carpeta de modelos" con la ruta, su origen y el espacio libre, y
los botones **Cambiar…** (selector nativo y luego `POST /api/models/dir`) y **Abrir carpeta**. En
"Catálogo sin descargar", un botón **Descargar** (o **Reanudar** si hay `.part`) por modelo con
`[prebuilt]`; mientras baja, la fila muestra la etiqueta `live` y debajo una fila de progreso con
una barra en `--ember` (trabajo activo, regla Solo Arde Lo Activo), bytes, porcentaje y archivo en
curso de `GET /api/models/pull`, y **Cancelar**. Al terminar se recargan las tablas; los errores se
muestran con el texto del daemon.

Verificado el 2026-10-06 en M1 Pro 16 GB con `brasa serve` real (HOME temporal) contra un servidor
local que imita el layout de Hugging Face a 60 MiB/s con los archivos de `models/`: descarga,
cancelación, reanudación desde el `.part` (verificada por sha256), error de red legible y cambio de
carpeta por la API. `brasa pull qwen3-4b-q4 --endpoint <ese servidor>` verificó los sha256 del
manifiesto contra los archivos reales. El selector nativo (`choose folder`) y `open` no se
dispararon en esa prueba para no abrir ventanas; en los tests van inyectados.

## Consecuencias

- Instalar un modelo pasa de ≈7,6 GiB de safetensors más una conversión a 2,2–2,3 GiB ya
  convertidos, verificados por sha256.
- Descarga real verificada el 2026-10-07: `brasa pull qwen3-4b-q4` bajó los 2,21 GiB de
  huggingface.co en 1 min 48 s a una carpeta temporal y `brasa models verify` dio ok.
- Un `serve` arrancado fuera del checkout ya no busca en `./models`, sino en `Application Support`
  (o lo que diga la configuración); en el checkout no cambia nada porque `./models` existe.
- `brasa-catalog` suma `libc` (ya estaba en el workspace) para `statvfs`.
