# ADR 0020 — Manifiesto de modelos y descarga

Estado: aceptada (U3)

## Contexto

`brasa` necesita bajar pesos de Hugging Face, verificarlos y convertirlos al formato nativo
(ADR 0006). Hasta ahora el harness de benchmark tenía un registro provisorio de rutas. Falta un
manifiesto por modelo con el origen fijado (repo, revisión y hashes) y un cliente que descargue de
forma reanudable y verifique.

## Decisión

**Formato.** Un manifiesto TOML por modelo (`crates/catalog/manifests/*.toml`), embebido en el
binario con `include_str!` y también legible desde `$BRASA_CATALOG`. Campos:

```toml
name            = "qwen3-4b-q4"      # id del modelo convertido
family          = "qwen3"
source_repo     = "Qwen/Qwen3-4B"    # repo de Hugging Face
source_revision = "<sha40>"          # revisión fija (commit de HF)
hf_dir          = "qwen3-4b-hf"      # destino de los safetensors (entrada de `brasa convert`)
quant           = "q4_0 g32 …"       # cuantización destino
max_context     = 40960
license         = "Apache-2.0"
[[files]]
path   = "config.json"
size   = 726
sha256 = "<64 hex>"                  # sha256 del contenido del archivo
```

Los sha256 se toman de los `oid` LFS de la API de HF para los archivos grandes y del cálculo local
para los chicos; se verificaron contra la revisión `1cfa9a72` de `Qwen/Qwen3-4B` el 2026-10-05.

**Comandos.**

- `brasa models [--verify] [--json]`: lista los `.brasa` locales con tamaño, sha256 del
  encabezado, si la estructura abre y si entran en memoria con el contexto por defecto (planner,
  `ctx 4096`, KV f16). `--verify` recalcula los sha256 (lee el archivo entero).
- `brasa models verify <modelo>`: recalcula todos los sha256 contra el `.brasa`.
- `brasa pull <modelo>`: descarga los archivos del manifiesto a `models/<hf_dir>`. Reanuda con
  `Range` sobre el `.part` (rehasheando el parcial, porque el estado de sha2 no se persiste);
  verifica el sha256 y, si no coincide, borra el parcial y falla con el hash esperado y el
  obtenido. Un archivo ya presente y correcto no se vuelve a bajar. `--dry-run` y `--json`
  disponibles; `--endpoint` permite apuntar a un espejo.
- `brasa rm <modelo>`: borra la carpeta, pidiendo confirmación salvo `--yes`.

**Cliente HTTP.** Se agrega `ureq` 2 (blocking, TLS por rustls) en `brasa-catalog`. Es la
dependencia que habla HTTPS con Hugging Face; el resto del engine sigue sin red. Junto con `toml`
para el manifiesto, son las dos dependencias nuevas de esta tarea.

## Consecuencias

- El manifiesto va embebido, así que el binario conoce su modelo de fábrica sin archivos externos;
  `$BRASA_CATALOG` permite agregar manifiestos sin recompilar.
- `pull` no convierte: deja los safetensors y sugiere `brasa convert` (U4).
- Los tests de `pull` usan un servidor local chico (sin red) que soporta `Range`, y cubren la
  descarga, la reanudación desde un `.part` y el fallo limpio con un hash alterado.
