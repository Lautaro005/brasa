# ADR 0023 — Dependencias de desarrollo

Estado: aceptada (U2–U6)

## Contexto

Los tests del catálogo, del conversor nativo, del daemon y de la configuración necesitan
directorios temporales reales y un cliente HTTP en proceso para ejercitar el router de axum sin
levantar un servidor. Esas piezas no tienen que llegar al binario final.

## Decisión

Se agregan dos **dev-dependencies** (solo se compilan para `cargo test`, nunca se enlazan en el
binario de release):

- `tower` (feature `util`), en `brasa-daemon`: da `ServiceExt::oneshot` para mandar pedidos al
  `Router` de axum en los tests de `/ui`, `/api/status`, `/api/metrics`, `/api/plan`, `/api/agents`
  y `/api/bench`, sin abrir un puerto.
- `tempfile` 3, en `brasa-catalog`, `brasa-quant` y `brasa-daemon`: carpetas temporales que se
  limpian solas para los tests de `pull` (descarga y reanudación), de `brasa rm`
  (`local::resolve_child`), del conversor y del filtrado de reportes de `/api/bench`.

Las dependencias de **runtime** nuevas siguen siendo las documentadas por tarea: `toml` y `ureq`
en el catálogo (ADR 0020), `clap_complete` en el CLI (ADR 0022). Ninguna de ellas habla red salvo
`ureq`, que solo usa `brasa pull`.

## Consecuencias

- `Cargo.lock` incluye estas cajas, pero el binario no las usa: se pueden verificar con
  `cargo tree -e normal` frente a `cargo tree -e dev`.
- Un test futuro que necesite otra utilidad de este tipo debería sumarla acá y no como
  dependencia normal.
