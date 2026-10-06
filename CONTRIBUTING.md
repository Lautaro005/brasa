# Cómo contribuir a Brasa

Brasa es el engine de inferencia local para Apple Silicon. El proyecto está en desarrollo, antes
de 1.0: se corrige solo `main`. La arquitectura y las reglas completas están en
[CLAUDE.md](CLAUDE.md). El estado de cada fase y los resultados medidos están en los ADR
([docs/adr/](docs/adr/)) y en [docs/bench/baseline.md](docs/bench/baseline.md).

## Cómo compilar

Requisitos: macOS sobre Apple Silicon y una toolchain de Rust (la versión la fija
`rust-toolchain.toml`).

```bash
cargo build --release -p brasa-cli
./target/release/brasa --version
```

Los pesos no van en el repo (están en `.gitignore`): bajalos con `brasa pull qwen3-4b-q4` y
convertilos con `brasa convert`, o poné los pesos en `models/`.

## Antes de cada commit

```bash
./scripts/ci.sh
```

corre `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings` y
`cargo test --workspace`. Tiene que salir con 0.

## Reglas no negociables (resumen de CLAUDE.md)

1. Nada de Ollama, llama.cpp ni MLX en el camino caliente; solo como baseline en `crates/bench` y
   en `tools/`.
2. Ningún kernel Metal entra sin implementación de referencia en CPU, test de equivalencia
   numérica con tolerancia documentada y microbenchmark.
3. Correcto primero, rápido después: una optimización que cambia logits fuera de tolerancia se
   revierte.
4. Decode sin asignaciones: buffers, scratch y KV se preasignan al cargar el modelo.
5. Memoria presupuestada: si no entra, se baja el contexto o se rechaza; nunca swap silencioso.
6. Nada se afirma sin medir: no hay cifras de velocidad, memoria ni calidad en docs o README sin
   un reporte de `brasa benchmark` que las respalde.
7. Python solo offline, en `tools/`; el binario final no depende de Python.
8. La API compatible es transporte; los tipos internos son de `crates/core`.
9. Sin telemetría saliente y la GUI no carga nada de internet.

## Tests de GPU

Los tests que crean un dispositivo Metal **no** corren en la CI de GitHub (ver
[docs/adr/0024](docs/adr/0024-ci-en-github-actions.md)); hay que correrlos en una Mac:

```bash
cargo test -p brasa-kernels -- --nocapture
cargo bench -p brasa-kernels
cargo test --release -p brasa-models --test forward -- --ignored --nocapture --test-threads 1
cargo test --release -p brasa-runtime --test session -- --ignored --nocapture
```

En una Mac compartida, no corras dos trabajos de GPU a la vez: coordinalos antes de empezar y
liberá la GPU al terminar.

## Cómo proponer un ADR

Antes de una decisión de diseño no trivial (una dependencia nueva, el formato de un manifiesto, la
estructura de la GUI), agregá un ADR corto en `docs/adr/`, numerado con el siguiente número libre,
con este esqueleto:

```markdown
# ADR NNNN — Título

Estado: propuesta

## Contexto

## Decisión

## Consecuencias
```

No reescribas un ADR aceptado: si cambia la decisión, escribí uno nuevo que reemplace al anterior.

## Pull requests

- Un commit por tarea, en castellano, chico y descriptivo. Sin la línea `Claude-Session: ...`.
- `./scripts/ci.sh` en 0 antes de commitear.
- Completá la checklist de la plantilla de PR ([pull_request_template.md](.github/pull_request_template.md)).
- Un kernel nuevo trae su referencia CPU, su test de equivalencia y su microbenchmark.

## Seguridad

No abras un issue público para una vulnerabilidad: leé [SECURITY.md](SECURITY.md) y usá el reporte
privado de la pestaña Security.

## Licencia

Al contribuir aceptás que tu aporte se distribuya bajo Apache-2.0 ([LICENSE](LICENSE)).
