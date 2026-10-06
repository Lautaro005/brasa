# ADR 0024 — CI en GitHub Actions (sin GPU)

Estado: aceptada (V2)

## Contexto

Los runners de GitHub son máquinas virtuales; los de macOS corren en centros de datos de Azure.
Verificado el 2026-10-06 contra la documentación vigente de runners alojados por GitHub:

- Estándar **arm64 (Apple Silicon)**: `macos-latest`, `macos-14`, `macos-15`, `macos-26`
  (M1, 3 CPU, 7 GB de RAM).
- Estándar **Intel**: `macos-15-intel`, `macos-26-intel`.

La documentación no garantiza acceso a una GPU usable desde el runner (habla de "larger runners"
con GPU para otras plataformas, no de Metal en macOS). Bajo la consigna de "si no queda claro,
asumir que no", se toma que **no hay GPU Metal utilizable** en la CI de GitHub.

Los tests de kernels, modelos, runtime, memoria, tuner, bench, daemon y CLI crean buffers y
pipelines de Metal: no son confiables en el runner y además no medirían nada útil (la velocidad de
una VM no es la del objetivo).

## Decisión

`.github/workflows/ci.yml` corre en `macos-15` (arm64 estándar; se fija por etiqueta explícita y no
`macos-latest`, que puede cambiar de imagen):

1. `cargo fmt --all --check`
2. `cargo clippy --workspace --all-targets -- -D warnings`
3. `cargo build --workspace`
4. `cargo test` de los crates que **no** dependen de `brasa-metal`: `brasa-core`, `brasa-quant`,
   `brasa-tokenizer` y `brasa-catalog`.

La única action es `actions/checkout`, fijada por SHA (con el tag en un comentario);
`permissions: contents: read`, sin secretos. `.github/dependabot.yml` pide actualizaciones
semanales de `cargo` y `github-actions`.

## Consecuencias

- Un pull request puede pasar la CI de GitHub y aun así romper los tests de GPU: `./scripts/ci.sh`
  (CI local, con GPU) sigue siendo la puerta antes de commitear.
- La CI de GitHub es la red de seguridad para lo que no toca Metal (formato, lints, compilación y
  los crates de datos), y avisa si alguien rompe la compilación en un `macos` arm64.
- Cuando haya una Mac en la nube con Metal confiable, el workflow puede crecer a un job con GPU;
  hasta entonces esa parte queda fuera.
