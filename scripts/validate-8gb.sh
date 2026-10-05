#!/usr/bin/env bash
# Validación en la Mac M2 8 GB. Corre los criterios de aceptación que requieren esa máquina y
# deja la evidencia en docs/bench/m2-8gb/. Uso: ./scripts/validate-8gb.sh
# Cada tarea que necesite validación en 8 GB agrega su paso acá.
set -euo pipefail
cd "$(dirname "$0")/.."

out="docs/bench/m2-8gb"
mkdir -p "$out"
commit="$(git rev-parse --short HEAD)"
mem=$(sysctl -n hw.memsize)
if [[ "$mem" != "8589934592" ]]; then
    echo "Aviso: esta máquina tiene $((mem >> 30)) GiB, no 8 GiB. Se sigue igual." >&2
fi

echo "==> build release ($commit)"
cargo build --release -p brasa-cli

echo "==> T0.2 brasa doctor"
./target/release/brasa doctor | tee "$out/doctor.txt"
./target/release/brasa doctor --json > "$out/doctor.json"
cargo test -p brasa-cli --test doctor 2>&1 | tee "$out/doctor-test.log"
grep -q "test result: ok" "$out/doctor-test.log"

{
    echo "commit: $commit"
    echo "fecha: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    sw_vers
} > "$out/meta.txt"

echo
echo "Validación 8 GB OK. Evidencia en $out/ (commitear esa carpeta o pegar su contenido)."
