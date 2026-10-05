#!/usr/bin/env bash
# T0.5: baselines de llama.cpp y MLX-LM en 2K, 8K y 16K (más llama.cpp con KV Q8 en 16K, el
# perfil de agente). Requiere los pesos de models/ (ver CLAUDE.md, sección Comandos).
# Uso: ./scripts/run-baselines.sh [carpeta_salida]
# Un contexto que falla (por ejemplo, por memoria en 8 GB) se registra y se sigue con el resto.
set -uo pipefail
cd "$(dirname "$0")/.."

out_args=()
[[ $# -ge 1 ]] && out_args=(--out-dir "$1")
cargo build --release -q -p brasa-cli || exit 1
brasa=./target/release/brasa

$brasa doctor | sed -n '/Memoria del sistema/,$p'
failures=0
run() {
    echo
    echo "==> brasa benchmark $*"
    if ! $brasa benchmark "$@" "${out_args[@]}"; then
        echo "FALLÓ: brasa benchmark $*"
        failures=$((failures + 1))
    fi
}

for ctx in 2048 8192 16384; do
    run --baseline llama.cpp --ctx "$ctx"
    run --baseline mlx-lm --ctx "$ctx"
done
run --baseline llama.cpp --ctx 16384 --label kv-q8 --engine-arg=-ctk --engine-arg=q8_0 \
    --engine-arg=-ctv --engine-arg=q8_0

echo
echo "baselines terminados con $failures fallas"
exit "$failures"
