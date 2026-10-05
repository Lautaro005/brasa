#!/usr/bin/env bash
# Genera el GGUF Q4_0 de Qwen3-4B para el baseline de llama.cpp (ver ADR 0002).
# Usa el conversor de llama.cpp fijado al mismo commit que el binario instalado.
# Uso: tools/make_gguf.sh [dir_hf] [dir_salida]
set -euo pipefail
cd "$(dirname "$0")/.."

hf_dir="${1:-models/qwen3-4b-hf}"
out_dir="${2:-models}"
llama_commit="$(llama-cli --version 2>&1 | sed -n 's/.*commit \([0-9a-f]*\).*/\1/p')"
[[ -n "$llama_commit" ]] || { echo "no se pudo leer el commit de llama.cpp" >&2; exit 1; }

conv_dir="models/.tools/llama.cpp-$llama_commit"
if [[ ! -f "$conv_dir/convert_hf_to_gguf.py" ]]; then
    # Checkout parcial: el conversor importa los paquetes conversion/ y gguf-py/ del repo.
    rm -rf "$conv_dir"
    git clone -q --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp "$conv_dir"
    git -C "$conv_dir" sparse-checkout set --no-cone /convert_hf_to_gguf.py /conversion/ /gguf-py/
    git -C "$conv_dir" checkout -q "$llama_commit"
fi

f16="$out_dir/qwen3-4b-f16.gguf"
q4="$out_dir/qwen3-4b-q4_0.gguf"
[[ -f "$f16" ]] || .venv/bin/python "$conv_dir/convert_hf_to_gguf.py" "$hf_dir" --outtype f16 --outfile "$f16"
[[ -f "$q4" ]] || llama-quantize "$f16" "$q4" Q4_0
shasum -a 256 "$q4"
