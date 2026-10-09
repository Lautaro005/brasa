#!/usr/bin/env bash
# Instala el CLI `brasa` en macOS Apple Silicon: compila en release y copia el binario a BRASA_BIN_DIR.
#
#   Desde un checkout o un paquete ya extraído:  ./scripts/install.sh
#   En una línea, sin tener el código:           curl -fsSL <URL-de-este-script> | bash
#     - con BRASA_ARCHIVE_URL=<URL del .tar.gz> descarga ese paquete;
#     - sin esa variable clona https://github.com/Lautaro005/brasa.git.
#
# Variables opcionales:
#   BRASA_DIR             carpeta donde queda el código (por defecto ~/brasa)
#   BRASA_ARCHIVE_URL     paquete .tar.gz generado con scripts/package.sh
#   BRASA_ARCHIVE_SHA256  SHA-256 esperado del paquete (si se define, se verifica)
#   BRASA_BIN_DIR         carpeta del binario (por defecto ~/.local/bin)
set -euo pipefail

die() { echo "error: $*" >&2; exit 1; }

[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] \
    || die "Brasa necesita macOS en Apple Silicon (Metal)."
command -v cargo >/dev/null \
    || die "no hay cargo. Instala Rust con rustup (https://rustup.rs) y vuelve a correr este script."

# Si el script vive dentro de un checkout o paquete, lo usamos tal cual.
root=""
if [[ -n "${BASH_SOURCE[0]:-}" && -f "${BASH_SOURCE[0]}" ]]; then
    candidate="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    [[ -f "$candidate/crates/cli/Cargo.toml" ]] && root="$candidate"
fi

if [[ -z "$root" ]]; then
    dest="${BRASA_DIR:-$HOME/brasa}"
    [[ -e "$dest" ]] && die "$dest ya existe. Define BRASA_DIR con otra carpeta o bórrala."
    if [[ -n "${BRASA_ARCHIVE_URL:-}" ]]; then
        tmp="$(mktemp -d)"
        trap 'rm -rf "$tmp"' EXIT
        echo "==> descargando $BRASA_ARCHIVE_URL"
        curl -fL --progress-bar -o "$tmp/brasa.tar.gz" "$BRASA_ARCHIVE_URL"
        if [[ -n "${BRASA_ARCHIVE_SHA256:-}" ]]; then
            echo "$BRASA_ARCHIVE_SHA256  brasa.tar.gz" > "$tmp/brasa.sha256"
            (cd "$tmp" && shasum -a 256 -c brasa.sha256 >/dev/null) \
                || die "el SHA-256 del paquete no coincide con BRASA_ARCHIVE_SHA256"
            echo "==> SHA-256 verificado"
        fi
        mkdir -p "$dest"
        tar -xzf "$tmp/brasa.tar.gz" -C "$dest" --strip-components=1
    else
        command -v git >/dev/null || die "hace falta git para clonar, o define BRASA_ARCHIVE_URL."
        echo "==> clonando https://github.com/Lautaro005/brasa.git en $dest"
        git clone --depth 1 https://github.com/Lautaro005/brasa.git "$dest"
    fi
    root="$dest"
fi

echo "==> compilando brasa (release; la primera vez tarda varios minutos)"
(cd "$root" && cargo build --release -p brasa-cli)

target_dir="${CARGO_TARGET_DIR:-$root/target}"
bin_dir="${BRASA_BIN_DIR:-$HOME/.local/bin}"
mkdir -p "$bin_dir"
install -m 755 "$target_dir/release/brasa" "$bin_dir/brasa"
echo "==> instalado: $bin_dir/brasa"
"$bin_dir/brasa" --version || true

case ":$PATH:" in
    *":$bin_dir:"*) ;;
    *) echo "aviso: $bin_dir no está en PATH. Agrégalo en tu ~/.zshrc:  export PATH=\"$bin_dir:\$PATH\"" ;;
esac

cat <<EOF

Siguientes pasos:
  brasa doctor              # chip, memoria, Metal
  brasa pull qwen3-4b-q4    # baja los pesos (~2,2 GiB) desde Hugging Face
  brasa run qwen3-4b-q4     # chat en la terminal
EOF
