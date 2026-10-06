#!/usr/bin/env bash
# Aceptación de U6: verifica que cada subcomando de `brasa` citado en README.md y docs/guia/
# exista, corriendo `<bin> <sub> --help`. Por defecto usa el binario release; se puede apuntar a
# otro con BRASA_BIN.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="${BRASA_BIN:-./target/release/brasa}"
if [ ! -x "$BIN" ]; then
  echo "no existe $BIN; compilá con: cargo build --release -p brasa-cli" >&2
  exit 1
fi

status=0
subs=$(grep -rhoE '\bbrasa [a-z][a-z-]*' README.md docs/guia | awk '{print $2}' | sort -u)
for sub in $subs; do
  if "$BIN" "$sub" --help >/dev/null 2>&1; then
    echo "ok: brasa $sub"
  else
    echo "FALLA: brasa $sub --help" >&2
    status=1
  fi
done
exit $status
