#!/usr/bin/env bash
# Aceptación de U6 y de la documentación de la tanda 2: verifica que cada subcomando de `brasa`
# citado en los documentos exista (corriendo `<bin> <sub> --help`) y que todo enlace local
# (relativo) de los `*.md` de la raíz, `.github/` y `docs/` apunte a un archivo que existe.
# Por defecto usa el binario release; se puede apuntar a otro con BRASA_BIN.
set -euo pipefail
cd "$(dirname "$0")/.."

BIN="${BRASA_BIN:-./target/release/brasa}"
if [ ! -x "$BIN" ]; then
  echo "no existe $BIN; compilá con: cargo build --release -p brasa-cli" >&2
  exit 1
fi

status=0
subs=$(grep -rhoE '\bbrasa [a-z][a-z-]*' README.md SECURITY.md CONTRIBUTING.md .github docs/guia 2>/dev/null \
  | awk '{print $2}' | sort -u || true)
for sub in $subs; do
  if "$BIN" "$sub" --help >/dev/null 2>&1; then
    echo "ok: brasa $sub"
  else
    echo "FALLA: brasa $sub --help" >&2
    status=1
  fi
done

echo "==> enlaces locales"
docs=$({ ls ./*.md 2>/dev/null || true; find .github docs -name '*.md' -type f 2>/dev/null || true; } | sort -u)
for f in $docs; do
  dir=$(dirname "$f")
  while IFS= read -r target; do
    case "$target" in
      http://* | https://* | mailto:* | '#'* | '') continue ;;
      /*) continue ;; # rutas absolutas: no se validan contra el repo
    esac
    target="${target%%#*}"
    [ -n "$target" ] || continue
    if [ -e "$dir/$target" ]; then
      echo "ok: $f -> $target"
    else
      echo "FALLA: enlace roto en $f: $target" >&2
      status=1
    fi
  done < <(grep -oE '\]\([^)]+\)' "$f" | sed -E 's/^\]\((.*)\)$/\1/' || true)
done

exit $status
