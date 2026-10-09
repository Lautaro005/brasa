#!/usr/bin/env bash
# Empaqueta el código de Brasa en dist/ como un .tar.gz con su SHA-256.
# Incluye los archivos del árbol de trabajo que git no ignora (también los cambios sin commitear);
# deja afuera pesos, build, entorno Python, worktrees y las notas locales de sesión (RESUMEN-*.md).
# Uso: scripts/package.sh
set -euo pipefail
cd "$(dirname "$0")/.."

commit="$(git rev-parse --short=7 HEAD)"
fecha="$(date -u +%Y%m%d)"
name="brasa-$fecha-$commit"
if [[ -n "$(git status --porcelain)" ]]; then
    name="$name-wip"
    echo "aviso: hay cambios sin commitear; el paquete los incluye (sufijo -wip)" >&2
fi

mkdir -p dist
out="dist/$name.tar.gz"
# -z: nombres separados por NUL, así los espacios no rompen la lista.
git ls-files -z -co --exclude-standard \
    | grep -zv '^RESUMEN-.*\.md$' \
    | tar --null -T - -czf "$out" -s ",^,$name/," --no-mac-metadata

(cd dist && shasum -a 256 "$name.tar.gz" > "$name.tar.gz.sha256")

echo "paquete: $out ($(du -h "$out" | cut -f1))"
echo "sha256:  $(cut -d' ' -f1 "$out.sha256")"
echo "archivos: $(tar -tzf "$out" | wc -l | tr -d ' ')"
