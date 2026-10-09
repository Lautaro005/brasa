# ADR 0035 — Paquete del código e instalación en un comando

Estado: aceptada (2026-10-09)

## Contexto

Para usar Brasa había que clonar el repo y compilarlo a mano, y no había forma de pasar la app a
otra Mac sin Git ni de bajarla en una línea. El repo pesa 41 GB en disco de trabajo, pero casi todo
es regenerable o local: pesos (`models/`, 2,2–7,5 GiB por variante), build (`target/`), entorno
Python (`.venv/`), worktrees de subagentes y notas de sesión. El código y la documentación juntos
ocupan unos 28 MB comprimidos.

Los pesos no entran en el paquete: `brasa pull` ya los baja de Hugging Face con verificación por
hash (ADR 0031) y con control de espacio (ADR 0034).

## Decisión

- **`scripts/package.sh`** genera `dist/brasa-<fecha>-<commit>.tar.gz` y su `.sha256`. Toma los
  archivos que git no ignora del árbol de trabajo, así que incluye cambios sin commitear (en ese
  caso el nombre lleva `-wip`). Excluye `RESUMEN-*.md`, que son notas locales de sesión. `dist/`
  queda en `.gitignore`.
- **`scripts/install.sh`** compila `brasa` en release y copia el binario a `BRASA_BIN_DIR`
  (por defecto `~/.local/bin`). Reglas:
  - exige macOS arm64 (Metal);
  - no instala Rust: si falta `cargo`, sale con el enlace a rustup;
  - si se ejecuta dentro de un checkout o paquete, lo usa tal cual;
  - si no, descarga `BRASA_ARCHIVE_URL` (verifica `BRASA_ARCHIVE_SHA256` si se define) o clona el
    repo en `BRASA_DIR` (por defecto `~/brasa`);
  - no descarga pesos ni modifica el PATH: imprime los siguientes pasos.
- La instalación en una línea (`curl ... | BRASA_ARCHIVE_URL=... bash`) necesita una URL pública
  del paquete. Esa publicación (GitHub Release u otro host) no se hizo: requiere autorización
  explícita.

## Verificación

Medido en M1 Pro 16 GB (macOS 27.0):

- El script entró por stdin (`cat scripts/install.sh | bash`) desde fuera del repo, con
  `BRASA_ARCHIVE_URL=file://…` y SHA-256: descargó, verificó, extrajo, compiló y copió el binario.
  La compilación tardó 34,8 s reutilizando las dependencias ya compiladas (recompiló los 9 crates
  del workspace). `exit=0`. Esta corrida fue sobre el paquete previo a agregar este ADR y el bit de
  ejecución de `install.sh`; la diferencia es solo documentación y permisos, no código.
- El paquete final (`brasa-20261009-e6eba0a-wip.tar.gz`, 28 MB, 587 archivos) no contiene
  `models/`, `target/`, `.venv/`, `.claude/`, `dist/`, planes QWEN ni `RESUMEN-*.md`.
- Limitación medida: sin `.git` (en un paquete), `brasa --version` muestra `(desconocido)` como
  commit, porque `crates/cli/build.rs` lee el commit de git y no lo encuentra.

## Consecuencias

- Un paquete con cambios sin commitear incluye código que no está en `main` (hoy, el WIP de
  TurboQuant, ADR 0033, opt-in). Para distribuir solo lo commiteado, hay que commitear antes.
- El paquete no contiene binarios: cada máquina compila. Compilar desde cero tarda varios minutos
  (la primera vez).
- El instalador no modifica `~/.zshrc`: si `~/.local/bin` no está en PATH, lo avisa.

## Alternativas consideradas

- **Binario precompilado y firmado**: requiere Developer ID, notarización y un runner macOS en CI.
  No se hizo: el proyecto no tiene esa infraestructura todavía.
- **Instalar Rust con rustup desde el script**: es ejecutar un instalador remoto sin que el usuario
  lo pida. Se prefirió que el script falle con instrucciones.
- **Solo `git clone` + `cargo build`** (lo que había): sigue funcionando y está en el README.
