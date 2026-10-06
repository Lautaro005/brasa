# Trabajo delegado a Qwen Code (DeepSeek), tanda 2

Segunda tanda de tareas en paralelo con Claude. Las reglas de `QWEN-DEEPSEEK.md` siguen vigentes
(dónde trabajar, qué podés tocar, reglas de CLAUDE.md y coordinación de GPU). Si algo de acá
contradice esas reglas, mandan ellas, salvo lo que se dice en "Cambios respecto de la tanda 1".

Mientras tanto, Claude cierra la fase 3 (T3.6) en `brasa/`, rama `fase-3`: benchmarks válidos a
2K, 8K y 16K, más la demo de Claude Code. **Esos benchmarks necesitan la GPU libre.**

## Cambios respecto de la tanda 1

- **Ya no hay que mergear desde `fase-3`:** la rama `ui` quedó integrada en `main` (PR #3, `a036895`).
  Empezá con V0: traé `origin/main` a `ui`.
- **ADR:** numerá desde **0024**; 0020 a 0023 ya existen.
- **Marcador de GPU también para `ci.sh`.** `./scripts/ci.sh` corre los tests de kernels, que usan
  la GPU. Mientras corra `ci.sh` o cualquier cosa que use GPU, creá
  `/Users/lauti/Desktop/test/.brasa-gpu-en-uso`, con una línea que diga qué corrés y desde cuándo.
  Borralo al terminar, aunque falle.
- **Prohibido lo que publica o cambia cosas fuera del repo.**
  - Nada de push, PR, publicar la landing ni cambiar la configuración del repo en GitHub.
  - Si una tarea necesita algo de eso (habilitar private vulnerability reporting, GitHub Pages),
    anotalo en "Pedidos al usuario" al final y seguí.
- **Zonas nuevas permitidas:** `.github/`, `site/`, `SECURITY.md` y `CONTRIBUTING.md`.

## Tareas, en orden

### V0. Ponerse al día con `main`

- `git fetch origin && git merge origin/main` en la rama `ui`.
  - No debería haber conflictos: `main` contiene todo lo de `ui`.
  - Si los hay, quedate con la versión de `main` y anotalo en "Estado".
- **Aceptación:** `git log origin/main..ui` muestra solo el merge, y `./scripts/ci.sh` sale 0 (con
  el marcador de GPU mientras corre).

### V1. Política de seguridad para GitHub (`SECURITY.md`)

Verificá el formato y la ubicación contra la documentación vigente de GitHub, no de memoria. Se
busca en la raíz, en `.github/` o en `docs/`. Elegí la raíz y justificalo en el commit.

Contenido, en castellano con un resumen corto en inglés al principio:

- **Versiones soportadas:** el proyecto está en desarrollo, pre-1.0. Se corrige solo `main`.
- **Cómo reportar:**
  - Por el reporte privado de vulnerabilidades de GitHub ("Report a vulnerability" en la pestaña
    Security).
  - No pongas ningún email personal. Si hace falta un contacto alternativo, dejá un `TODO` en
    "Pedidos al usuario".
  - Qué incluir en el reporte y en qué plazo se responde: proponé plazos razonables y marcalos
    como compromiso de buena fe, no como SLA.
- **Modelo de amenazas del proyecto.** Tiene que ser verdadero: verificalo en el código y citá
  archivos.
  - `brasa serve` escucha en `127.0.0.1` por defecto y no tiene autenticación. Exponerlo en otra
    interfaz es decisión del usuario y queda fuera de alcance salvo que el default sea inseguro.
  - Integridad de pesos: sha256 por tensor y del bloque de datos (`brasa models verify`).
  - Catálogo: manifiestos validados, sin rutas fuera de `models/`, y `rm` acotado (R1 y R2).
  - GUI embebida: sin recursos remotos y sin telemetría (regla 9 de CLAUDE.md).
  - Archivos de pesos maliciosos (`.brasa`/safetensors con encabezados manipulados): dentro de
    alcance.
- **Qué no es una vulnerabilidad:** calidad de las respuestas del modelo, prompt injection
  hacia un agente que el usuario conectó, y consumo de memoria dentro de lo que el planner
  declara.
- **Aceptación:**
  - `SECURITY.md` en la raíz, con enlaces que existen (`scripts/check-docs.sh` los valida; si no
    cubre la raíz, extendelo).
  - Cada afirmación del modelo de amenazas tiene un archivo:línea que la respalda en un comentario
    HTML al final del archivo.

### V2. CI en GitHub Actions (sin GPU)

Los runners de GitHub no sirven para los tests de Metal. Verificá en la documentación vigente qué
runners macOS arm64 existen y si tienen GPU utilizable; si no queda claro, asumí que no.

- **`.github/workflows/ci.yml`:** `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D
  warnings` y `cargo build --workspace`.
  - Agregá también `cargo test` de los crates que no necesitan GPU. Probá en local cuáles pasan
    sin dispositivo Metal; si no hay forma de probarlo, limitate a los que no dependen de
    `brasa-metal`.
  - Las versiones de las actions van fijadas por SHA, con el tag en un comentario.
  - `permissions: contents: read`, sin secretos.
- Sumá **Dependabot** (`.github/dependabot.yml`) para `cargo` y `github-actions`, con frecuencia
  semanal.
- ADR 0024 corto: qué corre en GitHub, qué no y por qué (GPU).
- **Aceptación:**
  - el YAML pasa `actionlint` si lo tenés (si no, validalo con un parser YAML desde `.venv`);
  - `ci.sh` sigue en 0;
  - nada de push: Claude o el usuario lo suben.

### V3. Plantillas y guía de contribución

- **`.github/ISSUE_TEMPLATE/`:**
  - bug, con `brasa doctor --json`, comando, salida y versión;
  - pedido de función;
  - y `config.yml`, que manda los temas de seguridad a `SECURITY.md`.
- **`.github/pull_request_template.md`:** una checklist de CLAUDE.md:
  - `ci.sh` en 0;
  - un kernel nuevo trae referencia CPU, test de equivalencia y microbenchmark;
  - sin cifras sin reporte.
- **`CONTRIBUTING.md`:** cómo compilar, las reglas no negociables resumidas (enlazando
  CLAUDE.md), cómo correr los tests de GPU y cómo proponer un ADR.
- **Aceptación:** los enlaces pasan `check-docs.sh` y `ci.sh` sale 0.

### V4. Landing page con `/impeccable`

Usá el skill `/impeccable` para el diseño y seguí su flujo completo: dirección, maqueta y build.

- **Ubicación:** `site/`, estático (HTML y CSS, JS mínimo), autocontenido.
  - Sin CDN, sin fuentes remotas, sin analytics. Las fuentes y las imágenes van dentro de `site/`.
  - Que se pueda abrir con doble clic en `site/index.html`.
- **Público:** desarrolladores con Mac Apple Silicon que quieren correr agentes de código (Claude
  Code, Codex, Cline, OpenCode) contra un modelo local.
- **Contenido, todo verificable en el repo:**
  - qué es Brasa: engine propio en Rust + Metal, sin llama.cpp, MLX ni Ollama en el camino caliente;
  - API compatible con OpenAI y Anthropic;
  - `brasa connect`;
  - planner de memoria que no hace swap silencioso;
  - GUI;
  - catálogo y conversión nativa;
  - instalación desde el código (los comandos de `README.md`);
  - capturas reales de `docs/gui/*.jpg`;
  - enlaces a GitHub, `SECURITY.md` y la licencia Apache-2.0.
- **Cifras: regla 6 de CLAUDE.md, sin excepciones.**
  - No pongas velocidades ni comparaciones con llama.cpp o MLX.
  - Dejá una sección "Rendimiento" que diga que las mediciones están en `docs/bench/baseline.md`
    y enlace ahí.
  - Claude va a cerrar T3.6 y, si los números lo respaldan, después se agrega una tabla citando
    el reporte.
- **Accesibilidad básica:** contraste AA, navegación con teclado, `alt` en las imágenes.
  Responsive hasta 360 px de ancho.
- **Publicación (GitHub Pages u otro):** no la hagas. Dejá en `site/README.md` cómo publicarla y
  anotalo en "Pedidos al usuario".
- **Aceptación:**
  - `site/index.html` abre sin red: verificalo cortando la red o revisando que no haya ningún
    `http(s)://` salvo los enlaces a GitHub;
  - no tiene ninguna cifra de rendimiento;
  - incluye las capturas y pasa un chequeo de enlaces locales.

### V5. Model Manager por API (fase 4, parte del daemon)

PLAN.md, fase 4: `load`, `idle`, `pause`, `resume` y `stop`. Escribí antes el ADR 0025 con la
semántica de cada operación.

- **Propuesta para discutir en el ADR:**
  - `idle` libera los pesos y la KV y deja el daemon vivo;
  - `load` vuelve a cargar con el plan de memoria;
  - `pause`/`resume` frenan la cola sin descargar;
  - `stop` cierra el daemon de forma ordenada.
- **Solo en `crates/daemon/` y `crates/cli/` (`brasa model load|idle|…`).**
  - El modelo se descarga soltando la `Session`, desde el hilo del engine; no hace falta tocar
    `brasa-runtime`.
  - Si hace falta, anotalo en "Pedidos a Claude".
- `/v1/*` no cambia de comportamiento con el modelo cargado.
  - Con el modelo en `idle`, un pedido a `/v1/*` lo vuelve a cargar o devuelve un error claro en el
    formato del cliente; elegí uno en el ADR.
  - `/api/status` informa el estado.
- **Aceptación:**
  - tests con el engine simulado para cada transición y para un pedido a `/v1/*` en cada estado;
  - `ci.sh` en 0;
  - `brasa ps` muestra el estado;
  - con el modelo real, como máximo una prueba corta con el marcador de GPU: `idle` baja la huella
    del proceso, medida con `brasa ps`.

### V6. Codex editando por shell (pendiente de la fase 2)

Qwen3-4B no genera parches `apply_patch` válidos (`docs/demos/`, fase 2). Investigá en la
documentación vigente de Codex qué configuración permite que el modelo edite archivos por comandos
de shell en lugar de `apply_patch`, por ejemplo en el catálogo de modelos de
`model_catalog_json` o en las opciones de herramientas. Verificá la versión instalada con
`codex --version`.

- Implementalo en `brasa connect codex` y documentalo en `docs/guia/agentes.md`.
- **Prueba real:** `CODEX_HOME` temporal, nunca el del usuario, y `brasa serve` con el marcador de
  GPU.
  - Hacela **solo si el marcador de Claude no está y no hay un `brasa benchmark` corriendo**:
    chequealo con `pgrep -fl "brasa benchmark"`.
  - La tarea es la misma demo que Claude Code: arreglar un bug en un `calc.py` de prueba, en un
    directorio temporal, y que los tests pasen.
- **Aceptación:** el registro de la demo en `docs/demos/` (sin datos personales), o, si no se
  logra, qué se probó y por qué falla.

## Estado (actualizalo al cerrar cada tarea)

| Tarea | Estado | Commit |
|---|---|---|
| V0 | hecha | e08494a |
| V1 | hecha | 4d8887c |
| V2 | hecha | 284a138 |
| V3 | hecha | 692ae89 |
| V4 | hecha | 6c902c5 |
| V5 | pendiente | |
| V6 | pendiente | |

## Pedidos al usuario

(Lo que requiere publicar o cambiar la configuración del repo en GitHub.)

- **V1:** habilitar el **reporte privado de vulnerabilidades** en GitHub (Settings → Security and
  quality → Advanced Security → Private vulnerability reporting → Enable). Sin eso, el botón
  "Report a vulnerability" de la pestaña Security no aparece y `SECURITY.md` queda sin canal de
  reporte.
- **V1 (TODO):** definir un **contacto alternativo** de seguridad (por ejemplo, un alias o un
  buzón dedicado) para poner en `SECURITY.md`. Hoy no hay ninguno y no se publica un email
  personal.
- **V2:** subir a `main` `.github/workflows/ci.yml` y `.github/dependabot.yml` para que la CI y
  Dependabot se activen (no hago push). Hasta que no estén en la rama por defecto, no corren.
- **V3:** las plantillas de issues, su `config.yml` y la plantilla de PR también tienen que estar
  en la rama por defecto para que GitHub las muestre.
- **V4:** publicar la landing (GitHub Pages u otro). No se hizo. Cómo: subir `site/` a la rama por
  defecto y en **Settings → Pages** elegir esa rama y la carpeta `site/` (o mover los archivos a
  `docs/` y elegir esa carpeta). Está documentado en `site/README.md`.

## Pedidos a Claude

(Lo que necesite tocar zonas prohibidas.)
