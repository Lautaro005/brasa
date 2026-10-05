# CLAUDE.md — Brasa

Brasa es un engine de inferencia local, propio y open source, optimizado para Apple Silicon. Se opera desde la terminal (`brasa`), expone una API local compatible con OpenAI y Anthropic, y más adelante tendrá una GUI web. Su uso principal es servir **agentes con herramientas** (Codex, Claude Code, Pi, Hermes, Cline, OpenCode), no chat general.

Documento de arquitectura completo: "Arquitectura del engine de inferencia para Apple Silicon" (Claude Docs). Plan de tareas: `PLAN.md`.

## Hardware objetivo

- **Desarrollo y perfil 16 GB:** MacBook Pro M1 Pro, 16 GB.
- **Perfil 8 GB:** Mac M2, 8 GB. Todo cambio de memoria o kernels se valida también acá.
- Primer modelo: **Qwen3-4B** (transformer denso, GQA, Apache-2.0), cuantización Q4 por grupos de 32.

## Reglas no negociables

1. **Nada de Ollama, llama.cpp ni MLX en el camino caliente.** Pueden usarse solo en `crates/bench` como baseline y en `tools/` para generar referencias.
2. **Ningún kernel Metal entra sin tres cosas:** implementación de referencia en CPU, test de equivalencia numérica con tolerancia documentada y microbenchmark.
3. **Correcto primero, rápido después.** Una optimización que cambia logits fuera de tolerancia se revierte.
4. **Decode sin asignaciones:** buffers, scratch y KV se preasignan al cargar el modelo. Cualquier `alloc` en el loop de decode es un bug.
5. **Memoria presupuestada:** antes de cargar se calcula pesos + KV + workspace + scratch + margen del sistema. Si no entra, se baja el contexto o se rechaza; nunca swap silencioso.
6. **Nada se afirma sin medir.** Velocidad, memoria y calidad se reportan como "medido en <chip> <RAM>" o "estimado". No se escriben cifras de speedup en docs o README sin un reporte de `brasa benchmark` que las respalde.
7. **Python solo offline** (`tools/`): conversión, cuantización, logits de referencia. El binario final no depende de Python.
8. **La API compatible es transporte.** Los tipos internos son propios (`crates/core`); los formatos OpenAI y Anthropic se traducen en `crates/daemon`.
9. **Sin telemetría saliente.** El fingerprint de hardware es local y no incluye número de serie ni identificadores personales.

## Estructura

```text
crates/core        tipos comunes (DType, Shape, errores, config)
crates/metal       dispositivo, buffers, command queues, cache de pipelines
crates/kernels     fuentes .metal y registro de variantes por chip
crates/quant       empaquetado Q4/Q8 y formato nativo de pesos
crates/tokenizer   BPE y chat template
crates/models      adaptadores por familia (qwen3 primero) y grafo de forward
crates/memory      planner de memoria, KV cache, presupuesto por perfil
crates/tuner       autotuner, fingerprint, base de tuning
crates/runtime     sesión: prefill, decode, sampling, prefix cache
crates/catalog     manifiestos, descarga, verificación por hash
crates/bench       harness y reportes comparables
crates/daemon      API HTTP, scheduler, Model Manager
crates/cli         binario `brasa`
tools/             Python offline
fixtures/          logits y tokens de referencia
docs/adr/          decisiones de arquitectura
```

Las dependencias solo apuntan hacia abajo: `cli` y `daemon` dependen de `runtime`; `runtime` de `models`, `memory`, `tuner`; todos de `metal` y `core`. `metal` no depende de nada del producto.

## Comandos

```bash
./scripts/ci.sh                                # fmt --check + clippy -D warnings + test (CI local)
cargo build --release
cargo test --workspace
cargo run -p brasa-cli -- doctor               # chip, núcleos CPU/GPU, RAM, macOS, Metal, presión
cargo run -p brasa-cli -- doctor --json
cargo run --release -p brasa-cli -- run qwen3-4b-q4                      # chat interactivo
cargo run --release -p brasa-cli -- run qwen3-4b-q4 --no-think -p "Hola"  # una respuesta
./scripts/validate-8gb.sh                      # solo en la M2 8 GB; evidencia en docs/bench/m2-8gb/
cargo test -p brasa-kernels -- --nocapture     # equivalencia numérica GPU vs referencia CPU
cargo bench -p brasa-kernels                   # microbenchmarks de kernels (tiempo de GPU)

# Baselines (ver docs/adr/0002). Pesos en models/ (gitignored):
.venv/bin/hf download Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c --local-dir models/qwen3-4b-hf
tools/make_gguf.sh                             # models/qwen3-4b-q4_0.gguf
.venv/bin/python tools/make_bench_prompts.py   # fixtures/bench/prompt-{2048,8192,16384}.txt
cargo run --release -p brasa-cli -- benchmark --baseline llama.cpp --ctx 2048
cargo run --release -p brasa-cli -- benchmark --baseline mlx-lm --ctx 2048
./scripts/run-baselines.sh                     # T0.5: 2K/8K/16K en ambos engines
python3 tools/bench_table.py                   # regenera docs/bench/baseline.md

# Fixtures de referencia (ADR 0003): FP32 propio + validación contra transformers BF16
.venv/bin/python tools/make_fixtures.py all    # fixtures/qwen3-4b/ (~10 min en M1 Pro)
.venv/bin/python tools/make_tokenizer_cases.py # casos borde del tokenizer contra HF tokenizers
cargo test -p brasa-tokenizer                  # T1.2: template y tokens contra las fixtures

# Formato nativo (ADR 0006)
.venv/bin/python tools/convert_brasa.py models/qwen3-4b-hf models/qwen3-4b-q4
cargo test --release -p brasa-quant --test roundtrip -- --ignored --nocapture   # T1.3
.venv/bin/python tools/make_fixtures.py q4     # fixtures/qwen3-4b-q4/: referencia con pesos decuantizados
cargo test --release -p brasa-models --test layers -- --ignored --nocapture     # T1.5
cargo test --release -p brasa-models --test forward -- --ignored --nocapture    # T1.6 (~8 min)
cargo test --release -p brasa-models --test decode_alloc -- --ignored --nocapture  # regla 4
cargo test --release -p brasa-runtime --test session -- --ignored --nocapture   # T1.7 greedy y prefijo
```

Previstos (todavía no existen):

```bash
cargo run -p brasa-cli -- benchmark --model qwen3-4b-q4 --ctx 2048     # engine propio, fase 1
```

(Mantener esta sección verdadera: mover cada comando a la lista de arriba cuando exista.)

## Requisitos de API para agentes

- `POST /v1/chat/completions` con streaming SSE y `tools`/`tool_calls`.
- `POST /v1/responses` (subconjunto mínimo con streaming y herramientas).
- `POST /v1/messages` formato Anthropic con `tool_use`, para Claude Code.
- `GET /v1/models` informa el **contexto real del perfil de memoria**, no el nominal del modelo.
- Cancelación de generación y errores en el formato del cliente.
- Prefix caching: los agentes reenvían el mismo prompt largo en cada turno; el TTFT depende de reutilizar su KV.
- Suite `conformance` en CI contra el SDK oficial de OpenAI y un cliente Anthropic. Verificar los formatos contra la versión actual de cada herramienta antes de cerrar la fase 2.

## Contexto para agentes

Los agentes de código mandan prompts de 15K a 30K tokens. Un perfil de 2K de contexto sirve para pruebas pero no para este caso de uso. Perfil por defecto en modo agente: **16K con KV Q8**, y medir en la Mac de 8 GB si es estable.

## Flujo de trabajo

- Trabajar una tarea de `PLAN.md` por vez. Cada tarea tiene criterio de aceptación verificable por comando.
- Antes de una decisión de diseño no trivial, escribir un ADR corto en `docs/adr/`.
- Registrar resultados de benchmark en `docs/bench/` con chip, RAM, macOS, commit del engine y commit de pesos.
- Si un criterio no se cumple, no avanzar de fase: documentar el bloqueo.

## Git

- Commits pequeños y descriptivos, uno por tarea.
- No incluir la línea `Claude-Session: ...` en los mensajes de commit. Co-Authored-By puede quedar.
- No hacer push ni abrir PR sin pedirlo.
