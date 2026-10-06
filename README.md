# Brasa

Engine de inferencia local propio para Apple Silicon. Se opera desde la terminal (`brasa`),
expone una API local compatible con OpenAI y Anthropic, y trae una GUI web embebida. Su uso
principal es servir **agentes con herramientas** (Claude Code, Codex, Cline, OpenCode), no chat
general.

- Primer modelo: **Qwen3-4B** (transformer denso, GQA, Apache-2.0), cuantización Q4 por grupos
  de 32 en el formato nativo `.brasa`.
- Sin dependencias de Python ni de Ollama, llama.cpp o MLX en el camino caliente.
- Sin telemetría saliente: la GUI no carga nada de internet.

Las cifras de velocidad y memoria medidas están en
[docs/bench/baseline.md](docs/bench/baseline.md); este README solo las cita.

## Requisitos

- macOS sobre Apple Silicon (Metal). Perfiles objetivo: 16 GB y 8 GB.
- Toolchain de Rust (`rustup`); la versión la fija `rust-toolchain.toml`.
- Los pesos se bajan aparte (no van en el repo).

## Instalación desde fuente

```bash
git clone https://github.com/lautaro005/brasa
cd brasa
cargo build --release -p brasa-cli
./target/release/brasa --version
```

También se puede instalar en el `PATH` con:

```bash
cargo install --path crates/cli
```

## Primeros pasos

1. Diagnóstico del hardware y la memoria:

```bash
brasa doctor
brasa doctor --json
```

2. Bajá y convertí el modelo (o usá uno ya presente en `models/`). El manifiesto con el repo, la
   revisión y los sha256 viene embebido:

```bash
brasa pull qwen3-4b-q4
brasa convert models/qwen3-4b-hf models/qwen3-4b-q4
brasa models
brasa models verify qwen3-4b-q4
```

3. Chat en la terminal:

```bash
brasa run qwen3-4b-q4
brasa run qwen3-4b-q4 --no-think -p "Hola"
```

4. Servidor local (API OpenAI/Anthropic + GUI en `/ui`):

```bash
brasa serve qwen3-4b-q4 --ctx 16384
```

5. Estado de un servidor corriendo:

```bash
brasa ps
```

6. Configuración para un agente:

```bash
brasa connect claude-code
brasa connect codex
```

El resto de los subcomandos:

```bash
brasa plan qwen3-4b-q4 --ctx 16384
brasa config show
brasa completions zsh
brasa rm qwen3-4b-q4
```

## Guías

- [Uso con agentes](docs/guia/agentes.md): Claude Code, Codex, Cline y OpenCode.
- [GUI web](docs/guia/gui.md).
- [Memoria y KV](docs/guia/memoria-y-kv.md): perfiles y `--kv`.
- [Solución de problemas](docs/guia/problemas.md).

## Estructura

```text
crates/core        tipos comunes
crates/metal       dispositivo, buffers y pipelines de Metal
crates/kernels     kernels .metal y variantes por chip
crates/quant       formato nativo .brasa, cuantización y conversión
crates/tokenizer   BPE y chat template
crates/models      adaptadores por familia (qwen3) y forward
crates/memory      planner de memoria y presupuesto por perfil
crates/tuner       fingerprint y autotuning
crates/runtime     sesión: prefill, decode, sampling, prefix cache
crates/catalog     manifiestos, descarga y verificación
crates/bench       harness y reportes de benchmark
crates/daemon      API HTTP, GUI embebida y estado
crates/cli         binario `brasa`
tools/             Python offline (referencias y utilidades)
docs/adr/          decisiones de arquitectura
```

## Licencia

Apache-2.0 (ver [LICENSE](LICENSE)). Los pesos de Qwen3-4B se distribuyen bajo su propia
licencia.
