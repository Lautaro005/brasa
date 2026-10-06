# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Static HTML/CSS with minimal JS in `site/`, self-contained (pinned by the V4 brief): openable by
double click, no build step, no CDN, no remote fonts.
Embedded GUI in `crates/daemon/assets/` (HTML/CSS/JS compiled into the daemon, served at `/ui`, no build step).

## Users

Developers on Apple Silicon Macs who want to run **coding agents** (Claude Code, Codex, Cline,
OpenCode) against a model running locally. Their situation: they already use one of those tools and
want it pointed at a local engine instead of a cloud API; they care about memory use on a 16 GB or
8 GB machine and about not depending on Ollama, llama.cpp or MLX.

## Product Purpose

Brasa is a local inference engine written in Rust on Metal. It serves a model (Qwen3-4B today)
through a local API compatible with OpenAI and Anthropic, plus an embedded web GUI, so an agent can
work against a local model. Success: a developer can clone, build, serve, and point their agent at
Brasa without leaving the machine.

## Positioning

Its own Rust + Metal engine with **no llama.cpp, MLX or Ollama in the hot path**, and a memory
planner that rejects a context that does not fit instead of swapping silently.

## Operating Context

Terminal-first, on macOS over Apple Silicon. `brasa serve` on loopback; `brasa connect` prints the
configuration for each agent; the GUI lives at `/ui`; weights live in `models/` (gitignored, never
in the repo). Development is on a 16 GB M1 Pro and an 8 GB M2.

## Capabilities and Constraints

Confirmed capabilities: OpenAI and Anthropic compatible API (`/v1/chat/completions`,
`/v1/responses`, `/v1/messages`, `/v1/models`) with streaming, cancellation and tool calling; prefix
cache; `brasa connect`; memory planner; embedded GUI with no remote resources; model catalog with
manifests, `brasa pull`, `brasa models verify` and `brasa rm`; native conversion via `brasa convert`
(no Python in the binary); install from source (README commands).

Hard constraints: **no performance, memory or quality figures without a `brasa benchmark` report**
(CLAUDE.md rule 6) — the landing may only point to `docs/bench/baseline.md`; no outbound telemetry;
no CDN, remote fonts or analytics. There is no published installer, no releases, no pricing, no
customers.

## Brand Commitments

Name: Brasa. License: Apache-2.0. Documentation is written in Spanish. Confirmed for this surface:
the landing is **bilingual ES + EN**, and typography uses **system fonts** (no webfont files).

## Evidence on Hand

Real screenshots in `docs/gui/`: `chat-dos-turnos.jpg`, `chat-cancelado.jpg`, `estado.jpg`. Real
commands in `README.md`. Benchmark methodology and baselines in `docs/bench/baseline.md` (numbers
live there, not on the landing). Absent and not to be fabricated: testimonials, customers, star
counts, latency/speed figures, a hosted demo, a download button.

## Product Principles

1. Nothing is claimed without measuring it.
2. Local and private by default: loopback, no telemetry, nothing leaves the machine.
3. Memory is budgeted: a context that does not fit is rejected, never silently swapped.
4. No third-party engine in the hot path.
5. The real path is an agent with tools, not general chat.

## Accessibility & Inclusion

WCAG AA contrast, keyboard navigation, `alt` text on images, and a layout that holds down to 360 px
(pinned by the V4 brief).
