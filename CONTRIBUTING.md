# Contributing to Brasa

Brasa is a local inference engine for Apple Silicon. The project is pre-1.0 and under active
development: only `main` gets fixes. The full architecture and rules live in
[CLAUDE.md](CLAUDE.md). The state of each phase and the measured results are in the ADRs
([docs/adr/](docs/adr/)) and in [docs/bench/baseline.md](docs/bench/baseline.md).

The project's working language is Spanish (ADRs, commit messages, code comments, the CLI and the
GUI). Issues and pull requests in English are welcome.

## Building

Requirements: macOS on Apple Silicon and a Rust toolchain (the version is pinned by
`rust-toolchain.toml`).

```bash
cargo build --release -p brasa-cli
./target/release/brasa --version
```

Weights are not in the repository (they are in `.gitignore`). Download them with
`brasa pull qwen3-4b-q4`, or put them in your models folder (`brasa config show` prints which one
is in use).

## Before every commit

```bash
./scripts/ci.sh
```

It runs `cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings` and
`cargo test --workspace`. It has to exit with 0.

## Non-negotiable rules (summary of CLAUDE.md)

1. No Ollama, llama.cpp or MLX in the hot path; they are allowed only as baselines in
   `crates/bench` and in `tools/`.
2. No Metal kernel lands without a CPU reference implementation, a numerical equivalence test with
   a documented tolerance, and a microbenchmark.
3. Correct first, fast later: an optimization that moves logits outside the tolerance is reverted.
4. No allocations in decode: buffers, scratch and the KV cache are preallocated when the model
   loads.
5. Budgeted memory: if it does not fit, the context is lowered or the load is refused; never
   silent swapping.
6. Nothing is claimed without measuring: no speed, memory or quality figures in the docs or the
   README without a `brasa benchmark` report that backs them.
7. Python only offline, in `tools/`; the final binary does not depend on Python.
8. The compatible APIs are transport; internal types belong to `crates/core`.
9. No outbound telemetry, and the GUI loads nothing from the internet.

## GPU tests

Tests that create a Metal device do **not** run in GitHub's CI (see
[docs/adr/0024](docs/adr/0024-ci-en-github-actions.md)); run them on a Mac:

```bash
cargo test -p brasa-kernels -- --nocapture
cargo bench -p brasa-kernels
cargo test --release -p brasa-models --test forward -- --ignored --nocapture --test-threads 1
cargo test --release -p brasa-runtime --test session -- --ignored --nocapture
```

On a shared Mac, don't run two GPU jobs at the same time: benchmarks measured while something else
uses the GPU are not valid.

## Proposing an ADR

Before a non-trivial design decision (a new dependency, a manifest format, the structure of the
GUI), add a short ADR in `docs/adr/` with the next free number and this skeleton:

```markdown
# ADR NNNN — Title

Estado: propuesta

## Contexto

## Decisión

## Consecuencias
```

Don't rewrite an accepted ADR: if the decision changes, write a new one that supersedes it.

## Pull requests

- One small, descriptive commit per task. No `Claude-Session: ...` line.
- `./scripts/ci.sh` exits with 0 before committing.
- Fill in the PR template checklist ([pull_request_template.md](.github/pull_request_template.md)).
- A new kernel comes with its CPU reference, its equivalence test and its microbenchmark.

## Security

Don't open a public issue for a vulnerability: read [SECURITY.md](SECURITY.md) and use private
vulnerability reporting in the Security tab.

## License

By contributing you agree that your contribution is distributed under Apache-2.0
([LICENSE](LICENSE)).
