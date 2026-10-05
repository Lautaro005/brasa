"""Genera los prompts de benchmark con un número exacto de tokens (ver ADR 0002).

El corpus es el propio repo de Brasa (docs y código, Apache-2.0) concatenado con encabezados de
archivo, parecido a lo que un agente de código manda. Si no alcanza, se repite cíclicamente: el
contenido no afecta la velocidad de un modelo denso. Los prompts generados se commitean en
fixtures/bench/ para que no cambien cuando cambie el repo.

Uso: .venv/bin/python tools/make_bench_prompts.py [--tokenizer models/qwen3-4b-hf]
"""

import argparse
import hashlib
import json
import pathlib

from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONTEXTS = [2048, 8192, 16384]
GEN = 128
SOURCES = ["CLAUDE.md", "PLAN.md", "docs/adr", "crates", "scripts", "LICENSE"]


def corpus() -> str:
    files = []
    for src in SOURCES:
        p = ROOT / src
        files += sorted(p.rglob("*")) if p.is_dir() else [p]
    parts = []
    for f in files:
        if f.is_file() and not f.name.startswith(".") and f.suffix in {".md", ".rs", ".toml", ".sh", ""} and "target" not in f.parts:
            parts.append(f"### Archivo: {f.relative_to(ROOT)}\n\n{f.read_text()}\n")
    return "\n".join(parts)


def exact_prompt(tok: Tokenizer, text: str, n: int) -> str:
    ids = tok.encode(text, add_special_tokens=False).ids
    while len(ids) < n + 64:
        ids = ids + ids
    # Al decodificar y re-tokenizar el corte puede moverse un token; se ajusta hasta que cierre.
    k = n
    for _ in range(32):
        prompt = tok.decode(ids[:k])
        got = len(tok.encode(prompt, add_special_tokens=False).ids)
        if got == n:
            return prompt
        k += n - got
    raise RuntimeError(f"no se pudo obtener exactamente {n} tokens")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=str(ROOT / "models/qwen3-4b-hf"))
    args = ap.parse_args()
    tok = Tokenizer.from_file(str(pathlib.Path(args.tokenizer) / "tokenizer.json"))
    out = ROOT / "fixtures/bench"
    out.mkdir(parents=True, exist_ok=True)
    text = corpus()
    manifest = {"gen_tokens": GEN, "tokenizer": "Qwen/Qwen3-4B@1cfa9a72", "prompts": {}}
    for ctx in CONTEXTS:
        n = ctx - GEN
        prompt = exact_prompt(tok, text, n)
        path = out / f"prompt-{ctx}.txt"
        path.write_text(prompt)
        manifest["prompts"][str(ctx)] = {
            "file": path.name,
            "prompt_tokens": n,
            "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        }
        print(f"ctx {ctx}: {n} tokens -> {path.relative_to(ROOT)}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
