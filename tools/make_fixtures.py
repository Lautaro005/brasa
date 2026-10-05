"""Genera fixtures/qwen3-4b/: prompts, tokens, logits de referencia FP32 y continuación greedy.

Dos fases en procesos separados (para no tener los dos modelos en memoria a la vez):
  reference   forward FP32 propio (tools/qwen3_ref.py); escribe las fixtures y guarda los logits
              completos de todas las posiciones en models/.tmp/ para la fase siguiente.
  crosscheck  transformers BF16 (implementación oficial) sobre las mismas secuencias; registra
              en el manifest el error de logits y la coincidencia de top-1 contra la referencia.

Uso: .venv/bin/python tools/make_fixtures.py [all|reference|crosscheck]  (ver ADR 0003)
"""

import hashlib
import json
import pathlib
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models/qwen3-4b-hf"
MODEL_COMMIT = "1cfa9a7208912126459214e8b04321603b3df60c"
OUT = ROOT / "fixtures/qwen3-4b"
TMP = ROOT / "models/.tmp/ref_logits"
GREEDY = 32
TOPK = 20


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(tokenizer, p: dict) -> str:
    if "text" in p:
        return p["text"]
    if "text_file" in p:
        return (ROOT / p["text_file"]).read_text()
    kwargs = {}
    if "enable_thinking" in p:
        kwargs["enable_thinking"] = p["enable_thinking"]
    return tokenizer.apply_chat_template(
        p["messages"], tools=p.get("tools"), add_generation_prompt=True, tokenize=False, **kwargs
    )


def write(path: pathlib.Path, arr: np.ndarray, dtype: str) -> dict:
    arr = np.ascontiguousarray(arr, dtype=np.dtype(dtype).newbyteorder("<"))
    path.write_bytes(arr.tobytes())
    return {"file": path.name, "dtype": dtype, "shape": list(arr.shape), "sha256": sha256(path)}


def reference() -> None:
    from transformers import AutoTokenizer

    sys.path.insert(0, str(ROOT / "tools"))
    from qwen3_ref import Qwen3Ref

    torch.set_num_threads(8)
    spec = json.loads((ROOT / "tools/fixture_prompts.json").read_text())["prompts"]
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    texts = [render(tok, p) for p in spec]
    ids = [tok.encode(t, add_special_tokens=False) for t in texts]

    t0 = time.time()
    model = Qwen3Ref(MODEL_DIR)
    caches = [model.new_cache() for _ in spec]
    logits = model.forward_many([torch.tensor(x) for x in ids], caches)
    print(f"prefill de {len(spec)} prompts: {time.time() - t0:.1f} s", flush=True)
    rows = [[l] for l in logits]
    greedy = [[int(l[-1].argmax())] for l in logits]
    for step in range(GREEDY - 1):
        new = model.forward_many([torch.tensor([g[-1]]) for g in greedy], caches)
        for i, l in enumerate(new):
            rows[i].append(l)
            greedy[i].append(int(l[-1].argmax()))
        print(f"greedy paso {step + 1}/{GREEDY - 1}: {time.time() - t0:.1f} s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": {"repo": "Qwen/Qwen3-4B", "commit": MODEL_COMMIT},
        "reference": "tools/qwen3_ref.py, FP32 sobre pesos BF16 (ADR 0003)",
        "torch": torch.__version__,
        "greedy_tokens": GREEDY,
        "topk": TOPK,
        "layout": "little-endian; logits por posición de prompt + greedy[:-1]; la fila j predice el token j+1; greedy genera siempre 32 tokens (no se detiene en EOS)",
        "prompts": [],
    }
    for p, text, prompt_ids, r, g in zip(spec, texts, ids, rows, greedy):
        d = OUT / p["id"]
        d.mkdir(exist_ok=True)
        (d / "prompt.txt").write_text(text)
        full = torch.cat(r).numpy()  # [P + GREEDY - 1, V]
        assert full.shape[0] == len(prompt_ids) + GREEDY - 1
        top = torch.topk(torch.from_numpy(full), TOPK, dim=-1)
        lse = torch.logsumexp(torch.from_numpy(full), dim=-1)
        files = {
            "prompt": {"file": "prompt.txt", "sha256": sha256(d / "prompt.txt")},
            "tokens": write(d / "tokens.i32", np.array(prompt_ids), "int32"),
            "greedy": write(d / "greedy.i32", np.array(g), "int32"),
            "last_logits": write(d / "last_logits.f32", full[len(prompt_ids) - 1], "float32"),
            "topk_ids": write(d / "topk_ids.i32", top.indices.numpy(), "int32"),
            "topk_logits": write(d / "topk_logits.f32", top.values.numpy(), "float32"),
            "logsumexp": write(d / "logsumexp.f32", lse.numpy(), "float32"),
        }
        full.astype("<f4").tofile(TMP / f"{p['id']}.f32")
        manifest["prompts"].append(
            {
                "id": p["id"],
                "kind": "text" if ("text" in p or "text_file" in p) else "chat",
                "prompt_tokens": len(prompt_ids),
                "greedy_text": tok.decode(g),
                "files": files,
            }
        )
        print(f"{p['id']}: {len(prompt_ids)} tokens, greedy: {tok.decode(g)!r}", flush=True)
    save_manifest(manifest)


def crosscheck() -> None:
    from transformers import AutoModelForCausalLM

    manifest = json.loads((OUT / "manifest.json").read_text())
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).to(device).eval()
    vocab = model.config.vocab_size
    summary = {"impl": f"transformers BF16 en {device}", "max_abs": 0.0, "top1_agree": 0, "positions": 0}
    for p in manifest["prompts"]:
        d = OUT / p["id"]
        prompt_ids = np.fromfile(d / "tokens.i32", dtype="<i4")
        greedy = np.fromfile(d / "greedy.i32", dtype="<i4")
        seq = np.concatenate([prompt_ids, greedy[:-1]])
        ref = torch.from_numpy(np.fromfile(TMP / f"{p['id']}.f32", dtype="<f4").reshape(len(seq), vocab))
        with torch.no_grad():
            out = model(torch.tensor(seq[None], device=device)).logits[0].float().cpu()
        diff = (out - ref).abs()
        agree = int((out.argmax(-1) == ref.argmax(-1)).sum())
        kl = torch.nn.functional.kl_div(
            torch.log_softmax(out, -1), torch.log_softmax(ref, -1), log_target=True, reduction="none"
        ).sum(-1)
        p["crosscheck"] = {
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "top1_agree": agree,
            "positions": len(seq),
            "mean_kl": float(kl.mean()),
            "max_kl": float(kl.max()),
        }
        summary["max_abs"] = max(summary["max_abs"], float(diff.max()))
        summary["top1_agree"] += agree
        summary["positions"] += len(seq)
        print(f"{p['id']}: {p['crosscheck']}", flush=True)
    manifest["crosscheck"] = summary
    save_manifest(manifest)


def save_manifest(manifest: dict) -> None:
    hashes = sorted(f["sha256"] for p in manifest["prompts"] for f in p["files"].values())
    manifest["fixtures_sha256"] = hashlib.sha256("".join(hashes).encode()).hexdigest()
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    if phase == "reference":
        reference()
    elif phase == "crosscheck":
        crosscheck()
    elif phase == "all":
        for ph in ("reference", "crosscheck"):
            subprocess.run([sys.executable, __file__, ph], check=True)
    else:
        sys.exit(f"fase desconocida: {phase}")


if __name__ == "__main__":
    main()
