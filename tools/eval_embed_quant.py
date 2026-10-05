"""Calidad de la tabla de embeddings (= lm_head) según su cuantización (decisión de T3.5).

Usa la referencia FP32 con los pesos de `models/qwen3-4b-q4/model.brasa` (lineales q4_0, como
Brasa) y reemplaza la tabla de embeddings atada por la original BF16 cuantizada con cada esquema:
q8_0 (el actual), q6_0 (6,5 bits, ~Q6_K de llama.cpp) y q4_0. Para cada esquema hace teacher
forcing sobre las secuencias de fixtures/qwen3-4b (prompt + greedy FP32, 2898 posiciones) y
reporta, contra la referencia FP32 sin cuantizar:
  - coincidencia de top-1 (la métrica "Q4 vs FP32" de T1.6);
  - KL(FP32 ‖ esquema) estimada sobre el top-k de la referencia (media y p99 por posición).

Con `--act f16`, además redondea a f16 la entrada de cada proyección lineal (lo que haría un GEMM
con activaciones en f16, decisión de T3.3).

Con `--wts f16`, los pesos lineales decuantizados se redondean a f16 (d · q no es exacto en f16).

Uso: .venv/bin/python tools/eval_embed_quant.py [--schemes q8_0,q6_0,q4_0] [--act f32|f16] [--wts f32|f16]
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from convert_brasa import BLOCK  # noqa: E402
from qwen3_ref import Qwen3Ref, Weights  # noqa: E402


def fake_quant(w: torch.Tensor, scheme: str) -> torch.Tensor:
    """Valores que ve el engine con la tabla en `scheme` (mismas reglas que convert_brasa.py)."""
    b = w.reshape(-1, BLOCK).float()
    if scheme == "q8_0":
        d = (b.abs().amax(1) / 127.0).half().float()
        lo, hi, off = -127, 127, 0
    elif scheme in ("q6_0", "q4_0"):
        # Valor de mayor módulo con signo, como q4_0: d = m / -(2^(bits-1)).
        m = b.gather(1, b.abs().argmax(1, keepdim=True)).squeeze(1)
        half_range = 32 if scheme == "q6_0" else 8
        d = (m / -float(half_range)).half().float()
        lo, hi, off = 0, 2 * half_range - 1, half_range
    else:
        raise ValueError(scheme)
    safe = torch.where(d == 0, torch.ones_like(d), d)[:, None]
    q = (torch.round(b / safe) + off).clamp(lo, hi)
    q = torch.where(d[:, None] == 0, torch.full_like(q, off), q)
    return ((q - off) * d[:, None]).reshape(w.shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", default="q8_0,q6_0,q4_0")
    ap.add_argument("--act", choices=["f32", "f16"], default="f32")
    ap.add_argument("--wts", choices=["f32", "f16"], default="f32", help="pesos lineales decuantizados redondeados a f16")
    a = ap.parse_args()
    torch.set_num_threads(8)
    fx = ROOT / "fixtures/qwen3-4b"
    manifest = json.loads((fx / "manifest.json").read_text())
    k = manifest["topk"]
    seqs, refs = [], []
    for p in manifest["prompts"]:
        d = fx / p["id"]
        prompt = np.fromfile(d / "tokens.i32", dtype="<i4")
        greedy = np.fromfile(d / "greedy.i32", dtype="<i4")
        seq = np.concatenate([prompt, greedy[:-1]])
        ids = torch.from_numpy(np.fromfile(d / "topk_ids.i32", dtype="<i4").reshape(len(seq), k)).long()
        vals = torch.from_numpy(np.fromfile(d / "topk_logits.f32", dtype="<f4").reshape(len(seq), k))
        lse = torch.from_numpy(np.fromfile(d / "logsumexp.f32", dtype="<f4"))
        seqs.append(torch.from_numpy(seq).long())
        refs.append((ids, vals - lse[:, None]))  # log-probabilidades FP32 del top-k

    original = Weights(ROOT / "models/qwen3-4b-hf").get("model.embed_tokens.weight")
    model = Qwen3Ref(ROOT / "models/qwen3-4b-hf", brasa=ROOT / "models/qwen3-4b-q4/model.brasa")
    model.act_dtype = torch.float16 if a.act == "f16" else None
    if a.wts == "f16":
        get = model.w.get
        model.w.get = lambda name: get(name).half().float() if name.endswith("proj.weight") else get(name)
    for scheme in a.schemes.split(","):
        t0 = time.time()
        model.embed = fake_quant(original, scheme)
        agree, n, kls = 0, 0, []
        for seq, (ids, logp_ref) in zip(seqs, refs):
            logits = model.forward_many([seq], [model.new_cache()])[0]
            logp = torch.log_softmax(logits, -1).gather(1, ids)
            p_ref = logp_ref.exp()
            kls.append((p_ref * (logp_ref - logp)).sum(-1))
            agree += int((logits.argmax(-1) == ids[:, 0]).sum())
            n += len(seq)
        kl = torch.cat(kls)
        bits = {"q8_0": 8.5, "q6_0": 6.5, "q4_0": 4.5}[scheme]
        mb = original.numel() * bits / 8 / 1e6
        print(
            f"{scheme} act {a.act} pesos {a.wts}: {bits} bits/peso ({mb:.0f} MB) | top-1 vs FP32 {agree}/{n} ({100 * agree / n:.2f} %) | "
            f"KL top-{k} media {kl.mean():.2e}, p99 {kl.quantile(0.99):.2e} | {time.time() - t0:.0f} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
