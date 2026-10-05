"""Sensibilidad de los logits al redondear la KV cache (ADR 0009).

Compara la referencia con K y V redondeados (`--kv f16` o `q8_0`) contra la misma referencia con
una perturbación relativa de `--eps` en K y V antes de redondear. Esa perturbación es del orden del error del engine
contra la referencia en FP32 (T1.6: ~5e-6 en logits). Si con esta perturbación los logits ya se
mueven más que la tolerancia de T1.6, la tolerancia de 1e-4 no se puede pedir con KV redondeada
aunque el engine fuera perfecto: un error ínfimo cambia el redondeo de algunos elementos en un paso
del tipo de la caché.

Uso: .venv/bin/python tools/kv_rounding_sensitivity.py [--kv f16|q8_0] [--eps 1e-6] [--prompts id,id,...]
"""

import argparse
import json
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from qwen3_ref import Qwen3Ref  # noqa: E402


class Perturbed(Qwen3Ref):
    """Referencia que multiplica K y V por (1 + eps·ruido) antes de redondear."""

    def __init__(self, *a, eps: float, seed: int, **kw):
        super().__init__(*a, **kw)
        self.eps = eps
        self.gen = torch.Generator().manual_seed(seed)
        base = self.round_kv  # redondeo al tipo de la caché

        def round_kv(x: torch.Tensor) -> torch.Tensor:
            noise = torch.empty_like(x).uniform_(-1.0, 1.0, generator=self.gen)
            return base(x * (1 + self.eps * noise))

        self.round_kv = round_kv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv", choices=["f16", "q8_0"], default="f16")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--prompts", default="chat-no-think,tools-roundtrip,long-context")
    a = ap.parse_args()

    torch.set_num_threads(8)
    fx = ROOT / "fixtures/qwen3-4b-q4"
    manifest = json.loads((fx / "manifest.json").read_text())
    want = a.prompts.split(",")
    ids = []
    for p in manifest["prompts"]:
        if p["id"] in want:
            raw = (fx / p["id"] / "tokens.i32").read_bytes()
            ids.append((p["id"], torch.frombuffer(bytearray(raw), dtype=torch.int32).long()))

    model_dir = ROOT / "models/qwen3-4b-hf"
    brasa = ROOT / "models/qwen3-4b-q4/model.brasa"
    kv_dtype = {"f16": torch.float16, "q8_0": "q8_0"}[a.kv]
    base = Qwen3Ref(model_dir, brasa=brasa, kv_dtype=kv_dtype)
    pert = Perturbed(model_dir, brasa=brasa, kv_dtype=kv_dtype, eps=a.eps, seed=0)
    for pid, seq in ids:
        ab = base.forward_many([seq], [base.new_cache()])[0]
        ap_ = pert.forward_many([seq], [pert.new_cache()])[0]
        lb, lp = ab[-1], ap_[-1]
        rel = float((lb - lp).abs().max() / lb.abs().max())
        same = int(lb.argmax()) == int(lp.argmax())
        # Teacher forcing: posiciones donde el top-1 cambia, y la mayor brecha top-1/top-2
        # relativa (a |top-1|) de la referencia sin perturbar entre las que cambian.
        top2 = ab.topk(2, dim=-1).values
        flips = ab.argmax(-1) != ap_.argmax(-1)
        gaps = ((top2[:, 0] - top2[:, 1]) / top2[:, 0].abs())[flips]
        worst = float(gaps.max()) if len(gaps) else 0.0
        print(
            f"{pid:22} KV {a.kv} eps {a.eps:.0e}: logits {rel:.2e} (max|Δ|/max|ref|), mismo top-1: {same}; "
            f"top-1 distinto en {int(flips.sum())}/{len(seq)} posiciones (brecha relativa máx. {worst:.1e})",
            flush=True,
        )


if __name__ == "__main__":
    main()
