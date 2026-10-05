"""Convierte un modelo Qwen3 (safetensors BF16 de Hugging Face) al formato nativo `.brasa`.

Esquemas y layout: ADR 0006. Proyecciones lineales en q4_0, embeddings (= lm_head) en q8_0,
normas en f32. Genera una carpeta con model.brasa, tokenizer.json y tokenizer_config.json.

Uso: .venv/bin/python tools/convert_brasa.py models/qwen3-4b-hf models/qwen3-4b-q4
"""

import argparse
import hashlib
import json
import pathlib
import shutil
import struct
import time

import numpy as np
import torch
from safetensors import safe_open

CONVERTER = "convert_brasa.py v1"
PAGE = 16384
BLOCK = 32
MAGIC = b"BRSA"
VERSION = 1
BYTES_PER_BLOCK = {"q4_0": 18, "q8_0": 34}


def align(n: int) -> int:
    return (n + PAGE - 1) // PAGE * PAGE


def dtype_for(name: str, ndim: int) -> str:
    if ndim == 1:
        return "f32"
    if name == "model.embed_tokens.weight":
        return "q8_0"
    return "q4_0"


def nbytes(dtype: str, shape: list[int]) -> int:
    n = int(np.prod(shape))
    if dtype == "f32":
        return 4 * n
    assert shape[-1] % BLOCK == 0, f"última dimensión {shape[-1]} no es múltiplo de {BLOCK}"
    return n // BLOCK * BYTES_PER_BLOCK[dtype]


def quant_q4_0(w: np.ndarray) -> bytes:
    blocks = w.reshape(-1, BLOCK).astype(np.float32)
    idx = np.abs(blocks).argmax(axis=1)
    m = blocks[np.arange(len(blocks)), idx]  # valor de mayor módulo, con signo
    d16 = (m / -8.0).astype(np.float16)
    d = d16.astype(np.float32)
    safe = np.where(d == 0, 1.0, d)
    q = np.clip(np.rint(blocks / safe[:, None]) + 8, 0, 15).astype(np.uint8)
    q[d == 0] = 8
    packed = q[:, :16] | (q[:, 16:] << 4)
    out = np.concatenate([d16.view(np.uint8).reshape(-1, 2), packed], axis=1)
    return out.tobytes()


def quant_q8_0(w: np.ndarray) -> bytes:
    blocks = w.reshape(-1, BLOCK).astype(np.float32)
    d16 = (np.abs(blocks).max(axis=1) / 127.0).astype(np.float16)
    d = d16.astype(np.float32)
    safe = np.where(d == 0, 1.0, d)
    q = np.clip(np.rint(blocks / safe[:, None]), -127, 127).astype(np.int8)
    q[d == 0] = 0
    out = np.concatenate([d16.view(np.uint8).reshape(-1, 2), q.view(np.uint8)], axis=1)
    return out.tobytes()


def header_bytes(meta: dict) -> bytes:
    js = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode()
    return MAGIC + struct.pack("<IQ", VERSION, len(js)) + js


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=pathlib.Path)
    ap.add_argument("dst", type=pathlib.Path)
    ap.add_argument("--source-repo", default="Qwen/Qwen3-4B")
    ap.add_argument("--source-commit", default="1cfa9a7208912126459214e8b04321603b3df60c")
    args = ap.parse_args()

    config = json.loads((args.src / "config.json").read_text())
    assert config["model_type"] == "qwen3", "solo Qwen3 por ahora"
    index = json.loads((args.src / "model.safetensors.index.json").read_text())["weight_map"]
    files = {f: safe_open(str(args.src / f), framework="pt") for f in set(index.values())}

    # 1) Plan: dtype, forma y tamaño de cada tensor (no hace falta cuantizar para saberlo).
    tensors = []
    for name in sorted(index):
        shape = list(files[index[name]].get_slice(name).get_shape())
        dtype = dtype_for(name, len(shape))
        tensors.append({"name": name, "dtype": dtype, "shape": shape, "nbytes": nbytes(dtype, shape)})

    meta = {
        "format": "brasa",
        "version": VERSION,
        "converter": CONVERTER,
        "model": {"family": "qwen3", "source_repo": args.source_repo, "source_commit": args.source_commit, "config": config},
        "quant": {"block": BLOCK, "scheme": "q4_0 lineales, q8_0 embeddings/lm_head, f32 normas (ADR 0006)"},
        "page": PAGE,
        "tensors": tensors,
        "data_sha256": "0" * 64,
    }
    for t in tensors:
        t["offset"] = 0
        t["sha256"] = "0" * 64
    # 2) Offsets absolutos alineados a página. Hashes y offsets tienen largo fijo una vez
    #    calculados, así que el encabezado definitivo mide lo mismo que el provisorio.
    data_start = PAGE
    while True:
        off = data_start
        for t in tensors:
            t["offset"] = off
            off = align(off + t["nbytes"])
        if len(header_bytes(meta)) <= data_start:
            break
        data_start += PAGE
    total = off
    provisional = len(header_bytes(meta))

    # 3) Cuantizar y escribir cada tensor en su offset.
    args.dst.mkdir(parents=True, exist_ok=True)
    out_path = args.dst / "model.brasa"
    t0 = time.time()
    with open(out_path, "wb") as f:
        f.truncate(total)
        for i, t in enumerate(tensors):
            w = files[index[t["name"]]].get_tensor(t["name"]).to(torch.float32).numpy()
            if t["dtype"] == "f32":
                data = w.astype("<f4").tobytes()
            elif t["dtype"] == "q4_0":
                data = quant_q4_0(w)
            else:
                data = quant_q8_0(w)
            assert len(data) == t["nbytes"], t["name"]
            f.seek(t["offset"])
            f.write(data)
            t["sha256"] = hashlib.sha256(data).hexdigest()
            if i % 50 == 0:
                print(f"{i}/{len(tensors)} {t['name']} ({time.time() - t0:.0f} s)", flush=True)
        meta["data_sha256"] = hashlib.sha256("".join(t["sha256"] for t in tensors).encode()).hexdigest()
        header = header_bytes(meta)
        assert len(header) == provisional, "el encabezado cambió de tamaño"
        f.seek(0)
        f.write(header)

    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(args.src / name, args.dst / name)
    print(f"{out_path}: {total / 2**30:.2f} GiB, {len(tensors)} tensores, data_sha256 {meta['data_sha256']} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
