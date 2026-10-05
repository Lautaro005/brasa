"""Lector de `.brasa` en numpy (ADR 0006): decuantiza q4_0/q8_0 a f32. Solo para tools/."""

import json
import pathlib
import struct

import numpy as np

BLOCK = 32


def _f16_scales(raw: np.ndarray) -> np.ndarray:
    return raw[:, :2].copy().view(np.float16).astype(np.float32)  # [nb, 1]


def dequantize(dtype: str, data: np.ndarray, shape: list[int]) -> np.ndarray:
    n = int(np.prod(shape))
    if dtype == "f32":
        return data.view("<f4").reshape(shape).copy()
    if dtype == "q4_0":
        b = data.reshape(-1, 18)
        d = _f16_scales(b)
        qs = b[:, 2:]
        lo = (qs & 0x0F).astype(np.int32) - 8
        hi = (qs >> 4).astype(np.int32) - 8
        q = np.concatenate([lo, hi], axis=1).astype(np.float32)
        return (d * q).reshape(shape)
    if dtype == "q8_0":
        b = data.reshape(-1, 34)
        d = _f16_scales(b)
        q = b[:, 2:].copy().view(np.int8).astype(np.float32)
        return (d * q).reshape(shape)
    raise ValueError(f"dtype {dtype} con {n} elementos no soportado")


class BrasaFile:
    def __init__(self, path: str | pathlib.Path):
        self.map = np.memmap(path, dtype=np.uint8, mode="r")
        assert bytes(self.map[:4]) == b"BRSA", "no es un .brasa"
        version, n = struct.unpack("<IQ", bytes(self.map[4:16]))
        assert version == 1
        self.meta = json.loads(bytes(self.map[16 : 16 + n]))
        self.tensors = {t["name"]: t for t in self.meta["tensors"]}

    def config(self) -> dict:
        return self.meta["model"]["config"]

    def get_f32(self, name: str) -> np.ndarray:
        t = self.tensors[name]
        data = self.map[t["offset"] : t["offset"] + t["nbytes"]]
        return dequantize(t["dtype"], np.asarray(data), t["shape"])
