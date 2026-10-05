"""Forward de referencia de Qwen3 en FP32 exacto sobre los pesos BF16 originales (ADR 0003).

Lee cada tensor por mmap desde los safetensors y lo sube a FP32 en el momento de usarlo, capa por
capa: el pico de memoria es la tabla de embeddings en FP32 (1,5 GB) más una capa (~400 MB).
Procesa varias secuencias a la vez en orden capa-mayor, para convertir cada capa una sola vez por
paso. Es deliberadamente simple y legible: documenta la matemática que Brasa debe reproducir.
"""

import json
import math
import pathlib

import torch
from safetensors import safe_open

LAYER_TENSORS = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]


class BrasaWeights:
    """Pesos decuantizados de un `.brasa` (ADR 0006): la referencia ve exactamente lo que ve Brasa."""

    def __init__(self, path: pathlib.Path):
        import sys

        sys.path.insert(0, str(pathlib.Path(__file__).parent))
        from brasa_file import BrasaFile

        self.file = BrasaFile(path)

    def get(self, name: str) -> torch.Tensor:
        return torch.from_numpy(self.file.get_f32(name))


class Weights:
    def __init__(self, model_dir: pathlib.Path):
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        self.where = index["weight_map"]
        self.files = {f: safe_open(str(model_dir / f), framework="pt") for f in set(self.where.values())}

    def get(self, name: str) -> torch.Tensor:
        return self.files[self.where[name]].get_tensor(name).to(torch.float32)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """RoPE estilo NeoX (rotate_half). x: [T, H, D]. Ángulos en FP64, aplicados en FP32."""
    d = x.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    ang = positions.to(torch.float64)[:, None] * inv_freq[None, :]  # [T, D/2]
    cos = torch.cat([ang.cos(), ang.cos()], -1).to(torch.float32)[:, None, :]
    sin = torch.cat([ang.sin(), ang.sin()], -1).to(torch.float32)[:, None, :]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rotated = torch.cat([-x2, x1], -1)
    return x * cos + rotated * sin


class Qwen3Ref:
    def __init__(
        self,
        model_dir: str | pathlib.Path,
        brasa: str | pathlib.Path | None = None,
        kv_dtype: torch.dtype | None = None,
    ):
        """`model_dir`: safetensors originales. Con `brasa`, usa los pesos decuantizados de ese archivo.
        Con `kv_dtype` (p. ej. torch.float16), K y V se redondean a ese tipo al entrar en la caché,
        después de QK-norm y RoPE, como hace el engine (ADR 0009); el resto sigue en FP32."""
        self.kv_dtype = kv_dtype
        model_dir = pathlib.Path(model_dir)
        self.cfg = json.loads((model_dir / "config.json").read_text())
        c = self.cfg
        assert c["model_type"] == "qwen3" and c["tie_word_embeddings"]
        self.n_layers = c["num_hidden_layers"]
        self.n_heads = c["num_attention_heads"]
        self.n_kv = c["num_key_value_heads"]
        self.head_dim = c["head_dim"]
        self.eps = c["rms_norm_eps"]
        self.theta = c["rope_theta"]
        self.w = BrasaWeights(pathlib.Path(brasa)) if brasa else Weights(model_dir)
        self.embed = self.w.get("model.embed_tokens.weight")  # [V, H], también lm_head
        self.final_norm = self.w.get("model.norm.weight")

    def round_kv(self, x: torch.Tensor) -> torch.Tensor:
        """K o V tal como quedan en la caché (redondeados a `kv_dtype` si hay)."""
        if self.kv_dtype is None:
            return x
        return x.to(self.kv_dtype).to(torch.float32)

    def new_cache(self) -> list:
        return [None] * self.n_layers

    def forward_many(
        self, seqs: list[torch.Tensor], caches: list[list], capture: dict | None = None
    ) -> list[torch.Tensor]:
        """Procesa tokens nuevos de varias secuencias. Devuelve logits FP32 [T_i, V] por secuencia.

        caches[i][capa] = (K, V) con K, V: [T_pasado, n_kv, head_dim], K ya rotada.
        """
        xs = [self.embed[s] for s in seqs]
        if capture is not None:
            capture["embed"] = [x.clone() for x in xs]
        starts = [0 if c[0] is None else c[0][0].shape[0] for c in caches]
        group = self.n_heads // self.n_kv
        scale = 1.0 / math.sqrt(self.head_dim)
        for li in range(self.n_layers):
            w = {k: self.w.get(f"model.layers.{li}.{k}") for k in LAYER_TENSORS}
            for i, x in enumerate(xs):
                t = x.shape[0]
                pos = torch.arange(starts[i], starts[i] + t)
                h = rms_norm(x, w["input_layernorm.weight"], self.eps)
                q = (h @ w["self_attn.q_proj.weight"].T).view(t, self.n_heads, self.head_dim)
                k = (h @ w["self_attn.k_proj.weight"].T).view(t, self.n_kv, self.head_dim)
                v = (h @ w["self_attn.v_proj.weight"].T).view(t, self.n_kv, self.head_dim)
                q = rope(rms_norm(q, w["self_attn.q_norm.weight"], self.eps), pos, self.theta)
                k = rope(rms_norm(k, w["self_attn.k_norm.weight"], self.eps), pos, self.theta)
                k, v = self.round_kv(k), self.round_kv(v)
                if caches[i][li] is not None:
                    k = torch.cat([caches[i][li][0], k])
                    v = torch.cat([caches[i][li][1], v])
                caches[i][li] = (k, v)
                kk = k.repeat_interleave(group, dim=1)  # cabeza de query j usa la KV j // group
                vv = v.repeat_interleave(group, dim=1)
                scores = torch.einsum("thd,shd->hts", q, kk) * scale
                key_pos = torch.arange(kk.shape[0])
                scores = scores.masked_fill(key_pos[None, None, :] > pos[None, :, None], float("-inf"))
                p = torch.softmax(scores, dim=-1)
                o = torch.einsum("hts,shd->thd", p, vv).reshape(t, self.n_heads * self.head_dim)
                x = x + o @ w["self_attn.o_proj.weight"].T
                h = rms_norm(x, w["post_attention_layernorm.weight"], self.eps)
                gate = torch.nn.functional.silu(h @ w["mlp.gate_proj.weight"].T)
                x = x + (gate * (h @ w["mlp.up_proj.weight"].T)) @ w["mlp.down_proj.weight"].T
                xs[i] = x
            if capture is not None:
                capture[li] = [x.clone() for x in xs]
            del w
        return [rms_norm(x, self.final_norm, self.eps) @ self.embed.T for x in xs]
