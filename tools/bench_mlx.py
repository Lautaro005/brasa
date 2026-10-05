"""Driver de baseline MLX-LM para `brasa benchmark --baseline mlx-lm` (ver ADR 0002).

Antes de medir hace un calentamiento mínimo (1 token con un prompt de 16 tokens), equivalente
al que llama.cpp hace al cargar el modelo y excluye de sus tiempos. Procesa el prompt como texto plano (sin chat template), genera exactamente `--gen` tokens en
greedy con EOS bloqueado e imprime en stdout una línea JSON con los tiempos. Los tiempos se toman
en cada token emitido: TTFT = primer token - inicio; decode = último token - primer token.

Uso: .venv/bin/python tools/bench_mlx.py --model DIR --prompt FILE --gen 128
"""

import argparse
import json
import sys
import time

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--gen", type=int, default=128)
    args = ap.parse_args()

    model, tokenizer = load(args.model)
    with open(args.prompt) as f:
        text = f.read()
    prompt = tokenizer.encode(text, add_special_tokens=False)
    eos = sorted(tokenizer.eos_token_ids)

    def block_eos(_tokens, logits):
        logits[:, eos] = -float("inf")
        return logits

    sampler = make_sampler(temp=0.0)
    for _ in stream_generate(model, tokenizer, mx.array(prompt[:16]), max_tokens=1, sampler=sampler):
        pass
    mx.reset_peak_memory()

    times = []
    t0 = time.perf_counter()
    for _ in stream_generate(
        model,
        tokenizer,
        mx.array(prompt),
        max_tokens=args.gen,
        sampler=sampler,
        logits_processors=[block_eos],
    ):
        times.append(time.perf_counter())

    ttft = times[0] - t0
    decode = times[-1] - times[0]
    json.dump(
        {
            "prompt_tokens": len(prompt),
            "gen_tokens": len(times),
            "ttft_ms": ttft * 1e3,
            "decode_ms": decode * 1e3,
            "mlx_peak_memory_bytes": mx.get_peak_memory(),
            "mlx_version": mx.__version__,
        },
        sys.stdout,
    )
    print()


if __name__ == "__main__":
    main()
