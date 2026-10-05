"""Genera fixtures/qwen3-4b/tokenizer_cases.json: casos borde tokenizados con el tokenizer oficial
(HF `tokenizers`), para validar el BPE propio de crates/tokenizer (ADR 0005).

Uso: .venv/bin/python tools/make_tokenizer_cases.py
"""

import json
import pathlib

from tokenizers import Tokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
CASES = [
    "",
    " ",
    "a",
    "Hello world",
    "  leading and trailing spaces  ",
    "multiple    spaces\tand\ttabs",
    "line1\nline2\r\nline3\n\n\nline4",
    "\n\n  \n",
    "trailing newline\n",
    "    indented code\n        more",
    "I'm you're he'll they've it's we'd I'M YOU'LL",
    "numbers 1234567890 3.14159 1e-10 -42 1,000,000",
    "日本語のテキストと中文文本",
    "한국어 텍스트",
    "emoji 🔥🦀👨‍👩‍👧‍👦 flags 🇦🇷",
    "é café (NFC combina)",
    "Ångström ﬁ ligature",
    "<|im_start|>user\nhola<|im_end|>\n<|im_start|>assistant\n",
    "texto<|im_end|>pegado<|endoftext|>",
    "<|im_start parcial y <tool_call sin cerrar",
    "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"x\": 1}}\n</tool_call>",
    "<think>\n\n</think>\n\n",
    "fn main() {\n\tprintln!(\"{}\", x?);\n}\n",
    "def f(x):\n    return x**2  # comentario\n",
    "URL https://example.com/path?q=1&r=2#frag",
    "!!!??? ... --- *** ``` ~~~",
    "mixed nbsp thin　ideographic space",
    "\x00\x01 control \x7f",
]


def main() -> None:
    tok = Tokenizer.from_file(str(ROOT / "fixtures/qwen3-4b/tokenizer/tokenizer.json"))
    cases = [{"text": t, "ids": tok.encode(t, add_special_tokens=False).ids} for t in CASES]
    bench = (ROOT / "fixtures/bench/prompt-16384.txt").read_text()
    cases.append({"text_file": "fixtures/bench/prompt-16384.txt", "ids": tok.encode(bench, add_special_tokens=False).ids})
    out = ROOT / "fixtures/qwen3-4b/tokenizer_cases.json"
    out.write_text(json.dumps({"tokenizer": "Qwen/Qwen3-4B@1cfa9a72 (HF tokenizers)", "cases": cases}, ensure_ascii=False) + "\n")
    print(f"{len(cases)} casos -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
