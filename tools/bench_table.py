"""Genera docs/bench/baseline.md a partir de los reportes JSON de `brasa benchmark`.

Toma, por máquina, engine, variante y contexto, el reporte válido más reciente (o el más reciente
a secas, marcado, si ninguno es válido). Uso: python3 tools/bench_table.py
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
BENCH = ROOT / "docs/bench"
GIB = 1 << 30


def latest_reports():
    best = {}
    for f in sorted(BENCH.glob("*/*.json")):
        r = json.loads(f.read_text())
        if r.get("schema") != 1:
            continue
        settings = r["engine"]["settings"]
        # Los reportes de llama.cpp con mmap no son comparables en memoria (ADR 0002).
        if r["engine"]["name"] == "llama.cpp" and "none" not in settings:
            continue
        key = (f.parent.name, r["engine"]["name"], r["engine"]["label"], r["ctx"])
        cur = best.get(key)
        rank = (r["valid"], r["timestamp"])
        if cur is None or rank > (cur[0]["valid"], cur[0]["timestamp"]):
            best[key] = (r, f)
    return best


def main() -> None:
    best = latest_reports()
    lines = [
        "# Baselines",
        "",
        "Generado con `tools/bench_table.py` desde los reportes de `brasa benchmark` en esta",
        "carpeta. Metodología: [ADR 0002](../adr/0002-metodologia-de-benchmark.md). Prompt de",
        "`ctx - 128` tokens + 128 generados, greedy, mediana de 3 corridas tras 1 de calentamiento.",
        "Memoria = pico de footprint del proceso.",
        "",
    ]
    for machine in sorted({k[0] for k in best}):
        rows = sorted((k, v) for k, v in best.items() if k[0] == machine)
        r0 = rows[0][1][0]
        m = r0["machine"]
        lines += [
            f"## {m['chip']} {m['memory_bytes'] // GIB} GB ({m['gpu_cores']} núcleos GPU), "
            f"macOS {m['macos_version']} ({m['macos_build']})",
            "",
            "| Engine | Variante | Ctx | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Memoria (GiB) | Válido | Reporte |",
            "|---|---|---:|---:|---:|---:|---:|---|---|",
        ]
        for (_, engine, label, ctx), (r, f) in rows:
            s = r["summary"]
            valid = "sí" if r["valid"] else "no: " + "; ".join(r["invalid_reasons"])
            lines.append(
                f"| {engine} | {label or 'default'} | {ctx} | {s['ttft_ms']['median']:.0f} "
                f"| {s['prefill_tok_s']['median']:.1f} | {s['decode_tok_s']['median']:.1f} "
                f"| {s['peak_footprint_bytes']['median'] / GIB:.2f} | {valid} "
                f"| [{f.name}]({f.parent.name}/{f.name}) |"
            )
        versions = sorted({f"{v[0]['engine']['name']} {v[0]['engine']['version']}" for _, v in rows})
        commits = sorted({v[0]["brasa_commit"] for _, v in rows})
        weights = sorted({f"{v[0]['model']['source_repo']}@{v[0]['model']['source_commit'][:8]}" for _, v in rows})
        lines += [
            "",
            f"Versiones: {', '.join(versions)}. Commit de Brasa: {', '.join(commits)}. "
            f"Pesos: {', '.join(weights)}.",
            "",
        ]
        notes = sorted({n for _, v in rows for n in v[0]["notes"]})
        lines += [f"- Nota: {n}" for n in notes] + ([""] if notes else [])
    (BENCH / "baseline.md").write_text("\n".join(lines))
    print(f"escrito {BENCH / 'baseline.md'}")


if __name__ == "__main__":
    main()
