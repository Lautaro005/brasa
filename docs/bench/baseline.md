# Baselines

Generado con `tools/bench_table.py` desde los reportes de `brasa benchmark` en esta
carpeta. Metodología: [ADR 0002](../adr/0002-metodologia-de-benchmark.md). Prompt de
`ctx - 128` tokens + 128 generados, greedy, mediana de 3 corridas tras 1 de calentamiento.
Memoria = pico de footprint del proceso.

## Apple M1 Pro 16 GB (16 núcleos GPU), macOS 27.0 (26A428)

| Engine | Variante | Ctx | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Memoria (GiB) | Válido | Reporte |
|---|---|---:|---:|---:|---:|---:|---|---|
| brasa | default | 2048 | 6413 | 299.4 | 50.2 | 2.55 | sí | [brasa-qwen3-4b-q4-ctx2048-20261006T175553Z.json](m1pro-16gb/brasa-qwen3-4b-q4-ctx2048-20261006T175553Z.json) |
| brasa | default | 8192 | 41263 | 195.4 | 34.6 | 3.40 | sí | [brasa-qwen3-4b-q4-ctx8192-20261006T180555Z.json](m1pro-16gb/brasa-qwen3-4b-q4-ctx8192-20261006T180555Z.json) |
| brasa | default | 16384 | 118051 | 137.7 | 28.0 | 4.54 | sí | [brasa-qwen3-4b-q4-ctx16384-20261006T183130Z.json](m1pro-16gb/brasa-qwen3-4b-q4-ctx16384-20261006T183130Z.json) |
| brasa | fase1 | 2048 | 106582 | 18.0 | 9.8 | 2.95 | sí | [brasa-fase1-qwen3-4b-q4-ctx2048-20261005T133509Z.json](m1pro-16gb/brasa-fase1-qwen3-4b-q4-ctx2048-20261005T133509Z.json) |
| brasa | kv-q8 | 16384 | 121711 | 133.6 | 27.1 | 3.48 | sí | [brasa-kv-q8-qwen3-4b-q4-ctx16384-20261006T184008Z.json](m1pro-16gb/brasa-kv-q8-qwen3-4b-q4-ctx16384-20261006T184008Z.json) |
| llama.cpp | default | 2048 | 3919 | 490.0 | 50.8 | 3.18 | sí | [llamacpp-qwen3-4b-q4-ctx2048-20261005T033424Z.json](m1pro-16gb/llamacpp-qwen3-4b-q4-ctx2048-20261005T033424Z.json) |
| llama.cpp | default | 8192 | 22677 | 355.6 | 38.0 | 4.03 | sí | [llamacpp-qwen3-4b-q4-ctx8192-20261005T033709Z.json](m1pro-16gb/llamacpp-qwen3-4b-q4-ctx8192-20261005T033709Z.json) |
| llama.cpp | default | 16384 | 62590 | 259.7 | 28.6 | 5.20 | sí | [llamacpp-qwen3-4b-q4-ctx16384-20261005T034415Z.json](m1pro-16gb/llamacpp-qwen3-4b-q4-ctx16384-20261005T034415Z.json) |
| llama.cpp | kv-q8 | 16384 | 62210 | 261.3 | 20.1 | 4.12 | sí | [llamacpp-kv-q8-qwen3-4b-q4-ctx16384-20261007T152845Z.json](m1pro-16gb/llamacpp-kv-q8-qwen3-4b-q4-ctx16384-20261007T152845Z.json) |
| mlx-lm | default | 2048 | 5411 | 354.8 | 48.9 | 3.98 | sí | [mlxlm-qwen3-4b-q4-ctx2048-20261005T033518Z.json](m1pro-16gb/mlxlm-qwen3-4b-q4-ctx2048-20261005T033518Z.json) |
| mlx-lm | default | 8192 | 27542 | 292.8 | 33.6 | 5.51 | sí | [mlxlm-qwen3-4b-q4-ctx8192-20261005T033937Z.json](m1pro-16gb/mlxlm-qwen3-4b-q4-ctx8192-20261005T033937Z.json) |
| mlx-lm | default | 16384 | 68809 | 236.2 | 23.5 | 7.57 | sí | [mlxlm-qwen3-4b-q4-ctx16384-20261005T034942Z.json](m1pro-16gb/mlxlm-qwen3-4b-q4-ctx16384-20261005T034942Z.json) |

Versiones: brasa 0.0.1 (444dfb12aee4), brasa 0.0.1 (8dc9fa30d3a2), llama.cpp 0.5.0 (build 11146, commit 7fe450e19), mlx-lm 0.32.0, mlx 0.32.3. Commit de Brasa: 444dfb12aee4, 8dc9fa30d3a2, d16f20500a1f, d5c9855fadaf. Pesos: Qwen/Qwen3-4B@1cfa9a72.

## Notas

- Cuantización: llama.cpp `Q4_0` ocupa 4,70 bits/peso efectivos (algunos tensores en más
  bits); MLX con grupos de 32 ocupa 5,0 bits/peso (escala y bias BF16 por grupo). Brasa
  apunta a Q4 g32 con escala FP16 (4,5 bits/peso en las matrices).
- `kv-q8`: llama.cpp con `-ctk q8_0 -ctv q8_0`, el perfil de agente de Brasa. Ahorra
  ~1 GiB en 16K pero baja el decode respecto de KV FP16.
- M2 8 GB: pendiente; se agrega al correr `scripts/validate-8gb.sh` en esa máquina.
- Los reportes de llama.cpp con mmap (anteriores a ADR 0002 rev. `-lm none`) no entran en
  esta tabla porque su footprint no incluye los pesos.
