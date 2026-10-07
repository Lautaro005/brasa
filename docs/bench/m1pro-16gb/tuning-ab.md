# Autotuner de decode: quick tune y A/B (ADR 0029)

Medido en Apple M1 Pro, 16 GB, 16 núcleos de GPU, macOS 27.0 (26A428), 2026-10-07. Engine:
rama `fase-4-autotuner` (commit de los kernels `faa52a7`) contra `main` (`0a178aa`). Pesos:
`models/qwen3-4b-q4` (Qwen/Qwen3-4B `1cfa9a72`, Q4 por grupos de 32, lm_head q6_0). Todas las
corridas con el lock de GPU (`.brasa-gpu.sh`); la máquina tenía ~11 GiB de swap ocupado.

## Quick tune

`BRASA_TUNING_DIR=<tmp> brasa tune` (quick: 15 rondas, KV f16 y Q8, cachés 2048 y 16 384):

| Corrida | Tiempo informado | Pared | Entradas distintas del valor por defecto |
|---|---|---|---|
| 1 | 5,7 s | 6 s | 1 (`gemv_scaled_swiglu` SG=4, +4,3 %) |
| 2 | 5,6 s | 6 s | 0 |
| 3 | 5,7 s | 5 s | 0 |

Aceptación de fase 4 (quick tune < 60 s): cumplida en la M1 Pro. Las diferencias entre candidatos
son de 0–2 %, salvo `SG=1` (3–7 % más lento en varios kernels): en esta máquina los valores
fijados a mano son los mejores o empatan. El cambio de la corrida 1 no se repite y no mueve el
decode (tabla siguiente).

## A/B del decode

`profile_decode` (32 pasos, tiempo de GPU por token), 5 corridas alternadas por configuración:
`main`; la rama con `BRASA_TUNING=off` (valores por defecto); la rama con la base de la corrida 1
(la que cambió `gemv_scaled_swiglu` a SG=4).

| Posición, KV | main | rama, sin base | rama, con base |
|---|---|---|---|
| 2000, f16 (mediana GPU ms/token) | 19,35 | 19,35 | 19,36 |
| 16 000, Q8 (mediana GPU ms/token) | 35,98 | 36,01 | 35,99 |

Diferencias de 0,1 % o menos, dentro del ruido: el tuning no empeora el decode, y en la M1 Pro
tampoco lo mejora.

## Por qué variantes compiladas

Primera versión: los kernels leían `[[simdgroups_per_threadgroup]]` en tiempo de ejecución, sin
variantes. `decode_breakdown 2000`, 4 corridas alternadas, mediana en ms por token (36 capas):

| Kernel | main | runtime |
|---|---|---|
| `gemv_scaled_swiglu` | 5,90 | 5,97 (+1,2 %) |
| `decode_attention_lanes` + reduce | 3,33 | 3,36 (+1 %) |

En `profile_decode` a 2K esa versión quedó ~0,8 % más lenta (10 pares). La versión final
(constante de compilación; variantes compiladas al cargar solo si la base las pide), en otra tanda
de 4 corridas alternadas:

| Kernel | main | final |
|---|---|---|
| `gemv_scaled_swiglu` | 5,85 | 5,84 |
| `decode_attention_lanes` + reduce | 3,46 | 3,45 |
| `gemv_scaled3` | 2,01 | 2,01 |

## Corridas descartadas

Una tanda con el escritorio grabando pantalla dio tiempos 20–40 % más altos y dispersos (GPU
30 ms/token a 2K) y una elección espuria (`attn_decode_lanes` SG=8, "+20,7 %"). Se descartó y se
agregó el control de dispersión del tuner (ADR 0029).
