#!/usr/bin/env bash
# Validación en la Mac M2 8 GB. Corre los criterios de aceptación que requieren esa máquina y
# deja la evidencia en docs/bench/m2-8gb/. Uso: ./scripts/validate-8gb.sh
# Cada tarea que necesite validación en 8 GB agrega su paso acá.
set -euo pipefail
cd "$(dirname "$0")/.."

out="docs/bench/m2-8gb"
mkdir -p "$out"
commit="$(git rev-parse --short HEAD)"
mem=$(sysctl -n hw.memsize)
if [[ "$mem" != "8589934592" ]]; then
    echo "Aviso: esta máquina tiene $((mem >> 30)) GiB, no 8 GiB. Se sigue igual." >&2
fi

echo "==> build release ($commit)"
cargo build --release -p brasa-cli

echo "==> T0.2 brasa doctor"
./target/release/brasa doctor | tee "$out/doctor.txt"
./target/release/brasa doctor --json > "$out/doctor.json"
cargo test -p brasa-cli --test doctor 2>&1 | tee "$out/doctor-test.log"
grep -q "test result: ok" "$out/doctor-test.log"

echo "==> T0.5 baselines (llama.cpp y MLX-LM, 2K/8K/16K)"
# Prerrequisitos: Rust, Xcode + Metal Toolchain, `brew install llama.cpp`, Python 3.12.
if [[ ! -x .venv/bin/python ]]; then
    python3.12 -m venv .venv
    .venv/bin/pip install -q -r tools/requirements.txt
fi
if [[ ! -f models/qwen3-4b-hf/config.json ]]; then
    .venv/bin/hf download Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c \
        --local-dir models/qwen3-4b-hf
fi
[[ -f models/qwen3-4b-q4_0.gguf ]] || tools/make_gguf.sh
if [[ ! -d models/qwen3-4b-mlx-q4g32 ]]; then
    .venv/bin/mlx_lm.convert --hf-path models/qwen3-4b-hf -q --q-bits 4 --q-group-size 32 \
        --mlx-path models/qwen3-4b-mlx-q4g32
fi
echo "Cerrá las demás aplicaciones antes de seguir (Enter para continuar)."
read -r _
./scripts/run-baselines.sh "$out" 2>&1 | tee "$out/baselines.log" || true

echo "==> T1.7 brasa run (generación sin swap creciente)"
[[ -f models/qwen3-4b-q4/model.brasa ]] || \
    .venv/bin/python tools/convert_brasa.py models/qwen3-4b-hf models/qwen3-4b-q4
./target/release/brasa run qwen3-4b-q4 --no-think --seed 42 --max-tokens 200 \
    -p "Explicá en tres oraciones qué es una caché de KV en un transformer." \
    2>&1 | tee "$out/run.txt"
printf 'What is 17 * 23? Answer briefly.\nAnd divided by 17?\n' | \
    ./target/release/brasa run qwen3-4b-q4 --seed 7 --max-tokens 400 2>&1 | tee "$out/run-chat.txt"
grep "swap del sistema" "$out/run.txt" "$out/run-chat.txt"

echo "==> T1.8 planner: un contexto que no entra se rechaza; uno que entra se acepta"
./target/release/brasa doctor --json > "$out/doctor-plan.json"
if ./target/release/brasa run qwen3-4b-q4 --ctx 16384 --kv f32 -p hola > "$out/plan-rechazo.txt" 2>&1; then
    echo "ERROR: ctx 16384 debería rechazarse en 8 GB con KV f32" | tee -a "$out/plan-rechazo.txt"
else
    cat "$out/plan-rechazo.txt"
fi
./target/release/brasa plan qwen3-4b-q4 --ctx 4096 | tee "$out/plan-acepta.txt"
./target/release/brasa run qwen3-4b-q4 --ctx 4096 --no-think --max-tokens 50 -p "Decí hola." \
    2>&1 | tee -a "$out/plan-acepta.txt"

echo "==> T3.4 perfil de agente: 16K con KV Q8 (plan y benchmark sin swap creciente)"
./target/release/brasa plan qwen3-4b-q4 --ctx 16384 --kv q8_0 | tee "$out/plan-16k-q8.txt"
./target/release/brasa benchmark --ctx 16384 --kv q8_0 --label kv-q8 --out-dir "$out" \
    2>&1 | tee "$out/bench-16k-q8.log"

echo "==> Fase 4: autotuner (quick < 60 s) y perfil 8 GB de punta a punta"
./target/release/brasa doctor | sed -n '/^Tuning/,/^$/p' | tee "$out/tuning-antes.txt"
t0=$(date +%s)
./target/release/brasa tune 2>&1 | tee "$out/tune.txt"
echo "pared: $(( $(date +%s) - t0 )) s" | tee -a "$out/tune.txt"
./target/release/brasa doctor --json > "$out/doctor-tuning.json"
./target/release/brasa config show | tee "$out/config-show.txt"
grep -q "serve kv.*q8_0" "$out/config-show.txt"
# A/B del decode con y sin la base (mismo binario), 2K y 16K Q8.
BRASA_TUNING=off ./target/release/brasa benchmark --ctx 2048 --label tuning-off --out-dir "$out" \
    2>&1 | tee "$out/bench-2k-tuning-off.log"
./target/release/brasa benchmark --ctx 2048 --label tuning-on --out-dir "$out" \
    2>&1 | tee "$out/bench-2k-tuning-on.log"
BRASA_TUNING=off ./target/release/brasa benchmark --ctx 16384 --kv q8_0 --label kv-q8-tuning-off \
    --out-dir "$out" 2>&1 | tee "$out/bench-16k-q8-tuning-off.log"
./target/release/brasa benchmark --ctx 16384 --kv q8_0 --label kv-q8-tuning-on --out-dir "$out" \
    2>&1 | tee "$out/bench-16k-q8-tuning-on.log"
# Presión con el contexto declarado: serve con los valores del perfil (16K, KV Q8) y una
# conversación larga; memoria y swap antes y después.
./target/release/brasa serve qwen3-4b-q4 --port 18080 > "$out/serve-perfil.log" 2>&1 &
serve_pid=$!
for _ in $(seq 1 120); do
    curl -sf http://127.0.0.1:18080/api/status > /dev/null && break
    sleep 1
done
curl -sf http://127.0.0.1:18080/api/status > "$out/serve-status-antes.json"
python3 - "$out" <<'PY'
import json, sys, urllib.request
out = sys.argv[1]
long = " ".join(f"linea {i}: el agente lee archivos y llama herramientas." for i in range(1500))
body = {"model": "qwen3-4b-q4", "max_tokens": 64,
        "messages": [{"role": "user", "content": long + "\nResumí en una oración."}]}
req = urllib.request.Request("http://127.0.0.1:18080/v1/chat/completions",
                             data=json.dumps(body).encode(), headers={"content-type": "application/json"})
for i in range(3):
    with urllib.request.urlopen(req, timeout=600) as r:
        u = json.load(r)["usage"]
    print("turno", i, u, flush=True)
PY
curl -sf http://127.0.0.1:18080/api/status > "$out/serve-status-despues.json"
kill "$serve_pid"
grep "tuning:" "$out/serve-perfil.log"

{
    echo "commit: $commit"
    echo "fecha: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    sw_vers
} > "$out/meta.txt"

echo
echo "Validación 8 GB OK. Evidencia en $out/ (commitear esa carpeta o pegar su contenido)."
