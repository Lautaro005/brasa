# Validación en M2 8 GB

Correr en la Mac M2 8 GB, con el repo en el mismo commit que se quiere validar:

```bash
./scripts/validate-8gb.sh
```

El script deja acá `doctor.txt`, `doctor.json`, `doctor-test.log`, `baselines.log`, los reportes
JSON de `brasa benchmark` y `meta.txt`. La primera vez descarga Qwen3-4B (7,5 GB) y genera el GGUF
y la variante MLX (unos 13 GB más en `models/`). Commitear estos
archivos (o pegarlos en la conversación) como evidencia de los criterios que exigen la M2.

| Tarea | Criterio en 8 GB | Estado |
|---|---|---|
| T0.2 | `brasa doctor` coincide con `system_profiler` | pendiente |
| T0.5 | baselines llama.cpp y MLX-LM en 2K, 8K y 16K | pendiente |
| T1.7 | `brasa run` genera texto coherente sin swap creciente (`run.txt`, `run-chat.txt`) | pendiente |
| T1.8 | rechaza ctx 16384 con mensaje claro y acepta 4096 (`plan-rechazo.txt`, `plan-acepta.txt`) | pendiente |
| Fase 4 | `brasa tune` (quick) en menos de 60 s (`tune.txt`); el decode con la base no empeora (`bench-*-tuning-{off,on}.log`) | pendiente |
| Fase 4 | `serve` usa el perfil 8 GB (16K, KV Q8; `config-show.txt`, `serve-perfil.log`) y la presión queda estable con una conversación de ~16K tokens (`serve-status-{antes,despues}.json`: `system.pressure`, swap) | pendiente |
