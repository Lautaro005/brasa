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
