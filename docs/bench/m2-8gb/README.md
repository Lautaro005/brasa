# Validación en M2 8 GB

Correr en la Mac M2 8 GB, con el repo en el mismo commit que se quiere validar:

```bash
./scripts/validate-8gb.sh
```

El script deja acá `doctor.txt`, `doctor.json`, `doctor-test.log` y `meta.txt`. Commitear estos
archivos (o pegarlos en la conversación) como evidencia de los criterios que exigen la M2.

| Tarea | Criterio en 8 GB | Estado |
|---|---|---|
| T0.2 | `brasa doctor` coincide con `system_profiler` | pendiente |
