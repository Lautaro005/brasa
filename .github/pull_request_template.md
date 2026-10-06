## Qué cambia

<!-- Una o dos frases. Enlazá el issue si hay. -->

## Checklist (de CLAUDE.md)

- [ ] `./scripts/ci.sh` sale 0 (fmt, clippy `-D warnings` y tests).
- [ ] Si es un **kernel nuevo**: trae implementación de referencia en CPU, test de equivalencia
      numérica con tolerancia documentada y microbenchmark (regla 2).
- [ ] No hay cifras de velocidad, memoria ni calidad sin un reporte de `brasa benchmark` que las
      respalde (regla 6).
- [ ] Si toca una decisión de diseño no trivial (dependencia, formato, estructura), hay un ADR
      corto en `docs/adr/`.
- [ ] Sin telemetría saliente y la GUI no carga nada de internet (reglas 3 y 9).

Reglas completas: [../CLAUDE.md](../CLAUDE.md). Proceso: [../CONTRIBUTING.md](../CONTRIBUTING.md).
