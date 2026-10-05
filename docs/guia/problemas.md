# Guía — solución de problemas

## No se encontró el modelo

```
no se encontró el modelo "qwen3-4b-q4" ...
```

Mirá los modelos locales y, si falta, bajalo y convertí:

```bash
brasa models
brasa pull qwen3-4b-q4
brasa convert models/qwen3-4b-hf models/qwen3-4b-q4
```

## El contexto no entra en memoria

El planner lo rechaza con el máximo que sí entra:

```
el contexto 32768 no entra en memoria: ... Contexto máximo que entra: 20480 (--ctx 20480).
```

Usá ese `--ctx`, bajá el tipo de KV (`--kv q8_0`) o mirá el plan antes de cargar:

```bash
brasa plan qwen3-4b-q4 --ctx 32768 --kv q8_0
```

## El servidor no responde

Comprobá que esté corriendo y consultá su estado y métricas:

```bash
brasa ps
```

Si `brasa ps` no conecta, revisá el puerto con el que levantaste `brasa serve` y usá `--port`.

## El sistema está bajo presión de memoria o usa swap

Mirá el diagnóstico antes de medir o de subir el contexto:

```bash
brasa doctor
```

Si la presión es alta o hay swap, cerrá aplicaciones: las mediciones no son representativas en ese
estado.

## Verificar los pesos

```bash
brasa models verify qwen3-4b-q4
```

Si algún sha256 no coincide, el archivo `.brasa` está dañado; volvé a bajarlo y convertirlo.
