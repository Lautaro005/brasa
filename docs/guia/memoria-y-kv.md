# Guía — perfiles de memoria y `--kv`

Antes de cargar, Brasa calcula pesos + KV + workspace + margen del sistema y compara el total con
el presupuesto de la máquina. Si no entra, rechaza el contexto con el máximo que sí entra; nunca
usa swap en silencio.

## Ver el plan sin cargar el modelo

```bash
brasa plan qwen3-4b-q4 --ctx 16384
brasa plan qwen3-4b-q4 --ctx 16384 --kv q8_0
brasa plan qwen3-4b-q4 --ctx 32768 --perfil 8gb
brasa plan qwen3-4b-q4 --ctx 32768 --perfil 8gb --json
```

`--perfil 8gb` o `--perfil 16gb` simulan el presupuesto de otra Mac; sin `--perfil` se mide la
actual. La salida informa el contexto máximo que entra, que es el valor a usar en `--ctx`.

## Tipo de KV

`--kv` elige el almacenamiento de la caché de atención:

- `f16`: por defecto.
- `q8_0`: la mitad de memoria que f16; el perfil de agente de 16K.
- `f32`: para verificar contra las referencias.

```bash
brasa run qwen3-4b-q4 --kv q8_0
brasa serve qwen3-4b-q4 --ctx 16384 --kv q8_0
```

## Perfiles

- **16 GB:** perfil de desarrollo. Para agentes, 16K con KV Q8.
- **8 GB:** validar cada cambio también acá; conviene mirar la presión con `brasa doctor` antes de
  medir.

## Valores por defecto

El modelo, el contexto, el puerto y el KV se pueden fijar en `~/.config/brasa/config.toml`. Los
flags mandan sobre el archivo.

```bash
brasa config show
```

Los números medidos de memoria y velocidad están en
[../bench/baseline.md](../bench/baseline.md).
