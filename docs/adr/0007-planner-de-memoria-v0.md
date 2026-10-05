# ADR 0007 — Planner de memoria v0

Estado: aceptada (T1.8)

## Contexto

Regla 5 de CLAUDE.md: antes de cargar se calcula pesos + KV + workspace + scratch + margen; si no
entra, se baja el contexto o se rechaza, nunca swap silencioso. Hay que decidir contra qué
presupuesto se compara.

## Decisión

**Plan** (bytes exactos, calculados antes de cargar, solo con el encabezado del `.brasa`):

- `pesos`: suma de `nbytes` de los tensores del archivo.
- `kv`: `2 · capas · ctx · kv_heads · head_dim · bytes_por_elemento` (f32 en la fase 1).
- `workspace`: los buffers que reserva `Qwen3::load` (activaciones de un bloque de prefill,
  logits, tabla RoPE `ctx · head_dim`). Con la atención tiled no hay scratch de puntajes; la KV
  cache se reserva con capacidad alineada a 32 posiciones por capa.
  Un test compara esta fórmula con lo que el modelo reserva de verdad.
- `overhead`: memoria del proceso fuera de esos buffers (tokenizer, runtime de Metal, binario,
  sampler). Constante medida (ver abajo) redondeada hacia arriba.

**Presupuesto** = `recommendedMaxWorkingSetSize` de Metal − 512 MiB de holgura.
`recommendedMaxWorkingSetSize` es el límite que macOS recomienda para la memoria que usa la GPU
de un proceso; ya deja lugar para el sistema (medido: 11,84 GiB en la M1 Pro 16 GB). La holgura
cubre el resto del proceso y errores de estimación. Un plan que supera el presupuesto se rechaza
con un mensaje que detalla el plan y el contexto máximo que sí entra.

**Memoria disponible ahora** (otras apps abiertas): si el plan supera lo que el kernel informa como
disponible, se avisa pero no se rechaza: macOS libera caché de archivos y comprime memoria
inactiva antes de usar swap. La verificación final es empírica: `brasa run` y `brasa benchmark`
reportan el crecimiento de swap.

**Perfiles simulados** (`brasa plan --perfil 8gb|16gb`): para planear sin la máquina. El
presupuesto de un perfil sin medir es una estimación (`2/3` de la RAM para 8 GB) y se marca como
tal; se reemplaza por el valor real cuando `brasa doctor` corra en esa Mac.

## Consecuencias

- El contexto que se informe por API (`/v1/models`, fase 2) sale de este planner, no del máximo
  nominal del modelo.
- Con KV f32 (fase 1) el contexto que entra es la mitad que con KV f16. Desde T3.1 (ADR 0009) el
  tipo de KV es parte de la sesión y el planner lo cuenta en bytes por bloque de 32 elementos
  (`kv_block_bytes`: 128 en f32, 64 en f16, 34 en Q8).
