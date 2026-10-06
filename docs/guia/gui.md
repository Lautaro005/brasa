# Guía — GUI web

`brasa serve` sirve un dashboard en `/ui`. No hay framework ni paso de build: el HTML, el CSS y el
JS van embebidos en el binario y no se carga nada de internet.

```bash
brasa serve qwen3-4b-q4 --ctx 16384
```

Abrí `http://127.0.0.1:8080/ui` en el navegador.

## Pantallas

1. **Chat.** Streaming por `/v1/chat/completions`. El razonamiento (`<think>`) se muestra plegable
   y se activa o desactiva con el switch. Hay parámetros de muestreo, botón para cancelar (corta la
   conexión y el servidor cancela) e historial en `localStorage`.
2. **Estado.** Modelo, plan de memoria contra el presupuesto, prefix cache, TTFT, tok/s de decode y
   cola, refrescado cada 2 s.
3. **Benchmarks.** Tabla y gráficos de los reportes de `docs/bench/`; los inválidos se marcan.
4. **Plan.** Formulario de contexto, tipo de KV y perfil; usa el planner sin cargar el modelo.
5. **Agentes.** Lo que imprime `brasa connect` para cada herramienta, con botón de copiar.

El servidor expone los datos que consume la GUI:

```bash
brasa ps
```

El tema sigue al sistema (claro u oscuro) y la disposición funciona a 1280 px y en ancho de
teléfono.
