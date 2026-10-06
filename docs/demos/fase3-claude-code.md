# Demo de cierre de la fase 3: Claude Code contra brasa con el perfil de agente

Medido en Apple M1 Pro 16 GB, macOS 27.0, engine a036895 (main tras el PR #3), pesos qwen3-4b-q4
(Qwen/Qwen3-4B 1cfa9a7, `data_sha256` 7cc86859…). Claude Code 2.1.274, fecha 2026-10-06.

Repite la demo de la fase 2 ([fase2-claude-code.md](fase2-claude-code.md)) con el perfil de
agente: contexto de 16K y **KV Q8** (ADR 0009). La de la fase 2 usaba KV f16. La configuración de
Claude Code se aisló con `CLAUDE_CONFIG_DIR` apuntando a un directorio temporal.

La tarea es un repo con `calc.py` (`suma` devuelve `a - b`) y `test_calc.py`, que falla.

```bash
BRASA_DEBUG=1 ./target/release/brasa serve qwen3-4b-q4 --ctx 16384 --kv q8_0
# variables de `brasa connect claude-code --ctx 16384`
claude -p "Los tests de test_calc.py fallan. Encontrá el bug en calc.py, arreglalo con Edit y corré python3 -m unittest para confirmar." \
  --model qwen3-4b-q4 --tools Read,Edit,Bash --permission-mode acceptEdits --allowedTools "Bash(python3 -m unittest*)"
```

Resultado: `Read` → `Edit` (`return a - b` → `return a + b`) → `Bash` (`python3 -m unittest`: OK) →
respuesta final.
- Son 4 pedidos a `/v1/messages`, sin compactaciones ni errores.
- La corrida completa de `claude -p` tardó 26,2 s de reloj.
- El diff final cambia solo esa línea.

Log del daemon (`BRASA_DEBUG=1`; rutas acortadas):

```text
[brasa] llamada Read {"file_path":"calc.py"}
[brasa] fin ToolCalls Usage { input_tokens: 3440, output_tokens: 118, cached_tokens: 0 }
[brasa] llamada Edit {"file_path":"calc.py","old_string":"def suma(a, b):\n    return a - b","new_string":"def suma(a, b):\n    return a + b","replace_all":false}
[brasa] fin ToolCalls Usage { input_tokens: 3629, output_tokens: 171, cached_tokens: 3436 }
[brasa] llamada Bash {"command":"python3 -m unittest test_calc.py","timeout":120000}
[brasa] fin ToolCalls Usage { input_tokens: 3921, output_tokens: 72, cached_tokens: 3625 }
The tests have passed successfully after fixing the `suma` function in calc.py. [...]
```

`brasa ps` al terminar:

```text
Pedidos
  /v1/messages              4
Tokens      prompt 15036, generados 410, reutilizados 10978
cancelados  0
TTFT ms     último 768.2, media 4026.9, p50 1481.9 (n=4)
decode t/s  último 45.7, media 45.7, p50 45.8 (n=4)
errores     ninguno
```

- **Prefix cache.** Desde el segundo pedido se reutiliza ~94–95 % del prompt (3436/3629, 3625/3921).
  En el total de la sesión son 10 978 de 15 036 tokens; incluye el primer pedido, que no tiene
  caché.
- **TTFT.** El primer turno paga el prefill completo de ~3,4K tokens. Los siguientes bajan a
  0,8–1,5 s.
- **Decode.** ~46 tok/s en estos turnos cortos con KV Q8. La demo corrió sin controlar la carga de
  la máquina (en paralelo había otro agente compilando y corriendo tests), así que es una cifra de
  la demo, no un benchmark. Los benchmarks válidos están en `docs/bench/`.
