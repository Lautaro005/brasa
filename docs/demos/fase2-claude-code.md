# Demo fase 2: Claude Code contra brasa

Medido en Apple M1 Pro 16 GB, macOS 27.0, engine 728c10e + cambios de este commit,
pesos qwen3-4b-q4 (Qwen/Qwen3-4B 1cfa9a7). Claude Code 2.1.274, fecha 2026-10-05.

Tarea: repo con `calc.py` (`suma` devuelve `a - b`) y `test_calc.py` que falla.

```bash
./target/release/brasa serve qwen3-4b-q4 --ctx 16384
# variables de `brasa connect claude-code` (incluye CLAUDE_CODE_MAX_OUTPUT_TOKENS=2048 y
# CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT=1)
claude -p "Los tests de test_calc.py fallan. Encontrá el bug en calc.py, arreglalo con Edit y corré python3 -m unittest para confirmar." \
  --model qwen3-4b-q4 --tools Read,Edit,Bash --permission-mode acceptEdits --allowedTools "Bash(python3 -m unittest*)"
```

Resultado: `Read` → `Edit` (`return a - b` → `return a + b`) → `Bash` (`python3 -m unittest`: OK) →
respuesta final. 4 pedidos, sin compactaciones, prefix cache reutilizando casi todo el prompt.

Log del daemon (`BRASA_DEBUG=1`; rutas acortadas):

```text
[brasa] llamada Read {"file_path":"calc.py"}
[brasa] fin ToolCalls Usage { input_tokens: 3429, output_tokens: 103, cached_tokens: 0 }
[brasa] llamada Edit {"file_path":"calc.py","old_string":"return a - b","new_string":"return a + b","replace_all":false}
[brasa] fin ToolCalls Usage { input_tokens: 3634, output_tokens: 135, cached_tokens: 3425 }
[brasa] llamada Bash {"command":"python3 -m unittest test_calc.py","timeout":120000}
[brasa] fin ToolCalls Usage { input_tokens: 3885, output_tokens: 49, cached_tokens: 3630 }
The tests are passing successfully now. The bug in the `suma` function has been fixed by changing `return a - b` to `return a + b`. The function now correctly returns the sum of two numbers.
[brasa] fin Stop Usage { input_tokens: 3987, output_tokens: 45, cached_tokens: 3881 }
```

Antes de `CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT`, Claude Code compactaba después de
cada turno (~3,6K tokens de prompt); el resumen del modelo de 4B inventó un bug en `promedio` y el
modelo lo "arregló", rompiendo los tests (ADR 0008).
