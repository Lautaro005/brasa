# Demo fase 4: Codex editando por shell contra brasa

Medido en Apple M1 Pro 16 GB, macOS 27.0, engine b3c7725 + cambios de este commit, pesos
qwen3-4b-q4 (Qwen/Qwen3-4B 1cfa9a7, sha256 de encabezado 7cc86859). Codex CLI 0.153.4, fecha
2026-10-06.

Tarea: repo con `calc.py` (`suma` devuelve `a - b`) y `test_calc.py` que falla.

## Por qué edita por shell y no con `apply_patch`

El catálogo de modelos de Codex define, por modelo, si se le ofrece la herramienta `apply_patch`
(`apply_patch_tool_type`). Un modelo de 4B no genera parches `apply_patch` válidos, así que la
entrada de Brasa en el catálogo **no** declara ese campo: Codex no le da el parche y lo obliga a
editar con comandos de shell (`shell_type = unified_exec`). Es lo que imprime
`brasa connect codex`.

La entrada del catálogo necesita `base_instructions` (o `model_messages.instructions_template`).
Las usamos **propias** y cortas: el catálogo de OpenAI trae una plantilla de ~17 KB que es de
ellos, y un 4B anda mejor con instrucciones mínimas y directas que dicen cómo editar por shell.

## Configuración

`CODEX_HOME` temporal (`/tmp/...`), nunca el del usuario. Tres archivos:
`config.toml` (el proveedor), `brasa-models.json` (el catálogo, tal cual lo imprime
`brasa connect codex`) y `brasa.config.toml` (el perfil). El catálogo se generó extrayendo el
bloque `JSON` de la salida de `brasa connect codex`, así que la demo valida ese texto.

```bash
./target/release/brasa serve qwen3-4b-q4 --ctx 32768 --kv q8_0
CODEX_HOME=/tmp/brasa-v6-demo/codex-home codex exec --profile brasa --cd /tmp/brasa-v6-demo/repo \
  --skip-git-repo-check --sandbox workspace-write \
  "Los tests de test_calc.py fallan. Encontra el bug en calc.py, arreglalo editando el archivo con comandos de shell (no hay herramienta apply_patch) y corre python3 -m unittest para confirmar."
```

Codex no pidió credenciales: el proveedor va con `requires_openai_auth = false`.

## Resultado

`exec_command` (`python3 -c` con `pathlib`, reemplazo de texto) → `exec_command`
(`python3 -m unittest`: OK) → resumen final. Dos llamadas, ninguna a `apply_patch`; el parche no
existe en el prompt. `./scripts/ci.sh` no cambió.

Log del daemon (`BRASA_DEBUG=1`; JSON recortado):

```text
[brasa] llamada exec_command {"cmd":"python3 -c \"import pathlib;p=pathlib.Path('calc.py');p.write_text(p.read_text().replace('return a - b','return a + b'))\""}
[brasa] fin ToolCalls Usage { input_tokens: 5325, output_tokens: 54, cached_tokens: 0 }
[brasa] llamada exec_command {"cmd":"python3 -m unittest"}
[brasa] fin ToolCalls Usage { input_tokens: 5430, output_tokens: 24, cached_tokens: 5321 }
Los tests de `test_calc.py` pasan ahora después de corregir el bug en `calc.py`. …
[brasa] fin Stop Usage { input_tokens: 5523, output_tokens: 76, cached_tokens: 5426 }
```

El reemplazo lo hizo el modelo con la receta que le dimos en `base_instructions`; el `sed -i` que
intentó en el primer ensayo falla en macOS sin `-i ''`.

## Lo que falló antes

En un primer intento con instrucciones más laxas el modelo intentó
`sed -i 's/def suma(a, b):\n    return a - b/…/'` (falla en macOS) y después emitió una llamada a
herramienta **como texto** (`<tool_call>…</tool_call>`), que Codex no parsea: los tests quedaron
fallando. Aclarar en `base_instructions` que las herramientas se llaman por la interfaz de
tool-calling, dar la receta exacta de `python3 -c` y nombrar `exec_command` fue lo que lo arregló.
