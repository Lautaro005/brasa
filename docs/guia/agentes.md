# Guía — uso con agentes

Brasa expone una API local compatible con OpenAI y Anthropic pensada para agentes. Los pasos son
los mismos para todas las herramientas: levantá el servidor y generá la configuración.

```bash
brasa serve qwen3-4b-q4 --ctx 16384
```

El perfil recomendado para agentes es 16K con KV Q8; los prompts de agentes rondan los 10K–30K
tokens. Ver [memoria y KV](memoria-y-kv.md). Por defecto el servidor escucha en `127.0.0.1:8080`.

`brasa connect <herramienta>` imprime la configuración; no escribe archivos del usuario. La misma
configuración se ve en la GUI, en la pantalla **Agentes**.

## Claude Code

Usa la API Messages. `brasa connect claude-code` imprime las variables de entorno. Como el prompt
completo supera la ventana del modelo, la configuración sugiere limitar las herramientas y fijar
`CLAUDE_CODE_MAX_CONTEXT_TOKENS` al contexto real.

```bash
brasa connect claude-code
brasa serve qwen3-4b-q4 --ctx 16384
```

## Codex

Habla la API Responses. `brasa connect codex` imprime el proveedor, un catálogo de modelos y el
perfil con la ventana de contexto.

```bash
brasa connect codex
```

El catálogo (`~/.codex/brasa-models.json`) define la entrada del modelo **sin**
`apply_patch_tool_type`: así Codex no le ofrece el parche `apply_patch` y el modelo edita los
archivos con comandos de shell (`shell_type = unified_exec`), que es lo que un 4B sí sabe hacer.
Medido en [la demo de la fase 4](../demos/fase4-codex.md).

## Cline y OpenCode

Usan un proveedor compatible con OpenAI. `brasa connect cline` y `brasa connect opencode` imprimen
la Base URL, la API key de prueba, el modelo y el contexto.

```bash
brasa connect cline
brasa connect opencode
```

## Comprobar que responde

Con el servidor corriendo, el estado y las métricas se consultan con:

```bash
brasa ps
brasa ps --json
```
