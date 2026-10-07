# ADR 0028 — Conectar agentes escribiendo su configuración

Estado: aceptada (2026-10-06)

## Contexto

Hasta ahora `brasa connect <herramienta>` y la pestaña Agentes de la GUI solo **muestran** la
configuración para apuntar un agente al daemon; el usuario la copia a mano. El usuario pidió un
botón que conecte directamente. Eso significa escribir archivos en su carpeta personal, desde una
página web y desde la CLI, así que hay que acotar qué se toca.

Opciones que se le presentaron al usuario:

1. Archivos propios de Brasa + agregar lo mínimo a la config de la herramienta, con respaldo.
2. Solo lanzadores (`codex-brasa`, `claude-brasa`, …), sin editar ninguna config existente.
3. Editar la config principal para que la herramienta use Brasa por defecto.

Eligió la 1.

## Decisión

Una sola implementación, `brasa_daemon::connect_apply`, que usan `POST /api/agents/<herramienta>/connect`
(botón **Conectar** de la GUI) y `brasa connect <herramienta> --apply`. Nunca cambia la herramienta
que el usuario usa por defecto: agrega una opción para usar Brasa.

**Codex** (`$CODEX_HOME` o `~/.codex`):
- Escribe `brasa-models.json` (el catálogo sin `apply_patch`, V6) y `brasa.config.toml` (el perfil),
  que son de Brasa y se reescriben.
- En `config.toml` **agrega** la tabla `[model_providers.brasa]` solo si no existe, al final del
  archivo y sin reescribir el resto: así se conservan comentarios y orden. Antes copia el archivo a
  `config.toml.brasa-bak-<segundos unix>`.
- Si la tabla ya existe con la misma `base_url`, no toca nada. Si existe con otra URL, o el archivo
  no es TOML válido, devuelve error y no escribe: editar una config ajena que no se entiende es peor
  que pedir que se haga a mano.
- Uso: `codex --profile brasa`.

**Claude Code**: no se toca `~/.claude` (sus `settings.json` cambiarían todas las sesiones del
usuario). Se crea el lanzador `~/.local/bin/claude-brasa`: un script `sh` con las mismas variables
que imprime `brasa connect claude-code`, que termina en `exec claude "$@"`. Lleva una marca en la
primera línea de comentario. Si ya existe un archivo con ese nombre sin la marca, devuelve error y
no lo pisa. Si `~/.local/bin` no está en el `PATH`, la respuesta lo avisa.

**OpenCode** (`$XDG_CONFIG_HOME/opencode` o `~/.config/opencode`): según la documentación vigente
(opencode.ai/docs/providers y /docs/config, consultada el 2026-10-06), un proveedor compatible con
OpenAI se declara en `opencode.json` con `npm: "@ai-sdk/openai-compatible"`, `options.baseURL` y
`models`.
- Si no hay config, se crea con solo `provider.brasa`.
- Si hay `opencode.json` en JSON estricto, se agrega o actualiza **solo** la clave `provider.brasa`,
  con respaldo previo.
- Si el archivo tiene comentarios (JSONC) o existe `opencode.jsonc`, devuelve error con el texto
  para pegarlo a mano: reescribir JSONC perdería los comentarios.
- No se fija `model` por defecto. No se escribe `limit` porque la documentación consultada no lo
  documenta para proveedores propios.
- Uso: elegir `brasa/<modelo>` en OpenCode.

**Cline**: guarda su configuración dentro de VS Code (ajustes y almacenamiento secreto de la
extensión). No se escribe; la API responde 400 y la GUI no muestra el botón.

**Reglas comunes:**
- Escrituras atómicas (archivo temporal + `rename`).
- Rutas fijas debajo del `HOME` del proceso: ningún camino sale del pedido.
- La respuesta lista los archivos escritos, respaldados y sin cambios, y cómo usar la herramienta.
- El endpoint es un `POST`, así que lo cubre el middleware de origen (`SECURITY.md`, «Pedidos desde el navegador»):
  una página de otro origen no puede dispararlo.

## Consecuencias

- Conectar es repetible: una segunda vez no cambia nada salvo que haya cambiado el contexto o el
  puerto (se reescriben los archivos propios).
- Deshacer: borrar los archivos propios (`brasa.config.toml`, `brasa-models.json`, `claude-brasa`) y
  restaurar el respaldo o quitar la tabla/clave `brasa`.
- Los tests escriben en un `HOME` temporal; nada toca la carpeta real del usuario.
