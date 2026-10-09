# ADR 0034 — Storage manager: espacio de la carpeta de modelos

Estado: aceptada (2026-10-09)

## Contexto

Desde ADR 0031 la app baja pesos de 2,2–2,3 GiB por modelo (y `--desde-fuente`, 7,6 GiB de
safetensors) a una carpeta elegible, y una descarga cancelada deja `.part` para reanudar. Faltaba
ver qué ocupa cada cosa, no quedarse sin disco por una descarga (el daemon solo comparaba con el
espacio libre; `brasa pull` no comparaba nada) y limpiar descargas abandonadas. Lo que ya existía y
se reutiliza: `dirs::free_bytes` (`statvfs`), las reglas de borrado de `local::resolve_child` (sin
symlinks, ni en la base ni en el modelo), la reanudación por `Range` de `pull` y el 507 de
`POST /api/models/pull`.

## Decisión

**Módulo `brasa_catalog::storage`**, usado por el CLI y el daemon:

- **Inventario** de la carpeta de modelos sin seguir symlinks (ni contarlos): modelos completos
  (carpeta con `model.brasa`), descargas a medias (cada `.part` con su tamaño y antigüedad por
  `mtime`) y lo demás, clasificado solo para informar: safetensors de origen (`hf_dir` de un
  manifiesto), descarga sin terminar, carpeta o archivo no reconocido, symlink (con su destino).
  Volumen con `statvfs` del ancestro existente más cercano (la carpeta predeterminada se crea con
  la primera descarga).
- **Reserva**: bytes del volumen que una descarga no puede usar. Chequeo previo: si
  `falta_bajar > libre − reserva`, se rechaza con un mensaje que dice cuánto hace falta, cuánto hay
  libre, la reserva, lo disponible y cuánto falta. `falta_bajar` descuenta lo que ya está en disco
  (archivos completos y `.part`), así que reanudar pide solo el resto. Si el volumen no se puede
  leer no se bloquea (la escritura fallaría igual con el error del sistema).
- **Limpieza** de `.part` con al menos N días: dry-run por defecto. Lo único que borra son `.part`
  regulares y las carpetas que quedan vacías (`remove_dir`, que falla si tienen algo). Antes de
  borrar vuelve a mirar cada archivo: tiene que seguir siendo un `.part` regular, seguir siendo
  viejo (si se reanudó entre el recorrido y el borrado, se salta) y su ruta real tiene que ser
  exactamente la recorrida dentro de la carpeta. Una carpeta de modelos que pasa por un symlink se
  rechaza salvo `--seguir-symlink-base` (la regla de `brasa rm`). El daemon además excluye la
  carpeta de la descarga en curso.

**Configuración** (`[storage]` del archivo de ADR 0022, `CONFIG_KEYS` suma `storage`):

```toml
[storage]
reserve_gib = 2            # por defecto 2 GiB; 0 la desactiva; tope 1024
partial_max_age_days = 7   # por defecto 7; mínimo 1
```

El umbral mínimo de 1 día hace que la limpieza nunca alcance una descarga en curso aunque no sepa
de ella (`brasa storage clean` en una terminal mientras el daemon baja): una descarga activa
escribe su `.part` todo el tiempo. `brasa config show` muestra los dos valores con su origen.

**CLI**: `brasa storage [--json]` (estado) y `brasa storage clean [--apply] [--dias N]
[--seguir-symlink-base]`. `brasa pull` hace el chequeo de espacio antes de descargar y su
`--dry-run` informa lo que falta bajar y lo disponible. Borrar un modelo sigue siendo `brasa rm`.

**API del daemon** (errores como `{"error": "..."}`; `POST`/`DELETE` pasan por `origin.rs`):

| Método y ruta | Qué hace |
|---|---|
| `GET /api/storage` | el JSON de `brasa storage --json` más `loaded`/`deletable` por modelo, `served_model`, `pull` (descarga en curso) y `config` |
| `POST /api/storage/clean {apply?, older_than_days?}` | dry-run salvo `"apply": true`; 422 con un campo desconocido, 400 con umbral 0; 409 si la base pasa por un symlink |
| `DELETE /api/storage/models/{name} {"confirm": "<name>"}` | 400 sin el nombre exacto o si `resolve_child` lo rechaza (symlink, `..`, sin `model.brasa`); 404 si no existe; 409 si es el modelo que sirve el servidor o se está descargando |
| `POST /api/models/pull` | el 507 usa el chequeo con la reserva |
| `GET /api/models` | suma `reserve_bytes` y `available_bytes` |

**GUI** (pantalla Modelos, sección "Almacenamiento"; ver DESIGN.md): barra del volumen en grises con
la reserva como hueco con borde, que pasa a `--bad` si lo libre no supera la reserva; modelos con
**Borrar** en dos pasos (el primer clic arma, 4 s); descargas a medias con su antigüedad; **Limpiar
descargas viejas** en dos pasos donde el primer clic pide el dry-run y muestra qué borraría; lo no
reconocido se lista sin acciones. Todo con nodos (el test de assets ahora prohíbe `.innerHTML`,
`.outerHTML`, `insertAdjacentHTML` y `document.write`).

## Alternativas

- **Reserva como porcentaje del disco**: en un disco de 2 TB, un 5 % son 100 GiB que nadie
  necesita reservar; en uno de 256 GB, 12 GiB. Un número fijo es más fácil de explicar.
- **Borrar también las carpetas de descargas sin terminar** (con los tokenizers ya bajados): son
  pocos MB y borrarlas obligaría a decidir qué es "de Brasa" en una carpeta que el usuario puede
  compartir con otras cosas. Se informan y se borran solo si quedan vacías.
- **Chequear el espacio durante la descarga**: el chequeo previo cubre el caso común; si otro
  proceso llena el disco a mitad, la escritura falla con el error del sistema, el `.part` queda y
  se reanuda.
- **Edad por `atime`** o por fecha de inicio: `mtime` es lo que actualiza una descarga activa.

## Consecuencias

- Una descarga ya no puede dejar el volumen con menos de 2 GiB libres (por defecto). En una Mac con
  poco disco esto puede rechazar una descarga que antes entraba justo: el mensaje dice cuánto falta
  y cómo bajar la reserva.
- Verificado sin red ni GPU: tests unitarios del módulo (symlinks adentro y como base, exclusión,
  carpetas vacías, re-verificación), tests de API con un Hugging Face falso local (507 por la
  reserva sin tocar la red; descarga, cancelación, limpieza que respeta el `.part` reciente y
  reanudación verificada por sha256) y la GUI contra un daemon con engine simulado. La descarga real
  desde huggingface.co con el storage manager queda pendiente de autorización del usuario.
