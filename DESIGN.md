---
name: Brasa
description: Engine de inferencia local para Apple Silicon; landing en site/ y GUI embebida en /ui.
colors:
  landing-case: "#17150f"
  landing-panel: "#d7d4cd"
  landing-ink: "#17150f"
  landing-amber: "#b8701a"
  landing-led-on: "#1c7f47"
  gui-surface: "#f5f4f1"
  gui-ground: "#e8e6e1"
  gui-rail: "#dcd9d3"
  gui-field: "#fbfaf8"
  gui-hover: "#ebe8e3"
  gui-ink: "#1d1b18"
  gui-ink-2: "#514c45"
  gui-ink-3: "#635d56"
  gui-line: "#cdc9c1"
  gui-line-2: "#b3ada4"
  gui-cold: "#d3cfc8"
  gui-ember: "#b8410e"
  gui-ember-ink: "#9a360b"
  gui-live: "#d4520f"
  gui-bad: "#9e1b1b"
  gui-bad-bg: "#f6e2de"
  gui-heat-0: "#d3cfc8"
  gui-heat-1: "#6e1a0f"
  gui-heat-2: "#93240c"
  gui-heat-3: "#b5350b"
  gui-heat-4: "#c94a0a"
  gui-heat-5: "#c4620a"
  gui-heat-6: "#b47900"
  gui-seg-weights: "#3a3530"
  gui-seg-kv: "#7a7168"
  gui-seg-workspace: "#a49b90"
  gui-seg-overhead: "#c4bcb1"
  gui-selection: "#f2c9a4"
typography:
  gui-view-title:
    fontFamily: "system-ui, -apple-system, SF Pro Text, Helvetica Neue, sans-serif"
    fontSize: "22px"
    fontWeight: 650
    letterSpacing: "-0.01em"
  gui-figure:
    fontFamily: "system-ui, -apple-system, SF Pro Text, Helvetica Neue, sans-serif"
    fontSize: "24px"
    fontWeight: 650
    letterSpacing: "-0.01em"
    fontFeature: "tnum"
  gui-wordmark:
    fontFamily: "system-ui, -apple-system, SF Pro Text, Helvetica Neue, sans-serif"
    fontSize: "20px"
    fontWeight: 700
    letterSpacing: "-0.02em"
  gui-total:
    fontFamily: "system-ui, -apple-system, SF Pro Text, Helvetica Neue, sans-serif"
    fontSize: "17px"
    fontWeight: 600
    fontFeature: "tnum"
  gui-panel-title:
    fontFamily: "system-ui, -apple-system, SF Pro Text, Helvetica Neue, sans-serif"
    fontSize: "15px"
    fontWeight: 650
  gui-body:
    fontFamily: "system-ui, -apple-system, SF Pro Text, Helvetica Neue, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.45
  gui-note:
    fontFamily: "system-ui, -apple-system, SF Pro Text, Helvetica Neue, sans-serif"
    fontSize: "13px"
    fontWeight: 400
  gui-label:
    fontFamily: "system-ui, -apple-system, SF Pro Text, Helvetica Neue, sans-serif"
    fontSize: "12px"
    fontWeight: 500
  gui-mono:
    fontFamily: "ui-monospace, SF Mono, Menlo, monospace"
    fontSize: "12.5px"
rounded:
  gui-cell: "2px"
  gui-bar: "3px"
  gui-code: "4px"
  gui-control: "6px"
  gui-composer: "10px"
  gui-popover: "8px"
  gui-pill: "999px"
  landing-panel: "7px"
spacing:
  gui-cell-gap: "2px"
  gui-control-gap: "8px"
  gui-panel-top: "14px"
  gui-panel-bottom: "22px"
  gui-column-gutter: "28px"
  gui-main-x: "32px"
  gui-main-x-narrow: "16px"
components:
  gui-button:
    backgroundColor: "{colors.gui-field}"
    textColor: "{colors.gui-ink}"
    rounded: "{rounded.gui-control}"
    padding: "8px 12px"
  gui-button-hover:
    backgroundColor: "{colors.gui-hover}"
  gui-button-primary:
    backgroundColor: "{colors.gui-ink}"
    textColor: "{colors.gui-surface}"
    rounded: "{rounded.gui-control}"
    padding: "8px 12px"
  gui-button-primary-hover:
    backgroundColor: "{colors.gui-ink-2}"
  gui-button-ghost:
    textColor: "{colors.gui-ink-2}"
    rounded: "{rounded.gui-control}"
    padding: "8px 12px"
  gui-button-danger:
    backgroundColor: "{colors.gui-field}"
    textColor: "{colors.gui-bad}"
    rounded: "{rounded.gui-control}"
  gui-button-danger-confirm:
    backgroundColor: "{colors.gui-bad}"
    textColor: "{colors.gui-surface}"
  gui-input:
    backgroundColor: "{colors.gui-field}"
    textColor: "{colors.gui-ink}"
    rounded: "{rounded.gui-control}"
    padding: "7px 9px"
  gui-tag:
    textColor: "{colors.gui-ink-2}"
    rounded: "{rounded.gui-pill}"
    padding: "1px 8px 1px 6px"
  gui-tag-live:
    textColor: "{colors.gui-ember-ink}"
  gui-nav-item:
    textColor: "{colors.gui-ink-2}"
    rounded: "{rounded.gui-control}"
    padding: "7px 10px"
  gui-nav-item-current:
    backgroundColor: "{colors.gui-surface}"
    textColor: "{colors.gui-ink}"
  gui-heat-cell:
    backgroundColor: "{colors.gui-heat-0}"
    rounded: "{rounded.gui-cell}"
    height: "40px"
---

# Design

Brasa tiene dos superficies con dos mundos visuales distintos, ambos aprobados por el usuario:

- **Landing** (`site/`): "rack de estudio con patchbay".
- **GUI embebida** (`crates/daemon/assets/`, servida en `/ui`): "Ceniza y brasa".

No se mezclan: la GUI no hereda el aluminio, los tornillos ni el ámbar de la landing, y la landing no
hereda la hoja de ceniza de la GUI. Una superficie nueva elige a cuál pertenece. Los tokens del
frontmatter llevan el prefijo `landing-` o `gui-`; el frontmatter guarda los valores del tema claro de
la GUI y `.impeccable/design.json` los del tema oscuro.

---

## Landing (`site/`): rack de estudio con patchbay

Sistema visual de la landing de Brasa (`site/`). Registra lo que ya está construido para que
cualquier superficie futura lo herede. La dirección se eligió con el flujo de `impeccable`
(PRODUCT.md, `concept-seed`, elección de la dirección en la ronda con el usuario).

### Mundo

Un **rack de estudio con patchbay**. El engine es un equipo de outboard: paneles de aluminio
anodizado claro atornillados a un rack oscuro, rótulos grabados, agujas ámbar y luces de estado. El
agente se "parchea" al engine por un cable.

### Paleta

| Rol | Valor | Uso |
|---|---|---|
| `--case` | `#17150f` | chasis del rack (fondo de la página) |
| `--panel` / `--panel-hi` / `--panel-lo` | `#d7d4cd` / `#e9e7e2` / `#bdbab2` | cara de aluminio (degradado vertical) |
| `--edge` | `#8b887f` | borde del panel |
| `--ink` / `--ink-2` | `#17150f` / `#46433c` | tinta grabada y texto secundario (AA sobre el panel) |
| `--amber` / `--amber-hi` | `#b8701a` / `#e0a03a` | señal viva: aguja, cable parcheado, relleno del jack |
| `--amber-deep` | `#8a5210` | contorno de estado sobre aluminio (≥3:1) |
| `--led-on` | `#1c7f47` | LED de estado |
| `--paper` / `--paper-2` | `#eae5db` / `#a7a298` | texto sobre el chasis oscuro |

Estrategia: **Comprometida**. El aluminio es la superficie; el ámbar queda reservado para lo vivo y
no se usa como decoración.

### Tipografía

Fuentes del sistema (pinneado por el encargo): `--sans` para prosa, `--mono` para rótulos, cifras y
comandos. Piso de tamaño 12.8 px (`0.8rem`); rótulos en mayúsculas con tracking ≤0.2em solo cuando
son cortos. Jerarquía: `h1` clamp(2.05–3.6rem) → `h3` 1.08rem → `h2` de panel 0.84rem.

### Materiales y composición

- El panel es la unidad: `padding:3px`, borde de 1px, radio 7px, sombra suave de 8px y bisel interior;
  tornillos dibujados en las esquinas (`::before`/`::after`).
- Rack: rieles laterales con agujeros de tornillo (se ocultan por debajo de 760 px).
- Rótulo de sección: una franja con el `h2` (que **es** el título, sin volanta) y una pista corta al lado.
- Ancho máximo 1240px; retícula fluida con `minmax(0,1fr)`; se sostiene hasta 360 px.

### Componentes y estados

- **Patchbay** (interacción firma): cuatro botones `jack` con `aria-pressed` dentro de un `role="group"`
  etiquetado; al elegir uno, el cable SVG (curva cuadrática con `vector-effect="non-scaling-stroke"`)
  se dibuja con `stroke-dashoffset` y el panel de salida muestra `brasa connect <agente>` en una
  región `aria-live`. El cable termina siempre en el jack de entrada porque comparten celda.
- **Vúmetro**: dial SVG con la aguja que se asienta una vez al entrar en pantalla.
- **Bloques del plan**: filas sobre un mismo eje medido; proporciones ilustrativas, nunca cifras.
- **Tiras de especificaciones**: lista de canales, no tarjetas de icono+título+texto.
- **Estados**: hover (brillo y borde), activo (baja 1px), foco (`outline` de tinta + halo claro del
  panel, visible tanto sobre aluminio como sobre el chasis), deshabilitado donde aplique.

### Movimiento

Un solo gesto: el parche (0.62s, ease-out exponencial) y el asentamiento de la aguja (1s). El
contenido se ve por defecto; `prefers-reduced-motion` lo apaga.

### Idiomas

Bilingüe ES/EN con pares de nodos `.es`/`.en` conmutados por una clase en `body` (sin JS se ve
castellano). El JS actualiza lo que el CSS no puede: `<title>`, `meta description`, `aria-label` y
`alt`.

### Restricciones que la próxima superficie debe heredar

- Sin recursos remotos: sin CDN, sin fuentes remotas, sin analytics. Solo enlaces a GitHub y los JPG
  locales de `site/img/`.
- Ninguna cifra de rendimiento ni de memoria: las mediciones van a `docs/bench/baseline.md`.
- WCAG AA, foco visible, teclado, `alt`, y composición hasta 360 px.

---

## GUI (`/ui`): ceniza y brasa

Registrado a partir de lo construido en `crates/daemon/assets/` (`index.html`, `app.css`, `app.js`).
Contrato de dirección: `.impeccable/surfaces/crates-daemon-assets-index-html.md`. Los valores
normativos están en el frontmatter (tema claro, prefijo `gui-`) y en `.impeccable/design.json` (tema
oscuro, rampas, movimiento, cortes).

### Overview

**Creative North Star: "Ceniza y brasa"**

La interfaz es ceniza; solo arde lo que está trabajando, y cuánto arde es la medida. Toda la GUI es
una hoja gris cálida y plana (claro: papel ceniza; oscuro: carbón apagado) con tinta casi negra, y la
única fuente de color es la rampa de incandescencia, que aparece solo cuando hay dato o trabajo en
curso: tok/s, prefill en marcha, generación en streaming, el modelo ocupado. Con el modelo quieto, la
pantalla es casi monocroma; mirarla de reojo alcanza para saber si un agente está generando.

Es una herramienta de operación densa, pensada para quedar abierta en una pestaña mientras Claude Code
o Codex trabajan. La densidad la ordena una sola línea de 1px, no tarjetas ni sombras. Rechaza el
dashboard por defecto de tarjetas con sparklines y un azul de acento.

**Key Characteristics:**
- Una sola hoja plana; las zonas se separan con líneas de 1px (`--line`).
- Color = dato o actividad. Nada decorativo arde.
- Rampa de calor en 7 pasos planos con escalas fijas, nunca relativas a la ventana.
- Fuente del sistema, cifras tabulares en todo lo que se mide; mono solo para comandos, hashes y rutas.
- Tema claro y oscuro según `prefers-color-scheme`, sin conmutador.
- Toda cifra viene de `/api/*` o `/v1/*`.

### Colors

Neutros cálidos de ceniza en planos lisos más una rampa de incandescencia reservada al dato. Cada token
tiene valor claro (frontmatter) y oscuro (sidecar); se declaran como propiedades de `:root` y el tema
oscuro las redefine bajo `@media (prefers-color-scheme: dark)`.

#### Primary
- **Brasa** (`gui-ember`): marca de dato sobre neutro: la línea del pico de memoria medido sobre la
  barra de presupuesto. En oscuro sube a un naranja encendido.
- **Brasa entintada** (`gui-ember-ink`): texto y borde de la etiqueta `live` (prefill o generación en
  curso), con contraste AA sobre la hoja.
- **Llama viva** (`gui-live`): el estado que está ocurriendo ahora: LED del modelo ocupado, casillero de
  prefill en curso, punto de la etiqueta `live`, cursor de streaming del chat. Siempre late (ver
  Motion).

#### Rampa de incandescencia (dato)
- **Ceniza fría** (`gui-heat-0`): segundo sin actividad; fondo de cada casillero.
- **De rojo profundo a ámbar** (`gui-heat-1` a `gui-heat-6`): seis pasos planos de calor. En oscuro la
  rampa termina en blanco cálido (`#f9c968`); en claro se detiene en un ámbar oscuro (`#b47900`) para
  que el último paso siga leyéndose sobre papel ceniza. Es la única divergencia del contrato (que pedía
  terminar en blanco cálido en ambos temas): el build gana por legibilidad.

#### Neutral
- **Papel ceniza** (`gui-surface`): la hoja de contenido; también el texto del botón primario.
- **Ceniza de fondo** (`gui-ground`): fondo de bloques de código, mensajes del usuario y `pre`.
- **Riel** (`gui-rail`): la barra lateral de navegación.
- **Campo** (`gui-field`): fondo de botones, campos y compositor.
- **Hover** (`gui-hover`): respuesta al puntero en nav y botones.
- **Tinta** (`gui-ink`), **tinta 2** (`gui-ink-2`), **tinta 3** (`gui-ink-3`): texto principal,
  secundario y notas/rótulos de tabla. `gui-ink` es también el foco en claro, el botón primario y la
  barra de Brasa en los benchmarks.
- **Línea** (`gui-line`) y **línea 2** (`gui-line-2`): la retícula de 1px y los bordes de controles.
- **Ceniza fría** (`gui-cold`): pista vacía de barras y LED apagado.
- **Segmentos de memoria** (`gui-seg-weights`, `gui-seg-kv`, `gui-seg-workspace`, `gui-seg-overhead`):
  cuatro grises escalonados para pesos, caché KV, workspace y overhead. En oscuro la escala se invierte
  (pesos el más claro). Son grises a propósito: la memoria reservada no es trabajo.
- **Error** (`gui-bad`, `gui-bad-bg`): servidor desconectado, fallos, botón de detener confirmado y
  barra de memoria al 95 % del presupuesto o más (todos los segmentos pasan a `gui-bad`).
- **Selección** (`gui-selection`): `::selection`.

#### Named Rules
**The Solo Arde Lo Activo Rule.** `gui-ember`, `gui-live` y la rampa `gui-heat-1..6` marcan únicamente
dato medido o trabajo en curso. Nunca acento de marca, nunca botón, nunca enlace, nunca decoración. El
botón primario es tinta, no brasa; la barra propia en benchmarks es tinta, no brasa.

**The Escala Fija Rule.** El calor se mide contra escalas fijas, no contra el máximo de la ventana.
Decode: pasos de 10 tok/s (nivel 1 por encima de 0.05 y menos de 10, ..., nivel 6 a partir de 50).
Prefill: pasos de 100 tok/s de prompt procesado (nivel 6 a partir de 500). El nivel 0 es ceniza. La
leyenda lo dice en pantalla: "decode: pasos de 10 (≥ 50) · prefill: pasos de 100 (≥ 500)".

**The Plano Rule.** Color plano, sin degradés. La única excepción construida es el rayado del valor
inválido en los benchmarks (`repeating-linear-gradient` de `gui-bad`), que es un patrón de estado, no un
degradé.

### Typography

**Display Font:** ninguna; no hay fuente de display.
**Body Font:** fuente del sistema (`system-ui, -apple-system, "SF Pro Text", "Helvetica Neue", sans-serif`)
**Label/Mono Font:** `ui-monospace, "SF Mono", Menlo, monospace`

**Character:** Una sola escala chica y pareja, como un folio: la jerarquía sale del peso (650) y del
tamaño en pasos cortos, no de una tipografía de titular. Todas las cifras son tabulares.

#### Hierarchy
- **Cifra de memoria** (650, 24px, -0.01em, tabular): la ocupación actual en el bloque de memoria.
- **Título de vista** (650, 22px, -0.01em): el `h1` de cada vista, con su subtítulo al lado en la misma
  línea base.
- **Wordmark** (700, 20px, -0.02em): "Brasa" en el riel.
- **Total** (600, 17px, tabular): los contadores acumulados "Desde el arranque".
- **Título de panel** (650, 15px): el `h2` de cada zona; también el veredicto del plan.
- **Body** (400, 14px, 1.45): texto base; los mensajes del chat a 1.55 y como máximo 72ch.
- **Nota** (400, 13px): notas de panel, pies, mensajes de control, botones (500, 13px).
- **Rótulo** (400–500, 12px): encabezados de tabla, etiquetas de carril y de campo, ejes, leyenda,
  etiquetas `tag`. Es el piso de tamaño.
- **Mono** (12.5px o 0.9em): comandos, hashes, rutas y bloques de configuración de agentes.

#### Named Rules
**The Cifras Tabulares Rule.** Toda cifra que se actualiza (tok/s, GiB, contadores, tiempos) va con
`font-variant-numeric: tabular-nums` para que no baile al refrescar.

**The Mono Solo Para Máquina Rule.** La mono es para lo que se copia a una máquina (comandos, hashes,
rutas, JSON/TOML); los números y rótulos van en la sans.

### Layout

Barra lateral fija de 224px (`position: sticky`, alto de la ventana) y contenido fluido
(`minmax(0, 1fr)`) con relleno de 28px arriba y 32px a los lados, ancho máximo de vista 1360px. Cada
vista: título y subtítulo, y debajo una pila de zonas separadas por la línea de 1px superior de cada
una (14px arriba, 22px abajo).

- **Monitor:** la tira de calor ocupa el ancho completo arriba; debajo, dos columnas (fluida | 360px)
  divididas por una línea vertical de 1px con 28px a cada lado: pedidos y totales a la izquierda,
  memoria y modelo a la derecha.
- **Chat:** columna de conversación fluida | 300px de estadísticas y parámetros (sticky), divididas por
  línea de 1px.
- **Benchmarks, Agentes:** rejillas `auto-fit`/`auto-fill` (mínimos 340px y 380px) con 28px de
  separación entre columnas y sin separación vertical: las líneas de cada zona hacen de borde.
- **Totales:** rejilla `auto-fill` de 150px separada por líneas verticales de 1px; la línea de la
  primera columna de cada fila se recorta.

Responsive:
- **≤1100px:** Monitor y Chat pasan a una columna; el lateral se vuelve una rejilla de dos columnas
  debajo, sin línea vertical.
- **≤820px:** el riel se vuelve una barra superior sticky (marca, nav horizontal con scroll y desvanecido
  en el borde derecho, estado del modelo con su LED); se ocultan los íconos de la nav, el meta de la marca
  y el nombre del modelo. Relleno 20px/16px. Todo pasa a una columna; los carriles de calor pierden la
  columna de rótulos (rótulo arriba, alineado a la izquierda), el carril de decode baja a 32px de alto y el
  hueco entre casilleros a 1px.

### Elevation & Depth

Plana. No hay sombras de elevación: la profundidad es tonal (riel más oscuro que la hoja, `gui-ground`
detrás de código y mensajes del usuario) y estructural (líneas de 1px). Las únicas `box-shadow` son
anillos `inset` de 1–2px que dibujan un borde: el ítem actual de la nav, el contorno del LED y el
casillero del segundo actual en la tira de calor.

**Excepción: lo que se superpone.** Lo que flota sobre la hoja necesita separarse de ella, así que
es lo único con sombra de elevación:
- la lista desplegable de los campos propios: radio 8px, sombra
  `0 2px 4px rgb(30 20 10 / 0.08), 0 8px 24px rgb(30 20 10 / 0.14)`;
- la lista de conversaciones y el panel del chat cuando se abren superpuestos en pantallas angostas:
  sombra lateral `±4px 0 24px rgb(0 0 0 / 0.18)`;
- la perilla del interruptor: `0 1px 2px rgb(0 0 0 / 0.25)`.

En la nav angosta, una máscara (`#000` → transparente en los últimos 28px) avisa que hay más
pestañas.

#### Named Rules
**The Una Hoja Rule.** Una sola hoja: las zonas se separan con una línea de 1px, sin tarjetas, sin
fondos de panel distintos y sin sombras.

### Shapes

Radios chicos y funcionales. Casilleros de calor y muestras de leyenda 2px (1px en la mini tira del
chat); barra de memoria 3px; código en línea 4px; botones, campos, nav y bloques `pre` 6px; compositor
y burbuja del usuario 10px (la burbuja con la esquina inferior derecha a 2px); etiquetas en píldora
(999px); LED circular de 10px. Íconos de trazo de 1.5px a 18px (16px dentro de botones) desde un sprite
SVG inline.

### Components

#### Tira de calor (firma)
Dos carriles de 120 casilleros, uno por segundo de los últimos 120 s, alimentados por `/api/activity`
cada segundo: prefill (16px de alto) y decode (40px). El color de cada casillero es su nivel en la
escala fija (ver Colors). Un segundo con prefill todavía en curso se pinta `gui-live` y late. El
casillero actual lleva un anillo de 1px (`gui-line-2`). Leyenda de 7 muestras a la derecha del título,
eje de tiempo debajo (−120 s … ahora) y un resumen en texto que reemplaza a la gráfica para lectores de
pantalla (los carriles van `aria-hidden`). Cada casillero tiene `title` con su cifra. El chat repite una
versión de 60 casilleros de 18px en su lateral.

#### LED de estado del modelo
Círculo de 10px en el riel, con el nombre del estado al lado. `loaded`: tinta 2 sólida. `busy`:
`gui-live` latiendo (1.6s). `loading`: `gui-heat-3` latiendo rápido (0.8s). `paused`: aro de 2px.
`stopped`/`offline`: aro de 1.5px vacío. Apagado por defecto: ceniza fría con contorno.

#### Barra de presupuesto de memoria
Barra de 14px apilada en cuatro segmentos grises (pesos, KV, workspace, overhead) separados por 1px del
color de la hoja, sobre pista `gui-cold`; una marca vertical de 2px en `gui-ember` señala el pico
medido. A 95 % del presupuesto o más (o si el plan no entra), todos los segmentos pasan a `gui-bad`.
Leyenda en una lista `dt/dd` con un cuadrado de 10px por segmento. Se reusa en la vista Plan.

#### Buttons
- **Shape:** 6px, borde 1px `gui-line-2`, 8px × 12px, 500 13px, ícono de 16px.
- **Default:** fondo `gui-field`. Hover: `gui-hover` y borde `gui-ink-3`. Activo: baja 1px.
  Deshabilitado: opacidad 0.45 y `not-allowed`. Ocupado: `aria-busy` con cursor de progreso.
- **Primary:** tinta sobre papel invertido (`gui-ink` de fondo); hover a `gui-ink-2`. Una por formulario
  (Enviar, Calcular, Generar).
- **Ghost:** transparente, texto `gui-ink-2` (Borrar conversación).
- **Danger:** texto `gui-bad`; "Detener servidor" pide confirmación en dos pasos: el primer clic lo
  rellena de `gui-bad` con "¿Detener? Confirmar" durante 4 s.

#### Tags de estado
Píldora de 12px con borde 1px y un punto de 6px del color del texto. `live` (prefill, generando):
`gui-ember-ink` con punto `gui-live` latiendo. `bad`: `gui-bad`. `ok`: punto de tinta. `cold`: punto
tenue.

#### Inputs / Fields
- **Style:** fondo `gui-field`, borde 1px `gui-line-2`, 6px, 7px × 9px; rótulo de 12px encima.
- **Focus:** contorno de 2px `gui-focus` sin separación y borde del mismo color. Hover: borde `gui-ink-3`.
- **Checkbox:** 16px con `accent-color` de tinta.
- **Compositor del chat:** caja de 10px que contiene el `textarea` sin borde y la barra de acciones; el
  foco se marca en la caja (borde `gui-ink-3`).

#### Tables
Tablas de datos a ancho completo, cifras tabulares, encabezado de 12px en `gui-ink-3`, filas separadas
por 1px, números alineados a la derecha, filas inactivas en `gui-ink-3`. Envueltas en un contenedor con
scroll horizontal.

#### Navigation
Riel con wordmark, seis vistas (Monitor, Chat, Modelos, Benchmarks, Plan, Agentes) con ícono de 18px y
el estado del modelo abajo. Ítem: 500, texto `gui-ink-2`, 7px × 10px, 6px. Hover `gui-hover`. Actual
(`aria-current="page"`): fondo de la hoja con anillo de 1px `gui-line-2`. Por debajo de 820px, barra
superior horizontal (ver Layout).

#### Chat
Mensajes del asistente sin caja, a 72ch; los del usuario a la derecha sobre `gui-ground` con borde de
1px. El razonamiento va en un `details` con línea izquierda de 1px. El streaming muestra un cursor de
bloque `gui-live` que parpadea en dos pasos.

#### Estados vacíos y de error
Vacíos: texto de hasta 60ch con la primera frase en negrita, sin ilustración. Desconectado: franja
`gui-bad-bg` con texto y borde `gui-bad` sobre la vista.

#### Motion
Transiciones de 150ms `ease-out` en fondo, color y borde; 100ms en el desplazamiento de 1px del botón
activo. Un único latido, `ember` (opacidad a 0.55 a mitad de ciclo), aplicado solo a lo que está vivo:
LED ocupado 1.6s, LED cargando 0.8s, casillero de prefill y punto de `tag.live` 1.2s, cursor de
streaming 1s en dos pasos. `prefers-reduced-motion: reduce` apaga todas las animaciones y transiciones.

### Do's and Don'ts

#### Do:
- **Do** mostrar solo cifras que vienen de `/api/*` o `/v1/*` (estado, métricas, actividad, modelos,
  bench, plan, agentes y la propia respuesta del chat). Si el dato falta, mostrar "—".
- **Do** pintar con `gui-live`/`gui-ember`/`gui-heat-1..6` solo dato medido o trabajo en curso, y
  dejar que lo vivo lata con `ember`.
- **Do** medir el calor contra las escalas fijas: decode en pasos de 10 tok/s hasta ≥50, prefill en
  pasos de 100 tok/s hasta ≥500.
- **Do** separar zonas con la línea superior de 1px (`gui-line`) y columnas con una línea vertical de 1px.
- **Do** usar cifras tabulares en todo lo que se actualiza y acompañar cada gráfica con un resumen en
  texto.
- **Do** definir cada color nuevo en ambos temas (claro en `:root`, oscuro bajo
  `prefers-color-scheme: dark`).
- **Do** mantener todo embebido: sin CDN, sin fuentes remotas, sin telemetría.

#### Don't:
- **Don't** usar tarjetas, fondos de panel ni sombras de elevación para agrupar.
- **Don't** usar brasa o la rampa de calor como acento de marca, en botones, enlaces o decoración.
- **Don't** escalar el calor al máximo de la ventana visible, ni usar degradés en la rampa.
- **Don't** mostrar estimaciones, cifras calculadas en el cliente sin dato de origen o cifras de
  ejemplo.
- **Don't** usar sparklines ni un azul de acento.
- **Don't** traer el aluminio, los tornillos ni el ámbar de la landing a la GUI.
