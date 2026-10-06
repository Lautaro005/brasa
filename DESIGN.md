# Design

Sistema visual de la landing de Brasa (`site/`). Registra lo que ya está construido para que
cualquier superficie futura lo herede. La dirección se eligió con el flujo de `impeccable`
(PRODUCT.md, `concept-seed`, elección de la dirección en la ronda con el usuario).

## Mundo

Un **rack de estudio con patchbay**. El engine es un equipo de outboard: paneles de aluminio
anodizado claro atornillados a un rack oscuro, rótulos grabados, agujas ámbar y luces de estado. El
agente se "parchea" al engine por un cable.

## Paleta

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

## Tipografía

Fuentes del sistema (pinneado por el encargo): `--sans` para prosa, `--mono` para rótulos, cifras y
comandos. Piso de tamaño 12.8 px (`0.8rem`); rótulos en mayúsculas con tracking ≤0.2em solo cuando
son cortos. Jerarquía: `h1` clamp(2.05–3.6rem) → `h3` 1.08rem → `h2` de panel 0.84rem.

## Materiales y composición

- El panel es la unidad: `padding:3px`, borde de 1px, radio 7px, sombra suave de 8px y bisel interior;
  tornillos dibujados en las esquinas (`::before`/`::after`).
- Rack: rieles laterales con agujeros de tornillo (se ocultan por debajo de 760 px).
- Rótulo de sección: una franja con el `h2` (que **es** el título, sin volanta) y una pista corta al lado.
- Ancho máximo 1240px; retícula fluida con `minmax(0,1fr)`; se sostiene hasta 360 px.

## Componentes y estados

- **Patchbay** (interacción firma): cuatro botones `jack` con `aria-pressed` dentro de un `role="group"`
  etiquetado; al elegir uno, el cable SVG (curva cuadrática con `vector-effect="non-scaling-stroke"`)
  se dibuja con `stroke-dashoffset` y el panel de salida muestra `brasa connect <agente>` en una
  región `aria-live`. El cable termina siempre en el jack de entrada porque comparten celda.
- **Vúmetro**: dial SVG con la aguja que se asienta una vez al entrar en pantalla.
- **Bloques del plan**: filas sobre un mismo eje medido; proporciones ilustrativas, nunca cifras.
- **Tiras de especificaciones**: lista de canales, no tarjetas de icono+título+texto.
- **Estados**: hover (brillo y borde), activo (baja 1px), foco (`outline` de tinta + halo claro del
  panel, visible tanto sobre aluminio como sobre el chasis), deshabilitado donde aplique.

## Movimiento

Un solo gesto: el parche (0.62s, ease-out exponencial) y el asentamiento de la aguja (1s). El
contenido se ve por defecto; `prefers-reduced-motion` lo apaga.

## Idiomas

Bilingüe ES/EN con pares de nodos `.es`/`.en` conmutados por una clase en `body` (sin JS se ve
castellano). El JS actualiza lo que el CSS no puede: `<title>`, `meta description`, `aria-label` y
`alt`.

## Restricciones que la próxima superficie debe heredar

- Sin recursos remotos: sin CDN, sin fuentes remotas, sin analytics. Solo enlaces a GitHub y los JPG
  locales de `site/img/`.
- Ninguna cifra de rendimiento ni de memoria: las mediciones van a `docs/bench/baseline.md`.
- WCAG AA, foco visible, teclado, `alt`, y composición hasta 360 px.
