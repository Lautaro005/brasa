# Landing de Brasa (`site/`)

Página de presentación estática y autocontenida. Se abre con doble clic en `index.html`; no hay
paso de build ni dependencias.

## Archivos

- `index.html`: la página (HTML + CSS + JS mínimo, todo embebido).
- `img/`: capturas reales de la GUI, copiadas de `docs/gui/` (`chat-dos-turnos.jpg`,
  `chat-cancelado.jpg`, `estado.jpg`).

## Sin red

La página no carga nada de internet: sin CDN, sin fuentes remotas, sin analytics, sin telemetría.
La tipografía usa las fuentes del sistema del visitante (`ui-sans-serif` y `ui-monospace`), así que
no hay archivos de fuentes que incrustar. Las únicas referencias externas son enlaces a GitHub (el
repositorio, `SECURITY.md`, la licencia y `docs/bench/baseline.md`); el resto son los tres JPG de
`img/`.

## Cifras

No hay ninguna cifra de rendimiento: la regla 6 de `CLAUDE.md` prohíbe afirmar sin medir. La sección
"Rendimiento" enlaza a `docs/bench/baseline.md`, donde están los reportes con chip, RAM, macOS y
commit.

## Idiomas

Bilingüe castellano/inglés con un conmutador (ES/EN). Sin JavaScript se muestra el castellano. La
elección se guarda en `localStorage` dentro de un `try/catch`.

## Accesibilidad

Contraste AA, foco visible, navegación por teclado (el patchbay son botones con `aria-pressed` y la
salida es una región `aria-live`), `alt` en las tres capturas y composición fluida hasta 360 px de
ancho.

## Cómo publicarla (no está hecha)

Se publica sola con GitHub Pages: el workflow `.github/workflows/sitio.yml` sube esta carpeta tal
cual en cada push a `main` que toque `site/` (también se puede lanzar a mano desde Actions). El
repositorio tiene Pages con origen "GitHub Actions"; la URL es https://lautaro005.github.io/brasa/.
