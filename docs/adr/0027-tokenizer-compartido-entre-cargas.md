# ADR 0027 — Tokenizer compartido entre cargas

Estado: aceptada

## Contexto

Con `brasa serve qwen3-4b-q4 --ctx 16384 --kv q8_0`, cada ciclo `POST /api/model/load` →
`POST /api/model/idle` (ADR 0025) dejaba ~26 MB más de huella en idle (medido en M1 Pro 16 GB,
macOS 27.0 26A428: 156 → 183 → 209 → 235 → 261 → 287 MB). En `footprint` crecía "Malloc Large"
(+8 regiones por ciclo); los buffers de Metal no.

Diagnóstico (ejemplo temporal que repite cargar y soltar en un solo proceso, 5 ciclos):

| Qué se repite                                   | Huella tras cada ciclo (MB)       |
|-------------------------------------------------|-----------------------------------|
| `Session::load_with_budget` completo            | 61 → 152 → 179 → 206 → 234        |
| solo `Tokenizer::from_dir`                      | 48 → 89 → 115 → 143 → 170         |
| solo parsear `tokenizer.json` a `serde_json`    | 34 → 64 → 91 → 118 → 145          |
| `Context::new` + `Qwen3::load` (Context nuevo)  | 17,6 → 17,9 → 18,0 → 18,4 → 18,6  |
| `Qwen3::load` con un Context compartido         | 17,8 → 17,8 → 17,8 → 17,9 → 17,9  |
| `Budget::this_machine` / `Context::new` solos   | constante (4,5)                   |

`heap` sobre el daemon en idle muestra solo ~13 MB vivos en malloc; `vmmap` muestra la diferencia
como "Malloc Large (empty)": memoria **ya liberada** que el allocator del sistema no devuelve ni
reutiliza en la carga siguiente. No es un leak del código: un bucle que solo hace `vec![0u8; 11 MB]`
y un `Vec` que crece también sube ~5–7 MB por vuelta, y con `MallocSpaceEfficient=1` el ciclo del
tokenizer queda plano (4,1 MB). `malloc_zone_pressure_relief` no recupera nada (devuelve 0). El
sistema la recupera en algún momento: en una corrida de 10 ciclos la huella subió hasta 313 MB y
después bajó sola a 177 y a 121 MB, sin patrón controlable. Mientras tanto cuenta en la huella,
que es lo que mira jetsam y lo que informa `/api/status`.

Descartado con medición: autorelease pool alrededor de la carga (sin cambio: +26 MB por ciclo
igual), Context/librería MSL nueva por carga (~0,2 MB por ciclo, dentro del ruido del criterio) e
hilos nuevos (las 2 regiones de Stack son un hilo del pool bloqueante de tokio, que no crece).

## Decisión

`Session` toma el tokenizer de una caché del proceso en `brasa-runtime` (`shared_tokenizer`):
`tokenizer.json` se parsea una sola vez por carpeta de modelo y las cargas siguientes reciben un
`Arc<Tokenizer>`. La clave es la carpeta canonicalizada; si cambian el tamaño o la fecha de
`tokenizer.json` o `tokenizer_config.json`, la entrada se reemplaza.

`idle` sigue soltando pesos, KV y Context Metal; lo que queda vivo es el tokenizer (~13 MB en
malloc), que no es memoria de GPU.

No se comparte el `Context` entre cargas: costaría hacerlo `Send` o vivir fuera de la `Session`, y
la ganancia medida (~0,2 MB por ciclo) no lo justifica hoy.

## Consecuencias

- Medido en el daemon, 10 ciclos `load` → `idle` (300 ms tras cada idle): 155 MB (ciclo 2) →
  157 MB (ciclo 10), "Malloc Large" fijo en 81 MB / 21 regiones. Antes: 183 → 313 MB en los
  ciclos 2 a 7, +8 regiones por ciclo.
- Cualquier asignación grande y transitoria repetida en caliente (no solo el tokenizer) tiene el
  mismo efecto sobre la huella en este allocator. Al agregar trabajo por carga o por pedido,
  medir la huella con varios ciclos, no solo el pico.
