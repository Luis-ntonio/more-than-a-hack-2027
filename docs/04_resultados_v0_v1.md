# 04 · Resultados v0 y v1

```bash
docker compose run --rm dev python -m possession.make_submission --model v0
docker compose run --rm dev python -m possession.make_submission --model v1
docker compose run --rm dev python evaluation/run_evaluation.py --bootstrap 1000
docker compose run --rm dev python analysis/sweep_v1.py     # sensibilidad de v1
```

Las salidas van a `evaluation/results/v0.csv` y `v1.csv` (gitignored). Ambas pasan
`validate_submission.py` y son **reproducibles bit a bit**: el mismo SHA-256 en ejecuciones repetidas
dentro del contenedor.

## Qué hace cada versión

| | v0 | v1 |
|---|---|---|
| Equipo por frame | jugador más cercano (binario) | evidencia continua: `tanh(Δdistancia)` ponderada por `exp(−d/2,5 m)` |
| Pelota no visible | se arrastra el último equipo | sin evidencia de equipo; el HMM mantiene el estado |
| Pelota muerta | solo regla de pre-saque | P(D) heurística: hueco de pelota largo + jugadores lentos + cámara quieta + pelota fuera del campo |
| Temporalidad | voto mayoritario ±2 s | HMM de 3 estados (H/A/D) con Viterbi y costes de transición (H↔A 25, ↔D 15 nats) |
| Reglas fijas | `D` antes del saque inicial de cada tiempo (la 2ª parte se califica unos 2 min antes) | igual |

Los parámetros de v1 están fijados **a mano a partir del EDA** y no se ajustaron contra la ventana
pública (`possession/models.py`, `V1_PARAMS`).

## Ventana pública (5 min, 17 posesiones): solo para comprobar que todo es correcto

| submission | spellF1 | IC 90 % bootstrap | macroF1 | F1 vivo/muerto | posesiones (GT 17) |
|---|---|---|---|---|---|
| v0 | 0.431 | 0.287–0.574 | 0.527 | 0.891 | 22 |
| **v1** | 0.389 | 0.238–0.541 | **0.762** | **0.921** | 20 |
| baseline_smooth125 | 0.424 | 0.267–0.594 | 0.581 | 0.899 | 16 |
| baseline_smooth50 (suelo) | 0.326 | 0.214–0.447 | 0.621 | 0.899 | 26 |

**Los intervalos se solapan por completo: con 17 posesiones esta ventana no puede ordenar v0, v1 y
los baselines.** Lo único que sí muestra con claridad es que v1 resuelve mucho mejor la pelota
muerta: el macroF1 pasa de 0,53–0,62 a 0,76.

## Partido completo (sin GT): diagnósticos agregados

| | v0 | v1 |
|---|---|---|
| % de frames `D` | 1,8 % | 48,5 % |
| posesiones | 369 | 278 |
| duración media y mediana | 16,9 s / 8,5 s | 11,8 s / 8,3 s |
| posesiones de menos de 2 s | 42 | 7 |

**Por qué preferimos v1 aunque la ventana pública no lo confirme.** En la ventana pública solo el 20 %
de los frames son `D`. El tiempo efectivo de juego típico en LaLiga ronda 55 de ~100 minutos, así que
en el partido completo cabe esperar un 40–50 % de pelota muerta. v0 casi no marca `D`, de modo que
fusiona posesiones separadas por paradas (`H D H` → una sola `H`) y asigna equipo a cada parada.
Esto penaliza el spellF1 justo en la parte del partido que la ventana pública no representa.

## Sensibilidad de v1 (`analysis/sweep_v1.py`)

| cambio | % D | posesiones |
|---|---|---|
| por defecto | 48,5 % | 278 |
| coste H↔A 15 / 40 | 48,4 % / 48,6 % | 341 / 249 |
| sesgo D −3 / −1 | 41,9 % / 55,6 % | 289 / 284 |
| coste ↔D 8 / 30 | 48,6 % / 48,3 % | 322 / 261 |

El número de posesiones responde sobre todo al coste H↔A, y la cantidad de `D` al sesgo de pelota
muerta. **Esos dos parámetros son los que hay que fijar con el conjunto de validación etiquetado a
mano** (sección 4 de la propuesta). Sin él, cualquier elección es una apuesta.

## Próximos pasos

1. **Etiquetado de validación** (bloquea todo lo demás): con 4–6 bloques de 5 min se pueden fijar el
   coste H↔A y el sesgo D con evidencia.
2. **v2**: un clasificador vivo/muerto entrenado, en lugar de la heurística, y que el equipo que
   ejecuta el saque determine la posesión al reanudar.
3. **v3**: modelo de geometría colectiva con pseudoetiquetas, para inferir el equipo durante los
   huecos de pelota (ahora v1 solo arrastra el estado).
