# 07 · Entrenamiento con datos externos

**Por qué existe este documento.** La aclaración de la organización del 3 de noviembre de 2026 tiene
prioridad sobre `RULES.md` y sobre lo anterior: el material del partido provisto se puede usar como
**entrada**, pero no para entrenar, ajustar, calibrar ni validar. Las etiquetas a mano quedan
prohibidas, incluso para validar. Si un número del partido cambia un parámetro del pipeline, es
calibración.

Antes de esa aclaración el equipo usó otro enfoque, con etiquetas a mano sobre el partido y parámetros
ajustados con él. Ese enfoque está **retirado**: su código, sus documentos y sus cifras se quitaron del
repositorio, y las etiquetas nunca estuvieron en él. La solución está en el paquete `external/`, que no
lee etiquetas a mano, GT ni resultados anteriores.

## Flujo

```bash
docker compose run --rm dev python -m external.idsse download   # 7 partidos, ~2,6 GB
docker compose run --rm dev python -m external.idsse parse
docker compose run --rm dev python -m external.skillcorner download
docker compose run --rm dev python -m external.skillcorner parse
# las descargas se comprueban contra external/downloads.sha256
docker compose run --rm dev python -m external.train            # 5 partidos, valida en 2
docker compose run --rm dev python -m enrich.merge              # balón enriquecido
docker compose run --rm dev python -m external.train --final    # 7 partidos + SkillCorner en el de equipo
docker compose run --rm dev python -m external.predict          # evaluation/results/external.csv
docker compose run --rm dev python evaluation/validate_submission.py evaluation/results/external.csv
```

| pieza | qué hace |
|---|---|
| `external/idsse.py` | descarga IDSSE y lo convierte a arrays con `BallPossession` y `BallStatus` |
| `external/features.py` | 149 variables que existen en las dos fuentes: distancias al balón, jugadores cerca, balón, toques, ventanas |
| `external/degrade.py` | hace que el tracking óptico de IDSSE se parezca a un tracking automático, con relleno simulado del detector |
| `external/train.py` | dos LightGBM y un HMM con costes sacados de las etiquetas de IDSSE; valida en partidos retenidos |
| `external/predict.py` | aplica el modelo al partido, que entra solo como dato |

## Datos

**IDSSE / DFL** (Bassek et al. 2025, CC-BY 4.0): 7 partidos de la Bundesliga, ~11 h, tracking óptico a
25 fps. Figshare responde a los scripts con un desafío anti-bots, así que se usa el espejo oficial
`pysport/idsse-data` en una revisión fija. Los IDs son `J03WMX`, `J03WN1`, `J03WOH`, `J03WOY`, `J03WPY`,
`J03WQQ` y `J03WR9`.

Por qué sus etiquetas sirven: `CHALLENGE.md` dice que la GT se deriva mecánicamente de un tracking
corregido, con `D` donde ese tracking marca el balón muerto y, si no, el equipo que nombra como dueño.
`BallStatus` y `BallPossession` tienen esa misma estructura.

- `BallPossession` nunca está vacío, igual que la regla del reto.
- Hay 3 061 posesiones, unas 4–5 por minuto. El 73 % dura menos de 8 s y el 30 % menos de 2 s.
- El balón está muerto en el 41–52 % de los frames.

**Riesgo sin resolver:** son proveedores distintos, y su definición de "dueño" puede variar, sobre
todo en balones divididos. No se puede medir contra la GT.

## Decisiones para no calibrar con el partido

- **Degradación:** la intensidad se sortea en rangos amplios cada 5 minutos, sin usar cifras del
  partido, ni siquiera el 41 % de balón perdido que publica `DATA.md`. Se busca que el modelo aguante
  todo el rango.
- **Radios y ventanas:** son elecciones físicas, y solo se ajustan contra IDSSE.
- **Costes del HMM:** salen de las transiciones de las etiquetas de IDSSE.
- **Dirección de ataque:** se calcula sola, con la posición media de cada equipo en cada parte.
- **Saque inicial:** se usan las anclas que publica la organización en `games.py`.
- **`external/predict.py`** solo imprime chequeos de formato. El CSV no se puntúa con `run_evaluation.py`.

## Resultados (solo IDSSE)

Entrenado con 5 partidos y evaluado en 2 retenidos (`J03WOY`, `J03WR9`):

| modelo | spellF1 limpio | macroF1 limpio | spellF1 degradado | macroF1 degradado |
|---|---|---|---|---|
| jugador más cercano | 0,101 | 0,816 | 0,070 | 0,735 |
| LightGBM frame a frame | 0,337 | 0,901 | 0,190 | 0,815 |
| **LightGBM + HMM** | **0,596** | 0,902 | **0,441** | 0,819 |

- Pesar las emisiones del HMM entre ×0,5 y ×2 cambia el spellF1 menos de 0,01, así que se queda en ×1.
- Las cifras no predicen el resultado en el partido real: el nivel de degradación es una suposición.

## Detector de balón

Los pesos, el recorte y los umbrales se eligen en vídeo externo, sin usar el partido.

```bash
docker compose run --rm dev python -m external.soccertrack videos
docker compose run --rm gpu python -m external.detector_eval detect --name original_tiles640
docker compose run --rm dev python -m external.pbdet
docker compose run --rm gpu python -m external.finetune_detector
docker compose run --rm dev python -m external.detector_eval report --thresholds
```

**Banco de pruebas: SoccerTrack v2** (CC-BY 4.0), 4 partidos en panorámica 4096×1080 a 25 fps. No trae
cajas de balón y su trayectoria se interpola entre eventos, así que se puntúa solo en 2 000 eventos
`PASS` y `DRIVE`, donde el balón está anclado y en el suelo. Los píxeles pasan al campo con el modelo
ojo de pez, la homografía y una corrección polinómica sobre los 65 puntos clave de cada partido
(error dejando uno fuera: mediana 0,33–0,56 m). El máximo de acierto aparece con desfase de 0–5 frames
en los 4 partidos, lo que confirma que la métrica mide algo real.

**Afinado: PBDet-Soccer** (Zenodo 21821917, CC BY 4.0), 2 457 cajas de balón en train, imágenes de
1280 px. `external/finetune_detector.py` parte de los pesos originales y entrena 30 épocas a 640 px.

| detector | algún candidato | top1 < 2 m | top1 < 5 m | top5 < 2 m | top5 < 5 m |
|---|---|---|---|---|---|
| **original, `tiles640`** | 85,7 % | **5,1 %** | **7,0 %** | **9,6 %** | **13,7 %** |
| original, `full1280` | 66,6 % | 0,8 % | 1,8 % | 1,0 % | 2,4 % |
| afinado con PBDet, `tiles640` | 41,9 % | 1,0 % | 1,4 % | 1,1 % | 1,7 % |

El azar (posición del balón 5 s después) da 0,1 % en top1 < 2 m.

- **Decisión: pesos originales y `tiles640`.** El afinado se descarta.
- El afinado mejoró en su propia validación (mAP50 0,70) y empeoró en panorámica, con confianzas que
  casi nunca pasan de 0,3. Encaja con que PBDet no se parezca a una panorámica, pero la causa no está
  medida: es una sola ejecución y no se probaron otras recetas.
- Con el original, la precisión a 2 m sube con la confianza: 5,9 % en 0,05, 6,6 % en 0,30, 7,8 % en 0,50
  y 12,3 % en 0,70, conservando el 19,9 % de los frames. Sirve para ordenar umbrales. Los valores
  absolutos no se trasladan: SoccerTrack es mucho más difícil que la cámara táctica.

## Validación en SkillCorner

```bash
docker compose run --rm dev python -m external.skillcorner download   # 20 partidos, ~1,8 GB
docker compose run --rm dev python -m external.skillcorner parse
docker compose run --rm dev python -m external.validate_skillcorner
```

Tracking real de broadcast a 10 fps de la A-League 2024/25, en un commit fijado. El README anuncia 10
partidos, pero `matches.json` lista 20. Esta tabla es de los modelos entrenados solo con IDSSE, antes de sumar SkillCorner al de equipo. Se aplican sin cambios, con las
posiciones interpoladas a 25 fps. La verdad es `D` fuera de una fase de juego y, dentro, el último
equipo en `possession.group`.

| modelo (`detected`, 20 partidos) | spellF1 | macroF1 | posesiones | reales |
|---|---|---|---|---|
| jugador más cercano | 0,063 | 0,738 | 2 137 | 318 |
| LightGBM frame a frame | 0,156 | 0,796 | 1 199 | 318 |
| **LightGBM + HMM** | **0,372** | 0,800 | 330 | 318 |

- Mide acuerdo entre dos definiciones de posesión, no acierto en el reto, y la cámara no es la táctica.
- Con el relleno de SkillCorner (`extrapolated`), el HMM da 0,361 / 0,793.
- Muerto o vivo coincide en el 89,8 % de los frames; el equipo, en el 74,9 % de los vivos con tracking.
- El 25,6 % de los frames no tiene tracking y su verdad es `D` en el 96,5 %, quizá por construcción.
  Sin ellos, el HMM da 0,385 / 0,783.
- La licencia es MIT y el equipo revisó que no prohíbe este uso. El README pide citar la fuente.

**¿Entrenar también con SkillCorner?** Se probó con los mismos 5 partidos de IDSSE y validación cruzada
en 4 bloques, de modo que cada partido de SkillCorner queda retenido una vez. Solo se entrena con frames
con tracking y cada fuente pesa lo mismo. El criterio se fijó antes de ver cifras: adoptarlo si mejora
SkillCorner retenido sin empeorar IDSSE más que el ruido (desviación 0,022).

```bash
docker compose run --rm dev python -m external.skillcorner_training
```

| modelo | SkillCorner spellF1 | macroF1 | equipo | posesiones | IDSSE degradado + relleno, spellF1 |
|---|---|---|---|---|---|
| **solo IDSSE** | 0,372 | 0,804 | 74,2 % | 323 | **0,527** |
| + SkillCorner en los dos modelos | 0,490 | 0,869 | 83,5 % | 227 | 0,499 |
| + SkillCorner solo en el de equipo | 0,411 | 0,819 | 76,3 % | 262 | 0,504 |

- A peso igual no se adopta ninguna: ambas ganan en los 20 partidos de SkillCorner, pero pierden 0,028 y
  0,023 en IDSSE, por encima del ruido.
- La ganancia puede ser, en parte, aprender la definición de SkillCorner: es mucho mayor cuando también
  entrena su `D`, y predice 227 posesiones frente a 318 reales. Por eso se descarta usarlo en el modelo de
  muerto o vivo.

**Peso de SkillCorner en el modelo de equipo.** Tres semillas, cada una con su par de partidos de IDSSE
retenidos, y los mismos 4 bloques de SkillCorner. Las diferencias se comparan con el modelo solo IDSSE de
la misma semilla. El criterio se fijó antes: pérdida media en IDSSE de 0,010 como máximo y mejora en
SkillCorner en las 3 semillas.

```bash
docker compose run --rm dev python -m external.skillcorner_weight
```

| peso | SkillCorner spellF1, diferencia | mínima | IDSSE spellF1, diferencia | ¿pasa? |
|---|---|---|---|---|
| **0,10** | +0,031 | +0,026 | −0,007 | **sí** |
| 0,25 | +0,037 | +0,033 | −0,011 | no |
| 0,50 | +0,039 | +0,034 | −0,015 | no |

- **Decisión: peso 0,10, solo en el modelo de equipo.** El modelo final entrena con los 7 partidos de IDSSE
  y los 20 de SkillCorner, que quedan sin datos retenidos: las cifras de esta tabla son la validación.
- El coste en IDSSE es pequeño pero real: aparece en las 3 semillas (desviación 0,002).
- No se sabe cuál de las dos fuentes se parece más al partido, y no se puede medir sin usarlo. La mejora
  es una apuesta razonable, no una garantía.

**Variables nuevas para el modelo de equipo.** Tres grupos en `external/features.py`: continuidad (balance
de toques y tiempo reteniéndolos), ataque (velocidad del balón y del bloque de jugadores hacia la portería
rival) y forma (área de cada equipo y jugadores detrás del balón). El modelo de muerto o vivo no cambia.
La regla se fijó antes: no bajar de media en ninguna fuente, suma de ganancias positiva en cada semilla y
de al menos 0,005 de media.

```bash
docker compose run --rm dev python -m external.team_variables
docker compose run --rm dev python -m external.team_variables --follow-up
```

| columnas añadidas | semillas | SkillCorner spellF1, diferencia | IDSSE spellF1, diferencia | ¿pasa? |
|---|---|---|---|---|
| continuidad (24) | 0–2 | +0,000 | +0,001 | no |
| forma (20) | 0–2 | −0,003 | +0,011 | no |
| ataque (20) | 0–2 | −0,004 | +0,036 | no |
| todas (64) | 0–2 | −0,006 | +0,048 | no |
| solo balón (10) | 3–5 | −0,004 | +0,015 | no |
| solo bloque (10) | 3–5 | −0,004 | +0,032 | no |
| ataque (20) | 3–5 | −0,005 | +0,040 | no |

- **Decisión: se adoptan las 20 columnas de ataque, solo en el modelo de equipo (169 columnas).** Ninguna
  variante pasó la regla fijada antes. Se adopta por decisión expresa, después de ver los resultados: gana
  unas 8–10 veces lo que cuesta. El modelo de muerto o vivo sigue con 149, y su archivo no cambia.
- Las semillas 3–5 usan pares retenidos nuevos. La ganancia de ataque en IDSSE se repite con ellos.
- La hipótesis de que la pérdida en SkillCorner venía del bloque, porque solo se ven los jugadores en
  cámara, queda descartada: el balón solo pierde lo mismo.
- La regla exige no perder nada en ninguna fuente, más estricta que la del peso de SkillCorner, que admitía
  perder 0,010. Con esa tolerancia, ataque pasaría.
- La pérdida en SkillCorner es pequeña, aparece en las 6 semillas y no tiene explicación todavía.
- Las tablas anteriores a esta se midieron con 149 columnas en el modelo de equipo.

**Frente al baseline de la organización.** El algoritmo de `starter/baseline.py` (jugador más cercano,
`D` sin balón y voto por mayoría) frente al nuestro, en partidos externos que ninguno de los dos ha
visto. Nuestros modelos se entrenan aquí solo con 5 partidos de IDSSE. El baseline lee el tracking sin
el relleno del detector, como en el partido. No se compara con los 5 minutos públicos: la aclaración
del 3 de noviembre lo prohíbe.

```bash
docker compose run --rm dev python -m external.baseline_compare
```

| datos | baseline ±50 (el suelo publicado) | mejor ventana del baseline | nuestro modelo |
|---|---|---|---|
| IDSSE limpio, 2 partidos | 0,293 | 0,293 (±50) | **0,590** |
| IDSSE degradado, 3 semillas | 0,192 | 0,192 (±50) | **0,568** |
| SkillCorner, 20 partidos | 0,251 | 0,251 (±50) | **0,361** |

Las cifras son spellF1. La ventaja es mucho mayor en IDSSE, cuya degradación es simulada, que en
SkillCorner, que es tracking real de broadcast. Ninguna de las dos cifras predice el margen en el
partido.

**Smart Tagging como entrada.** Según `OVERVIEW.md`, `Team` es "el equipo que realiza la acción": el que
tiene el balón en las fases ofensivas, el que defiende en `DEFENDING IN *` y `DEFENSIVE TRANSITION`, y el
que saca en `SET PIECES`. Ningún dataset externo trae estas etiquetas, así que `external/tags.py` las
simula a partir de la verdad. Simula fases omitidas e inventadas, equipo equivocado, bordes desplazados,
fases que abarcan dos posesiones y `N/A` en parte de los saques. El nivel de error se sortea cada 5
minutos en rangos amplios y aporta 21 columnas. Se puntúa con etiquetas de cuatro calidades. La regla,
fijada antes: con calidad media, IDSSE gana al menos 0,005 en cada semilla y SkillCorner no baja; con
etiquetas inútiles, ninguna fuente pierde más de 0,003.

```bash
docker compose run --rm dev python -m external.tag_experiment
```

| etiquetas | en el modelo de equipo: SC / IDSSE | en los dos modelos: SC / IDSSE |
|---|---|---|
| buenas | +0,081 / +0,062 | +0,104 / +0,076 |
| medias | +0,029 / +0,015 | +0,038 / +0,022 |
| malas | +0,003 / −0,001 | +0,006 / −0,000 |
| inútiles | −0,004 / −0,005 | −0,005 / −0,007 |

- **Primer intento:** las dos variantes cumplen con etiquetas medias, pero con etiquetas inútiles pierden
  algo más de 0,003.
- **Segundo intento, con la misma regla:** se añaden tres columnas de fiabilidad, que miden cuánto
  coincide el equipo de las etiquetas con el equipo más cercano al balón (por parte y en ventanas de
  5 minutos). Se calculan igual al entrenar y en el partido. En datos sintéticos valen 0,96 con
  etiquetas buenas y 0,46 con inútiles.

| etiquetas | + fiabilidad, dos modelos: SC / IDSSE | + fiabilidad, solo equipo: SC / IDSSE |
|---|---|---|
| buenas | +0,112 / +0,085 | +0,095 / +0,073 |
| medias | +0,037 / +0,021 | +0,028 / +0,016 |
| malas | +0,002 / −0,002 | +0,001 / −0,001 |
| inútiles | −0,007 / −0,005 | −0,004 / −0,003 |

- **Decisión: ninguna adoptada.** La fiabilidad apenas cambia nada. Ninguna variante cumple el límite con
  etiquetas inútiles, aunque la de solo equipo se queda a 0,001. La repetición del primer intento dio
  cifras idénticas.
- La ganancia depende del ruido simulado: no se sabe cuánto se equivoca Metrica, ni se puede medir sin el
  partido.

**Duraciones de las posesiones (semi-Markov).** `external/durations.py` divide cada estado en k etapas,
así que su duración sigue una binomial negativa. k y la probabilidad de salida se ajustan por máxima
verosimilitud con las posesiones de IDSSE de entrenamiento; k = 1 es el HMM actual. Solo cambia el
decodificador, sobre los mismos modelos. La regla, fijada antes: IDSSE gana al menos 0,005 de media y
en cada semilla, y SkillCorner no baja.

```bash
docker compose run --rm dev python -m external.duration_experiment
```

- **Decisión: no se adopta.** El ajuste elige 1 etapa para `H` y `A` en las 3 semillas, o sea, el mismo
  HMM. Diferencia media: +0,001 en SkillCorner y +0,001 en IDSSE.
- La duración real de las posesiones ya se parece a una geométrica: el 30 % dura menos de 2 s.

## Regeneración desde un clon limpio

La entrega se regenera bit a bit desde cero: clon de la rama, descarga de IDSSE y SkillCorner, `merge`,
`train --final` y `predict`. Da `7d52e2acdc3ede04a49c6294800feba511fa0a5adfb7907e4a2b8b87db85f75f` y
los artefactos coinciden con `enrich/artefacts.sha256`.

- El clon solo recibe el partido en `data/` y las detecciones de balón fijadas en `cache/enriched/`. El
  detector no se vuelve a pasar (ver `enrich/README.md`).
- El primer intento falló: un archivo de tracking de SkillCorner llegó a medias sin ningún error.
  Desde entonces `external/download.py` comprueba el tamaño anunciado y el SHA-256 de cada uno de los 102
  archivos externos, fijados en `external/downloads.sha256`, y reintenta hasta 3 veces.

## Balón enriquecido

`enrich/merge.py` rellena los huecos del ATD con las detecciones guardadas, que salieron de los pesos
originales con `tiles640` (hash de pesos y de artefacto comprobados). `conf_min` y `margin_m` venían del
partido y ya no: se eligen fuera de él.

```bash
docker compose run --rm gpu python -m external.fill_eval detect
docker compose run --rm dev python -m external.fill_eval sweep
docker compose run --rm dev python -m external.fill_experiment
```

**Umbrales en SoccerTrack.** 240 ventanas de 1–4 s entre eventos anclados (11 026 frames): se oculta el
balón tras el primer evento y se rellena con el tracker de `merge`.

| `conf_min` | `min_run` | frames rellenados | a < 5 m del balón |
|---|---|---|---|
| 0,30 | 3 | 10,3 % | 68,5 % |
| **0,50** | **3** | 6,3 % | 87,1 % |
| 0,50 | 8 | 4,2 % | 92,0 % |
| 0,70 | 3 | 5,4 % | 88,6 % |

- Control con detecciones de otra ventana y el mismo ancla: 4,6 % a menos de 5 m. La precisión viene del
  detector, no del ancla.
- `conf_seed` casi no cambia nada; se queda en 0,50.
- `margin_m` pasa a 2,0 m por criterio físico: la misma tolerancia que `slack_m`.
- `keep_conf`, `anchor_s`, `interp_max` y `max_speed_ms` venían de la etapa anterior. Se comprobaron de
  uno en uno (`fill_eval params`), con interpolación. El criterio, fijado antes: cambiar solo si suben los
  frames bien rellenados y la precisión no baja más de 1 punto.
  - **`keep_conf` pasa de 0,70 a 0,60:** frames rellenados a < 5 m del 7,0 % al 8,7 %, con precisión de
    88,4 % a 88,9 %.
  - `interp_max` 10 también lo cumple (8,5 %, 88,2 %), pero se queda en 5: entre eventos la verdad es una
    recta, y eso favorece a la interpolación por construcción.
  - `anchor_s` y `max_speed_ms` no cambian ese 7,0 %, así que se quedan igual.
- Entre eventos la verdad es una recta, así que solo se usa la cifra a 5 m. Son ventanas sin elegir por
  dificultad, y los valores ordenan ajustes, no se trasladan al partido.

**Modelo.** `degrade(fill=True)` simula el relleno en parte de los frames perdidos, con rangos amplios:
5–60 % de lo perdido, 60–95 % de tramos correctos y errores de 5–25 m en los demás. La variable `filled`
marca esos frames. Dos modelos con los mismos 5 partidos; 2 partidos retenidos por 3 semillas de
degradación, HMM:

| modelo | IDSSE degradado | IDSSE degradado + relleno | SkillCorner `detected` |
|---|---|---|---|
| sin relleno | 0,467 | 0,505 | 0,371 |
| **con relleno** | 0,472 | **0,527** | 0,372 |

Las cifras son spellF1, con desviación de 0,02–0,04. Rellenar ayuda, entrenar sabiéndolo añade algo más,
y no se pierde nada donde no hay relleno. La ganancia depende de los rangos supuestos y viene de una sola
semilla de entrenamiento. La decisión no usa ninguna cifra del partido.

`evaluation/results/external.csv` usa ya el balón enriquecido y el modelo final con SkillCorner y las
columnas de ataque: formato válido, 2 ejecuciones completas idénticas, SHA-256
`7d52e2acdc3ede04a49c6294800feba511fa0a5adfb7907e4a2b8b87db85f75f`.

## Pendiente

1. **Balón enriquecido:** falta medir cuánto rellena una cámara táctica; solo se puede acotar con datos externos.
2. **Equipo con balón:** sigue siendo lo más débil, y falta explicar la pequeña pérdida en SkillCorner.
3. **Repo público:** el historial de git se conserva y contiene lo retirado.
4. **Consulta a la organización:** si las estadísticas agregadas del partido que ve un asistente de IA
   cuentan como "subir datos".
