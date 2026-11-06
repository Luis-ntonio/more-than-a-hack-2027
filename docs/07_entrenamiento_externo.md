# 07 · Entrenamiento con datos externos

**Por qué existe este documento.** La aclaración de la organización del 3 de noviembre de 2026 tiene
prioridad sobre `RULES.md` y sobre lo anterior: el material del partido provisto se puede usar como
**entrada**, pero no para entrenar, ajustar, calibrar ni validar. Las etiquetas a mano quedan
prohibidas, incluso para validar. Si un número del partido cambia un parámetro del pipeline, es
calibración.

Por eso, los documentos 05 y 06 describen un enfoque **retirado**: el pre-etiquetador, sus cifras y
sus etiquetas ya no forman parte de la solución. El trabajo nuevo está en la rama `external-training`
y en el paquete `external/`, que no lee etiquetas a mano, GT ni resultados anteriores.

## Flujo

```bash
docker compose run --rm dev python -m external.idsse download   # 7 partidos, ~2,6 GB
docker compose run --rm dev python -m external.idsse parse
docker compose run --rm dev python -m external.train            # 5 partidos, valida en 2
docker compose run --rm dev python -m external.train --final    # 7 partidos
docker compose run --rm dev python -m external.predict          # evaluation/results/external.csv
docker compose run --rm dev python evaluation/validate_submission.py evaluation/results/external.csv
```

| pieza | qué hace |
|---|---|
| `external/idsse.py` | descarga IDSSE y lo convierte a arrays con `BallPossession` y `BallStatus` |
| `external/features.py` | 148 variables que existen en las dos fuentes: distancias al balón, jugadores cerca, balón, toques, ventanas |
| `external/degrade.py` | hace que el tracking óptico de IDSSE se parezca a un tracking automático |
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
  Tampoco se comparan con las de 05, que usan otros partidos y otra verdad.

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

## Pendiente

1. **Balón enriquecido:** `external/predict.py` usa el ATD crudo. Las detecciones crudas con los pesos
   originales son válidas y no hay que repetirlas. Falta volver a generar `merge` con umbrales físicos
   o externos, porque `conf_min` y `margin_m` salieron de este partido.
2. **SkillCorner:** validar con tracking de broadcast real.
3. **Repo público:** retirar lo contaminado (`labeler/` como validación, `possession/prelabel.py`,
   `enrich/finetune_ball.py`, `enrich/jersey_team.py`, `V1_PARAMS`) y decidir qué pasa con el historial.
4. **Consulta a la organización:** si las estadísticas agregadas del partido que ve un asistente de IA
   cuentan como "subir datos".
