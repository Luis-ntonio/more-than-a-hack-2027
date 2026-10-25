# Enriquecimiento del balón con visión (Roboflow sports)

El ATD pierde el balón en el 41 % de los frames calificados. Este paquete lo busca en el vídeo
táctico con el detector YOLO de [roboflow/sports](https://github.com/roboflow/sports) (MIT) y
rellena esos huecos en una **copia nueva** del ATD. Los datos originales y la caché existente no se
tocan.

Todo corre en local, en la imagen GPU. Ningún frame sale de la máquina.

```bash
docker compose build gpu                                              # una vez (~11 GB)
docker compose run --rm gpu bash enrich/download_models.sh            # pesos -> models/ (gitignored)

docker compose run --rm gpu python -m enrich.detect_ball --mode calibrate --diagnose   # ~3 min
docker compose run --rm gpu python -m enrich.detect_ball --mode missing               # ~75 min
docker compose run --rm gpu python -m enrich.detect_ball --mode windows               # ~6 min
docker compose run --rm gpu python -m enrich.detect_ball --mode visible               # ~106 min, para los toques
docker compose run --rm dev python -m enrich.merge --simulate                         # elegir parámetros
docker compose run --rm dev python -m enrich.merge                                    # escribe la copia
```

Uso en el código: `possession.io.load_atd(enriched=True)`. Devuelve los mismos arrays que el ATD,
más `ball_source` (0 ninguno, 1 ATD, 2 detector, 3 interpolado) y `ball_conf`.

## Archivos (todos en `cache/enriched/`, gitignored: son datos derivados del partido)

| archivo | contenido |
|---|---|
| `<game>_ball_detections.npz` | top-5 candidatos por frame sin balón en el ATD: píxel, confianza, tamaño |
| `<game>_ball_detections_windows.npz` | lo mismo, en tramos donde el ATD sí tiene balón (para simular) |
| `<game>_atd_enriched.npz` | ATD con los huecos rellenados |

## Qué se ha medido

**Calibración** (2000 frames donde el ATD tiene balón):

- El modelo siempre detecta *algo* con confianza alta, pero el candidato principal solo está a
  menos de 2 m del balón real en el 52 % de los casos.
- La confianza no distingue: los fallos tienen la misma confianza que los aciertos (0,84 frente a
  0,86).
- **Los 5 candidatos contienen el balón real en el 64 % de los frames** (80 % a menos de 5 m).

**Diagnóstico:**

- La alineación de frames es correcta: el pico de acierto está en el desfase 0.
- No hay un falso positivo fijo: los fallos están repartidos por 344 zonas de la imagen.
- Los fallos son *otros objetos parecidos a un balón*: balones de repuesto en la banda, cabezas,
  medias.

Por eso `merge.py` no se queda con el candidato de mayor confianza. Se queda con el que encaja con la
trayectoria (alcanzable a velocidad física desde la última posición aceptada), descarta los que caen
fuera del campo y exige tramos de varios frames seguidos.

**Simulación de huecos** (`merge --simulate`, 40 tramos de 5 s con el balón del ATD oculto):

| | rellena | acierto <2 m (de lo rellenado) | acierto <5 m | acierto <2 m (de todo) |
|---|---|---|---|---|
| top-1 sin seguimiento | 100 % | 52 % | — | 52 % |
| `merge` (parámetros por defecto) | 84 % | **67,5 %** | 82 % | 56,5 % |

El techo es el 64 % de frames con el balón real entre los 5 candidatos. Ya se está cerca: para
mejorar más hay que mejorar el detector (afinarlo con frames de este partido) o usar a los
jugadores como pista.

**Pase completo (ATD enriquecido):** el balón pasa de estar visible en el 59 % al **87 %** de los
frames calificados. Se rellena el 67 % de los frames sin balón.

> **Cuidado.** Los huecos largos (>30 s) se rellenan en un 76 %, y muchos son pelota muerta. En esos
> casos el detector encuentra *un* balón (en manos del que saca, o uno de repuesto). Si se usa el
> balón enriquecido como si fuera el del ATD, se pierde la señal "hueco largo = pelota muerta", que
> es la que más pesa en v1. Los modelos deben recibir `ball_source` y `ball_conf` como features, y
> conservar la visibilidad del ATD original.

## Reproducibilidad

- Inferencia con `cudnn.deterministic` y pesos verificados por SHA-256, que `download_models.sh`
  imprime.
- En la misma GPU, el resultado es estable.
- La entrega debe regenerarse en la misma máquina o con la misma GPU. Si no se puede garantizar,
  hay que decidir si la submission final usa el ATD enriquecido o solo el original.

## Artefacto fijo (entrega)

La entrega no vuelve a pasar el detector: usa estos archivos tal cual, verificados por SHA-256.

```bash
docker compose run --rm dev sha256sum -c enrich/artefacts.sha256
```

Si se regenera alguno, hay que actualizar el manifiesto, y el resultado puede cambiar.

## Afinado del detector con este partido (`enrich/finetune_ball.py`): no mejora el relleno

```bash
docker compose run --rm gpu python -m enrich.finetune_ball dataset    # ~57 min
docker compose run --rm gpu python -m enrich.finetune_ball train      # 17 épocas, 3,8 h (parada anticipada)
docker compose run --rm gpu python -m enrich.finetune_ball evaluate
docker compose run --rm gpu python -m enrich.detect_ball --mode windows --last-min 20 --suffix _held_orig
docker compose run --rm gpu python -m enrich.detect_ball --mode windows --last-min 20 --suffix _held_ft \
    --weights models/football-ball-detection-ft.pt
docker compose run --rm dev python -m enrich.merge --simulate --windows-suffix _held_orig   # y _held_ft
```

**Etiquetas sin mano humana.** El balón del ATD se proyecta en la imagen con la homografía. Como el
ATD no tiene altura, solo se usa un frame si un candidato del detector original está a menos de 1 m
del balón del ATD (caja de ese candidato, el 85 % de los frames) o si el balón va a menos de 2 m/s,
es decir, por el suelo (caja proyectada, el 15 %). Hay 42 417 frames etiquetables; se usan 8 000 para
entrenar y 1 000 para validar. Los **últimos 20 min de la 2ª parte** quedan fuera, con 1 min de
separación, para evaluar.

**Evaluación en el tramo reservado** (2 000 frames con balón del ATD, troceado `tiles640`):

| modelo | detecta algo | top-1 < 1 m | top-1 < 2 m | top-5 < 2 m | top-1 < 5 m |
|---|---|---|---|---|---|
| original | 100 % | 39,1 % | 52,2 % | **61,8 %** | 65,6 % |
| afinado | 87,5 % | **43,6 %** | **57,8 %** | 59,2 % | **71,0 %** |

**De punta a punta, relleno de huecos simulado** (`merge --simulate`, 40 ventanas de 5 s, todas del
tramo reservado):

| detecciones | rellena | acierto < 2 m (de lo rellenado) | acierto < 2 m (de todo) |
|---|---|---|---|
| original | 75,8 % | 72,1 % | **54,6 %** |
| afinado | 72,4 % | 74,1 % | 53,7 % |
| afinado + original (candidatos combinados) | 70,3 % | 72,9 % | 51,3 % |

El afinado pone el balón en primer lugar más a menudo, pero propone menos candidatos, y `merge` ya
elegía bien entre los 5 primeros. El resultado de punta a punta es un empate, y la combinación es
peor. **Se mantiene el modelo original**: las detecciones de `enrich/artefacts.sha256` no cambian.
El modelo afinado queda en `models/football-ball-detection-ft.pt` (gitignored).

## Balón del broadcast (`enrich/detect_ball_broadcast.py`)

Pasa el detector por el vídeo del broadcast en los frames que muestran en directo un hueco del
táctico (alineación: `possession/broadcast.py`) y proyecta los candidatos con la homografía del
broadcast. El candidato principal acierta más que en el vídeo táctico (63,7 % frente a 58,2 % a
menos de 2 m, con `full1280`). Aun así, `merge --broadcast` solo sube el relleno del 48,7 % al
49,8 %, y el modelo de posesión no mejora (docs/05, §4.12). Escribe archivos aparte
(`*_ball_detections_broadcast.npz`, `*_atd_enriched_bc.npz`) que no forman parte del artefacto fijo.
