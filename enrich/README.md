# Enriquecimiento del balón con visión

Este paquete busca el balón en el vídeo táctico, en los frames donde el ATD lo perdió, con el detector
YOLO de [roboflow/sports](https://github.com/roboflow/sports) (MIT). Rellena esos huecos en una **copia
nueva** del ATD. Los datos originales no se tocan.

Todo corre en local, en la imagen GPU. Ningún frame sale de la máquina.

```bash
docker compose build gpu                                     # una vez (~11 GB)
docker compose run --rm gpu bash enrich/download_models.sh   # pesos -> models/ (gitignored)
docker compose run --rm gpu python -m enrich.detect_ball     # detecciones en los huecos del ATD
docker compose run --rm dev python -m enrich.merge           # escribe la copia enriquecida
```

Uso en el código: `possession.io.load_atd(enriched=True)`. Devuelve los mismos arrays que el ATD, más
`ball_source` (0 ninguno, 1 ATD, 2 detector, 3 interpolado) y `ball_conf`.

`merge.py` no se queda con el candidato de mayor confianza: elige el que se puede alcanzar a velocidad
física desde la última posición aceptada, descarta los que caen fuera del campo y exige tramos de
varios frames seguidos.

Los pesos, el recorte y los umbrales se eligen fuera del partido, en SoccerTrack. Las cifras y el
método están en [`docs/07_entrenamiento_externo.md`](../docs/07_entrenamiento_externo.md).

## Archivos (en `cache/enriched/`, gitignored: son datos derivados del partido)

| archivo | contenido |
|---|---|
| `<game>_ball_detections.npz` | top-5 candidatos por frame sin balón en el ATD: píxel, confianza, tamaño |
| `<game>_atd_enriched.npz` | ATD con los huecos rellenados |

## Reproducibilidad

- La inferencia usa `cudnn.deterministic`, y `download_models.sh` imprime el SHA-256 de los pesos.
- La entrega no vuelve a pasar el detector: usa los archivos fijados en `enrich/artefacts.sha256`.

```bash
docker compose run --rm dev sha256sum -c enrich/artefacts.sha256
```

Si se regenera alguno, hay que actualizar el manifiesto, y el resultado puede cambiar.
