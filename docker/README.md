# Entorno del equipo (Docker)

Una sola imagen fijada (`python:3.11.13-slim` + `requirements.lock.txt`) para desarrollo, evaluación
y la ejecución que regenera la submission. El código y los datos se **montan** al ejecutar, nunca se
copian dentro de la imagen, porque los datos no se pueden redistribuir.

## Requisitos
- Docker Desktop (en Windows con backend WSL2).
- Los datos descomprimidos en `./data`, con los nombres que espera `games.py`:
  `data/tactical/BAR-ATH_tactical.mp4`, `data/broadcast/BAR-ATH_broadcast.mp4`, etc.

## Uso

```bash
docker compose build dev                              # una vez (o cuando cambie el lock)
docker compose run --rm dev                           # shell en /workspace
docker compose run --rm dev python evaluation/run_evaluation.py
docker compose run --rm dev python docker/smoke_test.py   # comprueba librerías, vídeo y ATD
docker compose up jupyter                             # JupyterLab en http://localhost:8888
```

- `/workspace` es el repo (bind mount): lo que edites fuera se ve dentro y al revés.
- `/workspace/data` está montado **en solo lectura**.

## Añadir una dependencia
1. Añadirla, con versión, a `docker/requirements.in`.
2. Borrar temporalmente `docker/requirements.lock.txt` y ejecutar `docker compose build dev`.
3. Regenerar el lock:
   `docker compose run --rm -T dev pip freeze > docker/requirements.lock.txt`
4. Hacer commit de los dos archivos. El resto del equipo vuelve a ejecutar `docker compose build dev`.

## GPU (más adelante)
La imagen actual es solo CPU. Si se aprueba la detección de pelota en vídeo, se añadirá una variante
con CUDA y PyTorch (`--gpus all` sobre WSL2).
