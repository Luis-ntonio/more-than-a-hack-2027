# 06 · Flujo del modelo: de los datos al CSV

Cómo está montado hoy el sistema: qué entra, qué se entrena, con qué datos, qué sale y cómo se
evalúa. Los resultados y los experimentos están en [05](05_etiquetado_y_preetiquetador.md); aquí
solo se enlazan.

## 1. Vista general

```
 DATOS DEL RETO (data/, nunca se suben)                   ETIQUETAS
 ┌───────────────────────────────────────┐   ┌──────────────────────────────────────┐
 │ táctico: ATD (XML), homografía (CSV), │   │ GT pública: 5 min oficiales          │
 │   vídeo (MP4), smart tagging (XML)    │   │ etiquetas a mano: labels/ (repo      │
 │ broadcast: ATD, homografía, vídeo     │   │   privado), 10 bloques de 5 min      │
 │   (solo en opciones experimentales)   │   └──────────────────┬───────────────────┘
 └──────┬──────────────────┬─────────────┘                      │
        │ (0) lectura      │ (1) visión, GPU                    │ (3) corrección del desfase
        │ possession/io.py │ enrich/ → ARTEFACTO FIJO           │ labeler/timing.py
        ▼                  ▼                                    ▼
 ┌──────────────┐   ┌───────────────────────┐   ┌──────────────────────────────┐
 │ ATD en caché │   │ ATD enriquecido:      │   │ etiquetas frame a frame      │
 │ smart tagging│   │ huecos del balón      │   │ H / A / D, alineadas al GT   │
 └──────┬───────┘   │ rellenados con YOLO   │   └──────────────┬───────────────┘
        │           └──────────┬────────────┘                  │
        └──────────┬───────────┘                               │
                   │ (2) variables por frame (158 columnas)    │
                   ▼                                           │
        ┌───────────────────────┐                              │
        │ matriz X, possession/ │◄─────────── entrenamiento ───┘
        │ prelabel.py           │
        └──────────┬────────────┘
                   │ (4) dos LightGBM → P(muerto), P(local | vivo)
                   │ (5) HMM de 3 estados (Viterbi) → H / A / D por frame
                   ▼
        ┌─────────────────────────────────────────────────────────────┐
        │ evaluation/results/prelabel.csv      (entrenado con todo)   │
        │ evaluation/results/prelabel_oof.csv  (cross-fitting)        │
        └──────────┬──────────────────────────────────────────────────┘
                   │ (6) evaluation/run_evaluation.py  → spellF1, macroF1
                   ▼
              scoreboard
```

Hay dos modelos de referencia que **no se entrenan**: `v0` y `v1` (`possession/models.py`), con
reglas y parámetros fijados a mano a partir del EDA. Se generan con `possession/make_submission.py`.

## 2. Entradas

| entrada | qué es | dónde se lee | lo usa |
|---|---|---|---|
| ATD táctico | posición (x, y normalizadas 0–1) y velocidad de 46 slots y del balón, a 25 fps; equipo de cada slot según el XML | `possession/io.py: load_atd()` | modelo |
| homografía táctica | matriz 3×3 por frame: campo normalizado → píxeles | `load_homography()` | modelo, visión |
| vídeo táctico | 199 319 frames | `enrich/` | visión (artefacto fijo) |
| smart tagging | 638 fases y balones parados de Metrica, en segundos del vídeo táctico | `load_smart_tagging()` | modelo |
| broadcast | ATD (51 slots), homografía y vídeo propios, 164 488 frames, con cortes y repeticiones | `load_atd(source="broadcast")` | solo opciones |
| GT pública | 5 min etiquetados por los organizadores | `evaluation/gt/` | entrenamiento, validación |
| etiquetas a mano | bloques de 5 min hechos con `labeler/` | `labels/*.json` | entrenamiento, validación |

Ningún dato del partido ni ninguna etiqueta a mano se sube al repo principal: `data/`, `cache/`,
`labels/` y `models/` están en `.gitignore`.

## 3. Etapas

### (0) Lectura y caché — `possession/io.py`

Lee cada XML y CSV una vez y los guarda en `cache/` (`<game>_atd_tactical.npz`,
`<game>_atd_broadcast.npz`, …). Todo lo demás lee la caché. También da los bloques calificados de
cada parte (`graded_blocks`) y los frames de los saques iniciales (`kickoff_frames`).

### (1) Visión con GPU — `enrich/` · artefacto fijo

El ATD pierde el balón en el 41 % de los frames calificados. El detector YOLOv8x de Roboflow lo
busca en el vídeo táctico, y `merge` acepta un candidato solo si encaja con la trayectoria:

| paso | comando | sale |
|---|---|---|
| detectar donde el ATD no tiene balón | `enrich.detect_ball --mode missing` | `cache/enriched/*_ball_detections.npz` |
| rellenar los huecos siguiendo la trayectoria | `enrich.merge` | `*_atd_enriched.npz` (balón + `ball_source` + `ball_conf`) |

El balón queda visible en el 79 % de los frames calificados (antes, el 59 %). Estos archivos **no se
regeneran en la entrega** (regla 3, opción b): se usan tal cual y se verifican con
`sha256sum -c enrich/artefacts.sha256`. El manifiesto también incluye las detecciones en los frames
con balón (`--mode visible`) y las ventanas de simulación (`--mode windows`).

Herramientas experimentales que **no usa el modelo por defecto** (05, §4):

- `detect_players.py`: cajas de jugadores;
- `jersey_team.py`: equipo por camiseta;
- `finetune_ball.py`: detector afinado con este partido;
- `detect_ball_broadcast.py`: el balón en el vídeo del broadcast.

### (2) Variables por frame — `possession/prelabel.py: frame_features()`

**158 columnas por frame**, en tres grupos:

- **23 directas** (`possession/features.py: compute()`, con el ATD original y con el enriquecido):
  - visibilidad del balón y longitud del hueco;
  - velocidad media de los jugadores y paneo de la cámara a 2 s;
  - las dos señales de v1: P(muerto) y la evidencia de equipo;
  - distancia de cada equipo al balón y su diferencia;
  - origen y confianza del balón enriquecido;
  - posición y velocidad del balón, distancia a la banda y si está fuera del campo;
  - centroide de cada equipo y su diferencia;
  - si es la 2ª parte.
- **117 de contexto**: 13 de esas señales promediadas en ventanas de 0,5, 2 y 5 s, centradas, hacia
  atrás y hacia delante (13 × 3 × 3). Es el "qué pasa alrededor del frame n".
- **18 del smart tagging** (`possession/smart_tagging.py`):
  - el equipo que implica cada código de fase;
  - segundos desde y hasta el inicio de un balón parado, su tipo y su equipo;
  - segundos desde y hasta un gol.

Hay más fuentes disponibles como opción (`--broadcast`, `--ball-broadcast`, `--touches`,
`--control`). Ninguna mejoró con 5 semillas; la tabla completa está en 05, §5.

### (3) Etiquetas de entrenamiento — `training_blocks()` + `labeler/timing.py`

- La **GT pública** (5 min) entra tal cual.
- Cada **bloque terminado** de `labels/` (los 10 de `luis`) se convierte en etiquetas por frame.
- Antes de entrenar, los cambios hechos a mano **se desplazan al timing del GT** según el desfase que
  cada anotador tiene en el bloque público: dead +1 f, restart +15 f, possession +4 f. Las etiquetas
  guardadas no cambian. `--no-shift` lo desactiva.

En total: 11 bloques, 55 min y 158 posesiones.

**Regla 1:** las etiquetas a mano solo sirven para **entrenar y validar**. Nunca se copian al CSV de
la entrega (§5).

### (4) Clasificadores — `fit()`

Dos LightGBM deterministas (`deterministic`, semilla con `--seed`, 0 por defecto), con 300 rondas,
entrenados con 1 de cada 5 frames, porque los vecinos son casi idénticos.

| modelo | con qué frames | predice |
|---|---|---|
| muerto / vivo | todos los etiquetados | P(D) |
| local / visitante | solo los vivos | P(H \| vivo) |

### (5) Decodificador — `possession/models.py: decode()`

- Las dos probabilidades dan tres emisiones por frame: log P(H), log P(A) y log P(D).
- Un **HMM de 3 estados** con Viterbi elige la secuencia más probable. Cada cambio cuesta: 25 nats
  H↔A y 15 nats a D o desde D. Así las posesiones no parpadean.
- Cada parte se decodifica desde su saque inicial; antes del saque es `D` por regla.

Los costes son los de v1. Ajustarlos por validación cruzada no mejoró (05, §4.1).

### (6) Salidas

| archivo | cómo se hace | para qué |
|---|---|---|
| `evaluation/results/prelabel.csv` | modelo entrenado con **todos** los bloques, aplicado a todo el partido | pre-etiquetar bloques nuevos en el etiquetador |
| `evaluation/results/prelabel_oof.csv` | igual, pero cada bloque etiquetado sale del modelo entrenado **sin ese bloque** | la puntuación honesta, y la base de la entrega |

En los bloques etiquetados `prelabel.csv` saca 0,99, pero es memoria: se entrenó con esas mismas
etiquetas. La cifra que vale es la de `prelabel_oof.csv`.

### (7) Evaluación — `evaluation/run_evaluation.py`

Puntúa cada CSV de `evaluation/results/` contra una verdad:

- `--gt public`: los 5 min oficiales (17 posesiones, muy ruidoso);
- `--gt labels/validation_luis.csv`: los bloques a mano, ya desplazados (`labeler.export --shift auto`).

La métrica que decide el ranking es **spellF1**: el F1 medio de posesiones emparejadas con IoU entre
0,3 y 0,7. **macroF1** desempata. Con 144 posesiones, diferencias de spellF1 menores de ~0,14 son
ruido. `prelabel` imprime además su propia validación dejando un bloque fuera, bloque a bloque.

## 4. Resultados actuales

| | spellF1 | macroF1 | correcciones necesarias |
|---|---|---|---|
| **pre-etiquetador** | **0,514** | **0,816** | 241 |
| v1 (sin entrenar) | 0,438 | 0,764 * | 346 |

Pre-etiquetador: validación dejando un bloque fuera, 11 bloques, media de 5 semillas.

\* El macroF1 de v1 sale del scoreboard contra los 10 bloques a mano (05, §2.2). v1 no se entrena,
así que no tiene validación dejando un bloque fuera.

El defecto conocido es que **se come las posesiones cortas** (< 8 s): predice ~120 posesiones donde
hay 158. Ninguna de las pruebas para recuperarlas funcionó (05, resumen y §3–4).

## 5. La entrega (pendiente)

1. **Cross-fitting en todo el partido**:
   - en los frames de un bloque etiquetado, la predicción del modelo entrenado sin ese bloque (lo que
     ya hace `prelabel_oof.csv`);
   - en el resto, la del modelo entrenado con todo.

   Así ningún frame de la entrega copia una etiqueta a mano (regla 1).
2. **Un comando de entrega** (`make_submission --model prelabel`, o equivalente) que:
   - verifique los hashes del artefacto;
   - entrene;
   - escriba el CSV de 199 319 filas;
   - lo pase por `validate_submission.py`.
3. **Reproducibilidad bit a bit**: dos ejecuciones dan el mismo SHA-256 del CSV.
4. **Repo público**:
   - el código y la documentación, sin `data/`, `cache/`, `labels/` ni `models/`;
   - hay que decidir cómo se reproduce la entrega sin las etiquetas a mano, que siguen en el repo
     privado: publicar el modelo entrenado, o documentar que hacen falta.

## 6. Mapa de archivos

| archivo | papel |
|---|---|
| `possession/io.py` | lectura de las dos fuentes y caché, bloques calificados, saques iniciales |
| `possession/features.py` | variables base por frame (distancias, huecos, velocidades) |
| `possession/smart_tagging.py` | variables del smart tagging |
| `possession/models.py`, `possession/decode.py` | v0, v1, HMM (`decode`), Viterbi, regla del saque inicial |
| `possession/prelabel.py` | **el modelo actual**: variables, etiquetas, LightGBM, validación, salidas |
| `possession/make_submission.py` | CSV de v0 y v1 |
| `possession/broadcast.py`, `possession/stack.py` | alineación del broadcast, modelo del broadcast y juez (opciones) |
| `possession/touches.py`, `possession/control.py` | toques y control del balón (opciones) |
| `enrich/*` | visión con GPU y artefacto fijo (`artefacts.sha256`) |
| `labeler/*` | etiquetador web, desfase de los anotadores, export, visualizador |
| `evaluation/*` | métrica oficial, validación del formato, scoreboard |
| `analysis/*` | EDA y diagnósticos; solo imprimen agregados |

## 7. Reproducir de principio a fin

```bash
# visión: verificar el artefacto (o regenerarlo una vez con la GPU: enrich/README.md)
docker compose run --rm dev sha256sum -c enrich/artefacts.sha256

# modelo: entrena, valida y escribe prelabel.csv y prelabel_oof.csv
docker compose run --rm dev python -m possession.prelabel

# evaluación honesta contra las etiquetas a mano
docker compose run --rm dev python -m labeler.export --annotator luis --shift auto
docker compose run --rm dev python evaluation/run_evaluation.py --gt labels/validation_luis.csv --bootstrap 1000
```
