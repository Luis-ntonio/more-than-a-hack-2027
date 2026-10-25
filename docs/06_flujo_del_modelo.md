# 06 · Flujo del modelo: de los datos al CSV

Este documento resume cómo está montado hoy el sistema: qué entra, qué se entrena, con qué datos,
qué sale y qué se evalúa. Los resultados y los experimentos están en
[05](05_etiquetado_y_preetiquetador.md), y aquí solo se enlazan.

## 1. Vista general

```
 DATOS DEL RETO (data/, nunca se suben)                ETIQUETAS
 ┌────────────────────────────────────┐   ┌──────────────────────────────────────┐
 │ ATD táctico (XML): jugadores, balón│   │ GT pública: 5 min oficiales          │
 │ homografía por frame (CSV)         │   │ etiquetas a mano: labels/ (repo      │
 │ vídeo táctico (MP4)                │   │   privado), 10 bloques de 5 min      │
 └──────┬──────────────────┬──────────┘   └──────────────────┬───────────────────┘
        │                  │                                 │
        │ (0) lectura      │ (1) visión, GPU                 │ (3) corrección del desfase
        │ possession/io.py │ enrich/  → ARTEFACTO FIJO       │ labeler/timing.py
        ▼                  ▼                                 ▼
 ┌──────────────┐   ┌───────────────────────┐   ┌──────────────────────────────┐
 │ ATD en caché │   │ ATD enriquecido:      │   │ etiquetas frame a frame      │
 │ cache/*.npz  │   │ huecos del balón      │   │ H / A / D, alineadas al GT   │
 └──────┬───────┘   │ rellenados con YOLO   │   └──────────────┬───────────────┘
        │           └──────────┬────────────┘                  │
        └──────────┬───────────┘                               │
                   │ (2) variables por frame (158 columnas)    │
                   ▼                                           │
        ┌───────────────────────┐                              │
        │ matriz X (n frames ×  │◄─────────── entrenamiento ───┘
        │ 158), possession/     │
        │ prelabel.py           │
        └──────────┬────────────┘
                   │ (4) dos LightGBM → P(muerto), P(local | vivo)
                   │ (5) HMM de 3 estados (Viterbi) → H / A / D por frame
                   ▼
        ┌────────────────────────────────────────────────────────────┐
        │ evaluation/results/prelabel.csv      (todo el entrenamiento) │
        │ evaluation/results/prelabel_oof.csv  (cross-fitting)        │
        └──────────┬─────────────────────────────────────────────────┘
                   │ (6) evaluation/run_evaluation.py  → spellF1, macroF1
                   ▼
              scoreboard
```

Hay dos modelos de referencia que **no se entrenan**: `v0` y `v1` (`possession/models.py`), con
reglas y parámetros fijados a mano a partir del EDA. Se generan con `possession/make_submission.py`.

## 2. Entradas

| entrada | qué es | dónde se lee |
|---|---|---|
| ATD táctico | posición (x, y normalizadas 0–1) y velocidad de 46 slots de jugador y del balón, a 25 fps; equipo de cada slot según el XML | `possession/io.py: load_atd()` |
| homografía | matriz 3×3 por frame: campo normalizado → píxeles del vídeo | `load_homography()` |
| vídeo táctico | MP4 de 199 319 frames; solo lo usa la etapa de visión | `enrich/` |
| GT pública | 5 min etiquetados por los organizadores | `evaluation/gt/` |
| etiquetas a mano | bloques de 5 min hechos con `labeler/`, en el repo privado de etiquetas | `labels/*.json` |

Ningún dato del partido ni ninguna etiqueta a mano se sube al repo principal: `data/`, `cache/` y
`labels/` están en `.gitignore`.

## 3. Etapas

### (0) Lectura y caché — `possession/io.py`

Lee el XML y el CSV una vez y los guarda como `cache/<game>_atd_tactical.npz`. Todo lo demás lee
la caché. También da los bloques calificados de cada parte (`graded_blocks`) y los frames de los
saques iniciales (`kickoff_frames`).

### (1) Visión con GPU — `enrich/` · **artefacto fijo**

El ATD pierde el balón en el 41 % de los frames calificados. El detector YOLOv8x de Roboflow lo
busca en el vídeo:

| paso | comando | sale |
|---|---|---|
| detectar donde el ATD no tiene balón | `enrich.detect_ball --mode missing` | `cache/enriched/*_ball_detections.npz` |
| detectar donde sí lo tiene (para los toques) | `enrich.detect_ball --mode visible` | `*_ball_detections_visible.npz` |
| ventanas de simulación | `enrich.detect_ball --mode windows` | `*_ball_detections_windows.npz` |
| rellenar los huecos siguiendo la trayectoria | `enrich.merge` | `*_atd_enriched.npz` (balón + `ball_source` + `ball_conf`) |

Estos archivos **no se regeneran en la entrega** (regla 3, opción b): se usan tal cual y se
verifican con `sha256sum -c enrich/artefacts.sha256`. El modelo depende de ellos, porque usa el
ATD enriquecido.

Hay herramientas de visión **experimentales que no usa el modelo por defecto**:
`detect_players.py` (cajas de jugadores), `jersey_team.py` (equipo por camiseta) y
`finetune_ball.py` (detector afinado). Sus resultados están en 05, secciones 14–17.

### (2) Variables por frame — `possession/prelabel.py: frame_features()`

158 columnas por frame:

- **23 directas:**
  - visibilidad del balón (ATD y enriquecido) y longitud del hueco;
  - velocidad media de los jugadores y paneo de la cámara a 2 s;
  - las dos señales de v1, P(muerto) y evidencia de equipo;
  - distancia de cada equipo al balón y su diferencia;
  - origen y confianza del balón enriquecido;
  - posición del balón, distancia a la banda, si está fuera del campo y su velocidad;
  - centroide de cada equipo y su diferencia;
  - si es la 2ª parte.
- **117 de contexto:** 13 de esas señales, promediadas en ventanas de 0,5, 2 y 5 s **centradas, hacia
  atrás y hacia delante** (13 × 3 × 3). Es el "qué pasa alrededor del frame n".

- **18 del smart tagging**, por defecto (`possession/smart_tagging.py`; `--no-tagging` las quita):
  - el equipo que implica cada código de fase;
  - segundos desde y hasta el inicio de un balón parado, su tipo y su equipo;
  - segundos desde y hasta un gol.

  Con 5 semillas suben el macroF1 de 0,803 a 0,816 y no cambian el spellF1 (05, secciones 19–20).

En total son **158 columnas**.

Opciones: `--broadcast` (el broadcast alineado con el táctico, `possession/broadcast.py`) y
`--ball-broadcast` (el balón detectado también en el vídeo del broadcast). Las dos son neutras con 5
semillas (05, secciones 20–21).

Las variables base salen de `possession/features.py: compute()`, una vez con el ATD original y otra
con el enriquecido. Opciones experimentales: `--touches` añade 14 variables de toques del balón del
vídeo y `--control` añade el control con cajas de jugadores. Las dos son neutras (05, secciones 12 y 17).

### (3) Etiquetas de entrenamiento — `training_blocks()` + `labeler/timing.py`

- La **GT pública** (5 min) entra tal cual.
- Cada **bloque terminado** de `labels/` (los 10 de `luis`: `1T-01`…`1T-07` y `2T-00`…`2T-02`) se
  pasa a etiquetas por frame.
- Antes de entrenar, los cambios hechos a mano **se desplazan al timing del GT**, según el desfase
  que cada anotador tiene en el bloque público: dead +1 f, restart +15 f, possession +4 f. Las
  etiquetas guardadas no se tocan. `--no-shift` lo desactiva.

En total son 11 bloques, 55 min y 158 posesiones.

**Regla 1:** las etiquetas a mano solo se usan para **entrenar y validar**. Nunca se copian al CSV
de la entrega (ver la sección 5).

### (4) Clasificadores — `fit()`

Dos modelos LightGBM, deterministas (`seed 0`, `deterministic`), con 300 rondas. Se entrena con 1 de
cada 5 frames, porque los vecinos son casi idénticos.

| modelo | con qué frames | predice |
|---|---|---|
| muerto / vivo | todos los etiquetados | P(D) |
| local / visitante | solo los vivos | P(H \| vivo) |

### (5) Decodificador — `possession/models.py: decode()`

- Las dos probabilidades se convierten en tres emisiones por frame: log P(H), log P(A), log P(D).
- Un **HMM de 3 estados** con Viterbi elige la secuencia más probable. Cada cambio cuesta: H↔A
  25 nats, a D o desde D 15 nats. Esto impone que las posesiones no parpadeen.
- Se decodifica cada parte desde el saque inicial. Antes del saque es `D` por regla.

Los costes son los de v1. Ajustarlos por validación cruzada no mejoró (05, sección 6).

### (6) Salidas

| archivo | cómo se hace | para qué |
|---|---|---|
| `evaluation/results/prelabel.csv` | modelo entrenado con **todos** los bloques y aplicado a todo el partido | pre-etiquetar bloques nuevos en el etiquetador |
| `evaluation/results/prelabel_oof.csv` | igual, pero **cada bloque etiquetado** se sustituye por la predicción del modelo entrenado **sin ese bloque** | la puntuación honesta, y la base de la entrega |

En los bloques etiquetados, `prelabel.csv` saca 0,99, pero es memoria: se entrenó con esas mismas
etiquetas. La cifra que vale es la de `prelabel_oof.csv`.

### (7) Evaluación — `evaluation/run_evaluation.py`

Puntúa cada CSV de `evaluation/results/` contra una verdad:

- `--gt public`: los 5 min oficiales (17 posesiones, muy ruidoso);
- `--gt labels/validation_luis.csv`: los bloques a mano, ya desplazados (`labeler.export --shift auto`).

La métrica que decide el ranking es **spellF1**: el F1 medio de posesiones emparejadas con IoU
0,3–0,7. Con 144 posesiones, diferencias menores de ~0,14 son ruido.

Además, `prelabel` imprime su propia validación dejando un bloque fuera, bloque por bloque.

## 4. Resultados actuales

| | spellF1 (validación dejando un bloque fuera, 11 bloques, media de 5 semillas) | macroF1 | correcciones necesarias |
|---|---|---|---|
| **pre-etiquetador** (con smart tagging) | **0,514** | **0,816** | 241 |
| v1 (sin entrenar) | 0,438 | 0,764 * | 346 |

\* El macroF1 de v1 sale del scoreboard contra los 10 bloques a mano (sección 9 de 05), no de la
validación dejando un bloque fuera, porque v1 no se entrena.

El defecto conocido es que **se come las posesiones cortas** (< 8 s): predice ~117 posesiones donde
hay 158. Se probaron nueve vías para recuperarlas, sin éxito (05, secciones 6–18).

## 5. La entrega (pendiente)

Lo que falta para el CSV final:

1. **Cross-fitting en todo el partido:**
   - en los frames de un bloque etiquetado, la predicción del modelo entrenado **sin** ese bloque (lo
     que ya hace `prelabel_oof.csv`);
   - en el resto del partido, la del modelo entrenado con todo.

   Así ningún frame de la entrega copia una etiqueta a mano (regla 1).
2. **Un comando de entrega** (`make_submission --model prelabel`, o equivalente) que:
   - verifique los hashes del artefacto;
   - entrene;
   - escriba el CSV de 199 319 filas;
   - lo pase por `validate_submission.py`.
3. **Reproducibilidad bit a bit:** dos ejecuciones dan el mismo SHA-256 del CSV.
4. **Repo público:**
   - el código y la documentación, sin `data/`, `cache/`, `labels/` ni `models/`;
   - las etiquetas a mano se quedan en el repo privado;
   - hay que decidir cómo se reproduce la entrega sin ellas: publicar el modelo entrenado, o
     documentar que hacen falta.

## 6. Mapa de archivos

| archivo | papel |
|---|---|
| `possession/io.py` | lectura de datos y caché, bloques calificados, saques iniciales |
| `possession/features.py` | variables base por frame (distancias, huecos, velocidades) |
| `possession/models.py` | v0, v1, HMM (`decode`), regla del saque inicial |
| `possession/decode.py` | Viterbi |
| `possession/prelabel.py` | **el modelo actual**: variables, etiquetas, LightGBM, validación, salidas |
| `possession/make_submission.py` | CSV de v0 y v1 |
| `possession/touches.py`, `possession/control.py` | variables experimentales (toques, control) |
| `enrich/*` | visión con GPU y artefacto fijo |
| `labeler/*` | etiquetador web, desfase de los anotadores, export, visualizador |
| `evaluation/*` | métrica oficial, validación del formato, scoreboard |
| `analysis/*` | EDA y diagnósticos; solo imprimen agregados |

## 7. Reproducir de principio a fin

```bash
# visión (una vez, GPU): o verificar el artefacto existente
docker compose run --rm dev sha256sum -c enrich/artefacts.sha256

# modelo: entrena, valida y escribe prelabel.csv y prelabel_oof.csv
docker compose run --rm dev python -m possession.prelabel

# evaluación honesta contra las etiquetas a mano
docker compose run --rm dev python -m labeler.export --annotator luis --shift auto
docker compose run --rm dev python evaluation/run_evaluation.py --gt labels/validation_luis.csv --bootstrap 1000
```
