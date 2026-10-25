# 05 · Etiquetado a mano, pre-etiquetador y experimentos

**Estado (25-10-2026).** Hay 55 min etiquetados a mano: el bloque público y 10 bloques de 5 min
(`1T-01`…`1T-07` y `2T-00`…`2T-02`), todos desde cero, anotador `luis`. Las etiquetas están solo en el
repo privado de etiquetas; aquí van solo cifras agregadas. El flujo completo del modelo está en
[06](06_flujo_del_modelo.md).

## Resumen

**Modelo actual**: el pre-etiquetador (`possession/prelabel.py`). Son dos LightGBM sobre 158
variables por frame (tracking, balón enriquecido con Roboflow y smart tagging) más el HMM de v1.

| | spellF1 | macroF1 | correcciones | posesiones (verdad 158) |
|---|---|---|---|---|
| **pre-etiquetador** (media de 5 semillas) | **0,514** | **0,816** | 241 | ~120 |
| v1 (sin entrenar) | 0,438 | — | 346 | — |

Las cifras salen de validar dejando un bloque fuera: cada bloque se predice con un modelo que no lo
vio. Hay 11 bloques. Contra los 10 bloques a mano (144 posesiones), las diferencias de spellF1
menores de ~0,14 son ruido.

**Su defecto**: se come las **posesiones cortas** (< 8 s), que son la mitad de todas. Un equipo
defiende, recupera el balón en su campo, lo saca 10–20 m y lo vuelve a perder. Sobre los límites
reales de la posesión la señal es clara, pero frame a frame casi no se ve (§3).

**Lo que se probó para recuperarlas o mejorar el modelo:**

| prueba | § | spellF1 (base ≈ 0,51) | veredicto |
|---|---|---|---|
| corregir el desfase del anotador al entrenar | 1 | público 0,413 → 0,538 | **en uso** |
| ajustar los costes del HMM | 4.1 | 0,521 (anidado) | no ayuda |
| ponderar frames por 1/duración de la posesión | 4.2 | 0,482–0,488 | no ayuda |
| toques en el balón del tracking | 4.2–4.3 | 0,507 | no ayuda |
| toques en el balón del vídeo (`--touches`) | 4.4 | 0,512 | neutro, opción |
| regla: equipo = último toque confirmado | 4.4 | 0,475–0,493 | peor |
| control del balón con cajas de jugadores | 4.5 | ≤ 0,435 como regla | peor |
| equipo por color de camiseta (`--control`) | 4.6 | 0,502 | neutro, opción |
| afinar el detector de balón con este partido | 4.7 | — (el relleno empata) | descartado |
| detectar posesiones cortas como segmentos | 4.8 | 0,49–0,51 | no ayuda |
| **smart tagging** | 4.9 | 0,514 (macroF1 +0,013) | **en uso** |
| broadcast alineado como variables (`--broadcast`) | 4.10 | 0,512 (macroF1 +0,006) | neutro, opción |
| modelo del broadcast + juez | 4.11 | 0,511 | neutro |
| balón detectado en el broadcast (`--ball-broadcast`) | 4.12 | 0,518 | neutro, opción |

**Conclusión**: el tracking y los dos vídeos dan casi toda la información frame a frame. Las
posesiones cortas fallan porque **hay que encontrar sus límites**, no por falta de señal sobre quién
está cerca del balón.

```bash
docker compose run --rm dev python -m labeler.timing                 # desfase frente al GT
docker compose run --rm dev python -m possession.prelabel            # entrena, valida, escribe prelabel*.csv
docker compose run --rm dev python -m labeler.export --annotator luis --shift auto
docker compose run --rm dev python evaluation/run_evaluation.py --gt labels/validation_luis.csv --bootstrap 1000
```

---

## 1. Desfase del anotador frente al GT (`labeler/timing.py`)

`1T-00` es la ventana pública: ahí se compara la etiqueta a mano con el GT oficial.

| tipo de cambio | n | mediana | media | corrección |
|---|---|---|---|---|
| a muerta (`dead`) | 4 | −1 f | −1,0 f | +1 f |
| reanudación (`restart`) | 4 | −15 f | −16,8 f | +15 f |
| cambio de posesión | 12 | −4,5 f | −7,2 f | +4 f |

El acuerdo por frame es del 97,9 %, con 20 cambios emparejados, 0 omitidos y 0 de más. El anotador
marca antes que el GT, y sobre todo las reanudaciones: el GT las pone cuando el balón ya rueda, no en
el golpeo (`analysis/gt_changes.py`). La corrección se aplica al entrenar y al exportar; las
etiquetas guardadas no cambian.

**Sirve.** En el bloque público, el único con verdad oficial, el spellF1 de la validación pasa de
**0,413** (sin corregir) a **0,538** (corregido), con el macroF1 igual (0,888 / 0,887). Es un solo
bloque de 17 posesiones: una señal, no una prueba.

## 2. Medición del pre-etiquetador

### 2.1 Validación dejando un bloque fuera (11 bloques, semilla 0, antes del tagging)

| bloque | spellF1 | macroF1 | spellF1 v1 | correcciones pre / v1 | cambios reales |
|---|---|---|---|---|---|
| público | 0,428 | 0,876 | 0,389 | 29 / 35 | 20 |
| 1T-01 | 0,610 | 0,859 | 0,465 | 14 / 29 | 16 |
| 1T-02 | 0,600 | 0,691 | 0,568 | 27 / 38 | 21 |
| 1T-03 | 0,431 | 0,795 | 0,287 | 27 / 40 | 20 |
| 1T-04 | 0,413 | 0,802 | 0,374 | 22 / 29 | 23 |
| 1T-05 | 0,522 | 0,805 | 0,387 | 19 / 35 | 15 |
| 1T-06 | 0,462 | 0,861 | 0,486 | 9 / 15 | 10 |
| 1T-07 | 0,574 | 0,846 | 0,416 | 19 / 26 | 15 |
| 2T-00 | 0,473 | 0,792 | 0,482 | 24 / 29 | 20 |
| 2T-01 | 0,557 | 0,740 | 0,509 | 24 / 39 | 24 |
| 2T-02 | 0,541 | 0,769 | 0,457 | 30 / 31 | 25 |
| **media / total** | **0,510** | 0,803 | 0,438 | **244 / 346** | 209 |

- Gana a v1 en 9 de 11 bloques y ahorra un 30 % de correcciones.
- **Funciona en la 2ª parte**, con los equipos cambiados de lado: 0,524 frente a 0,483 de v1. Pero
  cada bloque de la 2ª se entrenó también con los otros dos de la 2ª.
- Con 8 bloques (solo la 1ª parte) la media era 0,526. Al añadir la 2ª bajó el bloque público (0,538 →
  0,428). Con ~20 posesiones por bloque, eso es ruido.
- `1T-04` es el bloque más difícil para todos los modelos.

### 2.2 Scoreboard honesto contra las etiquetas a mano

`prelabel.py` escribe dos archivos:

- `prelabel.csv`: entrenado con todo. En los bloques etiquetados saca 0,99, pero es memoria: se
  entrenó con esas mismas etiquetas.
- `prelabel_oof.csv`: cada bloque etiquetado lo predice el modelo que no lo vio. Es la fila
  comparable.

Verdad: 10 bloques (50 min, 144 posesiones de 12,6 s de media), con el desfase corregido y bootstrap
de 1000 muestras. Semilla 0, antes del tagging.

| submission | spellF1 | macroF1 | transF1 | posesiones (verdad 144) | duración media |
|---|---|---|---|---|---|
| **prelabel_oof** | **0,499** | **0,812** | **0,380** | 110 | 16,6 s |
| v1 | 0,425 | 0,764 | 0,228 | 150 | 11,8 s |
| v0 | 0,300 | 0,386 | 0,137 | 207 | 14,5 s |
| baseline_smooth50 | 0,263 | 0,665 | 0,111 | 299 | 7,4 s |

- **Ruido**: con 144 posesiones, diferencias de spellF1 menores de ~0,14 no son significativas. La
  ventaja sobre v1 (+0,074) va en la misma dirección en 9 de 11 bloques, pero cabe en el ruido.
- **transF1** sí separa: 0,380 frente a 0,228. Los cambios que encuentra caen mucho más cerca del
  frame correcto (latencia media +0,4 f frente a −2,1 f).
- **El defecto**: 110 posesiones donde hay 144, porque se fusionan las cortas con sus vecinas. v1
  acierta el número (150) pero las coloca peor.

## 3. Diagnóstico: por qué se pierden las posesiones cortas

### 3.1 El clasificador no ve las posesiones cortas (`analysis/short_possessions.py`)

Las 166 posesiones verdaderas de los 11 bloques, por duración. `p_team` es la probabilidad fuera de
muestra del equipo correcto; "encontrada" significa que hay una posesión predicha del mismo equipo
con IoU ≥ 0,5.

| duración | n | p_team | frames bien | balón visto | encontrada |
|---|---|---|---|---|---|
| 0–2 s | 26 | 0,38 | 38 % | 87 % | 0 % |
| 2–4 s | 29 | 0,57 | 57 % | 97 % | 14 % |
| 4–8 s | 31 | 0,60 | 61 % | 96 % | 45 % |
| 8–15 s | 30 | 0,81 | 81 % | 97 % | 67 % |
| 15–30 s | 35 | 0,92 | 93 % | 92 % | 71 % |
| > 30 s | 15 | 0,89 | 90 % | 98 % | 93 % |

**La mitad de las posesiones dura menos de 8 s**, y en ellas el clasificador está casi al azar aunque
el balón se vea casi siempre. Con `p_team` ≈ 0,5 ningún decodificador las recupera. Con 8 bloques la
tabla era la misma: el problema no es la cantidad de etiquetas.

### 3.2 Qué es una posesión corta en el GT (`analysis/short_spell_profile.py`)

Se mide el balón **sobre los límites reales** de cada posesión que sigue a una del otro equipo, desde
el punto de vista del equipo que la gana. Se compara con ventanas de la misma duración dentro de
posesiones largas, tomando como "nuevo" al equipo sin el balón. La dirección de ataque sale de la
posición mediana de cada portero en cada parte.

| medida (2–4 s, n = 23) | corta | control | AUC |
|---|---|---|---|
| distancia mínima del equipo nuevo al balón | 0,29 m | 2,57 m | 0,88 (invertido) |
| % de frames en que el nuevo está más cerca | 64 % | 16 % | 0,84 |
| avance del balón hacia la portería que ataca | +11,6 m | −3,1 m | 0,82 |
| dónde empieza (desde su propia portería) | 34 m | 63 m | 0,75 (invertido) |

- Las de 4–8 s son parecidas (avance +19,6 m, AUC de cercanía 0,89). Las de 0–2 s, menos (AUC
  0,61–0,79).
- Después, el balón vuelve al equipo anterior el 59–74 % de las veces; si no, es balón muerto.
- **Es el patrón de un balón dividido**: el equipo que defiende recupera el balón en su campo, lo saca
  y lo pierde. Encaja con la convención del GT, que marca el cambio cuando el balón queda suelto tras
  la intercepción.
- **Sobre la posesión entera la señal es clara. Frame a frame no**: el equipo nuevo está más cerca
  solo el ~60 % del tiempo, y el HMM lo suaviza.

## 4. Experimentos

Todos se miden igual: validación dejando un bloque fuera, con la media de spellF1 de los bloques.

### 4.1 Costes del HMM

Se probaron 150 combinaciones (cambio de equipo 4–40, a muerta 4–25, desde muerta 4–25 nats),
eligiendo los costes de cada bloque sin él. Con 8 bloques:

- costes de v1 (25 / 15 / 15): 0,526 y 174 correcciones;
- mejor punto, elegido sobre los mismos bloques: 0,527;
- anidado: 0,521 y 209 correcciones.

La superficie es plana. **El límite no está en el decodificador.**

### 4.2 Toques en el balón del tracking y ponderación (8 bloques)

| variante | spellF1 | correcciones |
|---|---|---|
| pre-etiquetador | **0,526** | **174** |
| + variables de toque (último y siguiente equipo a 1,5 m y 3 m) | 0,507 | 183 |
| + toque + frames ponderados por duración^−0,5 de su posesión | 0,488 | — |
| + toque + frames ponderados por duración^−1 (como spellF1) | 0,482 | — |

Se quitaron del código.

### 4.3 ¿Hay toques en el tracking? (`analysis/touch_signal.py`)

Un "toque" es un giro o un cambio de velocidad del balón entre los 5 frames anteriores y los 5
siguientes, con un jugador cerca. Hay 109 cambios H↔A en 11 bloques. Con el umbral 35° / 3 m/s / 2,5 m:

| balón | toques/min | cambios con toque del nuevo equipo a ±1 s | azar (ventana a 3 s) | precisión | confirmado: precisión / cambios que conserva |
|---|---|---|---|---|---|
| tracking | 12,3 | 48 % | 25 % | 40 % | 52 % / 40 % |
| **detectado en el vídeo** | 27,3 | 53 % | 35 % | 38 % | **54 % / 75 %** |
| híbrido | 43,9 | 84 % | 53 % | 34 % | 43 % / 79 % |

- El tracking ve muy pocos toques. Aun con el balón visible en el 87 % de los frames cerca de un
  cambio, su trayectoria está suavizada o pegada al suelo.
- Solos, los toques apenas superan al azar. "Confirmado" (el mismo equipo vuelve a tocar en menos de
  3 s) separa "controla" de "desvía", y con el balón del vídeo conserva el 75 % de los cambios.
- El híbrido parece mejor, pero la mitad es casualidad, y alternar dos fuentes crea giros falsos.

El balón detectado necesita `enrich.detect_ball --mode visible` (94 214 frames, 106 min de GPU).

### 4.4 Variables de toque del vídeo (`possession/touches.py`, `--touches`)

Son 14 variables: último y siguiente toque, confirmado o no, tiempos y cambios de equipo confirmados.

| variante (11 bloques) | spellF1 | correcciones | posesiones (verdad 158) |
|---|---|---|---|
| pre-etiquetador | 0,510 | 244 | 117 |
| `--touches` | 0,512 | 234 | — |
| regla: equipo = último toque confirmado (`analysis/touch_rule.py`) | 0,475 | — | **158** |
| la misma regla, ignorando cambios de menos de 2 s | 0,493 | — | 140 |

La regla acierta el número de posesiones, pero la mitad de sus cambios son falsos.

### 4.5 Control del balón con cajas de jugadores (`enrich/detect_players.py`, `possession/control.py`)

- **Detector de jugadores de Roboflow** (YOLOv8x, 1280 px) sobre los 82 500 frames etiquetados (~40 min
  de GPU).
- **Equipo de cada caja**: el del jugador del ATD más cercano a sus pies (≤ 3 m).
- **Control**: el balón detectado a menos de `contact` alturas de caja de un jugador.

| contacto | control = equipo etiquetado (0–2 / 2–4 / 4–8 / 8–15 / > 15 s) |
|---|---|
| 0,15 | 46 / 52 / 58 / 86 / 89 % |
| 0,5 | 36 / 54 / 59 / — / — % |
| 1,0 | 47 / 53 / 56 / — / — % |

- En las posesiones largas, el detector confirma el equipo.
- En las cortas, quien toca el balón es el equipo etiquetado la mitad de las veces.
- Como detector de cambios, la precisión es del 9–12 %, y segmentar por control da ≤ 0,435.

### 4.6 Equipo por color de camiseta (`enrich/jersey_team.py`, `--control`)

Pone a prueba si el problema es la asignación de equipo en los duelos.

- **Entrenamiento**: LightGBM sobre el histograma HSV del torso sin el césped, con las cajas en las que
  el ATD es claro (69 %).
- **Evaluación**: se entrena con la 1ª parte y se evalúa con la 2ª.

| cajas | coincide con el ATD |
|---|---|
| 2ª parte, claras (acierto real) | **98,8 %** |
| 2ª parte, duelos | 93,5 % |
| 1ª parte, duelos | 89,6 % |

- Corrige el equipo del ATD en el 6–10 % de las cajas en duelo.
- Pero el control en las cortas apenas cambia: 46 / 56 / 62 % frente a 46 / 52 / 58 %.
- Segmentar por control llega a 0,468, y como variable `--control` saca 0,502.
- **La asignación de equipo explica poco. Manda la convención del GT.**

### 4.7 Afinar el detector de balón con este partido (`enrich/finetune_ball.py`)

- **Etiquetas**: salen del ATD, sin mano humana. Se usan los frames en los que el candidato del detector
  coincide con el balón del ATD (85 %), o en los que el balón va lento por el suelo (15 %).
- **Entrenamiento**: 8 000 frames; los últimos 20 min de la 2ª parte se reservan para evaluar.

| | top-1 < 2 m | top-5 < 2 m | relleno simulado: acierto < 2 m del total |
|---|---|---|---|
| original | 52,2 % | 61,8 % | **54,6 %** |
| afinado | **57,8 %** | 59,2 % | 53,7 % |

El modelo afinado pone el balón primero más a menudo, pero propone menos candidatos, y el relleno
empata. **Se mantiene el detector original.** Detalle en `enrich/README.md`.

### 4.8 Posesiones cortas como segmentos (`analysis/short_spell_proposals.py`)

- **Propuestas**: tramos en los que el equipo sin posesión según el pre-etiquetador está más cerca del
  balón.
- **Puntuación**: se describen con las medidas de §3.2 y se puntúan con LightGBM, que no ve el bloque
  que puntúa.

| unión de huecos / mínimo | cortas perdidas cubiertas | propuestas que aciertan | AUC | spellF1 anidado |
|---|---|---|---|---|
| 5 f / 10 f | 31 % de 54 | 5 % | 0,77 | 0,509 |
| 15 f / 10 f | 35 % | 6 % | 0,82 | 0,507 |
| 30 f / 5 f | 30 % | 5 % | 0,81 | 0,492 |
| 50 f / 5 f | 28 % | 6 % | 0,84 | 0,500 |

Las propuestas solo encajan con un tercio de las posesiones que faltan, y solo el 5 % de ellas
acierta. **Para ver la señal hay que conocer ya los límites.**

### 4.9 Smart tagging (`possession/smart_tagging.py`, por defecto)

Viene ya en el tiempo del vídeo táctico (`analysis/smart_tagging_vs_labels.py`):

| | resultado |
|---|---|
| equipo implicado por el código | BUILD UP 99 %, PROGRESSION 85 %, transiciones 71 %, SET PIECES 68 %, en el área / último tercio 56–67 % |
| reinicio real tras el inicio de un balón parado | córner 6,0 s, saque de puerta 4,9 s, banda 6,2 s, falta 12,2 s (mediana) |
| equipo que reinicia = equipo del balón parado | 100 % (12 de 12; los saques de banda no tienen equipo) |
| GOALS | 92–95 % de sus frames son D |

Son 18 variables: el equipo que implica cada código, los tiempos desde y hasta el balón parado (con
su tipo y su equipo) y los tiempos desde y hasta un gol. Ver §4.13 para el resultado.

### 4.10 Broadcast alineado con el táctico (`possession/broadcast.py`, `--broadcast`)

**Alineación.** Las dos grabaciones van en tiempo real, así que fuera de las repeticiones el frame
`j` del broadcast es el `j + c` del táctico. Sin identidades comunes (51 slots frente a 46), se
empareja por geometría: cada jugador del broadcast contra el más cercano de su equipo en el táctico,
más el balón cuando los dos lo tienen.

- **+4 013 f (+160,5 s)** en la 1ª parte y **+26 920 f (+1 076,8 s)** en la 2ª (el broadcast se salta
  el descanso).
- El 74 % de las muestras encaja a menos de 2 m. La alineación tarda ~4 min.

| estado del broadcast en cada frame etiquetado | frames | balón muerto |
|---|---|---|
| directo | 56,8 % | 24,3 % |
| campo visible pero no encaja | 20,9 % | 22,8 % |
| **plano cerrado o sin campo** | 22,2 % | **85,6 %** |

- Un plano cerrado es una señal fuerte de balón muerto.
- "No encaja" se comporta como directo: casi todo es directo mal emparejado, no repeticiones.
- Variables: estado, coste del emparejamiento, si hay balón en el broadcast, proporción de frames que
  no son directo a 2, 5 y 10 s, y la duración del tramo que no es directo.

### 4.11 Modelo del broadcast y juez (`possession/stack.py`)

- **Modelo del broadcast**: el mismo par de LightGBM con lo que ve el broadcast en directo (28
  columnas).
- **Juez**: un LightGBM simple (8 hojas) sobre las probabilidades de los dos modelos, sus medias a 1 y
  3 s, el estado del broadcast, el coste del emparejamiento y la disputa del balón en cada vista.
- **Validación**: el juez se entrena con las probabilidades fuera de muestra de los otros bloques. Hay
  un sesgo optimista pequeño: los modelos base que las dieron sí vieron el bloque evaluado.

| variante (semilla 0) | spellF1 | macroF1 | posesiones (verdad 158) |
|---|---|---|---|
| táctico | 0,510 | 0,803 | 120 |
| broadcast solo | 0,363 | 0,700 | 157 |
| media de los dos | 0,499 | 0,803 | 107 |
| juez | 0,511 | 0,806 | 102 |

Un juez de 31 hojas sobreajustaba y sacaba 0,474. El simple empata: el ángulo del broadcast no aporta
información nueva sobre los cambios.

### 4.12 Balón detectado en el broadcast (`enrich/detect_ball_broadcast.py`, `--ball-broadcast`)

El balón se ve más grande en el broadcast. Calibración en 2 000 frames que el broadcast muestra en
directo:

| detector | top-1 < 2 m | top-5 < 2 m | top-1 < 5 m |
|---|---|---|---|
| vídeo táctico, original | 58,2 % | **71,5 %** | 69,1 % |
| **broadcast, original `full1280`** | **63,7 %** | 67,0 % | **77,0 %** |
| broadcast, afinado `full1280` | 62,8 % | 63,2 % | 74,9 % |

- El broadcast muestra en directo el 30 % de los huecos (19 365 frames).
- De los 9 967 que el táctico no rellenaba, solo el 29 % tiene un candidato fiable dentro del campo.
- `merge --broadcast` sube el relleno del 48,7 % al 49,8 %.

### 4.13 Cinco semillas

Para separar mejora de ruido, las variantes que dependen de pocas décimas se repitieron con 5
semillas de LightGBM:

| variante | spellF1 | rango entre semillas | macroF1 | correcciones |
|---|---|---|---|---|
| sin tagging | 0,512 | 0,508–0,515 | 0,803 | 241 |
| **con tagging (por defecto)** | 0,514 | 0,497–0,525 | **0,816** | 241 |
| sin tagging + `--broadcast` | 0,512 | 0,494–0,524 | 0,809 | **231** |
| `--broadcast` + tagging | 0,515 | 0,509–0,519 | 0,810 | 235 |
| tagging + `--ball-broadcast` | 0,518 | 0,510–0,537 | 0,817 | 242 |

- **En spellF1 ninguna fuente mejora** fuera de la variación entre semillas (0,004–0,012). Con una sola
  semilla, el tagging y el broadcast parecían dar +0,011.
- **En macroF1 sí**, y en todas las semillas: el tagging +0,013. Por eso va por defecto.

## 5. Opciones, artefactos y reproducibilidad

| opción de `prelabel` | por defecto | necesita | efecto |
|---|---|---|---|
| desfase del anotador | sí (`--no-shift`) | — | §1 |
| smart tagging | sí (`--no-tagging`) | — | macroF1 +0,013 |
| `--broadcast` | no | `possession.broadcast` | neutro |
| `--ball-broadcast` | no | `detect_ball_broadcast gaps` + `merge --broadcast` | neutro |
| `--touches` | no | `detect_ball --mode visible` | neutro |
| `--control` | no | `detect_players` + `jersey_team` (solo bloques etiquetados) | neutro |
| `--seed` | 0 | — | variación entre semillas ±0,01 |

**Artefacto fijo (regla 3, opción b).** La inferencia en GPU no se regenera en la entrega. Las
detecciones de balón y el ATD enriquecido que usa el modelo por defecto se verifican con:

```bash
docker compose run --rm dev sha256sum -c enrich/artefacts.sha256
```

Con el artefacto verificado, `prelabel` es determinista: con la semilla 0 y el tagging, dos
ejecuciones dan las mismas 235 correcciones. Los archivos de las opciones experimentales (detecciones
de jugadores y del broadcast, alineación, `*_atd_enriched_bc.npz`) no están en el manifiesto, porque
el modelo por defecto no los usa.

## 6. Siguientes pasos

1. **La entrega** (06, §5): un CSV del partido completo con *cross-fitting* (regla 1), que se
   regenere bit a bit, y el repo listo para hacerlo público sin datos ni etiquetas.
2. Opcional: más bloques de la 2ª parte para validar mejor el cambio de lado.
3. Abierto: un modelo secuencial que aprenda los límites de las posesiones directamente (por ejemplo,
   sobre ventanas largas). Es arriesgado con 55 min de etiquetas.
