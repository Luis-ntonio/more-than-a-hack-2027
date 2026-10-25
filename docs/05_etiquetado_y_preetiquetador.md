# 05 · Etiquetado a mano y pre-etiquetador

Estado al 24-10-2026: **40 min de la 1ª parte y 15 min de la 2ª etiquetados** (bloques
`1T-00`…`1T-07` y `2T-00`…`2T-02`, todos desde cero, anotador `luis`). Las secciones 1–8 son con los
40 min de la 1ª parte; la 9 añade la 2ª. Las etiquetas están solo en el repo privado de etiquetas; aquí van solo
cifras agregadas.

```bash
docker compose run --rm dev python -m labeler.timing                       # desfase frente al GT
docker compose run --rm dev python -m possession.prelabel                  # entrena + validación cruzada
docker compose run --rm dev python -m possession.prelabel --no-shift       # igual, sin corregir el desfase
docker compose run --rm dev python -m labeler.export --annotator luis --shift auto
docker compose run --rm dev python evaluation/run_evaluation.py --gt labels/validation_luis.csv --bootstrap 1000
```

## 1. Desfase del anotador frente al GT (`labeler.timing`)

`1T-00` es la ventana pública: ahí se compara la etiqueta a mano con el GT oficial.

| tipo de cambio | n | mediana | media | corrección aplicada |
|---|---|---|---|---|
| a muerta (`dead`) | 4 | −1 f | −1,0 f | +1 f |
| reanudación (`restart`) | 4 | −15 f | −16,8 f | +15 f |
| cambio de posesión | 12 | −4,5 f | −7,2 f | +4 f |

Acuerdo por frame: 97,9 %; 20 cambios emparejados, 0 omitidos, 0 de más. El anotador marca los
cambios antes que el GT: las reanudaciones ~0,6 s antes, porque el GT las pone cuando la pelota ya
rueda, no en el golpeo (ver `analysis/gt_changes.py`). La corrección se aplica al entrenar y al
exportar; las etiquetas guardadas no se tocan.

## 2. ¿Sirve la corrección del desfase?

La única comparación limpia es el bloque público, porque su verdad es el GT oficial en los dos casos
(en los demás bloques la propia verdad cambia con el desfase).

| bloque público (validación cruzada) | spellF1 | macroF1 |
|---|---|---|
| entrenado sin corregir el desfase | 0,413 | 0,888 |
| **entrenado con el desfase corregido** | **0,538** | 0,887 |

Con el timing alineado, los cambios predichos caen donde los pone el GT, y spellF1 depende de eso
en las posesiones cortas. Es un solo bloque de 17 posesiones, así que es una señal, no una prueba.

## 3. Pre-etiquetador: validación dejando un bloque fuera

Cada fila se entrena con los otros 7 bloques y se puntúa en el que queda fuera. "Correcciones" son
los cambios que habría que añadir, mover o borrar para llegar a la etiqueta.

| bloque | spellF1 | macroF1 | spellF1 v1 | correcciones pre / v1 | cambios reales |
|---|---|---|---|---|---|
| público | 0,538 | 0,887 | 0,389 | 25 / 35 | 20 |
| 1T-01 | 0,667 | 0,826 | 0,465 | 17 / 29 | 16 |
| 1T-02 | 0,588 | 0,688 | 0,568 | 31 / 38 | 21 |
| 1T-03 | 0,455 | 0,824 | 0,287 | 27 / 40 | 20 |
| 1T-04 | 0,348 | 0,788 | 0,374 | 26 / 29 | 23 |
| 1T-05 | 0,585 | 0,804 | 0,387 | 20 / 35 | 15 |
| 1T-06 | 0,462 | 0,854 | 0,486 | 9 / 15 | 10 |
| 1T-07 | 0,567 | 0,824 | 0,416 | 19 / 26 | 15 |
| **media / total** | **0,526** | 0,812 | 0,422 | **174 / 247** | 140 |

Gana a v1 en 6 de 8 bloques y ahorra un 30 % de correcciones. Las cifras son las mismas que con
estos mismos bloques en la sesión anterior: no se ha añadido ningún bloque nuevo desde entonces.

## 4. Scoreboard honesto sobre las etiquetas a mano

`prelabel.py` escribe ahora también `evaluation/results/prelabel_oof.csv`: la predicción completa,
pero con cada bloque etiquetado sustituido por la del modelo que **no** lo vio al entrenar
(*out-of-fold*). Es la única fila comparable del scoreboard; `prelabel.csv` está entrenado con esas
mismas etiquetas y su 0,98 es memoria, no acierto.

Verdad: 7 bloques (`1T-01`…`1T-07`, 35 min, 92 posesiones de 13,2 s de media), con el desfase
corregido.

| submission | spellF1 | macroF1 | transF1 | posesiones (verdad 92) | duración media |
|---|---|---|---|---|---|
| prelabel (dentro de muestra, no vale) | 0,985 | 0,996 | 0,992 | 90 | 13,5 s |
| **prelabel_oof** | **0,501** | **0,820** | **0,344** | 78 | 16,1 s |
| v1 | 0,400 | 0,774 | 0,209 | 102 | 11,6 s |
| v0 | 0,261 | 0,372 | 0,136 | 146 | 14,4 s |
| baseline_smooth125 | 0,245 | 0,652 | 0,034 | 150 | 11,0 s |

**Ruido: con 92 posesiones, diferencias de spellF1 por debajo de ~0,17 no son significativas**
(bootstrap de 1000 muestras). La ventaja de +0,10 sobre v1 va en la dirección correcta y se repite
en 6 de 8 bloques, pero todavía cabe en el ruido.

## 5. Qué dicen los números

- **El modelo se queda corto de posesiones**: predice 78 donde hay 92, con una duración media de 16,1
  s frente a 13,2 s. Se come sobre todo las posesiones cortas (rebotes, intercepciones que se pierden
  enseguida), que son justo las que más pesan en spellF1. v1 se pasa por el lado contrario (102).
- El macroF1 (0,82) es bueno: el problema no es el equipo frame a frame, es **dónde empiezan y acaban
  las posesiones**, sobre todo las cortas. Las secciones 6–8 muestran que no se arregla en el
  decodificador: el clasificador no tiene evidencia en esas posesiones.
- `1T-04` es el bloque más difícil para todos los modelos (0,35): conviene revisarlo con el
  visualizador.

## 6. Ajuste del decodificador: no ayuda

Se probaron 150 combinaciones de costes de transición del HMM (cambio de equipo 4–40,
a muerta 4–25, desde muerta 4–25 nats) sobre las probabilidades fuera de muestra de cada bloque. La
elección es anidada: los costes con los que se puntúa un bloque se eligen sin él.

| costes | spellF1 medio (8 bloques) | correcciones |
|---|---|---|
| v1 (25 / 15 / 15) | **0,526** | **174** |
| mejor punto de la rejilla (40 / 4 / 25), elegido sobre los mismos bloques | 0,527 | — |
| anidado (elegido sin el bloque) | 0,521 | 209 |

La superficie es plana: ningún coste recupera las posesiones cortas. **El límite no está en el
decodificador.**

## 7. Por qué se pierden las posesiones cortas (`analysis/short_possessions.py`)

Las 113 posesiones verdaderas de los 8 bloques, por duración. `p_team` es la probabilidad fuera de
muestra del equipo correcto, y "encontrada" significa que hay una posesión predicha del mismo
equipo con IoU ≥ 0,5.

| duración | n | p_team | frames bien | balón visto | encontrada |
|---|---|---|---|---|---|
| 0–2 s | 18 | 0,41 | 37 % | 90 % | 0 % |
| 2–4 s | 21 | 0,55 | 57 % | 99 % | 24 % |
| 4–8 s | 20 | 0,56 | 55 % | 96 % | 45 % |
| 8–15 s | 18 | 0,83 | 83 % | 98 % | 67 % |
| 15–30 s | 25 | 0,94 | 94 % | 92 % | 80 % |
| > 30 s | 11 | 0,94 | 95 % | 97 % | 100 % |

- **La mitad de las posesiones dura menos de 8 s**, y en ellas el clasificador está al nivel del azar
  aunque el balón se vea casi siempre. Con `p_team` ≈ 0,5 ningún decodificador puede recuperarlas.
- Con proxies simples (scratch, solo agregados), el equipo etiquetado sí llega a menos de 1–1,5 m del
  balón en las posesiones cortas, y "último equipo a menos de 1,5 m" acierta el 72 % de los frames
  en las de 2–4 s. Pero solo el 51 % en las de 0–2 s, y entre el 72 % y el 85 % incluso en las largas:
  la relación balón–jugador del tracking es ruidosa (balón proyectado al suelo, cambios de ID).
## 8. Otras dos pruebas sin éxito

Todas se puntúan igual, con validación dejando un bloque fuera y el spellF1 medio de los 8 bloques.

| variante | spellF1 medio | correcciones |
|---|---|---|
| **pre-etiquetador original** | **0,526** | **174** |
| + variables de toque (último y siguiente equipo a 1,5 m y 3 m, tiempo desde y hasta el toque) | 0,507 | 183 |
| + toque + frames ponderados por duración^−0,5 de su posesión | 0,488 | — |
| + toque + frames ponderados por duración^−1 (cada posesión pesa igual, como spellF1) | 0,482 | — |

Ninguna ayuda, y todas las diferencias caen dentro del ruido. Con 8 bloques, el clasificador no
tiene de dónde sacar las posesiones cortas: la evidencia del tracking en ellas es casi aleatoria.
**Se quitaron del código**, y `prelabel.py` queda como estaba más la copia *out-of-fold*. El diff de
los experimentos se guardó fuera del repo, en local.

Lo que queda por probar, de mayor a menor potencial:
- **Más datos de otro régimen**: los 15 minutos de la 2ª parte que se están etiquetando.
- **Mejor evidencia de toque desde el vídeo**: la posición del balón en el suelo y los cambios de ID
  del tracking limitan la relación balón–jugador. Un detector de jugadores en el vídeo (Roboflow,
  con GPU) daría la distancia real en píxeles al balón detectado.
- **Revisar la convención del GT para las posesiones cortas**: si el GT marca una posesión cuando el
  balón queda "libre" tras una intercepción, puede que la señal no sea la cercanía sino el cambio de
  dirección del balón (el toque). Se puede medir con `analysis/gt_changes.py` en las cortas.

## 9. Con 15 min de la 2ª parte (11 bloques, 55 min)

Validación dejando un bloque fuera: cada bloque se predice con un modelo entrenado con los otros 10.

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

- **Funciona en la 2ª parte**: media de 0,524 en los tres bloques de la 2ª, frente a 0,483 de v1. Con
  los equipos cambiados de lado no se cae. Ojo: cada bloque de la 2ª se entrena también con los otros
  dos de la 2ª, así que no es una prueba pura de "entrenar en la 1ª y predecir la 2ª".
- Gana a v1 en 9 de 11 bloques y ahorra un 30 % de correcciones (244 frente a 346), igual que antes.
- El bloque público baja de 0,538 a 0,428 al añadir datos, y la media de la 1ª parte pasa de 0,526 a
  0,505. Con 17–20 posesiones por bloque, esto es ruido de bloque, no un empeoramiento.
- **Más datos no arreglan las posesiones cortas** (`analysis/short_possessions.py`, 166 posesiones):

| duración | n | p_team | frames bien | balón visto | encontrada |
|---|---|---|---|---|---|
| 0–2 s | 26 | 0,38 | 38 % | 87 % | 0 % |
| 2–4 s | 29 | 0,57 | 57 % | 97 % | 14 % |
| 4–8 s | 31 | 0,60 | 61 % | 96 % | 45 % |
| 8–15 s | 30 | 0,81 | 81 % | 97 % | 67 % |
| 15–30 s | 35 | 0,92 | 93 % | 92 % | 71 % |
| > 30 s | 15 | 0,89 | 90 % | 98 % | 93 % |

  Con un 47 % más de datos, `p_team` en las posesiones de menos de 8 s sigue en 0,38–0,60. El cuello
  de botella es la evidencia, no la cantidad de etiquetas.

Scoreboard sobre las etiquetas a mano: 10 bloques (`1T-01`…`2T-02`, 50 min, 144 posesiones de 12,6 s
de media), con el desfase corregido y bootstrap de 1000 muestras.

| submission | spellF1 | macroF1 | transF1 | posesiones (verdad 144) | duración media |
|---|---|---|---|---|---|
| prelabel (dentro de muestra, no vale) | 0,990 | 0,996 | 0,995 | 142 | 12,8 s |
| **prelabel_oof** | **0,499** | **0,812** | **0,380** | 110 | 16,6 s |
| v1 | 0,425 | 0,764 | 0,228 | 150 | 11,8 s |
| v0 | 0,300 | 0,386 | 0,137 | 207 | 14,5 s |
| baseline_smooth50 | 0,263 | 0,665 | 0,111 | 299 | 7,4 s |

- La banda de ruido baja de ~0,17 a **~0,14** con 144 posesiones. La ventaja sobre v1 (+0,074) sigue
  **dentro del ruido**, aunque va en la misma dirección en 9 de 11 bloques.
- Donde sí hay una diferencia clara es en **transF1: 0,380 frente a 0,228**. El pre-etiquetador pone
  los cambios que encuentra mucho más cerca del frame correcto (latencia media +0,4 f frente a
  −2,1 f).
- El defecto es el mismo que antes: **110 posesiones donde hay 144**, de 16,6 s de media frente a
  12,6 s. Se fusionan las posesiones cortas con sus vecinas. v1 acierta el número (150) pero las coloca
  peor.

## 10. ¿Hay señal de toque en el tracking? (`analysis/touch_signal.py`)

Un "toque" es un frame en el que el balón del tracking gira más de un ángulo o cambia de velocidad
más de un umbral entre los 5 frames anteriores y los 5 siguientes, con un jugador cerca. Su equipo
es el del jugador más cercano. Se mide sobre los 11 bloques (55 min, 109 cambios H↔A):

- *recall*: cambios con un toque del nuevo equipo a ±1 s;
- precisión: toques del rival que de verdad son un cambio hacia ese equipo.

| umbral (giro / Δv / radio) | toques/min | recall 0–2 s | 2–4 s | 4–8 s | 8–15 s | > 15 s | precisión |
|---|---|---|---|---|---|---|---|
| 25° / 3 m/s / 1,5 m | 11,8 | 36 % | 57 % | 42 % | 35 % | 40 % | 42 % |
| 35° / 3 m/s / 1,5 m | 10,1 | 36 % | 48 % | 37 % | 25 % | 36 % | 41 % |
| 50° / 4 m/s / 1,5 m | 5,1 | 18 % | 39 % | 32 % | 10 % | 24 % | 43 % |
| 35° / 3 m/s / 2,5 m | 12,3 | 45 % | 57 % | 53 % | 35 % | 48 % | 40 % |
| 35° / 3 m/s / 4 m | 13,1 | 50 % | 57 % | 58 % | 30 % | 48 % | 37 % |

- **El tracking no ve la mayoría de los toques.** Solo hay un toque del nuevo equipo cerca de entre el
  35 % y el 58 % de los cambios, y no depende de la duración de la posesión: las largas también salen
  mal. Esas se encuentran por la cercanía sostenida, no por el toque.
- **Y no es por falta de balón.** El balón del tracking se ve en el 87 % de los frames a ±1 s de un
  cambio; solo en el 14 % de los cambios falta más de la mitad de esa ventana. El balón está, pero su
  trayectoria no muestra los toques: ~10–13 toques/min es muy poco para un partido. Probablemente
  está suavizado, o pegado al suelo en los pases altos.
- **Precisión ~40 %**: de cada 5 toques del rival, 3 no son un cambio (desvíos, duelos perdidos,
  ruido). Una variable de toque sola no basta, y por eso las de la sección 8 no ayudaron.

**Conclusión: la información para las posesiones cortas no está en el tracking.** Para recuperarlas
hay que sacar los toques del vídeo.

## 11. Toques con el balón del vídeo (fase 1)

```bash
docker compose run --rm gpu python -m enrich.detect_ball --mode visible   # 94 214 frames, 106 min
docker compose run --rm dev python analysis/touch_signal.py --ball detected
docker compose run --rm dev python analysis/touch_signal.py --ball hybrid
```

`--mode visible` pasa el detector por todos los frames calificados que tienen balón del tracking.
Con `--mode missing`, cubre todo el partido. El "balón detectado" es, en cada frame, el candidato
más cercano al balón del tracking (a menos de 3 m), y en los huecos del tracking, el que tomó
`merge`. El híbrido usa el detectado donde lo hay y el del tracking en el resto.

- El detector encuentra el balón del tracking a menos de 3 m en el **60 %** de los frames, a una
  mediana de 0,68 m de él.
- Se añaden dos medidas:
  - **azar**: el mismo *recall* con la ventana desplazada 3 s del cambio, para ver cuánto es casualidad;
  - **confirmado**: un toque del rival seguido de otro toque del mismo equipo en menos de 3 s, antes
    de que toque el otro equipo. Separa "controla" de "desvía".

Umbral 35° / 3 m/s / 2,5 m:

| balón | toques/min | cambios con toque | azar | precisión | precisión confirmado | cambios que conserva |
|---|---|---|---|---|---|---|
| tracking | 12,3 | 48 % | 25 % | 40 % | 52 % | 40 % |
| **detectado** | 27,3 | 53 % | 35 % | 38 % | **54 %** | **75 %** |
| híbrido | 43,9 | 84 % | 53 % | 34 % | 43 % | 79 % |

- **Solos, los toques apenas superan al azar**: con más toques por minuto, el azar también sube. El
  híbrido parece muy bueno (84 %), pero la mitad es casualidad, y al alternar entre dos fuentes que
  difieren 0,68 m crea giros falsos.
- **Lo que sí funciona es la confirmación con el balón detectado**: la precisión sube del 38 % al
  54 % y se conserva el 75 % de los cambios verdaderos. Con el tracking, confirmar solo conserva el
  40 %, porque ve muy pocos toques. El balón del vídeo sí trae información que el tracking no tiene.
- Aun así, casi la mitad de los toques confirmados del rival no son un cambio. Como variable del
  modelo puede ayudar; como regla, no.

## 12. Siguientes pasos

1. Añadir al pre-etiquetador variables del balón detectado: último y siguiente toque **confirmado**
   y el tiempo desde y hasta él. Medirlo con la misma validación dejando un bloque fuera. Esto hace
   que el modelo dependa de la inferencia en GPU, así que antes hay que decidir el punto 3.
2. Entrega por *cross-fitting* (regla 1): cada bloque etiquetado se predice con el modelo que no lo
   vio, y el resto del partido con el modelo entrenado con todo. `prelabel_oof.csv` ya sigue ese
   esquema.
3. Decidir si la entrega usa detecciones de Roboflow (GPU): para la regla 3 hay que comprobar que
   son deterministas o fijar su salida como artefacto.
