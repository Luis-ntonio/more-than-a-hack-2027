# 03 · Propuesta de solución

**Objetivo:** maximizar el `spellF1` en el partido completo, es decir, recuperar las posesiones reales
como **segmentos enteros**, con el `macroF1` como desempate. La meta mínima es superar 0.2433 por más
de la banda de ruido (±0,04).

---

## 1. Principios de diseño (derivados del EDA y de la métrica)

1. **Es un problema de segmentación temporal, no de clasificación por frame.** Cada posesión inventada
   o partida cuesta mucho. La salida debe salir de un **decodificador temporal** con coste de
   transición, no de un clasificador frame a frame más un suavizado.
2. **`D` solo con confianza.** Una `D` espuria dentro de una posesión la parte en dos. En la duda, se
   mantiene el equipo anterior.
3. **Nunca escribir `X` en frames calificados.** Para spellF1, `X` equivale a repetir la última
   etiqueta. Escribir esa etiqueta explícitamente da lo mismo en spellF1 y puede ganar en macroF1.
   `X` se reserva para los frames no calificados.
4. **El 41 % del tiempo no hay pelota.** El equipo en posesión debe inferirse también a partir de la
   **geometría colectiva** de los 22 jugadores, no solo de la pelota.
5. **"Posesión controlada" = evidencia sostenida.** Un rival que toca o se acerca a la pelota no cambia
   la posesión. El cambio exige que la evidencia a favor del otro equipo se mantenga un tiempo (hay
   histéresis).
6. **No elegir variantes con la ventana pública.** Con 17 posesiones, la varianza supera a casi
   cualquier mejora. Hace falta un conjunto de validación propio.

---

## 2. Arquitectura

```
 ATD táctico ─┐
 Homografía ──┼─► [A] Limpieza y features ─┬─► [B] Detector vivo/muerto ── P(D) ─┐
 SmartTagging ┘    por frame y por ventana │                                     ├─► [D] Decodificador ─► [E] Reglas ─► CSV
 (Vídeo, opc.) ─► [F] Detector de pelota ──┴─► [C] Evidencia de equipo ── P(H) ─┘     HSMM / Viterbi    deterministas
```

### [A] Limpieza y features
- **Pelota:** eliminar saltos imposibles (más de 2 m por frame) y tramos estáticos de 2 s o más
  (falsos positivos), interpolar huecos de 8 frames o menos y suavizar con un filtro de Kalman
  (velocidad y aceleración).
- **Jugadores:** pasar a metros (105,3 × 68). Normalizar la orientación por tiempo para que cada equipo
  ataque siempre "hacia +x", usando la dirección del XML y validándola con los porteros (slots 24 y 46).
- **Features por frame:**
  - *Pelota y control:* distancia de cada equipo a la pelota (mínimo y softmin), diferencia
    `d_home − d_away`, velocidad de la pelota, ángulo del movimiento de la pelota hacia compañeros o
    rivales, y jugadores de cada equipo a menos de 2, 5 y 10 m.
  - *Geometría colectiva* (no necesita pelota): centroide y dispersión por equipo, área de la
    envolvente convexa, altura de la línea defensiva, jugadores por delante y por detrás del centroide
    rival, velocidad media y dirección del movimiento colectivo (quién avanza y quién repliega), y
    distancia entre bloques.
  - *Contexto de juego:* velocidad media global, movimiento de cámara (Δ de la homografía), número de
    jugadores trackeados, longitud del hueco de pelota actual y posición de la pelota al empezar el
    hueco (cerca de la banda o de la línea de fondo).
- **Features de ventana** (±1, ±3 y ±10 s): medias, mínimos y tendencias de todo lo anterior.

### [B] Detector vivo/muerto → P(D)
Señales que el EDA muestra como discriminantes:
- la pelota falta o lleva mucho tiempo ausente, con la velocidad de los jugadores baja (1,4 frente a
  2,4 m/s);
- la cámara está quieta;
- la pelota fue vista por última vez fuera o junto a una línea;
- hay una etiqueta de `SET PIECES` cercana: el saque marca el **final** de un tramo muerto;
- hay un gol, que implica pelota muerta hasta el siguiente `Kick Off (After Goal)`;
- patrones de reanudación, como jugadores agrupados en el área (córner), un jugador inmóvil sobre la
  pelota o un jugador en la banda (saque de banda).

Modelo: un clasificador sobre features de ventana (gradient boosting, ligero y explicable), con
salida calibrada. Después, la decisión se segmenta con duración mínima de ~2 s para `D`: el GT tiene
tramos de `D` de 5–20 s.

**Reglas fijas:**
- `D` desde el inicio calificado de la 2ª parte (110.808) hasta el saque inicial (≈113.660–113.742,
  a afinar con la pelota y el tracking).
- Después de cada gol, `D` hasta el saque.

### [C] Evidencia de equipo → P(H | vivo)
1. **Control cercano:** cuando la pelota es visible y lenta y hay un jugador a menos de 1–1,5 m, el
   control es casi seguro (91 % de acierto a menos de 1 m). Se pondera por distancia, en vez de un
   "más cercano" binario.
2. **Cadenas de pases:** que la pelota viaje de un jugador a otro del mismo equipo refuerza la
   posesión. Que vaya hacia un rival que luego la controla indica un cambio.
3. **Geometría colectiva:** un modelo entrenado para predecir el equipo **solo** con features que no
   dependen de la pelota. Es el que cubre los huecos.
4. **Prior del Smart Tagging**, ponderado por fiabilidad: Build Up, Creating Chances y Long Ball
   pesan mucho; Progression y las transiciones, poco; los códigos de zona (final third, box), casi
   nada. Se ignoran los espejos defensivos duplicados y los límites temporales exactos.
5. **Reanudación:** al pasar de `D` a viva, la posesión es del equipo que ejecuta el saque. Se
   determina por el jugador sobre la pelota en la reanudación, la etiqueta `Team` del saque y, en los
   saques de banda (N/A), el lado del jugador en la línea.

**Entrenamiento sin apenas GT, con pseudoetiquetas:**
- En todo el partido se extraen momentos de **control de alta confianza**: pelota visible, velocidad
  baja, un jugador a menos de 1 m durante 0,5 s o más y sin rival a menos de 2 m. Su equipo es casi
  seguro y da **miles de frames etiquetados** repartidos por los dos tiempos.
- Con ellos se entrena el modelo de geometría colectiva (solo features sin pelota). Así aprende cómo
  se coloca un equipo cuando ataca y cuando defiende, y se aplica a los frames sin pelota.
- Se valida con el GT público y con el etiquetado manual de validación (sección 4).

### [D] Decodificador temporal (el núcleo)
Un **HMM o semi-Markov (HSMM) de 3 estados `H`, `A`, `D`**, decodificado con Viterbi:
- **Emisiones:** la combinación log-lineal de P(D) de [B] y P(H | vivo) de [C].
- **Transiciones:**
  - `H↔A` con coste λ alto. Es la histéresis de la "posesión controlada": hace falta evidencia
    sostenida del otro equipo para pagar el cambio.
  - `vivo→D` solo es barato cuando P(D) es alta.
  - `D→H/A` se reparte según la evidencia de reanudación.
- **Duraciones (en la variante HSMM):** distribuciones de duración por estado. Para `D`, un mínimo de
  ~2 s. Para las posesiones no se fija mínimo duro, porque el GT tiene posesiones de 0,8 s; se usa un
  prior suave.
- **Ventaja:** optimiza la coherencia de segmentos de forma global, que es lo que mide spellF1. Sus
  pocos hiperparámetros (λ, el peso del prior del tagging y el umbral de D) se ajustan en validación.

### [E] Reglas deterministas finales
- `X` en todo frame no calificado. En los calificados, siempre `H`, `A` o `D`.
- Fusionar un `D` de menos de 1 s que separa dos posesiones del mismo equipo, salvo que haya evidencia
  fuerte de parada.
- Validación del archivo con `evaluation/validate_submission.py` como último paso del pipeline.

### [F] (Opcional, fase 2) Detección de pelota en el vídeo táctico
- Un detector preentrenado (p. ej. YOLO con la clase *sports ball*, o uno ajustado para balón de
  fútbol), ejecutado **en local**, solo en los frames calificados sin pelota (~65.000 frames). Las
  detecciones se proyectan al campo con la homografía inversa, que es 100 % válida en el táctico.
- Puede recuperar parte del 41 % sin pelota, sobre todo en juego vivo, y así mejorar [B] y [C].
- Coste estimado: de 1 a 3 h de GPU para esos frames a 1080p. Antes de comprometerse, se mide el
  recall en una muestra.
- **No se sube vídeo ni frames a servicios en la nube** (restricción de los términos).

**Se descarta inicialmente:** el ATD y el vídeo broadcast. No están sincronizados, un tercio no tiene
homografía y la mediana es de 14 jugadores por frame. Su valor marginal es bajo frente al coste de
alinearlos. Se documentará como "probado o descartado".

---

## 3. Hoja de ruta incremental

Cada versión es una submission válida y reproducible, y se mide en la validación propia.

| versión | contenido | qué mide |
|---|---|---|
| **v0** | Baseline del repo, pero cuando falta la pelota se arrastra el último equipo en vez de poner `D`, más las reglas fijas de pre-saque y gol | Cuánto daño hacía el "sin pelota ⇒ D" |
| **v1** | HMM de 3 estados sobre "más cercano ponderado por distancia" + P(D) heurística (hueco largo + velocidad baja + cámara quieta) | El salto de "suavizado" a "decodificación con coste de transición" |
| **v2** | Clasificador vivo/muerto entrenado + reanudaciones por saque | Calidad de `D` y de las transiciones después de una parada |
| **v3** | Modelo de geometría colectiva con pseudoetiquetas + prior del tagging ponderado | El equipo durante los huecos de pelota |
| **v4** | HSMM con duraciones + ajuste fino de λ y los pesos en validación | Fragmentación y posesiones cortas |
| **v5 (opc.)** | Pelota detectada en vídeo fusionada con el ATD | Recuperación de huecos |

---

## 4. Estrategia de validación

1. **GT público (5 min):** solo para comprobar que el pipeline es correcto (alineación, formato y
   significado de las etiquetas). Nunca para elegir entre variantes.
2. **Conjunto de validación etiquetado a mano.** Las reglas lo permiten: *"small amounts of
   hand-labelled data used to train or validate a model are fine"*.
   - Se etiquetan unos 4–6 bloques de 5 min repartidos en los dos tiempos (≈20–30 min, unas 80–120
     posesiones), viendo el vídeo táctico.
   - Hay que escribir un **protocolo** que imite el GT: cuándo una pelota está muerta, qué cuenta como
     control y qué hacer con rebotes y duelos aéreos. El protocolo se calibra primero reproduciendo los
     5 min públicos y midiendo el acuerdo con ellos.
   - Dos personas etiquetan un mismo bloque para medir el acuerdo entre anotadores.
   - Se usan **solo** para validar o seleccionar hiperparámetros. La submission sale del algoritmo.
3. **Métrica:** se usa el propio `possession_metrics.py` (spellF1 y macroF1) con `--bootstrap 1000`.
   Una mejora solo cuenta si los intervalos no se solapan.
4. **Diagnósticos sin GT:** número total de posesiones, distribución de duraciones, porcentaje de `D`
   y de posesión local, comparados entre versiones. Sirven para detectar fragmentación, no para
   ajustar.
5. **Ablaciones** para el documento técnico: quitar el tagging, quitar la geometría, quitar [B], etc.

---

## 5. Plan de trabajo (21 oct → 17 nov 2026, ~4 semanas)

| semana | foco | entregable interno |
|---|---|---|
| **S1** (21–27 oct) | Módulo de features (el entorno Docker ya está listo), v0 y v1. Protocolo de etiquetado y primeros 2 bloques de validación | v1 validada; protocolo escrito |
| **S2** (28 oct–3 nov) | Detector vivo/muerto (v2), reanudaciones, resto del etiquetado de validación. Prueba rápida de un detector de pelota en vídeo sobre una muestra | v2; decisión go/no-go sobre [F] |
| **S3** (4–10 nov) | Pseudoetiquetas + modelo de geometría (v3), prior del tagging, HSMM (v4). [F] si hay go | v4; ablaciones |
| **S4** (11–17 nov) | Congelar el modelo el **14 nov**. Comprobar la reproducibilidad bit a bit en una máquina limpia, escribir el documento técnico, validar y enviar **antes del 16 nov** (con un día de margen) | Submission final |

**Reparto sugerido (3–5 personas):**
1. Datos y features + reproducibilidad.
2. Detector vivo/muerto y reanudaciones.
3. Modelo de equipo (geometría y pseudoetiquetas) + tagging.
4. Decodificador HMM/HSMM y evaluación.
5. (Opcional) Visión: detector de pelota + etiquetado de validación.

El etiquetado de validación se reparte entre todos al principio.

---

## 6. Riesgos y mitigaciones

| riesgo | mitigación |
|---|---|
| Sobreajuste a la ventana pública o a la validación pequeña | Pocos hiperparámetros, bootstrap, bloques de validación repartidos por todo el partido |
| La definición de "controlada" y "muerta" del GT difiere de la intuición | Calibrar el protocolo y los modelos contra los 5 min públicos antes de etiquetar más |
| Pseudoetiquetas sesgadas (solo hay momentos "fáciles") | Usar features de geometría independientes de la pelota y validarlas en tramos sin pelota de la validación |
| No reproducible bit a bit (orden de punto flotante, semillas, versiones) | Dependencias fijadas, semillas fijas, modelos deterministas (o pesos guardados en el repo), comprobación con hash del CSV |
| El detector de vídeo consume demasiado tiempo | Es opcional, con go/no-go en S2; el pipeline principal no depende de él |
| Filtrar datos del partido en el repo público | `data/` y `analysis/cache/` en el gitignore; revisar `git status` antes de cada push |
| Usar LLMs en la nube con datos del reto | Prohibido por los términos: todo el procesamiento de datos y vídeo se hace en local |

---

## 7. Estructura propuesta del código entregable

```
possession/
  io.py            lectura de ATD, tagging y homografía (con caché)
  clean.py         limpieza y suavizado de pelota y jugadores
  features.py      features de frame y de ventana
  deadball.py      [B] P(D)
  team.py          [C] P(H|vivo), pseudoetiquetas, prior del tagging
  decode.py        [D] HMM/HSMM Viterbi
  rules.py         [E] reglas deterministas
  make_submission.py   punto de entrada único
validation/        protocolo de etiquetado y etiquetas manuales (sin datos del partido)
docs/              documento técnico
requirements.txt   versiones fijadas
```

Comando de reproducción:
`python -m possession.make_submission --data data/ --out <equipo>.csv && python evaluation/validate_submission.py <equipo>.csv`

---

## 8. Próximos pasos inmediatos

1. ~~Entorno común~~: hecho, en la imagen Docker (`docker/README.md`).
2. Implementar v0 y v1 y medirlas contra el GT público, solo para comprobar que todo es correcto.
3. Redactar el protocolo de etiquetado y reproducir los 5 min públicos para calibrarlo.
4. Decidir el equipo y los roles (son obligatorios de 3 a 5 miembros).
