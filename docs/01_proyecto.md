# 01 · Qué contiene el proyecto

**More than a Hack 2027 — Fase 1: Posesión del balón**
Organizan Metrica Sports, FC Barcelona (Barça Innovation Hub) y Mobile World Capital Barcelona.
Repositorio base: `https://github.com/metrica-sports/more-than-a-hack-2027`

---

## 1. El reto en una frase

Para **cada frame** del vídeo táctico de un partido completo (FC Barcelona vs Athletic Club, LaLiga
2024-25, jornada 2), decir **quién tiene la pelota**.

| etiqueta | significado |
|---|---|
| `H` | local (Barcelona) en posesión, pelota viva |
| `A` | visitante (Athletic Club) en posesión, pelota viva |
| `D` | pelota muerta (fuera de juego reglamentario: saque, falta, gol, etc.) |
| `X` | sin respuesta (el sistema se abstiene) |

- Entregable puntuado: un CSV `frame,label` con **exactamente 199.319 filas** (frames 0…199318).
- Se califican **158.857 frames** (los dos tiempos). Antes del partido, el descanso y el final no
  cuentan.
- Ranking automático, sin jurado: **`spellF1`** decide y **`macroF1`** desempata.
- Línea base publicada a batir: **spellF1 = 0.2433** (`baseline_smooth50`).

No es un reto de tracking ni de reconstrucción del estado del juego. El tracking, el etiquetado táctico
y la homografía vienen dados para ir directamente a la pregunta de posesión. Aun así, se permite
mejorarlos o reemplazarlos.

---

## 2. Definición de posesión (CHALLENGE.md)

- Mientras la pelota está viva, **siempre** hay un equipo en posesión. En balones divididos o sueltos,
  la posesión se queda con el último equipo que la tenía.
- Un **cambio de posesión** ocurre cuando un rival obtiene posesión **controlada**. Posibles proxies:
  conservar la pelota unos segundos, o encadenar más de dos pases.
- **No** cuentan como cambio: un toque sin control, una desviación que no intercepta el pase, o un
  duelo aéreo ganado sin control posterior.
- Una posesión es una **racha maximal** de `H` o de `A`. `H H D D H H` son **dos** posesiones.
- El GT se deriva **mecánicamente del tracking corregido** (no de vídeo ni de eventos): `D` donde el
  tracking corregido marca pelota muerta y, si no, el equipo que el tracking marca como dueño. No hay
  suavizado ni ajuste manual.
- No se pide quién "posee" una pelota muerta: si está muerta, es `D`.

---

## 3. Datos entregados (`data/`, 19 GB, no versionados)

| | **táctica** (referencia) | **broadcast** (opcional) |
|---|---|---|
| vídeo | 199.319 frames, 25 fps, 1920×1080 (`BAR-ATH_tactical-001.mp4`, 5,2 GB) | 164.488 frames, con cortes y repeticiones (`BAR-ATH_broadcast-002.mp4`, 14 GB) |
| tracking ATD | 199.319 registros, 46 slots | 164.488 registros, 51 slots |
| homografía | 199.319 filas, 100 % válida | 164.488 filas, 65,8 % válida |
| Smart Tagging | 638 instancias | — |
| pelota visible | 56,9 % de los frames | 50,5 % |

> **Ojo:** los vídeos llegan como `…-001.mp4` y `…-002.mp4`, pero `games.py` espera
> `BAR-ATH_tactical.mp4` y `BAR-ATH_broadcast.mp4`. Cada miembro del equipo debe renombrarlos al
> descomprimir los datos.

### 3.1 Tracking automático (ATD, formato FIFA/EPTS)
- `*_atd.xml`: metadatos con equipos, slots y dirección de ataque. Barcelona ataca de izquierda a
  derecha en la 1ª parte.
- `*_atd.txt`: una línea por frame, `frame_id:jug1;jug2;…;jug46:pelota`. Cada grupo es `x,y,velocidad`.
- Coordenadas **normalizadas 0–1**. Para pasar a metros: ×105,3 y ×68,0 (`games.py`).
- **Sin identidades**: hay 46 *slots* (tracklets) para 22 jugadores. Aparecen, desaparecen e
  intercambian identidad. Solo se conoce el equipo de cada slot: 24 de Barcelona y 22 del Athletic.
- **Sin corregir**: la pelota falta en el 41 % de los frames calificados.
- Alineación: `video_frame = frame_id − 1`.

### 3.2 Smart Tagging (SportsCode XML)
Fases tácticas y balón parado, inferidas automáticamente (pueden tener falsos positivos y falsos
negativos). 14 códigos: Build Up, Progression, Creating Chances, Defending in Attacking/Middle/Defensive
Third, Attacking/Defensive Transition, Set Pieces, Goals, Long Ball, Ball in Final Third, Ball in the
Box y Players in the Box. Las etiquetas son *Team*, *Half*, *Type*, *Side*, etc. El tiempo va en
segundos desde el inicio del vídeo táctico: `frame = round(seg × 25)`. **Todos los saques de banda
tienen `Team = N/A`.**

### 3.3 Homografía (CSV)
Una matriz 3×3 por frame que proyecta coordenadas de campo normalizadas (0–1) a píxeles, con columnas
`valid` y `reprojection_error`. Sirve para superponer el tracking en el vídeo o para pasar detecciones
propias del vídeo al campo.

### 3.4 Tramos calificados

| tramo | frames de vídeo | ¿calificado? |
|---|---|---|
| antes del partido | 0 – 11.184 | no |
| **1ª parte** | **11.185 – 90.651** | **sí** |
| descanso | 90.652 – 110.807 | no |
| **2ª parte** | **110.808 – 190.197** | **sí** |
| después del final | 190.198 – 199.318 | no |

---

## 4. Ground truth disponible

`evaluation/gt/BAR-ATH_possession_gt_public.csv` trae GT real **solo para los primeros 5 minutos**
(frames 11.185–18.684): H 3.454, A 2.577 y D 1.469 frames, con **17 posesiones** de 14,2 s de media.
El resto es `X`. El GT del partido completo es **privado**.

Es para verificar que el pipeline es correcto, **no para elegir entre variantes**. Con 17 posesiones,
la varianza es enorme: un error mueve el spellF1 unos 6 puntos. Los propios baselines lo demuestran,
porque la ventana pública elige `smooth125` y el partido completo prefiere `smooth50`.

---

## 5. Métricas (`evaluation/possession_metrics.py` es la especificación)

**macroF1**: F1 medio por clase (H, A, D) frame a frame. Su rango útil es ≈[0.16, 1] y el ruido es
de ±0,01. Cuenta `X` como fallo.

**spellF1** (ranking): cada posesión (racha maximal de H o A dentro de un tiempo) es un objeto. Las
posesiones predichas y las reales del mismo equipo se emparejan uno a uno con IoU temporal (greedy),
y se calcula el F1 con umbrales 0,3/0,4/0,5/0,6/0,7, luego se promedia. Una constante vale 0. Un
retraso de 1 frame es gratis; 1 s de retraso baja a 0,65. El ruido en el partido completo es de ±0,04.

Consecuencias prácticas:
- **Fragmentar destruye el score**: cada posesión inventada es un falso positivo y además rompe la
  real en trozos con IoU bajo. `baseline_raw` hace unas 1.950 posesiones en el partido y saca 0,05.
- **Una `D` espuria dentro de una posesión la parte en dos.**
- **`X` = repetir la última etiqueta** para spellF1 (relleno hacia delante dentro del tiempo). Escribir
  esa etiqueta explícitamente da exactamente lo mismo en spellF1 y puede sumar en macroF1. Por tanto,
  `X` solo tiene sentido en frames no calificados.

---

## 6. Estructura del repositorio

```
games.py                    constantes del partido, rutas de assets y alineación de frames
evaluation/
  possession_metrics.py     librería de métricas (la definición oficial)
  run_evaluation.py         scorer oficial (el mismo que usan los organizadores)
  validate_submission.py    validación previa obligatoria
  gt/                       GT público de 5 min + meta JSON
  results/                  4 baselines (aquí van las submissions)
starter/
  baseline.py               baseline "jugador más cercano + voto mayoritario", solo numpy
  load_atd.py               lectura del ATD (kloppy) y comprobación de alineación
  load_smart_tagging.py     tagging → intervalos de frames, sin kloppy
  load_homography.py        carga y proyección campo↔píxel
  utils.py                  dibujo del campo
  01/02/03_*.ipynb          notebooks guiados
data/                       datos del partido (gitignored)
analysis/                   [nuevo] scripts del EDA de este análisis (caché gitignored)
docs/                       [nuevo] estos documentos y figuras
```

Baselines del repo (los cuatro usan el mismo algoritmo y solo cambia la ventana del voto):

| baseline | macroF1 (completo) | spellF1 (completo) | spellF1 (público) | posesiones predichas (completo) |
|---|---|---|---|---|
| raw | 0.6532 | 0.0486 | 0.0378 | 1.949 |
| smooth25 | **0.6752** | 0.1996 | 0.2868 | 746 |
| smooth50 | 0.6724 | **0.2433** | 0.3256 | 594 |
| smooth125 | 0.6356 | 0.2242 | **0.4242** | 398 |

---

## 7. Entregables (antes del **17 de noviembre de 2026**)

1. **Archivo de posesión** `<equipo>.csv`. Validarlo siempre con
   `python evaluation/validate_submission.py <archivo>.csv`.
2. **Repositorio público de código**: fuente, dependencias fijadas y el comando que regenera **el
   mismo archivo exacto** (reproducibilidad bit a bit). **Sin los datos.**
3. **Documento técnico** `<equipo>.pdf` o `.md`, de pocas páginas: enfoque, assets usados y
   descartados, manejo del ruido, reglas de posesión implementadas, limitaciones y reproducción.

Un único formulario por equipo (hace falta cuenta de Google). No hay leaderboard ni feedback antes
del cierre. Los resultados se publican antes del 26 de noviembre de 2026.

## 8. Reglas clave

- Se permite **cualquier método**, siempre que los organizadores puedan acceder a él e
  inspeccionarlo, y que el documento lo explique. Se permiten modelos preentrenados, datos externos y
  LLMs.
- **Prohibido** etiquetar a mano la submission. Sí se permiten **pequeñas cantidades de etiquetado
  manual para entrenar o validar** un modelo.
- Prohibido usar el GT privado o cualquier otra fuente de posesión corregida de este partido.
- Equipos de 3 a 5 personas.
- Datos con licencia solo para la competición: no redistribuirlos ni publicar derivados que los
  reconstruyan.
- **Restricción sobre LLMs:** no se pueden subir vídeos, datasets ni extractos de los datos a LLMs
  alojados por terceros o en la nube. El análisis con los datos debe hacerse en local.

## 9. Entorno del equipo

Hay una imagen Docker común con Python 3.11.13 y las dependencias fijadas en
`docker/requirements.lock.txt`: numpy, pandas, scipy, scikit-learn, lightgbm, matplotlib, kloppy
3.18, OpenCV y JupyterLab. Las instrucciones están en [`docker/README.md`](../docker/README.md).
Está verificado que el evaluador da los mismos números que el README, que kloppy lee el ATD y que
OpenCV abre el vídeo táctico (199.319 frames a 25 fps).
