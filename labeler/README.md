# Etiquetador de posesión

Herramienta local para etiquetar a mano bloques de 5 minutos del vídeo táctico. Las etiquetas sirven
como conjunto de validación propio para ajustar los modelos.

```bash
docker compose up labeler          # abre http://localhost:8765
```

Todo se queda en tu máquina. El vídeo se lee de `data/` en solo lectura, y las etiquetas se guardan
en `labels/<bloque>__<anotador>.json`. Esa carpeta está en el gitignore y bloqueada para Claude Code.

## Visualizador de tracking

La pestaña **Visualizar** (http://localhost:8765/overlay) muestra el vídeo con capas que se activan
y desactivan con casillas o con las teclas `1`…`8`:

- líneas de la cancha;
- jugadores de cada equipo, con su número de slot;
- balón del tracking y su estela;
- balón con el que Roboflow rellenó los huecos;
- las 5 detecciones brutas de YOLO, con su confianza.

Todo se proyecta con la homografía de cada frame. Las cajas de los jugadores son aproximadas,
porque el tracking solo da la posición de los pies. Las de los candidatos de Roboflow son las
reales del detector. Las capas de Roboflow requieren haber ejecutado `enrich` (ver
`enrich/README.md`).

## Cómo se etiqueta

No se etiqueta frame a frame: **solo se marcan los cambios**. Al pulsar `H`, `A` o `D` en un frame,
la etiqueta vale desde ahí hasta el siguiente cambio. Un bloque de 5 minutos son unas pocas decenas
de pulsaciones.

1. Escribe tu nombre de anotador y elige un bloque. El selector muestra tu progreso y quién más lo
   ha etiquetado.
2. En el primer frame del bloque, pulsa la etiqueta del estado inicial.
3. Reproduce a 1× con `Espacio` y pulsa `H`, `A` o `D` cada vez que cambie el estado.
4. Afina cada cambio:
   - `[` y `]` saltan entre cambios.
   - `←` y `→` avanzan o retroceden un frame.
   - `Alt`+flechas mueven el cambio sobre el que estás.
   - El clip de **contexto** (panel derecho) repite en bucle ±2, ±4 u ±8 s alrededor del frame
     actual, con la etiqueta de cada momento superpuesta.
5. Cuando termines, marca **Bloque terminado**. El guardado es automático.

**Practica primero con el bloque `1T-00`.** Tiene la GT oficial: etiquétalo sin mirarla, pulsa
*Comparar con GT* y usa *Mostrar GT* para ver en qué difiere vuestro criterio del de la organización.
La GT manda (ver `CHALLENGE.md`).

## Dónde pone el GT los cambios, y cómo calibrarte

`analysis/gt_changes.py` mide qué pasa en el ATD en los frames de cambio del GT público:

- **Pelota muerta:** coincide con el pitido o con la salida del campo. Aquí no hay ambigüedad.
- **Reanudación:** el GT cambia de media unos 0,9 s *después* de que el balón se ponga en juego.
  En ese frame el balón ya va camino del receptor.
- **Cambio de posesión:** el GT cambia con el balón en movimiento (≈6,5 m/s) y sin nadie encima
  (≈2,7 m del nuevo equipo, ≈3,7 m del anterior). Unas veces es después del toque de
  interceptación y el rebote; otras, antes de que el nuevo equipo reciba. No hay una regla
  geométrica fija: son solo 12 cambios, y el balón del ATD tiene error.

Por eso **no intentéis clavar el frame del GT a ojo**. Marcad siempre en un momento claro y siempre
igual: el toque que gana el balón en un cambio de posesión, el golpeo en una reanudación, y el
pitido o la salida en una pelota muerta. Después, el export corrige vuestro desfase:

```bash
docker compose run --rm dev python -m labeler.timing                    # desfase de cada uno en 1T-00
docker compose run --rm dev python -m labeler.export --annotator all --shift auto
```

`labeler.timing` empareja tus cambios del bloque `1T-00` con los del GT y calcula la mediana del
desfase por tipo (pelota muerta, reanudación, posesión). `--shift auto` desplaza esa cantidad los
cambios de cada anotador en todos sus bloques. El pre-etiquetador aplica el mismo desplazamiento
antes de entrenar. **Hace falta haber etiquetado `1T-00` desde cero.** Sin ese bloque, tus etiquetas
se usan sin desplazar. Los bloques que partieron de un modelo tampoco se desplazan, porque sus
cambios sin tocar ya están en el timing del modelo.

El desplazamiento solo mueve cambios: no recupera posesiones que no marcaste. En `1T-00`, macroF1
pasa de 0,976 a 0,985 y spellF1 no cambia (0,827), porque lo que cuesta son las dos posesiones
cortas que faltaban. Lo prioritario es no saltarse posesiones cortas; el frame exacto lo arregla el
shift.

## Corregir sobre un modelo en vez de empezar de cero

1. En la barra superior, elige **Modelo**:

   - `v1` para los primeros bloques;
   - `prelabel` en cuanto el pre-etiquetador gane a v1 (ver más abajo).

   Su predicción aparece como una franja extra bajo la tuya en las dos líneas de tiempo (`M` la
   muestra u oculta).
2. Pulsa **Partir de la predicción**. Sus cambios se copian al bloque. Si el bloque ya tenía
   etiquetas, pide un segundo clic para no borrar trabajo por accidente.
3. Revisa los cambios de uno en uno:

   - `]` salta al siguiente.
   - Mira el clip de contexto.
   - Si está bien, sigue; si está desplazado, muévelo con `Alt`+flechas; si sobra, bórralo con
     `Supr`; si falta uno, colócate en el frame y pulsa `H`, `A` o `D`.
   - Entre cambios, reproduce a 1–2× para detectar posesiones que el modelo se saltó.
4. Marca **Bloque terminado** y sube el archivo al repo de etiquetas.

El bloque queda marcado como "desde v1" (o desde el modelo que hayas usado). Corregir sobre un
modelo arrastra sus sesgos: los casos dudosos se tienden a dejar como los marcó. Por eso conviene
etiquetar **2–3 bloques desde cero** y reservarlos para validar
(`labeler.export --exclude-seeded`).

## Pre-etiquetador que aprende de vuestros bloques

```bash
docker compose run --rm dev python -m possession.prelabel
```

- Entrena con la GT pública más todos los bloques marcados como terminados, ya desplazados al
  timing del GT, y escribe `evaluation/results/prelabel.csv`. Aparece en el selector de modelos sin
  reiniciar nada.
- Escribe también `evaluation/results/prelabel_oof.csv`: cada bloque etiquetado lo predice el modelo
  que no lo vio. Es la fila honesta al puntuar contra vuestras etiquetas; `prelabel` está entrenado
  con ellas. Resultados en `docs/05_etiquetado_y_preetiquetador.md`.
- Volved a ejecutarlo cada vez que terminéis uno o dos bloques.
- Imprime, por bloque, cuántas correcciones habrían hecho falta con él y con v1 (añadir, mover o
  borrar cambios). Cuando el total del pre-etiquetador sea claramente menor, pasad a usarlo.

## Usar las etiquetas para evaluar

```bash
docker compose run --rm dev python -m labeler.export --annotator luis   # o --annotator all
docker compose run --rm dev python evaluation/run_evaluation.py --gt labels/validation_luis.csv
```

El export genera un CSV `frame,label` de longitud completa. Los frames que no están en ningún bloque
terminado quedan como `X`, y el evaluador no los puntúa. Si dos personas etiquetan el mismo bloque,
`--annotator all` coge la versión más reciente. Comparar las dos versiones da una medida del acuerdo
entre anotadores.

## Compartir las etiquetas

Las etiquetas nunca van a este repo, que será público. Viven en un repo **privado** aparte, clonado
dentro de `labels/`:

```bash
git clone https://github.com/Luis-ntonio/more-than-a-hack-2027-labels.git labels
cd labels && git pull                                      # antes de etiquetar
git add *.json && git commit -m "1T-03 luis" && git push   # al terminar
```

Cada persona escribe solo sus propios archivos (`<bloque>__<anotador>.json`), así que no hay
conflictos. Para que los compañeros puedan clonarlo, el dueño los añade como colaboradores en GitHub.
