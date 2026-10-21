# Etiquetador de posesión

Herramienta local para etiquetar a mano bloques de 5 minutos del vídeo táctico. Las etiquetas sirven
como conjunto de validación propio para ajustar los modelos.

```bash
docker compose up labeler          # abre http://localhost:8765
```

Todo se queda en tu máquina. El vídeo se lee de `data/` en solo lectura, y las etiquetas se guardan
en `labels/<bloque>__<anotador>.json`. Esa carpeta está en el gitignore y bloqueada para Claude Code.

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
