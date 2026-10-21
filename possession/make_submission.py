"""Build a submission CSV from the provided data.

    python -m possession.make_submission --model v1 --out evaluation/results/v1.csv

Prints only aggregate statistics (label shares, possession counts) -- never frame-level data.
"""
import argparse
import os

import numpy as np

from possession import features as feats
from possession.io import FPS, GAME, GAMES, ROOT, graded_blocks, load_atd, load_homography
from possession.models import MODELS


def write_csv(path, labels):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))


def summarize(labels):
    """Aggregate view of a submission: label shares and possession spells on graded frames."""
    graded = np.zeros(len(labels), dtype=bool)
    spells = []
    for _, first, last in graded_blocks():
        graded[first:last + 1] = True
        window = labels[first:last + 1]
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(window)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if window[a] in ("H", "A"):
                spells.append(b - a)
    g = labels[graded]
    shares = {k: float(np.mean(g == k)) for k in "HADX"}
    spells = np.array(spells) / FPS
    print("graded frames: " + "  ".join("{} {:.1%}".format(k, v) for k, v in shares.items()))
    print("possessions: {}  mean {:.1f} s  median {:.1f} s  <2 s: {}".format(
        len(spells), spells.mean(), np.median(spells), int((spells < 2).sum())))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(MODELS), default="v1")
    parser.add_argument("--out", help="CSV path (default evaluation/results/<model>.csv)")
    args = parser.parse_args()

    out = args.out or os.path.join(ROOT, "evaluation", "results", args.model + ".csv")
    features = feats.compute(load_atd(), load_homography())
    labels = MODELS[args.model](features, GAMES[GAME]["n_video_frames"])
    write_csv(out, labels)
    print("{} -> {}".format(args.model, os.path.relpath(out, ROOT)))
    summarize(labels)


if __name__ == "__main__":
    main()
