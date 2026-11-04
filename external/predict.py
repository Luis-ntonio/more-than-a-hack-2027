"""Label the provided match with the models trained on external data.

    docker compose run --rm dev python -m external.train --final
    docker compose run --rm dev python -m external.predict
    docker compose run --rm dev python evaluation/validate_submission.py evaluation/results/external.csv

The match is input only: nothing here reads the ground truth or the hand labels, and no number
measured on the match sets a parameter. It prints format checks, not scores. Graded frames before
each half's kickoff are D, from the kickoff anchors the organisers publish in games.py; ungraded
frames are X.
"""
import os

import numpy as np

from external.features import compute
from external.train import decode, load
from possession.io import GAME, GAMES, PITCH, ROOT, graded_blocks, kickoff_frames, load_atd

OUT = os.path.join(ROOT, "evaluation", "results", "external.csv")


def match_view(first, last, half):
    """One graded block of the provided tracking as a view: metres, pitch centred on (0, 0)."""
    atd = load_atd()
    scale = np.array(PITCH, dtype=np.float64)
    rows = slice(first, last + 1)
    centre = lambda xy: (xy.astype(np.float64) - 0.5) * scale  # noqa: E731
    return {"players": centre(atd["players"][rows, :, :2]), "team": atd["team"].astype(int),
            "ball": centre(atd["ball"][rows]), "half": np.full(last - first + 1, half),
            "pitch": tuple(PITCH)}


def main():
    dead, team, log_t = load()
    n = GAMES[GAME]["n_video_frames"]
    labels = np.full(n, "X", dtype="<U1")
    kickoffs = kickoff_frames()
    for half, first, last in graded_blocks():
        start = max(first, kickoffs[half])   # decode from the kickoff on
        view = match_view(start, last, half)
        X, _ = compute(view)
        labels[start:last + 1] = decode(dead.predict(X), team.predict(X), view["half"], log_t)
        labels[first:start] = "D"
    with open(OUT, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label) for i, label in enumerate(labels))
    print("{} rows -> {}".format(n, os.path.relpath(OUT, ROOT)))


if __name__ == "__main__":
    main()
