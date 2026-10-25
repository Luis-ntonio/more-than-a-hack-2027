"""Why the pre-labeller misses short possessions: evidence or decoder?

    docker compose run --rm dev python analysis/short_possessions.py

For every true possession spell (H or A) in the hand-labelled blocks, compares the out-of-fold
classifier output inside it with what the decoder made of it, bucketed by spell length. Prints
aggregates only:
  - p_team: mean P(that team | alive) from the model that did not see the block
  - frames_right: share of its frames where that probability is above 0.5
  - ball_seen: share of its frames with a ball (tracking or Roboflow)
  - found: share of spells with a predicted spell of the same team covering >= 50% of it (IoU)
If p_team is high but found is low, the decoder smooths them away; if p_team is near 0.5, the
evidence is not there and no decoder setting can recover them.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession.io import FPS, GAME, GAMES, load_atd  # noqa: E402
from possession.models import V1_PARAMS, decode  # noqa: E402
from possession.prelabel import fit, frame_features, probabilities, training_blocks  # noqa: E402

BUCKETS_S = (0, 2, 4, 8, 15, 30, np.inf)


def spells(labels):
    edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(labels)]])
    return [(a, b, labels[a]) for a, b in zip(starts, ends)]


def main():
    n_frames = GAMES[GAME]["n_video_frames"]
    X, _ = frame_features()
    blocks = training_blocks()
    try:
        ball = np.isfinite(load_atd(enriched=True)["ball"][:, 0])
    except SystemExit:
        ball = np.isfinite(load_atd()["ball"][:, 0])
    rows = []  # (seconds, p_team, frames_right, ball_seen, found, alive_pred)
    for k, (_, first, last, labels) in enumerate(blocks):
        p_dead, p_home = probabilities(fit(X, blocks[:k] + blocks[k + 1:]), X, n_frames)
        pred = decode(p_dead, p_home, n_frames, V1_PARAMS)[first:last + 1]
        pred_spells = spells(pred)
        for a, b, label in spells(labels):
            if label not in "HA":
                continue
            ph = p_home[first + a:first + b]
            p_team = ph if label == "H" else 1 - ph
            best = 0.0
            for c, d, plabel in pred_spells:
                if plabel == label and c < b and d > a:
                    best = max(best, (min(b, d) - max(a, c)) / (max(b, d) - min(a, c)))
            rows.append(((b - a) / FPS, p_team.mean(), (p_team > 0.5).mean(),
                         ball[first + a:first + b].mean(), best >= 0.5,
                         (1 - p_dead[first + a:first + b]).mean()))
    rows = np.array(rows, dtype=float)
    print("{} true possession spells in {} blocks".format(len(rows), len(blocks)))
    print("{:>10} {:>5} {:>8} {:>13} {:>10} {:>7} {:>9}".format(
        "length", "n", "p_team", "frames_right", "ball_seen", "found", "P(alive)"))
    for lo, hi in zip(BUCKETS_S[:-1], BUCKETS_S[1:]):
        m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
        if not m.any():
            continue
        r = rows[m].mean(axis=0)
        print("{:>4.0f}-{:<5} {:>5} {:>8.2f} {:>13.0%} {:>10.0%} {:>7.0%} {:>9.2f}".format(
            lo, "inf s" if hi == np.inf else "{:.0f} s".format(hi), int(m.sum()), r[1], r[2], r[3],
            r[4], r[5]))


if __name__ == "__main__":
    main()
