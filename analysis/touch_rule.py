"""Segmenting possessions by confirmed touches instead of a per-frame classifier.

    docker compose run --rm dev python -m possession.prelabel        # writes prelabel_oof.csv
    docker compose run --rm dev python analysis/touch_rule.py

Keeps the out-of-fold pre-labeller's dead-ball frames and replaces its team on alive frames with
the team of the last confirmed touch on the detector's ball (possession.touches), falling back to
the pre-labeller where there is none yet. Scores both against every labelled block. Aggregates only.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import features as feats  # noqa: E402
from possession import touches as tf  # noqa: E402
from possession.io import AWAY, FPS, HOME, PITCH, ROOT, load_atd  # noqa: E402
from possession.prelabel import OOF_OUT, pm, training_blocks  # noqa: E402

MIN_RUN_S = (0.0, 1.0, 2.0)  # ignore a switch whose new team keeps it less than this


def main():
    oof = pd.read_csv(OOF_OUT)["label"].astype(str).values
    tracking = feats.clean_ball(load_atd()["ball"].astype(np.float64) * np.array(PITCH))
    ball = tf.detected_ball(tracking)
    c = tf.confirmed(tf.touches(ball, tf.team_distances(ball)))
    idx = np.flatnonzero(c >= 0)
    blocks = training_blocks()
    print("{} blocks; spellF1 / possessions per variant (truth has {})".format(
        len(blocks), sum(len([1 for a, b in zip(l[:-1], l[1:]) if a != b and b in "HA"])
                         for _, _, _, l in blocks)))
    variants = {"pre-labeller (oof)": oof}
    for min_run in MIN_RUN_S:
        team = np.full(len(oof), -1)
        for i, j in zip(idx, list(idx[1:]) + [len(oof)]):
            team[i:j] = c[i]
        # drop switches that do not hold for min_run: carry the previous team over them
        if min_run:
            edges = np.flatnonzero(team[1:] != team[:-1]) + 1
            for a, b in zip(edges, list(edges[1:]) + [len(team)]):
                if b - a < min_run * FPS and a > 0:
                    team[a:b] = team[a - 1]
        labels = oof.copy()
        alive = (labels != "D") & (labels != "X") & (team >= 0)
        labels[alive & (team == HOME)] = "H"
        labels[alive & (team == AWAY)] = "A"
        variants["touches, min run {:.0f} s".format(min_run)] = labels
    for name, pred in variants.items():
        spell, n_pred = [], 0
        for _, first, last, truth in blocks:
            p = pred[first:last + 1]
            spell.append(pm.evaluate(pm.label_codes(truth), pm.label_codes(p),
                                     bootstrap_n=0)["spell_f1_mean"])
            n_pred += len([1 for a, b in zip(p[:-1], p[1:]) if a != b and b in "HA"])
        print("  {:<22} spellF1 {:.3f}   possessions started {}".format(name, np.mean(spell), n_pred))


if __name__ == "__main__":
    main()
