"""Sensitivity of v1 to its main knobs. Prints aggregates only.

The public 5-minute window is reported for reference; with 17 possessions it cannot rank variants
(see EVALUATION.md), so read the full-match columns (D share, possession count) as the signal.

    docker compose run --rm dev python analysis/sweep_v1.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

from possession import features as feats  # noqa: E402
from possession.io import FPS, GAME, GAMES, graded_blocks, load_atd, load_homography  # noqa: E402
from possession.models import V1_PARAMS, v1  # noqa: E402

GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")

GRID = [
    {},
    {"cost_team_switch": 15.0},
    {"cost_team_switch": 40.0},
    {"dead_bias": -3.0},
    {"dead_bias": -1.0},
    {"cost_to_dead": 8.0, "cost_from_dead": 8.0},
    {"cost_to_dead": 30.0, "cost_from_dead": 30.0},
]


def full_match_stats(labels):
    spells, dead, total = [], 0, 0
    for _, first, last in graded_blocks():
        w = labels[first:last + 1]
        dead += int((w == "D").sum())
        total += len(w)
        bounds = np.concatenate(([0], np.flatnonzero(w[1:] != w[:-1]) + 1, [len(w)]))
        spells += [b - a for a, b in zip(bounds[:-1], bounds[1:]) if w[a] in "HA"]
    return dead / total, len(spells), np.mean(spells) / FPS


def main():
    features = feats.compute(load_atd(), load_homography())
    gt = pm.label_codes(pm.read_label_csv(GT))
    n = GAMES[GAME]["n_video_frames"]
    print("{:<40} {:>7} {:>6} {:>6} | {:>8} {:>8}".format(
        "override", "D_full", "poss", "dur_s", "spellF1*", "macroF1*"))
    for override in GRID:
        labels = v1(features, n, {**V1_PARAMS, **override})
        d_share, n_poss, dur = full_match_stats(labels)
        result = pm.evaluate(gt, pm.label_codes(labels))
        print("{:<40} {:>6.1%} {:>6} {:>6.1f} | {:>8.3f} {:>8.3f}".format(
            str(override or "default"), d_share, n_poss, dur,
            result["spell_f1_mean"], result["macro_f1"]))
    print("* public 5-minute window only: reference, not a ranking")


if __name__ == "__main__":
    main()
