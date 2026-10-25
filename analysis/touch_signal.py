"""Is there a touch signal in the tracking that marks possession changes, short ones included?

    docker compose run --rm dev python analysis/touch_signal.py

A touch is a frame where the ball turns by more than ANGLE degrees or changes speed by more than
DSPEED m/s between the 5 frames before and after it, with a player within RADIUS m; its team
is the nearest player's. Run over the public GT and every finished hand-labelled block (moved onto
the GT's timing), it prints aggregates only:
  - recall: share of true possession changes (H<->A) with a touch by the new team within +-TOL
    frames, by the length of the possession that starts there
  - precision: share of rival touches (by the team not in possession just before) that are such a
    change; the rest are deflections, tackles lost, or tracking noise
If both are high for short possessions, a touch feature should recover them; if recall is high but
precision low, rival touches are everywhere and something must tell real changes from the rest.

    docker compose run --rm dev python analysis/touch_signal.py --ball detected

does the same on the video detector's ball, which is not smoothed: the candidate nearest to the ATD
ball where there is one, merge.py's detector ball in the ATD's gaps.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.gt_changes import per_frame  # noqa: E402
from possession import touches as tf  # noqa: E402
from possession.io import AWAY, FPS, HOME  # noqa: E402
from possession.prelabel import training_blocks  # noqa: E402

TOL = FPS                        # a touch this close to a change counts for it
SETTINGS = [                     # (angle deg, dspeed m/s, radius m)
    (25, 3.0, 1.5), (35, 3.0, 1.5), (50, 4.0, 1.5), (35, 3.0, 2.5), (35, 3.0, 4.0),
]
BUCKETS_S = (0, 2, 4, 8, 15, np.inf)
CONFIRM_S = tf.CONFIRM_S
CHANCE_S = 3                    # the recall test repeated this far from each change: chance level


def code(label):
    return {"H": HOME, "A": AWAY}.get(label, -1)


def with_ball(f, ball):
    """per_frame()'s dict with another ball: nearest-player distances recomputed."""
    return dict(f, ball=ball, d=tf.team_distances(ball), visible=np.isfinite(ball[:, 0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ball", choices=["tracking", "detected", "hybrid"], default="tracking",
                        help="detected: the detector's unsmoothed ball (needs detect_ball --mode "
                             "visible and missing, and merge); hybrid: the detector's where it has "
                             "one, the tracking's elsewhere")
    args = parser.parse_args()
    f = per_frame()
    if args.ball != "tracking":
        tracking = f["ball"]
        detected = tf.detected_ball(f["ball"])
        both = np.isfinite(tracking[:, 0])
        matched = both & np.isfinite(detected[:, 0])
        print("detector ball on {:.0%} of the frames with a tracking ball (within {} m of it), "
              "median {:.2f} m apart".format(matched.sum() / both.sum(), tf.MATCH_M,
                                              np.median(np.linalg.norm(detected[matched] - tracking[matched], axis=1))))
        if args.ball == "hybrid":
            detected = np.where(np.isfinite(detected[:, :1]), detected, tracking)
        f = with_ball(f, detected)
    blocks = training_blocks()
    minutes = sum(last - first + 1 for _, first, last, _ in blocks) / FPS / 60
    print("{} blocks, {:.0f} min".format(len(blocks), minutes))
    for angle, dspeed, radius in SETTINGS:
        team = tf.touches(f["ball"], f["d"], angle, dspeed, radius)
        found = {b: [0, 0] for b in BUCKETS_S[:-1]}
        rival, rival_true, own = 0, 0, 0
        chance = [0, 0]
        confirmed, confirmed_true, missed_unconfirmed = 0, 0, 0
        for _, first, last, labels in blocks:
            lab = np.array([code(x) for x in labels])
            edges = np.flatnonzero(lab[1:] != lab[:-1]) + 1
            ends = np.concatenate([edges[1:], [len(lab)]])
            changes = []
            for e, end in zip(edges, ends):
                if lab[e - 1] >= 0 and lab[e] >= 0:  # H<->A
                    changes.append((e, lab[e]))
                    length = (end - e) / FPS
                    b = max(x for x in BUCKETS_S[:-1] if x <= length)
                    window = team[first + max(e - TOL, 0):first + min(e + TOL + 1, len(lab))]
                    found[b][0] += bool((window == lab[e]).any())
                    found[b][1] += 1
                    # chance: the same test CHANCE_S away from the change, either side
                    for s in (-CHANCE_S * FPS, CHANCE_S * FPS):
                        lo = first + min(max(e + s - TOL, 0), len(lab) - 1)
                        chance[0] += bool((team[lo:lo + 2 * TOL + 1] == lab[e]).any())
                        chance[1] += 1
            for i in np.flatnonzero(team[first:last + 1] >= 0):
                prev = lab[max(i - 15, 0)]
                if prev < 0:
                    continue
                if team[first + i] == prev:
                    own += 1
                    continue
                rival += 1
                is_change = any(abs(i - e) <= TOL and t == team[first + i] for e, t in changes)
                rival_true += is_change
                # confirmed: the same team touches again within CONFIRM_S, before the other one does
                later = team[first + i + 1:first + min(i + 1 + CONFIRM_S * FPS, last + 1 - first)]
                later = later[later >= 0]
                if len(later) and later[0] == team[first + i]:
                    confirmed += 1
                    confirmed_true += is_change
                elif is_change:
                    missed_unconfirmed += 1
        n_changes = sum(v[1] for v in found.values())
        if (angle, dspeed, radius) == SETTINGS[0]:
            seen = [f["visible"][first + max(e - TOL, 0):first + e + TOL + 1].mean()
                    for _, first, last, labels in blocks
                    for e in np.flatnonzero(np.array([code(x) for x in labels])[1:]
                                            != np.array([code(x) for x in labels])[:-1]) + 1
                    if code(labels[e - 1]) >= 0 and code(labels[e]) >= 0]
            print("ball visible within +-{} f of a change: mean {:.0%}, "
                  "under half the window for {:.0%} of changes".format(
                      TOL, np.mean(seen), np.mean(np.array(seen) < 0.5)))
        print("\nangle > {} deg or |dv| > {} m/s, player within {} m: {:.1f} touches/min "
              "({:.1f} by the team in possession, {:.1f} by its rival)".format(
                  angle, dspeed, radius, (own + rival) / minutes, own / minutes, rival / minutes))
        print("  recall, by length of the possession that starts at the change "
              "({} changes):".format(n_changes))
        print("    " + "   ".join("{:.0f}-{}s {:.0%} (n={})".format(
            b, "{:.0f}".format(hi) if hi < np.inf else "inf", v[0] / max(v[1], 1), v[1])
            for (b, v), hi in zip(found.items(), BUCKETS_S[1:])))
        print("  chance: {:.0%} with the window moved {} s off the change; all changes: {:.0%}".format(
            chance[0] / max(chance[1], 1), CHANCE_S,
            sum(v[0] for v in found.values()) / max(n_changes, 1)))
        print("  precision: {:.0%} of rival touches are a change to that team ({} of {})".format(
            rival_true / max(rival, 1), rival_true, rival))
        print("  confirmed by a 2nd touch of that team within {} s: precision {:.0%} ({} of {}), "
              "keeps {:.0%} of the true ones".format(
                  CONFIRM_S, confirmed_true / max(confirmed, 1), confirmed_true, confirmed,
                  confirmed_true / max(confirmed_true + missed_unconfirmed, 1)))


if __name__ == "__main__":
    main()
