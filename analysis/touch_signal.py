"""Is there a touch signal in the tracking that marks possession changes, short ones included?

    docker compose run --rm dev python analysis/touch_signal.py

A touch is a frame where the ball turns by more than ANGLE degrees or changes speed by more than
DSPEED m/s between the WINDOW frames before and after it, with a player within RADIUS m; its team
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
from enrich.geometry import pixels_to_pitch  # noqa: E402
from labeler.overlay import ENRICHED_DIR  # noqa: E402
from possession.io import AWAY, FPS, GAME, HOME, PITCH, load_atd, load_homography  # noqa: E402
from possession.prelabel import training_blocks  # noqa: E402

WINDOW = 5                       # frames each side for the before / after velocity
TOL = FPS                        # a touch this close to a change counts for it
SETTINGS = [                     # (angle deg, dspeed m/s, radius m)
    (25, 3.0, 1.5), (35, 3.0, 1.5), (50, 4.0, 1.5), (35, 3.0, 2.5), (35, 3.0, 4.0),
]
BUCKETS_S = (0, 2, 4, 8, 15, np.inf)
CONFIRM_S = 3                    # a rival touch followed by another of its team this soon
CHANCE_S = 3                    # the recall test repeated this far from each change: chance level
MATCH_M = 3.0                   # --ball detected: a candidate this close to the ATD ball is it


def touches(f, angle, dspeed, radius):
    """Per-frame touch team (HOME, AWAY, or -1), at most one touch per WINDOW frames."""
    ball = f["ball"]
    n = len(ball)
    before = np.full((n, 2), np.nan)
    after = np.full((n, 2), np.nan)
    before[WINDOW:] = (ball[WINDOW:] - ball[:-WINDOW]) * FPS / WINDOW
    after[:-WINDOW] = (ball[WINDOW:] - ball[:-WINDOW]) * FPS / WINDOW
    sb, sa = np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1)
    cos = np.einsum("ij,ij->i", before, after) / np.maximum(sb * sa, 1e-9)
    turn = (sb > 1.0) & (sa > 1.0) & (cos < np.cos(np.radians(angle)))
    kick = np.abs(sa - sb) > dspeed
    d_home, d_away = f["d"][HOME], f["d"][AWAY]
    nearest = np.fmin(d_home, d_away)
    event = (turn | kick) & (nearest < radius)
    team = np.where(event, np.where(d_home <= d_away, HOME, AWAY), -1)
    # keep the first frame of each run of events
    keep = event & ~np.concatenate(([False], event[:-1]))
    last = -WINDOW
    for i in np.flatnonzero(keep):
        if i - last < WINDOW:
            keep[i] = False
        else:
            last = i
    return np.where(keep, team, -1)


def code(label):
    return {"H": HOME, "A": AWAY}.get(label, -1)


def detected_ball(f):
    """The detector's ball in metres: where the ATD has a ball, the candidate nearest to it (within
    MATCH_M); where it has none, the ball merge.py took from the detector. NaN elsewhere."""
    scale = np.array(PITCH, dtype=np.float64)
    homography = load_homography()
    ball = np.full_like(f["ball"], np.nan)
    z = np.load(os.path.join(ENRICHED_DIR, "{}_ball_detections_visible.npz".format(GAME)))
    for frame, cands in zip(z["frames"], z["det"]):
        cands = cands[np.isfinite(cands[:, 2])]
        if not len(cands) or not np.isfinite(f["ball"][frame, 0]):
            continue
        x, y = pixels_to_pitch(homography[frame], cands[:, 0], cands[:, 1])
        xy = np.stack([x, y], axis=1) * scale
        dist = np.linalg.norm(xy - f["ball"][frame], axis=1)
        best = int(np.nanargmin(dist)) if np.isfinite(dist).any() else None
        if best is not None and dist[best] < MATCH_M:
            ball[frame] = xy[best]
    rich = load_atd(enriched=True)
    from_detector = rich["ball_source"] == 2
    ball[from_detector] = rich["ball"][from_detector].astype(np.float64) * scale
    return ball


def with_ball(f, ball):
    """per_frame()'s dict with another ball: nearest-player distances recomputed."""
    atd = load_atd()
    players = atd["players"][:, :, :2].astype(np.float64) * np.array(PITCH, dtype=np.float64)
    dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    d = {}
    for team in (HOME, AWAY):
        d[team] = dist[:, atd["team"] == team].min(axis=1)
        d[team][~np.isfinite(d[team])] = np.nan
    return dict(f, ball=ball, d=d, visible=np.isfinite(ball[:, 0]))


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
        detected = detected_ball(f)
        both = np.isfinite(tracking[:, 0])
        matched = both & np.isfinite(detected[:, 0])
        print("detector ball on {:.0%} of the frames with a tracking ball (within {} m of it), "
              "median {:.2f} m apart".format(matched.sum() / both.sum(), MATCH_M,
                                              np.median(np.linalg.norm(detected[matched] - tracking[matched], axis=1))))
        if args.ball == "hybrid":
            detected = np.where(np.isfinite(detected[:, :1]), detected, tracking)
        f = with_ball(f, detected)
    blocks = training_blocks()
    minutes = sum(last - first + 1 for _, first, last, _ in blocks) / FPS / 60
    print("{} blocks, {:.0f} min".format(len(blocks), minutes))
    for angle, dspeed, radius in SETTINGS:
        team = touches(f, angle, dspeed, radius)
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
