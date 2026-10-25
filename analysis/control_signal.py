"""Does ball control from the video's player boxes mark possessions, short ones included?

    docker compose run --rm gpu python -m enrich.detect_players --frames labelled
    docker compose run --rm dev python -m possession.prelabel          # for prelabel_oof.csv
    docker compose run --rm dev python analysis/control_signal.py

Over every finished hand-labelled block (on the GT's timing), prints aggregates only:
  - per possession length: share of its frames with a controlling team, and how often that team is
    the labelled one
  - control switches (a run of at least MIN frames by the other team) as a change detector: recall
    within +-1 s, the same at 3 s off the change (chance), and precision
  - possessions segmented by control on the out-of-fold pre-labeller's alive frames, by spellF1
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import control as pc  # noqa: E402
from possession.io import AWAY, FPS, HOME  # noqa: E402
from possession.prelabel import OOF_OUT, pm, training_blocks  # noqa: E402

BUCKETS_S = (0, 2, 4, 8, 15, np.inf)
TOL = FPS
CHANCE_S = 3
MIN_FRAMES = (1, 3, 6)
HOLD_S = (0.0, 1.0, 2.0)
MAX_CARRY_S = 15   # a control episode names the team for at most this long without another one


def code(label):
    return {"H": HOME, "A": AWAY}.get(label, -1)


def spells(lab):
    edges = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    return list(zip(np.concatenate([[0], edges]), np.concatenate([edges, [len(lab)]])))


def switches(ep):
    """Frames where control passes to the other team than the last one (NONE gaps skipped)."""
    out = np.full_like(ep, pc.NONE)
    idx = np.flatnonzero(ep >= 0)
    at = idx[1:][ep[idx[1:]] != ep[idx[:-1]]]
    out[at] = ep[at]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--contact", type=float, default=pc.CONTACT,
                        help="ball within this many box heights of a player box is contact")
    args = parser.parse_args()
    ctrl, stats = pc.control(contact=args.contact)
    print("{} frames with player detections: a matched ball box in {:.0%}, team assigned to {:.0%} "
          "of player boxes".format(stats["frames"], stats["with_ball"], stats["boxes_assigned"]))
    blocks = [(n, f, l, np.array([code(x) for x in lab]), lab) for n, f, l, lab in training_blocks()]

    print("\ncontrol inside true possessions (per frame):")
    print("{:>9} {:>4} {:>9} {:>10} {:>10}".format("length", "n", "control", "contested", "right"))
    rows = []
    for _, first, last, lab, _ in blocks:
        for a, b in spells(lab):
            if lab[a] < 0:
                continue
            c = ctrl[first + a:first + b]
            known = (c == HOME) | (c == AWAY)
            rows.append(((b - a) / FPS, known.mean(), (c == pc.CONTESTED).mean(),
                         (c[known] == lab[a]).mean() if known.any() else np.nan))
    rows = np.array(rows)
    for lo, hi in zip(BUCKETS_S[:-1], BUCKETS_S[1:]):
        m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
        print("{:>3.0f}-{:<5} {:>4} {:>9.0%} {:>10.0%} {:>10.0%}".format(
            lo, "{:.0f} s".format(hi) if hi < np.inf else "inf", int(m.sum()), rows[m, 1].mean(),
            rows[m, 2].mean(),
            np.nanmean(rows[m, 3]) if np.isfinite(rows[m, 3]).any() else np.nan))

    print("\ncontrol switches as a change detector (+-{} f):".format(TOL))
    for min_frames in MIN_FRAMES:
        sw = switches(pc.episodes(ctrl, min_frames))
        hits, n, chance, chance_n, true_sw, all_sw = 0, 0, 0, 0, 0, 0
        short_hits, short_n = 0, 0
        for _, first, last, lab, _ in blocks:
            s = sw[first:last + 1]
            sp = spells(lab)
            changes = [(a, lab[a], (b - a) / FPS) for (a, b), (pa, _) in zip(sp[1:], sp[:-1])
                       if lab[a] >= 0 and lab[pa] >= 0]
            for e, team, length in changes:
                hit = (s[max(e - TOL, 0):e + TOL + 1] == team).any()
                hits += hit
                n += 1
                if length < 4:
                    short_hits += hit
                    short_n += 1
                for off in (-CHANCE_S * FPS, CHANCE_S * FPS):
                    lo = min(max(e + off - TOL, 0), len(s) - 1)
                    chance += (s[lo:lo + 2 * TOL + 1] == team).any()
                    chance_n += 1
            for i in np.flatnonzero(s >= 0):
                all_sw += 1
                true_sw += any(abs(i - e) <= TOL and t == s[i] for e, t, _ in changes)
        print("  runs >= {} f: {:.1f} switches/min; recall {:.0%} (under 4 s: {:.0%}), chance {:.0%}, "
              "precision {:.0%}".format(
                  min_frames, all_sw / (sum(l - f + 1 for _, f, l, _, _ in blocks) / FPS / 60),
                  hits / n, short_hits / max(short_n, 1), chance / chance_n, true_sw / max(all_sw, 1)))

    oof = pd.read_csv(OOF_OUT)["label"].astype(str).values
    print("\nsegmenting by control on the pre-labeller's alive frames (spellF1, possessions started; "
          "truth {}):".format(sum(len([1 for (a, b) in spells(lab)[1:] if lab[a] >= 0])
                                  for _, _, _, lab, _ in blocks)))
    variants = {"pre-labeller (oof)": oof}
    for min_frames in MIN_FRAMES:
        ep = pc.episodes(ctrl, min_frames)
        team = np.full(len(oof), pc.NONE)
        idx = np.flatnonzero(ep >= 0)
        for i, j in zip(idx, list(idx[1:]) + [len(oof)]):
            team[i:min(j, i + MAX_CARRY_S * FPS)] = ep[i]
        for hold in HOLD_S:
            t = team.copy()
            if hold:
                edges = np.flatnonzero(t[1:] != t[:-1]) + 1
                for a, b in zip(edges, list(edges[1:]) + [len(t)]):
                    if b - a < hold * FPS:
                        t[a:b] = t[a - 1]
            labels = oof.copy()
            alive = (labels != "D") & (labels != "X") & (t >= 0)
            labels[alive & (t == HOME)] = "H"
            labels[alive & (t == AWAY)] = "A"
            variants["control runs>={}f hold {:.0f}s".format(min_frames, hold)] = labels
    for name, pred in variants.items():
        spell, started = [], 0
        for _, first, last, _, truth in blocks:
            p = pred[first:last + 1]
            spell.append(pm.evaluate(pm.label_codes(truth), pm.label_codes(p),
                                     bootstrap_n=0)["spell_f1_mean"])
            started += len([1 for a, b in zip(p[:-1], p[1:]) if a != b and b in "HA"])
        print("  {:<28} {:.3f}   {}".format(name, np.mean(spell), started))


if __name__ == "__main__":
    main()
