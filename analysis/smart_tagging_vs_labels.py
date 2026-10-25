"""How the Smart Tagging lines up with the GT and the hand labels: team by code, set-piece timing.

    docker compose run --rm dev python analysis/smart_tagging_vs_labels.py

Over the public GT and every finished hand-labelled block (on the GT's timing), aggregates only:
  - per code: share of the labelled alive frames it covers where its implied team holds the ball
    (attacking codes: the tagged team; defending codes: the other one)
  - per set-piece type: where the restart (a change from D to H/A in the labels) falls relative to
    the instance's start, and whether the team after it is the tagged one
  - the same for GOALS against the next D stretch
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession.io import AWAY, FPS, HOME, load_smart_tagging  # noqa: E402
from possession.prelabel import training_blocks  # noqa: E402

TEAM = {"barcelona": HOME, "atletic club": AWAY}   # BAR-ATH: Barcelona at home
DEFENDING = ("DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD",
             "DEFENDING IN ATTACKING THIRD", "DEFENSIVE TRANSITION")
SEARCH_S = 40


def main():
    tags = load_smart_tagging()
    tags["team"] = tags["Team"].fillna("N/A").str.strip().str.lower().map(TEAM).fillna(-1).astype(int)
    blocks = training_blocks()
    n = max(l for _, _, l, _ in blocks) + 1
    label = np.full(n, "X", dtype="<U1")
    for _, first, last, lab in blocks:
        label[first:last + 1] = lab
    alive = (label == "H") | (label == "A")
    code = np.where(label == "H", HOME, np.where(label == "A", AWAY, -1))
    print("{} labelled frames, {} alive".format(int((label != "X").sum()), int(alive.sum())))

    print("\nteam implied by each code, on labelled alive frames it covers:")
    print("  {:<32} {:>6} {:>10} {:>8}".format("code", "inst", "frames", "right"))
    for c, g in sorted(tags.groupby("code"), key=lambda kv: -len(kv[1])):
        right, total, inst = 0, 0, 0
        for _, t in g.iterrows():
            if t["team"] < 0:
                continue
            implied = t["team"] if c not in DEFENDING else (AWAY if t["team"] == HOME else HOME)
            f0, f1 = max(t["f0"], 0), min(t["f1"], n)
            m = alive[f0:f1]
            if m.any():
                inst += 1
                total += int(m.sum())
                right += int((code[f0:f1][m] == implied).sum())
        if total:
            print("  {:<32} {:>6} {:>10} {:>7.0%}".format(c, inst, total, right / total))

    # restarts in the labels
    restarts = [(i, code[i]) for i in np.flatnonzero((label[1:] != label[:-1]) & (label[:-1] == "D")) + 1
                if code[i] >= 0]
    restart_frames = np.array([r[0] for r in restarts])
    sp = tags[tags["code"] == "SET PIECES"]
    print("\nset pieces vs the nearest labelled restart (D -> H/A) within {} s of their start:".format(SEARCH_S))
    print("  {:<24} {:>4} {:>8} {:>16} {:>13} {:>12}".format(
        "type", "n", "matched", "offset median s", "offset IQR s", "team right"))
    for typ, g in sp.groupby(sp["Type"].fillna("?")):
        offs, team_ok, inside = [], [], 0
        for _, t in g.iterrows():
            if not (label[max(t["f0"], 0):min(t["f1"], n)] != "X").any():
                continue
            inside += 1
            if not len(restart_frames):
                continue
            j = int(np.argmin(np.abs(restart_frames - t["f0"])))
            off = (restart_frames[j] - t["f0"]) / FPS
            if abs(off) <= SEARCH_S:
                offs.append(off)
                if t["team"] >= 0:
                    team_ok.append(restarts[j][1] == t["team"])
        if inside:
            q = np.percentile(offs, [25, 50, 75]) if offs else [np.nan] * 3
            print("  {:<24} {:>4} {:>8} {:>16.1f} {:>6.1f}..{:<5.1f} {:>12}".format(
                typ, inside, len(offs), q[1], q[0], q[2],
                "{:.0%} of {}".format(np.mean(team_ok), len(team_ok)) if team_ok else "-"))
    print("  labelled restarts: {}; within {} s of some set piece's start: {:.0%}".format(
        len(restart_frames), SEARCH_S, np.mean([np.min(np.abs(sp["f0"].values - r)) <= SEARCH_S * FPS
                                                 for r in restart_frames]) if len(restart_frames) else 0))

    goals = tags[tags["code"] == "GOALS"]
    for _, t in goals.iterrows():
        f0, f1 = max(t["f0"], 0), min(t["f1"], n)
        if (label[f0:f1] != "X").any():
            print("  GOALS instance: {:.0%} of its labelled frames are D".format(
                np.mean(label[f0:f1][label[f0:f1] != "X"] == "D")))


if __name__ == "__main__":
    main()
