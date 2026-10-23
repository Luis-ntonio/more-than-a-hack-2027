"""How early or late each annotator marks changes, measured on the public block against the GT.

    docker compose run --rm dev python -m labeler.timing                 # every annotator
    docker compose run --rm dev python -m labeler.timing --annotator luis

Each change is one of three kinds: "dead" (to D), "restart" (D to H/A) and "possession" (H <-> A).
An annotator's change is matched one-to-one to the nearest GT change of the same transition within
10 s. The median offset per kind is the shift labeler.export --shift auto and possession.prelabel
apply to that annotator's blocks, so hand labels end up on the GT's timing. Blocks started from a
model's prediction are not shifted: their untouched changes are already on the model's timing. A
public block that was itself started from a model does not calibrate anything. Prints offsets and
counts only.
"""
import argparse
import glob
import json
import os

import numpy as np

from labeler.server import (LABELS, LABELS_DIR, build_blocks, changes_of, dense, label_path, pm,
                            _public_gt)
from possession.io import FPS

KINDS = ("dead", "restart", "possession")
MAX_GAP = 10 * FPS
MIN_MATCHES = 3  # fewer matches than this for a kind and its shift is 0


def kind_of(previous, label):
    if label == "D":
        return "dead"
    return "restart" if previous == "D" else "possession"


def transitions(labels, first):
    """Dense labels from frame `first` -> [(frame, previous, new)], X frames dropped."""
    points = changes_of(labels, first)
    return [(frame, prev, new) for (_, prev), (frame, new) in zip(points, points[1:])]


def match(truth, mine):
    """One-to-one greedy match by distance, same transition only. -> [(kind, offset)], leftovers."""
    pairs = sorted((abs(m[0] - t[0]), i, j) for i, t in enumerate(truth) for j, m in enumerate(mine)
                   if t[1:] == m[1:] and abs(m[0] - t[0]) <= MAX_GAP)
    used_t, used_m, matched = set(), set(), []
    for _, i, j in pairs:
        if i not in used_t and j not in used_m:
            used_t.add(i)
            used_m.add(j)
            matched.append((kind_of(*truth[i][1:]), mine[j][0] - truth[i][0]))
    missed = [truth[i] for i in range(len(truth)) if i not in used_t]
    extra = [mine[j] for j in range(len(mine)) if j not in used_m]
    return matched, missed, extra


def public_comparison(annotator):
    """(matched, missed, extra, agreement, seeded) over the public block(s), or None."""
    gt, public_range = _public_gt()
    if gt is None:
        return None
    gt_labels = np.array(pm.LABELS)[gt]
    matched, missed, extra, agree, n, found, seeded = [], [], [], 0, 0, False, False
    for block in build_blocks(public_range):
        path = label_path(block["id"], annotator)
        if not block["public"] or not os.path.isfile(path):
            continue
        found = True
        with open(path) as handle:
            doc = json.load(handle)
        seeded = seeded or bool(doc.get("seeded_from"))
        first, last = block["first"], block["last"]
        truth = gt_labels[first:last + 1]
        graded = np.isin(truth, LABELS)
        mine = np.where(graded, dense(doc["changes"], first, last), "X")
        m, mi, ex = match(transitions(np.where(graded, truth, "X"), first), transitions(mine, first))
        matched += m
        missed += mi
        extra += ex
        agree += int((mine[graded] == truth[graded]).sum())
        n += int(graded.sum())
    return (matched, missed, extra, agree / max(n, 1), seeded) if found else None


def calibrate(annotator):
    """{kind: frames to add to that annotator's changes}, or None without a from-scratch public block."""
    comparison = public_comparison(annotator)
    if comparison is None or comparison[4]:
        return None
    shifts = {}
    for kind in KINDS:
        offsets = [o for k, o in comparison[0] if k == kind]
        shifts[kind] = -int(round(np.median(offsets))) if len(offsets) >= MIN_MATCHES else 0
    return shifts


def shift_changes(changes, shifts, first, last):
    """Move each change by its kind's shift, keeping them inside the block and strictly increasing.

    The first entry is the block's starting state, not a change, and stays where it is. A change
    pushed onto or past the next one stops a frame short of it, so no state is lost; it is only
    dropped when there is no room left at all.
    """
    if not changes:
        return changes
    moved = [min(max(frame + shifts[kind_of(prev[1], label)], first + 1), last)
             for prev, (frame, label) in zip(changes, changes[1:])]
    for i in range(len(moved) - 2, -1, -1):
        moved[i] = min(moved[i], moved[i + 1] - 1)
    out = [list(changes[0])]
    for frame, (_, label) in zip(moved, changes[1:]):
        if frame > out[-1][0]:
            out.append([frame, label])
    return out


def shifted_changes(doc, cache):
    """A saved block's changes on the GT's timing. cache maps annotator -> calibrate() result."""
    if doc.get("seeded_from"):
        return doc["changes"]
    if doc["annotator"] not in cache:
        cache[doc["annotator"]] = calibrate(doc["annotator"])
    shifts = cache[doc["annotator"]]
    return shift_changes(doc["changes"], shifts, doc["first"], doc["last"]) if shifts else doc["changes"]


def annotators():
    names = set()
    for path in glob.glob(os.path.join(LABELS_DIR, "*__*.json")):
        names.add(os.path.basename(path).split("__", 1)[1][:-len(".json")])
    return sorted(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annotator", default="all")
    args = parser.parse_args()

    names = annotators() if args.annotator == "all" else [args.annotator]
    for name in names:
        comparison = public_comparison(name)
        if comparison is None:
            if args.annotator != "all":
                print("{}: no labels on the public block".format(name))
            continue
        matched, missed, extra, agreement, seeded = comparison
        print("== {}   frame agreement {:.1%}   matched {}   missed {}   extra {}{}".format(
            name, agreement, len(matched), len(missed), len(extra),
            "   (started from a model: not used to calibrate)" if seeded else ""))
        if missed:
            print("   missed:  " + ", ".join("{}>{}".format(p, n) for _, p, n in missed))
        if extra:
            print("   extra:   " + ", ".join("{}>{}".format(p, n) for _, p, n in extra))
        print("   {:<11} {:>3} {:>8} {:>8} {:>6} {:>6}  {}".format(
            "kind", "n", "median", "mean", "early", "late", "shift (frames added on export)"))
        shifts = calibrate(name)
        for kind in KINDS:
            off = np.array([o for k, o in matched if k == kind])
            if not len(off):
                print("   {:<11} {:>3}".format(kind, 0))
                continue
            print("   {:<11} {:>3} {:>+7.1f}f {:>+7.1f}f {:>6} {:>6}  {:+d}{}".format(
                kind, len(off), np.median(off), off.mean(), int((off < 0).sum()),
                int((off > 0).sum()), shifts[kind],
                "" if len(off) >= MIN_MATCHES else "  (too few to calibrate)"))
        off = np.array([o for _, o in matched])
        if len(off):
            print("   all: mean |offset| {:.1f} f ({:.2f} s), signed mean {:+.1f} f".format(
                np.abs(off).mean(), np.abs(off).mean() / FPS, off.mean()))


if __name__ == "__main__":
    main()
