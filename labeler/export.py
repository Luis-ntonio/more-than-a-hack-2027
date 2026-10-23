"""Merge hand-labelled blocks into one full-length frame,label CSV the evaluator reads as GT.

    docker compose run --rm dev python -m labeler.export --annotator luis
    docker compose run --rm dev python evaluation/run_evaluation.py --gt labels/validation_luis.csv

Frames nobody labelled are X, which the evaluator never scores. Blocks not marked done are left
out unless --include-unfinished is given. Blocks started from a model's prediction ("Partir de la
predicción") are anchored to that model; --exclude-seeded keeps them out of a validation set.
--shift auto moves each annotator's changes onto the GT's timing, by the per-kind median offset
labeler.timing measures on their public block. Annotators without one, and blocks started from a
model, are exported unshifted.
Prints counts only.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from labeler.server import LABELS_DIR, build_blocks, dense, _public_gt
from labeler.timing import shifted_changes
from games import GAME, GAMES
from possession.io import FPS


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annotator", required=True,
                        help="whose labels to export; 'all' takes, per block, the most recent file")
    parser.add_argument("--include-unfinished", action="store_true")
    parser.add_argument("--include-public", action="store_true",
                        help="also export the public block (it already has official GT)")
    parser.add_argument("--exclude-seeded", action="store_true",
                        help="leave out blocks that started from a model's prediction")
    parser.add_argument("--shift", choices=("none", "auto"), default="none",
                        help="auto: move changes by each annotator's offset to the GT (labeler.timing)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    blocks = build_blocks(_public_gt()[1])
    labels = np.full(GAMES[GAME]["n_video_frames"], "X", dtype="<U1")
    used, skipped, shifts = [], [], {}
    for block in blocks:
        if block["public"] and not args.include_public:
            continue
        docs = []
        for name in os.listdir(LABELS_DIR) if os.path.isdir(LABELS_DIR) else []:
            prefix = block["id"] + "__"
            if name.startswith(prefix) and name.endswith(".json"):
                annotator = name[len(prefix):-len(".json")]
                if args.annotator in ("all", annotator):
                    with open(os.path.join(LABELS_DIR, name)) as handle:
                        docs.append(json.load(handle))
        if not docs:
            continue
        doc = max(docs, key=lambda d: d.get("updated") or "")
        if not doc.get("done") and not args.include_unfinished:
            skipped.append(block["id"] + " (not done)")
            continue
        if doc.get("seeded_from") and args.exclude_seeded:
            skipped.append("{} (from {})".format(block["id"], doc["seeded_from"]))
            continue
        changes = shifted_changes(doc, shifts) if args.shift == "auto" else doc["changes"]
        window = dense(changes, block["first"], block["last"])
        labels[block["first"]:block["last"] + 1] = window
        used.append((block["id"], doc["annotator"], float((window != "X").mean()),
                     doc.get("seeded_from", "")))

    out = args.out or os.path.join(LABELS_DIR, "validation_{}.csv".format(args.annotator))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(out, index=False)

    for block_id, annotator, coverage, seeded in used:
        print("  {:<6} {:<16} {:>6.1%} labelled{}".format(
            block_id, annotator, coverage, "   started from " + seeded if seeded else ""))
    if skipped:
        print("  skipped: " + ", ".join(skipped))
    for annotator, shift in sorted(shifts.items()):
        print("  shift {:<16} {}".format(annotator, ", ".join(
            "{} {:+d} f".format(kind, frames) for kind, frames in shift.items())
            if shift else "none (no from-scratch labels on the public block)"))
    graded = labels != "X"
    print("{} blocks, {} frames ({:.1f} min) -> {}".format(
        len(used), int(graded.sum()), graded.sum() / FPS / 60, os.path.relpath(out)))


if __name__ == "__main__":
    main()
