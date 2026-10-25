"""Pre-labeller for the labelling loop: learns from every finished block, pre-labels the rest.

    docker compose run --rm dev python -m possession.prelabel

Trains two LightGBM models on window features around each frame -- alive vs dead, then home vs
away on alive frames -- from the public GT plus every block marked done in labels/. Their
probabilities go through v1's HMM, and the result is written to evaluation/results/prelabel.csv,
which the labeller offers as a model ("Partir de la prediccion"). Rerun after each finished block.
Hand labels are first moved onto the GT's timing with each annotator's offset (labeler.timing), so
the pre-labels land where the GT would put the changes; --no-shift trains on them as saved.

It is a labelling aid, not the submission: it is trained on hand labels of this very match.
Reports leave-one-block-out scores and correction counts (aggregates only) so the team can see
whether it saves work compared with v1. It also writes evaluation/results/prelabel_oof.csv: the
full-data prediction with every labelled block replaced by the prediction of the model that did not
see it, so scoring it against the hand labels is an honest out-of-sample number.
"""
import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from possession import features as feats
from possession import touches as touch_features
from possession.io import (AWAY, FPS, GAME, GAMES, HOME, PITCH, ROOT, graded_blocks,
                           kickoff_frames, load_atd, load_homography)
from possession.models import V1_PARAMS, _apply_prekickoff_rule, dead_probability, decode, team_logit

import sys
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from labeler.timing import shifted_changes  # noqa: E402

LABELS_DIR = os.path.join(ROOT, "labels")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv".format(GAME))
OUT = os.path.join(ROOT, "evaluation", "results", "prelabel.csv")
OOF_OUT = os.path.join(ROOT, "evaluation", "results", "prelabel_oof.csv")
WINDOWS_S = (0.5, 2.0, 5.0)
TRAIN_STRIDE = 5          # neighbouring frames are near-duplicates; every 5th is plenty
MATCH_TOL = FPS           # a predicted change within 1 s of the true one counts as right
LGB_PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 40,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "seed": 0, "deterministic": True, "force_row_wise": True, "num_threads": 4, "verbose": -1,
}
N_ROUNDS = 300


# -- features ------------------------------------------------------------------------------------

def frame_features(use_touches=False):
    """(n, k) per-frame feature matrix and names, from the original and the enriched ATD, plus
    ball touches from the detector's ball (possession.touches) if use_touches."""
    homography = load_homography()
    atd = load_atd()
    base = feats.compute(atd, homography)
    try:
        rich_atd = load_atd(enriched=True)
        rich = feats.compute(rich_atd, homography)
        source = rich_atd["ball_source"].astype(np.float64)
        conf = np.nan_to_num(rich_atd["ball_conf"].astype(np.float64), nan=0.0)
        ball = rich_atd["ball"].astype(np.float64) * np.array(PITCH)
    except SystemExit:
        print("  (no enriched ATD: run enrich.merge for the ball-from-video features)")
        rich, source, conf = base, np.zeros(len(base["d_home"])), np.zeros(len(base["d_home"]))
        ball = atd["ball"].astype(np.float64) * np.array(PITCH)

    length, width = PITCH
    players = atd["players"][:, :, :2].astype(np.float64) * np.array(PITCH)
    team = atd["team"]
    cols = {}

    def add(name, values):
        cols[name] = np.asarray(values, dtype=np.float64)

    def capped(d):
        return np.minimum(np.where(np.isfinite(d), d, 60.0), 60.0)

    add("atd_visible", base["ball_visible"])
    add("gap_len_s", base["gap_len_s"])
    add("speed_2s", base["speed_2s"])
    add("pan_2s", base["pan_2s"])
    add("v1_p_dead", dead_probability(base))
    add("v1_team_logit", team_logit(base))
    add("rich_visible", rich["ball_visible"])
    add("rich_d_home", capped(rich["d_home"]))
    add("rich_d_away", capped(rich["d_away"]))
    add("rich_diff", capped(rich["d_away"]) - capped(rich["d_home"]))
    add("ball_source", source)
    add("ball_conf", conf)
    add("ball_x", ball[:, 0])
    add("ball_y", ball[:, 1])
    add("ball_edge_m", np.minimum.reduce([ball[:, 0], length - ball[:, 0], ball[:, 1], width - ball[:, 1]]))
    add("out_of_pitch", rich["out_of_pitch"])
    step = np.linalg.norm(np.diff(ball, axis=0, prepend=ball[:1]), axis=1) * FPS
    add("ball_speed", np.where(step < 50, step, np.nan))
    for side, name in ((HOME, "home"), (AWAY, "away")):
        xy = players[:, team == side]
        finite = np.isfinite(xy[:, :, 0])
        n = np.maximum(finite.sum(axis=1), 1)
        add(name + "_cx", np.where(finite, xy[:, :, 0], 0).sum(axis=1) / n)
        add(name + "_cy", np.where(finite, xy[:, :, 1], 0).sum(axis=1) / n)
    add("cx_diff", cols["home_cx"] - cols["away_cx"])
    kickoffs = kickoff_frames()
    add("half2", np.arange(len(step)) >= kickoffs[2] - 10 * 60 * FPS)
    if use_touches:
        if touch_features.available():
            tracking = feats.clean_ball(atd["ball"].astype(np.float64) * np.array(PITCH))
            for name, values in touch_features.features(tracking).items():
                add(name, values)
        else:
            print("  (no detections on the ATD's ball frames: run enrich.detect_ball --mode visible"
                  " for the touch features)")

    # window summaries: the "context around frame n"
    for name in ("atd_visible", "rich_visible", "rich_diff", "v1_team_logit", "v1_p_dead",
                 "speed_2s", "pan_2s", "ball_speed", "ball_x", "ball_edge_m", "ball_conf",
                 "out_of_pitch", "cx_diff"):
        values = np.where(np.isfinite(cols[name]), cols[name], np.nan)
        for w in WINDOWS_S:
            half = int(w * FPS)
            mean = feats.rolling_nanmean(values, half)
            add("{}_m{}".format(name, w), mean)
            add("{}_past{}".format(name, w), np.concatenate([[np.nan] * half, mean[:-half]]))
            add("{}_next{}".format(name, w), np.concatenate([mean[half:], [np.nan] * half]))
    names = list(cols)
    return np.column_stack([cols[k] for k in names]).astype(np.float32), names


# -- labels --------------------------------------------------------------------------------------

def dense(changes, first, last):
    out = np.full(last - first + 1, "X", dtype="<U1")
    for (frame, label), nxt in zip(changes, changes[1:] + [[last + 1, None]]):
        out[frame - first:nxt[0] - first] = label
    return out


def training_blocks(shift=True):
    """[(name, first, last, labels)] from the public GT and every finished block in labels/."""
    blocks = []
    gt = np.array([str(v).strip().upper() for v in pm.read_label_csv(PUBLIC_GT)["label"].values])
    graded = np.flatnonzero(gt != "X")
    blocks.append(("public", int(graded[0]), int(graded[-1]), gt[graded[0]:graded[-1] + 1]))
    latest = {}
    if os.path.isdir(LABELS_DIR):
        for name in sorted(os.listdir(LABELS_DIR)):
            if "__" not in name or not name.endswith(".json"):
                continue
            with open(os.path.join(LABELS_DIR, name)) as handle:
                doc = json.load(handle)
            if not doc.get("done") or "first" not in doc:
                continue
            if doc["block"] not in latest or doc["updated"] > latest[doc["block"]]["updated"]:
                latest[doc["block"]] = doc
    shifts = {}
    for block_id, doc in sorted(latest.items()):
        if doc["first"] <= blocks[0][2] and doc["last"] >= blocks[0][1]:
            continue  # the public block: the official GT wins
        blocks.append((block_id + ("*" if doc.get("seeded_from") else ""), doc["first"], doc["last"],
                       dense(shifted_changes(doc, shifts) if shift else doc["changes"],
                             doc["first"], doc["last"])))
    return blocks


# -- model ---------------------------------------------------------------------------------------

def fit(X, blocks):
    rows, y_dead, y_home = [], [], []
    for _, first, last, labels in blocks:
        idx = np.arange(first, last + 1)[::TRAIN_STRIDE]
        lab = labels[::TRAIN_STRIDE]
        keep = lab != "X"
        rows.append(idx[keep])
        y_dead.append(lab[keep] == "D")
        y_home.append(lab[keep] == "H")
    rows, y_dead, y_home = np.concatenate(rows), np.concatenate(y_dead), np.concatenate(y_home)
    dead_model = lgb.train(LGB_PARAMS, lgb.Dataset(X[rows], y_dead.astype(int)), N_ROUNDS)
    alive = ~y_dead
    team_model = lgb.train(LGB_PARAMS, lgb.Dataset(X[rows[alive]], y_home[alive].astype(int)),
                           N_ROUNDS)
    return dead_model, team_model


def probabilities(models, X, n_frames):
    """Per-frame P(dead) and P(home | alive) on the graded frames, 0.5 elsewhere."""
    dead_model, team_model = models
    graded = np.zeros(n_frames, dtype=bool)
    for _, first, last in graded_blocks():
        graded[first:last + 1] = True
    p_dead, p_home = np.full(n_frames, 0.5), np.full(n_frames, 0.5)
    p_dead[graded] = dead_model.predict(X[graded])
    p_home[graded] = team_model.predict(X[graded])
    return p_dead, p_home


def predict(models, X, n_frames):
    return decode(*probabilities(models, X, n_frames), n_frames, V1_PARAMS)


# -- evaluation ----------------------------------------------------------------------------------

def changes_in(labels):
    edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    return [(int(i), labels[i]) for i in edges]


def corrections(truth, pred):
    """Edits to turn pred into truth: true changes with no same-label prediction within 1 s must
    be added or moved, predicted changes with no true counterpart must be deleted."""
    t, p = changes_in(truth), changes_in(pred)
    used = set()
    missed = 0
    for frame, label in t:
        match = next((j for j, (f, l) in enumerate(p)
                      if j not in used and l == label and abs(f - frame) <= MATCH_TOL), None)
        if match is None:
            missed += 1
        else:
            used.add(match)
    return missed + (len(p) - len(used)), len(t)


def score_block(truth_block, pred_block):
    n = len(truth_block)
    gt = pm.label_codes(truth_block)
    pr = pm.label_codes(pred_block)
    result = pm.evaluate(gt, pr, bootstrap_n=0) if (gt != pm.X).any() else None
    edits, n_true = corrections(truth_block, pred_block)
    return result, edits, n_true, n


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-cv", action="store_true", help="skip leave-one-block-out scoring")
    parser.add_argument("--touches", action="store_true",
                        help="add touch features from the video detector's ball (needs the GPU "
                             "detections; neutral in leave-one-block-out, docs/05 section 12)")
    parser.add_argument("--no-shift", action="store_true",
                        help="train on hand labels as saved, without moving them onto the GT's timing")
    args = parser.parse_args()

    n_frames = GAMES[GAME]["n_video_frames"]
    X, names = frame_features(use_touches=args.touches)
    blocks = training_blocks(shift=not args.no_shift)
    print("training blocks: " + ", ".join("{} ({:.1f} min)".format(b[0], (b[2] - b[1] + 1) / FPS / 60)
                                          for b in blocks) + "   (* = started from a model)")

    from possession.models import v1
    v1_labels = v1(feats.compute(load_atd(), load_homography()), n_frames)
    cv_blocks = blocks
    if len(blocks) == 1:
        # nothing to hold out yet: split the public 5 minutes into 1-minute folds. Noisy, and
        # neighbouring folds share window context, but it beats no estimate at all.
        name, first, last, labels = blocks[0]
        cuts = np.linspace(0, last - first + 1, 6).astype(int)
        cv_blocks = [("pub-{}".format(i), first + a, first + b - 1, labels[a:b])
                     for i, (a, b) in enumerate(zip(cuts[:-1], cuts[1:]))]
    if not args.no_cv:
        blocks_cv = cv_blocks
        print("\nleave-one-block-out (train on the others, score the held-out block){}:".format(
            "  [public split in 1-minute folds: rough]" if cv_blocks is not blocks else ""))
        print("{:<8} {:>9} {:>9} {:>9} | {:>11} {:>11}".format(
            "block", "spellF1", "macroF1", "v1 spell", "edits pre", "edits v1"))
        totals = np.zeros(3)
        oof = {}
        for k, (name, first, last, labels) in enumerate(blocks_cv):
            models = fit(X, blocks_cv[:k] + blocks_cv[k + 1:])
            pred = predict(models, X, n_frames)[first:last + 1]
            oof[(first, last)] = pred
            res, edits, n_true, _ = score_block(labels, pred)
            res_v1, edits_v1, _, _ = score_block(labels, v1_labels[first:last + 1])
            totals += (edits, edits_v1, n_true)
            print("{:<8} {:>9.3f} {:>9.3f} {:>9.3f} | {:>5} / {:<4} {:>5} / {:<4}".format(
                name, res["spell_f1_mean"], res["macro_f1"], res_v1["spell_f1_mean"],
                edits, n_true, edits_v1, n_true))
        print("total edits: pre-labeller {:.0f}, v1 {:.0f}  (true changes {:.0f})".format(*totals))

    models = fit(X, blocks)
    full = predict(models, X, n_frames)
    labels = _apply_prekickoff_rule(full.copy())
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pd.DataFrame({"frame": np.arange(n_frames), "label": labels}).to_csv(OUT, index=False)
    if not args.no_cv:
        for (first, last), pred in oof.items():
            full[first:last + 1] = pred
        oof_labels = _apply_prekickoff_rule(full)
        pd.DataFrame({"frame": np.arange(n_frames), "label": oof_labels}).to_csv(OOF_OUT, index=False)
        print("out-of-fold copy (labelled blocks predicted by the model that did not see them) -> "
              + os.path.relpath(OOF_OUT, ROOT))
    graded = np.concatenate([labels[f:l + 1] for _, f, l in graded_blocks()])
    print("\nfull match: {:.1%} D, {} feature columns -> {}".format(
        np.mean(graded == "D"), len(names), os.path.relpath(OUT, ROOT)))
    print("in the labeller: Modelo -> prelabel -> 'Partir de la prediccion'")


if __name__ == "__main__":
    main()
