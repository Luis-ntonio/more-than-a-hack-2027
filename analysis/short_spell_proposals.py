"""Find short possessions as whole segments instead of frame by frame, and score the result.

    docker compose run --rm dev python -m possession.prelabel          # prelabel_oof.csv
    docker compose run --rm dev python analysis/short_spell_proposals.py

The pre-labeller misses short possessions because frame by frame the evidence is weak, while over
the whole spell it is clear (analysis/short_spell_profile.py). So:
  1. proposals: on the out-of-fold pre-labeller's alive frames, runs where the team it does *not*
     give the ball to is nearer to the ball (gaps up to MERGE_F frames merged, at least MIN_F long,
     at most MAX_S);
  2. each proposal gets the profile's measures over the whole run, plus its length and how much
     nearer the other team is; a proposal is a hit if a true spell of that team overlaps it with
     IoU >= HIT_IOU;
  3. a small LightGBM model scores proposals of each block, trained on the other blocks'; proposals
     above a threshold become possessions of the other team.
The threshold is picked nested (on the other blocks) for each block. Prints spellF1 per block and
mean, against the pre-labeller alone. Aggregates only.
"""
import argparse
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.short_spell_profile import attack_sign, frame_data, half_of, measure  # noqa: E402
from possession.io import AWAY, FPS, HOME  # noqa: E402
from possession.prelabel import OOF_OUT, pm, training_blocks  # noqa: E402

MERGE_F = 5
MIN_F = 10
MAX_S = 10
HIT_IOU = 0.3
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
LGB_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 8, "min_child_samples": 15,
              "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1, "lambda_l2": 1.0,
              "seed": 0, "deterministic": True, "force_row_wise": True, "num_threads": 4, "verbose": -1}
N_ROUNDS = 200
CODE = {"H": HOME, "A": AWAY}
LABEL = {HOME: "H", AWAY: "A"}


def runs(mask):
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return list(zip(edges[::2], edges[1::2]))


def spells(labels):
    edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    return list(zip(np.concatenate([[0], edges]), np.concatenate([edges, [len(labels)]])))


def proposals(f, signs, base, first, last, merge_f=MERGE_F, min_f=MIN_F):
    """[(a, b, new team, features)] in block-relative frames."""
    lab = base[first:last + 1]
    poss = np.array([CODE.get(x, -1) for x in lab])
    d_home, d_away = f["d"][HOME][first:last + 1], f["d"][AWAY][first:last + 1]
    d_poss = np.where(poss == HOME, d_home, d_away)
    d_other = np.where(poss == HOME, d_away, d_home)
    mask = (poss >= 0) & np.isfinite(d_poss) & np.isfinite(d_other) & (d_other < d_poss)
    merged = []
    for a, b in runs(mask):
        if merged and a - merged[-1][1] <= merge_f and poss[a] == poss[merged[-1][0]]:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    out = []
    for a, b in merged:
        if b - a < min_f or b - a > MAX_S * FPS or poss[a] < 0:
            continue
        new = AWAY if poss[a] == HOME else HOME
        m = measure(f, first + a, first + b, new, signs[(half_of(first + a), new)])
        if m is None:
            continue
        m["len_s"] = (b - a) / FPS
        m["margin_m"] = np.nanmedian(d_poss[a:b] - d_other[a:b])
        # how long the pre-labeller's possession around it is
        sp = [(c, e) for c, e in spells(lab) if c <= a < e]
        m["host_s"] = (sp[0][1] - sp[0][0]) / FPS if sp else np.nan
        out.append((a, b, new, m))
    return out


def hit(a, b, new, truth):
    for c, e in spells(truth):
        if CODE.get(truth[c], -1) == new:
            inter = min(b, e) - max(a, c)
            if inter > 0 and inter / (max(b, e) - min(a, c)) >= HIT_IOU:
                return True
    return False


def coverage(props, blocks, base):
    """Share of true spells under 8 s after a spell of the other team that the pre-labeller
    misses (no same-team predicted spell with IoU >= 0.5) and a proposal hits."""
    missed, covered = 0, 0
    for p, (_, first, last, truth) in zip(props, blocks):
        pred = base[first:last + 1]
        pred_spells = [(c, e, pred[c]) for c, e in spells(pred)]
        sp = spells(truth)
        for (c, e), (pc, _) in zip(sp[1:], sp[:-1]):
            if truth[c] not in CODE or truth[pc] not in CODE or (e - c) >= 8 * FPS:
                continue
            found = any(l == truth[c] and min(e, d) - max(c, a) > 0 and
                        (min(e, d) - max(c, a)) / (max(e, d) - min(c, a)) >= 0.5
                        for a, d, l in pred_spells)
            if found:
                continue
            missed += 1
            covered += any(new == CODE[truth[c]] and min(e, b) - max(c, a) > 0 and
                           (min(e, b) - max(c, a)) / (max(e, b) - min(c, a)) >= HIT_IOU
                           for a, b, new, _ in p)
    return missed, covered


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--merge-f", type=int, default=MERGE_F)
    parser.add_argument("--min-f", type=int, default=MIN_F)
    args = parser.parse_args()
    f = frame_data()
    signs = attack_sign()
    base = pd.read_csv(OOF_OUT)["label"].astype(str).values
    blocks = training_blocks()
    props = [proposals(f, signs, base, first, last, args.merge_f, args.min_f)
             for _, first, last, _ in blocks]
    missed, covered = coverage(props, blocks, base)
    print("true spells under 8 s the pre-labeller misses: {}; hit by a proposal: {} ({:.0%})".format(
        missed, covered, covered / max(missed, 1)))
    names = list(props[0][0][3])
    X = [np.array([[m[k] for k in names] for _, _, _, m in p], dtype=np.float64) for p in props]
    y = [np.array([hit(a, b, new, truth) for a, b, new, _ in p]) for p, (_, _, _, truth) in zip(props, blocks)]
    n_prop = sum(len(p) for p in props)
    n_hit = sum(int(v.sum()) for v in y)
    print("{} proposals in {} blocks ({:.1f}/min), {} of them hits ({:.0%})".format(
        n_prop, len(blocks), n_prop / (sum(l - f_ + 1 for _, f_, l, _ in blocks) / FPS / 60), n_hit,
        n_hit / max(n_prop, 1)))

    # out-of-fold scores
    scores = []
    for k in range(len(blocks)):
        Xt = np.concatenate([X[j] for j in range(len(blocks)) if j != k])
        yt = np.concatenate([y[j] for j in range(len(blocks)) if j != k])
        model = lgb.train(LGB_PARAMS, lgb.Dataset(Xt, yt.astype(int)), N_ROUNDS)
        scores.append(model.predict(X[k]) if len(X[k]) else np.array([]))
    all_s, all_y = np.concatenate(scores), np.concatenate(y)
    order = np.argsort(-all_s)
    ranks = np.argsort(np.argsort(all_s)) + 1
    auc = (ranks[all_y].sum() - all_y.sum() * (all_y.sum() + 1) / 2) / (all_y.sum() * (~all_y).sum())
    print("out-of-fold AUC of the proposal score: {:.2f}".format(auc))
    for t in THRESHOLDS:
        keep = all_s >= t
        print("  score >= {:.1f}: {:>4} proposals kept, precision {:.0%}, recall of hits {:.0%}".format(
            t, int(keep.sum()), all_y[keep].mean() if keep.any() else np.nan,
            all_y[keep].sum() / max(all_y.sum(), 1)))

    # spellF1 per block and threshold
    spell = np.zeros((len(THRESHOLDS) + 1, len(blocks)))
    for k, (_, first, last, truth) in enumerate(blocks):
        pred_base = base[first:last + 1]
        spell[0, k] = pm.evaluate(pm.label_codes(truth), pm.label_codes(pred_base), bootstrap_n=0)["spell_f1_mean"]
        for i, t in enumerate(THRESHOLDS):
            pred = pred_base.copy()
            for (a, b, new, _), s in zip(props[k], scores[k]):
                if s >= t:
                    pred[a:b] = LABEL[new]
            spell[i + 1, k] = pm.evaluate(pm.label_codes(truth), pm.label_codes(pred), bootstrap_n=0)["spell_f1_mean"]
    print("\nmean spellF1 over {} blocks: pre-labeller {:.3f}".format(len(blocks), spell[0].mean()))
    for i, t in enumerate(THRESHOLDS):
        print("  + proposals with score >= {:.1f}: {:.3f}".format(t, spell[i + 1].mean()))
    nested = [spell[1 + int(np.argmax(np.delete(spell[1:], k, axis=1).mean(axis=1))), k]
              for k in range(len(blocks))]
    print("  nested (threshold picked without the block): {:.3f}  (blocks better / worse than the "
          "pre-labeller: {} / {})".format(np.mean(nested), int(np.sum(np.array(nested) > spell[0] + 1e-9)),
                                          int(np.sum(np.array(nested) < spell[0] - 1e-9))))


if __name__ == "__main__":
    main()
