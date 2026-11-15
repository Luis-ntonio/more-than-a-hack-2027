"""Does Smart Tagging help as input? Measured without the provided match.

    docker compose run --rm dev python -m external.tag_experiment

No external dataset has Metrica's tags, so external.tags simulates them from each match's truth,
with error levels drawn over wide ranges for training. Any gain then depends on how noisy the real
tags are, which cannot be measured. So the models are scored on held-out matches with tags at four
fixed levels (external.tags.LEVELS, from good to useless), and the rule asks for two things: a gain
when tags are mid, and no loss when they are useless, which is what over-trusting them would cost.

Arms, on the final pipeline's columns (compute_final) with SkillCorner at SC_WEIGHT in the team model:
  base                no tags
  +tags both          the 21 tag columns in both models
  +tags both +trust   also the 3 trust columns: how far tags agree with the nearest team
  +tags team +trust   the 24 columns in the team model only
Same seeds, held-out IDSSE pairs and SkillCorner folds as external.team_variables. SkillCorner's tags
are simulated from its own truth, as IDSSE's are.

An arm is adopted if, at "mid", its mean IDSSE gain is at least MIN_GAIN and positive on every seed
and SkillCorner does not fall on average; and, at "useless", neither source falls by more than
MAX_LOSS on average. Among those, the largest mean gain at "mid" over both sources. These rules were
set before seeing any result.

First attempt, without the trust columns: both tag arms gained at mid and lost 0.004-0.007 at
useless, so neither passed. The trust arms are the second attempt, under the same rule.
"""
import os

import numpy as np

from external import idsse, skillcorner as sc
from external.degrade import degrade
from external.features import compute_final, swap_teams
from external.skillcorner_training import EVAL_SEEDS, FOLDS, train
from external.skillcorner_weight import SPLITS, fit_team
from external.tags import LEVELS, columns, flip, nearest_sign, simulate
from external.train import LGB_PARAMS, SC_WEIGHT, STRIDE, decode, score, swap_labels, transition_costs, view_of
from external.validate_skillcorner import view_of as sc_view

MIN_GAIN = 0.005
MAX_LOSS = 0.003
N_TAGS, N_TRUST = 21, 24
ARMS = {"base": (0, 0), "+tags both": (N_TAGS, N_TAGS), "+tags both +trust": (N_TRUST, N_TRUST),
        "+tags team +trust": (0, N_TRUST)}   # tag columns read by (dead model, team model)


def with_tags(view, tags, labels_len):
    X, names = compute_final(view)
    T = columns(tags, labels_len, nearest_sign(X, names), view["half"])[0]
    assert T.shape[1] == N_TRUST
    return X, T


def training_rows(view, labels, tags, keep):
    out = []
    for v, lab, tg in ((view, labels, tags), (swap_teams(view), swap_labels(labels), flip(tags))):
        X, T = with_tags(v, tg, len(lab))
        out.append((np.hstack([X, T])[keep][::STRIDE], lab[keep][::STRIDE]))
    return np.concatenate([x for x, _ in out]), np.concatenate([y for _, y in out])


def eval_set(view, truth, key):
    X, names = compute_final(view)
    near = nearest_sign(X, names)
    T = {lv: columns(simulate(truth, np.random.default_rng(key + [i]), LEVELS[lv]), len(truth), near, view["half"])[0]
         for i, lv in enumerate(LEVELS)}
    return X, T


def load_idsse(seed, pair):
    rng, tag_rng = np.random.default_rng(seed), np.random.default_rng(1000 + seed)
    rows, labels = [], []
    for match in [m for m in idsse.MATCHES if m not in pair]:
        d = idsse.load(match)
        truth = idsse.to_labels(d)
        labels.append(truth)
        view = degrade(view_of(d), d["ball"][:, 2], rng, fill=True)
        rows.append(training_rows(view, truth, simulate(truth, tag_rng), np.ones(len(truth), dtype=bool)))
        print("idsse", match, flush=True)
    held = []
    for k, match in enumerate(pair):
        d = idsse.load(match)
        truth = idsse.to_labels(d)
        for s in EVAL_SEEDS:
            view = degrade(view_of(d), d["ball"][:, 2], np.random.default_rng(s), fill=True)
            held.append(eval_set(view, truth, [seed, k, s]) + (truth, d["half"]))
    return (np.concatenate([x for x, _ in rows]), np.concatenate([y for _, y in rows]),
            transition_costs(labels), held)


def load_corner():
    out = {}
    for k, match in enumerate(sc.match_ids()):
        d = np.load(os.path.join(sc.DATA_DIR, match + ".npz"))
        v, truth = sc_view(d, "detected")
        has = v.pop("has")
        view = dict(v, team=d["team"], pitch=tuple(d["pitch"].tolist()))
        X_t, y_t = training_rows(view, truth, simulate(truth, np.random.default_rng([7, k])), has)
        out[match] = (X_t, y_t) + eval_set(view, truth, [99, k]) + (truth, has, v["half"])
        print("skillcorner", match, flush=True)
    return out


def scored(dead, team, k_dead, k_team, X, T, truth, half, log_t, has=None):
    pred = decode(dead.predict(np.hstack([X, T[:, :k_dead]])), team.predict(np.hstack([X, T[:, :k_team]])),
                  half, log_t)
    s = score(truth, pred)
    keep = (truth != "D") & (np.ones(len(truth), dtype=bool) if has is None else has)
    return s + (np.mean(pred[keep] == truth[keep]),)


def run_seed(seed, corner):
    LGB_PARAMS["seed"] = seed
    X_i, y_i, log_t, held_idsse = load_idsse(seed, SPLITS[seed])
    nb = held_idsse[0][0].shape[1]
    sizes = sorted({k for pair in ARMS.values() for k in pair})
    dead = {k: train(X_i[:, :nb + k], y_i == "D", None) for k in sorted({a for a, _ in ARMS.values()})}
    matches = sorted(corner)
    sc_rows = {(a, lv): {} for a in ARMS for lv in LEVELS}
    id_rows = {(a, lv): [] for a in ARMS for lv in LEVELS}
    for fold in range(FOLDS):
        held = matches[fold::FOLDS]
        rows = [corner[m][:2] for m in matches if m not in held]
        X_s, y_s = np.concatenate([x for x, _ in rows]), np.concatenate([y for _, y in rows])
        team = {k: fit_team(X_i[:, :nb + k], y_i, X_s[:, :nb + k], y_s, SC_WEIGHT)
                for k in sizes if any(t == k for _, t in ARMS.values())}
        for arm, (kd, kt) in ARMS.items():
            for lv in LEVELS:
                if arm == "base" and lv != "mid":
                    continue
                for m in held:
                    _, _, X, T, truth, has, half = corner[m]
                    sc_rows[arm, lv][m] = scored(dead[kd], team[kt], kd, kt, X, T[lv], truth, half, log_t, has)
                id_rows[arm, lv].append([scored(dead[kd], team[kt], kd, kt, X, T[lv], truth, half, log_t)
                                         for X, T, truth, half in held_idsse])
        print("seed", seed, "fold", fold, "done", flush=True)
    for lv in LEVELS:
        sc_rows["base", lv], id_rows["base", lv] = sc_rows["base", "mid"], id_rows["base", "mid"]
    return {key: (np.array([sc_rows[key][m] for m in matches]).mean(axis=0), np.mean(id_rows[key], axis=(0, 1)))
            for key in sc_rows}


def main():
    corner = load_corner()
    runs = [run_seed(seed, corner) for seed in SPLITS]
    print("R {:<18} {:<8} {:>4} {:>9} {:>6} {:>6} {:>9} {:>7} {:>8} {:>8}".format(
        "arm", "tags", "seed", "SC spell", "poss", "team", "ID spell", "ID team", "d SC", "d ID"))
    for seed, r in zip(SPLITS, runs):
        for lv in LEVELS:
            for arm in ARMS:
                s, i = r[arm, lv]
                b_s, b_i = r["base", lv]
                print("R {:<18} {:<8} {:>4} {:>9.3f} {:>6.0f} {:>6.3f} {:>9.3f} {:>7.3f} {:>+8.3f} {:>+8.3f}".format(
                    arm, lv, seed, s[0], s[2], s[3], i[0], i[3], s[0] - b_s[0], i[0] - b_i[0]))
    passed = []
    for arm in list(ARMS)[1:]:
        d = {lv: (np.array([r[arm, lv][0][0] - r["base", lv][0][0] for r in runs]),
                  np.array([r[arm, lv][1][0] - r["base", lv][1][0] for r in runs])) for lv in LEVELS}
        for lv in LEVELS:
            ds, di = d[lv]
            print("R mean {} at {}: d SC {:+.3f} (min {:+.3f}), d ID {:+.3f} (min {:+.3f})".format(
                arm, lv, ds.mean(), ds.min(), di.mean(), di.min()))
        ds, di = d["mid"]
        us, ui = d["useless"]
        ok = bool(di.mean() >= MIN_GAIN and np.all(di > 0) and ds.mean() >= 0
                  and us.mean() >= -MAX_LOSS and ui.mean() >= -MAX_LOSS)
        print("R {} passes {}".format(arm, ok))
        if ok:
            passed.append((ds.mean() + di.mean(), arm))
    print("R adopted:", max(passed)[1] if passed else None)


if __name__ == "__main__":
    main()
