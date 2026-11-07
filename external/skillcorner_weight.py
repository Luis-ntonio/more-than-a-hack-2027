"""How much should SkillCorner weigh in the team model? Measured without the provided match.

    docker compose run --rm dev python -m external.skillcorner_weight

external.skillcorner_training showed that SkillCorner helps on its own matches and costs something on
IDSSE at equal weight. Here the dead/alive model stays on IDSSE, and the team model adds SkillCorner
at WEIGHTS of IDSSE's total weight. Three seeds, each with its own held-out IDSSE pair; four folds
over SkillCorner, so each of its matches is held out once per seed. Differences are paired with the
IDSSE-only model of the same seed.

A weight is adopted if the mean loss on IDSSE is at most MAX_LOSS and SkillCorner improves on every
seed; among those, the one with the largest gain. These rules were set before seeing any result.
"""
import lightgbm as lgb
import numpy as np

from external.skillcorner_training import (FOLDS, fit_pair, load_idsse_eval, load_idsse_training,
                                           load_skillcorner)
from external.train import LGB_PARAMS, N_ROUNDS, decode, score

WEIGHTS = (0.10, 0.25, 0.50)
SPLITS = {0: ("J03WOY", "J03WR9"), 1: ("J03WMX", "J03WPY"), 2: ("J03WN1", "J03WQQ")}
MAX_LOSS = 0.010


def fit_team(X_i, y_i, X_s, y_s, weight):
    """Team model on alive frames; SkillCorner's rows add up to `weight` of IDSSE's."""
    a, b = y_i != "D", y_s != "D"
    X = np.concatenate([X_i[a], X_s[b]])
    y = np.concatenate([y_i[a], y_s[b]]) == "H"
    w = np.concatenate([np.full(a.sum(), 1.0 / a.sum()), np.full(b.sum(), weight / b.sum())])
    return lgb.train(LGB_PARAMS, lgb.Dataset(X, y.astype(int), weight=w * len(w) / w.sum()), N_ROUNDS)


def result(p_dead, team, log_t, X, truth, half, has=None):
    pred = decode(p_dead, team.predict(X), half, log_t)
    s = score(truth, pred)
    if has is None:
        return s + (np.nan,)
    alive = (truth != "D") & has
    return s + (np.mean(pred[alive] == truth[alive]),)


def run_seed(seed, corner):
    LGB_PARAMS["seed"] = seed
    held_pair = SPLITS[seed]
    X_i, y_i, log_t = load_idsse_training(np.random.default_rng(seed), held_pair)
    held_idsse = load_idsse_eval(held_pair)
    dead, base_team = fit_pair([(X_i, y_i)])
    matches = sorted(corner)
    p_sc = {m: dead.predict(corner[m][2]) for m in matches}
    p_id = [dead.predict(X) for X, _, _ in held_idsse]

    def on_idsse(team):
        return np.array([result(p, team, log_t, X, t, h) for p, (X, t, h) in zip(p_id, held_idsse)])

    def on_sc(team, held):
        return {m: result(p_sc[m], team, log_t, corner[m][2], corner[m][3], corner[m][5], corner[m][4])
                for m in held}

    base_id, base_sc = on_idsse(base_team), on_sc(base_team, matches)
    sc_rows = {w: {} for w in WEIGHTS}
    id_rows = {w: [] for w in WEIGHTS}
    for fold in range(FOLDS):
        held = matches[fold::FOLDS]
        rows = [corner[m][:2] for m in matches if m not in held]
        X_s, y_s = np.concatenate([x for x, _ in rows]), np.concatenate([y for _, y in rows])
        for w in WEIGHTS:
            team = fit_team(X_i, y_i, X_s, y_s, w)
            sc_rows[w].update(on_sc(team, held))
            id_rows[w].append(on_idsse(team))
        print("seed", seed, "fold", fold, "done", flush=True)
    base_sc = np.array([base_sc[m] for m in matches])
    out = {"base": (base_sc.mean(axis=0), base_id.mean(axis=0))}
    for w in WEIGHTS:
        r = np.array([sc_rows[w][m] for m in matches])
        i = np.mean(id_rows[w], axis=0)
        out[w] = (r.mean(axis=0), i.mean(axis=0), (r[:, 0] - base_sc[:, 0]).mean(), (i[:, 0] - base_id[:, 0]).mean())
    return out


def main():
    corner = load_skillcorner()
    runs = [run_seed(seed, corner) for seed in SPLITS]
    print("R {:<6} {:>5} {:>9} {:>8} {:>6} {:>6} {:>9} {:>9} {:>9}".format(
        "weight", "seed", "SC spell", "SC macro", "poss", "team", "ID spell", "d SC", "d ID"))
    for seed, r in zip(SPLITS, runs):
        s, i = r["base"]
        print("R {:<6} {:>5} {:>9.3f} {:>8.3f} {:>6.0f} {:>6.3f} {:>9.3f}".format("base", seed, s[0], s[1], s[2], s[3], i[0]))
        for w in WEIGHTS:
            s, i, ds, di = r[w]
            print("R {:<6} {:>5} {:>9.3f} {:>8.3f} {:>6.0f} {:>6.3f} {:>9.3f} {:>+9.3f} {:>+9.3f}".format(
                w, seed, s[0], s[1], s[2], s[3], i[0], ds, di))
    passed = []
    for w in WEIGHTS:
        ds = np.array([r[w][2] for r in runs])
        di = np.array([r[w][3] for r in runs])
        ok = bool(di.mean() >= -MAX_LOSS and np.all(ds > 0))
        print("R mean weight {}: d SC {:+.3f} (min {:+.3f}), d ID {:+.3f} (sd {:.3f}), passes {}".format(
            w, ds.mean(), ds.min(), di.mean(), di.std(), ok))
        if ok:
            passed.append((ds.mean(), w))
    print("R adopted:", max(passed)[1] if passed else None)


if __name__ == "__main__":
    main()
