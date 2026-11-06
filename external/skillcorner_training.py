"""Does training on SkillCorner as well as IDSSE help? Measured without the provided match.

    docker compose run --rm dev python -m external.skillcorner_training

Three arms, all on the same 5 IDSSE training matches and with IDSSE's HMM costs:
  idsse        both models on IDSSE
  +sc both     both models also on SkillCorner
  +sc team     the dead/alive model on IDSSE, the team model also on SkillCorner
SkillCorner's dead frames follow another definition, which is why the third arm exists. Four folds
over its 20 matches, so each one is held out once. Only frames with tracking are trained on, and each
source weighs the same. Scored on held-out SkillCorner and on held-out IDSSE, degraded with fill.
"""
import os

import lightgbm as lgb
import numpy as np

from external import idsse, skillcorner as sc
from external.degrade import degrade
from external.features import compute, swap_teams
from external.train import (LGB_PARAMS, N_ROUNDS, STRIDE, VAL_MATCHES, decode, score, swap_labels,
                            transition_costs, view_of)
from external.validate_skillcorner import view_of as sc_view

FOLDS = 4
EVAL_SEEDS = (1, 2, 3)


def training_rows(view, labels, keep):
    """Strided rows of a view and of the same view with the teams swapped."""
    out = []
    for v, lab in ((view, labels), (swap_teams(view), swap_labels(labels))):
        X, _ = compute(v)
        out.append((X[keep][::STRIDE].astype(np.float32), lab[keep][::STRIDE]))
    return (np.concatenate([x for x, _ in out]), np.concatenate([y for _, y in out]))


def load_idsse_training(rng):
    rows, labels = [], []
    for match in [m for m in idsse.MATCHES if m not in VAL_MATCHES]:
        d = idsse.load(match)
        truth = idsse.to_labels(d)
        labels.append(truth)
        view = degrade(view_of(d), d["ball"][:, 2], rng, fill=True)
        rows.append(training_rows(view, truth, np.ones(len(truth), dtype=bool)))
        print("idsse", match, flush=True)
    return (np.concatenate([x for x, _ in rows]), np.concatenate([y for _, y in rows]),
            transition_costs(labels))


def load_skillcorner():
    """{match: (training X, training y, full X, truth, has, half)}."""
    out = {}
    for match in sc.match_ids():
        d = np.load(os.path.join(sc.DATA_DIR, match + ".npz"))
        v, truth = sc_view(d, "detected")
        has = v.pop("has")
        view = dict(v, team=d["team"], pitch=tuple(d["pitch"].tolist()))
        X, _ = compute(view)
        out[match] = training_rows(view, truth, has) + (X.astype(np.float32), truth, has, v["half"])
        print("skillcorner", match, flush=True)
    return out


def load_idsse_eval():
    out = []
    for match in VAL_MATCHES:
        d = idsse.load(match)
        truth = idsse.to_labels(d)
        for seed in EVAL_SEEDS:
            view = degrade(view_of(d), d["ball"][:, 2], np.random.default_rng(seed), fill=True)
            out.append((compute(view)[0].astype(np.float32), truth, d["half"]))
    return out


def train(X, y, w):
    return lgb.train(LGB_PARAMS, lgb.Dataset(X, y.astype(int), weight=w), N_ROUNDS)


def fit_pair(sources):
    """Dead and team models on [(X, labels)], each source with the same total weight."""
    X = np.concatenate([x for x, _ in sources])
    labels = np.concatenate([y for _, y in sources])
    weight = np.concatenate([np.full(len(y), 1.0 / len(y)) for _, y in sources]) * len(labels) / len(sources)
    dead = labels == "D"
    return train(X, dead, weight), train(X[~dead], labels[~dead] == "H", weight[~dead])


def scored(dead, team, log_t, X, truth, half, has=None):
    p_dead, p_home = dead.predict(X), team.predict(X)
    pred = decode(p_dead, p_home, half, log_t)
    s = score(truth, pred)
    if has is None:
        return s + (np.nan,)
    alive = (truth != "D") & has
    return s + (np.mean(pred[alive] == truth[alive]),)


def main():
    rng = np.random.default_rng(0)
    X_i, y_i, log_t = load_idsse_training(rng)
    corner = load_skillcorner()
    held_idsse = load_idsse_eval()
    base_dead, base_team = fit_pair([(X_i, y_i)])
    print("base models done", flush=True)
    matches = sorted(corner)
    on_sc, on_idsse = {}, {}
    for fold in range(FOLDS):
        held = matches[fold::FOLDS]
        rows = [corner[m][:2] for m in matches if m not in held]
        both_dead, both_team = fit_pair([(X_i, y_i), (np.concatenate([x for x, _ in rows]),
                                                      np.concatenate([y for _, y in rows]))])
        arms = {"idsse": (base_dead, base_team), "+sc both": (both_dead, both_team),
                "+sc team": (base_dead, both_team)}
        for arm, (dead, team) in arms.items():
            for m in held:
                _, _, X, truth, has, half = corner[m]
                on_sc.setdefault(arm, {})[m] = scored(dead, team, log_t, X, truth, half, has)
            rows_i = [scored(dead, team, log_t, X, truth, half) for X, truth, half in held_idsse]
            on_idsse.setdefault(arm, []).append(np.mean(rows_i, axis=0))
        print("fold", fold, "done", flush=True)

    base = np.array([on_sc["idsse"][m] for m in matches])
    print("{:<10} {:>9} {:>9} {:>6} {:>7} {:>11} {:>8} | {:>10} {:>9}".format(
        "arm", "SC spell", "SC macro", "poss", "team", "d spell", "sd d", "IDSSE spell", "macro"))
    for arm in on_sc:
        r = np.array([on_sc[arm][m] for m in matches])
        diff = r[:, 0] - base[:, 0]
        i = np.mean(on_idsse[arm], axis=0)
        print("{:<10} {:>9.3f} {:>9.3f} {:>6.0f} {:>7.3f} {:>+11.3f} {:>8.3f} | {:>10.3f} {:>9.3f}".format(
            arm, r[:, 0].mean(), r[:, 1].mean(), r[:, 2].mean(), r[:, 3].mean(), diff.mean(), diff.std(),
            i[0], i[1]))
    print("matches better than idsse:", {arm: int(np.sum(np.array([on_sc[arm][m][0] for m in matches]) > base[:, 0]))
                                          for arm in on_sc})


if __name__ == "__main__":
    main()
