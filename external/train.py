"""Train the possession models on external data and validate them on held-out external matches.

    docker compose run --rm dev python -m external.train            # 5 matches, score on the other 2
    docker compose run --rm dev python -m external.train --final    # all 7, saved for external.predict

Two LightGBM models on the features of external/features.py -- dead / alive, then home / away on
alive frames -- trained on degraded IDSSE matches with a simulated detector fill of part of the lost
ball (external.fill_experiment shows why), both ways round (teams swapped), then a 3-state
HMM whose transition costs come from the training labels. Scored with the official metrics on
held-out IDSSE matches, clean and degraded, next to a nearest-player baseline. The final team model
also sees SkillCorner at a small weight (external.skillcorner_weight). Nothing here reads the
provided match.
"""
import argparse
import os
import sys

import lightgbm as lgb
import numpy as np

from external import idsse
from external.degrade import degrade
from external.features import compute, swap_teams
from possession.decode import viterbi

sys.path.insert(0, os.path.join(idsse.ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

MODEL_DIR = os.path.join(idsse.ROOT, "external_data", "models")
VAL_MATCHES = ("J03WOY", "J03WR9")
STRIDE = 5
LGB_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 100,
              "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1, "lambda_l2": 1.0,
              "seed": 0, "deterministic": True, "force_row_wise": True, "num_threads": 8, "verbose": -1}
N_ROUNDS = 300
SC_WEIGHT = 0.10    # SkillCorner's share of IDSSE's weight in the final team model
STATES = np.array(["H", "A", "D"])


def view_of(d):
    return {"players": d["players"].astype(np.float64), "team": d["team"].astype(int),
            "ball": d["ball"][:, :2].astype(np.float64), "half": d["half"],
            "pitch": tuple(d["pitch"].tolist())}


def swap_labels(labels):
    return np.where(labels == "H", "A", np.where(labels == "A", "H", labels))


def transition_costs(all_labels):
    """(3, 3) log transition matrix from label counts; every change gets a floor of one count."""
    counts = np.ones((3, 3))
    index = {"H": 0, "A": 1, "D": 2}
    for labels in all_labels:
        codes = np.array([index[x] for x in labels])
        np.add.at(counts, (codes[:-1], codes[1:]), 1)
    return np.log(counts / counts.sum(axis=1, keepdims=True))


def decode(p_dead, p_home, half, log_transition):
    """Viterbi per half, on the two models' probabilities."""
    p_dead = np.clip(p_dead, 1e-6, 1 - 1e-6)
    p_home = np.clip(p_home, 1e-6, 1 - 1e-6)
    emission = np.column_stack([np.log1p(-p_dead) + np.log(p_home),
                                np.log1p(-p_dead) + np.log1p(-p_home), np.log(p_dead)])
    out = np.empty(len(half), dtype="<U1")
    for h in np.unique(half):
        rows = half == h
        out[rows] = STATES[viterbi(emission[rows], log_transition)]
    return out


def nearest_team(X, names):
    """The team of the nearest player, carried through frames without a ball."""
    diff = X[:, names.index("d_diff")]
    last = np.maximum.accumulate(np.where(np.isfinite(diff), np.arange(len(diff)), 0))
    return np.where(diff[last] >= 0, "H", "A")


def score(truth, pred):
    r = pm.evaluate(pm.label_codes(truth), pm.label_codes(pred), bootstrap_n=0)
    spells = int(np.sum((pred[1:] != pred[:-1]) & np.isin(pred[1:], ["H", "A"])))
    return r["spell_f1_mean"], r["macro_f1"], spells


def fit(matches, rng, fill=False, team_extra=None):
    """(dead model, team model, log transition matrix, feature names) from these IDSSE matches.

    team_extra: (X, labels) added to the team model only, adding up to SC_WEIGHT of IDSSE's weight.
    """
    rows_X, y_dead, y_home, train_labels = [], [], [], []
    names = None
    for match in matches:
        d = idsse.load(match)
        labels = idsse.to_labels(d)
        train_labels.append(labels)
        view = degrade(view_of(d), d["ball"][:, 2], rng, fill=fill)
        for v, lab in ((view, labels), (swap_teams(view), swap_labels(labels))):
            X, names = compute(v)
            rows_X.append(X[::STRIDE])
            y_dead.append(lab[::STRIDE] == "D")
            y_home.append(lab[::STRIDE] == "H")
        print("features {}: {} columns".format(match, X.shape[1]), flush=True)
    X = np.concatenate(rows_X)
    y_dead, y_home = np.concatenate(y_dead), np.concatenate(y_home)
    del rows_X
    dead = lgb.train(LGB_PARAMS, lgb.Dataset(X, y_dead.astype(int)), N_ROUNDS)
    alive = ~y_dead
    X_team, y_team, weight = X[alive], y_home[alive], None
    if team_extra is not None:
        X_s, labels_s = team_extra
        extra = labels_s != "D"
        X_team = np.concatenate([X_team, X_s[extra]])
        y_team = np.concatenate([y_team, labels_s[extra] == "H"])
        weight = np.concatenate([np.full(alive.sum(), 1.0 / alive.sum()),
                                 np.full(extra.sum(), SC_WEIGHT / extra.sum())])
        weight *= len(weight) / weight.sum()
    team = lgb.train(LGB_PARAMS, lgb.Dataset(X_team, y_team.astype(int), weight=weight), N_ROUNDS)
    return dead, team, transition_costs(train_labels), names


def save(dead, team, log_t):
    os.makedirs(MODEL_DIR, exist_ok=True)
    dead.save_model(os.path.join(MODEL_DIR, "dead.txt"))
    team.save_model(os.path.join(MODEL_DIR, "team.txt"))
    np.save(os.path.join(MODEL_DIR, "log_transition.npy"), log_t)


def load():
    return (lgb.Booster(model_file=os.path.join(MODEL_DIR, "dead.txt")),
            lgb.Booster(model_file=os.path.join(MODEL_DIR, "team.txt")),
            np.load(os.path.join(MODEL_DIR, "log_transition.npy")))


def validate(dead, team, log_t, names):
    print("\n{:<9} {:<9} {:<24} {:>8} {:>8} {:>6} {:>6}".format(
        "match", "tracking", "model", "spellF1", "macroF1", "poss", "truth"))
    results = {}
    for match in VAL_MATCHES:
        d = idsse.load(match)
        truth = idsse.to_labels(d)
        n_true = int(np.sum((truth[1:] != truth[:-1]) & np.isin(truth[1:], ["H", "A"])))
        base = view_of(d)
        for name, view in (("clean", base),
                           ("degraded", degrade(base, d["ball"][:, 2], np.random.default_rng(1)))):
            Xv, _ = compute(view)
            pd_, ph = dead.predict(Xv), team.predict(Xv)
            frame = np.where(pd_ >= 0.5, "D", np.where(ph >= 0.5, "H", "A"))
            nearest = np.where(pd_ >= 0.5, "D", nearest_team(Xv, names))
            for label, pred in (("nearest + model's D", nearest), ("frame by frame", frame),
                                ("HMM", decode(pd_, ph, d["half"], log_t))):
                s = score(truth, pred)
                results.setdefault((name, label), []).append(s)
                print("{:<9} {:<9} {:<24} {:>8.3f} {:>8.3f} {:>6} {:>6}".format(
                    match, name, label, s[0], s[1], s[2], n_true), flush=True)
    print("\nmean over the held-out matches")
    for (name, label), rows in results.items():
        r = np.mean(rows, axis=0)
        print("  {:<9} {:<24} spellF1 {:.3f}  macroF1 {:.3f}  poss {:.0f}".format(name, label, *r))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--final", action="store_true", help="train on all matches and save the models")
    args = parser.parse_args()
    LGB_PARAMS["seed"] = args.seed
    rng = np.random.default_rng(args.seed)
    if args.final:
        from external.skillcorner_training import load_skillcorner
        rows = [m[:2] for m in load_skillcorner().values()]
        extra = np.concatenate([x for x, _ in rows]), np.concatenate([y for _, y in rows])
        dead, team, log_t, _ = fit(idsse.MATCHES, rng, fill=True, team_extra=extra)
        save(dead, team, log_t)
        print("-> " + os.path.relpath(MODEL_DIR, idsse.ROOT))
        return
    dead, team, log_t, names = fit([m for m in idsse.MATCHES if m not in VAL_MATCHES], rng, fill=True)
    print("transition per frame: stay H {:.4f}, H>A {:.4f}, H>D {:.4f}, D>H {:.4f}".format(
        *np.exp([log_t[0, 0], log_t[0, 1], log_t[0, 2], log_t[2, 0]])))
    validate(dead, team, log_t, names)


if __name__ == "__main__":
    main()
