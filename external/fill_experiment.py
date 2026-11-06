"""Does a detector-filled ball help the possession model? Measured without the provided match.

    docker compose run --rm dev python -m external.fill_experiment

Two models on the same 5 IDSSE matches: one trained without fill, one with simulated detector fill
and the `filled` flag. Both are scored by HMM on the held-out matches, degraded with and without
fill, over several degradation seeds. SkillCorner has no fill: it checks that the fill-aware model
loses nothing when the flag is never set.
"""
import os

import numpy as np

from external import idsse, skillcorner as sc
from external.degrade import degrade
from external.features import compute
from external.train import VAL_MATCHES, decode, fit, score, view_of
from external.validate_skillcorner import view_of as sc_view

EVAL_SEEDS = (1, 2, 3)


def on_idsse(model, fill):
    dead, team, log_t, _ = model
    rows = []
    for match in VAL_MATCHES:
        d = idsse.load(match)
        truth = idsse.to_labels(d)
        for seed in EVAL_SEEDS:
            view = degrade(view_of(d), d["ball"][:, 2], np.random.default_rng(seed), fill=fill)
            X, _ = compute(view)
            rows.append(score(truth, decode(dead.predict(X), team.predict(X), d["half"], log_t)))
    return np.mean(rows, axis=0), np.std([r[0] for r in rows])


def on_skillcorner(model):
    dead, team, log_t, _ = model
    rows = []
    for match in sc.match_ids():
        d = np.load(os.path.join(sc.DATA_DIR, match + ".npz"))
        v, truth = sc_view(d, "detected")
        v.pop("has")
        X, _ = compute(dict(v, team=d["team"], pitch=tuple(d["pitch"].tolist())))
        rows.append(score(truth, decode(dead.predict(X), team.predict(X), v["half"], log_t)))
    return np.mean(rows, axis=0), np.std([r[0] for r in rows])


def main():
    train = [m for m in idsse.MATCHES if m not in VAL_MATCHES]
    models = {"no fill": fit(train, np.random.default_rng(0)),
              "fill-aware": fit(train, np.random.default_rng(0), fill=True)}
    print("{:<12} {:<26} {:>8} {:>6} {:>8} {:>6}".format("model", "scored on", "spellF1", "sd", "macroF1", "poss"))
    for name, model in models.items():
        for label, result in (("IDSSE degraded", on_idsse(model, False)), ("IDSSE degraded + fill", on_idsse(model, True)),
                              ("SkillCorner detected", on_skillcorner(model))):
            r, sd = result
            print("{:<12} {:<26} {:>8.3f} {:>6.3f} {:>8.3f} {:>6.0f}".format(name, label, r[0], sd, r[1], r[2]), flush=True)


if __name__ == "__main__":
    main()
