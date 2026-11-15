"""Do new variables help the team model? Measured without the provided match.

    docker compose run --rm dev python -m external.team_variables

Team assignment is the weakest part, and today's columns mostly look at who is near the ball now.
Three candidate groups from external.features: continuity (who has kept touching it), attack (which
way the ball and the block of players are going) and shape (who is spread out, who is behind the
ball). The dead/alive model does not change. Each arm trains the team model as the final one is
trained, with SkillCorner at SC_WEIGHT; same seeds, held-out IDSSE pairs and SkillCorner folds as
external.skillcorner_weight, paired with the arm without new columns.

An arm is adopted if its mean spell F1 does not fall on either source, the sum of both gains is
positive on every seed and at least MIN_GAIN on average; among those, the largest sum. These rules
were set before seeing any result.

No arm passed: attack gained a lot on IDSSE and lost a little on SkillCorner, where only the players
on camera are seen. --follow-up tests that reading: ball and block of players apart, same rule, on
seeds and held-out pairs that played no part in suggesting it. That reading was wrong, and no arm
passed again. The attack columns went into the team model anyway, by a decision taken after seeing
these results and recorded in docs/07: the IDSSE gain repeated and is several times the loss.
"""
import argparse

import numpy as np

from external.features import compute_team
from external.skillcorner_training import FOLDS, load_idsse_eval, load_idsse_training, load_skillcorner, train
from external.skillcorner_weight import SPLITS, fit_team, result
from external.train import LGB_PARAMS, SC_WEIGHT

MIN_GAIN = 0.005
FOLLOW_UP = {3: ("J03WOH", "J03WN1"), 4: ("J03WOY", "J03WMX"), 5: ("J03WR9", "J03WPY")}


def synthetic_view():
    """A tiny made-up view: only the column names are wanted from it."""
    n = 400
    rng = np.random.default_rng(0)
    return {"players": rng.normal(size=(n, 4, 2)), "team": np.array([0, 0, 1, 1]), "ball": rng.normal(size=(n, 2)),
            "half": np.ones(n, dtype=int), "pitch": (105.0, 68.0)}


def arms(follow_up=False):
    """{arm: column indices}."""
    _, names, groups = compute_team(synthetic_view())
    new = {c for cols in groups.values() for c in cols}
    base = [i for i, c in enumerate(names) if c not in new]
    out = {"base": base}
    if follow_up:
        for name in ("ball_vx", "block_vx"):
            out["+" + name] = base + [names.index(c) for c in groups["attack"] if c.startswith(name)]
        out["+attack"] = base + [names.index(c) for c in groups["attack"]]
        return out
    for group, cols in groups.items():
        out["+" + group] = base + [names.index(c) for c in cols]
    out["+all"] = list(range(len(names)))
    return out


def run_seed(seed, pair, corner, columns):
    LGB_PARAMS["seed"] = seed
    X_i, y_i, log_t = load_idsse_training(np.random.default_rng(seed), pair, compute_team)
    held_idsse = load_idsse_eval(pair, compute_team)
    base = columns["base"]
    dead = train(X_i[:, base], y_i == "D", None)
    matches = sorted(corner)
    p_sc = {m: dead.predict(corner[m][2][:, base]) for m in matches}
    p_id = [dead.predict(X[:, base]) for X, _, _ in held_idsse]
    sc_rows = {arm: {} for arm in columns}
    id_rows = {arm: [] for arm in columns}
    for fold in range(FOLDS):
        held = matches[fold::FOLDS]
        rows = [corner[m][:2] for m in matches if m not in held]
        X_s, y_s = np.concatenate([x for x, _ in rows]), np.concatenate([y for _, y in rows])
        for arm, cols in columns.items():
            team = fit_team(X_i[:, cols], y_i, X_s[:, cols], y_s, SC_WEIGHT)
            for m in held:
                c = corner[m]
                sc_rows[arm][m] = result(p_sc[m], team, log_t, c[2][:, cols], c[3], c[5], c[4])
            id_rows[arm].append([result(p, team, log_t, X[:, cols], t, h, np.ones(len(t), dtype=bool))
                                 for p, (X, t, h) in zip(p_id, held_idsse)])
        print("seed", seed, "fold", fold, "done", flush=True)
    return {arm: (np.array([sc_rows[arm][m] for m in matches]).mean(axis=0),
                  np.mean(id_rows[arm], axis=(0, 1))) for arm in columns}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--follow-up", action="store_true")
    args = parser.parse_args()
    splits = FOLLOW_UP if args.follow_up else SPLITS
    columns = arms(args.follow_up)
    corner = load_skillcorner(compute_team)
    runs = [run_seed(seed, pair, corner, columns) for seed, pair in splits.items()]
    print("R {:<12} {:>5} {:>5} {:>9} {:>6} {:>6} {:>9} {:>7} {:>8} {:>8}".format(
        "arm", "seed", "cols", "SC spell", "poss", "team", "ID spell", "ID team", "d SC", "d ID"))
    for seed, r in zip(splits, runs):
        for arm, (s, i) in r.items():
            print("R {:<12} {:>5} {:>5} {:>9.3f} {:>6.0f} {:>6.3f} {:>9.3f} {:>7.3f} {:>+8.3f} {:>+8.3f}".format(
                arm, seed, len(columns[arm]), s[0], s[2], s[3], i[0], i[3],
                s[0] - r["base"][0][0], i[0] - r["base"][1][0]))
    passed = []
    for arm in columns:
        if arm == "base":
            continue
        ds = np.array([r[arm][0][0] - r["base"][0][0] for r in runs])
        di = np.array([r[arm][1][0] - r["base"][1][0] for r in runs])
        both = ds + di
        ok = bool(ds.mean() >= 0 and di.mean() >= 0 and np.all(both > 0) and both.mean() >= MIN_GAIN)
        print("R mean {}: d SC {:+.3f} (min {:+.3f}), d ID {:+.3f} (min {:+.3f}), sum {:+.3f}, passes {}".format(
            arm, ds.mean(), ds.min(), di.mean(), di.min(), both.mean(), ok))
        if ok:
            passed.append((both.mean(), arm))
    print("R adopted:", max(passed)[1] if passed else None)


if __name__ == "__main__":
    main()
