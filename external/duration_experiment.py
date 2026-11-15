"""Does decoding with possession lengths (external.durations) beat the plain HMM? Without the match.

    docker compose run --rm dev python -m external.duration_experiment

Decoding only: the models are the final pipeline's (compute_final columns, dead/alive on IDSSE, team
on IDSSE plus SkillCorner at SC_WEIGHT), trained per seed and fold as in external.team_variables,
and both decoders read the same probabilities. Lengths and what follows each state come from the
training IDSSE matches' labels, like the HMM's costs.

The semi-Markov decoder is adopted if its mean spell F1 gain on held-out IDSSE is at least MIN_GAIN
and positive on every seed, and held-out SkillCorner does not fall on average. These rules were set
before seeing any result.
"""
import numpy as np

from external import durations, idsse
from external.features import compute_final
from external.skillcorner_training import FOLDS, load_idsse_eval, load_idsse_training, load_skillcorner, train
from external.skillcorner_weight import SPLITS, fit_team
from external.train import LGB_PARAMS, SC_WEIGHT, decode, score

MIN_GAIN = 0.005
N_DEAD = 149   # the dead/alive model reads compute()'s columns only


def scored(pred, truth, has=None):
    keep = (truth != "D") & (np.ones(len(truth), dtype=bool) if has is None else has)
    return score(truth, pred) + (np.mean(pred[keep] == truth[keep]),)


def run_seed(seed, corner):
    LGB_PARAMS["seed"] = seed
    pair = SPLITS[seed]
    X_i, y_i, log_t = load_idsse_training(np.random.default_rng(seed), pair, compute_final)
    model = durations.fit([idsse.to_labels(idsse.load(m)) for m in idsse.MATCHES if m not in pair])
    print("R seed {} stages {}".format(seed, {str(s): (k, round(q, 4)) for s, (k, q) in model["stages"].items()}))
    held_idsse = load_idsse_eval(pair, compute_final)
    dead = train(X_i[:, :N_DEAD], y_i == "D", None)
    matches = sorted(corner)
    p_sc = {m: dead.predict(corner[m][2][:, :N_DEAD]) for m in matches}
    p_id = [dead.predict(X[:, :N_DEAD]) for X, _, _ in held_idsse]
    rows = {"HMM": ({}, []), "semi-Markov": ({}, [])}
    for fold in range(FOLDS):
        held = matches[fold::FOLDS]
        train_rows = [corner[m][:2] for m in matches if m not in held]
        X_s, y_s = np.concatenate([x for x, _ in train_rows]), np.concatenate([y for _, y in train_rows])
        team = fit_team(X_i, y_i, X_s, y_s, SC_WEIGHT)
        cases = [(p_sc[m], corner[m][2], corner[m][3], corner[m][5], corner[m][4], m) for m in held]
        cases += [(p, X, t, h, None, None) for p, (X, t, h) in zip(p_id, held_idsse)]
        fold_id = {"HMM": [], "semi-Markov": []}
        for p_dead, X, truth, half, has, m in cases:
            p_home = team.predict(X)
            for name, pred in (("HMM", decode(p_dead, p_home, half, log_t)),
                               ("semi-Markov", durations.decode(p_dead, p_home, half, model))):
                if m is None:
                    fold_id[name].append(scored(pred, truth))
                else:
                    rows[name][0][m] = scored(pred, truth, has)
        for name in rows:
            rows[name][1].append(fold_id[name])
        print("seed", seed, "fold", fold, "done", flush=True)
    return {name: (np.array([sc_rows[m] for m in matches]).mean(axis=0), np.mean(id_rows, axis=(0, 1)))
            for name, (sc_rows, id_rows) in rows.items()}


def main():
    corner = load_skillcorner(compute_final)
    runs = [run_seed(seed, corner) for seed in SPLITS]
    print("R {:<12} {:>4} {:>9} {:>6} {:>6} {:>9} {:>6} {:>7} {:>8} {:>8}".format(
        "decoder", "seed", "SC spell", "poss", "team", "ID spell", "poss", "ID team", "d SC", "d ID"))
    for seed, r in zip(SPLITS, runs):
        for name, (s, i) in r.items():
            b_s, b_i = r["HMM"]
            print("R {:<12} {:>4} {:>9.3f} {:>6.0f} {:>6.3f} {:>9.3f} {:>6.0f} {:>7.3f} {:>+8.3f} {:>+8.3f}".format(
                name, seed, s[0], s[2], s[3], i[0], i[2], i[3], s[0] - b_s[0], i[0] - b_i[0]))
    ds = np.array([r["semi-Markov"][0][0] - r["HMM"][0][0] for r in runs])
    di = np.array([r["semi-Markov"][1][0] - r["HMM"][1][0] for r in runs])
    ok = bool(di.mean() >= MIN_GAIN and np.all(di > 0) and ds.mean() >= 0)
    print("R mean semi-Markov: d SC {:+.3f} (min {:+.3f}), d ID {:+.3f} (min {:+.3f}), passes {}".format(
        ds.mean(), ds.min(), di.mean(), di.min(), ok))


if __name__ == "__main__":
    main()
