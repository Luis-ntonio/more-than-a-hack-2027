"""Score saved models on SkillCorner's real broadcast tracking. Trains nothing.

    docker compose run --rm dev python -m external.validate_skillcorner

The final team model has now seen SkillCorner, so scoring it here is no longer a held-out figure; the
held-out ones come from external.skillcorner_training and external.skillcorner_weight.

SkillCorner runs at 10 fps, so positions are interpolated to the 25 fps the features expect. Its
truth is built by external.skillcorner: another provider's definition of possession, so a score here
measures agreement between two definitions on hard tracking, not accuracy on the task.
  detected: only what the camera saw; the rest is NaN, as a tracker that loses the ball would give
  extrapolated: SkillCorner's own fill of what is off screen
Scored on the whole timeline and on frames with tracking only: about a quarter of each match has
none, mostly replays, and the models never saw that during training.
"""
import os

import numpy as np

from external import skillcorner as sc
from external.features import FPS, compute
from external.train import decode, load, nearest_team, score

SOURCE_FPS = 10


def upsample(values):
    """Linear interpolation to 25 fps; the earlier value where the next one is missing."""
    n = int((len(values) - 1) * FPS / SOURCE_FPS) + 1
    position = np.arange(n) * SOURCE_FPS / FPS
    i = np.floor(position).astype(int)
    j = np.minimum(i + 1, len(values) - 1)
    w = (position - i).reshape((-1,) + (1,) * (values.ndim - 1))
    with np.errstate(invalid="ignore"):
        mixed = values[i] * (1 - w) + values[j] * w
    return np.where(np.isfinite(values[j]), mixed, values[i])


def nearest_frame(values):
    n = int((len(values) - 1) * FPS / SOURCE_FPS) + 1
    return values[np.minimum(np.round(np.arange(n) * SOURCE_FPS / FPS).astype(int), len(values) - 1)]


def view_of(d, variant):
    players, ball = d["players"].astype(np.float64), d["ball"].astype(np.float64)
    if variant == "detected":
        players[~d["seen"]] = np.nan
        ball[~d["ball_seen"]] = np.nan
    out = {"players": [], "ball": [], "half": [], "labels": [], "has": []}
    for h in (1, 2):
        rows = d["half"] == h
        out["players"].append(upsample(players[rows]))
        out["ball"].append(upsample(ball[rows]))
        out["labels"].append(nearest_frame(d["labels"][rows]))
        out["has"].append(nearest_frame(d["has_data"][rows]))
        out["half"].append(np.full(len(out["ball"][-1]), h))
    joined = {k: np.concatenate(v) for k, v in out.items()}
    labels = joined.pop("labels")
    return joined, np.where(labels == "N", "D", labels)


def main():
    dead, team, log_t = load()
    names = None
    results = {}
    frame_stats = []
    print("{:<9} {:<13} {:<15} {:<9} {:>8} {:>8} {:>6} {:>6}".format(
        "match", "tracking", "model", "frames", "spellF1", "macroF1", "poss", "truth"))
    for match in sc.match_ids():
        d = np.load(os.path.join(sc.DATA_DIR, match + ".npz"))
        for variant in ("detected", "extrapolated"):
            v, truth = view_of(d, variant)
            has = v.pop("has")
            view = dict(v, team=d["team"], pitch=tuple(d["pitch"].tolist()))
            X, names = compute(view)
            p_dead, p_home = dead.predict(X), team.predict(X)
            preds = {"nearest": np.where(p_dead >= 0.5, "D", nearest_team(X, names)),
                     "frame by frame": np.where(p_dead >= 0.5, "D", np.where(p_home >= 0.5, "H", "A")),
                     "HMM": decode(p_dead, p_home, view["half"], log_t)}
            for label, pred in preds.items():
                for scope, rows in (("all", np.ones(len(truth), dtype=bool)), ("tracked", has)):
                    t, p = truth[rows], pred[rows]
                    s = score(t, p)
                    n_true = int(np.sum((t[1:] != t[:-1]) & np.isin(t[1:], ["H", "A"])))
                    results.setdefault((variant, label, scope), []).append(s + (n_true,))
            hmm = preds["HMM"]
            alive = (truth != "D") & has
            frame_stats.append((variant, np.mean((hmm == "D") == (truth == "D")), np.mean(hmm[alive] == truth[alive]),
                                np.mean(truth[~has] == "D"), np.mean(truth[has] == "D"), np.mean(hmm == "D")))
        print(match, "done", flush=True)
    print("\nmean over {} matches".format(len(sc.match_ids())))
    print("{:<13} {:<15} {:<8} {:>8} {:>8} {:>6} {:>6}".format("tracking", "model", "frames", "spellF1", "macroF1", "poss", "truth"))
    for (variant, label, scope), rows in results.items():
        r = np.mean(rows, axis=0)
        sd = np.std([x[0] for x in rows])
        print("{:<13} {:<15} {:<8} {:>8.3f} {:>8.3f} {:>6.0f} {:>6.0f}   spellF1 sd {:.3f}".format(variant, label, scope, r[0], r[1], r[2], r[3], sd))
    for variant in ("detected", "extrapolated"):
        r = np.mean([x[1:] for x in frame_stats if x[0] == variant], axis=0)
        print("{}: dead/alive agreement {:.3f} | team agreement when alive {:.3f} | truth D with no tracking {:.3f}, "
              "with tracking {:.3f} | predicted D {:.3f}".format(variant, *r))


if __name__ == "__main__":
    main()
