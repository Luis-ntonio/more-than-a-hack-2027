"""The organisers' baseline against our pipeline, on external matches neither has trained on.

    docker compose run --rm dev python -m external.baseline_compare

The baseline is starter/baseline.py's algorithm: team of the player nearest the ball, D where the
ball is lost, then a majority vote over +-window frames (the published floor uses 50). It reads the
raw tracking, so it never sees the detector fill; our pipeline does, as on the match. Both are scored
with the official metrics on the same frames:
  IDSSE: the 2 held-out matches, clean and degraded with fill by 3 seeds; the baseline gets the
         same degradation with the filled frames lost again
  SkillCorner: the 20 matches, camera view only ("detected")
Our models here are trained on the 5 other IDSSE matches only, so every SkillCorner match is held
out too. Nothing here reads the provided match, and nothing printed sets a parameter.
"""
import os

import numpy as np

from external import idsse, skillcorner as sc
from external.degrade import degrade
from external.features import compute_final
from external.train import VAL_MATCHES, decode, fit, predict_both, score, view_of
from external.validate_skillcorner import view_of as sc_view
from starter.baseline import AWAY, HOME, NOBODY, smooth

WINDOWS = (0, 25, 50, 125)
SEEDS = (1, 2, 3)


def baseline_codes(X, names):
    d_home, d_away = X[:, names.index("d_home")], X[:, names.index("d_away")]
    seen = np.isfinite(d_home) & np.isfinite(d_away)
    return np.where(seen, np.where(d_home <= d_away, HOME, AWAY), NOBODY).astype(np.int8)


def both(dead, team, log_t, view, base_view, truth, out, key):
    X, names = compute_final(view)
    out.setdefault((key, "ours (models + HMM)"), []).append(score(truth, decode(*predict_both(dead, team, X), view["half"], log_t)))
    codes = baseline_codes(*compute_final(base_view))
    for w in WINDOWS:
        out.setdefault((key, "baseline +-{}".format(w)), []).append(score(truth, smooth(codes, w)))


def main():
    dead, team, log_t, _ = fit([m for m in idsse.MATCHES if m not in VAL_MATCHES], np.random.default_rng(0), fill=True)
    out, truths = {}, {}
    for match in VAL_MATCHES:
        d = idsse.load(match)
        truth = idsse.to_labels(d)
        base = view_of(d)
        both(dead, team, log_t, base, base, truth, out, "IDSSE clean")
        for seed in SEEDS:
            v = degrade(base, d["ball"][:, 2], np.random.default_rng(seed), fill=True)
            raw = dict(v, ball=np.where(v["filled"][:, None], np.nan, v["ball"]), filled=np.zeros_like(v["filled"]))
            both(dead, team, log_t, v, raw, truth, out, "IDSSE degraded")
        truths.setdefault("IDSSE", []).append(int(np.sum((truth[1:] != truth[:-1]) & np.isin(truth[1:], ["H", "A"]))))
    for match in sc.match_ids():
        d = np.load(os.path.join(sc.DATA_DIR, match + ".npz"))
        v, truth = sc_view(d, "detected")
        v.pop("has")
        view = dict(v, team=d["team"], pitch=tuple(d["pitch"].tolist()))
        both(dead, team, log_t, view, view, truth, out, "SkillCorner")
        truths.setdefault("SkillCorner", []).append(int(np.sum((truth[1:] != truth[:-1]) & np.isin(truth[1:], ["H", "A"]))))
    print("R true possessions per match: IDSSE {:.0f}, SkillCorner {:.0f}".format(
        np.mean(truths["IDSSE"]), np.mean(truths["SkillCorner"])))
    print("R {:<15} {:<22} {:>8} {:>8} {:>6}".format("data", "method", "spellF1", "macroF1", "poss"))
    for (key, method), rows in out.items():
        r = np.mean(rows, axis=0)
        print("R {:<15} {:<22} {:>8.3f} {:>8.3f} {:>6.0f}".format(key, method, *r))


if __name__ == "__main__":
    main()
