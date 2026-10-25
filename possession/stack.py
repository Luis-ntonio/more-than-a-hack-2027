"""A broadcast-only model and a judge that combines it with the tactical one, scored out of fold.

    docker compose run --rm dev python -m possession.broadcast     # alignment, once
    docker compose run --rm dev python -m possession.stack [--seed N]

The broadcast sees play from another angle, with its own tracking (players and ball from the TV
camera), mapped onto the tactical timeline by possession/broadcast.py. Three models:

  tactical   the pre-labeller (possession/prelabel.py), its default features
  broadcast  the same two LightGBM models (dead / alive, home / away), on what the broadcast
             shows: its ball-to-team distances, ball and player counts, its state (live, close-up,
             ...) and window summaries. NaN where the broadcast is not live.
  judge      two LightGBM models on both models' probabilities plus context -- broadcast state and
             match cost, how contested the ball is in each view, both probabilities' 1 s and 3 s
             means -- so it can learn where to trust which view.

Leave-one-block-out: for each held-out block k, tactical and broadcast models are trained on the
other blocks; the judge is trained on the other blocks' out-of-fold probabilities (each block's
from the base models that did not see it) and applied with the fold-k ones. Every variant is
decoded with v1's HMM. (The base models that produced block j's out-of-fold probabilities did see
block k, so the judge's training inputs are not fully independent of k: a small optimism.)
Prints spellF1, macroF1 and possessions per variant. Aggregates only.
"""
import argparse

import lightgbm as lgb
import numpy as np

from possession import broadcast as bc
from possession import features as feats
from possession import prelabel as pl
from possession.io import AWAY, FPS, GAME, GAMES, HOME, PITCH, load_atd
from possession.models import V1_PARAMS, _apply_prekickoff_rule, decode

WINDOWS_S = (0.5, 2.0, 5.0)
JUDGE_ROUNDS = 200
# the judge sees few, strong inputs and only 10 blocks to learn from: keep it simple
JUDGE_PARAMS = {"num_leaves": 8, "min_child_samples": 200, "learning_rate": 0.03, "lambda_l2": 5.0}


def broadcast_view(n_frames):
    """(n_frames, k) matrix of what the live broadcast shows at each tactical frame, and names."""
    z = np.load(bc.ALIGN_PATH)
    live = z["live"][:n_frames]
    j = z["bc_frame"][:n_frames]
    b = load_atd(source="broadcast")
    scale = np.array(PITCH, dtype=np.float64)
    players = np.full((n_frames,) + b["players"].shape[1:2] + (2,), np.nan)
    ball = np.full((n_frames, 2), np.nan)
    players[live] = b["players"][j[live], :, :2].astype(np.float64) * scale
    ball[live] = b["ball"][j[live]].astype(np.float64) * scale
    dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    d = {t: dist[:, b["team"] == t].min(axis=1) for t in (HOME, AWAY)}
    for t in d:
        d[t][~np.isfinite(d[t])] = np.nan
    cols = dict(bc.features(n_frames))
    cols["bcv_live"] = live.astype(np.float64)
    cols["bcv_ball_visible"] = np.isfinite(ball[:, 0]).astype(np.float64)
    cols["bcv_players"] = np.where(live, np.isfinite(players[:, :, 0]).sum(axis=1), np.nan)
    cols["bcv_d_home"] = np.minimum(d[HOME], 60.0)
    cols["bcv_d_away"] = np.minimum(d[AWAY], 60.0)
    cols["bcv_diff"] = cols["bcv_d_away"] - cols["bcv_d_home"]
    cols["bcv_ball_x"], cols["bcv_ball_y"] = ball[:, 0], ball[:, 1]
    step = np.linalg.norm(np.diff(ball, axis=0, prepend=ball[:1]), axis=1) * FPS
    cols["bcv_ball_speed"] = np.where(step < 50, step, np.nan)
    for name in ("bcv_ball_visible", "bcv_diff", "bcv_ball_speed", "bcv_live"):
        for w in WINDOWS_S:
            cols["{}_m{}".format(name, w)] = feats.rolling_nanmean(cols[name], int(w * FPS))
    names = list(cols)
    return np.column_stack([cols[k] for k in names]).astype(np.float32), names


def smooth(p, seconds):
    return feats.rolling_nanmean(p, int(seconds * FPS))


def judge_inputs(tac, brd, context):
    """Per-frame judge features from (p_dead, p_home) of both views and context columns."""
    cols = [tac[0], tac[1], brd[0], brd[1]]
    for p in (tac[0], tac[1], brd[0], brd[1]):
        cols += [smooth(p, 1.0), smooth(p, 3.0)]
    cols += [tac[1] - brd[1], tac[0] - brd[0]]
    return np.column_stack(cols + context).astype(np.float32)


def fit_pair(X, rows, y_dead, y_home):
    params = dict(pl.LGB_PARAMS, **JUDGE_PARAMS)
    dead = lgb.train(params, lgb.Dataset(X[rows], y_dead.astype(int)), JUDGE_ROUNDS)
    alive = ~y_dead
    team = lgb.train(params, lgb.Dataset(X[rows[alive]], y_home[alive].astype(int)), JUDGE_ROUNDS)
    return dead, team


def score(blocks, k, labels_full):
    _, first, last, truth = blocks[k]
    pred = labels_full[first:last + 1]
    r = pl.pm.evaluate(pl.pm.label_codes(truth), pl.pm.label_codes(pred), bootstrap_n=0)
    started = int(np.sum((pred[1:] != pred[:-1]) & np.isin(pred[1:], ["H", "A"])))
    return r["spell_f1_mean"], r["macro_f1"], started


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    pl.LGB_PARAMS["seed"] = args.seed

    n = GAMES[GAME]["n_video_frames"]
    X_tac, _ = pl.frame_features()
    X_bc, bc_names = broadcast_view(n)
    blocks = pl.training_blocks()
    print("{} blocks; broadcast view: {} columns".format(len(blocks), len(bc_names)))

    # base models, one pair per held-out block, probabilities on every frame
    tac_p, bc_p = [], []
    for k in range(len(blocks)):
        others = blocks[:k] + blocks[k + 1:]
        tac_p.append(pl.probabilities(pl.fit(X_tac, others), X_tac, n))
        bc_p.append(pl.probabilities(pl.fit(X_bc, others), X_bc, n))

    # context for the judge: broadcast state and cost, how contested the ball is in each view
    st = bc.states(n).astype(np.float64)
    cost = np.nan_to_num(np.load(bc.ALIGN_PATH)["cost"][:n], nan=20.0)
    rich = pl.feats.compute(load_atd(enriched=True), pl.load_homography())
    with np.errstate(invalid="ignore"):
        contested_tac = np.clip(np.abs(np.nan_to_num(rich["d_away"] - rich["d_home"], nan=30.0)), 0, 30)
    contested_bc = np.clip(np.abs(np.nan_to_num(X_bc[:, bc_names.index("bcv_diff")], nan=30.0)), 0, 30)
    context = [st, cost, contested_tac, contested_bc]

    variants = {"tactical": [], "broadcast": [], "mean of both": [], "judge": []}
    for k in range(len(blocks)):
        # judge training rows: other blocks, inputs from the base models that did not see them
        Xj, yd, yh = [], [], []
        for j in range(len(blocks)):
            if j == k:
                continue
            _, first, last, truth = blocks[j]
            idx = np.arange(first, last + 1)[::pl.TRAIN_STRIDE]
            lab = truth[::pl.TRAIN_STRIDE]
            keep = lab != "X"
            Z = judge_inputs(tac_p[j], bc_p[j], context)
            Xj.append(Z[idx[keep]])
            yd.append(lab[keep] == "D")
            yh.append(lab[keep] == "H")
        Xj, yd, yh = np.concatenate(Xj), np.concatenate(yd), np.concatenate(yh)
        judge = fit_pair(Xj, np.arange(len(Xj)), yd, yh)
        Zk = judge_inputs(tac_p[k], bc_p[k], context)
        p_judge = (judge[0].predict(Zk), judge[1].predict(Zk))
        for name, (pd_, ph) in (("tactical", tac_p[k]), ("broadcast", bc_p[k]),
                                ("mean of both", ((tac_p[k][0] + bc_p[k][0]) / 2,
                                                  (tac_p[k][1] + bc_p[k][1]) / 2)),
                                ("judge", p_judge)):
            labels = _apply_prekickoff_rule(decode(pd_, ph, n, V1_PARAMS))
            variants[name].append(score(blocks, k, labels))

    truth_poss = sum(int(np.sum((t[1:] != t[:-1]) & np.isin(t[1:], ["H", "A"]))) for _, _, _, t in blocks)
    print("seed {}; mean over blocks (spellF1, macroF1), possessions started (truth {}):".format(
        args.seed, truth_poss))
    for name, rows in variants.items():
        rows = np.array(rows)
        print("  {:<14} spellF1 {:.3f}  macroF1 {:.3f}  possessions {:>4}".format(
            name, rows[:, 0].mean(), rows[:, 1].mean(), int(rows[:, 2].sum())))
    print("  per block spellF1, judge minus tactical: " + " ".join(
        "{:+.2f}".format(a[0] - b[0]) for a, b in zip(variants["judge"], variants["tactical"])))


if __name__ == "__main__":
    main()
