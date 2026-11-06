"""Per-frame features that exist both in external tracking and in the provided match.

Everything works on a common view, in metres with the pitch centred on (0, 0):
  players (n, slots, 2), NaN when absent; team (slots,), 0 home, 1 away
  ball (n, 2), NaN when lost; half (n,); pitch (length, width)
  filled (n,), optional: True where the ball came from a detector, not from the tracking
No height, no camera, no tagging: a model trained on external data cannot see those. Features are
written from the home team's point of view, and x runs towards the goal home attacks in that half,
so swapping the teams (swap_teams) is a valid augmentation. Radii and windows are physical choices.
"""
import warnings

import numpy as np

FPS = 25
WINDOWS_S = (0.5, 1.0, 2.0, 5.0)
NEAR_M = (3.0, 6.0)
TOUCH_M = 1.5     # about a stride: a player this close can play the ball
CAP_M = 40.0
CAP_S = 30.0
MAX_BALL_MS = 40.0


def rolling_mean(values, half):
    """Centred nan-mean over +-half frames; NaN where the window has no value."""
    finite = np.isfinite(values)
    total = np.concatenate([[0.0], np.cumsum(np.where(finite, values, 0.0))])
    count = np.concatenate([[0], np.cumsum(finite)])
    i = np.arange(len(values))
    lo, hi = np.maximum(i - half, 0), np.minimum(i + half + 1, len(values))
    n = count[hi] - count[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[hi] - total[lo]) / n, np.nan)


def attack_sign(view):
    """(n,) +1 where home attacks towards +x. A team's mean x over a half is on its own side."""
    sign = np.ones(len(view["ball"]))
    home = view["team"] == 0
    for half in np.unique(view["half"]):
        rows = view["half"] == half
        x = view["players"][rows, :, 0]
        sign[rows] = 1.0 if np.nanmean(x[:, home]) < np.nanmean(x[:, ~home]) else -1.0
    return sign


def run_info(mask):
    """Seconds since the run of True each frame is in started, until it ends, and its length."""
    n = len(mask)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    since, until, length = np.zeros(n), np.zeros(n), np.zeros(n)
    for a, b in zip(edges[::2], edges[1::2]):
        since[a:b] = np.arange(b - a) / FPS
        until[a:b] = np.arange(b - a, 0, -1) / FPS
        length[a:b] = (b - a) / FPS
    return np.minimum(since, CAP_S), np.minimum(until, CAP_S), np.minimum(length, CAP_S)


def last_next(team_at):
    """Signed team (+1 home, -1 away, 0 none) of the last and next touch, and seconds to each."""
    n = len(team_at)
    frames = np.arange(n)
    has = team_at != 0
    last = np.maximum.accumulate(np.where(has, frames, -1))
    nxt = np.minimum.accumulate(np.where(has, frames, n)[::-1])[::-1]
    return (np.where(last >= 0, team_at[np.maximum(last, 0)], 0.0),
            np.where(last >= 0, np.minimum((frames - last) / FPS, CAP_S), CAP_S),
            np.where(nxt < n, team_at[np.minimum(nxt, n - 1)], 0.0),
            np.where(nxt < n, np.minimum((nxt - frames) / FPS, CAP_S), CAP_S))


def compute(view):
    """(n, k) float32 matrix and its column names."""
    players, team, ball = view["players"], view["team"], view["ball"]
    length, width = view["pitch"]
    sign = attack_sign(view)
    visible = np.isfinite(ball[:, 0])
    dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    d = {}
    near = {}
    for t in (0, 1):
        mine = np.where(np.isfinite(dist[:, team == t]), dist[:, team == t], np.inf)
        d[t] = np.where(visible, np.minimum(mine.min(axis=1), CAP_M), np.nan)
        for r in NEAR_M:
            near[t, r] = np.where(visible, (mine < r).sum(axis=1), np.nan)

    cols = {}
    cols["visible"] = visible.astype(float)
    cols["filled"] = view.get("filled", np.zeros(len(ball), dtype=bool)).astype(float)
    cols["gap_since_s"], cols["gap_until_s"], cols["gap_len_s"] = run_info(~visible)
    cols["d_home"], cols["d_away"] = d[0], d[1]
    cols["d_diff"] = d[1] - d[0]
    cols["d_min"] = np.fmin(d[0], d[1])
    for r in NEAR_M:
        cols["near_diff_{}".format(r)] = near[0, r] - near[1, r]
        cols["near_sum_{}".format(r)] = near[0, r] + near[1, r]

    x, y = ball[:, 0] * sign, ball[:, 1]
    cols["ball_x"], cols["ball_abs_y"] = x, np.abs(y)
    cols["edge_m"] = np.minimum(length / 2 - np.abs(x), width / 2 - np.abs(y))
    step = np.linalg.norm(np.diff(ball, axis=0, prepend=ball[:1]), axis=1) * FPS
    speed = np.where(step < MAX_BALL_MS, rolling_mean(step, 2), np.nan)
    cols["ball_speed"] = speed
    cols["ball_accel"] = np.abs(np.diff(speed, prepend=speed[:1])) * FPS
    v = np.diff(ball, axis=0, prepend=ball[:1])
    before, after = np.roll(v, 5, axis=0), np.roll(v, -5, axis=0)
    cos = (before * after).sum(axis=1) / np.maximum(
        np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1), 1e-9)
    cols["ball_turn"] = np.clip(cos, -1, 1)

    home = team == 0
    px = players[:, :, 0] * sign[:, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cx_home, cx_away = np.nanmean(px[:, home], axis=1), np.nanmean(px[:, ~home], axis=1)
        moved = np.linalg.norm(np.diff(players, axis=0, prepend=players[:1]), axis=2) * FPS
        pace = np.nanmean(np.where(moved < 12, moved, np.nan), axis=1)
    cols["cx_home"], cols["cx_away"], cols["cx_diff"] = cx_home, cx_away, cx_home - cx_away
    cols["pace"] = rolling_mean(pace, FPS)
    cols["n_players"] = np.isfinite(players[:, :, 0]).sum(axis=1).astype(float)

    touch = np.where(cols["d_min"] < TOUCH_M, np.where(d[0] <= d[1], 1.0, -1.0), 0.0)
    (cols["touch_last"], cols["touch_since_s"], cols["touch_next"],
     cols["touch_until_s"]) = last_next(touch)
    cols["touch_agree"] = cols["touch_last"] * cols["touch_next"]

    for name in ("visible", "d_diff", "d_min", "near_diff_3.0", "near_sum_3.0", "ball_speed",
                 "ball_x", "edge_m", "pace", "cx_diff"):
        for w in WINDOWS_S:
            half = int(w * FPS)
            mean = rolling_mean(cols[name], half)
            cols["{}_m{}".format(name, w)] = mean
            cols["{}_past{}".format(name, w)] = np.concatenate([np.full(half, np.nan), mean[:-half]])
            cols["{}_next{}".format(name, w)] = np.concatenate([mean[half:], np.full(half, np.nan)])
    names = list(cols)
    return np.column_stack([cols[k] for k in names]).astype(np.float32), names


def swap_teams(view):
    """The same match with home and away exchanged."""
    return dict(view, team=1 - view["team"])
