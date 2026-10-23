"""What the ATD looks like at the public GT's change frames: where the GT puts a change relative to
the touch, the kick or the ball leaving play.

    docker compose run --rm dev python analysis/gt_changes.py

For every GT change, offsets are in frames relative to the GT change (negative = before it).
Prints per-change offsets and distances, never frame numbers or positions.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from labeler.server import pm, _public_gt  # noqa: E402
from labeler.timing import kind_of, transitions  # noqa: E402
from possession.features import clean_ball, rolling_nanmean  # noqa: E402
from possession.io import AWAY, FPS, HOME, PITCH, load_atd  # noqa: E402

BEFORE, AFTER = 3 * FPS, 2 * FPS
CLOSE_M = 1.5      # a player this close to the ball is "on it"
KICK_MS = 3.0      # a restart's ball moving faster than this for KICK_RUN frames has been played
KICK_RUN = 3


def per_frame():
    atd = load_atd()
    scale = np.array(PITCH, dtype=np.float64)
    ball = clean_ball(atd["ball"].astype(np.float64) * scale)
    players = atd["players"][:, :, :2].astype(np.float64) * scale
    dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    d = {HOME: dist[:, atd["team"] == HOME].min(axis=1), AWAY: dist[:, atd["team"] == AWAY].min(axis=1)}
    for team in d:
        d[team][~np.isfinite(d[team])] = np.nan
    step = np.linalg.norm(np.diff(ball, axis=0, prepend=ball[:1]), axis=1) * FPS
    visible = np.isfinite(ball[:, 0])
    # nearest team, carried through frames without a ball
    nearest = np.where(visible, np.where(d[HOME] <= d[AWAY], HOME, AWAY), -1)
    carry = np.maximum.accumulate(np.where(nearest >= 0, np.arange(len(nearest)), 0))
    nearest = nearest[carry]
    out = visible & ((ball[:, 0] < 0) | (ball[:, 0] > scale[0]) | (ball[:, 1] < 0) | (ball[:, 1] > scale[1]))
    return {"ball": ball, "d": d, "speed": rolling_nanmean(step, 2), "visible": visible,
            "nearest": nearest, "out": out}


def first_true(mask, lo, offset_of):
    hits = np.flatnonzero(mask)
    return offset_of(lo + hits[0]) if len(hits) else None


def last_true(mask, lo, offset_of):
    hits = np.flatnonzero(mask)
    return offset_of(lo + hits[-1]) if len(hits) else None


def sustained(mask, run):
    """True where mask holds for `run` frames starting there."""
    if len(mask) < run:
        return np.zeros_like(mask)
    window = np.lib.stride_tricks.sliding_window_view(mask, run).all(axis=1)
    return np.concatenate((window, np.zeros(run - 1, dtype=bool)))


def describe(change, f):
    frame, prev, new = change
    lo, hi = frame - BEFORE, frame + AFTER + 1
    rel = lambda i: int(i - frame)  # noqa: E731
    row = {"kind": kind_of(prev, new), "change": "{}>{}".format(prev, new),
           "visible": bool(f["visible"][frame]), "speed": f["speed"][frame]}
    if row["kind"] == "possession":
        n, o = (HOME, AWAY) if new == "H" else (AWAY, HOME)
        d_new, d_old = f["d"][n][lo:hi], f["d"][o][lo:hi]
        row["d_new"], row["d_old"] = f["d"][n][frame], f["d"][o][frame]
        row["new_close"] = first_true(d_new < CLOSE_M, lo, rel)
        row["old_last_close"] = last_true(d_old[:BEFORE + 1] < CLOSE_M, lo, rel)
        row["touch"] = rel(lo + np.nanargmin(np.where(np.isfinite(d_new), d_new, np.inf)))
        row["touch_d"] = np.nanmin(d_new) if np.isfinite(d_new).any() else np.nan
        # start of the current run of "new team nearest", looking back from the change
        near = f["nearest"][lo:frame + 1] == n
        row["nearest_since"] = (rel(lo + len(near) - np.argmin(near[::-1]))
                                if near[-1] and not near.all() else (None if not near[-1] else -BEFORE))
        row["new_close_after_old"] = None
        if row["old_last_close"] is not None:
            start = row["old_last_close"] + BEFORE + 1
            row["new_close_after_old"] = first_true(d_new[start:] < CLOSE_M, lo + start, rel)
    elif row["kind"] == "restart":
        moving = sustained(np.nan_to_num(f["speed"][lo:hi]) > KICK_MS, KICK_RUN)
        row["kick"] = first_true(moving, lo, rel)
        n = HOME if new == "H" else AWAY
        row["d_new"] = f["d"][n][frame]
        spot = np.nanmedian(f["ball"][lo:frame - FPS], axis=0) if np.isfinite(
            f["ball"][lo:frame - FPS, 0]).any() else np.full(2, np.nan)
        row["moved_m"] = float(np.linalg.norm(f["ball"][frame] - spot))
    else:
        row["ball_out"] = first_true(f["out"][lo:hi], lo, rel)
        row["ball_lost"] = first_true(~f["visible"][lo:hi], lo, rel)
    return row


def fmt(value, spec):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-".rjust(len(format(0, spec)))
    return format(value, spec)


def summary(rows, key):
    values = np.array([r[key] for r in rows if r.get(key) is not None
                       and np.isfinite(r[key])], dtype=float)
    if not len(values):
        return "{:<20} n= 0".format(key)
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return "{:<20} n={:2d}  median {:+7.1f}  IQR [{:+6.1f}, {:+6.1f}]".format(key, len(values), med, q1, q3)


def main():
    gt, public_range = _public_gt()
    first, last = public_range
    labels = np.array(pm.LABELS)[gt[first:last + 1]]
    f = per_frame()
    rows = [describe(c, f) for c in transitions(labels, first)]

    print("{} GT changes in the public window ({:.1f} min). Offsets in frames (25 = 1 s), "
          "negative = before the GT change. Distances in m.".format(
              len(rows), (last - first + 1) / FPS / 60))

    poss = [r for r in rows if r["kind"] == "possession"]
    print("\n-- possession changes ({})".format(len(poss)))
    print("  {:<5} {:>4} {:>6} {:>6} {:>6} {:>6} {:>8} {:>8} {:>8} {:>8} {:>8}".format(
        "", "vis", "d_new", "d_old", "speed", "touch", "touch_d", "new<1.5", "old<1.5", "nearest", "new|old"))
    for r in poss:
        print("  {:<5} {:>4} {} {} {} {} {} {} {} {} {}".format(
            r["change"], "y" if r["visible"] else "n", fmt(r["d_new"], "6.1f"), fmt(r["d_old"], "6.1f"),
            fmt(r["speed"], "6.1f"), fmt(r["touch"], "+6d"), fmt(r["touch_d"], "8.1f"),
            fmt(r["new_close"], "+8d"), fmt(r["old_last_close"], "+8d"), fmt(r["nearest_since"], "+8d"),
            fmt(r["new_close_after_old"], "+8d")))
    for key in ("touch", "new_close", "old_last_close", "nearest_since", "new_close_after_old",
                "d_new", "d_old", "speed"):
        print("  " + summary(poss, key))

    restarts = [r for r in rows if r["kind"] == "restart"]
    print("\n-- restarts ({})".format(len(restarts)))
    for r in restarts:
        print("  {:<5} vis {}  kick {}  moved {} m  d_new {}  speed {}".format(
            r["change"], "y" if r["visible"] else "n", fmt(r["kick"], "+4d"), fmt(r["moved_m"], "5.1f"),
            fmt(r["d_new"], "5.1f"), fmt(r["speed"], "5.1f")))
    for key in ("kick", "moved_m", "d_new", "speed"):
        print("  " + summary(restarts, key))

    dead = [r for r in rows if r["kind"] == "dead"]
    print("\n-- ball goes dead ({})".format(len(dead)))
    for r in dead:
        print("  {:<5} vis {}  ball_out {}  ball_lost {}".format(
            r["change"], "y" if r["visible"] else "n", fmt(r["ball_out"], "+4d"), fmt(r["ball_lost"], "+4d")))
    for key in ("ball_out", "ball_lost"):
        print("  " + summary(dead, key))

    print("\nlegend: touch = frame the new team gets closest to the ball (-3 s..+2 s); new<1.5 = new "
          "team first within 1.5 m; old<1.5 = old team last within 1.5 m before the change; nearest = "
          "start of the run where the new team is nearest; new|old = new team within 1.5 m after the "
          "old team let go; kick = ball first moving > {} m/s for {} frames".format(KICK_MS, KICK_RUN))


if __name__ == "__main__":
    main()
