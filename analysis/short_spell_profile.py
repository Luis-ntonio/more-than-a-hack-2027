"""What the ball does in short possessions, against stretches with no change: is there a signal?

    docker compose run --rm dev python analysis/short_spell_profile.py

For every possession spell (H or A) that follows one of the other team, in the public GT and every
finished hand-labelled block (on the GT's timing), it measures the ball over the spell from the
point of view of the team that gets it (the "new" team):
  adv_m       how far the ball moves towards the goal the new team attacks
  vmax        top ball speed (m/s); kick0 = top speed in its first 0.5 s
  dmin_new    closest the new team's nearest player gets to the ball (m)
  near_new    share of frames the new team is nearer to the ball than the old one
  start_x     where it starts, in the new team's attacking direction (0 own goal line, 105 the other)
The same is measured on control windows: random stretches of the same lengths inside long spells
(over 8 s, at least 2 s from either end), with the team *not* in possession as the "new" one.
AUC is how well each measure tells a real short spell from a control window (0.5 = not at all).
Also prints what follows a short spell. Aggregates only. Attacking directions come from the median
position of each team's goalkeeper slot in each half.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import features as feats  # noqa: E402
from possession.io import AWAY, FPS, HOME, PITCH, graded_blocks, load_atd  # noqa: E402
from possession.prelabel import training_blocks  # noqa: E402

GK_SLOT = {HOME: 23, AWAY: 45}   # 0-based ATD slots of the two goalkeepers (DATA.md)
GROUPS_S = ((0, 2), (2, 4), (4, 8))
LONG_S = 8
EDGE_S = 2
SEED = 0


def code(label):
    return {"H": HOME, "A": AWAY}.get(label, -1)


def attack_sign():
    """{(half, team): +1 if that team attacks towards x = 105 in that half, else -1}."""
    atd = load_atd()
    out = {}
    for half, first, last in graded_blocks():
        for team, slot in GK_SLOT.items():
            x = np.nanmedian(atd["players"][first:last + 1, slot, 0])
            out[(half, team)] = 1.0 if x < 0.5 else -1.0
    return out


def measure(f, a, b, new, sign):
    """Measures of frames [a, b) with `new` as the new team (sign: its attacking direction)."""
    ball = f["ball"][a:b]
    seen = np.isfinite(ball[:, 0])
    if seen.sum() < 2:
        return None
    pts = ball[seen]
    speed = f["speed"][a:b]
    old = AWAY if new == HOME else HOME
    d_new, d_old = f["d"][new][a:b], f["d"][old][a:b]
    both = np.isfinite(d_new) & np.isfinite(d_old)
    x0 = pts[0, 0] if sign > 0 else PITCH[0] - pts[0, 0]
    return {
        "adv_m": (pts[-1, 0] - pts[0, 0]) * sign,
        "vmax": np.nanmax(speed) if np.isfinite(speed).any() else np.nan,
        "kick0": np.nanmax(speed[:FPS // 2]) if np.isfinite(speed[:FPS // 2]).any() else np.nan,
        "dmin_new": np.nanmin(d_new) if np.isfinite(d_new).any() else np.nan,
        "near_new": np.mean(d_new[both] < d_old[both]) if both.any() else np.nan,
        "start_x": x0,
    }


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if not len(pos) or not len(neg):
        return np.nan
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def frame_data():
    """Per-frame ball (enriched ATD, metres), each team's nearest-player distance and ball speed."""
    atd = load_atd(enriched=True)
    scale = np.array(PITCH, dtype=np.float64)
    ball = feats.clean_ball(atd["ball"].astype(np.float64) * scale)
    players = atd["players"][:, :, :2].astype(np.float64) * scale
    dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    d = {}
    for team in (HOME, AWAY):
        d[team] = dist[:, atd["team"] == team].min(axis=1)
        d[team][~np.isfinite(d[team])] = np.nan
    step = np.linalg.norm(np.diff(ball, axis=0, prepend=ball[:1]), axis=1) * FPS
    return {"ball": ball, "d": d, "speed": np.where(step < 40, step, np.nan)}


def half_of(frame):
    return next(h for h, a, b in graded_blocks() if a <= frame <= b)


def main():
    f = frame_data()
    signs = attack_sign()
    print("attacking direction (+1 towards x=105): " + ", ".join(
        "{} {} {:+.0f}".format("1st" if h == 1 else "2nd", "home" if t == HOME else "away", s)
        for (h, t), s in sorted(signs.items())))

    rng = np.random.default_rng(SEED)
    short = {g: [] for g in GROUPS_S}
    follow = {g: {"back to old team": 0, "dead ball": 0, "other": 0} for g in GROUPS_S}
    long_spells = []
    for _, first, last, labels in training_blocks():
        lab = np.array([code(x) for x in labels])
        dead = labels == "D"
        edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
        starts = np.concatenate([[0], edges])
        ends = np.concatenate([edges, [len(labels)]])
        for k, (a, b) in enumerate(zip(starts, ends)):
            if lab[a] < 0:
                continue
            length = (b - a) / FPS
            if length >= LONG_S:
                long_spells.append((first + a, first + b, lab[a]))
            if k == 0 or lab[starts[k - 1]] < 0 or lab[starts[k - 1]] == lab[a]:
                continue
            for g in GROUPS_S:
                if g[0] <= length < g[1]:
                    m = measure(f, first + a, first + b, lab[a], signs[(half_of(first + a), lab[a])])
                    if m is not None:
                        short[g].append(m)
                    if b < len(labels):
                        nxt = lab[b] if not dead[b] else None
                        key = ("dead ball" if dead[b] else
                               "back to old team" if nxt == lab[starts[k - 1]] else "other")
                        follow[g][key] += 1
    names = list(next(iter(short.values()))[0])
    for g in GROUPS_S:
        control = []
        lengths = [None] * len(short[g])
        for _ in range(4 * len(short[g])):
            a, b, team = long_spells[rng.integers(len(long_spells))]
            n = int(rng.uniform(*g) * FPS) or 1
            lo, hi = a + EDGE_S * FPS, b - EDGE_S * FPS - n
            if hi <= lo:
                continue
            s = int(rng.integers(lo, hi))
            new = AWAY if team == HOME else HOME
            m = measure(f, s, s + n, new, signs[(half_of(s), new)])
            if m is not None:
                control.append(m)
        print("\npossessions of {}-{} s (n={}) vs {} control windows inside long possessions".format(
            g[0], g[1], len(short[g]), len(control)))
        print("  {:<10} {:>14} {:>14} {:>6}".format("measure", "short: median", "control", "AUC"))
        for name in names:
            pos = [m[name] for m in short[g]]
            neg = [m[name] for m in control]
            print("  {:<10} {:>14.2f} {:>14.2f} {:>6.2f}".format(
                name, np.nanmedian(pos), np.nanmedian(neg), auc(pos, neg)))
        total = max(sum(follow[g].values()), 1)
        print("  what follows: " + ", ".join("{} {:.0%}".format(k, v / total)
                                              for k, v in follow[g].items()))


if __name__ == "__main__":
    main()
