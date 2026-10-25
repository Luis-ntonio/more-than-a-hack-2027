"""Per-frame features from Metrica's Smart Tagging (timed on the tactical video, no alignment needed).

The tagging is automatic, not ground truth, and its intervals are phases of fixed lengths rather
than possessions (docs/02_eda.md), so it goes in as evidence for the model, never as labels.
Measured on the hand-labelled blocks (analysis/smart_tagging_vs_labels.py): the team a code implies
holds the ball 56-99% of its frames depending on the code, a set piece's restart comes 5-12 s after
its start, and the team that restarts is the tagged one.
"""
import numpy as np

from possession.io import AWAY, FPS, HOME, load_smart_tagging

TEAM = {"barcelona": HOME, "atletic club": AWAY}   # BAR-ATH: Barcelona at home
DEFENDING = ("DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD",
             "DEFENDING IN ATTACKING THIRD", "DEFENSIVE TRANSITION")
SET_PIECE_TYPES = {"throw-in": 1, "corner kick": 2, "goal kick": 3, "free kick": 4,
                   "kick off (after goal)": 5, "kick off (start)": 6}
CAP_S = 60.0


def _since_until(starts, n):
    """Seconds since the last start and until the next one, capped at CAP_S."""
    frames = np.arange(n)
    mark = np.zeros(n, dtype=bool)
    mark[starts[(starts >= 0) & (starts < n)]] = True
    last = np.maximum.accumulate(np.where(mark, frames, -1))
    nxt = np.minimum.accumulate(np.where(mark, frames, n)[::-1])[::-1]
    return (np.where(last >= 0, np.minimum((frames - last) / FPS, CAP_S), CAP_S),
            np.where(nxt < n, np.minimum((nxt - frames) / FPS, CAP_S), CAP_S), last)


def features(n_frames):
    """{name: (n_frames,) array}."""
    tags = load_smart_tagging()
    team = tags["Team"].fillna("N/A").str.strip().str.lower().map(TEAM).fillna(-1).astype(int).values
    out = {}
    # one column per code: +1 when it implies home has the ball, -1 away, 0 not active or no team
    for c in sorted(tags["code"].unique()):
        if c in ("SET PIECES", "GOALS"):
            continue
        col = np.zeros(n_frames)
        for i in np.flatnonzero(tags["code"].values == c):
            if team[i] < 0:
                continue
            implied = team[i] if c not in DEFENDING else (AWAY if team[i] == HOME else HOME)
            f0, f1 = max(tags["f0"].values[i], 0), min(tags["f1"].values[i], n_frames)
            col[f0:f1] += 1.0 if implied == HOME else -1.0
        out["st_" + c.lower().replace(" ", "_")] = np.clip(col, -1, 1)

    sp = np.flatnonzero(tags["code"].values == "SET PIECES")
    starts = tags["f0"].values[sp]
    since, until, last = _since_until(starts, n_frames)
    out["st_setpiece_since_s"], out["st_setpiece_until_s"] = since, until
    kind = np.zeros(n_frames)
    side = np.zeros(n_frames)
    by_start = {}
    for i in sp:
        typ = str(tags["Type"].values[i]).strip().lower() if "Type" in tags else ""
        by_start[tags["f0"].values[i]] = (SET_PIECE_TYPES.get(typ, 0),
                                          1.0 if team[i] == HOME else -1.0 if team[i] == AWAY else 0.0)
    has = last >= 0
    lookup = np.array([by_start.get(f, (0, 0.0)) for f in last[has]], dtype=np.float64).reshape(-1, 2)
    kind[has], side[has] = lookup[:, 0], lookup[:, 1]
    out["st_setpiece_type"], out["st_setpiece_team"] = kind, side

    goals = tags["f0"].values[tags["code"].values == "GOALS"]
    out["st_goal_since_s"], out["st_goal_until_s"], _ = _since_until(goals, n_frames)
    return out
