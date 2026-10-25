"""Ball touches from the video detector's ball, which is not smoothed like the ATD's.

A touch is a frame where the ball turns by more than ANGLE_DEG or changes speed by more than
DSPEED_MS between the WINDOW frames before and after it, with a player within RADIUS_M; its team
is the nearest player's. A touch is confirmed when the next touch within CONFIRM_S is by the same
team: control rather than a deflection. analysis/touch_signal.py measures how these line up with
possession changes (docs/05, section 11).

The detector's ball needs enrich.detect_ball --mode missing and --mode visible plus enrich.merge.
Those detections are a fixed artefact of the submission (GPU inference is not regenerated): this
module only reads them.
"""
import os

import numpy as np

from enrich.geometry import pixels_to_pitch
from possession.io import AWAY, CACHE_DIR, FPS, GAME, HOME, PITCH, load_atd, load_homography

ENRICHED_DIR = os.path.join(CACHE_DIR, "enriched")
VISIBLE_DETECTIONS = os.path.join(ENRICHED_DIR, "{}_ball_detections_visible.npz".format(GAME))
WINDOW = 5          # frames each side for the before / after velocity
ANGLE_DEG = 35.0
DSPEED_MS = 3.0
RADIUS_M = 2.5
CONFIRM_S = 3
MATCH_M = 3.0       # a detector candidate this close to the ATD ball is taken as the ball


def available():
    return os.path.isfile(VISIBLE_DETECTIONS)


def detected_ball(tracking_m):
    """The detector's ball in metres: where the ATD has a ball (tracking_m, metres), the candidate
    nearest to it within MATCH_M; where it has none, the ball merge.py took from the detector.
    NaN elsewhere."""
    scale = np.array(PITCH, dtype=np.float64)
    homography = load_homography()
    ball = np.full_like(tracking_m, np.nan)
    z = np.load(VISIBLE_DETECTIONS)
    for frame, cands in zip(z["frames"], z["det"]):
        cands = cands[np.isfinite(cands[:, 2])]
        if not len(cands) or not np.isfinite(tracking_m[frame, 0]):
            continue
        x, y = pixels_to_pitch(homography[frame], cands[:, 0], cands[:, 1])
        dist = np.linalg.norm(np.stack([x, y], axis=1) * scale - tracking_m[frame], axis=1)
        if np.isfinite(dist).any() and np.nanmin(dist) < MATCH_M:
            best = int(np.nanargmin(dist))
            ball[frame] = np.array([x[best], y[best]]) * scale
    rich = load_atd(enriched=True)
    from_detector = rich["ball_source"] == 2
    ball[from_detector] = rich["ball"][from_detector].astype(np.float64) * scale
    return ball


def team_distances(ball, atd=None):
    """{HOME: d, AWAY: d}: per-frame distance in metres from the ball to each team's nearest player."""
    atd = load_atd() if atd is None else atd
    players = atd["players"][:, :, :2].astype(np.float64) * np.array(PITCH, dtype=np.float64)
    dist = np.linalg.norm(players - ball[:, None, :], axis=2)
    dist = np.where(np.isfinite(dist), dist, np.inf)
    d = {}
    for team in (HOME, AWAY):
        d[team] = dist[:, atd["team"] == team].min(axis=1)
        d[team][~np.isfinite(d[team])] = np.nan
    return d


def touches(ball, d, angle=ANGLE_DEG, dspeed=DSPEED_MS, radius=RADIUS_M):
    """Per-frame touch team (HOME, AWAY, or -1), at most one touch per WINDOW frames."""
    n = len(ball)
    before = np.full((n, 2), np.nan)
    after = np.full((n, 2), np.nan)
    before[WINDOW:] = (ball[WINDOW:] - ball[:-WINDOW]) * FPS / WINDOW
    after[:-WINDOW] = (ball[WINDOW:] - ball[:-WINDOW]) * FPS / WINDOW
    sb, sa = np.linalg.norm(before, axis=1), np.linalg.norm(after, axis=1)
    cos = np.einsum("ij,ij->i", before, after) / np.maximum(sb * sa, 1e-9)
    turn = (sb > 1.0) & (sa > 1.0) & (cos < np.cos(np.radians(angle)))
    kick = np.abs(sa - sb) > dspeed
    d_home, d_away = d[HOME], d[AWAY]
    event = (turn | kick) & (np.fmin(d_home, d_away) < radius)
    team = np.where(event, np.where(d_home <= d_away, HOME, AWAY), -1)
    # keep the first frame of each run of events
    keep = event & ~np.concatenate(([False], event[:-1]))
    last = -WINDOW
    for i in np.flatnonzero(keep):
        if i - last < WINDOW:
            keep[i] = False
        else:
            last = i
    return np.where(keep, team, -1)


def confirmed(team):
    """Touches whose next touch within CONFIRM_S is by the same team; -1 elsewhere."""
    out = np.full_like(team, -1)
    idx = np.flatnonzero(team >= 0)
    for i, j in zip(idx[:-1], idx[1:]):
        if j - i <= CONFIRM_S * FPS and team[j] == team[i]:
            out[i] = team[i]
    return out


def last_next(events):
    """Signed team (+1 home, -1 away, 0 none) of the last / next event, and seconds since / until it,
    capped at 60."""
    n = len(events)
    frames = np.arange(n)
    sign = np.where(events == HOME, 1.0, np.where(events == AWAY, -1.0, 0.0))
    has = events >= 0
    last = np.maximum.accumulate(np.where(has, frames, -1))
    nxt = np.minimum.accumulate(np.where(has, frames, n)[::-1])[::-1]
    return (np.where(last >= 0, sign[np.maximum(last, 0)], 0.0),
            np.where(last >= 0, np.minimum((frames - last) / FPS, 60.0), 60.0),
            np.where(nxt < n, sign[np.minimum(nxt, n - 1)], 0.0),
            np.where(nxt < n, np.minimum((nxt - frames) / FPS, 60.0), 60.0))


def features(tracking_m):
    """Per-frame touch features from the detector's ball, as {name: array}."""
    ball = detected_ball(tracking_m)
    team = touches(ball, team_distances(ball))
    out = {}
    for name, events in (("touch", team), ("ctouch", confirmed(team))):
        last, since, nxt, until = last_next(events)
        out[name + "_last"], out[name + "_since_s"] = last, since
        out[name + "_next"], out[name + "_until_s"] = nxt, until
        out[name + "_agree"] = last * nxt
    # a confirmed touch by the other team than the confirmed touch before it: a candidate change
    c = confirmed(team)
    idx = np.flatnonzero(c >= 0)
    switch = np.full_like(c, -1)
    switch[idx[1:][c[idx[1:]] != c[idx[:-1]]]] = c[idx[1:][c[idx[1:]] != c[idx[:-1]]]]
    last, since, nxt, until = last_next(switch)
    out["cswitch_since_s"], out["cswitch_until_s"] = since, until
    out["cswitch_last"], out["cswitch_next"] = last, nxt
    return out
