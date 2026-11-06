"""Make clean optical tracking look like automatic tracking from a single camera.

IDSSE has every player and the ball in every frame; automatic tracking loses the ball in crowded
and aerial moments, drops players and is noisy. A model trained on the clean data would lean on
signals that are not there at inference, so training data is degraded first.

Strengths are drawn at random over WIDE ranges, per stretch of play, and are not fitted to the
provided match: a number measured on that match must not set a parameter of the pipeline. The
aim is a model that holds up over the whole range, not one tuned to any level.
"""
import numpy as np

from external.features import FPS

CHUNK_S = 300                      # a new level of degradation every 5 minutes
LOST_SHARE = (0.1, 0.7)            # share of frames the ball tends to be lost for
GAP_MEDIAN_S = (0.5, 6.0)
CROWD_M, CROWD_GAIN = 3.0, 3.0     # gaps start this much more readily with 3+ players around
AIR_M, AIR_GAIN = 1.5, 4.0         # ... and with the ball above this height
BALL_NOISE_M = (0.1, 1.0)
PLAYER_NOISE_M = (0.05, 0.5)
PLAYER_LOST = (0.0, 0.25)          # share of the time a player is missing
PLAYER_GAP_S = (1.0, 20.0)
# Detector fill of the lost frames. SoccerTrack's panorama sits at the low end of FILL_SHARE; the
# ranges stay wide because a tactical camera is easier and how much easier cannot be measured here.
FILL_SHARE = (0.05, 0.6)           # share of the lost frames that get filled
FILL_RUN_S = (0.2, 2.0)
FILL_RIGHT = (0.6, 0.95)           # share of filled runs that follow the real ball
FILL_NOISE_M = (0.3, 2.0)
FILL_WRONG_M = (5.0, 25.0)         # a run on a head or a sock: this far from the ball


def smooth_noise(rng, n, sigma, seconds=0.4):
    """Gaussian noise with about `seconds` of memory, so tracks wobble instead of jittering."""
    k = max(int(seconds * FPS), 1)
    walk = np.cumsum(rng.standard_normal((n + k, 2)), axis=0)
    return sigma * (walk[k:] - walk[:-k]) / np.sqrt(k)


def drop_gaps(rng, weights, share, median_s):
    """(n,) bool, True where lost: gaps of lognormal length whose starts follow `weights`."""
    n = len(weights)
    lost = np.zeros(n, dtype=bool)
    mean_len = median_s * FPS * np.exp(0.5)   # lognormal with sigma 1
    n_gaps = int(share * n / mean_len)
    if n_gaps == 0:
        return lost
    starts = rng.choice(n, size=n_gaps, p=weights / weights.sum())
    lengths = rng.lognormal(np.log(median_s * FPS), 1.0, size=n_gaps).astype(int) + 1
    for s, g in zip(starts, lengths):
        lost[s:s + g] = True
    return lost


def fill_lost(rng, truth, lost):
    """(positions, filled): what a detector would put in some of the lost frames of one chunk."""
    filled = np.zeros(len(lost), dtype=bool)
    positions = np.full(truth.shape, np.nan)
    if not lost.any():
        return positions, filled
    share = rng.uniform(*FILL_SHARE) * lost.mean()
    picked = drop_gaps(rng, lost.astype(float), share, rng.uniform(*FILL_RUN_S)) & lost
    right, noise = rng.uniform(*FILL_RIGHT), rng.uniform(*FILL_NOISE_M)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], picked.astype(np.int8), [0]])))
    for a, b in zip(edges[::2], edges[1::2]):
        positions[a:b] = truth[a:b] + noise * rng.standard_normal((b - a, 2))
        if rng.uniform() > right:
            angle = rng.uniform(0, 2 * np.pi)
            positions[a:b] += rng.uniform(*FILL_WRONG_M) * np.array([np.cos(angle), np.sin(angle)])
        filled[a:b] = True
    return positions, filled


def degrade(view, ball_z, rng, fill=False):
    """A degraded copy of a view. ball_z is only used to decide when the ball gets lost.

    With fill, some lost frames get a detector-like ball, flagged in the view's "filled".
    """
    players = view["players"].copy()
    ball = view["ball"].copy()
    n, slots = players.shape[:2]
    crowd = (np.linalg.norm(players - ball[:, None, :], axis=2) < CROWD_M).sum(axis=1)
    weights = 1.0 + CROWD_GAIN * (crowd >= 3) + AIR_GAIN * (ball_z > AIR_M)
    chunk = CHUNK_S * FPS
    filled = np.zeros(n, dtype=bool)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        m = b - a
        lost = drop_gaps(rng, weights[a:b], rng.uniform(*LOST_SHARE), rng.uniform(*GAP_MEDIAN_S))
        truth = ball[a:b].copy()
        ball[a:b] += smooth_noise(rng, m, rng.uniform(*BALL_NOISE_M))
        ball[a:b][lost] = np.nan
        if fill:
            positions, filled[a:b] = fill_lost(rng, truth, lost)
            ball[a:b][filled[a:b]] = positions[filled[a:b]]
        sigma = rng.uniform(*PLAYER_NOISE_M)
        share = rng.uniform(*PLAYER_LOST)
        for s in range(slots):
            players[a:b, s] += smooth_noise(rng, m, sigma)
            gone = drop_gaps(rng, np.ones(m), share, rng.uniform(*PLAYER_GAP_S))
            players[a:b, s][gone] = np.nan
    return dict(view, players=players, ball=ball, filled=filled)
