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


def degrade(view, ball_z, rng):
    """A degraded copy of a view. ball_z is only used to decide when the ball gets lost."""
    players = view["players"].copy()
    ball = view["ball"].copy()
    n, slots = players.shape[:2]
    crowd = (np.linalg.norm(players - ball[:, None, :], axis=2) < CROWD_M).sum(axis=1)
    weights = 1.0 + CROWD_GAIN * (crowd >= 3) + AIR_GAIN * (ball_z > AIR_M)
    chunk = CHUNK_S * FPS
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        m = b - a
        lost = drop_gaps(rng, weights[a:b], rng.uniform(*LOST_SHARE), rng.uniform(*GAP_MEDIAN_S))
        ball[a:b] += smooth_noise(rng, m, rng.uniform(*BALL_NOISE_M))
        ball[a:b][lost] = np.nan
        sigma = rng.uniform(*PLAYER_NOISE_M)
        share = rng.uniform(*PLAYER_LOST)
        for s in range(slots):
            players[a:b, s] += smooth_noise(rng, m, sigma)
            gone = drop_gaps(rng, np.ones(m), share, rng.uniform(*PLAYER_GAP_S))
            players[a:b, s][gone] = np.nan
    return dict(view, players=players, ball=ball)
