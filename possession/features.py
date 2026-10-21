"""Per-frame features. All distances in metres, all windows centred, all NaN-aware."""
import numpy as np

from possession.io import FPS, PITCH, HOME, AWAY

JUMP_M = 2.0  # a ball moving more than this in one frame is > 50 m/s: a detection glitch


def rolling_nanmean(values, half_width):
    """Centred mean over +/- half_width frames, ignoring NaN. NaN where the window is empty."""
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    csum = np.concatenate(([0.0], np.cumsum(filled)))
    ccount = np.concatenate(([0], np.cumsum(finite)))
    index = np.arange(len(values))
    low = np.maximum(0, index - half_width)
    high = np.minimum(len(values), index + half_width + 1)
    count = ccount[high] - ccount[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, (csum[high] - csum[low]) / np.maximum(count, 1), np.nan)


def run_lengths(mask):
    """For every frame, the length of the run of True it belongs to (0 where False)."""
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    out = np.zeros(len(mask), dtype=np.int64)
    for start, stop in zip(starts, stops):
        out[start:stop] = stop - start
    return out


def clean_ball(ball_m):
    """Drop isolated spikes: a visible frame far from both its neighbours is a glitch."""
    ball = ball_m.copy()
    step = np.linalg.norm(np.diff(ball, axis=0), axis=1)
    jump_in = np.concatenate(([False], step > JUMP_M))
    jump_out = np.concatenate((step > JUMP_M, [False]))
    ball[jump_in & jump_out] = np.nan
    return ball


def compute(atd, homography):
    """Dict of per-frame feature arrays used by the models."""
    length, width = PITCH
    scale = np.array([length, width], dtype=np.float64)
    players = atd["players"][:, :, :2].astype(np.float64) * scale
    speed = atd["players"][:, :, 2].astype(np.float64)
    ball = clean_ball(atd["ball"].astype(np.float64) * scale)
    team = atd["team"]

    visible = np.isfinite(ball[:, 0])
    dist = np.linalg.norm(players - ball[:, None, :], axis=2)  # NaN where either is missing
    dist = np.where(np.isfinite(dist), dist, np.inf)
    d_home = dist[:, team == HOME].min(axis=1)
    d_away = dist[:, team == AWAY].min(axis=1)

    finite = np.isfinite(speed)
    n_speed = finite.sum(axis=1)
    mean_speed = np.where(n_speed > 0, np.where(finite, speed, 0.0).sum(axis=1) / np.maximum(n_speed, 1),
                          np.nan)
    pan = np.abs(np.diff(homography[:, 2], prepend=homography[0, 2]))

    out_of_pitch = visible & ((ball[:, 0] < -1) | (ball[:, 0] > length + 1)
                              | (ball[:, 1] < -1) | (ball[:, 1] > width + 1))

    return {
        "ball_visible": visible,
        "d_home": d_home,
        "d_away": d_away,
        "gap_len_s": run_lengths(~visible) / FPS,
        "speed_2s": rolling_nanmean(mean_speed, 2 * FPS),
        "pan_2s": rolling_nanmean(pan, 2 * FPS),
        "out_of_pitch": out_of_pitch,
    }
