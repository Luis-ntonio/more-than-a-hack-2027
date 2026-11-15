"""Fill the ATD's ball gaps with detector candidates and save an enriched copy under a new path.

    docker compose run --rm dev python -m enrich.merge

Reads the original ATD (never modified) and cache/enriched/<game>_ball_detections.npz, writes
cache/enriched/<game>_atd_enriched.npz with the same arrays as possession.io.load_atd plus
ball_source (per frame: 0 none, 1 ATD, 2 detector, 3 interpolated) and ball_conf. Load it with
load_atd(enriched=True). Prints aggregates only.

A candidate becomes the ball only if it is plausible in time: it must be reachable from the last
accepted position at a physical speed, and accepted positions must form runs of several frames.
That is what removes heads, socks and penalty spots the detector mistakes for a ball.
"""
import argparse
import json
import os

import numpy as np

from enrich.detect_ball import OUT_DIR
from enrich.geometry import pixels_to_pitch
from games import GAME
from possession.io import FPS, PITCH, ROOT, graded_blocks, load_atd, load_homography

SOURCE_NONE, SOURCE_ATD, SOURCE_DETECTOR, SOURCE_INTERP = 0, 1, 2, 3
PARAMS = {
    "conf_min": 0.50,       # below this a candidate is ignored (chosen on SoccerTrack: external.fill_eval)
    "conf_seed": 0.50,      # starting a track with nothing to anchor it; no effect on SoccerTrack
    "max_speed_ms": 40.0,   # a struck ball tops out around 30-35 m/s
    "slack_m": 2.0,         # homography + detection error on top of the speed bound
    "margin_m": 2.0,        # candidates further than this outside the pitch are discarded: a ball
                            # out of the pitch is out of play. Same tolerance as slack_m
    "anchor_s": 1.0,        # an ATD position this recent anchors the start of a gap
    "min_run": 3,           # accepted runs shorter than this are dropped...
    "keep_conf": 0.60,      # ...unless every frame in them is at least this confident (SoccerTrack)
    "interp_max": 5,        # holes up to this many frames between accepted positions are filled
}


def candidates_in_metres(det, frames, homography, p):
    """(n, K, 3) metres x/y + conf; NaN where a candidate is missing, weak or off the pitch."""
    scale = np.array(PITCH, dtype=np.float64)
    out = np.full(det.shape[:2] + (3,), np.nan)
    for i, frame in enumerate(frames):
        x, y = pixels_to_pitch(homography[frame], det[i, :, 0], det[i, :, 1])
        out[i, :, 0], out[i, :, 1], out[i, :, 2] = x * scale[0], y * scale[1], det[i, :, 2]
    m = p["margin_m"]
    bad = ((out[:, :, 2] < p["conf_min"]) | ~np.isfinite(out[:, :, 0])
           | (out[:, :, 0] < -m) | (out[:, :, 0] > scale[0] + m)
           | (out[:, :, 1] < -m) | (out[:, :, 1] > scale[1] + m))
    out[bad] = np.nan
    return out


def track_gap(cands, anchor, anchor_age, p):
    """Greedy forward pass over one gap. Returns (len, 3) accepted x/y/conf, NaN where none."""
    accepted = np.full((len(cands), cands.shape[2]), np.nan)
    last, age = anchor, anchor_age
    for i, frame_cands in enumerate(cands):
        age += 1
        valid = frame_cands[np.isfinite(frame_cands[:, 2])]
        if not len(valid):
            continue
        if last is None:
            best = valid[np.argmax(valid[:, 2])]
            if best[2] < p["conf_seed"]:
                continue
        else:
            reach = p["max_speed_ms"] * age / FPS + p["slack_m"]
            dist = np.hypot(valid[:, 0] - last[0], valid[:, 1] - last[1])
            ok = dist <= reach
            if not ok.any():
                if age > p["anchor_s"] * FPS:      # lost for long enough: allow a fresh seed
                    best = valid[np.argmax(valid[:, 2])]
                    if best[2] < p["conf_seed"]:
                        continue
                else:
                    continue
            else:
                # nearest reachable, with confidence breaking near-ties
                score = dist[ok] / reach - 0.5 * valid[ok, 2]
                best = valid[ok][np.argmin(score)]
        accepted[i] = best
        last, age = best[:2], 0
    return accepted


def drop_short_runs(accepted, p):
    found = np.isfinite(accepted[:, 2])
    edges = np.diff(np.concatenate(([0], found.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if stop - start < p["min_run"] and not np.all(accepted[start:stop, 2] >= p["keep_conf"]):
            accepted[start:stop] = np.nan
    return accepted


def fill_gap(cands, anchor, p, end_anchor=None):
    """Accepted positions in one gap. With p["backward"] and an anchor after the gap, the tracker
    also runs backwards from it, and each frame keeps the pass whose anchor is nearer in time."""
    forward = drop_short_runs(track_gap(cands, anchor, 0, p), p)
    if not p.get("backward") or end_anchor is None:
        return forward
    backward = drop_short_runs(track_gap(cands[::-1], end_anchor, 0, p), p)[::-1]
    n = len(cands)
    i = np.arange(n)
    from_start = i + 1 if anchor is not None else np.full(n, np.inf)
    use_back = np.isfinite(backward[:, 2]) & (~np.isfinite(forward[:, 2]) | (n - i < from_start))
    return np.where(use_back[:, None], backward, forward)


def interpolate_holes(ball, source, p):
    """Linear fill of short holes whose both ends are known (ATD or detector)."""
    known = np.isfinite(ball[:, 0])
    idx = np.flatnonzero(known)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= p["interp_max"] + 1:
            t = (np.arange(a + 1, b) - a) / (b - a)
            ball[a + 1:b] = ball[a] + t[:, None] * (ball[b] - ball[a])
            source[a + 1:b] = SOURCE_INTERP
    return ball, source


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for key, value in PARAMS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args()
    p = {key: getattr(args, key) for key in PARAMS}

    det_path = os.path.join(OUT_DIR, "{}_ball_detections.npz".format(GAME))
    if not os.path.isfile(det_path):
        raise SystemExit("missing {}: run enrich.detect_ball first".format(det_path))
    z = np.load(det_path)
    frames, det = z["frames"], z["det"]
    atd = load_atd()
    homography = load_homography()
    scale = np.array(PITCH, dtype=np.float64)

    n = len(atd["ball"])
    ball_m = atd["ball"].astype(np.float64) * scale
    source = np.where(np.isfinite(ball_m[:, 0]), SOURCE_ATD, SOURCE_NONE).astype(np.int8)
    conf = np.where(source == SOURCE_ATD, 1.0, np.nan)

    cands = np.full((n, det.shape[1], 3), np.nan)
    cands[frames] = candidates_in_metres(det, frames, homography, p)

    graded = np.zeros(n, dtype=bool)
    for _, first, last in graded_blocks():
        graded[first:last + 1] = True
    missing = graded & (source == SOURCE_NONE)
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    gaps = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    for start, stop in gaps:
        anchor = ball_m[start - 1] if start > 0 and source[start - 1] == SOURCE_ATD else None
        accepted = fill_gap(cands[start:stop], anchor, p)
        hit = np.isfinite(accepted[:, 2])
        ball_m[start:stop][hit] = accepted[hit, :2]
        conf[start:stop][hit] = accepted[hit, 2]
        source[start:stop][hit] = SOURCE_DETECTOR

    ball_m, source = interpolate_holes(ball_m, source, p)
    ball_m[~graded & (source == SOURCE_INTERP)] = np.nan   # only enrich graded frames
    source[~graded & (source == SOURCE_INTERP)] = SOURCE_NONE

    out = os.path.join(OUT_DIR, "{}_atd_enriched.npz".format(GAME))
    np.savez_compressed(out, players=atd["players"], team=atd["team"],
                        ball=(ball_m / scale).astype(np.float32), ball_source=source,
                        ball_conf=conf.astype(np.float32))
    with open(out[:-len(".npz")] + "_meta.json", "w") as handle:
        json.dump(p, handle, indent=1)

    gap_len = np.zeros(n)
    for start, stop in gaps:
        gap_len[start:stop] = (stop - start) / FPS
    filled = missing & (source >= SOURCE_DETECTOR)
    print("graded frames: {}   ATD ball missing: {} ({:.1%})".format(
        int(graded.sum()), int(missing.sum()), missing.sum() / graded.sum()))
    print("filled: {:.1%} of missing  (detector {:.1%}, interpolated {:.1%})".format(
        filled.sum() / missing.sum(), (missing & (source == SOURCE_DETECTOR)).sum() / missing.sum(),
        (missing & (source == SOURCE_INTERP)).sum() / missing.sum()))
    print("ball now visible on {:.1%} of graded frames (was {:.1%})".format(
        (graded & (source != SOURCE_NONE)).sum() / graded.sum(),
        (graded & (atd["ball"][:, 0] == atd["ball"][:, 0])).sum() / graded.sum()))
    print("filled share by gap length:")
    for lo, hi in ((0, 1), (1, 5), (5, 10), (10, 30), (30, 1e9)):
        sel = missing & (gap_len > lo) & (gap_len <= hi)
        if sel.any():
            print("  {:>4}-{:<4} s  {:7d} frames  {:6.1%} filled".format(
                lo, "inf" if hi > 1e8 else int(hi), int(sel.sum()), filled[sel].mean()))
    print("-> " + os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
