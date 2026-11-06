"""Choose the gap-filling thresholds of enrich.merge on SoccerTrack, without the provided match.

    docker compose run --rm gpu python -m external.fill_eval detect
    docker compose run --rm dev python -m external.fill_eval sweep
    docker compose run --rm dev python -m external.fill_eval params

A window runs from one anchored event to the next. The ball is known at the first event, as the last
ATD position is before a real gap; it is then hidden, the frames in between go through the detector
and enrich.merge's tracker, and the result is scored where the ball is known again, at the second
event. Between events the truth is a straight line, good enough for the 5 m figures only. These
windows are not chosen for being hard to see, and the panorama is harder than a tactical camera, so
the numbers rank settings; their absolute values do not carry over.
"""
import argparse
import json
import os

import numpy as np

from enrich.merge import PARAMS, fill_gap, interpolate_holes
from external import soccertrack as st
from external.detector_eval import HALF_LENGTH, HALF_WIDTH, LABELS, Camera

PER_MATCH = 60
MIN_FRAMES, MAX_FRAMES = 25, 100
PATH = os.path.join(st.DATA_DIR, "fill_windows.npz")
SWEEP = [dict(conf_min=c, conf_seed=s, min_run=r)
         for c in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7) for s in (0.5, 0.7) for r in (3, 8)]
PARAM_SWEEP = ([{}] + [{"keep_conf": v} for v in (0.5, 0.6, 0.8, 0.9, 1.01)]
               + [{"anchor_s": v} for v in (0.5, 2.0, 4.0)]
               + [{"interp_max": v} for v in (0, 2, 10, 25)]
               + [{"max_speed_ms": v} for v in (25.0, 30.0, 50.0)])


def ball_of(match):
    b = np.load(os.path.join(st.DATA_DIR, "ball", match + "_1st_ball.npz"))
    return np.column_stack([b["x"], b["y"]]).astype(np.float64)


def windows(match, rng):
    """(n, 2) 1-based frames of consecutive first-half events MIN_FRAMES..MAX_FRAMES apart."""
    with open(os.path.join(st.DATA_DIR, "bas", match, match + "_12_class_events.json")) as handle:
        actions = json.load(handle)["actions"]
    ball = ball_of(match)
    frames = sorted({a["frame"] for a in actions if a["label"] in LABELS
                     and str(a["gameTime"]).startswith("1 ") and a["frame"] <= len(ball)})
    pairs = [(a, b) for a, b in zip(frames[:-1], frames[1:]) if MIN_FRAMES <= b - a <= MAX_FRAMES
             and np.isfinite(ball[a - 1, 0]) and np.isfinite(ball[b - 1, 0])]
    keep = rng.choice(len(pairs), size=min(PER_MATCH, len(pairs)), replace=False)
    return np.array(pairs)[np.sort(keep)]


def detect(args):
    from enrich.detect_ball import WEIGHTS, BallDetector, run
    rng = np.random.default_rng(args.seed)
    detector = BallDetector(WEIGHTS, "tiles640", 0.05)
    out = {}
    for match in st.MATCHES:
        pairs = windows(match, rng)
        frames = np.unique(np.concatenate([np.arange(a + 1, b + 1) for a, b in pairs]))
        video = os.path.join(st.DATA_DIR, "videos", match, match + "_panorama_1st_half.mp4")
        det = run(detector, frames - 1, 4, match, video=video)
        pitch = np.full(det.shape[:2] + (2,), np.nan)
        seen = np.isfinite(det[:, :, 2])
        pitch[seen] = Camera(match).to_pitch(det[seen][:, :2].astype(np.float64))
        out[match + "_pairs"], out[match + "_frames"] = pairs, frames
        out[match + "_conf"], out[match + "_pitch"] = det[:, :, 2], pitch
        print(match, len(pairs), "windows", len(frames), "frames", flush=True)
    np.savez_compressed(PATH, **out)


def fill_windows(z, match, p, control=False, interp=False):
    """Per frame of every window: filled flag and distance to the truth, plus the last frame's.

    control feeds each window the detections of the next one, which have nothing to do with its
    ball: what still counts as right then comes from the anchor, not from the detector.
    interp also fills short holes as merge does, from the anchor onwards.
    """
    ball = ball_of(match)
    frames = z[match + "_frames"]
    lookup = {f: i for i, f in enumerate(frames.tolist())}
    conf, pitch = z[match + "_conf"], z[match + "_pitch"]
    margin = p["margin_m"]
    cands = np.concatenate([pitch, conf[:, :, None]], axis=2).astype(np.float64)
    bad = ((cands[:, :, 2] < p["conf_min"]) | ~np.isfinite(cands[:, :, 0])
           | (np.abs(cands[:, :, 0]) > HALF_LENGTH + margin) | (np.abs(cands[:, :, 1]) > HALF_WIDTH + margin))
    cands[bad] = np.nan
    filled, dist, end_filled, end_dist, errors = [], [], [], [], []
    pairs = z[match + "_pairs"].tolist()
    for i, (a, b) in enumerate(pairs):
        rows = [lookup[f] for f in range(a + 1, b + 1)]
        if control:
            c, e = pairs[(i + 1) % len(pairs)]
            other = [lookup[f] for f in range(c + 1, e + 1)]
            rows = [other[j % len(other)] for j in range(len(rows))]
        acc = fill_gap(cands[rows], ball[a - 1], p)
        if interp:
            track = np.vstack([ball[a - 1], acc[:, :2]])
            track, _ = interpolate_holes(track, np.zeros(len(track), dtype=np.int8), p)
            acc[:, :2] = track[1:]
            acc[:, 2] = np.where(np.isfinite(track[1:, 0]), 1.0, np.nan)
        t = np.linspace(0, 1, b - a + 1)[1:, None]
        truth = ball[a - 1] + t * (ball[b - 1] - ball[a - 1])
        hit = np.isfinite(acc[:, 2])
        d = np.hypot(acc[:, 0] - truth[:, 0], acc[:, 1] - truth[:, 1])
        filled.append(hit)
        dist.append(d)
        end = slice(len(hit) - 3, len(hit))         # annotations agree with the video to ~3 frames
        end_filled.append(hit[end].any())
        end_dist.append(np.nanmin(d[end]) if hit[end].any() else np.nan)
        errors.append(acc[hit, :2] - truth[hit])
    return (np.concatenate(filled), np.concatenate(dist), np.array(end_filled), np.array(end_dist),
            np.concatenate(errors))


def sweep(args):
    z = np.load(PATH)
    print("{:>8} {:>9} {:>7} {:>9} {:>10} {:>10} {:>10} {:>10}".format(
        "conf_min", "conf_seed", "min_run", "filled", "ok5|fill", "ok5|all", "end filled", "end ok2"))
    for override in SWEEP:
        p = {**PARAMS, "margin_m": args.margin_m, **override}
        parts = [fill_windows(z, m, p) for m in st.MATCHES]
        filled = np.concatenate([x[0] for x in parts])
        dist = np.concatenate([x[1] for x in parts])
        end_filled = np.concatenate([x[2] for x in parts])
        end_dist = np.concatenate([x[3] for x in parts])
        ok5 = filled & (dist < 5)
        print("{conf_min:>8} {conf_seed:>9} {min_run:>7}".format(**override), "{:>9.1%} {:>10.1%} {:>10.1%} {:>10.1%} {:>10.1%}".format(
            filled.mean(), ok5.sum() / max(filled.sum(), 1), ok5.mean(), end_filled.mean(),
            np.mean(end_filled & (end_dist < 2))))
    print("control: detections from another window, same anchor")
    for conf_min in (0.3, 0.5):
        p = {**PARAMS, "margin_m": args.margin_m, "conf_min": conf_min, "conf_seed": 0.5, "min_run": 3}
        parts = [fill_windows(z, m, p, control=True) for m in st.MATCHES]
        filled = np.concatenate([x[0] for x in parts])
        dist = np.concatenate([x[1] for x in parts])
        print("  conf_min {}: filled {:.1%}, ok5|fill {:.1%}".format(
            conf_min, filled.mean(), (filled & (dist < 5)).sum() / max(filled.sum(), 1)))


def params(args):
    """One parameter at a time around enrich.merge's values, with merge's interpolation."""
    z = np.load(PATH)
    print("{:<26} {:>8} {:>9} {:>9}".format("override", "filled", "ok5|fill", "ok5|all"))
    for override in PARAM_SWEEP:
        parts = [fill_windows(z, m, {**PARAMS, **override}, interp=True) for m in st.MATCHES]
        filled = np.concatenate([x[0] for x in parts])
        dist = np.concatenate([x[1] for x in parts])
        ok5 = filled & (dist < 5)
        print("{:<26} {:>8.1%} {:>9.1%} {:>9.1%}".format(
            str(override or "merge"), filled.mean(), ok5.sum() / max(filled.sum(), 1), ok5.mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["detect", "sweep", "params"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--margin-m", type=float, default=2.0)
    args = parser.parse_args()
    {"detect": detect, "sweep": sweep, "params": params}[args.step](args)


if __name__ == "__main__":
    main()
