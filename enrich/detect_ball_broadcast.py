"""Detect the ball in the broadcast video where it shows the tactical frame live, for the gaps.

    docker compose run --rm dev python -m possession.broadcast                       # alignment first
    docker compose run --rm gpu python -m enrich.detect_ball_broadcast calibrate     # measure
    docker compose run --rm gpu python -m enrich.detect_ball_broadcast gaps          # the real run

The broadcast frames a smaller part of the pitch, closer, so the ball is bigger in the image. Each
tactical frame the broadcast shows live (possession/broadcast.py) with a valid broadcast homography
has a broadcast frame; the Roboflow detector runs on it and its candidates are projected onto the
pitch with the *broadcast* homography, which puts them in the same normalised coordinates as the
tactical ATD.

calibrate  a seeded sample of such frames where the tactical ATD has the ball: each candidate is
           scored against it, for the original and the fine-tuned weights and both tilings, next to
           the tactical-video detector on the same frames (detect_ball --mode visible)
gaps       every graded tactical frame the broadcast shows live where the tactical ATD has no ball;
           writes cache/enriched/<game>_ball_detections_broadcast.npz: tactical frame, broadcast
           frame, candidates in broadcast pixels and on the pitch (normalised)

Everything runs locally on the GPU container. Output is derived match data: cache/ is gitignored.
"""
import argparse
import os

import numpy as np

from enrich.detect_ball import OUT_DIR, TOP_K, WEIGHTS, BallDetector, graded_frames, run
from enrich.finetune_ball import FT_WEIGHTS
from enrich.geometry import pixels_to_pitch
from games import GAME, video_path
from possession.broadcast import ALIGN_PATH
from possession.io import PITCH, ROOT, load_atd, load_homography

N_CALIBRATE = 2000


def candidates(tactical_frames, which):
    """(tactical frames, broadcast frames) where the broadcast is live with a pitch view, restricted to
    graded tactical frames with (which="visible") or without (which="missing") a tactical ATD ball."""
    z = np.load(ALIGN_PATH)
    hb = load_homography(source="broadcast")
    atd = load_atd()
    i = tactical_frames
    j = z["bc_frame"][i]
    ok = (j >= 0) & z["live"][i]
    ok[ok] &= np.isfinite(hb[j[ok], 0])
    has_ball = np.isfinite(atd["ball"][i, 0])
    ok &= has_ball if which == "visible" else ~has_ball
    return i[ok], j[ok]


def project(det, bc_frames):
    """Candidates in broadcast pixels -> (n, TOP_K, 2) normalised pitch, with the broadcast homography."""
    hb = load_homography(source="broadcast")
    out = np.full(det.shape[:2] + (2,), np.nan)
    for k, j in enumerate(bc_frames):
        x, y = pixels_to_pitch(hb[j], det[k, :, 0], det[k, :, 1])
        out[k, :, 0], out[k, :, 1] = x, y
    return out


def by_broadcast_order(i, j):
    """Sort both by broadcast frame (the video is read forwards), dropping repeated broadcast frames."""
    order = np.argsort(j, kind="stable")
    i, j = i[order], j[order]
    first = np.concatenate([[True], j[1:] != j[:-1]])
    return i[first], j[first]


def report(label, pitch, truth):
    scale = np.array(PITCH, dtype=np.float64)
    dist = np.linalg.norm((pitch - truth[:, None, :]) * scale, axis=2)
    top = np.where(np.isfinite(dist[:, 0]), dist[:, 0], np.inf)
    best = np.nanmin(np.where(np.isfinite(dist), dist, np.inf), axis=1)
    print("{:<30} {:>7.1%} {:>8.1%} {:>8.1%} {:>10.1%} {:>8.1%}".format(
        label, np.isfinite(pitch[:, 0, 0]).mean(), np.mean(top < 1), np.mean(top < 2),
        np.mean(best < 2), np.mean(top < 5)))


def calibrate(args):
    i, j = candidates(graded_frames(), "visible")
    rng = np.random.default_rng(args.seed)
    pick = np.sort(rng.choice(len(i), size=min(N_CALIBRATE, len(i)), replace=False))
    i, j = by_broadcast_order(i[pick], j[pick])
    truth = load_atd()["ball"][i].astype(np.float64)
    print("{} tactical frames with an ATD ball, shown live by the broadcast".format(len(i)))
    print("{:<30} {:>7} {:>8} {:>8} {:>10} {:>8}".format("detector", "any", "top1<1m", "top1<2m",
                                                         "top5<2m", "top1<5m"))
    # the tactical-video detector on the same frames
    z = np.load(os.path.join(OUT_DIR, "{}_ball_detections_visible.npz".format(GAME)))
    row = {int(f): k for k, f in enumerate(z["frames"])}
    ht = load_homography()
    tac = np.full((len(i), TOP_K, 2), np.nan)
    for k, f in enumerate(i):
        if int(f) in row:
            d = z["det"][row[int(f)]]
            x, y = pixels_to_pitch(ht[f], d[:, 0], d[:, 1])
            tac[k, :, 0], tac[k, :, 1] = x, y
    report("tactical video, original", tac, truth)
    for name, weights in (("original", WEIGHTS), ("fine-tuned", FT_WEIGHTS)):
        for tiling in args.tilings:
            det = run(BallDetector(weights, tiling, 0.05), j, 8, "{} {}".format(name, tiling),
                      video=video_path(source="broadcast"))
            report("broadcast, {} {}".format(name, tiling), project(det, j), truth)


def gaps(args):
    i, j = candidates(graded_frames(), "missing")
    i, j = by_broadcast_order(i, j)
    print("{} graded tactical frames without an ATD ball, shown live by the broadcast".format(len(i)))
    det = run(BallDetector(args.weights, args.tiling, 0.05), j, 8, args.tiling,
              video=video_path(source="broadcast"))
    order = np.argsort(i)
    out = os.path.join(OUT_DIR, "{}_ball_detections_broadcast.npz".format(GAME))
    np.savez_compressed(out, frames=i[order].astype(np.int32), bc_frames=j[order].astype(np.int32),
                        det=det[order], pitch=project(det, j)[order].astype(np.float32))
    print("frames with >=1 candidate: {:.1%} -> {}".format(np.isfinite(det[:, 0, 2]).mean(),
                                                         os.path.relpath(out, ROOT)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["calibrate", "gaps"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tilings", nargs="+", default=["tiles640", "full1280"], help="calibrate")
    parser.add_argument("--tiling", default="tiles640", help="gaps")
    parser.add_argument("--weights", default=WEIGHTS, help="gaps")
    args = parser.parse_args()
    {"calibrate": calibrate, "gaps": gaps}[args.step](args)


if __name__ == "__main__":
    main()
