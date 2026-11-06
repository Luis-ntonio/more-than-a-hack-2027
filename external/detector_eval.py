"""Score a ball detector on SoccerTrack v2's panoramic video, the closest external match to the task.

    docker compose run --rm gpu python -m external.detector_eval detect --name original
    docker compose run --rm dev python -m external.detector_eval report

SoccerTrack has no ball boxes and its ball track is interpolated between events, so a detector is
only scored on the frames of PASS and DRIVE events, where the track is anchored and the ball is on
the ground at a player's foot. Candidates are mapped from raw panorama pixels to the pitch with the
camera's fisheye model, its homography and a polynomial correction fitted on the 65 hand-annotated
keypoints of each match (leave-one-out error about 0.5 m). Annotations and video agree to within a
few frames, so the numbers compare detectors, they do not measure absolute accuracy. "chance" is the
same score against where the ball is CHANCE_S later.

Tiling, weights and thresholds for the provided match are chosen here. Its own video plays no part.
"""
import argparse
import ast
import json
import os

import cv2
import numpy as np

from external import soccertrack as st

PER_MATCH = 500
LABELS = ("PASS", "DRIVE")
CHANCE_S = 5
DEGREE, RIDGE = 4, 1e-2
HALF_LENGTH, HALF_WIDTH = 52.5, 34.0
CONFS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def terms(uv):
    u, v = uv[:, 0] / HALF_LENGTH - 1, uv[:, 1] / HALF_WIDTH - 1
    return np.column_stack([u ** i * v ** j for i in range(DEGREE + 1) for j in range(DEGREE + 1 - i)])


class Camera:
    """Raw panorama pixels -> metres on the pitch, centred on (0, 0)."""

    def __init__(self, match):
        base = os.path.join(st.DATA_DIR, "raw", match, match + "_")
        with open(base + "keypoints.json") as handle:
            keypoints = json.load(handle)
        z = np.load(base + "camera_intrinsics.npz")
        self.K, self.D, self.Knew = z["K"], z["D"], z["Knew"]
        self.inverse = np.linalg.inv(np.load(base + "homography.npy"))
        pitch = np.array([ast.literal_eval(k) for k in keypoints], dtype=np.float64)
        uv = self.homography(np.array(list(keypoints.values()), dtype=np.float64))
        A = terms(uv)
        self.correction = np.linalg.solve(A.T @ A + RIDGE * np.eye(A.shape[1]), A.T @ (pitch - uv))

    def homography(self, pixels):
        und = cv2.fisheye.undistortPoints(pixels.reshape(-1, 1, 2), self.K, self.D, P=self.Knew)
        v = np.column_stack([und.reshape(-1, 2), np.ones(len(pixels))]) @ self.inverse.T
        return v[:, :2] / v[:, 2:3]

    def to_pitch(self, pixels):
        uv = self.homography(pixels)
        return uv + terms(uv) @ self.correction - np.array([HALF_LENGTH, HALF_WIDTH])


def event_frames(match, rng):
    """Sorted 1-based frames of a sample of the first half's PASS and DRIVE events with a ball."""
    with open(os.path.join(st.DATA_DIR, "bas", match, match + "_12_class_events.json")) as handle:
        actions = json.load(handle)["actions"]
    ball = np.load(os.path.join(st.DATA_DIR, "ball", match + "_1st_ball.npz"))
    frames = sorted({a["frame"] for a in actions if a["label"] in LABELS
                     and str(a["gameTime"]).startswith("1 ") and a["frame"] <= len(ball["x"])})
    frames = np.array([f for f in frames if np.isfinite(ball["x"][f - 1])])
    return np.sort(rng.choice(frames, size=min(PER_MATCH, len(frames)), replace=False))


def path_of(name):
    return os.path.join(st.DATA_DIR, "eval_{}.npz".format(name))


def detect(args):
    from enrich.detect_ball import WEIGHTS, BallDetector, run
    out = {}
    rng = np.random.default_rng(args.seed)
    detector = BallDetector(args.weights or WEIGHTS, args.tiling, 0.05)
    for match in st.MATCHES:
        frames = event_frames(match, rng)
        video = os.path.join(st.DATA_DIR, "videos", match, match + "_panorama_1st_half.mp4")
        det = run(detector, frames - 1, 4, match, video=video)
        camera = Camera(match)
        pitch = np.full(det.shape[:2] + (2,), np.nan)
        seen = np.isfinite(det[:, :, 2])
        pitch[seen] = camera.to_pitch(det[seen][:, :2].astype(np.float64))
        out[match + "_frames"], out[match + "_conf"], out[match + "_pitch"] = frames, det[:, :, 2], pitch
    np.savez_compressed(path_of(args.name), **out)
    print("-> eval_{}.npz".format(args.name))


def distances(z, match, flip, shift=0):
    """(n, K) metres from each candidate to the ball track at that frame (+ shift)."""
    b = np.load(os.path.join(st.DATA_DIR, "ball", match + "_1st_ball.npz"))
    index = np.minimum(z[match + "_frames"] - 1 + shift, len(b["x"]) - 1)
    truth = np.column_stack([b["x"][index], (-1 if flip else 1) * b["y"][index]]).astype(np.float64)
    return np.linalg.norm(z[match + "_pitch"] - truth[:, None, :], axis=2)


def report(args):
    names = sorted(f[5:-4] for f in os.listdir(st.DATA_DIR) if f.startswith("eval_") and f.endswith(".npz"))
    print("{:<26} {:>6} {:>8} {:>8} {:>8} {:>8} {:>13}".format(
        "detector", "any", "top1<2m", "top1<5m", "top5<2m", "top5<5m", "chance top1<2"))
    for name in names:
        z = np.load(path_of(name))
        # the ball's y axis: whichever sign puts the candidates nearer, per match
        rows = []
        for match in st.MATCHES:
            med = [np.nanmedian(np.nanmin(np.where(np.isfinite(distances(z, match, f)), distances(z, match, f), 1e3), axis=1))
                   for f in (False, True)]
            flip = med[1] < med[0]
            d = distances(z, match, flip)
            c = distances(z, match, flip, CHANCE_S * 25)
            rows.append((d, c, z[match + "_conf"]))
        d = np.concatenate([r[0] for r in rows])
        c = np.concatenate([r[1] for r in rows])
        conf = np.concatenate([r[2] for r in rows])
        top = np.where(np.isfinite(d[:, 0]), d[:, 0], np.inf)
        best = np.min(np.where(np.isfinite(d), d, np.inf), axis=1)
        chance = np.where(np.isfinite(c[:, 0]), c[:, 0], np.inf)
        print("{:<26} {:>6.1%} {:>8.1%} {:>8.1%} {:>8.1%} {:>8.1%} {:>13.1%}".format(
            name, np.isfinite(conf[:, 0]).mean(), np.mean(top < 2), np.mean(top < 5), np.mean(best < 2),
            np.mean(best < 5), np.mean(chance < 2)))
        if args.thresholds:
            print("   conf >=   frames kept   precision@2m   precision@5m")
            for t in CONFS:
                keep = conf[:, 0] >= t
                if keep.any():
                    print("     {:.2f}     {:>7.1%}      {:>7.1%}       {:>7.1%}".format(
                        t, keep.mean(), np.mean(top[keep] < 2), np.mean(top[keep] < 5)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["detect", "report"])
    parser.add_argument("--name", default="original")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--tiling", default="tiles640")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--thresholds", action="store_true")
    args = parser.parse_args()
    {"detect": detect, "report": report}[args.step](args)


if __name__ == "__main__":
    main()
