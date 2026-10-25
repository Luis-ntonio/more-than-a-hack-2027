"""Fine-tune the Roboflow ball detector on this match, with labels made from the ATD (no hand labels).

    docker compose run --rm gpu python -m enrich.finetune_ball dataset    # frames + YOLO labels
    docker compose run --rm gpu python -m enrich.finetune_ball train      # -> models/football-ball-detection-ft.pt
    docker compose run --rm gpu python -m enrich.finetune_ball evaluate   # original vs fine-tuned, held-out frames

Labels. On graded frames where the ATD has the ball, its position is projected into the image
with the frame's homography. The ATD has no height, so a lofted ball projects onto the grass below
it; to keep wrong boxes out, a frame is labelled only
  - matched: when a candidate of the original detector (detect_ball --mode visible) lies within
    MATCH_M of the ATD ball, and the box is that candidate's (its centre and size), or
  - projected: when no candidate is near but the ATD ball is slow (under SLOW_MS: on the ground),
    with a box of the ball's size at the projection -- the balls the original model misses.
Every other object in a labelled frame is background, which is what teaches the model to stop
taking spare balls and heads for the ball.

Split. The last HOLDOUT_MIN minutes of the 2nd half are held out (after a GUARD_S gap), so the
evaluation is on frames far in time from anything trained on. evaluate scores both models on a
seeded sample of held-out frames with an ATD ball (all of them, lofted ones included), like
detect_ball --mode calibrate.

Everything runs locally on the GPU container; frames are written under cache/ (gitignored).
"""
import argparse
import hashlib
import os
import shutil

import cv2
import numpy as np

from enrich.detect_ball import (OUT_DIR, TOP_K, WEIGHTS, BallDetector, graded_frames, read_frames,
                                run)
from enrich.geometry import pixels_to_pitch
from games import GAME, video_path
from possession.io import CACHE_DIR, FPS, PITCH, ROOT, graded_blocks, load_atd, load_homography

DATA_DIR = os.path.join(CACHE_DIR, "finetune", "ball")
RUNS_DIR = os.path.join(CACHE_DIR, "finetune", "runs")
FT_WEIGHTS = os.path.join(ROOT, "models", "football-ball-detection-ft.pt")
MATCH_M = 1.0
SLOW_MS = 2.0
BALL_M = 0.22
HOLDOUT_MIN = 20
GUARD_S = 60
N_TRAIN = 8000
N_VAL = 1000
N_EVAL = 2000
EPOCHS = 40
IMGSZ = 1280
BATCH = 4


def project(h, x_norm, y_norm):
    """Normalised pitch -> image pixels with one frame's homography (row-major 3x3)."""
    m = np.asarray(h, dtype=np.float64).reshape(3, 3)
    px, py, w = m @ np.array([x_norm, y_norm, 1.0])
    return px / w, py / w


def split():
    """(train frames, held-out frames): graded frames with an ATD ball, the held-out ones the last
    HOLDOUT_MIN minutes of the 2nd half."""
    atd = load_atd()
    graded = graded_frames()
    graded = graded[np.isfinite(atd["ball"][graded, 0])]
    _, first, last = [b for b in graded_blocks() if b[0] == 2][0]
    held_from = last - HOLDOUT_MIN * 60 * FPS
    train = graded[graded < held_from - GUARD_S * FPS]
    return train, graded[graded >= held_from]


def make_labels(frames):
    """{frame: (cx, cy, size, kind)} for the frames that can be labelled (see the module doc)."""
    atd = load_atd()
    homography = load_homography()
    scale = np.array(PITCH, dtype=np.float64)
    ball = atd["ball"].astype(np.float64)
    step = np.linalg.norm(np.diff(ball * scale, axis=0, prepend=ball[:1] * scale), axis=1) * FPS
    z = np.load(os.path.join(OUT_DIR, "{}_ball_detections_visible.npz".format(GAME)))
    det_at = dict(zip(z["frames"].tolist(), range(len(z["frames"]))))
    wanted = set(frames.tolist())
    out = {}
    for frame in sorted(wanted):
        h = homography[frame]
        x, y = ball[frame]
        if not (np.all(np.isfinite(h)) and 0 <= x <= 1 and 0 <= y <= 1):
            continue
        px, py = project(h, x, y)
        # ball size in pixels from the local scale
        qx, qy = project(h, x + BALL_M / scale[0], y)
        size = max(np.hypot(qx - px, qy - py), 6.0)
        if frame in det_at:
            cands = z["det"][det_at[frame]]
            cands = cands[np.isfinite(cands[:, 2])]
            if len(cands):
                cx, cy = pixels_to_pitch(h, cands[:, 0], cands[:, 1])
                dist = np.hypot((cx - x) * scale[0], (cy - y) * scale[1])
                if np.nanmin(dist) < MATCH_M:
                    c = cands[int(np.nanargmin(dist))]
                    out[frame] = (float(c[0]), float(c[1]), float(c[3]), "matched")
                    continue
        if step[frame] < SLOW_MS:
            out[frame] = (px, py, size, "projected")
    return out


def dataset(args):
    train, held = split()
    labels = make_labels(np.concatenate([train, held]))
    rng = np.random.default_rng(args.seed)
    train = np.array([f for f in train if f in labels])
    # validation for training (early stopping) comes from the train range, never the held-out one
    pick = np.sort(rng.choice(train, size=min(N_TRAIN + N_VAL, len(train)), replace=False))
    val_mask = np.zeros(len(pick), dtype=bool)
    val_mask[rng.choice(len(pick), size=min(N_VAL, len(pick)), replace=False)] = True
    kinds = [labels[f][3] for f in pick]
    print("train range: {} labellable frames ({:.0%} matched, {:.0%} projected); "
          "using {} train + {} val".format(len(train), np.mean([k == "matched" for k in kinds]),
                                           np.mean([k == "projected" for k in kinds]),
                                           (~val_mask).sum(), val_mask.sum()))
    if os.path.isdir(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    for part in ("train", "val"):
        os.makedirs(os.path.join(DATA_DIR, "images", part))
        os.makedirs(os.path.join(DATA_DIR, "labels", part))
    part_of = {int(f): ("val" if v else "train") for f, v in zip(pick, val_mask)}
    for frame, image in read_frames(video_path(), pick):
        if image is None:
            continue
        part = part_of[frame]
        height, width = image.shape[:2]
        cx, cy, size, _ = labels[frame]
        name = "f{:06d}".format(frame)
        cv2.imwrite(os.path.join(DATA_DIR, "images", part, name + ".jpg"), image,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(DATA_DIR, "labels", part, name + ".txt"), "w") as handle:
            handle.write("0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                cx / width, cy / height, size / width, size / height))
    with open(os.path.join(DATA_DIR, "data.yaml"), "w") as handle:
        handle.write("path: {}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n".format(DATA_DIR))
    print("-> " + os.path.relpath(DATA_DIR, ROOT))


def train(args):
    from ultralytics import YOLO
    model = YOLO(WEIGHTS)
    model.train(data=os.path.join(DATA_DIR, "data.yaml"), imgsz=IMGSZ, epochs=args.epochs,
                batch=BATCH, seed=args.seed, deterministic=True, workers=2, patience=10,
                optimizer="SGD", lr0=0.002, project=RUNS_DIR, name="ball", exist_ok=True,
                plots=False, verbose=False)
    best = os.path.join(RUNS_DIR, "ball", "weights", "best.pt")
    shutil.copyfile(best, FT_WEIGHTS)
    with open(FT_WEIGHTS, "rb") as handle:
        print("-> {}  sha256 {}".format(os.path.relpath(FT_WEIGHTS, ROOT),
                                        hashlib.sha256(handle.read()).hexdigest()))


def evaluate(args):
    _, held = split()
    rng = np.random.default_rng(args.seed)
    sample = np.sort(rng.choice(held, size=min(N_EVAL, len(held)), replace=False))
    atd = load_atd()
    homography = load_homography()
    scale = np.array(PITCH, dtype=np.float64)
    truth = atd["ball"][sample].astype(np.float64) * scale
    ball = atd["ball"].astype(np.float64) * scale
    speed = np.linalg.norm(np.diff(ball, axis=0, prepend=ball[:1]), axis=1)[sample] * FPS
    slow = speed < SLOW_MS
    print("held-out: last {} min of the 2nd half, {} frames with an ATD ball ({:.0%} slow)".format(
        HOLDOUT_MIN, len(sample), slow.mean()))
    print("{:<10} {:<9} {:>7} {:>8} {:>8} {:>10} {:>8} {:>12}".format(
        "weights", "tiling", "any", "top1<1m", "top1<2m", "top5<2m", "top1<5m", "top1<2m slow"))
    for label, weights in (("original", WEIGHTS), ("finetuned", FT_WEIGHTS)):
        for tiling in args.tilings:
            det = run(BallDetector(weights, tiling, 0.05), sample, 8, "{} {}".format(label, tiling))
            dist = np.full((len(sample), TOP_K), np.nan)
            for i, frame in enumerate(sample):
                x, y = pixels_to_pitch(homography[frame], det[i, :, 0], det[i, :, 1])
                dist[i] = np.hypot(x * scale[0] - truth[i, 0], y * scale[1] - truth[i, 1])
            top = np.where(np.isfinite(dist[:, 0]), dist[:, 0], np.inf)
            best = np.nanmin(np.where(np.isfinite(dist), dist, np.inf), axis=1)
            print("{:<10} {:<9} {:>7.1%} {:>8.1%} {:>8.1%} {:>10.1%} {:>8.1%} {:>12.1%}".format(
                label, tiling, np.isfinite(det[:, 0, 2]).mean(), np.mean(top < 1), np.mean(top < 2),
                np.mean(best < 2), np.mean(top < 5), np.mean(top[slow] < 2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["dataset", "train", "evaluate"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--tilings", nargs="+", default=["tiles640", "full1280"])
    args = parser.parse_args()
    {"dataset": dataset, "train": train, "evaluate": evaluate}[args.step](args)


if __name__ == "__main__":
    main()
