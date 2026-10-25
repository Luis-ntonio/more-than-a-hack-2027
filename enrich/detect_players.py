"""Detect players, goalkeepers, referees and the ball in the tactical video (Roboflow sports YOLO).

    docker compose run --rm gpu python -m enrich.detect_players --limit 500       # smoke test
    docker compose run --rm gpu python -m enrich.detect_players --frames labelled  # ~82k frames
    docker compose run --rm gpu python -m enrich.detect_players --frames graded    # whole match

Writes up to MAX_BOXES boxes per frame -- x1, y1, x2, y2 in pixels, confidence, class (0 ball,
1 goalkeeper, 2 player, 3 referee), NaN-padded -- to cache/enriched/<game>_player_detections_<frames>.npz.
possession/control.py gives each box a team from the nearest ATD player and finds who has the ball.

labelled  the frames of every finished hand-labelled block (labels/), to measure before the full run
graded    every graded frame of the match

Everything runs locally on the GPU container. Output is derived match data: cache/ is gitignored.
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

from enrich.detect_ball import OUT_DIR, graded_frames, read_frames
from games import GAME, video_path
from possession.io import ROOT

WEIGHTS = os.path.join(ROOT, "models", "football-player-detection.pt")
MAX_BOXES = 40
IMGSZ = 1280       # what the model was trained at


class PlayerDetector:
    def __init__(self, weights, conf, half=True):
        import torch
        from ultralytics import YOLO
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.model = YOLO(weights)
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.half = half and self.device == 0
        self.conf = conf

    def __call__(self, frames):
        """BGR frames -> (len(frames), MAX_BOXES, 6) float32, highest confidence first."""
        results = self.model.predict(frames, imgsz=IMGSZ, conf=self.conf, half=self.half,
                                     device=self.device, verbose=False)
        out = np.full((len(frames), MAX_BOXES, 6), np.nan, dtype=np.float32)
        for i, result in enumerate(results):
            boxes = result.boxes
            if boxes is None or not len(boxes):
                continue
            rows = np.column_stack([boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(),
                                    boxes.cls.cpu().numpy()])
            rows = rows[np.argsort(-rows[:, 4], kind="stable")][:MAX_BOXES]
            out[i, :len(rows)] = rows
        return out


def labelled_frames():
    from possession.prelabel import training_blocks
    return np.unique(np.concatenate([np.arange(first, last + 1)
                                     for _, first, last, _ in training_blocks()]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", choices=["labelled", "graded"], default="labelled")
    parser.add_argument("--conf", type=float, default=0.1, help="keep boxes above this")
    parser.add_argument("--batch", type=int, default=8, help="frames per GPU batch")
    parser.add_argument("--limit", type=int, default=0, help="only the first N frames (smoke test)")
    args = parser.parse_args()
    if not os.path.isfile(WEIGHTS):
        raise SystemExit("missing {}: run  bash enrich/download_models.sh".format(WEIGHTS))

    wanted = labelled_frames() if args.frames == "labelled" else graded_frames()
    if args.limit:
        wanted = wanted[:args.limit]
    print("{} frames ({})".format(len(wanted), args.frames))
    detector = PlayerDetector(WEIGHTS, args.conf)
    out = np.full((len(wanted), MAX_BOXES, 6), np.nan, dtype=np.float32)
    pending, index, started = [], 0, time.time()
    for _, image in read_frames(video_path(), wanted):
        if image is None:
            image = np.zeros((1080, 1920, 3), dtype=np.uint8)  # undecodable: detects nothing
        pending.append(image)
        if len(pending) == args.batch:
            out[index:index + len(pending)] = detector(pending)
            index += len(pending)
            pending.clear()
            if index % (args.batch * 250) == 0:
                rate = index / (time.time() - started)
                print("  {}/{} frames, {:.1f} fps, ~{:.0f} min left".format(
                    index, len(wanted), rate, (len(wanted) - index) / rate / 60), flush=True)
    if pending:
        out[index:index + len(pending)] = detector(pending)
    print("  {} frames in {:.1f} min".format(len(wanted), (time.time() - started) / 60))

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "{}_player_detections_{}.npz".format(GAME, args.frames))
    np.savez_compressed(path, frames=wanted.astype(np.int32), boxes=out)
    with open(WEIGHTS, "rb") as handle:
        sha = hashlib.sha256(handle.read()).hexdigest()
    with open(path[:-len(".npz")] + "_meta.json", "w") as handle:
        json.dump({"conf": args.conf, "imgsz": IMGSZ, "max_boxes": MAX_BOXES,
                   "weights_sha256": sha, "limit": args.limit}, handle, indent=1)
    cls = out[:, :, 5]
    print("boxes per frame (median): players {:.0f}, goalkeepers {:.0f}, referees {:.0f}; "
          "ball in {:.0%} of frames -> {}".format(
              np.median((cls == 2).sum(axis=1)), np.median((cls == 1).sum(axis=1)),
              np.median((cls == 3).sum(axis=1)), (cls == 0).any(axis=1).mean(),
              os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    main()
