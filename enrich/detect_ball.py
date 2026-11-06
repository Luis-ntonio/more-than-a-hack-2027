"""Detect the ball in the tactical video with the Roboflow sports YOLO model where the ATD lost it.

    docker compose run --rm gpu python -m enrich.detect_ball

Takes every graded frame where the ATD has no ball and writes the top-K candidates per frame (pixels
+ confidence) to cache/enriched/<game>_ball_detections.npz; merge.py decides which of them become
ball positions. Weights and tiling were chosen on external video (external.detector_eval).

Everything runs locally on the GPU container. Output is derived match data: cache/ is gitignored.
"""
import argparse
import hashlib
import json
import os
import queue
import threading
import time

import cv2
import numpy as np

from games import GAME, video_path
from possession.io import CACHE_DIR, ROOT, graded_blocks, load_atd

OUT_DIR = os.path.join(CACHE_DIR, "enriched")
WEIGHTS = os.path.join(ROOT, "models", "football-ball-detection.pt")
TOP_K = 5
DEDUPE_PX = 12          # candidates closer than this (overlapping tiles) are the same object
SEEK_GAP = 250          # beyond this many frames, seek instead of decoding through
TILING = "tiles640"     # chosen on SoccerTrack (external.detector_eval)
TILINGS = {
    # Roboflow's own example slices the frame into 640 px tiles for the ball (InferenceSlicer).
    "tiles640": {"tile": 640, "overlap": 128, "imgsz": 640},
    "full1280": {"tile": None, "overlap": 0, "imgsz": 1280},
}


def tile_origins(width, height, tile, overlap):
    def starts(size):
        if size <= tile:
            return [0]
        n = int(np.ceil((size - overlap) / (tile - overlap)))
        return [round(i * (size - tile) / (n - 1)) for i in range(n)]
    return [(x, y) for y in starts(height) for x in starts(width)]


class BallDetector:
    def __init__(self, weights, tiling, conf, half=True):
        import torch
        from ultralytics import YOLO
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.model = YOLO(weights)
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.half = half and self.device == 0
        self.spec = TILINGS[tiling]
        self.conf = conf

    def __call__(self, frames):
        """BGR frames -> one (TOP_K, 4) array per frame: px, py, conf, box size. NaN-padded."""
        spec = self.spec
        images, owners = [], []
        for i, frame in enumerate(frames):
            if spec["tile"] is None:
                images.append(frame)
                owners.append((i, 0, 0))
                continue
            height, width = frame.shape[:2]
            for x, y in tile_origins(width, height, spec["tile"], spec["overlap"]):
                images.append(frame[y:y + spec["tile"], x:x + spec["tile"]])
                owners.append((i, x, y))
        results = self.model.predict(images, imgsz=spec["imgsz"], conf=self.conf, half=self.half,
                                     device=self.device, verbose=False)
        found = [[] for _ in frames]
        for (i, x, y), result in zip(owners, results):
            boxes = result.boxes
            if boxes is None or not len(boxes):
                continue
            xywh = boxes.xywh.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            for (cx, cy, w, h), c in zip(xywh, conf):
                found[i].append((cx + x, cy + y, float(c), max(w, h)))
        out = np.full((len(frames), TOP_K, 4), np.nan, dtype=np.float32)
        for i, candidates in enumerate(found):
            kept = []
            for cand in sorted(candidates, key=lambda c: -c[2]):
                if all((cand[0] - k[0]) ** 2 + (cand[1] - k[1]) ** 2 > DEDUPE_PX ** 2 for k in kept):
                    kept.append(cand)
                if len(kept) == TOP_K:
                    break
            if kept:
                out[i, :len(kept)] = kept
        return out


def read_frames(path, wanted, depth=64):
    """Yield (frame, BGR image) for the sorted frame indices in `wanted`, decoding on a thread."""
    buffer = queue.Queue(maxsize=depth)

    def worker():
        cap = cv2.VideoCapture(path)
        position = 0
        for frame in wanted:
            if frame < position or frame - position > SEEK_GAP:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
                position = int(frame)
            while position < frame:
                cap.grab()
                position += 1
            ok, image = cap.read()
            position += 1
            buffer.put((int(frame), image if ok else None))
        buffer.put(None)
        cap.release()

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = buffer.get()
        if item is None:
            return
        yield item


def run(detector, wanted, batch, label, video=None):
    """Detections for every frame in `wanted` (of `video`, default the tactical one), in order.
    Prints throughput only."""
    out = np.full((len(wanted), TOP_K, 4), np.nan, dtype=np.float32)
    pending, index, started = [], 0, time.time()

    def flush():
        nonlocal index
        if not pending:
            return
        images = [img for _, img in pending]
        out[index:index + len(pending)] = detector(images)
        index += len(pending)
        pending.clear()

    for frame, image in read_frames(video or video_path(), wanted):
        if image is None:
            image = np.zeros((1080, 1920, 3), dtype=np.uint8)  # undecodable: detects nothing
        pending.append((frame, image))
        if len(pending) == batch:
            flush()
            if index % (batch * 250) == 0:
                rate = index / (time.time() - started)
                print("  [{}] {}/{} frames, {:.1f} fps, ~{:.0f} min left".format(
                    label, index, len(wanted), rate, (len(wanted) - index) / rate / 60), flush=True)
    flush()
    print("  [{}] {} frames in {:.1f} min".format(label, len(wanted), (time.time() - started) / 60))
    return out


def graded_frames():
    return np.concatenate([np.arange(first, last + 1) for _, first, last in graded_blocks()])


def detect_missing(args, atd):
    graded = graded_frames()
    wanted = graded[~np.isfinite(atd["ball"][graded, 0])]
    if args.limit:
        wanted = wanted[:args.limit]
    print("{} graded frames without ATD ball".format(len(wanted)))
    detector = BallDetector(args.weights, TILING, args.conf)
    det = run(detector, wanted, args.batch, TILING)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "{}_ball_detections.npz".format(GAME))
    np.savez_compressed(out, frames=wanted.astype(np.int32), det=det)
    with open(args.weights, "rb") as handle:
        weights_sha = hashlib.sha256(handle.read()).hexdigest()
    with open(out[:-len(".npz")] + "_meta.json", "w") as handle:
        json.dump({"tiling": TILING, "conf": args.conf, "top_k": TOP_K,
                   "weights_sha256": weights_sha, "limit": args.limit}, handle, indent=1)
    with_det = np.isfinite(det[:, 0, 2])
    print("frames with >=1 candidate: {:.1%} -> {}".format(with_det.mean(), os.path.relpath(out, ROOT)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--conf", type=float, default=0.05, help="keep candidates above this")
    parser.add_argument("--batch", type=int, default=8, help="frames per GPU batch")
    parser.add_argument("--weights", default=WEIGHTS, help="detector weights")
    parser.add_argument("--limit", type=int, default=0, help="only the first N frames (smoke test)")
    args = parser.parse_args()
    if not os.path.isfile(args.weights):
        raise SystemExit("missing {}: run  bash enrich/download_models.sh".format(args.weights))
    detect_missing(args, load_atd())


if __name__ == "__main__":
    main()
