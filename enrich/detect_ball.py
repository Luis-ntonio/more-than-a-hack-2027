"""Detect the ball in the tactical video with the Roboflow sports YOLO model where the ATD lost it.

    docker compose run --rm gpu python -m enrich.detect_ball --mode calibrate   # measure first
    docker compose run --rm gpu python -m enrich.detect_ball --mode missing     # the real run

calibrate  a seeded sample of graded frames where the ATD *does* have the ball, so each detection
           can be checked against it. Prints recall and precision in metres per tiling, nothing
           per frame. Optimistic by construction: a ball the ATD sees is an easier ball.
missing    every graded frame where the ATD has no ball. Writes the top-K candidates per frame
           (pixels + confidence) to cache/enriched/<game>_ball_detections.npz; merge.py decides
           which of them become ball positions.
visible    every graded frame where the ATD does have the ball, to <game>_ball_detections_visible.npz.
           With missing it covers the whole match: the detector's unsmoothed path, for touches.
windows    seeded stretches where the ATD has the ball on every frame, saved separately so that
           `merge --simulate` can hide the ATD ball and measure how well the gap filling recovers
           it. Same optimism caveat as calibrate.

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

from enrich.geometry import pixels_to_pitch
from games import GAME, video_path
from possession.io import CACHE_DIR, FPS, PITCH, ROOT, graded_blocks, load_atd, load_homography

OUT_DIR = os.path.join(CACHE_DIR, "enriched")
WEIGHTS = os.path.join(ROOT, "models", "football-ball-detection.pt")
TOP_K = 5
DEDUPE_PX = 12          # candidates closer than this (overlapping tiles) are the same object
SEEK_GAP = 250          # beyond this many frames, seek instead of decoding through
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


def calibrate(args, atd, homography):
    rng = np.random.default_rng(args.seed)
    graded = graded_frames()
    visible = graded[np.isfinite(atd["ball"][graded, 0])]
    sample = np.sort(rng.choice(visible, size=min(args.sample, len(visible)), replace=False))
    scale = np.array(PITCH, dtype=np.float64)
    truth = atd["ball"][sample].astype(np.float64) * scale
    print("calibration: {} graded frames where the ATD has the ball (seed {})".format(
        len(sample), args.seed))
    for tiling in args.tilings:
        detector = BallDetector(args.weights, tiling, args.conf)
        det = run(detector, sample, args.batch, tiling)
        dist = np.full((len(sample), TOP_K), np.nan)
        for i, frame in enumerate(sample):
            x, y = pixels_to_pitch(homography[frame], det[i, :, 0], det[i, :, 1])
            dist[i] = np.hypot(x * scale[0] - truth[i, 0], y * scale[1] - truth[i, 1])
        top_conf = det[:, 0, 2]
        top_dist = dist[:, 0]
        any_det = np.isfinite(top_conf)
        print("\n== {} ==".format(tiling))
        print("  any detection        {:6.1%}".format(any_det.mean()))
        for metres in (1, 2, 5):
            print("  top-1 within {} m     {:6.1%}   any of top-{} {:6.1%}".format(
                metres, np.mean(top_dist < metres), TOP_K, np.mean(np.nanmin(
                    np.where(np.isfinite(dist), dist, np.inf), axis=1) < metres)))
        print("  conf >=   coverage   precision@2m")
        for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
            keep = top_conf >= t
            precision = np.mean(top_dist[keep] < 2) if keep.any() else float("nan")
            print("    {:.1f}     {:6.1%}     {:6.1%}".format(t, keep.mean(), precision))
        if args.diagnose:
            diagnose(sample, det, dist, homography, atd, scale)


def diagnose(sample, det, dist, homography, atd, scale):
    """Why top-1 misses: frame misalignment, or a fixed object the model keeps firing on."""
    print("  -- diagnose")
    # 1. alignment: score candidates against the ATD ball k frames away
    for k in (-50, -25, -10, -5, -2, -1, 0, 1, 2, 5, 10, 25, 50):
        shifted = np.clip(sample + k, 0, len(atd["ball"]) - 1)
        truth = atd["ball"][shifted].astype(np.float64) * scale
        hits = []
        for i, frame in enumerate(sample):
            x, y = pixels_to_pitch(homography[frame], det[i, :, 0], det[i, :, 1])
            d = np.hypot(x * scale[0] - truth[i, 0], y * scale[1] - truth[i, 1])
            hits.append(np.nanmin(np.where(np.isfinite(d), d, np.inf)) < 2)
        print("     truth at frame{:+4d}: any-of-top-5 within 2 m {:6.1%}".format(k, np.mean(hits)))
    # 2. hotspots: do the misses pile up on the same pixels?
    miss = ~(dist[:, 0] < 2) & np.isfinite(det[:, 0, 2])
    cells = np.floor(det[miss, 0, :2] / 40).astype(int)
    if len(cells):
        uniq, counts = np.unique(cells, axis=0, return_counts=True)
        order = np.argsort(-counts)[:5]
        print("     top-1 misses: {}, in {} distinct 40 px cells; busiest cells:".format(
            int(miss.sum()), len(uniq)))
        for j in order:
            print("       pixel ~({:4d},{:4d})  {:5.1%} of misses".format(
                uniq[j][0] * 40 + 20, uniq[j][1] * 40 + 20, counts[j] / miss.sum()))
    hit = dist[:, 0] < 2
    for name, sel in (("hits", hit), ("misses", miss)):
        if sel.any():
            print("     {:<7} conf median {:.2f}  box px median {:.1f}  top-1 dist median {:.1f} m".format(
                name, np.median(det[sel, 0, 2]), np.median(det[sel, 0, 3]),
                np.median(dist[sel, 0])))


def detect_missing(args, atd):
    graded = graded_frames()
    wanted = graded[~np.isfinite(atd["ball"][graded, 0])]
    if args.limit:
        wanted = wanted[:args.limit]
    print("{} graded frames without ATD ball".format(len(wanted)))
    detector = BallDetector(args.weights, args.tilings[0], args.conf)
    det = run(detector, wanted, args.batch, args.tilings[0])
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "{}_ball_detections.npz".format(GAME))
    np.savez_compressed(out, frames=wanted.astype(np.int32), det=det)
    with open(args.weights, "rb") as handle:
        weights_sha = hashlib.sha256(handle.read()).hexdigest()
    with open(out[:-len(".npz")] + "_meta.json", "w") as handle:
        json.dump({"tiling": args.tilings[0], "conf": args.conf, "top_k": TOP_K,
                   "weights_sha256": weights_sha, "limit": args.limit}, handle, indent=1)
    with_det = np.isfinite(det[:, 0, 2])
    print("frames with >=1 candidate: {:.1%} -> {}".format(with_det.mean(), os.path.relpath(out, ROOT)))


def detect_visible(args, atd):
    """Detections on every graded frame that has an ATD ball: with --mode missing, the whole match.
    The detector's ball is not smoothed, so its path shows touches the ATD's does not."""
    graded = graded_frames()
    wanted = graded[np.isfinite(atd["ball"][graded, 0])]
    if args.limit:
        wanted = wanted[:args.limit]
    print("{} graded frames with ATD ball".format(len(wanted)))
    detector = BallDetector(args.weights, args.tilings[0], args.conf)
    det = run(detector, wanted, args.batch, args.tilings[0])
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "{}_ball_detections_visible.npz".format(GAME))
    np.savez_compressed(out, frames=wanted.astype(np.int32), det=det)
    print("frames with >=1 candidate: {:.1%} -> {}".format(np.isfinite(det[:, 0, 2]).mean(),
                                                         os.path.relpath(out, ROOT)))


def detect_windows(args, atd):
    """Detections on contiguous stretches with the ATD ball visible, plus the frame before each."""
    rng = np.random.default_rng(args.seed)
    length = args.window
    visible = np.isfinite(atd["ball"][:, 0])
    starts = []
    for half, first, last in graded_blocks():
        if args.last_min:  # only the last minutes of the 2nd half (enrich.finetune_ball's held-out)
            if half != 2:
                continue
            first = max(first, last - args.last_min * 60 * FPS)
        ok = visible[first:last + 1]
        csum = np.concatenate(([0], np.cumsum(ok)))
        full = np.flatnonzero(csum[length + 1:] - csum[:-length - 1] == length + 1) + first
        starts.append(full)  # window [s+1, s+length], anchor s, all visible
    starts = np.concatenate(starts)
    chosen = []
    for s in rng.permutation(starts):
        if all(abs(s - c) > length for c in chosen):
            chosen.append(int(s))
        if len(chosen) == args.n_windows:
            break
    chosen = np.sort(np.array(chosen))
    wanted = np.concatenate([np.arange(s + 1, s + 1 + length) for s in chosen])
    print("{} windows of {} frames ({} frames)".format(len(chosen), length, len(wanted)))
    detector = BallDetector(args.weights, args.tilings[0], args.conf)
    det = run(detector, wanted, args.batch, args.tilings[0])
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "{}_ball_detections_windows{}.npz".format(GAME, args.suffix))
    np.savez_compressed(out, anchors=chosen.astype(np.int32), length=length,
                        frames=wanted.astype(np.int32), det=det)
    print("-> " + os.path.relpath(out, ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["calibrate", "missing", "visible", "windows"], required=True)
    parser.add_argument("--tilings", nargs="+", choices=list(TILINGS), default=None,
                        help="calibrate: which to compare (default all); missing: the first is used")
    parser.add_argument("--conf", type=float, default=0.05, help="keep candidates above this")
    parser.add_argument("--batch", type=int, default=8, help="frames per GPU batch")
    parser.add_argument("--sample", type=int, default=2000, help="calibrate: frames to check")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", default=WEIGHTS,
                        help="detector weights (models/football-ball-detection-ft.pt: enrich.finetune_ball)")
    parser.add_argument("--diagnose", action="store_true",
                        help="calibrate: test frame alignment and look for fixed false positives")
    parser.add_argument("--limit", type=int, default=0, help="missing, visible: only the first N (smoke test)")
    parser.add_argument("--n-windows", type=int, default=40, help="windows: how many stretches")
    parser.add_argument("--window", type=int, default=125, help="windows: frames per stretch")
    parser.add_argument("--last-min", type=int, default=0,
                        help="windows: draw them from the last N minutes of the 2nd half only")
    parser.add_argument("--suffix", default="", help="windows: appended to the output file name")
    args = parser.parse_args()
    args.tilings = args.tilings or (list(TILINGS) if args.mode == "calibrate" else ["tiles640"])
    if not os.path.isfile(args.weights):
        raise SystemExit("missing {}: run  bash enrich/download_models.sh".format(args.weights))
    atd = load_atd()
    if args.mode == "calibrate":
        calibrate(args, atd, load_homography())
    elif args.mode == "windows":
        detect_windows(args, atd)
    elif args.mode == "visible":
        detect_visible(args, atd)
    else:
        detect_missing(args, atd)


if __name__ == "__main__":
    main()
