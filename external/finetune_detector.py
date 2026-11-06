"""Fine-tune the Roboflow ball detector on PBDet-Soccer. No frame of the provided match is involved.

    docker compose run --rm dev python -m external.pbdet
    docker compose run --rm gpu python -m external.finetune_detector
    docker compose run --rm gpu python -m external.detector_eval detect --name pbdet --weights models/football-ball-detection-pbdet.pt

Keeps the ball class only; images with no ball stay as negatives, which is what teaches the model
to stop taking heads and socks for a ball. PBDet's images are 1280 px and its ball about 26 px, so
training at 640 px brings the ball to the size it has in a 640 px tile of a panorama. Whether that
helps is decided by external.detector_eval, not here.
"""
import glob
import hashlib
import os
import shutil

from external.pbdet import DATA_DIR, ROOT

BASE_WEIGHTS = os.path.join(ROOT, "models", "football-ball-detection.pt")
OUT_WEIGHTS = os.path.join(ROOT, "models", "football-ball-detection-pbdet.pt")
RUNS_DIR = os.path.join(ROOT, "external_data", "runs")
LAYOUT = os.path.join(DATA_DIR, "ball")
BALL_CLASS = "1"
SPLITS = {"train": "train", "val": "valid"}


def build():
    """YOLO layout with the ball as class 0. Images are moved, not copied: disk is short."""
    for split, source in SPLITS.items():
        images = os.path.join(LAYOUT, "images", split)
        labels = os.path.join(LAYOUT, "labels", split)
        os.makedirs(images, exist_ok=True)
        os.makedirs(labels, exist_ok=True)
        for path in glob.glob(os.path.join(DATA_DIR, source, "*.jpg")):
            os.replace(path, os.path.join(images, os.path.basename(path)))
        for path in glob.glob(os.path.join(DATA_DIR, "yolo_labels_" + source, "*.txt")):
            with open(path) as handle:
                rows = [line.split() for line in handle]
            with open(os.path.join(labels, os.path.basename(path)), "w") as handle:
                for row in rows:
                    if len(row) >= 5 and row[0] == BALL_CLASS:
                        handle.write("0 " + " ".join(row[1:5]) + "\n")
    with open(os.path.join(LAYOUT, "data.yaml"), "w") as handle:
        handle.write("path: {}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n".format(LAYOUT))


def main():
    from ultralytics import YOLO
    build()
    model = YOLO(BASE_WEIGHTS)
    model.train(data=os.path.join(LAYOUT, "data.yaml"), imgsz=640, epochs=30, batch=16, seed=0,
                deterministic=True, workers=2, patience=8, optimizer="SGD", lr0=0.002,
                project=RUNS_DIR, name="pbdet", exist_ok=True, plots=False, verbose=False)
    shutil.copyfile(os.path.join(RUNS_DIR, "pbdet", "weights", "best.pt"), OUT_WEIGHTS)
    with open(OUT_WEIGHTS, "rb") as handle:
        print("-> {}  sha256 {}".format(os.path.relpath(OUT_WEIGHTS, ROOT),
                                        hashlib.sha256(handle.read()).hexdigest()))


if __name__ == "__main__":
    main()
