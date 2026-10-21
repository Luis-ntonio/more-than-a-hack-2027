"""Container smoke test: libraries import, data is mounted, video and ATD are readable."""
import os
import sys

sys.path.insert(0, os.getcwd())
import cv2
import kloppy
import lightgbm
import sklearn
from kloppy import metrica

from games import asset_paths

paths = asset_paths()
cap = cv2.VideoCapture(paths["video"])
ok, frame = cap.read()
print("video:", ok, frame.shape if ok else None,
      int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "frames @", cap.get(cv2.CAP_PROP_FPS), "fps")
dataset = metrica.load_tracking_epts(meta_data=paths["atd_meta"], raw_data=paths["atd_raw"], limit=1000)
print("kloppy", kloppy.__version__, "->", len(dataset.records), "records, teams",
      [str(team) for team in dataset.metadata.teams])
print("lightgbm", lightgbm.__version__, "| scikit-learn", sklearn.__version__)
writable = os.access(os.environ["MTAH_DATA_DIR"], os.W_OK)
print("data mount read-only:", not writable)
