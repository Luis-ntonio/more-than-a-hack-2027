"""Per-frame overlay data for the /overlay page: tracking, homography and ball detections.

Everything is sent in the coordinates it is stored in -- players and balls on the normalised pitch
(0-1), detector candidates in video pixels -- together with each frame's homography, and the
browser projects it onto the video. Detector candidates that project outside the pitch are left
out, as enrich.merge does. Arrays are loaded on the first request and kept in memory.
"""
import os
import threading

import numpy as np

from enrich.geometry import pixels_to_pitch
from games import GAME
from possession.io import CACHE_DIR, PITCH, load_atd, load_homography

ENRICHED_DIR = os.path.join(CACHE_DIR, "enriched")
MAX_SPAN = 500  # frames per request


def _round(values, digits):
    return [None if not np.isfinite(v) else round(float(v), digits) for v in values]


class Overlay:
    def __init__(self):
        self._lock = threading.Lock()
        self._data = None

    def _load(self):
        with self._lock:
            if self._data is not None:
                return self._data
            atd = load_atd()
            data = {
                "players": atd["players"][:, :, :2],
                "ball": atd["ball"],
                "team": atd["team"],
                "homography": load_homography(),
                "enriched": None,
                "detections": None,
            }
            enriched = os.path.join(ENRICHED_DIR, "{}_atd_enriched.npz".format(GAME))
            if os.path.isfile(enriched):
                z = np.load(enriched)
                data["enriched"] = (z["ball"], z["ball_source"])
            detections = os.path.join(ENRICHED_DIR, "{}_ball_detections.npz".format(GAME))
            if os.path.isfile(detections):
                z = np.load(detections)
                data["detections"] = (z["frames"], z["det"])
            self._data = data
            return data

    def meta(self):
        d = self._load()
        return {
            "pitch": list(PITCH), "team": d["team"].astype(int).tolist(),
            "has_enriched": d["enriched"] is not None, "has_detections": d["detections"] is not None,
            "max_span": MAX_SPAN,
        }

    def sources(self):
        """One character per frame of the match: where its ball comes from.

        0 none, 1 tracking (ATD), 2 Roboflow detector, 3 interpolated -- the codes of
        enrich.merge's ball_source. Without the enriched copy, only 0 and 1 appear.
        """
        d = self._load()
        if "sources" not in d:
            if d["enriched"] is not None:
                codes = d["enriched"][1].astype(np.int8)
            else:
                codes = np.isfinite(d["ball"][:, 0]).astype(np.int8)
            d["sources"] = (codes + ord("0")).astype(np.uint8).tobytes().decode("ascii")
        return {"codes": d["sources"]}

    def frames(self, first, count):
        """List of per-frame dicts for [first, first + count), clipped to the match."""
        d = self._load()
        n = len(d["ball"])
        first = max(0, min(int(first), n - 1))
        last = min(n, first + max(1, min(int(count), MAX_SPAN)))
        if d["detections"] is not None:
            det_frames, det = d["detections"]
            lo, hi = np.searchsorted(det_frames, [first, last])
            det_at = {int(f): det[i] for i, f in zip(range(lo, hi), det_frames[lo:hi])}
        else:
            det_at = {}
        out = []
        for f in range(first, last):
            h = d["homography"][f]
            xy = d["players"][f]
            seen = np.flatnonzero(np.isfinite(xy[:, 0]))
            row = {
                "h": _round(h, 6) if np.all(np.isfinite(h)) else None,
                "p": [[int(s), round(float(xy[s, 0]), 5), round(float(xy[s, 1]), 5)] for s in seen],
                "b": _round(d["ball"][f], 5) if np.isfinite(d["ball"][f, 0]) else None,
            }
            if d["enriched"] is not None:
                ball, source = d["enriched"]
                if source[f] >= 2 and np.isfinite(ball[f, 0]):
                    row["e"] = _round(ball[f], 5) + [int(source[f])]
            if f in det_at:
                cands = det_at[f]
                cands = cands[np.isfinite(cands[:, 2])]
                if np.all(np.isfinite(h)) and len(cands):
                    x, y = pixels_to_pitch(h, cands[:, 0], cands[:, 1])
                    cands = cands[(x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)]  # on the pitch only
                # px, py, confidence, box size in px
                row["d"] = [[round(float(c[0]), 1), round(float(c[1]), 1), round(float(c[2]), 3),
                             round(float(c[3]), 1)] for c in cands]
            out.append(row)
        return {"first": first, "frames": out}
