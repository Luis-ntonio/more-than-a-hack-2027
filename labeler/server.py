"""Local labelling tool: streams the tactical video and stores hand labels as change points.

    docker compose up labeler        ->  http://localhost:8765

Everything stays on this machine. The video is read from ./data (mounted read-only) and the labels
go to ./labels/<block>__<annotator>.json, which is gitignored. A label file only stores the frames
where the label changes; every frame up to the next change inherits it.
"""
import argparse
import datetime
import json
import mimetypes
import os
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from games import GAME, GAMES, video_path  # noqa: E402
from possession.io import FPS, graded_blocks, kickoff_frames  # noqa: E402

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
LABELS_DIR = os.path.join(ROOT, "labels")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "{}_possession_gt_public.csv".format(GAME))
PREDICTIONS_DIR = os.path.join(ROOT, "evaluation", "results")  # make_submission writes here

BLOCK_FRAMES = 5 * 60 * FPS
MIN_TAIL_FRAMES = 60 * FPS      # a leftover shorter than this is merged into the previous block
CHUNK = 1 << 20
MAX_RANGE = 16 << 20            # cap open-ended ranges; the browser asks again when it needs more
ANNOTATOR_RE = re.compile(r"^[a-z0-9_-]{1,32}$")
LABELS = ("H", "A", "D")


def _public_gt():
    """Full-length GT codes and the [first, last] frames it grades, or (None, None) if absent."""
    if not os.path.isfile(PUBLIC_GT):
        return None, None
    codes = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    graded = np.flatnonzero(codes != pm.X)
    return codes, (int(graded[0]), int(graded[-1])) if len(graded) else None


def build_blocks(public_range):
    """5-minute blocks from each half's kickoff to the end of its graded window."""
    kickoffs = kickoff_frames()
    blocks = []
    for half, first, last in graded_blocks():
        start = max(first, kickoffs[half])
        bounds = list(range(start, last + 1, BLOCK_FRAMES)) + [last + 1]
        if len(bounds) > 2 and bounds[-1] - bounds[-2] < MIN_TAIL_FRAMES:
            del bounds[-2]
        for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
            minute = (lo - kickoffs[half]) / FPS / 60 + (45 if half == 2 else 0)
            public = bool(public_range and lo <= public_range[1] and hi - 1 >= public_range[0])
            blocks.append({
                "id": "{}T-{:02d}".format(half, i), "half": half, "first": lo, "last": hi - 1,
                "minute_from": round(minute, 1),
                "minute_to": round(minute + (hi - lo) / FPS / 60, 1),
                "public": public,
            })
    return blocks


def label_path(block_id, annotator):
    return os.path.join(LABELS_DIR, "{}__{}.json".format(block_id, annotator))


def dense(changes, first, last):
    """Change points -> one label per frame of [first, last]; frames before the first change are X."""
    out = np.full(last - first + 1, "X", dtype="<U1")
    for (frame, label), nxt in zip(changes, changes[1:] + [[last + 1, None]]):
        out[frame - first:nxt[0] - first] = label
    return out


def validate(doc, blocks_by_id):
    block = blocks_by_id.get(doc.get("block"))
    if block is None:
        return "unknown block"
    if not ANNOTATOR_RE.match(str(doc.get("annotator", ""))):
        return "annotator must match {}".format(ANNOTATOR_RE.pattern)
    changes = doc.get("changes")
    if not isinstance(changes, list):
        return "changes must be a list"
    previous = None
    for item in changes:
        if (not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], int)
                or item[1] not in LABELS):
            return "each change must be [frame, 'H'|'A'|'D']"
        if not block["first"] <= item[0] <= block["last"]:
            return "change at {} is outside the block".format(item[0])
        if previous is not None and item[0] <= previous:
            return "changes must be strictly increasing"
        previous = item[0]
    return None


def changes_of(labels, first):
    """Dense labels for a window starting at `first` -> change points [frame, label], X dropped."""
    edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    return [[first + int(i), str(labels[i])] for i in np.concatenate(([0], edges))
            if labels[i] in LABELS]


class App:
    def __init__(self):
        self._predictions = {}
        self.video = video_path()
        self.gt, public_range = _public_gt()
        self.blocks = build_blocks(public_range)
        self.blocks_by_id = {b["id"]: b for b in self.blocks}
        self.config = {
            "game": GAME, "fps": FPS, "n_frames": GAMES[GAME]["n_video_frames"],
            "blocks": self.blocks, "kickoffs": kickoff_frames(),
            "video_available": os.path.isfile(self.video),
        }

    def status(self):
        """Per block: who has labelled it, how much, and whether they marked it done."""
        rows = {}
        if os.path.isdir(LABELS_DIR):
            for name in sorted(os.listdir(LABELS_DIR)):
                match = re.match(r"^(.+)__([a-z0-9_-]+)\.json$", name)
                if not match or match.group(1) not in self.blocks_by_id:
                    continue
                with open(os.path.join(LABELS_DIR, name)) as handle:
                    doc = json.load(handle)
                block = self.blocks_by_id[match.group(1)]
                labels = dense(doc["changes"], block["first"], block["last"])
                rows.setdefault(match.group(1), []).append({
                    "annotator": match.group(2), "done": bool(doc.get("done")),
                    "coverage": round(float((labels != "X").mean()), 4),
                    "updated": doc.get("updated"), "seeded_from": doc.get("seeded_from", ""),
                })
        return rows

    def prediction_names(self):
        if not os.path.isdir(PREDICTIONS_DIR):
            return []
        return sorted(name[:-len(".csv")] for name in os.listdir(PREDICTIONS_DIR)
                      if name.endswith(".csv"))

    def prediction(self, name, block_id):
        """A model's labels on one block, as change points. Reloaded when the CSV changes."""
        block = self.blocks_by_id.get(block_id)
        if block is None or name not in self.prediction_names():
            return None
        path = os.path.join(PREDICTIONS_DIR, name + ".csv")
        mtime = os.path.getmtime(path)
        if name not in self._predictions or self._predictions[name][0] != mtime:
            frame = pm.read_label_csv(path)
            labels = np.array([str(v).strip().upper() for v in frame["label"].values], dtype="<U1")
            self._predictions[name] = (mtime, labels)
        labels = self._predictions[name][1][block["first"]:block["last"] + 1]
        return changes_of(labels, block["first"])

    def compare(self, block_id, annotator, reveal):
        """Score one annotator's labels on the public block against the official GT."""
        block = self.blocks_by_id.get(block_id)
        if block is None or not block["public"] or self.gt is None:
            return None
        changes = []
        path = label_path(block_id, annotator)
        if ANNOTATOR_RE.match(annotator) and os.path.isfile(path):
            with open(path) as handle:
                changes = json.load(handle)["changes"]
        n = self.config["n_frames"]
        first, last = block["first"], block["last"]
        pred = np.full(n, "X", dtype="<U1")
        pred[first:last + 1] = dense(changes, first, last)
        gt = np.full(n, pm.X, dtype=self.gt.dtype)
        gt[first:last + 1] = self.gt[first:last + 1]
        result = pm.evaluate(gt, pm.label_codes(pred), bootstrap_n=0)
        out = {key: result[key] for key in (
            "spell_f1_mean", "macro_f1", "alive_dead_f1", "n_gt_spells", "n_pred_spells",
            "n_graded", "n_abstain_on_graded")}
        if reveal:
            out["gt_changes"] = changes_of(np.array(pm.LABELS)[self.gt[first:last + 1]], first)
        return out


class Handler(BaseHTTPRequestHandler):
    app = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        if not self.path.startswith("/video"):
            sys.stderr.write("[labeler] " + fmt % args + "\n")

    # -- helpers ---------------------------------------------------------------------------------

    def _json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status, message):
        self._json({"error": message}, status)

    def _static(self, rel):
        path = os.path.realpath(os.path.join(STATIC_DIR, rel))
        if not path.startswith(os.path.realpath(STATIC_DIR) + os.sep) or not os.path.isfile(path):
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        with open(path, "rb") as handle:
            body = handle.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(path)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _video(self):
        path = self.app.video
        if not os.path.isfile(path):
            return self._error(HTTPStatus.NOT_FOUND, "video not found: " + path)
        size = os.path.getsize(path)
        start, end = 0, size - 1
        match = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if match:
            if match.group(1):
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else size - 1
            else:  # suffix range: the last N bytes
                start = max(0, size - int(match.group(2)))
            end = min(end, size - 1, start + MAX_RANGE - 1)
            if start > end:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", "bytes */{}".format(size))
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", "bytes {}-{}/{}".format(start, end, size))
        else:
            self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        try:
            with open(path, "rb") as handle:
                handle.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = handle.read(min(CHUNK, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # the browser dropped the request after a seek; that is normal

    # -- routes ----------------------------------------------------------------------------------

    def do_GET(self):
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/":
            return self._static("index.html")
        if url.path.startswith("/static/"):
            return self._static(url.path[len("/static/"):])
        if url.path == "/video":
            return self._video()
        if url.path == "/api/config":
            return self._json(self.app.config)
        if url.path == "/api/status":
            return self._json(self.app.status())
        if url.path == "/api/labels":
            block, annotator = query.get("block"), query.get("annotator", "")
            if block not in self.app.blocks_by_id or not ANNOTATOR_RE.match(annotator):
                return self._error(HTTPStatus.BAD_REQUEST, "bad block or annotator")
            path = label_path(block, annotator)
            if not os.path.isfile(path):
                return self._json({"block": block, "annotator": annotator, "changes": [],
                                   "done": False, "notes": ""})
            with open(path) as handle:
                return self._json(json.load(handle))
        if url.path == "/api/predictions":
            return self._json(self.app.prediction_names())
        if url.path == "/api/prediction":
            changes = self.app.prediction(query.get("name"), query.get("block"))
            if changes is None:
                return self._error(HTTPStatus.BAD_REQUEST, "unknown prediction or block")
            return self._json(changes)
        if url.path == "/api/compare":
            result = self.app.compare(query.get("block"), query.get("annotator", ""),
                                      query.get("reveal") == "1")
            if result is None:
                return self._error(HTTPStatus.BAD_REQUEST, "only the public block has GT")
            return self._json(result)
        return self._error(HTTPStatus.NOT_FOUND, "not found")

    def do_PUT(self):
        if urlparse(self.path).path != "/api/labels":
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        try:
            doc = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        except (ValueError, json.JSONDecodeError):
            return self._error(HTTPStatus.BAD_REQUEST, "invalid JSON")
        problem = validate(doc, self.app.blocks_by_id)
        if problem:
            return self._error(HTTPStatus.BAD_REQUEST, problem)
        block = self.app.blocks_by_id[doc["block"]]
        record = {
            "block": doc["block"], "annotator": doc["annotator"],
            "first": block["first"], "last": block["last"],
            "changes": doc["changes"], "done": bool(doc.get("done")),
            "notes": str(doc.get("notes", ""))[:5000],
            # which model's output the annotator started from, if any: such a block is anchored
            # to that model, so export can keep it out of validation (--exclude-seeded)
            "seeded_from": str(doc.get("seeded_from", ""))[:64],
            "updated": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        os.makedirs(LABELS_DIR, exist_ok=True)
        path = label_path(doc["block"], doc["annotator"])
        with open(path + ".tmp", "w") as handle:
            json.dump(record, handle, indent=1)
        os.replace(path + ".tmp", path)
        return self._json({"ok": True, "updated": record["updated"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    Handler.app = App()
    if not Handler.app.config["video_available"]:
        print("[labeler] WARNING: tactical video not found at " + Handler.app.video)
    print("[labeler] {} blocks, open http://localhost:{}".format(len(Handler.app.blocks), args.port))
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
