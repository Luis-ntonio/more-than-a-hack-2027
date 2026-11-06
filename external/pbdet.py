"""PBDet-Soccer (Wu, Zenodo 10.5281/zenodo.21821917, CC-BY 4.0): boxes of players and the ball.

    docker compose run --rm dev python -m external.pbdet

Downloads the archive to external_data/pbdet/ (gitignored), checks its MD5 and prints aggregates of
the annotations: images and boxes per split, and how big the ball is. Used to train the ball
detector; the provided match plays no part in it.
"""
import collections
import glob
import hashlib
import os
import shutil
import urllib.request
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "external_data", "pbdet")
URL = "https://zenodo.org/api/records/21821917/files/soccer_player.zip/content"
MD5 = "903971d11430d69c535a8eeaaa865c89"


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "soccer_player.zip")
    if not os.path.isfile(path):
        request = urllib.request.Request(URL, headers={"User-Agent": "python-urllib"})
        with urllib.request.urlopen(request) as source, open(path + ".part", "wb") as target:
            shutil.copyfileobj(source, target, 1 << 20)
        os.replace(path + ".part", path)
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != MD5:
        raise SystemExit("bad checksum: {}".format(path))
    return path


def main():
    path = download()
    print("checksum OK, {:.2f} GB".format(os.path.getsize(path) / 1e9))
    if not glob.glob(os.path.join(DATA_DIR, "**", "data.yaml"), recursive=True):
        with zipfile.ZipFile(path) as archive:
            archive.extractall(DATA_DIR)
    yaml_path = glob.glob(os.path.join(DATA_DIR, "**", "data.yaml"), recursive=True)[0]
    print(os.path.relpath(yaml_path, DATA_DIR))
    with open(yaml_path) as handle:
        print(handle.read()[:400])
    folders = collections.defaultdict(lambda: [0, collections.Counter(), []])
    for label in glob.glob(os.path.join(DATA_DIR, "**", "*.txt"), recursive=True):
        parts = os.path.relpath(label, DATA_DIR).replace("\\", "/").split("/")
        split = "/".join(parts[:-1])
        entry = folders[split]
        entry[0] += 1
        with open(label) as handle:
            for line in handle:
                f = line.split()
                if len(f) >= 5:
                    entry[1][f[0]] += 1
                    entry[2].append((int(f[0]), float(f[3]), float(f[4])))
    for split, (n, classes, boxes) in sorted(folders.items()):
        print("{:<28} {:>6} label files  boxes by class {}".format(split, n, dict(classes)))
        b = np.array(boxes) if boxes else np.zeros((0, 3))
        for c in sorted(set(b[:, 0].astype(int).tolist())):
            s = np.sqrt(b[b[:, 0] == c, 1] * b[b[:, 0] == c, 2])
            print("    class {}: box side as a share of the image, p10/p50/p90 {:.4f} / {:.4f} / {:.4f}".format(
                c, *np.percentile(s, [10, 50, 90])))


if __name__ == "__main__":
    main()
