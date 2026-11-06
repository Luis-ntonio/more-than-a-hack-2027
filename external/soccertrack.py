"""SoccerTrack v2 (CC-BY 4.0, gated on Hugging Face): panoramic full-pitch video with ball annotations.

    docker compose run --rm dev python -m external.soccertrack annotations   # small files, prints their shape

Used only to train and tune the ball detector, never for possession. Files go to
external_data/soccertrack/ (gitignored). The Hugging Face token is read from external_data/.hf_token
and is never printed.
"""
import argparse
import json
import os
import shutil
import urllib.error
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "external_data", "soccertrack")
TOKEN_PATH = os.path.join(ROOT, "external_data", ".hf_token")
REVISION = "325bd4c57ee8e135ab652f3921d274526ada496b"
REPO = "datasets/atomscott/soccertrack-v2"
API = "https://huggingface.co/api/{}/tree/{}/".format(REPO, REVISION)
FILE = "https://huggingface.co/{}/resolve/{}/".format(REPO, REVISION)


def request(url):
    with open(TOKEN_PATH) as handle:
        token = handle.read().strip()
    return urllib.request.Request(url, headers={"Authorization": "Bearer " + token})


def fetch(remote):
    """Download one file of the dataset, unless it is already here. Returns its local path."""
    path = os.path.join(DATA_DIR, remote)
    if os.path.isfile(path) and os.path.getsize(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with urllib.request.urlopen(request(FILE + remote)) as source, open(path + ".part", "wb") as target:
        shutil.copyfileobj(source, target, 1 << 20)
    if not os.path.getsize(path + ".part"):
        raise SystemExit("empty download: {}".format(remote))
    os.replace(path + ".part", path)
    return path


def listing(folder):
    with urllib.request.urlopen(request(API + folder + "?recursive=true")) as response:
        return [e for e in json.load(response) if e["type"] == "file"]


def annotations(_):
    try:
        files = listing("ball")
        path = fetch(files[0]["path"])
    except urllib.error.HTTPError as error:
        raise SystemExit("access failed: HTTP {}".format(error.code))
    print("access OK; {} ball files".format(len(files)))
    for entry in files:
        fetch(entry["path"])
    z = np.load(path, allow_pickle=False)
    print("keys of {}:".format(os.path.basename(path)))
    for key in z.files:
        a = z[key]
        line = "  {:<16} shape {} dtype {}".format(key, a.shape, a.dtype)
        if a.dtype.kind in "fiu" and a.size:
            finite = a[np.isfinite(a)] if a.dtype.kind == "f" else a
            line += "  finite {:.1%}  min {:.3f}  max {:.3f}".format(
                finite.size / a.size, float(finite.min()), float(finite.max()))
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["annotations"])
    args = parser.parse_args()
    {"annotations": annotations}[args.step](args)


if __name__ == "__main__":
    main()
