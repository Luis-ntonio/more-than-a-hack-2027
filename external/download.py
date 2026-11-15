"""Downloads that are complete and identical to the ones the submission was built from.

A server can close a connection early without an error, and the file then looks fine until it is
parsed (it happened with a SkillCorner tracking file). So every file is checked against the size the
server announces and against the SHA-256 pinned in external/downloads.sha256, and fetched again up
to ATTEMPTS times. A file already on disk is checked too.
"""
import hashlib
import os
import urllib.request

from external.idsse import ROOT

MANIFEST = os.path.join(ROOT, "external", "downloads.sha256")
ATTEMPTS = 3


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pinned():
    """{path relative to the repository: SHA-256}."""
    out = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as handle:
            for line in handle:
                if line.strip():
                    digest, rel = line.split(None, 1)
                    out[rel.strip().lstrip("*")] = digest
    return out


def fetch(url, path, headers=None):
    """Download url to path unless a good copy is there. Raises if it cannot get a good one."""
    rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
    expected = pinned().get(rel)
    if os.path.exists(path) and os.path.getsize(path) > 0 and (expected is None or sha256(path) == expected):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    problem = None
    for _ in range(ATTEMPTS):
        request = urllib.request.Request(url, headers=dict({"User-Agent": "research-download"}, **(headers or {})))
        with urllib.request.urlopen(request) as response, open(path + ".part", "wb") as handle:
            announced = response.headers.get("Content-Length")
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                handle.write(block)
        size = os.path.getsize(path + ".part")
        if size == 0:
            problem = "empty"
        elif announced is not None and size != int(announced):
            problem = "{} of {} bytes".format(size, announced)
        elif expected is not None and sha256(path + ".part") != expected:
            problem = "SHA-256 differs from the pinned one"
        else:
            os.replace(path + ".part", path)
            return
    raise RuntimeError("bad download of {}: {}".format(rel, problem))
