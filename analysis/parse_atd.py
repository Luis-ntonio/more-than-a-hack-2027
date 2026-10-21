"""Parse tactical + broadcast ATD into npz caches (scratchpad only, never in the repo)."""
import sys, os
import xml.etree.ElementTree as ET
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "cache")
os.makedirs(OUT, exist_ok=True)


def teams(meta):
    r = ET.parse(meta).getroot()
    ids = [t.get("id") for t in r.iter("Team")]
    slots = [p.get("teamId") for p in r.iter("Player")]
    return np.array([0 if s == ids[0] else 1 for s in slots], dtype=np.int8)


def parse(src):
    d = os.path.join(ROOT, "data", src)
    meta = os.path.join(d, f"BAR-ATH_{src}_atd.xml")
    raw = os.path.join(d, f"BAR-ATH_{src}_atd.txt")
    team = teams(meta)
    n_slots = len(team)
    lines = open(raw).read().split("\n")
    lines = [l for l in lines if l]
    n = len(lines)
    P = np.full((n, n_slots, 3), np.nan, dtype=np.float32)
    B = np.full((n, 3), np.nan, dtype=np.float32)
    ids = np.zeros(n, dtype=np.int64)
    for i, l in enumerate(lines):
        a, b, c = l.split(":")
        ids[i] = int(a)
        P[i] = np.array(b.replace(";", ",").split(","), dtype=np.float32).reshape(n_slots, 3)
        B[i] = np.array(c.split(","), dtype=np.float32)
    np.savez_compressed(os.path.join(OUT, f"atd_{src}.npz"), P=P, B=B, ids=ids, team=team)
    print(src, n, n_slots, "team0:", int((team == 0).sum()), "team1:", int((team == 1).sum()))


for s in sys.argv[1:]:
    parse(s)
