"""IDSSE / DFL open data (Bassek et al. 2025, CC-BY 4.0): download and turn into arrays.

    docker compose run --rm dev python -m external.idsse download
    docker compose run --rm dev python -m external.idsse parse
    docker compose run --rm dev python -m external.idsse stats

Seven Bundesliga matches with full optical tracking at 25 fps and, per frame, BallPossession
(1 home, 2 away) and BallStatus (0 dead, 1 alive). Raw files go to external_data/idsse/ (gitignored),
one <match>.npz per match with both halves concatenated:
  players (n, slots, 2)  metres, pitch centred on (0, 0), NaN when not on the pitch
  team    (slots,)       0 home, 1 away
  ball    (n, 3)         metres, with height
  possession (n,)        0 none, 1 home, 2 away
  status  (n,)           0 dead, 1 alive
  half    (n,)           1 or 2
  pitch   (2,)           length, width in metres
"""
import argparse
import os
import shutil
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "external_data", "idsse")
# figshare answers scripts with a WAF challenge, so the official mirror is used, at a fixed revision
REVISION = "a715a38dfbaf5f58e431727c2b78d174101a703c"
URL = "https://huggingface.co/datasets/pysport/idsse-data/resolve/{}/{{}}".format(REVISION)
COMPETITION = {"J03WMX": "000001", "J03WN1": "000001", "J03WOH": "000002", "J03WOY": "000002",
               "J03WPY": "000002", "J03WQQ": "000002", "J03WR9": "000002"}
MATCHES = sorted(COMPETITION)
REMOTE = {"meta": "DFL_02_01_matchinformation", "event": "DFL_03_02_events_raw",
          "tracking": "DFL_04_03_positions_raw_observed"}
HOME, AWAY = 0, 1
HALVES = {"firstHalf": 1, "secondHalf": 2}


def raw_path(match, kind):
    return os.path.join(DATA_DIR, "{}_{}.xml".format(match, kind))


def array_path(match):
    return os.path.join(DATA_DIR, "{}.npz".format(match))


def download(_):
    os.makedirs(DATA_DIR, exist_ok=True)
    for match in MATCHES:
        for kind, prefix in REMOTE.items():
            path = raw_path(match, kind)
            if os.path.isfile(path) and os.path.getsize(path):
                continue
            name = "{}_DFL-COM-{}_DFL-MAT-{}.xml".format(prefix, COMPETITION[match], match)
            with urllib.request.urlopen(URL.format(name)) as source, open(path + ".part", "wb") as target:
                shutil.copyfileobj(source, target, 1 << 20)
            if not os.path.getsize(path + ".part"):
                raise SystemExit("empty download: {}".format(name))
            os.replace(path + ".part", path)
            print("{} {}: {:.1f} MB".format(match, kind, os.path.getsize(path) / 1e6), flush=True)


def read_meta(match):
    """(pitch (length, width), {person id: team}, ball-less team ids) from the match information."""
    root = ET.parse(raw_path(match, "meta")).getroot()
    info = root.find(".//General")
    env = root.find(".//Environment")
    pitch = (float(env.get("PitchX")), float(env.get("PitchY")))
    teams = {t.get("TeamId"): (HOME if t.get("Role") == "home" else AWAY) for t in root.iter("Team")}
    people = {}
    for team in root.iter("Team"):
        for player in team.iter("Player"):
            people[player.get("PersonId")] = teams[team.get("TeamId")]
    return pitch, people, info


def parse_match(match):
    pitch, people, _ = read_meta(match)
    # frame numbers first: the ball's frameset covers every frame of a half
    tracks, ball_sets = {}, {}
    for _, element in ET.iterparse(raw_path(match, "tracking"), events=("end",)):
        if element.tag != "FrameSet":
            continue
        half = HALVES.get(element.get("GameSection"))
        person = element.get("PersonId")
        if half is None:
            element.clear()
            continue
        frames = element.findall("Frame")
        n = np.array([int(f.get("N")) for f in frames])
        xy = np.array([[float(f.get("X")), float(f.get("Y"))] for f in frames], dtype=np.float32)
        if frames and frames[0].get("BallStatus") is not None:   # only the ball carries these
            z = np.array([float(f.get("Z", "nan")) for f in frames], dtype=np.float32)
            pos = np.array([int(f.get("BallPossession", 0)) for f in frames], dtype=np.int8)
            st = np.array([int(f.get("BallStatus", 0)) for f in frames], dtype=np.int8)
            ball_sets[half] = (n, xy, z, pos, st)
        elif person in people:   # referees have framesets too
            tracks.setdefault(person, {})[half] = (n, xy)
        element.clear()
    slots = sorted(tracks)
    team = np.array([people[s] for s in slots], dtype=np.int8)
    parts = {k: [] for k in ("players", "ball", "possession", "status", "half")}
    for half in sorted(ball_sets):
        n, xy, z, pos, st = ball_sets[half]
        index = {int(v): i for i, v in enumerate(n)}
        players = np.full((len(n), len(slots), 2), np.nan, dtype=np.float32)
        for s, person in enumerate(slots):
            if half in tracks[person]:
                pn, pxy = tracks[person][half]
                rows = np.array([index.get(int(v), -1) for v in pn])
                ok = rows >= 0
                players[rows[ok], s] = pxy[ok]
        parts["players"].append(players)
        parts["ball"].append(np.column_stack([xy, z]))
        parts["possession"].append(pos)
        parts["status"].append(st)
        parts["half"].append(np.full(len(n), half, dtype=np.int8))
    out = {k: np.concatenate(v) for k, v in parts.items()}
    np.savez_compressed(array_path(match), team=team, pitch=np.array(pitch, dtype=np.float32), **out)
    return out


def parse(_):
    for match in MATCHES:
        out = parse_match(match)
        print("{}: {} frames, {} slots".format(match, len(out["ball"]), out["players"].shape[1]), flush=True)


def load(match):
    z = np.load(array_path(match))
    return {k: z[k] for k in z.files}


def stats(_):
    for match in MATCHES:
        d = load(match)
        n = len(d["ball"])
        print("{}: {:.1f} min, alive {:.1%}, possession none/home/away {:.1%}/{:.1%}/{:.1%}, "
              "players on pitch median {:.0f}, pitch {}".format(
                  match, n / 25 / 60, (d["status"] == 1).mean(), (d["possession"] == 0).mean(),
                  (d["possession"] == 1).mean(), (d["possession"] == 2).mean(),
                  np.median(np.isfinite(d["players"][:, :, 0]).sum(axis=1)), d["pitch"].tolist()))


def to_labels(d):
    """H / A / D per frame: D where the ball is dead, else the team in possession (X if none)."""
    labels = np.where(d["possession"] == 1, "H", np.where(d["possession"] == 2, "A", "X"))
    return np.where(d["status"] == 0, "D", labels)


def spell_lengths(labels, half):
    """[(label, seconds)] of runs of equal labels within each half."""
    cut = np.flatnonzero((labels[1:] != labels[:-1]) | (half[1:] != half[:-1])) + 1
    bounds = np.concatenate([[0], cut, [len(labels)]])
    return [(labels[a], (b - a) / 25) for a, b in zip(bounds[:-1], bounds[1:])]


def label_stats(_):
    edges = (0, 2, 4, 8, 15, 30, np.inf)
    total = np.zeros(len(edges) - 1)
    print("match     D share  X share  H/A spells  per min  median s  dead spells  median s")
    for match in MATCHES:
        d = load(match)
        labels = to_labels(d)
        runs = spell_lengths(labels, d["half"])
        poss = np.array([s for l, s in runs if l in "HA"])
        dead = np.array([s for l, s in runs if l == "D"])
        total += np.histogram(poss, bins=edges)[0]
        print("{}  {:>6.1%}  {:>6.1%}  {:>10}  {:>7.1f}  {:>8.1f}  {:>11}  {:>8.1f}".format(
            match, (labels == "D").mean(), (labels == "X").mean(), len(poss),
            len(poss) / (len(labels) / 1500), np.median(poss), len(dead), np.median(dead)))
    print("possession spells by length: " + "  ".join(
        "{}-{}s {:.0%}".format(a, b, t / total.sum()) for a, b, t in zip(edges[:-1], edges[1:], total)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["download", "parse", "stats", "labels"])
    args = parser.parse_args()
    {"download": download, "parse": parse, "stats": stats, "labels": label_stats}[args.step](args)


if __name__ == "__main__":
    main()
