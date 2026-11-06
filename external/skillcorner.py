"""SkillCorner open data: 20 A-League matches of real broadcast tracking at 10 fps.

    docker compose run --rm dev python -m external.skillcorner download
    docker compose run --rm dev python -m external.skillcorner schema
    docker compose run --rm dev python -m external.skillcorner parse
    docker compose run --rm dev python -m external.skillcorner stats

Used only to check how models trained on IDSSE hold up on tracking that is really automatic. It
trains nothing. The repository is pinned to one commit, and its tracking files are served by Git LFS.
"""
import argparse
import collections
import csv
import json
import os
import urllib.request

import numpy as np

from external.idsse import ROOT

COMMIT = "4340d274572876239c154c90bc507a9b3250a656"
RAW = "https://raw.githubusercontent.com/SkillCorner/opendata/{}/data/".format(COMMIT)
LFS = "https://media.githubusercontent.com/media/SkillCorner/opendata/{}/data/".format(COMMIT)
DATA_DIR = os.path.join(ROOT, "external_data", "skillcorner")
MIN_TRACKING_BYTES = 1000000      # an LFS pointer is a few hundred bytes


def fetch(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "research-download"})
    with urllib.request.urlopen(request) as response, open(path + ".part", "wb") as handle:
        while True:
            block = response.read(1 << 20)
            if not block:
                break
            handle.write(block)
    if os.path.getsize(path + ".part") == 0:
        raise RuntimeError("empty download: " + url)
    os.replace(path + ".part", path)


def match_ids():
    with open(os.path.join(DATA_DIR, "matches.json")) as handle:
        return [str(m["id"]) for m in json.load(handle)]


def path_of(match, suffix):
    return os.path.join(DATA_DIR, "matches", match, "{}_{}".format(match, suffix))


def download():
    fetch(RAW + "matches.json", os.path.join(DATA_DIR, "matches.json"))
    for match in match_ids():
        for suffix in ("match.json", "phases_of_play.csv", "dynamic_events.csv"):
            fetch("{}matches/{}/{}_{}".format(RAW, match, match, suffix), path_of(match, suffix))
        fetch("{}matches/{}/{}_tracking_extrapolated.jsonl".format(LFS, match, match),
              path_of(match, "tracking_extrapolated.jsonl"))
        size = os.path.getsize(path_of(match, "tracking_extrapolated.jsonl"))
        if size < MIN_TRACKING_BYTES:
            raise RuntimeError("tracking of {} is {} bytes: LFS pointer?".format(match, size))
        print(match, "tracking {:.0f} MB".format(size / 1e6), flush=True)


def shape(value):
    """Keys and types of a value, without its contents."""
    if isinstance(value, dict):
        return {k: shape(v) for k, v in value.items()}
    if isinstance(value, list):
        return [shape(value[0])] if value else []
    return type(value).__name__


def schema():
    match = match_ids()[0]
    print("matches", len(match_ids()))
    with open(path_of(match, "match.json")) as handle:
        print("match.json keys", sorted(json.load(handle)))
    for suffix in ("phases_of_play.csv", "dynamic_events.csv"):
        with open(path_of(match, suffix), newline="") as handle:
            print(suffix, "columns", next(csv.reader(handle)))
    groups, ball_seen, rows, ball_keys = collections.Counter(), collections.Counter(), 0, None
    with open(path_of(match, "tracking_extrapolated.jsonl")) as handle:
        for line in handle:
            r = json.loads(line)
            if rows == 0:
                print("first record", shape(r))
            if r.get("player_data") and ball_keys is None:
                ball_keys = shape(r)
                print("first record with players", ball_keys)
            rows += 1
            groups[str((r.get("possession") or {}).get("group"))] += 1
            ball_seen[str((r.get("ball_data") or {}).get("is_detected"))] += 1
    print("records", rows, "| possession group", groups.most_common(), "| ball is_detected", ball_seen.most_common())


def in_play_frames(match):
    """Sets of frames covered by a phase of play, the only marker of the ball being in play."""
    frames = set()
    with open(path_of(match, "phases_of_play.csv"), newline="") as handle:
        for row in csv.DictReader(handle):
            frames.update(range(int(row["frame_start"]), int(row["frame_end"]) + 1))
    return frames


def labels_of(groups, in_play):
    """D out of play; in play, the last team named, or the next one at the start of a stretch."""
    out = np.full(len(groups), "D", dtype="<U1")
    a = 0
    while a < len(groups):
        if not in_play[a]:
            a += 1
            continue
        b = a
        while b < len(groups) and in_play[b]:
            b += 1
        named = [i for i in range(a, b) if groups[i]]
        if named:
            last = groups[named[0]]
            for i in range(a, b):
                last = groups[i] or last
                out[i] = last
        else:
            out[a:b] = "N"          # nobody named in the whole stretch
        a = b
    return out


def parse_match(match):
    with open(path_of(match, "match.json")) as handle:
        meta = json.load(handle)
    slots = {p["id"]: i for i, p in enumerate(meta["players"])}
    team = np.array([0 if p["team_id"] == meta["home_team"]["id"] else 1 for p in meta["players"]])
    with open(path_of(match, "tracking_extrapolated.jsonl")) as handle:
        records = [json.loads(line) for line in handle]
    by_frame = {r["frame"]: r for r in records}
    half_of = [(r["frame"], r["period"]) for r in records if r.get("period") in (1, 2)]
    spans = [(min(f for f, p in half_of if p == h), max(f for f, p in half_of if p == h)) for h in (1, 2)]
    frames = np.concatenate([np.arange(a, b + 1) for a, b in spans])
    half = np.concatenate([np.full(b - a + 1, h + 1) for h, (a, b) in enumerate(spans)])
    n = len(frames)
    players = np.full((n, len(slots), 2), np.nan, dtype=np.float32)
    seen = np.zeros((n, len(slots)), dtype=bool)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    ball_seen = np.zeros(n, dtype=bool)
    has_data = np.zeros(n, dtype=bool)
    groups = [None] * n
    code = {"home team": "H", "away team": "A"}
    for i, f in enumerate(frames.tolist()):
        r = by_frame.get(f)
        if r is None:
            continue
        for p in r.get("player_data") or []:
            s = slots.get(p["player_id"])
            if s is not None:
                players[i, s] = p["x"], p["y"]
                seen[i, s] = bool(p["is_detected"])
                has_data[i] = True
        b = r.get("ball_data") or {}
        if b.get("x") is not None:
            ball[i] = b["x"], b["y"]
            ball_seen[i] = bool(b.get("is_detected"))
        groups[i] = code.get((r.get("possession") or {}).get("group"))
    play = in_play_frames(match)
    in_play = np.array([f in play for f in frames.tolist()])
    labels = np.concatenate([labels_of(groups[a:b], in_play[a:b]) for a, b in
                             ((0, int((half == 1).sum())), (int((half == 1).sum()), n))])
    np.savez_compressed(os.path.join(DATA_DIR, match + ".npz"), players=players, seen=seen, team=team, ball=ball,
                        ball_seen=ball_seen, has_data=has_data, half=half, labels=labels,
                        named=np.array([g is not None for g in groups]),
                        pitch=np.array([meta["pitch_length"], meta["pitch_width"]], dtype=np.float64))


def parse():
    for match in match_ids():
        parse_match(match)
        print(match, flush=True)


def stats():
    keys = ("minutes", "data", "ball_xy", "ball_det", "players_det", "named", "D", "unnamed_N", "poss_per_min")
    print(" ".join("{:>11}".format(k) for k in keys))
    rows, lengths = [], []
    for match in match_ids():
        d = np.load(os.path.join(DATA_DIR, match + ".npz"))
        lab, has = d["labels"], d["has_data"]
        change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        bounds = np.concatenate([[0], change, [len(lab)]])
        for a, b in zip(bounds[:-1], bounds[1:]):
            if lab[a] in ("H", "A"):
                lengths.append((b - a) / 10)
        spells = int(np.sum((lab[1:] != lab[:-1]) & np.isin(lab[1:], ["H", "A"])))
        minutes = len(lab) / 600
        rows.append((minutes, has.mean(), np.isfinite(d["ball"][:, 0])[has].mean(), d["ball_seen"][has].mean(),
                     d["seen"][has].sum(axis=1).mean(), d["named"].mean(), (lab == "D").mean(),
                     (lab == "N").mean(), spells / minutes))
    r = np.array(rows)
    for name, line in (("mean", r.mean(axis=0)), ("min", r.min(axis=0)), ("max", r.max(axis=0))):
        print(" ".join("{:>11.3f}".format(v) for v in line), name)
    lengths = np.array(lengths)
    print("spells", len(lengths), "| < 2 s {:.1%} | < 8 s {:.1%} | median {:.1f} s".format(
        np.mean(lengths < 2), np.mean(lengths < 8), np.median(lengths)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["download", "schema", "parse", "stats"])
    args = parser.parse_args()
    {"download": download, "schema": schema, "parse": parse, "stats": stats}[args.step]()


if __name__ == "__main__":
    main()
