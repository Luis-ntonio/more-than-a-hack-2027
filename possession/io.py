"""Read the provided assets into numpy arrays, with an on-disk cache.

Everything is indexed by tactical video frame (= submission row). The cache lives in ./cache,
which is gitignored: it is derived match data and must never be published.
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, anchor_video_frame, half_video_blocks, require  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "cache")
FPS = GAMES[GAME]["fps"]
PITCH = GAMES[GAME]["pitch"]  # metres (length, width)
HOME, AWAY = 0, 1


def _track_teams(meta_path):
    """0 for home slots, 1 for away slots. The first team in the metadata is home."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)


def load_atd(game=GAME):
    """Tactical ATD as arrays: players (n, slots, 3) x/y/speed, ball (n, 2) x/y, team (slots,).

    Coordinates stay normalised 0-1 here; features.py converts to metres.
    """
    cache = os.path.join(CACHE_DIR, "{}_atd_tactical.npz".format(game))
    if os.path.isfile(cache):
        z = np.load(cache)
        return {"players": z["players"], "ball": z["ball"], "team": z["team"]}

    meta_path, raw_path = require(["atd_meta", "atd_raw"], game)
    team = _track_teams(meta_path)
    n_frames = GAMES[game]["n_video_frames"]
    players = np.full((n_frames, len(team), 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            frame = int(parts[0]) - 1  # frame_id counts from 1, video frames from 0
            values = np.array(parts[1].replace(";", ",").split(","), dtype=np.float32)
            players[frame] = values.reshape(len(team), 3)
            ball[frame] = np.array(parts[2].split(",")[:2], dtype=np.float32)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache, players=players, ball=ball, team=team)
    return {"players": players, "ball": ball, "team": team}


def load_homography(game=GAME):
    """(n, 9) homography coefficients, row-major, NaN where the frame has no valid matrix."""
    cache = os.path.join(CACHE_DIR, "{}_homography_tactical.npy".format(game))
    if os.path.isfile(cache):
        return np.load(cache)
    path = require(["homography"], game)[0]
    frame = pd.read_csv(path)
    cols = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]
    matrix = frame[cols].to_numpy(dtype=np.float64)
    matrix[frame["valid"].to_numpy() == 0] = np.nan
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(cache, matrix)
    return matrix


def load_smart_tagging(game=GAME):
    """Smart Tagging instances as a DataFrame with video-frame bounds f0/f1."""
    path = require(["smart_tagging"], game)[0]
    rows = []
    for inst in ET.parse(path).getroot().iter("instance"):
        row = {"start": float(inst.findtext("start")), "end": float(inst.findtext("end")),
               "code": inst.findtext("code").strip().upper()}
        for label in inst.findall("label"):
            row[label.findtext("group")] = label.findtext("text")
        rows.append(row)
    tags = pd.DataFrame(rows)
    tags["f0"] = (tags["start"] * FPS).round().astype(int)
    tags["f1"] = (tags["end"] * FPS).round().astype(int)
    return tags


def graded_blocks(game=GAME):
    """[(half, first_frame, last_frame)] inclusive, in video frames."""
    return half_video_blocks(game)


def kickoff_frames(game=GAME):
    """{half: video frame of the kickoff}, from the anchors the organisers publish in games.py."""
    entry = GAMES[game]
    return {1: anchor_video_frame(entry["kickoff"]), 2: anchor_video_frame(entry["kickoff_2nd"])}
