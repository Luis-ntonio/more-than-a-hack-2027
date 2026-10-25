"""Team of each detected player from the colour of its shirt, trained on the ATD's clear-cut cases.

    docker compose run --rm gpu python -m enrich.detect_players --frames labelled   # first
    docker compose run --rm gpu python -m enrich.jersey_team features   # colour features per box
    docker compose run --rm gpu python -m enrich.jersey_team train      # classifier + evaluation

possession/control.py names a box's team after the nearest ATD player to its feet, which fails in
duels, where a rival stands as close. Here a box is labelled with that team only when it is clear
cut -- the nearest ATD player within CLEAR_M and no player of the other team within RIVAL_M -- and
a classifier learns the shirt colour: an HSV histogram of the torso with the grass masked out, and
the mean and spread of its colour. It is trained on the 1st-half blocks and evaluated on the
2nd-half ones (the teams have changed ends), on clear-cut boxes and on duels, where it is the
ATD it disagrees with.

Writes per-box P(home) for every player box of the detection file to
cache/enriched/<game>_jersey_team_<frames>.npz (NaN where not a player or not classified).
Everything runs locally; output is derived match data under cache/ (gitignored).
"""
import argparse
import os

import cv2
import lightgbm as lgb
import numpy as np

from enrich.detect_ball import OUT_DIR, read_frames
from enrich.geometry import pixels_to_pitch
from games import GAME, video_path
from possession.control import GOALKEEPER, PLAYER, detections_path
from possession.io import AWAY, HOME, PITCH, ROOT, kickoff_frames, load_atd, load_homography

CLEAR_M = 1.0
RIVAL_M = 3.0
H_BINS, S_BINS, V_BINS = 8, 4, 4
TORSO = (0.15, 0.50, 0.25, 0.75)   # rows and columns of the box, as fractions
GRASS_H = (30, 90)                 # OpenCV hue range of the pitch
LGB_PARAMS = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 40,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "seed": 0,
              "deterministic": True, "force_row_wise": True, "num_threads": 8, "verbose": -1}
N_ROUNDS = 300


def features_path(frames):
    return os.path.join(OUT_DIR, "{}_jersey_features_{}.npz".format(GAME, frames))


def output_path(frames):
    return os.path.join(OUT_DIR, "{}_jersey_team_{}.npz".format(GAME, frames))


def box_features(image, box):
    """Colour features of a box's torso, or None when it is too small or all grass."""
    x1, y1, x2, y2 = box[:4]
    w, h = x2 - x1, y2 - y1
    r0, r1, c0, c1 = TORSO
    ya, yb = int(y1 + r0 * h), int(y1 + r1 * h)
    xa, xb = int(x1 + c0 * w), int(x1 + c1 * w)
    crop = image[max(ya, 0):max(yb, 0), max(xa, 0):max(xb, 0)]
    if crop.size < 3 * 16:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    keep = ~((hsv[:, 0] >= GRASS_H[0]) & (hsv[:, 0] <= GRASS_H[1]) & (hsv[:, 1] > 60))
    if keep.sum() < 8:
        return None
    hsv = hsv[keep].astype(np.float32)
    hist, _ = np.histogramdd(hsv, bins=(H_BINS, S_BINS, V_BINS), range=((0, 180), (0, 256), (0, 256)))
    lab = cv2.cvtColor(crop.reshape(-1, 1, 3)[keep], cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return np.concatenate([hist.ravel() / keep.sum(), lab.mean(axis=0) / 255, lab.std(axis=0) / 255])


def atd_labels(frame, boxes, h, players_m, team):
    """(team, clear) per box: the nearest ATD player's team within RIVAL_M, and whether it is
    clear cut (nearest within CLEAR_M, no rival within RIVAL_M)."""
    n = len(boxes)
    out_team, clear = np.full(n, -1), np.zeros(n, dtype=bool)
    seen = np.isfinite(players_m[:, 0])
    if not seen.any() or not np.all(np.isfinite(h)):
        return out_team, clear
    x, y = pixels_to_pitch(h, (boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3])
    feet = np.stack([x, y], axis=1) * np.array(PITCH, dtype=np.float64)
    dist = np.linalg.norm(feet[:, None, :] - players_m[None, seen], axis=2)
    teams = team[seen]
    for i in range(n):
        if not np.isfinite(dist[i]).all():
            continue
        j = int(np.argmin(dist[i]))
        if dist[i, j] >= RIVAL_M:
            continue
        out_team[i] = teams[j]
        rival = dist[i][teams != teams[j]]
        clear[i] = dist[i, j] < CLEAR_M and (not len(rival) or rival.min() >= RIVAL_M)
    return out_team, clear


def features(args):
    z = np.load(detections_path(args.frames))
    frames, boxes = z["frames"], z["boxes"]
    atd = load_atd()
    homography = load_homography()
    players = atd["players"][:, :, :2].astype(np.float64) * np.array(PITCH, dtype=np.float64)
    n_feat = H_BINS * S_BINS * V_BINS + 6
    feats = np.full(boxes.shape[:2] + (n_feat,), np.nan, dtype=np.float16)
    team = np.full(boxes.shape[:2], -1, dtype=np.int8)
    clear = np.zeros(boxes.shape[:2], dtype=bool)
    row_of = {int(f): i for i, f in enumerate(frames)}
    for frame, image in read_frames(video_path(), frames):
        if image is None:
            continue
        i = row_of[frame]
        b = boxes[i]
        people = np.flatnonzero(np.isfinite(b[:, 4]) & ((b[:, 5] == PLAYER) | (b[:, 5] == GOALKEEPER)))
        if not len(people):
            continue
        t, c = atd_labels(frame, b[people], homography[frame], players[frame], atd["team"])
        team[i, people], clear[i, people] = t, c
        for k in people:
            f = box_features(image, b[k])
            if f is not None:
                feats[i, k] = f
    np.savez_compressed(features_path(args.frames), frames=frames, feats=feats, team=team, clear=clear)
    has = np.isfinite(feats[:, :, 0])
    print("{} frames, {} player boxes with features; ATD team on {:.0%}, clear cut {:.0%}".format(
        len(frames), int(has.sum()), (team[has] >= 0).mean(), clear[has].mean()))


def train(args):
    z = np.load(features_path(args.frames))
    frames, feats, team, clear = z["frames"], z["feats"].astype(np.float32), z["team"], z["clear"]
    cls = np.load(detections_path(args.frames))["boxes"][:, :, 5]
    has = np.isfinite(feats[:, :, 0]) & (cls == PLAYER)   # goalkeepers keep the ATD's team
    half2 = frames >= kickoff_frames()[2]
    rows, cols = np.nonzero(has)
    second = half2[rows]
    X = feats[rows, cols]
    y = team[rows, cols]
    c = clear[rows, cols]
    fit = ~second & c & (y >= 0)
    model = lgb.train(LGB_PARAMS, lgb.Dataset(X[fit], (y[fit] == HOME).astype(int)), N_ROUNDS)
    p = model.predict(X)
    pred = np.where(p >= 0.5, HOME, AWAY)
    print("trained on {} clear-cut player boxes of the 1st half".format(int(fit.sum())))
    for name, m in (("2nd half, clear cut", second & c & (y >= 0)),
                    ("2nd half, duels (ATD unsure)", second & ~c & (y >= 0)),
                    ("1st half, duels (ATD unsure)", ~second & ~c & (y >= 0))):
        conf = np.abs(p[m] - 0.5) > 0.4
        print("  {:<30} n={:>7}  agrees with ATD {:.1%}  (confident {:.0%} of them: agrees {:.1%})".format(
            name, int(m.sum()), np.mean(pred[m] == y[m]), conf.mean(), np.mean(pred[m][conf] == y[m][conf])))
    # out-of-sample for the 1st half too: refit on the 2nd half's clear-cut boxes
    fit2 = second & c & (y >= 0)
    model2 = lgb.train(LGB_PARAMS, lgb.Dataset(X[fit2], (y[fit2] == HOME).astype(int)), N_ROUNDS)
    p_out = np.where(second, p, model2.predict(X))
    out = np.full(team.shape, np.nan, dtype=np.float32)
    out[rows, cols] = p_out
    np.savez_compressed(output_path(args.frames), frames=frames, p_home=out)
    print("1st half from the model trained on the 2nd and vice versa -> " +
          os.path.relpath(output_path(args.frames), ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["features", "train"])
    parser.add_argument("--frames", default="labelled", choices=["labelled", "graded"])
    args = parser.parse_args()
    {"features": features, "train": train}[args.step](args)


if __name__ == "__main__":
    main()
