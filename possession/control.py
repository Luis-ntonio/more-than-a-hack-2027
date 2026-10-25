"""Who has the ball, from the video detector's player and ball boxes.

Each player or goalkeeper box gets a team from the ATD: its foot point (bottom centre) is projected
onto the pitch and matched to the nearest ATD player within ASSIGN_M. The ball is the detector's
ball box nearest to the ATD ball (within BALL_MATCH_M), or its most confident one on the pitch
where the ATD has none. A box is in contact with the ball when the ball is within CONTACT box
heights of it; the frame's control is the team of the contacts, or CONTESTED if both teams touch it.

Needs enrich.detect_players. Its output is a fixed artefact of the submission, like the ball
detections (enrich/artefacts.sha256): this module only reads it.
"""
import os

import numpy as np

from enrich.geometry import pixels_to_pitch
from possession.io import AWAY, CACHE_DIR, GAME, HOME, PITCH, load_atd, load_homography

ENRICHED_DIR = os.path.join(CACHE_DIR, "enriched")
ASSIGN_M = 3.0
BALL_MATCH_M = 3.0
CONTACT = 0.15      # in box heights, outside the box
JERSEY_SURE = 0.1   # jersey=True: trust the shirt colour when P(home) is below this or above 1 - it
NONE, CONTESTED = -1, -2
BALL, GOALKEEPER, PLAYER = 0, 1, 2


def detections_path(frames="labelled"):
    return os.path.join(ENRICHED_DIR, "{}_player_detections_{}.npz".format(GAME, frames))


def jersey_path(frames="labelled"):
    return os.path.join(ENRICHED_DIR, "{}_jersey_team_{}.npz".format(GAME, frames))


def available(frames="labelled"):
    return os.path.isfile(detections_path(frames))


def _team_of_boxes(h, boxes, players_m, team):
    """Team (HOME / AWAY / NONE) of each box from the nearest ATD player to its foot point."""
    foot_x, foot_y = (boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]
    x, y = pixels_to_pitch(h, foot_x, foot_y)
    feet = np.stack([x, y], axis=1) * np.array(PITCH, dtype=np.float64)
    seen = np.isfinite(players_m[:, 0])
    if not seen.any() or not np.isfinite(feet).all():
        return np.full(len(boxes), NONE)
    dist = np.linalg.norm(feet[:, None, :] - players_m[None, seen], axis=2)
    best = dist.argmin(axis=1)
    return np.where(dist[np.arange(len(boxes)), best] < ASSIGN_M, team[seen][best], NONE)


def control(frames="labelled", contact=CONTACT, jersey=False):
    """(n_frames,) control per frame: HOME, AWAY, CONTESTED, or NONE (no ball, no contact, or no
    detections for that frame), plus the share of frames with a ball box and with a team-assigned
    box, for reporting. jersey=True takes a player box's team from its shirt colour
    (enrich.jersey_team) where the classifier is sure (P within JERSEY_SURE of 0 or 1), the ATD's
    elsewhere."""
    z = np.load(detections_path(frames))
    p_home = None
    if jersey:
        j = np.load(jersey_path(frames))
        assert np.array_equal(j["frames"], z["frames"])
        p_home = j["p_home"]
    atd = load_atd()
    homography = load_homography()
    scale = np.array(PITCH, dtype=np.float64)
    players = atd["players"][:, :, :2].astype(np.float64) * scale
    tracking = atd["ball"].astype(np.float64) * scale
    out = np.full(len(atd["ball"]), NONE)
    with_ball, assigned = 0, []
    for row, (frame, boxes) in enumerate(zip(z["frames"], z["boxes"])):
        h = homography[frame]
        valid = np.isfinite(boxes[:, 4])
        boxes = boxes[valid]
        if not len(boxes) or not np.all(np.isfinite(h)):
            continue
        balls = boxes[boxes[:, 5] == BALL]
        is_person = (boxes[:, 5] == PLAYER) | (boxes[:, 5] == GOALKEEPER)
        people = boxes[is_person]
        if not len(balls) or not len(people):
            continue
        bx, by = (balls[:, 0] + balls[:, 2]) / 2, (balls[:, 1] + balls[:, 3]) / 2
        x, y = pixels_to_pitch(h, bx, by)
        on_pitch = (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)
        if np.isfinite(tracking[frame, 0]):
            dist = np.linalg.norm(np.stack([x, y], axis=1) * scale - tracking[frame], axis=1)
            if not (dist < BALL_MATCH_M).any():
                continue
            k = int(np.nanargmin(dist))
        elif on_pitch.any():
            k = int(np.flatnonzero(on_pitch)[0])  # boxes are sorted by confidence
        else:
            continue
        with_ball += 1
        teams = _team_of_boxes(h, people, players[frame], atd["team"])
        if p_home is not None:
            p = p_home[row][valid][is_person]
            sure = np.abs(p - 0.5) > 0.5 - JERSEY_SURE
            teams = np.where(sure, np.where(p >= 0.5, HOME, AWAY), teams)
        assigned.append(np.mean(teams != NONE))
        height = people[:, 3] - people[:, 1]
        dx = np.maximum(0, np.maximum(people[:, 0] - bx[k], bx[k] - people[:, 2])) / height
        dy = np.maximum(0, np.maximum(people[:, 1] - by[k], by[k] - people[:, 3])) / height
        touching = set(teams[(np.maximum(dx, dy) < contact) & (teams != NONE)].tolist())
        if len(touching) == 1:
            out[frame] = touching.pop()
        elif len(touching) == 2:
            out[frame] = CONTESTED
    stats = {"frames": len(z["frames"]), "with_ball": with_ball / max(len(z["frames"]), 1),
             "boxes_assigned": float(np.mean(assigned)) if assigned else 0.0}
    return out, stats


def episodes(ctrl, min_frames):
    """Per-frame team of runs of at least min_frames frames with the same control (HOME / AWAY),
    NONE elsewhere: short flickers are dropped."""
    out = np.full_like(ctrl, NONE)
    edges = np.flatnonzero(ctrl[1:] != ctrl[:-1]) + 1
    for a, b in zip(np.concatenate([[0], edges]), np.concatenate([edges, [len(ctrl)]])):
        if ctrl[a] in (HOME, AWAY) and b - a >= min_frames:
            out[a:b] = ctrl[a]
    return out


def features(frames="labelled", jersey=True, min_frames=3):
    """Per-frame control features as {name: array}: team (+1 home, -1 away, 0 none) of the last and
    next control run of at least min_frames frames, seconds since / until it, whether they agree,
    and the control in the frame itself (0.5 * sign when contested is not counted). Frames outside
    the detection file get no control, so these features are only meaningful on its frames."""
    from possession import features as feats
    from possession.io import FPS
    from possession.touches import last_next
    ctrl, _ = control(frames, jersey=jersey)
    ep = episodes(ctrl, min_frames)
    last, since, nxt, until = last_next(ep)
    now = np.where(ctrl == HOME, 1.0, np.where(ctrl == AWAY, -1.0, 0.0))
    out = {"ctl_last": last, "ctl_since_s": since, "ctl_next": nxt, "ctl_until_s": until,
           "ctl_agree": last * nxt, "ctl_now": now}
    for w in (1.0, 3.0):
        out["ctl_mean_{}".format(w)] = np.nan_to_num(
            feats.rolling_nanmean(np.where(ctrl >= 0, now, np.nan), int(w * FPS)), nan=0.0)
    return out
