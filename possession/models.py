"""The submission models. Each returns one label per video frame ('H', 'A', 'D', 'X').

v0  nearest player + majority vote, but a missing ball carries the last team instead of
    becoming D, plus the deterministic pre-kickoff rule.
v1  3-state HMM (H / A / D) decoded with Viterbi: distance-weighted team evidence, a heuristic
    dead-ball probability, and transition costs that encode "possession changes need sustained
    evidence".
"""
import numpy as np

from possession.decode import viterbi
from possession.io import HOME, AWAY, graded_blocks, kickoff_frames

UNKNOWN = -1


def _labels_skeleton(n_frames):
    """X everywhere: ungraded frames stay X, graded ones are overwritten by the models."""
    return np.full(n_frames, "X", dtype="<U1")


def _apply_prekickoff_rule(labels):
    """Graded frames before each half's kickoff are dead ball (the 2nd half grades ~2 min early)."""
    kickoffs = kickoff_frames()
    for half, first, last in graded_blocks():
        labels[first:min(kickoffs[half], last + 1)] = "D"
    return labels


def _forward_fill(codes):
    """Replace UNKNOWN with the last known code, back-filling the head. All-unknown stays."""
    known = np.flatnonzero(codes != UNKNOWN)
    if not len(known):
        return codes
    carry = np.maximum.accumulate(np.where(codes != UNKNOWN, np.arange(len(codes)), -1))
    carry[carry < 0] = known[0]
    return codes[carry]


def _majority(codes, window):
    """Centred majority vote over +/- window frames between HOME and AWAY. Ties go home."""
    home = np.concatenate(([0], np.cumsum(codes == HOME)))
    away = np.concatenate(([0], np.cumsum(codes == AWAY)))
    index = np.arange(len(codes))
    low = np.maximum(0, index - window)
    high = np.minimum(len(codes), index + window + 1)
    return np.where(home[high] - home[low] >= away[high] - away[low], HOME, AWAY)


# -- v0 ------------------------------------------------------------------------------------------

V0_WINDOW = 50  # +/- 2 s, the published baseline's best window on the full match


def v0(features, n_frames, window=V0_WINDOW):
    nearest = np.where(features["d_home"] < features["d_away"], HOME, AWAY)
    nearest = np.where(features["ball_visible"], nearest, UNKNOWN)

    labels = _labels_skeleton(n_frames)
    for _, first, last in graded_blocks():
        codes = _forward_fill(nearest[first:last + 1].copy())
        codes = _majority(codes, window)
        labels[first:last + 1] = np.where(codes == HOME, "H", "A")
    return _apply_prekickoff_rule(labels)


# -- v1 ------------------------------------------------------------------------------------------

V1_PARAMS = {
    # team evidence: logit(H) = conf * gain * tanh((d_away - d_home) / scale)
    "team_gain": 2.5,
    "team_scale_m": 1.5,
    "conf_decay_m": 2.5,   # conf = exp(-d_nearest / decay): 91% right <1 m, ~60% at 3-5 m (EDA)
    # dead-ball logit, hand-set from the EDA contrasts (no fitting on the public window)
    "dead_bias": -2.0,
    "dead_gap_w": 3.0,     # inside a ball gap, scaled by gap length up to gap_full_s
    "gap_full_s": 8.0,
    "dead_speed_w": 2.0,   # players slower than speed_mid m/s (dead 1.4 vs alive 2.4)
    "speed_mid": 1.9,
    "speed_span": 0.6,
    "dead_pan_w": 1.0,     # camera still (pan median 0.10 without ball vs 0.62 with)
    "pan_mid": 0.3,
    "dead_oop_w": 3.0,     # ball seen outside the pitch
    # transition costs in nats per change
    "cost_team_switch": 25.0,
    "cost_to_dead": 15.0,
    "cost_from_dead": 15.0,
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def team_logit(features, p=V1_PARAMS):
    """Evidence for HOME over AWAY per frame; 0 when the ball is not visible."""
    d_home, d_away = features["d_home"], features["d_away"]
    nearest = np.minimum(d_home, d_away)
    conf = np.exp(-np.where(np.isfinite(nearest), nearest, np.inf) / p["conf_decay_m"])
    both = np.isfinite(d_home) & np.isfinite(d_away)
    diff = np.where(both, np.where(both, d_away, 0.0) - np.where(both, d_home, 0.0), 0.0)
    logit = conf * p["team_gain"] * np.tanh(diff / p["team_scale_m"])
    return np.where(features["ball_visible"], logit, 0.0)


def dead_probability(features, p=V1_PARAMS):
    missing = ~features["ball_visible"]
    gap = np.clip(features["gap_len_s"] / p["gap_full_s"], 0, 1) * missing
    speed = np.nan_to_num(features["speed_2s"], nan=p["speed_mid"])
    pan = np.nan_to_num(features["pan_2s"], nan=p["pan_mid"])
    z = (p["dead_bias"]
         + p["dead_gap_w"] * gap
         + p["dead_speed_w"] * np.clip((p["speed_mid"] - speed) / p["speed_span"], -1.5, 1.5)
         + p["dead_pan_w"] * np.clip((p["pan_mid"] - pan) / p["pan_mid"], -1, 1)
         + p["dead_oop_w"] * features["out_of_pitch"])
    return _sigmoid(z)


def v1(features, n_frames, p=V1_PARAMS):
    logit = team_logit(features, p)
    p_dead = np.clip(dead_probability(features, p), 1e-6, 1 - 1e-6)
    p_home = np.clip(_sigmoid(logit), 1e-6, 1 - 1e-6)
    log_emission = np.stack([
        np.log1p(-p_dead) + np.log(p_home),
        np.log1p(-p_dead) + np.log1p(-p_home),
        np.log(p_dead),
    ], axis=1)

    h_a, to_d, from_d = p["cost_team_switch"], p["cost_to_dead"], p["cost_from_dead"]
    log_transition = -np.array([
        [0.0, h_a, to_d],
        [h_a, 0.0, to_d],
        [from_d, from_d, 0.0],
    ])

    labels = _labels_skeleton(n_frames)
    kickoffs = kickoff_frames()
    for half, first, last in graded_blocks():
        start = max(first, kickoffs[half])  # before kickoff is D by rule, decode from kickoff on
        path = viterbi(log_emission[start:last + 1], log_transition,
                       log_start=np.log(np.array([0.5, 0.5, 1e-6])))
        labels[start:last + 1] = np.array(["H", "A", "D"])[path]
    return _apply_prekickoff_rule(labels)


MODELS = {"v0": v0, "v1": v1}
