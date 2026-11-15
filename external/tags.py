"""Smart Tagging as model input: simulated on external data for training, read from the XML at inference.

Smart Tagging is Metrica's automatic annotation of the tactical video: intervals with a code and a
Team tag, "the team performing the event" (OVERVIEW.md). So the team in possession for the attacking
codes, the team without the ball for DEFENDING IN * and DEFENSIVE TRANSITION, and the team taking the
restart for SET PIECES, where throw-ins carry N/A (DATA.md). It is not ground truth: it misses phases,
invents some, gets teams wrong and its intervals are phases, not possessions.

No external dataset has Metrica's tags, so training tags are simulated from each match's truth with
those errors. How often Metrica makes each error cannot be measured without the provided match, so
error levels are drawn per stretch of play over WIDE ranges, as external.degrade does, and no
number from the match's tagging sets one. An interval is (start frame, end frame, family, team):
family "attack", "defend" or "setpiece"; team 0 home, 1 away, None when the tag names no team.
"""
import numpy as np

from external.features import FPS, SLOW_WINDOWS_S, add_windows, last_next, rolling_mean, run_info


def nearest_sign(X, names):
    """+1 where the nearest player to the ball is home, -1 away, 0 where the ball is lost."""
    diff = X[:, names.index("d_away")] - X[:, names.index("d_home")]
    return np.sign(np.nan_to_num(diff, nan=0.0))

CHUNK_S = 300
ATTACK = ("BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION", "GOALS", "LONG BALL",
          "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX")
DEFEND = ("DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD", "DEFENDING IN DEFENSIVE THIRD",
          "DEFENSIVE TRANSITION")
SETPIECE = ("SET PIECES",)

# Ranges drawn per chunk for training. Wide on purpose, up to tags that carry almost nothing.
RANGES = {"wrong": (0.0, 0.45),      # share of intervals tagged with the wrong team
          "miss": (0.1, 0.9),        # share of possessions with no tag
          "min_s": (1.0, 8.0),       # possessions shorter than this are tagged less often
          "cover": (0.3, 1.0),       # least share of the possession an interval covers
          "jitter_s": (0.3, 4.0),    # spread of interval edges
          "span": (0.0, 0.3),        # share of intervals that run on over the next possession
          "defend": (0.2, 0.5),      # share of intervals in the defending family
          "restart": (0.3, 0.95),    # share of restarts tagged as set pieces
          "no_team": (0.2, 0.7),     # share of set pieces with N/A as team
          "false_per_min": (0.0, 2.0)}
# Fixed levels for scoring, from tags that say a lot to tags that say nothing.
LEVELS = {"good": {"wrong": 0.05, "miss": 0.2, "min_s": 2.0, "cover": 0.7, "jitter_s": 0.5, "span": 0.05,
                   "defend": 0.35, "restart": 0.9, "no_team": 0.4, "false_per_min": 0.2},
          "mid": {"wrong": 0.15, "miss": 0.4, "min_s": 4.0, "cover": 0.5, "jitter_s": 1.5, "span": 0.15,
                  "defend": 0.35, "restart": 0.7, "no_team": 0.5, "false_per_min": 0.5},
          "poor": {"wrong": 0.3, "miss": 0.7, "min_s": 6.0, "cover": 0.4, "jitter_s": 3.0, "span": 0.25,
                   "defend": 0.35, "restart": 0.5, "no_team": 0.6, "false_per_min": 1.5},
          "useless": {"wrong": 0.5, "miss": 0.9, "min_s": 8.0, "cover": 0.3, "jitter_s": 4.0, "span": 0.3,
                      "defend": 0.35, "restart": 0.3, "no_team": 0.7, "false_per_min": 2.0}}


def runs_of(labels):
    """[(start, end, label)] of constant label, end exclusive."""
    edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(labels)]])
    return [(int(a), int(b), labels[a]) for a, b in zip(starts, ends)]


def simulate(labels, rng, level=None):
    """Intervals a Smart Tagging-like annotator could give for a match with these labels (H, A, D)."""
    n = len(labels)
    chunk = CHUNK_S * FPS
    params = [level or {k: rng.uniform(*r) for k, r in RANGES.items()} for _ in range(0, n, chunk)]
    runs = runs_of(np.asarray(labels))
    out = []

    def wrong(team, p):
        return 1 - team if rng.uniform() < p["wrong"] else team

    for i, (a, b, lab) in enumerate(runs):
        p = params[a // chunk]
        if lab == "D":
            if i + 1 < len(runs) and runs[i + 1][2] != "D" and rng.uniform() < p["restart"]:
                team = wrong(0 if runs[i + 1][2] == "H" else 1, p)
                out.append((b - int(rng.uniform(1, 4) * FPS), b + int(rng.uniform(1, 4) * FPS), "setpiece",
                            None if rng.uniform() < p["no_team"] else team))
            continue
        if rng.uniform() > (1 - p["miss"]) * min(1.0, (b - a) / FPS / p["min_s"]):
            continue
        end = b
        if rng.uniform() < p["span"]:
            later = [r for r in runs[i + 1:i + 3] if r[2] != "D"]
            if later:
                end = later[0][1]
        for _ in range(rng.integers(1, 4)):
            length = int((end - a) * rng.uniform(p["cover"], 1.0))
            start = a + int(rng.integers(0, end - a - length + 1))
            s = start + int(rng.normal(0, p["jitter_s"]) * FPS)
            e = start + length + int(rng.normal(0, p["jitter_s"]) * FPS)
            if e - s < FPS:
                continue
            team = wrong(0 if lab == "H" else 1, p)
            family = "defend" if rng.uniform() < p["defend"] else "attack"
            out.append((s, e, family, 1 - team if family == "defend" else team))
    for c, p in enumerate(params):
        lo, hi = c * chunk, min((c + 1) * chunk, n)
        for _ in range(rng.poisson(p["false_per_min"] * (hi - lo) / FPS / 60)):
            s = int(rng.integers(lo, hi))
            out.append((s, s + int(rng.uniform(2, 10) * FPS), ("attack", "defend", "setpiece")[rng.integers(3)],
                        int(rng.integers(2))))
    return [(max(s, 0), min(e, n), f, t) for s, e, f, t in out if min(e, n) > max(s, 0)]


def flip(intervals):
    """The same tags with home and away exchanged, to go with external.features.swap_teams."""
    return [(s, e, f, None if t is None else 1 - t) for s, e, f, t in intervals]


TRUST_WINDOW_S = 300   # as long as a stretch of one tagging quality in the simulation


def trust_columns(vote, nearest, half):
    """How far the tags agree with the tracking: does the tagged team match the nearest team?

    Computed the same way in training and on the match, like the attack direction, so the model can
    learn how much to lean on the tags of whatever match it is given. No number from the provided
    match sets anything here.
    """
    v = np.sign(vote)
    both = (v != 0) & (nearest != 0)
    agree = np.where(both, (v == nearest).astype(float), np.nan)
    per_half = np.full(len(v), np.nan)
    for h in np.unique(half):
        rows = half == h
        if np.any(both[rows]):
            per_half[rows] = np.nanmean(agree[rows])
    local = rolling_mean(agree, int(TRUST_WINDOW_S * FPS / 2))
    return {"tag_trust_half": per_half, "tag_trust_local": local,
            "tag_vote_trusted": vote * (2 * np.nan_to_num(local, nan=0.5) - 1)}


def columns(intervals, n, nearest=None, half=None):
    """(n, k) float32 and names: tag evidence for the home team having the ball, frame by frame.

    nearest: (n,) +1 where the nearest player to the ball is home, -1 away, 0 unknown. With it and
    half, three trust columns go at the end.
    """
    sums = {k: np.zeros(n + 1) for k in ("attack", "defend", "setpiece")}
    counts = {k: np.zeros(n + 1) for k in sums}
    for s, e, family, team in intervals:
        sign = 0.0 if team is None else (1.0 if team == 0 else -1.0)
        sums[family][s] += sign
        sums[family][e] -= sign
        counts[family][s] += 1
        counts[family][e] -= 1
    vote = {k: np.cumsum(v)[:n] for k, v in sums.items()}
    count = {k: np.cumsum(v)[:n] for k, v in counts.items()}
    cols = {"tag_attack": vote["attack"], "tag_defend": -vote["defend"], "tag_setpiece": vote["setpiece"]}
    cols["tag_vote"] = cols["tag_attack"] + cols["tag_defend"] + cols["tag_setpiece"]
    cols["tag_n"] = count["attack"] + count["defend"] + count["setpiece"]
    cols["tag_in_setpiece"] = (count["setpiece"] > 0).astype(float)
    (cols["tag_last"], cols["tag_since_s"], cols["tag_next"],
     cols["tag_until_s"]) = last_next(np.sign(cols["tag_vote"]))
    cols["setpiece_since_s"], cols["setpiece_until_s"], _ = run_info(count["setpiece"] > 0)
    add_windows(cols, "tag_vote", SLOW_WINDOWS_S)
    if nearest is not None:
        cols.update(trust_columns(cols["tag_vote"], nearest, half))
    names = list(cols)
    return np.column_stack([cols[k] for k in names]).astype(np.float32), names


def from_smart_tagging(events, home_name, away_name, first, last):
    """Intervals of the Smart Tagging inside frames first..last, relative to first.

    events: starter.load_smart_tagging.load() output. Club names are matched case-insensitively,
    and each tag value must be one of the two clubs or N/A, so a spelling difference fails loudly
    instead of turning every tag into N/A.
    """
    def norm(name):
        return "".join(ch for ch in name.lower() if ch.isalnum())

    def family(code):
        code = (code or "").strip().upper()
        return "attack" if code in ATTACK else "defend" if code in DEFEND else "setpiece" if code in SETPIECE else None

    values = {e["tags"].get("Team") for e in events} - {None, "N/A"}
    clubs = {v: 0 if norm(v) in norm(home_name) or norm(home_name) in norm(v) else
             1 if norm(v) in norm(away_name) or norm(away_name) in norm(v) else None for v in values}
    unresolved = sorted(v for v, t in clubs.items() if t is None)
    if unresolved:
        # one club spelled differently: accept it as the other side only if the first resolved
        known = [t for t in clubs.values() if t is not None]
        if len(unresolved) == 1 and len(known) == 1:
            clubs[unresolved[0]] = 1 - known[0]
        else:
            raise ValueError("Smart Tagging teams that match neither club: {}".format(unresolved))
    out = []
    for e in events:
        f = family(e["code"])
        if f is None:
            continue
        s, t = max(e["start_frame"], first), min(e["end_frame"], last + 1)
        if t > s:
            out.append((s - first, t - first, f, clubs.get(e["tags"].get("Team"))))
    return out
