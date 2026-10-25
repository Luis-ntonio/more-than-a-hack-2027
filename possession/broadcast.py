"""Map the broadcast recording onto the tactical timeline, and features from it.

    docker compose run --rm dev python -m possession.broadcast     # align, print aggregates, cache

Both recordings are real time at 25 fps. While the broadcast shows a replay or a close-up the live
play goes on unseen, so outside those stretches a broadcast frame j shows tactical frame j + c for
an offset c that is constant over a continuous stretch of recording (one per half, if the feed was
cut at half time). There is no shared player identity between the two ATDs (51 slots vs 46), so
frames are matched by geometry: for every broadcast player, the distance to the nearest tactical
player of the same team, plus the ball.

  1. coarse: SAMPLES broadcast frames with at least MIN_PLAYERS players are matched against every
     STRIDE-th tactical frame; the offsets c = i - j that recur are the live ones;
  2. every broadcast frame is scored at each live offset; it is live at the offset where its cost
     is under LIVE_M, and each offset is kept on the stretch of broadcast frames it explains;
  3. per tactical frame: whether the broadcast was showing it live, the broadcast's ball there, and
     the match cost.

Writes cache/<game>_broadcast_alignment.npz (derived match data, gitignored). Prints aggregates.
"""
import os

import numpy as np

from possession.io import CACHE_DIR, FPS, GAME, PITCH, load_atd, load_homography

ALIGN_PATH = os.path.join(CACHE_DIR, "{}_broadcast_alignment.npz".format(GAME))
SAMPLES = 600
MIN_PLAYERS = 8
STRIDE = 5
LIVE_M = 2.0
MIN_SHARE = 0.05   # an offset is live if at least this share of the samples pick it
BALL_W = 0.5


def _metres(atd):
    scale = np.array(PITCH, dtype=np.float64)
    return atd["players"][:, :, :2].astype(np.float64) * scale, atd["ball"].astype(np.float64) * scale


def frame_cost(bp, bteam, bball, tp, tteam, tball):
    """Match cost (m) of one broadcast frame against tactical frames tp (n, slots, 2): mean over its
    players of the distance to the nearest tactical player of the same team, plus BALL_W times the
    ball distance where both have one. NaN where nothing can be compared."""
    seen = np.isfinite(bp[:, 0])
    total = np.zeros(len(tp))
    count = 0
    for team in (0, 1):
        pts = bp[seen & (bteam == team)]
        if not len(pts):
            continue
        cand = tp[:, tteam == team]                                   # (n, s, 2)
        d = np.linalg.norm(cand[:, :, None, :] - pts[None, None], axis=3)  # (n, s, k)
        d = np.where(np.isfinite(d), d, np.inf).min(axis=1)            # (n, k)
        total += np.minimum(d, 20.0).sum(axis=1)
        count += len(pts)
    if not count:
        return np.full(len(tp), np.nan)
    cost = total / count
    if np.isfinite(bball[0]):
        # only where both have a ball: a missing tactical ball must not make a frame look unmatched
        db = np.linalg.norm(tball - bball, axis=1)
        cost = cost + BALL_W * np.where(np.isfinite(db), np.minimum(db, 20.0), 0.0)
    return cost


def align():
    b, t = load_atd(source="broadcast"), load_atd()
    bp, bb = _metres(b)
    tp, tb = _metres(t)
    nb, nt = len(bb), len(tb)
    rng = np.random.default_rng(0)
    rich = np.flatnonzero(np.isfinite(bp[:, :, 0]).sum(axis=1) >= MIN_PLAYERS)
    sample = np.sort(rng.choice(rich, size=min(SAMPLES, len(rich)), replace=False))
    grid = np.arange(0, nt, STRIDE)
    offsets, best_cost = [], []
    for j in sample:
        cost = frame_cost(bp[j], b["team"], bb[j], tp[grid], t["team"], tb[grid])
        k = int(np.nanargmin(cost))
        offsets.append(grid[k] - j)
        best_cost.append(cost[k])
    offsets, best_cost = np.array(offsets), np.array(best_cost)
    # live offsets: modes of the coarse offsets (within 2 s), refined to the frame
    good = best_cost < LIVE_M
    modes = []
    for c in np.sort(offsets[good]):
        if modes and c - modes[-1][-1] <= 2 * FPS:
            modes[-1].append(c)
        else:
            modes.append([c])
    modes = [int(np.median(m)) for m in modes if len(m) >= MIN_SHARE * good.sum()]
    print("coarse: {} samples, {:.0%} matched under {} m; live offsets (tactical - broadcast): {}".format(
        len(sample), good.mean(), LIVE_M, ", ".join("{} f ({:+.1f} s)".format(c, c / FPS) for c in modes)))

    # refine each offset to the frame on its own samples
    refined = []
    for c in modes:
        mine = sample[good & (np.abs(offsets - c) <= 2 * FPS)]
        trial = np.arange(c - 2 * STRIDE, c + 2 * STRIDE + 1)
        scores = [np.nanmedian([frame_cost(bp[j], b["team"], bb[j], tp[[j + d]], t["team"], tb[[j + d]])[0]
                                for j in mine if 0 <= j + d < nt]) for d in trial]
        refined.append(int(trial[int(np.nanargmin(scores))]))
    print("refined offsets: " + ", ".join("{} f".format(c) for c in refined))

    # every broadcast frame at every offset
    cost = np.full((len(refined), nb), np.nan)
    for k, c in enumerate(refined):
        for j in range(nb):
            i = j + c
            if 0 <= i < nt and np.isfinite(bp[j, :, 0]).any():
                cost[k, j] = frame_cost(bp[j], b["team"], bb[j], tp[[i]], t["team"], tb[[i]])[0]
    # each offset owns the broadcast stretch where it is the live one most: split at the frame that
    # best separates them (offsets are ordered in time, one per continuous recording)
    owner = np.zeros(nb, dtype=int)
    if len(refined) > 1:
        live = np.nan_to_num(cost < LIVE_M).astype(int)
        cuts = []
        for k in range(len(refined) - 1):
            diff = np.cumsum(live[k] - live[k + 1])
            cuts.append(int(np.argmax(diff)) + 1)
        for k, cut in enumerate(cuts):
            owner[cut:] = k + 1
    own_cost = cost[owner, np.arange(nb)]
    homography = load_homography(source="broadcast")
    valid_h = np.isfinite(homography[:, 0])

    # onto the tactical timeline
    bc_frame = np.full(nt, -1, dtype=np.int64)
    for k, c in enumerate(refined):
        js = np.flatnonzero(owner == k)
        bc_frame[np.clip(js + c, 0, nt - 1)[(js + c >= 0) & (js + c < nt)]] = js[(js + c >= 0) & (js + c < nt)]
    has = bc_frame >= 0
    t_cost = np.full(nt, np.nan)
    t_cost[has] = own_cost[bc_frame[has]]
    live = has & (t_cost < LIVE_M)
    t_valid_h = np.zeros(nt, dtype=bool)
    t_valid_h[has] = valid_h[bc_frame[has]]
    t_ball = np.full((nt, 2), np.nan)
    t_ball[live] = b["ball"][bc_frame[live]]
    np.savez_compressed(ALIGN_PATH, offsets=np.array(refined), owner=owner, bc_frame=bc_frame,
                        cost=t_cost, live=live, valid_h=t_valid_h, ball=t_ball)

    from possession.io import graded_blocks
    graded = np.zeros(nt, dtype=bool)
    for _, a, z in graded_blocks():
        graded[a:z + 1] = True
    tball = np.isfinite(t["ball"][:, 0])
    bball = np.isfinite(t_ball[:, 0])
    both = tball & bball & graded
    print("graded tactical frames: broadcast live on {:.0%}, valid broadcast homography but not live "
          "(replay?) {:.0%}, no homography {:.0%}".format(
              live[graded].mean(), (has & t_valid_h & ~live)[graded].mean(), (has & ~t_valid_h)[graded].mean()))
    print("broadcast ball where the tactical ATD has none: {:.0%} of those graded frames; "
          "where both have it, median {:.2f} m apart".format(
              (bball & ~tball & graded).sum() / max((~tball & graded).sum(), 1),
              np.median(np.linalg.norm((t_ball[both] - t["ball"][both]) * np.array(PITCH), axis=1))
              if both.any() else np.nan))
    print("-> " + os.path.relpath(ALIGN_PATH, os.path.dirname(CACHE_DIR)))


if __name__ == "__main__":
    align()


def states(n_frames):
    """Per tactical frame: 0 broadcast live, 1 broadcast on the pitch but not live (replay),
    2 broadcast with no pitch view (close-up, crowd), 3 no broadcast frame mapped."""
    z = np.load(ALIGN_PATH)
    has = z["bc_frame"][:n_frames] >= 0
    return np.where(~has, 3, np.where(z["live"][:n_frames], 0, np.where(z["valid_h"][:n_frames], 1, 2)))


def features(n_frames):
    """{name: (n_frames,) array} for the pre-labeller, from cache/<game>_broadcast_alignment.npz."""
    from possession.features import rolling_nanmean
    z = np.load(ALIGN_PATH)
    st = states(n_frames)
    out = {"bc_state": st.astype(np.float64),
           "bc_cost": np.nan_to_num(z["cost"][:n_frames], nan=20.0),
           "bc_ball_visible": np.isfinite(z["ball"][:n_frames, 0]).astype(np.float64)}
    not_live = (st != 0).astype(np.float64)
    for w in (2.0, 5.0, 10.0):
        out["bc_notlive_{}".format(w)] = rolling_nanmean(not_live, int(w * FPS))
    # length of the stretch of not-live broadcast each frame is in, in seconds
    edges = np.flatnonzero(np.diff(np.concatenate([[0], (st != 0).astype(np.int8), [0]])))
    run = np.zeros(n_frames)
    for a, b in zip(edges[::2], edges[1::2]):
        run[a:b] = (b - a) / FPS
    out["bc_notlive_run_s"] = run
    return out
