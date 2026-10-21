"""EDA numbers: aggregates only, printed as JSON-ish text."""
import os, sys, json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP = os.path.join(ROOT, "analysis", "cache")
os.makedirs(SP, exist_ok=True)
sys.path.insert(0, ROOT)
from games import half_video_blocks

L, W = 105.3, 68.0
FPS = 25
R = {}


def runs(mask):
    """(start, length) of True runs."""
    m = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(m)
    s = np.where(d == 1)[0]
    e = np.where(d == -1)[0]
    return s, e - s


def q(a, qs=(5, 25, 50, 75, 95, 99)):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return {f"p{k}": round(float(np.percentile(a, k)), 3) for k in qs} | {"mean": round(float(a.mean()), 3), "n": int(a.size)}


z = np.load(os.path.join(SP, "atd_tactical.npz"))
P, B, ids, team = z["P"], z["B"], z["ids"], z["team"]
N = len(ids)
R["ids_contiguous_from_1"] = bool((ids == np.arange(1, N + 1)).all())
blocks = half_video_blocks("BAR-ATH")
graded = np.zeros(N, bool)
half = np.zeros(N, np.int8)
for h, a, b in blocks:
    graded[a:b + 1] = True
    half[a:b + 1] = h
R["graded_frames"] = int(graded.sum())

# ---- ball
bvis = np.isfinite(B[:, 0])
R["ball_speed_col_all_nan"] = bool(np.isnan(B[:, 2]).all())
R["ball_visible_all"] = round(float(bvis.mean()), 4)
R["ball_visible_graded"] = round(float(bvis[graded].mean()), 4)
for h, a, b in blocks:
    R[f"ball_visible_half{h}"] = round(float(bvis[a:b + 1].mean()), 4)
s, ln = runs(~bvis & graded)
R["ball_gap_runs_graded"] = int(len(ln))
R["ball_gap_len_s"] = q(ln / FPS)
for t in (1, 2, 5, 10, 30):
    R[f"ball_missing_frames_in_gaps_over_{t}s_share"] = round(float(ln[ln > t * FPS].sum() / ln.sum()), 4)
s2, ln2 = runs(bvis & graded)
R["ball_visible_runs_len_s"] = q(ln2 / FPS)

bx, by = B[:, 0] * L, B[:, 1] * W
v = np.hypot(np.diff(bx), np.diff(by)) * FPS  # m/s between consecutive frames
R["ball_speed_ms"] = q(v)
R["ball_speed_over_40ms_share"] = round(float(np.nanmean(v > 40)), 5)
R["ball_out_of_bounds_share"] = round(float(np.mean(((B[bvis, 0] < 0) | (B[bvis, 0] > 1) | (B[bvis, 1] < 0) | (B[bvis, 1] > 1)))), 4)
R["ball_x_range"] = [float(np.nanmin(B[:, 0])), float(np.nanmax(B[:, 0]))]
R["ball_y_range"] = [float(np.nanmin(B[:, 1])), float(np.nanmax(B[:, 1]))]

# ---- players
pvis = np.isfinite(P[:, :, 0])
cnt_h = pvis[:, team == 0].sum(1)
cnt_a = pvis[:, team == 1].sum(1)
R["players_per_frame_graded"] = q(cnt_h[graded] + cnt_a[graded])
R["home_per_frame_graded"] = q(cnt_h[graded])
R["away_per_frame_graded"] = q(cnt_a[graded])
R["frames_over_22_tracked_share"] = round(float(((cnt_h + cnt_a)[graded] > 22).mean()), 4)
R["frames_under_18_tracked_share"] = round(float(((cnt_h + cnt_a)[graded] < 18).mean()), 4)
seg = []
for k in range(P.shape[1]):
    _, l = runs(pvis[:, k] & graded)
    seg.extend(l)
seg = np.array(seg)
R["tracklets_graded"] = int(len(seg))
R["tracklet_len_s"] = q(seg / FPS)
R["slot_occupancy_graded"] = [round(float(x), 3) for x in pvis[graded].mean(0)]
sp = P[:, :, 2][graded]
R["player_speed_ms"] = q(sp)
R["player_speed_nan_when_visible_share"] = round(float(np.isnan(P[:, :, 2][pvis]).mean()), 4)
# static slots (goalkeepers / referees? check speed NaN + fixed position)
x0 = P[:, :, 0]
R["slot_std_x"] = [round(float(np.nanstd(x0[graded, k])), 3) if pvis[graded, k].any() else None for k in range(P.shape[1])]

# ---- nearest player
px, py = P[:, :, 0] * L, P[:, :, 1] * W
d = np.hypot(px - bx[:, None], py - by[:, None])
d_all = np.where(np.isfinite(d), d, np.inf)
near = np.argmin(d_all, 1)
dmin = d_all[np.arange(N), near]
dmin[~np.isfinite(dmin)] = np.nan
near_team = np.where(np.isfinite(dmin), team[near], -1)
dh = np.min(np.where(team[None, :] == 0, d_all, np.inf), 1)
da = np.min(np.where(team[None, :] == 1, d_all, np.inf), 1)
R["nearest_dist_m_graded"] = q(dmin[graded])
R["nearest_dist_over_3m_share"] = round(float(np.nanmean(dmin[graded] > 3)), 4)

# ---- GT public
gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))["label"].values
w = gt != "X"
R["gt_window"] = [int(np.where(w)[0].min()), int(np.where(w)[0].max())]
segs = []
lab = gt[w]
start = 0
for i in range(1, len(lab) + 1):
    if i == len(lab) or lab[i] != lab[start]:
        segs.append((lab[start], (i - start) / FPS))
        start = i
R["gt_segments"] = [(a, round(b, 2)) for a, b in segs]
R["gt_counts"] = {k: int((lab == k).sum()) for k in "HAD"}
gi = np.where(w)[0]
for k in "HAD":
    m = gi[lab == k]
    R[f"gt_{k}_ball_visible"] = round(float(bvis[m].mean()), 3)
    R[f"gt_{k}_nearest_dist_m"] = q(dmin[m], (25, 50, 75))
    R[f"gt_{k}_ball_speed_ms"] = q(v[np.minimum(m, N - 2)], (25, 50, 75))
    R[f"gt_{k}_players_tracked"] = q((cnt_h + cnt_a)[m], (25, 50, 75))
alive = gi[lab != "D"]
lab_alive = lab[lab != "D"]
nt = near_team[alive]
ok = nt >= 0
truth = np.where(lab_alive == "H", 0, 1)
R["gt_alive_nearest_team_accuracy_when_ball_visible"] = round(float((nt[ok] == truth[ok]).mean()), 4)
R["gt_alive_ball_visible"] = round(float(ok.mean()), 4)
# accuracy vs distance bins
bins = [0, 1, 2, 3, 5, 10, 100]
acc = {}
for lo, hi in zip(bins[:-1], bins[1:]):
    mm = ok & (dmin[alive] >= lo) & (dmin[alive] < hi)
    if mm.sum():
        acc[f"{lo}-{hi}m"] = (round(float((nt[mm] == truth[mm]).mean()), 3), int(mm.sum()))
R["gt_alive_nearest_acc_by_dist"] = acc
# dead ball: ball visible? where is it
dd = gi[lab == "D"]
R["gt_D_ball_out_of_bounds_share"] = round(float(np.nanmean((B[dd, 0] < 0) | (B[dd, 0] > 1) | (B[dd, 1] < 0) | (B[dd, 1] > 1))), 3)
R["gt_D_mean_player_speed"] = round(float(np.nanmean(P[dd, :, 2])), 3)
R["gt_alive_mean_player_speed"] = round(float(np.nanmean(P[alive, :, 2])), 3)

# ---- baselines on public window
for name in ("baseline_raw", "baseline_smooth25", "baseline_smooth50", "baseline_smooth125"):
    p = pd.read_csv(os.path.join(ROOT, "evaluation", "results", name + ".csv"))["label"].values
    lab_all = p[graded]
    nseg = 0
    for h, a, b in blocks:
        x = p[a:b + 1]
        poss = x[np.r_[True, x[1:] != x[:-1]]]
        nseg += int(np.isin(poss, ["H", "A"]).sum())
    R[f"{name}_full_match_possessions"] = nseg
    R[f"{name}_full_D_share"] = round(float((lab_all == "D").mean()), 3)

# ---- smart tagging
root = ET.parse(os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical_smart_tagging.xml")).getroot()
rows = []
for inst in root.iter("instance"):
    r = {"start": float(inst.findtext("start")), "end": float(inst.findtext("end")), "code": inst.findtext("code")}
    for lb in inst.findall("label"):
        r[lb.findtext("group")] = lb.findtext("text")
    rows.append(r)
st = pd.DataFrame(rows)
st["dur"] = st["end"] - st["start"]
st["f0"] = (st["start"] * FPS).round().astype(int)
st["f1"] = (st["end"] * FPS).round().astype(int)
R["st_n"] = len(st)
R["st_dur_by_code"] = st.groupby("code")["dur"].describe()[["count", "mean", "50%", "min", "max"]].round(1).to_dict("index")
R["st_team_by_code"] = st.groupby(["code", "Team"]).size().unstack(fill_value=0).to_dict("index")
R["st_type_team"] = st[st.code == "SET PIECES"].groupby(["Type", "Team"]).size().unstack(fill_value=0).to_dict("index")
# coverage of graded frames by team-tagged possession phases
POSS = ["BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION", "BALL IN FINAL THIRD", "BALL IN THE BOX", "LONG BALL", "PLAYERS IN THE BOX"]
DEF = ["DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD", "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"]
vote = np.zeros((N, 2), np.int32)
for _, r in st.iterrows():
    if r.Team not in ("Barcelona", "Atletic Club"):
        continue
    t = 0 if r.Team == "Barcelona" else 1
    if r.code in POSS:
        vote[r.f0:r.f1, t] += 1
    elif r.code in DEF:
        vote[r.f0:r.f1, 1 - t] += 1
cov = (vote.sum(1) > 0)
R["st_possession_prior_coverage_graded"] = round(float(cov[graded].mean()), 4)
conf = vote[:, 0] != vote[:, 1]
R["st_prior_conflict_share_graded"] = round(float((cov & ~conf)[graded].mean()), 4)
stp = np.where(vote[:, 0] > vote[:, 1], 0, 1)
m = cov[alive] & conf[alive]
R["st_prior_acc_on_gt_alive"] = (round(float((stp[alive][m] == truth[m]).mean()), 3), int(m.sum()), int(len(alive)))
# per-code accuracy on whole match is not possible; per-code in public window
pc = {}
for code in POSS + DEF:
    mm = np.zeros(N, bool)
    tt = np.full(N, -1)
    for _, r in st[(st.code == code) & st.Team.isin(["Barcelona", "Atletic Club"])].iterrows():
        t = 0 if r.Team == "Barcelona" else 1
        if code in DEF:
            t = 1 - t
        mm[r.f0:r.f1] = True
        tt[r.f0:r.f1] = t
    sel = mm[alive]
    if sel.sum():
        pc[code] = (round(float((tt[alive][sel] == truth[sel]).mean()), 3), int(sel.sum()))
R["st_per_code_acc_public"] = pc
# set pieces vs GT D
spf = st[st.code == "SET PIECES"]
R["st_setpieces_in_public_window"] = [(round(r.start, 1), r.Type, r.Team) for _, r in spf.iterrows() if R["gt_window"][0] <= r.f0 <= R["gt_window"][1]]
R["gt_D_runs_public"] = [(int(gi[0] + s_), round(l_ / FPS, 2)) for s_, l_ in zip(*runs(lab == "D"))]

# ---- homography
hg = pd.read_csv(os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical_homography.csv"))
R["hom_tac_valid"] = round(float(hg.valid.mean()), 4)
R["hom_tac_reproj_graded"] = q(hg.reprojection_error.values[graded])
R["hom_tac_reproj_ungraded"] = q(hg.reprojection_error.values[~graded])
hb = pd.read_csv(os.path.join(ROOT, "data", "broadcast", "BAR-ATH_broadcast_homography.csv"))
R["hom_bro_valid"] = round(float(hb.valid.mean()), 4)
R["hom_bro_reproj"] = q(hb.reprojection_error.values)
_, lnb = runs(hb.valid.values == 0)
R["hom_bro_invalid_runs"] = int(len(lnb))
R["hom_bro_invalid_len_s"] = q(lnb / FPS)
# camera motion: change in h02 (pan)
R["hom_tac_pan_change_graded"] = q(np.abs(np.diff(hg.h02.values))[graded[:-1]])

# ---- broadcast ATD
zb = np.load(os.path.join(SP, "atd_broadcast.npz"))
Bb, Pb = zb["B"], zb["P"]
R["bro_ball_visible"] = round(float(np.isfinite(Bb[:, 0]).mean()), 4)
R["bro_ball_visible_when_hom_valid"] = round(float(np.isfinite(Bb[hb.valid.values == 1, 0]).mean()), 4)
R["bro_players_per_frame"] = q(np.isfinite(Pb[:, :, 0]).sum(1))
# complementarity: is broadcast ball visible when tactical not? (unknown alignment; report counts only)

np.savez_compressed(os.path.join(SP, "derived.npz"), dmin=dmin, near_team=near_team, dh=dh, da=da, v=v, cnt_h=cnt_h, cnt_a=cnt_a, graded=graded, vote=vote)
st.to_pickle(os.path.join(SP, "st.pkl"))
print(json.dumps(R, indent=1, default=str))
