import os, sys, json
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP = os.path.join(ROOT, "analysis", "cache")
os.makedirs(SP, exist_ok=True)
sys.path.insert(0, ROOT)
from games import half_video_blocks
FPS = 25
R = {}
z = np.load(os.path.join(SP, "atd_tactical.npz"))
P, B, team = z["P"], z["B"], z["team"]
d = np.load(os.path.join(SP, "derived.npz"))
graded, near_team, dmin = d["graded"], d["near_team"], d["dmin"]
st = pd.read_pickle(os.path.join(SP, "st.pkl"))
N = len(B)
bvis = np.isfinite(B[:, 0])

# third ball column
c = B[:, 2]
R["ball_col3_nonnan_share_when_visible"] = round(float(np.isfinite(c[bvis]).mean()), 4)
if np.isfinite(c).any():
    R["ball_col3_q"] = [round(float(x), 3) for x in np.nanpercentile(c, [1, 25, 50, 75, 99])]

# goalkeepers: slot 24 (home) and 46 (away)
for k in (23, 45):
    x = P[:, k, 0]
    for h, a, b in half_video_blocks("BAR-ATH"):
        R[f"slot{k+1}_half{h}_x_median"] = round(float(np.nanmedian(x[a:b + 1])), 3)

# ball missing vs mean player speed (dead-ball proxy) on graded frames
ms = np.nanmean(P[:, :, 2], 1)
R["mean_player_speed_ball_visible"] = round(float(np.nanmean(ms[graded & bvis])), 3)
R["mean_player_speed_ball_missing"] = round(float(np.nanmean(ms[graded & ~bvis])), 3)
# ball missing vs camera pan speed
hg = pd.read_csv(os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical_homography.csv"))
pan = np.abs(np.r_[0, np.diff(hg.h02.values)])
R["cam_pan_ball_visible"] = round(float(np.median(pan[graded & bvis])), 3)
R["cam_pan_ball_missing"] = round(float(np.median(pan[graded & ~bvis])), 3)
# where was the ball last seen before a long gap (>5s)?
m = np.concatenate(([False], ~bvis & graded, [False])).astype(np.int8)
s = np.where(np.diff(m) == 1)[0]; e = np.where(np.diff(m) == -1)[0]
long = (e - s) > 5 * FPS
lx = B[np.maximum(s[long] - 1, 0), 0]; ly = B[np.maximum(s[long] - 1, 0), 1]
R["long_gaps"] = int(long.sum())
R["long_gap_last_seen_near_touchline_share"] = round(float(np.nanmean((lx < 0.03) | (lx > 0.97) | (ly < 0.04) | (ly > 0.96))), 3)
# long gaps overlapping a set-piece tag start within the gap or 3s after
spf = st[st.code == "SET PIECES"].f0.values
ov = [(np.any((spf >= a) & (spf <= b + 3 * FPS))) for a, b in zip(s[long], e[long])]
R["long_gaps_followed_by_setpiece_share"] = round(float(np.mean(ov)), 3)
# share of set pieces whose start falls inside a ball gap
R["setpiece_starts_inside_ball_gap_share"] = round(float(np.mean(~bvis[spf])), 3)

# kickoffs
R["kickoff_tags"] = [(int(r.f0), r.Type) for _, r in st[st.code == "SET PIECES"].iterrows() if "Kick Off" in str(r.Type)]

# GT transitions in public window
gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))["label"].values
idx = np.where(gt != "X")[0]
ch = [i for i in idx[1:] if gt[i] != gt[i - 1]]
tr = []
for i in ch:
    nt = near_team[i - 50:i + 50]
    # first frame after i-50 where the nearest team equals the new label
    tgt = {"H": 0, "A": 1}.get(gt[i], None)
    lag = None
    if tgt is not None:
        hit = np.where(nt == tgt)[0]
        lag = int(hit[0] - 50) if len(hit) else None
    tr.append({"frame": int(i), "from": gt[i - 1], "to": gt[i], "ball_visible": bool(bvis[i]),
               "dmin": None if np.isnan(dmin[i]) else round(float(dmin[i]), 2), "first_nearest_hit_lag_frames": lag})
R["gt_transitions"] = tr

# ball state proxies around D runs: is ball static / near touchline in D?
D = idx[gt[idx] == "D"]
bx, by = B[D, 0], B[D, 1]
R["gt_D_ball_near_boundary_share"] = round(float(np.nanmean((bx < 0.03) | (bx > 0.97) | (by < 0.04) | (by > 0.96))), 3)
v = d["v"]
R["gt_D_ball_static_share(<1ms)"] = round(float(np.nanmean(v[D] < 1)), 3)
A = idx[gt[idx] != "D"]
R["gt_alive_ball_static_share(<1ms)"] = round(float(np.nanmean(v[A] < 1)), 3)
print(json.dumps(R, indent=1, default=str))
