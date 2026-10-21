import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP = os.path.join(ROOT, "analysis", "cache")
os.makedirs(SP, exist_ok=True)
OUT = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, ROOT)
from games import half_video_blocks
FPS = 25
CH, CA, CD, CX = "#a50044", "#2b6cb0", "#9aa0a6", "#ffffff"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120})

z = np.load(os.path.join(SP, "atd_tactical.npz"))
P, B, team = z["P"], z["B"], z["team"]
d = np.load(os.path.join(SP, "derived.npz"))
graded, near_team, dmin, v = d["graded"], d["near_team"], d["dmin"], d["v"]
cnt = d["cnt_h"] + d["cnt_a"]
st = pd.read_pickle(os.path.join(SP, "st.pkl"))
N = len(B)
bvis = np.isfinite(B[:, 0])
blocks = half_video_blocks("BAR-ATH")
gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))["label"].values
ms = np.nanmean(np.where(np.isfinite(P[:, :, 2]), P[:, :, 2], np.nan), 1)


def runs(mask):
    m = np.concatenate(([False], mask, [False])).astype(np.int8)
    dd = np.diff(m)
    s = np.where(dd == 1)[0]; e = np.where(dd == -1)[0]
    return s, e - s


# 1. full-match timeline per 30 s
bin_ = 30 * FPS
nb = N // bin_
t = np.arange(nb) * 30 / 60
fig, ax = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True)
ax[0].plot(t, bvis[:nb * bin_].reshape(nb, -1).mean(1) * 100, color="#333")
ax[0].set_ylabel("pelota visible %")
ax[1].plot(t, cnt[:nb * bin_].reshape(nb, -1).mean(1), color="#333")
ax[1].axhline(22, ls=":", color="grey"); ax[1].set_ylabel("jugadores trackeados")
ax[2].plot(t, np.nan_to_num(ms[:nb * bin_]).reshape(nb, -1).mean(1), color="#333")
ax[2].set_ylabel("vel. media jug. (m/s)"); ax[2].set_xlabel("minuto del video táctico")
for a in ax:
    for h, s_, e_ in blocks:
        a.axvspan(s_ / FPS / 60, e_ / FPS / 60, color="#e8f5e9", zorder=0)
    for f in st[st.code == "SET PIECES"].f0:
        pass
ax[0].set_title("Partido completo (bandas verdes = tramos calificados), medias cada 30 s")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "01_timeline.png")); plt.close(fig)

# 2. ball gap and visible-run durations
s, ln = runs(~bvis & graded)
s2, ln2 = runs(bvis & graded)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
bins = np.logspace(-1.7, 2.3, 40)
ax[0].hist(ln / FPS, bins=bins, color="#333"); ax[0].set_xscale("log")
ax[0].set_title(f"Huecos de pelota (n={len(ln)}) — mediana {np.median(ln)/FPS:.1f} s"); ax[0].set_xlabel("duración (s)")
ax[1].hist(ln2 / FPS, bins=bins, color="#4a7"); ax[1].set_xscale("log")
ax[1].set_title(f"Tramos con pelota visible (n={len(ln2)})"); ax[1].set_xlabel("duración (s)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "02_ball_gaps.png")); plt.close(fig)

# 3. public window strip
a, b = 11185, 18685
x = np.arange(a, b)
def strip(ax, labels, title):
    col = np.array([{"H": CH, "A": CA, "D": CD}.get(l, CX) for l in labels])
    for c in (CH, CA, CD):
        m = col == c
        ax.vlines(x[m], 0, 1, colors=c, linewidth=1)
    ax.set_yticks([]); ax.set_ylabel(title, rotation=0, ha="right", va="center")
nt = np.where(near_team == 0, "H", np.where(near_team == 1, "A", "X"))
s50 = pd.read_csv(os.path.join(ROOT, "evaluation", "results", "baseline_smooth50.csv"))["label"].values
st_vote = d["vote"]
stp = np.where(st_vote.sum(1) == 0, "X", np.where(st_vote[:, 0] > st_vote[:, 1], "H", np.where(st_vote[:, 1] > st_vote[:, 0], "A", "X")))
fig, ax = plt.subplots(5, 1, figsize=(12, 4.2), sharex=True)
strip(ax[0], gt[a:b], "GT")
strip(ax[1], nt[a:b], "jugador más\ncercano")
strip(ax[2], s50[a:b], "baseline\nsmooth50")
strip(ax[3], stp[a:b], "Smart Tagging\n(prior equipo)")
ax[4].plot(x, np.where(bvis[a:b], dmin[a:b], np.nan), color="#333", lw=0.6)
ax[4].set_ylabel("dist. al más\ncercano (m)", rotation=0, ha="right", va="center"); ax[4].set_ylim(0, 12)
ax[4].set_xlabel("frame de video (ventana pública de 5 min)")
from matplotlib.patches import Patch
ax[0].legend(handles=[Patch(color=CH, label="H Barcelona"), Patch(color=CA, label="A Athletic"), Patch(color=CD, label="D muerta")], ncol=3, loc="lower right", bbox_to_anchor=(1, 1.05), frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "03_public_window.png")); plt.close(fig)

# 4. nearest accuracy vs distance (public, alive)
idx = np.where((gt == "H") | (gt == "A"))[0]
truth = np.where(gt[idx] == "H", 0, 1)
ok = near_team[idx] >= 0
edges = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 10])
acc, n = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = ok & (dmin[idx] >= lo) & (dmin[idx] < hi)
    acc.append((near_team[idx][m] == truth[m]).mean()); n.append(m.sum())
fig, ax = plt.subplots(figsize=(7, 3.4))
lbl = [f"{lo:g}-{hi:g}" for lo, hi in zip(edges[:-1], edges[1:])]
ax.bar(lbl, np.array(acc) * 100, color="#333")
for i, (a_, n_) in enumerate(zip(acc, n)):
    ax.text(i, a_ * 100 + 1, f"n={n_}", ha="center", fontsize=8)
ax.axhline(50, ls=":", color="grey"); ax.set_ylim(0, 105)
ax.set_ylabel("% acierto equipo"); ax.set_xlabel("distancia pelota–jugador más cercano (m)")
ax.set_title("Regla 'más cercano' vs GT (ventana pública, pelota viva)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "04_nearest_accuracy.png")); plt.close(fig)

# 5. smart tagging codes
g = st.groupby("code").agg(n=("dur", "size"), dur=("dur", "median")).sort_values("n")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
ax[0].barh(g.index, g.n, color="#333"); ax[0].set_title("instancias por código")
ax[1].barh(g.index, g.dur, color="#777"); ax[1].set_title("duración mediana (s)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "05_smart_tagging.png")); plt.close(fig)

# 6. ball heatmap and ball-missing heatmap (last seen)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
m = graded & bvis
ax[0].hist2d(B[m, 0], B[m, 1], bins=[60, 40], range=[[-0.05, 1.05], [-0.05, 1.05]], cmap="Greys")
ax[0].set_title("Posición de la pelota (frames calificados)")
s, ln = runs(~bvis & graded)
k = s[ln > 5 * FPS] - 1
ax[1].scatter(B[k, 0], B[k, 1], s=10, color=CH)
ax[1].set_xlim(-0.05, 1.05); ax[1].set_ylim(-0.05, 1.05)
ax[1].set_title("Última posición antes de un hueco > 5 s")
for a_ in ax:
    a_.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, color="#4a7")); a_.axvline(0.5, color="#4a7", lw=0.8)
    a_.set_aspect(68 / 105)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "06_ball_positions.png")); plt.close(fig)

# 7. ball-missing vs dynamics
fig, ax = plt.subplots(figsize=(7, 3.4))
bins = np.linspace(0, 5, 50)
ax.hist(ms[graded & bvis], bins=bins, density=True, alpha=0.6, label="pelota visible", color="#4a7")
ax.hist(ms[graded & ~bvis], bins=bins, density=True, alpha=0.6, label="pelota ausente", color=CH)
ax.set_xlabel("velocidad media de los jugadores (m/s)"); ax.legend(frameon=False)
ax.set_title("La pelota se pierde cuando el juego se frena")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "07_missing_vs_speed.png")); plt.close(fig)

# 8. players tracked distribution & tracklet lengths
pv = np.isfinite(P[:, :, 0])
seg = []
for kk in range(P.shape[1]):
    _, l = runs(pv[:, kk] & graded); seg.extend(l)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
ax[0].hist(cnt[graded], bins=np.arange(0, 27) - 0.5, color="#333"); ax[0].set_title("jugadores trackeados por frame"); ax[0].axvline(22, ls=":", color="grey")
ax[1].hist(np.array(seg) / FPS, bins=np.logspace(-1.5, 3.5, 40), color="#333"); ax[1].set_xscale("log")
ax[1].set_title(f"duración de tracklets (n={len(seg)})"); ax[1].set_xlabel("s")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "08_tracking_quality.png")); plt.close(fig)
print("ok", sorted(os.listdir(OUT)))
