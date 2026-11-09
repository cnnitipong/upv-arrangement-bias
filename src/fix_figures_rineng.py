# -*- coding: utf-8 -*-
"""Fix four RINENG manuscript figures + regenerate two bonus panels with 'Semi-direct' label."""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
A = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs", "figures_rineng")
os.makedirs(OUT, exist_ok=True)

m = pd.read_csv(os.path.join(A, "master_dataset.csv"))
m["quality"] = pd.Categorical(m["quality"], ["Doubtful", "Medium", "Good", "Excellent"], ordered=True)
E = json.load(open(os.path.join(A, "extra_results.json")))

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 11,
    "axes.edgecolor": "black", "axes.linewidth": 0.9, "axes.grid": True, "grid.color": "0.85",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": True, "legend.edgecolor": "0.5",
})

morder = ["Indirect", "Semi-Indirect", "Direct"]
DISPLAY = {"Indirect": "Indirect", "Semi-Indirect": "Semi-direct", "Direct": "Direct"}
morder_disp = [DISPLAY[mm] for mm in morder]
GMETH = {"Indirect": "0.78", "Semi-Indirect": "0.5", "Direct": "0.2"}
HMETH = {"Indirect": "....", "Semi-Indirect": "////", "Direct": ""}
MARK = {"Indirect": "o", "Semi-Indirect": "s", "Direct": "^"}
PANEL = (4.7, 3.7)


def save(fn):
    plt.savefig(os.path.join(OUT, fn))
    plt.close()


# ---------- 1. g1a_distribution.png ----------
fig, ax = plt.subplots(figsize=PANEL)
counts, bins, patches = ax.hist(m["vel"], bins=28, color="0.6", edgecolor="black", lw=0.6)
ymax = counts.max() * 1.15
ax.set_ylim(0, ymax)
for t in [3.0, 3.5, 4.5]:
    ax.axvline(t, ls="--", color="black", lw=0.9)
    ax.text(t, ymax * 0.97, f"{t}", ha="center", va="top", fontsize=8,
             bbox=dict(fc="white", ec="none", pad=1.5))
ax.set_xlabel("Pulse velocity (km/s)")
ax.set_ylabel("Count")
save("g1a_distribution.png")

# ---------- 2. gAa_incremental.png ----------
vd = E["variance_decomposition"]
fig, ax = plt.subplots(figsize=PANEL)
inc = [vd["R2_arrangement"], vd["R2_arrangement_plus_priority"] - vd["R2_arrangement"],
       vd["R2_full"] - vd["R2_arrangement_plus_priority"]]
labels = ["Arrangement", "+ Priority", "+ Column"]
greys = ["0.25", "0.55", "0.8"]
hatches = ["", "////", "...."]
bottom = 0
for v, l, g, h in zip(inc, labels, greys, hatches):
    ax.bar(0, v, bottom=bottom, color=g, edgecolor="black", hatch=h, width=0.5,
           label=f"{l} (+{v:.2f})")
    bottom += v
ax.set_xlim(-0.6, 0.6)
ax.set_ylim(0, 0.65)
ax.set_xticks([0])
ax.set_xticklabels(["All points"])
ax.set_ylabel("Cumulative $R^2$ (velocity)")
ax.legend(fontsize=8, loc="upper left", framealpha=1.0)
ax.annotate(f"total $R^2$ = {vd['R2_full']:.2f}", (0, bottom), textcoords="offset points",
            xytext=(0, 4), ha="center", fontsize=8.5)
save("gAa_incremental.png")

# ---------- 3. g4b_rebound_hist.png ----------
fig, ax = plt.subplots(figsize=PANEL)
counts, bins, patches = ax.hist(m["rebN"], bins=20, color="0.6", edgecolor="black", lw=0.6)
ax.axvline(55, ls="--", color="black", lw=1.0)
ymax = counts.max() * 1.15
ax.set_ylim(0, ymax)
ax.text(56.5, ymax * 0.97, "ceiling $N\\approx55$", fontsize=8, ha="left", va="top",
         bbox=dict(fc="white", ec="none", pad=1.5))
ax.set_xlabel("Rebound number, $N$")
ax.set_ylabel("Count")
save("g4b_rebound_hist.png")

# ---------- 4. m2_flow_overall.png ----------
def box(ax, x, y, w, h, text, fs=8.4, bold=False, emph=False):
    lw = 2.0 if emph else 1.1
    p = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                        boxstyle="round,pad=0.012,rounding_size=0.02",
                        facecolor="white", edgecolor="black", lw=lw, zorder=2)
    ax.add_patch(p)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if (bold or emph) else "normal", zorder=4)


def arrow_between(ax, x1, y1, h1, x2, y2, h2, gap=0.03):
    """Arrow from bottom edge of box1 (center x1,y1, height h1) to top edge of box2 (x2,y2,h2)."""
    ystart = y1 - h1 / 2 - gap
    yend = y2 + h2 / 2 + gap
    ax.add_patch(FancyArrowPatch((x1, ystart), (x2, yend), arrowstyle="-|>",
                                  mutation_scale=12, lw=1.2, color="black", zorder=3))


fig, ax = plt.subplots(figsize=(9.6, 5.8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 11)
ax.axis("off")

# Top / master boxes
top_x, top_y, top_w, top_h = 5, 10.3, 4.6, 0.9
box(ax, top_x, top_y, top_w, top_h,
    "Inspection report: 462 UPV points\n26 RC columns, post-earthquake", emph=True)

extr_x, extr_y, extr_w, extr_h = 5, 9.0, 3.9, 0.7
box(ax, extr_x, extr_y, extr_w, extr_h, "Extraction and validation against summary tables")

master_x, master_y, master_w, master_h = 5, 7.7, 5.0, 0.9
box(ax, master_x, master_y, master_w, master_h,
    "Master dataset: $v$, $N$, arrangement, quality,\npriority, grid coordinates, face geometry",
    emph=True)

arrow_between(ax, top_x, top_y, top_h, extr_x, extr_y, extr_h)
arrow_between(ax, extr_x, extr_y, extr_h, master_x, master_y, master_h)

# Column headers
left_x, right_x = 2.7, 7.3
head_y = 6.5
head_w, head_h = 3.9, 0.6
arrow_between(ax, master_x, master_y, master_h, left_x, head_y, head_h)
arrow_between(ax, master_x, master_y, master_h, right_x, head_y, head_h)

box(ax, left_x, head_y, head_w, head_h, "Diagnosis and correction", fs=9, emph=True)
box(ax, right_x, head_y, head_w, head_h, "Bias-aware triage framework", fs=9, emph=True)

p1 = ["Distributions and rebound saturation",
      "Variance decomposition ($\\eta^2$)",
      "Grade-arrangement association ($\\chi^2$, $V$)",
      "Arrangement correction $\\delta$",
      "Reclassification and grade reliability",
      "Probabilistic grading and repair decision"]
p2 = ["PCA (condition axis)",
      "Clustering (k-means, DBSCAN)",
      "Mahalanobis anomaly detection",
      "Heterogeneity (CoV) feature",
      "Composite concern index",
      "Comparison with manual priority"]

item_h = 0.5
pitch = 0.68
item_w = 3.9

y = head_y - pitch
prev_y, prev_h = head_y, head_h
for t in p1:
    box(ax, left_x, y, item_w, item_h, t, fs=8.4)
    arrow_between(ax, left_x, prev_y, prev_h, left_x, y, item_h)
    prev_y, prev_h = y, item_h
    y -= pitch
last_left_y, last_left_h = prev_y, prev_h

y = head_y - pitch
prev_y, prev_h = head_y, head_h
for t in p2:
    box(ax, right_x, y, item_w, item_h, t, fs=8.4)
    arrow_between(ax, right_x, prev_y, prev_h, right_x, y, item_h)
    prev_y, prev_h = y, item_h
    y -= pitch
last_right_y, last_right_h = prev_y, prev_h

# Bottom output box
bot_x, bot_y, bot_w, bot_h = 5, 0.55, 5.6, 0.8
box(ax, bot_x, bot_y, bot_w, bot_h,
    "Outputs: corrected, probabilistic grades and auditable repair triage", emph=True)

arrow_between(ax, left_x, last_left_y, last_left_h, bot_x, bot_y, bot_h)
arrow_between(ax, right_x, last_right_y, last_right_h, bot_x, bot_y, bot_h)

save("m2_flow_overall.png")

# ---------- Bonus: g1b_boxplot.png ----------
fig, ax = plt.subplots(figsize=PANEL)
data = [m.loc[m["method"] == mm, "vel"].values for mm in morder]
bp = ax.boxplot(data, tick_labels=morder_disp, patch_artist=True, widths=0.6,
                 medianprops=dict(color="black", lw=1.2),
                 whiskerprops=dict(color="black"), capprops=dict(color="black"),
                 flierprops=dict(marker="o", markersize=3, markerfacecolor="0.5", markeredgecolor="black"))
for p, mm in zip(bp["boxes"], morder):
    p.set_facecolor(GMETH[mm])
    p.set_edgecolor("black")
    p.set_hatch(HMETH[mm])
ax.set_ylabel("Pulse velocity (km/s)")
eps2, ci_lo, ci_hi = 0.459, 0.385, 0.531
ax.text(.03, .04,
        f"$\\varepsilon^2$ = {eps2:.2f} (95% CI {ci_lo:.2f} to {ci_hi:.2f}), $p<0.001$",
        transform=ax.transAxes, fontsize=8, bbox=dict(fc="white", ec="0.5"))
save("g1b_boxplot.png")

# ---------- Bonus: g4a_scatter.png ----------
fig, ax = plt.subplots(figsize=PANEL)
for mm in morder:
    s = m[m["method"] == mm]
    ax.scatter(s["rebN"], s["vel"], s=20, facecolor=GMETH[mm], edgecolor="black", lw=0.4,
               marker=MARK[mm], label=DISPLAY[mm], alpha=0.85)
b = np.polyfit(m["rebN"], m["vel"], 1)
xs = np.linspace(m["rebN"].min(), m["rebN"].max(), 50)
ax.plot(xs, np.polyval(b, xs), "k-", lw=1.4)
ax.set_xlabel("Rebound number, $N$")
ax.set_ylabel("Pulse velocity (km/s)")
ax.legend(fontsize=8)
save("g4a_scatter.png")

print("Done.")
