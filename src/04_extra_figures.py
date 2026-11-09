import os
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(HERE)
DATADIR=os.path.join(ROOT,"data")
OUTDIR=os.path.join(ROOT,"outputs")
os.makedirs(OUTDIR,exist_ok=True)
# -*- coding: utf-8 -*-
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT=OUTDIR; FIG=os.path.join(OUT,"figures_grey")
R=json.load(open(os.path.join(OUT,"extra_results.json")))
E=np.load(os.path.join(OUT,"_extra.npz"))
m=pd.read_csv(os.path.join(OUT,"master_dataset.csv"))
plt.rcParams.update({"figure.dpi":150,"savefig.dpi":300,"savefig.bbox":"tight",
 "font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],"font.size":10,
 "axes.edgecolor":"black","axes.linewidth":0.9,"axes.grid":True,"grid.color":"0.85",
 "grid.linewidth":0.6,"axes.axisbelow":True,"legend.frameon":True,"legend.edgecolor":"0.5"})
def save(fn): plt.savefig(os.path.join(FIG,fn)); plt.close()

# gA: variance decomposition (incremental R2 + partial eta2)
vd=R["variance_decomposition"]
fig,ax=plt.subplots(1,2,figsize=(9.5,3.7))
# incremental R2
inc=[vd["R2_arrangement"], vd["R2_arrangement_plus_priority"]-vd["R2_arrangement"], vd["R2_full"]-vd["R2_arrangement_plus_priority"]]
labels=["Arrangement","+ Priority","+ Column"]
bottom=0
greys=["0.25","0.55","0.8"]; hatches=["","////","...."]
for v,l,g,h in zip(inc,labels,greys,hatches):
    ax[0].bar(0,v,bottom=bottom,color=g,edgecolor="black",hatch=h,width=0.5,label=f"{l} (+{v:.2f})")
    bottom+=v
ax[0].set_xlim(-0.6,0.6); ax[0].set_ylim(0,0.62); ax[0].set_xticks([0]); ax[0].set_xticklabels(["All points"])
ax[0].set_ylabel("Cumulative $R^2$ (velocity)")
ax[0].set_title("(a) Incremental variance explained")
ax[0].legend(fontsize=7.5,loc="upper right",framealpha=1.0)
ax[0].annotate(f"total $R^2$ = {vd['R2_full']:.2f}",(0,bottom),textcoords="offset points",
    xytext=(0,4),ha="center",fontsize=8.5)
# partial eta2
pe=[vd["partial_eta2_arrangement_given_priority"],vd["partial_eta2_priority_given_arrangement"],vd["partial_eta2_arrangement_within_column"]]
pl=["Arrangement\n| priority","Priority\n| arrangement","Arrangement\nwithin column"]
ax[1].bar(pl,pe,color=["0.3","0.7","0.5"],edgecolor="black",hatch=["","////","...."])
for i,v in enumerate(pe): ax[1].annotate(f"{v:.2f}",(i,v),textcoords="offset points",xytext=(0,3),ha="center",fontsize=9)
ax[1].set_ylabel("Partial $\\eta^2$"); ax[1].set_title("(b) Partial effect sizes")
save("gA_variance.png")

# gB: ROC
fpr=E["fpr"]; tpr=E["tpr"]; o=np.argsort(fpr)
fig,ax=plt.subplots(figsize=(4.6,4.3))
ax.plot(fpr[o],tpr[o],color="black",lw=1.6,label=f"Classifier (AUC={R['roc']['cv_auc']})")
ax.plot([0,1],[0,1],ls="--",color="0.5",lw=1.0,label="Chance")
ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate")
ax.set_title("ROC: predicting a poor-quality grade"); ax.legend(fontsize=8,loc="lower right")
ax.set_xlim(0,1); ax.set_ylim(0,1)
save("gB_roc.png")

# gC: LOCO correction stability
loco=E["loco"]; base=float(E["base"])
fig,ax=plt.subplots(figsize=(6.0,3.3))
xs=np.arange(len(loco))
ax.axhline(base,color="black",lw=1.2,label=f"Full-sample offset = {base:.3f} km/s")
ax.axhspan(loco.min(),loco.max(),color="0.85",label=f"LOCO range [{loco.min():.3f}, {loco.max():.3f}]")
ax.scatter(xs,loco,s=28,facecolor="0.4",edgecolor="black",zorder=3)
ax.set_xlabel("Left-out column (index)"); ax.set_ylabel("Indirect$\\rightarrow$direct offset (km/s)")
ax.set_title("Leave-one-column-out stability of the correction"); ax.legend(fontsize=8,loc="upper right")
ax.set_ylim(base-0.06,base+0.06)
save("gC_loco.png")

# gD: classification reliability
flips=E["flips2"]; vel=E["vel"]
fig,ax=plt.subplots(1,2,figsize=(9.5,3.7))
ax[0].scatter(vel,flips,s=16,facecolor="0.55",edgecolor="black",linewidth=0.3)
for t in [3.0,3.5,4.5]: ax[0].axvline(t,ls="--",color="black",lw=0.9)
ax[0].set_xlabel("Pulse velocity (km/s)"); ax[0].set_ylabel("Grade-flip probability (2% noise)")
ax[0].set_title("(a) Classification fragility near thresholds")
rel=R["classification_reliability"]
lv=["1pct","2pct","3pct"]; vals=[rel[k]["pct_points_fragile_gt10"] for k in lv]
ax[1].bar([l.replace("pct","%") for l in lv],vals,color=["0.75","0.5","0.25"],edgecolor="black")
for i,v in enumerate(vals): ax[1].annotate(f"{v}%",(i,v),textcoords="offset points",xytext=(0,3),ha="center",fontsize=9)
ax[1].set_xlabel("Assumed measurement repeatability ($\\sigma$)"); ax[1].set_ylabel("% points fragile (flip prob > 0.10)")
ax[1].set_title("(b) Fragility vs measurement noise")
save("gD_reliability.png")

# gE: concern vs priority
cs=pd.read_pickle(os.path.join(OUT,"_cs_concern.pkl"))
fig,ax=plt.subplots(figsize=(5.0,4.0))
data=[cs.loc[cs["priority"]==p,"concern"].values for p in [1,2,3]]
bp=ax.boxplot(data,tick_labels=["P1","P2","P3"],patch_artist=True,widths=0.55,
    medianprops=dict(color="black",lw=1.2))
for patch,g in zip(bp["boxes"],["0.3","0.55","0.8"]): patch.set_facecolor(g); patch.set_edgecolor("black")
# overlay points
for i,p in enumerate([1,2,3]):
    yv=cs.loc[cs["priority"]==p,"concern"].values
    ax.scatter(np.full(len(yv),i+1)+np.random.uniform(-0.12,0.12,len(yv)),yv,s=20,facecolor="white",edgecolor="black",zorder=3)
ax.set_ylabel("Composite concern index (z)"); ax.set_xlabel("Engineers' manual priority")
ax.set_title(f"Concern index vs manual priority\n(Spearman $\\rho$={R['concern_vs_priority']['spearman_rho']}, p={R['concern_vs_priority']['perm_p']})")
save("gE_concern_priority.png")

print("EXTRA FIGURES:",[f for f in sorted(os.listdir(FIG)) if f.startswith('g') and f[1].isupper()])
