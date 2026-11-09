import os
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(HERE)
DATADIR=os.path.join(ROOT,"data")
OUTDIR=os.path.join(ROOT,"outputs")
os.makedirs(OUTDIR,exist_ok=True)
# -*- coding: utf-8 -*-
"""Deeper analysis (bias correction, reclassification, bootstrap CIs) + grayscale figures."""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

RNG=np.random.default_rng(7)
OUT=OUTDIR; FIG=os.path.join(OUT,"figures_grey")
os.makedirs(FIG,exist_ok=True)
m=pd.read_csv(os.path.join(OUT,"master_dataset.csv"))
m["quality"]=pd.Categorical(m["quality"],["Doubtful","Medium","Good","Excellent"],ordered=True)

# ---------- grayscale academic style ----------
plt.rcParams.update({
 "figure.dpi":150,"savefig.dpi":300,"savefig.bbox":"tight",
 "font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],"font.size":10,
 "axes.edgecolor":"black","axes.linewidth":0.9,"axes.grid":True,
 "grid.color":"0.85","grid.linewidth":0.6,"axes.axisbelow":True,
 "xtick.color":"black","ytick.color":"black","text.color":"black","axes.labelcolor":"black",
 "axes.titlesize":10.5,"legend.frameon":True,"legend.edgecolor":"0.5","legend.framealpha":1.0})
GREYS=["0.15","0.45","0.70","0.88"]   # dark->light
HATCH=["","////","....","xxxx"]
MARK=["o","s","^","D"]
RES={}
morder=["Indirect","Semi-Indirect","Direct"]; qorder=["Doubtful","Medium","Good","Excellent"]
GMETH={"Indirect":"0.78","Semi-Indirect":"0.5","Direct":"0.2"}
HMETH={"Indirect":"....","Semi-Indirect":"////","Direct":""}

def boot_mean_ci(x,n=5000):
    x=np.asarray(x,float); out=[np.mean(x[RNG.integers(0,len(x),len(x))]) for _ in range(n)]
    return float(np.mean(x)),float(np.percentile(out,2.5)),float(np.percentile(out,97.5))

# ============ DEEPER ANALYSIS ============
# (1) column-centred bias per arrangement with bootstrap CI
m["vel_cc"]=m["vel"]-m.groupby("col")["vel"].transform("mean")
bias={}
for mm in morder:
    mu,lo,hi=boot_mean_ci(m.loc[m["method"]==mm,"vel_cc"])
    bias[mm]={"mean":round(mu,3),"lo":round(lo,3),"hi":round(hi,3),"n":int((m["method"]==mm).sum())}
# (2) pairwise raw mean differences with bootstrap CI
def boot_diff_ci(a,b,n=5000):
    a=np.asarray(a,float);b=np.asarray(b,float);out=[]
    for _ in range(n):
        out.append(a[RNG.integers(0,len(a),len(a))].mean()-b[RNG.integers(0,len(b),len(b))].mean())
    return float(a.mean()-b.mean()),float(np.percentile(out,2.5)),float(np.percentile(out,97.5))
pairs={}
for x,y in [("Direct","Indirect"),("Direct","Semi-Indirect"),("Semi-Indirect","Indirect")]:
    d,lo,hi=boot_diff_ci(m.loc[m["method"]==x,"vel"],m.loc[m["method"]==y,"vel"])
    pairs[f"{x}-{y}"]={"diff":round(d,3),"lo":round(lo,3),"hi":round(hi,3)}
# (3) correction factor to direct-equivalent + reclassification
ccmean={mm:m.loc[m["method"]==mm,"vel_cc"].mean() for mm in morder}
offset={mm:ccmean["Direct"]-ccmean[mm] for mm in morder}   # add to reach direct datum
m["vel_corr"]=m["vel"]+m["method"].map(offset)
def grade(v):
    return "Excellent" if v>4.5 else "Good" if v>=3.5 else "Medium" if v>=3.0 else "Doubtful"
m["quality_corr"]=m["vel_corr"].map(grade)
# bootstrap CI for indirect offset
off_boot=[]
for _ in range(5000):
    s=m.sample(frac=1,replace=True,random_state=RNG.integers(1e9))
    cc=s["vel"]-s.groupby("col")["vel"].transform("mean")
    cm={mm:cc[s["method"]==mm].mean() for mm in morder}
    off_boot.append(cm["Direct"]-cm["Indirect"])
RES["correction"]={"offset_kms":{k:round(float(v),3) for k,v in offset.items()},
    "indirect_offset_ci95":[round(float(np.percentile(off_boot,2.5)),3),round(float(np.percentile(off_boot,97.5)),3)]}
# reclassification of Doubtful
doubt=m[m["quality"]=="Doubtful"]
resc=doubt[doubt["quality_corr"]!="Doubtful"]
RES["reclassification"]={"doubtful_original":int(len(doubt)),
    "doubtful_reclassified_after_correction":int(len(resc)),
    "pct_reclassified":round(100*len(resc)/len(doubt),1),
    "doubtful_remaining":int(len(doubt)-len(resc)),
    "of_indirect_doubtful":int((doubt["method"]=="Indirect").sum()),
    "indirect_doubtful_reclassified":int((resc["method"]=="Indirect").sum())}
RES["bias_cc"]=bias; RES["pairwise_diff"]=pairs
# (4) effect size epsilon2 with bootstrap CI
def kruskalH(groups):
    allv=np.concatenate(groups); r=pd.Series(allv).rank().values; N=len(allv); idx=0; H=0
    for gp in groups:
        ng=len(gp); rg=r[idx:idx+ng]; idx+=ng; H+=ng*(rg.mean()-(N+1)/2)**2
    return 12/(N*(N+1))*H
g=[m.loc[m["method"]==mm,"vel"].values for mm in morder]
H=kruskalH(g); N=len(m); eps2=H/(N-1)
e2b=[]
for _ in range(2000):
    s=m.sample(frac=1,replace=True,random_state=RNG.integers(1e9))
    gg=[s.loc[s["method"]==mm,"vel"].values for mm in morder]
    if all(len(x)>1 for x in gg): e2b.append(kruskalH(gg)/(len(s)-1))
RES["effect_size"]={"eps2":round(float(eps2),3),"ci95":[round(float(np.percentile(e2b,2.5)),3),round(float(np.percentile(e2b,97.5)),3)]}
json.dump(RES,open(os.path.join(OUT,"deep_results.json"),"w"),ensure_ascii=False,indent=2)
print(json.dumps(RES,ensure_ascii=False,indent=1))

# ============ FIGURES (grayscale) ============
def save(fn): plt.savefig(os.path.join(FIG,fn)); plt.close()

# G1: velocity distribution + by arrangement
fig,ax=plt.subplots(1,2,figsize=(9.5,3.7))
ax[0].hist(m["vel"],bins=28,color="0.6",edgecolor="black",linewidth=0.6)
for t in [3.0,3.5,4.5]:
    ax[0].axvline(t,ls="--",color="black",lw=0.9)
    ax[0].text(t,ax[0].get_ylim()[1]*0.93,f"{t}",ha="center",fontsize=8)
ax[0].set_title("(a) Pulse-velocity distribution"); ax[0].set_xlabel("Pulse velocity (km/s)"); ax[0].set_ylabel("Count")
data=[m.loc[m["method"]==mm,"vel"].values for mm in morder]
bp=ax[1].boxplot(data,tick_labels=morder,patch_artist=True,widths=0.6,
    medianprops=dict(color="black",lw=1.2),whiskerprops=dict(color="black"),
    capprops=dict(color="black"),flierprops=dict(marker="o",markersize=3,markerfacecolor="0.5",markeredgecolor="black"))
for patch,mm in zip(bp["boxes"],morder):
    patch.set_facecolor(GMETH[mm]); patch.set_edgecolor("black"); patch.set_hatch(HMETH[mm])
ax[1].set_title("(b) Velocity by transducer arrangement"); ax[1].set_ylabel("Pulse velocity (km/s)")
ax[1].text(.03,.04,f"$\\epsilon^2$={RES['effect_size']['eps2']}  (95% CI {RES['effect_size']['ci95']}),  p<0.001",
    transform=ax[1].transAxes,fontsize=8,bbox=dict(fc="white",ec="0.5"))
save("g1_velocity_arrangement.png")

# G2: bias with bootstrap CI (NEW) + pairwise diffs
fig,ax=plt.subplots(1,2,figsize=(9.5,3.7))
xs=np.arange(len(morder))
means=[bias[mm]["mean"] for mm in morder]
err=[[bias[mm]["mean"]-bias[mm]["lo"] for mm in morder],[bias[mm]["hi"]-bias[mm]["mean"] for mm in morder]]
ax[0].axhline(0,color="black",lw=0.8)
ax[0].errorbar(xs,means,yerr=err,fmt="o",color="black",capsize=4,markersize=7,markerfacecolor="0.4")
ax[0].set_xticks(xs); ax[0].set_xticklabels(morder); ax[0].set_ylabel("Column-centred velocity (km/s)")
ax[0].set_title("(a) Intrinsic arrangement bias (95% CI)")
for i,mm in enumerate(morder): ax[0].annotate(f"{means[i]:+.2f}",(xs[i],means[i]),textcoords="offset points",xytext=(8,0),fontsize=8)
pk=list(pairs.keys()); pv=[pairs[k]["diff"] for k in pk]
perr=[[pairs[k]["diff"]-pairs[k]["lo"] for k in pk],[pairs[k]["hi"]-pairs[k]["diff"] for k in pk]]
yp=np.arange(len(pk))
ax[1].axvline(0,color="black",lw=0.8)
ax[1].errorbar(pv,yp,xerr=perr,fmt="s",color="black",capsize=4,markersize=7,markerfacecolor="0.6")
ax[1].set_yticks(yp); ax[1].set_yticklabels(pk); ax[1].set_xlabel("Mean velocity difference (km/s)")
ax[1].set_title("(b) Pairwise differences (95% CI)")
for i,k in enumerate(pk): ax[1].annotate(f"{pairs[k]['diff']:.2f}",(pairs[k]['diff'],yp[i]),textcoords="offset points",xytext=(0,8),fontsize=8,ha="center")
save("g2_bias_ci.png")

# G3: quality composition by arrangement (grayscale shades)
fig,ax=plt.subplots(figsize=(5.2,3.8))
ct=pd.crosstab(m["method"],m["quality"]).reindex(morder)[qorder]
ctp=ct.div(ct.sum(1),axis=0)*100; bottom=np.zeros(len(morder))
for q,g_,h in zip(qorder,GREYS,HATCH):
    ax.bar(morder,ctp[q],bottom=bottom,color=g_,edgecolor="black",linewidth=0.6,hatch=h,label=q)
    bottom+=ctp[q].values
ax.set_ylabel("% of points"); ax.set_title("Quality class by transducer arrangement")
ax.legend(fontsize=8,title="Quality",loc="upper left",bbox_to_anchor=(1.01,1.0))
save("g3_quality_arrangement.png")

# G4: saturation
fig,ax=plt.subplots(1,2,figsize=(9.5,3.7))
for mm in morder:
    s=m[m["method"]==mm]
    ax[0].scatter(s["rebN"],s["vel"],s=20,facecolor=GMETH[mm],edgecolor="black",linewidth=0.4,
                  marker=MARK[morder.index(mm)],label=mm,alpha=0.85)
b=np.polyfit(m["rebN"],m["vel"],1); xs=np.linspace(m["rebN"].min(),m["rebN"].max(),50)
ax[0].plot(xs,np.polyval(b,xs),"k-",lw=1.4)
ax[0].set_xlabel("Rebound number, N"); ax[0].set_ylabel("Pulse velocity (km/s)")
ax[0].set_title("(a) UPV vs rebound (r = 0.23)"); ax[0].legend(fontsize=8)
ax[1].hist(m["rebN"],bins=20,color="0.6",edgecolor="black",linewidth=0.6)
ax[1].axvline(55,ls="--",color="black",lw=1.0); ax[1].text(55.2,ax[1].get_ylim()[1]*0.9,"ceiling N$\\approx$55",fontsize=8)
ax[1].set_title("(b) Rebound saturation (CoV 4.6% vs 15%)"); ax[1].set_xlabel("Rebound number, N"); ax[1].set_ylabel("Count")
save("g4_saturation.png")

# G5: reclassification after correction (NEW)
fig,ax=plt.subplots(1,2,figsize=(9.5,3.7))
# (a) before/after grade counts
ga=m["quality"].value_counts().reindex(qorder)
gb=pd.Series(m["quality_corr"]).value_counts().reindex(qorder)
x=np.arange(len(qorder)); w=0.38
ax[0].bar(x-w/2,ga.values,w,color="0.35",edgecolor="black",label="Original")
ax[0].bar(x+w/2,gb.values,w,color="0.8",edgecolor="black",hatch="////",label="After correction")
ax[0].set_xticks(x); ax[0].set_xticklabels(qorder,rotation=15); ax[0].set_ylabel("Count")
ax[0].set_title("(a) Grade counts: indirect/semi $\\rightarrow$ direct-equivalent"); ax[0].legend(fontsize=8)
# (b) doubtful fate
rc=RES["reclassification"]
labels=["Reclassified\n(no longer Doubtful)","Remain\nDoubtful"]
vals=[rc["doubtful_reclassified_after_correction"],rc["doubtful_remaining"]]
ax[1].bar(labels,vals,color=["0.8","0.3"],edgecolor="black",hatch=["////",""])
for i,v in enumerate(vals): ax[1].annotate(str(v),(i,v),textcoords="offset points",xytext=(0,3),ha="center",fontsize=9)
ax[1].set_ylabel("Doubtful points (n=35)")
ax[1].set_title(f"(b) {rc['pct_reclassified']}% of Doubtful were method artefacts")
save("g5_reclassification.png")

# G6: heterogeneity
cs=m.groupby("col").agg(vel_mean=("vel","mean"),vel_cov=("vel",lambda s:s.std()/s.mean()*100),
    pct_bad=("bad","mean"),priority=("priority","first"))
cs["pct_bad"]*=100
fig,ax=plt.subplots(figsize=(5.6,4.2))
szs=40+cs["pct_bad"]*6
sc=ax.scatter(cs["vel_mean"],cs["vel_cov"],s=szs,facecolor="0.6",edgecolor="black",linewidth=0.7)
b=np.polyfit(cs["vel_mean"],cs["vel_cov"],1); xs=np.linspace(cs["vel_mean"].min(),cs["vel_mean"].max(),30)
ax.plot(xs,np.polyval(b,xs),"k--",lw=1.2)
r=np.corrcoef(cs["vel_mean"],cs["vel_cov"])[0,1]
ax.set_xlabel("Column mean velocity (km/s)"); ax.set_ylabel("Within-column CoV (%)")
ax.set_title(f"Heterogeneity vs mean velocity (r = {r:.2f})")
ax.text(.97,.95,"marker size $\\propto$ % Doubtful",transform=ax.transAxes,ha="right",va="top",fontsize=8,bbox=dict(fc="white",ec="0.6"))
save("g6_heterogeneity.png")

# G7: clustering (numpy kmeans k=2) grayscale
F=m[["vel","rebN","gx","gy"]].dropna(); idx=F.index
Fz=(F-F.mean())/F.std()
def kmeans(X,k,it=200,rs=12):
    best=None
    for _ in range(rs):
        c=X[RNG.choice(len(X),k,replace=False)]
        for _ in range(it):
            d=((X[:,None]-c[None])**2).sum(2); lab=d.argmin(1)
            nc=np.array([X[lab==j].mean(0) if (lab==j).any() else c[j] for j in range(k)])
            if np.allclose(nc,c): c=nc; break
            c=nc
        I=((X-c[lab])**2).sum()
        if best is None or I<best[0]: best=(I,lab,c)
    return best
_,lab,_=kmeans(Fz.values,2)
U,S,Vt=np.linalg.svd((Fz-Fz.mean()).values,full_matrices=False); sc=U*S
mq=m.loc[idx]
fig,ax=plt.subplots(1,2,figsize=(9.5,3.7))
for q,gg,mk in zip(qorder,GREYS,MARK):
    sel=(mq["quality"]==q).values
    ax[0].scatter(sc[sel,0],sc[sel,1],s=22,facecolor=gg,edgecolor="black",linewidth=0.3,marker=mk,label=q)
ax[0].set_xlabel("PC1 (46%)"); ax[0].set_ylabel("PC2 (22%)"); ax[0].set_title("(a) PCA, by quality class"); ax[0].legend(fontsize=7)
for cl,gg,mk in zip([0,1],["0.25","0.75"],["o","s"]):
    sel=lab==cl
    ax[1].scatter(mq["gx"].values[sel],mq["gy"].values[sel],s=26,facecolor=gg,edgecolor="black",linewidth=0.3,marker=mk,label=f"Cluster {cl+1}")
ax[1].invert_yaxis(); ax[1].set_xlabel("Grid X (bay)"); ax[1].set_ylabel("Grid Y (gridline A-H)")
ax[1].set_title("(b) k-means clusters (k=2)"); ax[1].legend(fontsize=8)
save("g7_pca_clusters.png")

# G8: anomaly detection grayscale
A=m[["vel","rebN"]].dropna(); ia=A.index
mu=A.mean().values; inv=np.linalg.pinv(np.cov(A.values.T))
md=np.sqrt(((A.values-mu)@inv*(A.values-mu)).sum(1)); thr=np.percentile(md,97.5)
man=m.loc[ia].copy(); man["md"]=md; out=man[man["md"]>=thr]
fig,ax=plt.subplots(figsize=(5.8,4.3))
scsc=ax.scatter(A["rebN"],A["vel"],c=md,cmap="Greys",s=26,edgecolor="black",linewidth=0.3,vmin=0,vmax=md.max())
ax.scatter(out["rebN"],out["vel"],facecolors="none",edgecolors="black",s=150,linewidths=1.6,marker="o")
ax.set_xlabel("Rebound number, N"); ax.set_ylabel("Pulse velocity (km/s)")
ax.set_title(f"Anomaly detection (Mahalanobis, n={len(out)} flagged)")
cb=plt.colorbar(scsc,label="Mahalanobis distance")
save("g8_anomaly.png")

# G9: concern index grayscale
cz=cs.copy()
cz["z"]=(-(cz["vel_mean"]-cz["vel_mean"].mean())/cz["vel_mean"].std()
         +(cz["vel_cov"]-cz["vel_cov"].mean())/cz["vel_cov"].std()
         +(cz["pct_bad"]-cz["pct_bad"].mean())/(cz["pct_bad"].std()+1e-9))/3
cz=cz.sort_values("z",ascending=False).head(12).iloc[::-1]
fig,ax=plt.subplots(figsize=(6.2,4.4))
grad={1:"0.2",2:"0.5",3:"0.78"}
cols=[grad[p] for p in cz["priority"]]
ax.barh([n.replace("C1","").replace(" ","") for n in cz.index],cz["z"],color=cols,edgecolor="black")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="0.2",edgecolor="black",label="Priority 1"),
    Patch(facecolor="0.5",edgecolor="black",label="Priority 2"),
    Patch(facecolor="0.78",edgecolor="black",label="Priority 3")],fontsize=8,loc="lower right")
ax.set_xlabel("Composite structural-concern index (z)"); ax.set_title("Data-driven column concern ranking")
save("g9_concern.png")

# ---------- FIELD figure (grayscale composite) ----------
def to_grey(path):
    im=Image.open(path).convert("L")
    return np.asarray(im)
if all(os.path.exists(os.path.join(DATADIR,f)) for f in ("field_upv.png","field_grid.png")):
    fig,ax=plt.subplots(1,2,figsize=(9.5,3.7))
    ax[0].imshow(to_grey(os.path.join(DATADIR,"field_upv.png")),cmap="gray")
    ax[0].set_title("(a) In-situ UPV measurement (indirect arrangement)",fontsize=9.5); ax[0].axis("off")
    ax[1].imshow(to_grey(os.path.join(DATADIR,"field_grid.png")),cmap="gray")
    ax[1].set_title("(b) Column with marked UPV test-point grid",fontsize=9.5); ax[1].axis("off")
    plt.savefig(os.path.join(FIG,"g0_field.png"),dpi=200,bbox_inches="tight"); plt.close()
else:
    print("NOTE: field photographs not in data/ - skipping Figure g0_field.png (photographs not distributed)")

print("\nFIGURES:",sorted(os.listdir(FIG)))
