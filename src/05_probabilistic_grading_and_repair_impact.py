# -*- coding: utf-8 -*-
"""Probabilistic quality classification (#2) + repair-decision impact (#5)."""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
RNG=np.random.default_rng(20)
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
A=os.path.join(ROOT,"outputs")
FIG=os.path.join(A,"figures_grey"); os.makedirs(FIG,exist_ok=True)
m=pd.read_csv(os.path.join(A,"master_dataset.csv"))
D=json.load(open(os.path.join(A,"deep_results.json")))
off=D["correction"]["offset_kms"]                      # Indirect +0.237, Semi -0.055, Direct 0
ci=D["correction"]["indirect_offset_ci95"]             # [0.16,0.30]
sig_delta={"Indirect":(ci[1]-ci[0])/(2*1.96),"Semi-Indirect":0.030,"Direct":0.0}
SIG_REP=0.02     # 2% repeatability
THRESH=[(4.5,"Excellent"),(3.5,"Good"),(3.0,"Medium"),(-1,"Doubtful")]
classes=["Doubtful","Medium","Good","Excellent"]
def grade(v): return "Excellent" if v>4.5 else "Good" if v>=3.5 else "Medium" if v>=3.0 else "Doubtful"

# corrected (direct-equivalent) velocity
m["vel_corr"]=m["vel"]+m["method"].map(off)
M=20000
P=np.zeros((len(m),4))   # columns = classes order
for i,(v,meth) in enumerate(zip(m["vel_corr"].values,m["method"].values)):
    s=np.sqrt((SIG_REP*v)**2+sig_delta[meth]**2)
    samp=RNG.normal(v,s,M)
    g=np.where(samp>4.5,3,np.where(samp>=3.5,2,np.where(samp>=3.0,1,0)))
    P[i]=np.bincount(g,minlength=4)/M
prob=pd.DataFrame(P,columns=classes,index=m.index)
m["P_Doubtful"]=prob["Doubtful"]; m["P_Medium"]=prob["Medium"]; m["P_Good"]=prob["Good"]; m["P_Excellent"]=prob["Excellent"]
m["maxP"]=prob.max(axis=1); m["argclass"]=prob.idxmax(axis=1)

# ----- probabilistic summary -----
expected=prob.sum(axis=0)                     # expected count per class
hard=m["quality"].value_counts().reindex(classes)
ambiguous=(m["maxP"]<0.80).mean()*100         # % points with no confident grade
RES={
 "sigma_delta_indirect":round(sig_delta["Indirect"],4),
 "expected_class_counts":{k:round(float(expected[k]),1) for k in classes},
 "hard_class_counts":{k:int(hard[k]) for k in classes},
 "pct_points_ambiguous_maxP_lt_0.8":round(float(ambiguous),1),
 "pct_points_confident":round(float((m["maxP"]>=0.8).mean()*100),1),
}

# ----- repair-decision impact (#5) -----
# Repair trigger = "Doubtful" (poor integrity). Compare original hard / corrected hard / probabilistic.
m["quality_corr"]=m["vel_corr"].map(grade)
pts_orig=int((m["quality"]=="Doubtful").sum())
pts_corr=int((m["quality_corr"]=="Doubtful").sum())
pts_expected=float(prob["Doubtful"].sum())
pts_prob_gt50=int((m["P_Doubtful"]>0.5).sum())
def cols_with_poor(col): return m.groupby("col").apply(lambda d:(d[col]=="Doubtful").any()).sum()
cols_orig=int(cols_with_poor("quality")); cols_corr=int(cols_with_poor("quality_corr"))
# probabilistic: column flagged if expected doubtful points >=1 OR any point P>0.5
cols_prob=int(m.groupby("col")["P_Doubtful"].apply(lambda s:(s.sum()>=1.0) or (s>0.5).any()).sum())
RES["repair_impact"]={
 "doubtful_points_original":pts_orig,
 "doubtful_points_corrected":pts_corr,
 "doubtful_points_expected_prob":round(pts_expected,1),
 "doubtful_points_prob_gt50":pts_prob_gt50,
 "pct_reduction_points":round(100*(pts_orig-pts_corr)/pts_orig,1),
 "columns_triggering_repair_original":cols_orig,
 "columns_triggering_repair_corrected":cols_corr,
 "columns_triggering_repair_prob":cols_prob,
 "n_columns_total":int(m["col"].nunique()),
}
json.dump(RES,open(os.path.join(A,"novelty_results.json"),"w"),ensure_ascii=False,indent=2)
print(json.dumps(RES,ensure_ascii=False,indent=1))
m[["col","point","method","vel","vel_corr","quality","quality_corr","P_Doubtful","P_Medium","P_Good","P_Excellent","maxP","argclass"]].to_csv(os.path.join(A,"probabilistic_grades.csv"),index=False,encoding="utf-8-sig")

# ================= FIGURES =================
plt.rcParams.update({"figure.dpi":150,"savefig.dpi":300,"savefig.bbox":"tight",
 "font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],"font.size":11,
 "axes.edgecolor":"black","axes.linewidth":0.9,"axes.grid":True,"grid.color":"0.85",
 "grid.linewidth":0.6,"axes.axisbelow":True,"legend.frameon":True,"legend.edgecolor":"0.5"})
GREYS={"Doubtful":"0.15","Medium":"0.45","Good":"0.70","Excellent":"0.88"}
HATCH={"Doubtful":"","Medium":"////","Good":"....","Excellent":"xxxx"}
def save(fn): plt.savefig(os.path.join(FIG,fn)); plt.close()
PANEL=(4.7,3.7)

# Fig P1a: max-class probability vs velocity (ambiguity bands)
fig,ax=plt.subplots(figsize=PANEL)
ax.scatter(m["vel_corr"],m["maxP"],s=16,facecolor="0.55",edgecolor="black",lw=0.3)
for t in [3.0,3.5,4.5]: ax.axvline(t,ls="--",color="black",lw=0.9)
ax.axhline(0.8,ls=":",color="black",lw=1.0); ax.text(2.3,0.81,"confidence 0.8",fontsize=8)
ax.set_xlabel("Direct-equivalent velocity (km/s)"); ax.set_ylabel("Probability of assigned class"); save("p1a_confidence.png")

# Fig P1b: hard vs expected class counts
fig,ax=plt.subplots(figsize=PANEL)
x=np.arange(4); w=0.38
ax.bar(x-w/2,[hard[c] for c in classes],w,color="0.35",edgecolor="black",label="Hard (deterministic)")
ax.bar(x+w/2,[expected[c] for c in classes],w,color="0.8",edgecolor="black",hatch="////",label="Expected (probabilistic)")
ax.set_xticks(x); ax.set_xticklabels(classes,rotation=15); ax.set_ylabel("Number of points"); ax.legend(fontsize=8); save("p1b_counts.png")

# Fig P2a: repair-flagged points
fig,ax=plt.subplots(figsize=PANEL)
labs=["Original\n(hard)","Probabilistic\n(expected)","Corrected\n(hard)"]
vals=[pts_orig,pts_expected,pts_corr]
ax.bar(labs,vals,color=["0.25","0.55","0.8"],edgecolor="black",hatch=["","....","////"])
for i,v in enumerate(vals): ax.annotate(f"{v:.0f}" if i!=1 else f"{v:.1f}",(i,v),textcoords="offset points",xytext=(0,3),ha="center",fontsize=10)
ax.set_ylabel("Points flagged for repair (Doubtful)"); save("p2a_points.png")

# Fig P2b: columns triggering repair
fig,ax=plt.subplots(figsize=PANEL)
labs=["Original","Probabilistic","Corrected"]; vals=[cols_orig,cols_prob,cols_corr]
ax.bar(labs,vals,color=["0.25","0.55","0.8"],edgecolor="black",hatch=["","....","////"])
for i,v in enumerate(vals): ax.annotate(str(v),(i,v),textcoords="offset points",xytext=(0,3),ha="center",fontsize=10)
ax.set_ylabel(f"Columns triggering repair (of {RES['repair_impact']['n_columns_total']})"); save("p2b_columns.png")

print("\nFIGURES: p1a_confidence p1b_counts p2a_points p2b_columns")
