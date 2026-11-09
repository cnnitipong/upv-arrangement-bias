import os
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(HERE)
DATADIR=os.path.join(ROOT,"data")
OUTDIR=os.path.join(ROOT,"outputs")
os.makedirs(OUTDIR,exist_ok=True)
# -*- coding: utf-8 -*-
"""Additional rigorous analyses for the Structures submission."""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG=np.random.default_rng(11)
OUT=OUTDIR; FIG=os.path.join(OUT,"figures_grey")
m=pd.read_csv(os.path.join(OUT,"master_dataset.csv"))
m["quality"]=pd.Categorical(m["quality"],["Doubtful","Medium","Good","Excellent"],ordered=True)
R={}
morder=["Indirect","Semi-Indirect","Direct"]

# ---------- OLS helper ----------
def ols_sse(y,X):
    beta,_,_,_=np.linalg.lstsq(X,y,rcond=None)
    resid=y-X@beta; return float(resid@resid),beta
def dummies(s):
    d=pd.get_dummies(s,drop_first=True).astype(float); return d.values,list(d.columns)
y=m["vel"].values.astype(float); n=len(y); ybar=y.mean(); SST=float(((y-ybar)**2).sum())
ones=np.ones((n,1))
Xa,_=dummies(m["method"]); Xp,_=dummies(m["priority"].astype(str)); Xc,_=dummies(m["col"])
def cat(*mats): return np.hstack([ones]+list(mats))
SSE0,_=ols_sse(y,ones)
SSE_A,_=ols_sse(y,cat(Xa)); SSE_P,_=ols_sse(y,cat(Xp))
SSE_AP,_=ols_sse(y,cat(Xa,Xp)); SSE_APC,_=ols_sse(y,cat(Xa,Xp,Xc))
SSE_C,_=ols_sse(y,cat(Xc))
R["variance_decomposition"]={
 "R2_arrangement":round(1-SSE_A/SST,3),
 "R2_priority":round(1-SSE_P/SST,3),
 "R2_arrangement_plus_priority":round(1-SSE_AP/SST,3),
 "R2_column_fixed":round(1-SSE_C/SST,3),
 "R2_full":round(1-SSE_APC/SST,3),
 "partial_eta2_arrangement_given_priority":round((SSE_P-SSE_AP)/SSE_P,3),
 "partial_eta2_priority_given_arrangement":round((SSE_A-SSE_AP)/SSE_A,3),
 # arrangement effect WITHIN columns (partial, controlling all column means)
 "partial_eta2_arrangement_within_column":round((SSE_C-SSE_APC)/SSE_C,3),
}
# ICC (between-column share)
grand=ybar; ssb=sum(len(g)*(g.mean()-grand)**2 for _,g in m.groupby("col")["vel"])
R["icc_column"]=round(float(ssb/SST),3)
print("VARIANCE:",json.dumps(R["variance_decomposition"],indent=1),"ICC_col",R["icc_column"])

# ---------- Logistic odds ratios for Doubtful ~ arrangement ----------
m["bad"]=(m["quality"]=="Doubtful").astype(int)
def OR_ci(grp1,grp2,col="bad",n=10000):
    a=m.loc[m["method"]==grp1,col].values; b=m.loc[m["method"]==grp2,col].values
    def od(x): p=x.mean(); p=min(max(p,1e-6),1-1e-6); return p/(1-p)
    obs=od(a)/od(b); boot=[]
    for _ in range(n):
        aa=a[RNG.integers(0,len(a),len(a))]; bb=b[RNG.integers(0,len(b),len(b))]
        boot.append(od(aa)/od(bb))
    return round(float(obs),2),[round(float(np.percentile(boot,2.5)),2),round(float(np.percentile(boot,97.5)),2)]
orv,orci=OR_ci("Indirect","Direct")
orv2,orci2=OR_ci("Indirect","Semi-Indirect")
R["odds_ratio_doubtful"]={"indirect_vs_direct":{"OR":orv,"ci95":orci},
    "indirect_vs_semi":{"OR":orv2,"ci95":orci2}}
print("OR indirect/direct",orv,orci)

# ---------- Leave-one-column-out correction stability ----------
def offset_indirect(df):
    cc=df["vel"]-df.groupby("col")["vel"].transform("mean")
    return cc[df["method"]=="Direct"].mean()-cc[df["method"]=="Indirect"].mean()
base=offset_indirect(m); loco=[]
for c in m["col"].unique():
    sub=m[m["col"]!=c]
    if sub["method"].nunique()>=2 and (sub["method"]=="Indirect").any() and (sub["method"]=="Direct").any():
        loco.append((c,offset_indirect(sub)))
loco_vals=np.array([v for _,v in loco])
R["loco_correction"]={"base_offset":round(float(base),3),"loco_mean":round(float(loco_vals.mean()),3),
    "loco_min":round(float(loco_vals.min()),3),"loco_max":round(float(loco_vals.max()),3),
    "loco_sd":round(float(loco_vals.std()),3),"n_folds":len(loco)}
print("LOCO offset",R["loco_correction"])

# ---------- Monte-Carlo classification reliability ----------
def grade(v): return 3 if v>4.5 else 2 if v>=3.5 else 1 if v>=3.0 else 0  # Excellent..Doubtful index
sig_levels={"1pct":0.01,"2pct":0.02,"3pct":0.03}
rel={}
for name,frac in sig_levels.items():
    flips=np.zeros(n)
    for _ in range(2000):
        noisy=y*(1+RNG.normal(0,frac,n))
        flips+= (np.array([grade(v) for v in noisy])!=np.array([grade(v) for v in y]))
    flips/=2000
    rel[name]={"mean_flip_prob":round(float(flips.mean()),3),
               "pct_points_fragile_gt10":round(float((flips>0.10).mean()*100),1)}
m["flip2pct"]=None
# store per-point flip prob at 2% for figure
flips2=np.zeros(n)
for _ in range(3000):
    noisy=y*(1+RNG.normal(0,0.02,n))
    flips2+= (np.array([grade(v) for v in noisy])!=np.array([grade(v) for v in y]))
flips2/=3000; m["flip2"]=flips2
R["classification_reliability"]=rel
print("RELIABILITY",rel)

# ---------- Concern index vs manual priority agreement ----------
cs=m.groupby("col").agg(vel_mean=("vel","mean"),vel_cov=("vel",lambda s:s.std()/s.mean()*100),
    pct_bad=("bad","mean"),priority=("priority","first"))
cs["pct_bad"]*=100
cs["concern"]=(-(cs["vel_mean"]-cs["vel_mean"].mean())/cs["vel_mean"].std()
    +(cs["vel_cov"]-cs["vel_cov"].mean())/cs["vel_cov"].std()
    +(cs["pct_bad"]-cs["pct_bad"].mean())/(cs["pct_bad"].std()+1e-9))/3
def rank(a): return pd.Series(a).rank().values
rho=np.corrcoef(rank(cs["concern"]),rank(-cs["priority"]))[0,1]  # priority1=worst -> high concern
# permutation p
pr=cs["priority"].values; cv=cs["concern"].values; obs=abs(np.corrcoef(rank(cv),rank(-pr))[0,1])
cnt=0
for _ in range(20000):
    if abs(np.corrcoef(rank(cv),rank(-RNG.permutation(pr)))[0,1])>=obs: cnt+=1
R["concern_vs_priority"]={"spearman_rho":round(float(rho),3),"perm_p":round((cnt+1)/20001,4),
    "n_columns":int(len(cs))}
# disagreements: data-rank vs priority-rank
cs["data_rank"]=cs["concern"].rank(ascending=False)
cs_sorted=cs.sort_values("concern",ascending=False)
R["concern_vs_priority"]["top5_priority"]=[int(p) for p in cs_sorted["priority"].head(5)]
print("CONCERN vs PRIORITY rho",rho,R["concern_vs_priority"]["perm_p"])

# ---------- ROC for the bad-point classifier (reuse logreg) ----------
sup=m.copy()
sup["col_mean_excl"]=(sup.groupby("col")["vel"].transform("sum")-sup["vel"])/(sup.groupby("col")["vel"].transform("size")-1)
Xcat=pd.get_dummies(sup["method"],prefix="m"); Xface=pd.get_dummies(sup["face"],prefix="f")
X=pd.concat([sup[["gx","gy","priority","col_mean_excl"]],Xcat,Xface],axis=1).fillna(0.0).values.astype(float)
X=(X-X.mean(0))/(X.std(0)+1e-9); yb=sup["bad"].values.astype(float)
def logreg(X,y,l2=1.0,lr=0.1,ep=4000):
    w=np.zeros(X.shape[1]); b=0.0
    for _ in range(ep):
        p=1/(1+np.exp(-(X@w+b))); g=p-y; w-=lr*(X.T@g/len(y)+l2*w/len(y)); b-=lr*g.mean()
    return w,b
idx=RNG.permutation(n); folds=np.array_split(idx,5); scores=np.zeros(n)
for f in range(5):
    te=folds[f]; tr=np.concatenate([folds[i] for i in range(5) if i!=f])
    w,b=logreg(X[tr],yb[tr]); scores[te]=1/(1+np.exp(-(X[te]@w+b)))
# ROC
ths=np.linspace(0,1,200); P=yb.sum(); N2=len(yb)-P; tpr=[];fpr=[]
for t in ths:
    pred=scores>=t; tpr.append((pred&(yb==1)).sum()/P); fpr.append((pred&(yb==0)).sum()/N2)
order=np.argsort(fpr); auc=np.trapezoid(np.array(tpr)[order],np.array(fpr)[order])
R["roc"]={"cv_auc":round(float(abs(auc)),3)}
np.savez(os.path.join(OUT,"_extra.npz"),fpr=np.array(fpr),tpr=np.array(tpr),
    loco=loco_vals,base=base,flips2=flips2,vel=y,concern=cs["concern"].values,priority=cs["priority"].values,
    vd=np.array([R["variance_decomposition"]["R2_arrangement"],
                 R["variance_decomposition"]["partial_eta2_priority_given_arrangement"]]))
cs.to_pickle(os.path.join(OUT,"_cs_concern.pkl"))
json.dump(R,open(os.path.join(OUT,"extra_results.json"),"w"),ensure_ascii=False,indent=2)
print("\nALL EXTRA RESULTS:\n",json.dumps(R,ensure_ascii=False,indent=1))
