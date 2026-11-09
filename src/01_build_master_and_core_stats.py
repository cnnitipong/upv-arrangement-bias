import os
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(HERE)
DATADIR=os.path.join(ROOT,"data")
OUTDIR=os.path.join(ROOT,"outputs")
os.makedirs(OUTDIR,exist_ok=True)
# -*- coding: utf-8 -*-
"""
UPV structural-assessment dataset — full analysis (traditional + AI/ML).
Pure numpy/pandas/matplotlib implementation (no scipy/sklearn available).
Statistical inference via permutation + bootstrap (robust at n=462).
"""
import os, re, json, itertools, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)
BASE = DATADIR
OUT  = OUTDIR
FIG  = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi":130,"savefig.dpi":150,"font.size":10,
                     "axes.grid":True,"grid.alpha":0.25,"axes.axisbelow":True})

RESULTS = {}

# ----------------------------------------------------------------------
# 1. BUILD MASTER DATASET
# ----------------------------------------------------------------------
upv = pd.read_csv(os.path.join(BASE,"UPV_summary_all_locations.csv"))
upv.columns = ["col","point","vel","vel_mean","ksc","priority","method","quality_th"]

# per-point rebound number N (full 462-pt coverage) and MPa (partial)
reb = pd.read_csv(os.path.join(BASE,"rebound_hammer_direction.csv"))
reb.columns = ["col","point","method2","vel","rebN"]
reb["col"] = reb["col"].str.replace("เสา ","",regex=False).str.strip()
mpa = pd.read_csv(os.path.join(BASE,"Result_Mpa.csv"))
mpa.columns = ["col","point","vel","mpa"]
mpa["col"] = mpa["col"].str.replace("เสา ","",regex=False).str.strip()

def norm(s): return re.sub(r"\s+","",str(s))
for d in (upv,reb,mpa): d["key"]=d["col"].map(norm)+"|"+d["point"].map(norm)

m = upv.merge(reb[["key","rebN"]],on="key",how="left").merge(mpa[["key","mpa"]],on="key",how="left")
# derive MPa for all points from rebound N (linear fit on the covered subset)
_cov=m.dropna(subset=["mpa"]); _b=np.polyfit(_cov["rebN"],_cov["mpa"],1)
m["mpa"]=m["mpa"].fillna(pd.Series(np.polyval(_b,m["rebN"]),index=m.index))

# English quality
qmap={"ดีเยี่ยม":"Excellent","ดี":"Good","ปานกลาง":"Medium","แย่":"Doubtful"}
m["quality"]=m["quality_th"].map(qmap)
qorder=["Doubtful","Medium","Good","Excellent"]
m["quality"]=pd.Categorical(m["quality"],categories=qorder,ordered=True)

# ---- spatial parse from column name ----
def parse_grid(name):
    name=norm(name)
    if name=="C1": return (np.nan,np.nan,"core")
    pre,_,suf = name.partition(":")
    mn=re.search(r"C1[-.]?(\d)([AB])",pre)
    ms=re.search(r"C1[-.]([A-H])",suf)
    bay = int(mn.group(1)) if mn else np.nan
    sub = mn.group(2) if mn else ""
    line= (ord(ms.group(1))-64) if ms else np.nan      # A=1 ... H=8
    return (bay,line,sub)
g=m["col"].map(parse_grid)
m["bay"]=[x[0] for x in g]; m["line"]=[x[1] for x in g]; m["sub"]=[x[2] for x in g]
# grid_x combines bay and sub (B bay offset by 0.5), grid_y = line
m["gx"]=m["bay"]+m["sub"].map({"A":0.0,"B":0.5,"":0.0})
m["gy"]=m["line"]

# ---- face geometry from point id ----
def face(pid):
    p=re.match(r"[A-Za-z]+",str(pid)).group()
    return "single" if len(p)==1 else "pair"
m["face"]=m["point"].map(face)

m["priority"]=m["priority"].astype(int)
m["bad"]=(m["quality"]=="Doubtful").astype(int)

master_path=os.path.join(OUT,"master_dataset.csv")
m.to_csv(master_path,index=False,encoding="utf-8-sig")
print("MASTER:",m.shape,"| rebN coverage:",m["rebN"].notna().mean().round(3),
      "| mpa coverage:",m["mpa"].notna().mean().round(3))
RESULTS["dataset"]={"n_points":int(len(m)),"n_columns":int(m["col"].nunique()),
    "rebN_coverage":float(m["rebN"].notna().mean()),
    "vars":["vel","rebN","mpa","method","quality","priority","gx","gy","face"]}

# ----------------------------------------------------------------------
# helper stat functions (numpy)
# ----------------------------------------------------------------------
def pearson(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    ok=~(np.isnan(x)|np.isnan(y)); x,y=x[ok],y[ok]
    return float(np.corrcoef(x,y)[0,1]), x,y
def rank(a):
    a=np.asarray(a,float); order=a.argsort(); r=np.empty_like(order,float)
    r[order]=np.arange(len(a))
    # average ties
    _,inv,cnt=np.unique(a,return_inverse=True,return_counts=True)
    # simple tie handling via pandas
    return pd.Series(a).rank().values
def spearman(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    ok=~(np.isnan(x)|np.isnan(y)); x,y=x[ok],y[ok]
    return float(np.corrcoef(rank(x),rank(y))[0,1])
def perm_p_corr(x,y,r_obs,n=20000):
    x=np.asarray(x,float);y=np.asarray(y,float)
    cnt=0
    for _ in range(n):
        rp=np.corrcoef(x,RNG.permutation(y))[0,1]
        if abs(rp)>=abs(r_obs): cnt+=1
    return (cnt+1)/(n+1)
def boot_ci_corr(x,y,n=5000):
    x=np.asarray(x,float);y=np.asarray(y,float);N=len(x);out=[]
    for _ in range(n):
        idx=RNG.integers(0,N,N)
        out.append(np.corrcoef(x[idx],y[idx])[0,1])
    return float(np.percentile(out,2.5)),float(np.percentile(out,97.5))
def kruskal_H(groups):
    allv=np.concatenate(groups); r=pd.Series(allv).rank().values
    N=len(allv); idx=0; H=0
    for gp in groups:
        ng=len(gp); rg=r[idx:idx+ng]; idx+=ng
        H+=ng*(rg.mean()-(N+1)/2)**2
    H=12/(N*(N+1))*H
    return H
def perm_p_kruskal(groups,H_obs,n=20000):
    sizes=[len(g) for g in groups]; allv=np.concatenate(groups); cnt=0
    for _ in range(n):
        p=RNG.permutation(allv); gs=[];i=0
        for s in sizes: gs.append(p[i:i+s]); i+=s
        if kruskal_H(gs)>=H_obs: cnt+=1
    return (cnt+1)/(n+1)
def epsilon_sq(H,N): return float((H-1)/(N-1))*0 + float(H/( (N**2-1)/(N+1) ))  # H/(N-1)... use eps2=H/(N-1)
def eps2(H,N): return float(H/(N-1))
def chi2_stat(tab):
    tab=np.asarray(tab,float); rs=tab.sum(1,keepdims=True); cs=tab.sum(0,keepdims=True); tot=tab.sum()
    exp=rs@cs/tot; return float(np.nansum((tab-exp)**2/exp))
def cramers_v(tab):
    tab=np.asarray(tab,float); chi=chi2_stat(tab); n=tab.sum()
    k=min(tab.shape)-1; return float(np.sqrt(chi/(n*k))) if k>0 else np.nan
def perm_p_chi(cat1,cat2,chi_obs,n=20000):
    a=np.asarray(cat1);b=np.asarray(cat2);cnt=0
    for _ in range(n):
        tab=pd.crosstab(pd.Series(a),pd.Series(RNG.permutation(b))).values
        if chi2_stat(tab)>=chi_obs: cnt+=1
    return (cnt+1)/(n+1)

# ----------------------------------------------------------------------
# 2. DESCRIPTIVES
# ----------------------------------------------------------------------
desc=m[["vel","rebN","mpa"]].describe().T
desc["skew"]=[m[c].skew() for c in ["vel","rebN","mpa"]]
desc["kurt"]=[m[c].kurt() for c in ["vel","rebN","mpa"]]
desc["cov%"]=desc["std"]/desc["mean"]*100
desc.to_csv(os.path.join(OUT,"desc_overall.csv"))
RESULTS["descriptives"]=json.loads(desc.round(4).to_json(orient="index"))

by_pr=m.groupby("priority").agg(n=("vel","size"),vel_mean=("vel","mean"),vel_sd=("vel","std"),
      rebN_mean=("rebN","mean"),mpa_mean=("mpa","mean"),pct_bad=("bad","mean"))
by_pr["pct_bad"]*=100
by_pr.to_csv(os.path.join(OUT,"by_priority.csv"))
by_m=m.groupby("method").agg(n=("vel","size"),vel_mean=("vel","mean"),vel_sd=("vel","std"),
      vel_med=("vel","median"),pct_bad=("bad","mean"))
by_m["pct_bad"]*=100
by_m.to_csv(os.path.join(OUT,"by_method.csv"))
qdist=m["quality"].value_counts().reindex(qorder)
RESULTS["quality_distribution"]={k:int(v) for k,v in qdist.items()}
RESULTS["by_priority"]=json.loads(by_pr.round(3).to_json(orient="index"))
RESULTS["by_method"]=json.loads(by_m.round(3).to_json(orient="index"))
print("\nBY PRIORITY\n",by_pr.round(2))
print("\nBY METHOD\n",by_m.round(2))
print("\nQUALITY\n",qdist)

# ----------------------------------------------------------------------
# 3. CORRELATION UPV <-> STRENGTH (saturation test)
# ----------------------------------------------------------------------
r_vn,xn,yn=pearson(m["vel"],m["rebN"])
RESULTS["corr_vel_rebN"]={"pearson":round(r_vn,4),"spearman":round(spearman(m["vel"],m["rebN"]),4),
    "perm_p":round(perm_p_corr(xn,yn,r_vn,5000),5),"ci95":[round(c,4) for c in boot_ci_corr(xn,yn,2000)]}
r_vm,xm,ym=pearson(m["vel"],m["mpa"])
RESULTS["corr_vel_mpa"]={"pearson":round(r_vm,4),"spearman":round(spearman(m["vel"],m["mpa"]),4),
    "perm_p":round(perm_p_corr(xm,ym,r_vm,5000),5),"ci95":[round(c,4) for c in boot_ci_corr(xm,ym,2000)]}
# saturation evidence
RESULTS["rebound_saturation"]={"rebN_cov_pct":round(m["rebN"].std()/m["rebN"].mean()*100,3),
    "vel_cov_pct":round(m["vel"].std()/m["vel"].mean()*100,3),
    "rebN_min":float(m["rebN"].min()),"rebN_max":float(m["rebN"].max()),
    "note":"rebound numbers compressed near instrument ceiling N=55 (fc'=611 ksc); UPV carries the discriminating variance"}
print("\ncorr vel~rebN",RESULTS["corr_vel_rebN"])
print("corr vel~mpa",RESULTS["corr_vel_mpa"])

# ----------------------------------------------------------------------
# 4. GROUP DIFFERENCES
# ----------------------------------------------------------------------
def group_test(label,colname):
    groups=[m.loc[m[colname]==g,"vel"].dropna().values for g in sorted(m[colname].dropna().unique())]
    H=kruskal_H(groups); N=sum(len(g) for g in groups)
    p=perm_p_kruskal(groups,H,5000)
    res={"levels":[str(x) for x in sorted(m[colname].dropna().unique())],
         "group_n":[int(len(g)) for g in groups],
         "group_mean_vel":[round(float(g.mean()),3) for g in groups],
         "kruskal_H":round(H,3),"eps2":round(eps2(H,N),4),"perm_p":round(p,5)}
    RESULTS[f"veldiff_by_{label}"]=res
    print(f"\nvel by {label}: H={H:.2f} eps2={eps2(H,N):.3f} p={p:.4f}")
    return res
group_test("method","method")
group_test("priority","priority")
group_test("face","face")

# ----------------------------------------------------------------------
# 5. CONTINGENCY (quality x method, quality x priority, method x priority)
# ----------------------------------------------------------------------
def cont_test(a,b,na,nb):
    tab=pd.crosstab(m[a],m[b])
    chi=chi2_stat(tab.values); v=cramers_v(tab.values); p=perm_p_chi(m[a].values,m[b].values,chi,3000)
    RESULTS[f"chi2_{na}_x_{nb}"]={"chi2":round(chi,3),"cramers_v":round(v,4),"perm_p":round(p,5),
        "table":json.loads(tab.to_json())}
    print(f"\n{na} x {nb}: chi2={chi:.2f} V={v:.3f} p={p:.4f}")
cont_test("quality","method","quality","method")
cont_test("quality","priority","quality","priority")
cont_test("method","priority","method","priority")

# ----------------------------------------------------------------------
# 6. METHOD BIAS (within-column paired comparison)
# ----------------------------------------------------------------------
# For columns measured with >1 method, compare mean velocity by method controlling for column (column-centering)
m["vel_cc"]=m["vel"]-m.groupby("col")["vel"].transform("mean")
mb=m.groupby("method")["vel_cc"].agg(["mean","std","size"])
RESULTS["method_bias_columncentered"]=json.loads(mb.round(4).to_json(orient="index"))
print("\nmethod bias (column-centered vel)\n",mb.round(3))

# ----------------------------------------------------------------------
# 7. WITHIN-COLUMN HETEROGENEITY  (CoV as damage signature)
# ----------------------------------------------------------------------
col_stats=m.groupby("col").agg(n=("vel","size"),vel_mean=("vel","mean"),vel_cov=("vel",lambda s:s.std()/s.mean()*100),
    pct_bad=("bad","mean"),priority=("priority","first"),gx=("gx","first"),gy=("gy","first"),
    ksc=("ksc","first"))
col_stats["pct_bad"]*=100
col_stats=col_stats.sort_values("vel_mean")
col_stats.to_csv(os.path.join(OUT,"column_level_stats.csv"))
r_cov=np.corrcoef(col_stats["vel_mean"],col_stats["vel_cov"])[0,1]
RESULTS["heterogeneity"]={"corr_mean_vs_cov":round(float(r_cov),4),
    "interpretation":"negative => weaker columns are also more internally heterogeneous (localized damage)"}
print("\ncorr(col mean vel, col CoV):",round(r_cov,3))

# ----------------------------------------------------------------------
# 8. PCA (numpy SVD)
# ----------------------------------------------------------------------
feat_cols=["vel","rebN","gx","gy"]
F=m[feat_cols].dropna()
idx_keep=F.index
Fz=(F-F.mean())/F.std()
U,S,Vt=np.linalg.svd(Fz.values,full_matrices=False)
evr=(S**2)/np.sum(S**2)
RESULTS["pca"]={"explained_var_ratio":[round(float(x),4) for x in evr],
    "loadings_PC1":dict(zip(feat_cols,[round(float(x),3) for x in Vt[0]])),
    "loadings_PC2":dict(zip(feat_cols,[round(float(x),3) for x in Vt[1]]))}
scores=U*S
print("\nPCA EVR",evr.round(3))

# ----------------------------------------------------------------------
# 9. KMEANS (numpy) + silhouette
# ----------------------------------------------------------------------
def kmeans(X,k,iters=200,restarts=10):
    best=None
    for _ in range(restarts):
        c=X[RNG.choice(len(X),k,replace=False)]
        for _ in range(iters):
            d=((X[:,None,:]-c[None,:,:])**2).sum(2); lab=d.argmin(1)
            nc=np.array([X[lab==j].mean(0) if (lab==j).any() else c[j] for j in range(k)])
            if np.allclose(nc,c): c=nc; break
            c=nc
        inertia=((X-c[lab])**2).sum()
        if best is None or inertia<best[0]: best=(inertia,lab,c)
    return best
def silhouette(X,lab):
    # sample for speed
    n=len(X);
    D=np.sqrt(((X[:,None,:]-X[None,:,:])**2).sum(2))
    sl=[]
    for i in range(n):
        same=lab==lab[i]; same[i]=False
        if same.sum()==0: sl.append(0); continue
        a=D[i,same].mean()
        b=min(D[i,lab==j].mean() for j in set(lab) if j!=lab[i])
        sl.append((b-a)/max(a,b))
    return float(np.mean(sl))
Xc=Fz.values
sil={}
for k in range(2,7):
    inertia,lab,c=kmeans(Xc,k)
    sil[k]=round(silhouette(Xc,lab),4)
bestk=max(sil,key=sil.get)
inertia,lab,c=kmeans(Xc,bestk)
m_k=m.loc[idx_keep].copy(); m_k["cluster"]=lab
clus_prof=m_k.groupby("cluster").agg(n=("vel","size"),vel=("vel","mean"),rebN=("rebN","mean"),
    gx=("gx","mean"),gy=("gy","mean"),pct_bad=("bad","mean"))
clus_prof["pct_bad"]*=100
RESULTS["kmeans"]={"silhouette_by_k":sil,"chosen_k":int(bestk),
    "cluster_profile":json.loads(clus_prof.round(3).to_json(orient="index"))}
print("\nKMEANS silhouette",sil,"-> k=",bestk)
print(clus_prof.round(2))

# ----------------------------------------------------------------------
# 10. DBSCAN (numpy) for density anomalies
# ----------------------------------------------------------------------
def dbscan(X,eps,minpts):
    n=len(X); D=np.sqrt(((X[:,None,:]-X[None,:,:])**2).sum(2))
    lab=-np.ones(n,int); cid=0; visited=np.zeros(n,bool)
    for i in range(n):
        if visited[i]: continue
        visited[i]=True; nb=np.where(D[i]<=eps)[0]
        if len(nb)<minpts: continue
        cid+=1; lab[i]=cid; seeds=list(nb)
        k=0
        while k<len(seeds):
            j=seeds[k];k+=1
            if not visited[j]:
                visited[j]=True; nb2=np.where(D[j]<=eps)[0]
                if len(nb2)>=minpts: seeds+=list(nb2)
            if lab[j]==-1: lab[j]=cid
    return lab
dl=dbscan(Xc,eps=1.0,minpts=5)
RESULTS["dbscan"]={"eps":1.0,"minpts":5,"n_clusters":int(len(set(dl))-(1 if -1 in dl else 0)),
    "n_noise":int((dl==-1).sum())}
print("\nDBSCAN clusters",RESULTS["dbscan"])

# ----------------------------------------------------------------------
# 11. ANOMALY DETECTION (Mahalanobis on vel,rebN + kNN)
# ----------------------------------------------------------------------
A=m[["vel","rebN"]].dropna(); ia=A.index
mu=A.mean().values; cov=np.cov(A.values.T); inv=np.linalg.pinv(cov)
md=np.sqrt(((A.values-mu)@inv*(A.values-mu)).sum(1))
m_an=m.loc[ia].copy(); m_an["mahalanobis"]=md
# kNN distance
Az=(A-A.mean())/A.std()
Dk=np.sqrt(((Az.values[:,None,:]-Az.values[None,:,:])**2).sum(2))
knn=np.sort(Dk,1)[:,1:6].mean(1); m_an["knn_score"]=knn
thr=np.percentile(md,97.5)
anom=m_an[m_an["mahalanobis"]>=thr].sort_values("mahalanobis",ascending=False)
anom[["col","point","vel","rebN","method","quality","priority","mahalanobis"]].to_csv(
    os.path.join(OUT,"anomalies.csv"),index=False,encoding="utf-8-sig")
RESULTS["anomalies"]={"threshold_p97.5":round(float(thr),3),"n_flagged":int(len(anom)),
    "top":json.loads(anom.head(12)[["col","point","vel","rebN","method","quality","priority","mahalanobis"]].round(3).to_json(orient="records"))}
print("\nANOMALIES flagged:",len(anom))

# ----------------------------------------------------------------------
# 12. SUPERVISED ML: predict 'bad' (Doubtful) point from features (excl. velocity-threshold leakage)
#     features: spatial gx,gy, method(onehot), face, priority, neighbor mean velocity
# ----------------------------------------------------------------------
# neighbor feature: column mean velocity excluding the point
m["col_mean_excl"]=(m.groupby("col")["vel"].transform("sum")-m["vel"])/(m.groupby("col")["vel"].transform("size")-1)
sup=m.copy()
Xcat=pd.get_dummies(sup["method"],prefix="m")
Xface=pd.get_dummies(sup["face"],prefix="f")
Xsup=pd.concat([sup[["gx","gy","priority","col_mean_excl"]],Xcat,Xface],axis=1).fillna(sup[["gx","gy"]].mean().mean())
Xsup=Xsup.fillna(0.0)
ysup=sup["bad"].values.astype(float)
featnames=list(Xsup.columns)
Xv=Xsup.values.astype(float)
# standardize
Xs=(Xv-Xv.mean(0))/(Xv.std(0)+1e-9)

def logreg_train(X,y,l2=1.0,lr=0.1,epochs=4000):
    n,d=X.shape; w=np.zeros(d); b=0.0
    for _ in range(epochs):
        z=X@w+b; p=1/(1+np.exp(-z)); g=p-y
        w-=lr*(X.T@g/n+l2*w/n); b-=lr*g.mean()
    return w,b
def predict(X,w,b): return 1/(1+np.exp(-(X@w+b)))
def auc(y,s):
    pos=s[y==1]; neg=s[y==0]
    if len(pos)==0 or len(neg)==0: return np.nan
    c=0
    for p in pos: c+=(p>neg).sum()+0.5*(p==neg).sum()
    return c/(len(pos)*len(neg))
# 5-fold CV
idx=RNG.permutation(len(Xs)); folds=np.array_split(idx,5)
aucs=[];accs=[]
for f in range(5):
    te=folds[f]; tr=np.concatenate([folds[i] for i in range(5) if i!=f])
    w,b=logreg_train(Xs[tr],ysup[tr])
    s=predict(Xs[te],w,b); aucs.append(auc(ysup[te],s))
    accs.append(((s>=0.5).astype(int)==ysup[te]).mean())
# permutation importance (AUC drop)
w,b=logreg_train(Xs,ysup); base_auc=auc(ysup,predict(Xs,w,b))
imp={}
for j,fn in enumerate(featnames):
    Xp=Xs.copy(); Xp[:,j]=RNG.permutation(Xp[:,j])
    imp[fn]=round(base_auc-auc(ysup,predict(Xp,w,b)),4)
RESULTS["ml_predict_bad_logreg"]={"cv_auc_mean":round(float(np.nanmean(aucs)),4),
    "cv_auc_sd":round(float(np.nanstd(aucs)),4),"cv_acc_mean":round(float(np.mean(accs)),4),
    "base_rate_bad":round(float(ysup.mean()),4),
    "perm_importance":dict(sorted(imp.items(),key=lambda x:-x[1]))}
print("\nML predict-bad logreg CV-AUC",round(np.nanmean(aucs),3))

# Decision tree (CART, gini) small depth -> feature importance & interpretable rules
class Node: pass
def gini(y):
    if len(y)==0: return 0
    p=y.mean(); return 1-p**2-(1-p)**2
def best_split(X,y):
    n,d=X.shape; bg=gini(y); best=None
    for j in range(d):
        vals=np.unique(X[:,j])
        for t in vals:
            l=X[:,j]<=t; r=~l
            if l.sum()<10 or r.sum()<10: continue
            g=(l.sum()*gini(y[l])+r.sum()*gini(y[r]))/n
            if best is None or g<best[0]: best=(g,j,t)
    return best
def build(X,y,depth,maxd,imp):
    nd=Node(); nd.pred=y.mean()
    if depth>=maxd or len(y)<20 or y.std()==0: nd.leaf=True; return nd
    bs=best_split(X,y)
    if bs is None: nd.leaf=True; return nd
    g,j,t=bs; imp[j]+=(gini(y)-g)*len(y)
    nd.leaf=False; nd.j=j; nd.t=t
    l=X[:,j]<=t
    nd.left=build(X[l],y[l],depth+1,maxd,imp); nd.right=build(X[~l],y[~l],depth+1,maxd,imp)
    return nd
timp=np.zeros(len(featnames))
tree=build(Xv,ysup,0,3,timp)
timp=timp/timp.sum() if timp.sum()>0 else timp
RESULTS["ml_tree_importance"]=dict(sorted(zip(featnames,[round(float(x),4) for x in timp]),key=lambda x:-x[1]))
print("tree importance top:",sorted(zip(featnames,timp),key=lambda x:-x[1])[:5])

# ----------------------------------------------------------------------
# 13. SPATIAL AUTOCORRELATION (Moran's I) on column-mean velocity
# ----------------------------------------------------------------------
cs=col_stats.dropna(subset=["gx","gy"]).copy()
coords=cs[["gx","gy"]].values; v=cs["vel_mean"].values
n=len(v); D=np.sqrt(((coords[:,None,:]-coords[None,:,:])**2).sum(2))
W=(D<=1.5).astype(float); np.fill_diagonal(W,0)
z=v-v.mean(); S0=W.sum()
I=(n/S0)*(z@W@z)/(z@z)
# permutation null
Iperm=[]
for _ in range(5000):
    zp=RNG.permutation(z); Iperm.append((n/S0)*(zp@W@zp)/(zp@zp))
Iperm=np.array(Iperm); p_moran=( (np.abs(Iperm-Iperm.mean())>=abs(I-Iperm.mean())).sum()+1)/(5001)
RESULTS["spatial_moranI"]={"I":round(float(I),4),"E_I":round(-1/(n-1),4),"perm_p":round(float(p_moran),5),
    "n_columns":int(n),"interpretation":"I>0 => low/high velocity columns are spatially clustered on the P5 grid"}
print("\nMoran's I",round(I,3),"p",round(p_moran,4))

# ----------------------------------------------------------------------
# 14. COMPOSITE STRUCTURAL CONCERN INDEX (column ranking)
# ----------------------------------------------------------------------
cz=col_stats.copy()
cz["z_vel"]=-(cz["vel_mean"]-cz["vel_mean"].mean())/cz["vel_mean"].std()
cz["z_cov"]=(cz["vel_cov"]-cz["vel_cov"].mean())/cz["vel_cov"].std()
cz["z_bad"]=(cz["pct_bad"]-cz["pct_bad"].mean())/(cz["pct_bad"].std()+1e-9)
cz["concern_index"]=cz[["z_vel","z_cov","z_bad"]].mean(1)
cz=cz.sort_values("concern_index",ascending=False)
cz.to_csv(os.path.join(OUT,"concern_index.csv"),encoding="utf-8-sig")
RESULTS["concern_index_top5"]=json.loads(cz.head(5)[["vel_mean","vel_cov","pct_bad","priority","concern_index"]].round(3).to_json(orient="index"))
print("\nCONCERN INDEX top5\n",cz.head(5)[["vel_mean","vel_cov","pct_bad","concern_index"]].round(2))

# save results
with open(os.path.join(OUT,"results.json"),"w") as f:
    json.dump(RESULTS,f,ensure_ascii=False,indent=2)
print("\nSAVED results.json")
# stash objects for figure script
np.savez(os.path.join(OUT,"_state.npz"),scores=scores,lab=lab,
         coords=coords,vmean=v)
m.to_pickle(os.path.join(OUT,"_master.pkl"))
col_stats.to_pickle(os.path.join(OUT,"_colstats.pkl"))
cz.to_pickle(os.path.join(OUT,"_concern.pkl"))
print("DONE")
