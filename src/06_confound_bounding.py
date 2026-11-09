# -*- coding: utf-8 -*-
"""Bound the defect-proximity confound: re-estimate arrangement bias on 'sound' points.
Proxy for 'visually-sound concrete' = points NOT graded Doubtful (and a stricter Good/Excellent-only set).
Also: ambiguous-grade fraction at sigma_rep = 1/2/3%."""
import os, json
import numpy as np, pandas as pd
RNG=np.random.default_rng(31)
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
A=os.path.join(ROOT,"outputs")
m=pd.read_csv(os.path.join(A,"master_dataset.csv"))
morder=["Indirect","Semi-Indirect","Direct"]

def gap_and_bias(df):
    """Return direct-indirect mean gap and column-centred indirect deficit, with bootstrap CIs."""
    d=df.copy()
    d["vcc"]=d["vel"]-d.groupby("col")["vel"].transform("mean")
    gap=d.loc[d.method=="Direct","vel"].mean()-d.loc[d.method=="Indirect","vel"].mean()
    ind_cc=d.loc[d.method=="Indirect","vcc"].mean()
    # bootstrap
    gb=[]; ib=[]
    a=d.loc[d.method=="Direct","vel"].values; b=d.loc[d.method=="Indirect","vel"].values
    for _ in range(5000):
        gb.append(a[RNG.integers(0,len(a),len(a))].mean()-b[RNG.integers(0,len(b),len(b))].mean())
        s=d.sample(frac=1,replace=True,random_state=RNG.integers(1e9))
        scc=s["vel"]-s.groupby("col")["vel"].transform("mean")
        ib.append(scc[s.method=="Indirect"].mean())
    return (round(float(gap),3),[round(float(np.percentile(gb,2.5)),3),round(float(np.percentile(gb,97.5)),3)],
            round(float(ind_cc),3),[round(float(np.percentile(ib,2.5)),3),round(float(np.percentile(ib,97.5)),3)],
            int((d.method=="Direct").sum()),int((d.method=="Indirect").sum()))

R={}
g,gci,ic,ici,nd,ni=gap_and_bias(m)
R["full_sample"]={"n_direct":nd,"n_indirect":ni,"direct_indirect_gap":g,"gap_ci":gci,"indirect_cc_deficit":ic,"deficit_ci":ici}
sound=m[m["quality"]!="Doubtful"]
g,gci,ic,ici,nd,ni=gap_and_bias(sound)
R["sound_no_doubtful"]={"n_direct":nd,"n_indirect":ni,"direct_indirect_gap":g,"gap_ci":gci,"indirect_cc_deficit":ic,"deficit_ci":ici}
ge=m[m["quality"].isin(["Good","Excellent"])]
g,gci,ic,ici,nd,ni=gap_and_bias(ge)
R["sound_good_excellent_only"]={"n_direct":nd,"n_indirect":ni,"direct_indirect_gap":g,"gap_ci":gci,"indirect_cc_deficit":ic,"deficit_ci":ici}

# ----- ambiguous-grade fraction vs sigma_rep (probabilistic grade) -----
import json as _j
off=_j.load(open(os.path.join(A,"deep_results.json")))["correction"]["offset_kms"]
ci=_j.load(open(os.path.join(A,"deep_results.json")))["correction"]["indirect_offset_ci95"]
sig_delta={"Indirect":(ci[1]-ci[0])/(2*1.96),"Semi-Indirect":0.030,"Direct":0.0}
m["vel_corr"]=m["vel"]+m["method"].map(off)
def amb_fraction(sig_rep,M=8000):
    maxp=np.zeros(len(m))
    vc=m["vel_corr"].values; meth=m["method"].values
    for i in range(len(m)):
        s=np.sqrt((sig_rep*vc[i])**2+sig_delta[meth[i]]**2)
        samp=RNG.normal(vc[i],s,M)
        g=np.where(samp>4.5,3,np.where(samp>=3.5,2,np.where(samp>=3.0,1,0)))
        maxp[i]=np.bincount(g,minlength=4).max()/M
    return round(float((maxp<0.8).mean()*100),1)
R["ambiguous_pct_vs_sigma"]={"1pct":amb_fraction(0.01),"2pct":amb_fraction(0.02),"3pct":amb_fraction(0.03)}

json.dump(R,open(os.path.join(A,"confound_results.json"),"w"),indent=2)
print(json.dumps(R,indent=1))
