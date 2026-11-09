# -*- coding: utf-8 -*-
"""Revision analysis (Results in Engineering, round 1). See ANALYSIS_SPEC.md.
Run:  python3 src/revision_analysis.py     (about 1-2 minutes)
Writes outputs/revision_results.json, outputs/tables/*.tex, outputs/figures/*.png, outputs/SUMMARY.md
"""
import os, json, warnings, math, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats as sps
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.proportion import confint_proportions_2indep
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "master_dataset.csv"
WORK = ROOT / "outputs"; TAB = WORK / "tables"; FIG = WORK / "figures"
for d in (WORK, TAB, FIG): d.mkdir(parents=True, exist_ok=True)
SEED = 42
R = {}            # results
NOTES = {}
SUM = []          # summary lines (number, meaning)
def S(label, value, meaning=""):
    SUM.append((label, value, meaning))
def r3(x): return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)
def ci(a, lo=2.5, hi=97.5):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return [r3(np.percentile(a, lo)), r3(np.percentile(a, hi))]

m = pd.read_csv(DATA)
m["quality"] = pd.Categorical(m["quality"], ["Doubtful", "Medium", "Good", "Excellent"], ordered=True)
QORD = ["Doubtful", "Medium", "Good", "Excellent"]
MORD = ["Direct", "Semi-Indirect", "Indirect"]     # data label Semi-Indirect == "semi-direct" in the paper
ABBR = {"Direct": "D", "Semi-Indirect": "S", "Indirect": "I"}
N = len(m); y = m["vel"].values.astype(float)
m["key"] = m["col"].str.replace(" ", "") + "|" + m["point"].astype(str)
m["colid"] = pd.factorize(m["col"])[0]
COLS = list(m["col"].unique()); NC = len(COLS)
def grade(v):
    v = np.asarray(v, float)
    return np.where(v > 4.5, "Excellent", np.where(v >= 3.5, "Good", np.where(v >= 3.0, "Medium", "Doubtful")))
m["vel_cc"] = m["vel"] - m.groupby("col")["vel"].transform("mean")

# ======================================================================
# STEP 0  reproduce submitted numbers
# ======================================================================
def kruskalH_arr(v, g):
    r = sps.rankdata(v); n = len(v); H = 0.0
    for k in np.unique(g):
        s = g == k; H += s.sum() * (r[s].mean() - (n + 1) / 2) ** 2
    return 12 / (n * (n + 1)) * H
def chi2_stat(tab):
    tab = np.asarray(tab, float); e = tab.sum(1, keepdims=True) @ tab.sum(0, keepdims=True) / tab.sum()
    return float(np.sum((tab - e) ** 2 / e))
means = m.groupby("method")["vel"].mean()
ccm = m.groupby("method")["vel_cc"].mean()
off_exact = {k: ccm["Direct"] - ccm[k] for k in MORD}
DELTA = {"Direct": 0.0, "Semi-Indirect": -0.055, "Indirect": 0.237}      # submitted (rounded) offsets
tab34 = pd.crosstab(m["method"], m["quality"]).reindex(index=MORD, columns=QORD)
chi_34 = chi2_stat(tab34.values)
m["dbl"] = (m["quality"] == "Doubtful").astype(int)
t32 = pd.crosstab(m["method"], m["dbl"]).reindex(MORD)
chi_32 = chi2_stat(t32.values)
m["vel_c237"] = m["vel"] + m["method"].map(DELTA)
m["q_c237"] = grade(m["vel_c237"])
dbl0 = m["quality"] == "Doubtful"
resc = int((dbl0 & (m["q_c237"] != "Doubtful")).sum())

def ols_sse(X):
    b = np.linalg.lstsq(X, y, rcond=None)[0]; e = y - X @ b; return float(e @ e)
def dum(s): return pd.get_dummies(s, drop_first=True).astype(float).values
one = np.ones((N, 1)); SST = float(((y - y.mean()) ** 2).sum())
Xa, Xp, Xc = dum(m["method"]), dum(m["priority"].astype(str)), dum(m["col"])
def R2(*mats): return 1 - ols_sse(np.hstack([one] + list(mats))) / SST
R2s = {"arrangement": R2(Xa), "column": R2(Xc), "arrangement+column": R2(Xa, Xc),
       "priority": R2(Xp), "arrangement+priority": R2(Xa, Xp)}
step0 = {
 "mean_vel": {k: round(means[k], 3) for k in MORD},
 "gap_direct_minus_indirect": round(means["Direct"] - means["Indirect"], 3),
 "column_centred_means": {k: round(ccm[k], 3) for k in MORD},
 "delta_indirect_exact": round(off_exact["Indirect"], 4), "delta_semi_exact": round(off_exact["Semi-Indirect"], 4),
 "chi2_3x4_arrangement_by_quality": round(chi_34, 2), "chi2_3x2_arrangement_by_doubtful": round(chi_32, 2),
 "chi2_144.6_is": "3x4 arrangement x quality table" if abs(chi_34 - 144.6) < 0.1 else ("3x2 Doubtful-vs-other table" if abs(chi_32 - 144.6) < 0.1 else "neither"),
 "table_3x4": json.loads(tab34.to_json(orient="index")),
 "doubtful_direct": int(t32.loc["Direct", 1]), "doubtful_indirect": int(t32.loc["Indirect", 1]),
 "doubtful_semi": int(t32.loc["Semi-Indirect", 1]), "doubtful_total": int(dbl0.sum()),
 "reclassified_at_0.237_-0.055": resc,
 "R2": {k: round(v, 3) for k, v in R2s.items()},
}
chk = [("mean D", means["Direct"], 4.46, .005), ("mean S", means["Semi-Indirect"], 4.24, .005), ("mean I", means["Indirect"], 3.62, .005),
       ("gap", means["Direct"] - means["Indirect"], 0.85, .005),
       ("cc D", ccm["Direct"], .09, .005), ("cc S", ccm["Semi-Indirect"], .15, .005), ("cc I", ccm["Indirect"], -.14, .005),
       ("delta_ind", off_exact["Indirect"], .237, .0005), ("chi2", chi_34, 144.6, .05),
       ("dbl direct", t32.loc["Direct", 1], 2, 0), ("dbl indirect", t32.loc["Indirect", 1], 27, 0), ("dbl total", dbl0.sum(), 35, 0),
       ("reclassified", resc, 10, 0), ("R2 arr", R2s["arrangement"], .376, .0005), ("R2 col", R2s["column"], .344, .0005),
       ("R2 arr+col", R2s["arrangement+column"], .471, .0005), ("R2 prio", R2s["priority"], .094, .0005), ("R2 arr+prio", R2s["arrangement+priority"], .422, .0005)]
bad_chk = [(n, float(a), b) for n, a, b, t in chk if abs(a - b) > t + 1e-9]
step0["all_match"] = not bad_chk
R["step0_reproduction"] = step0
if bad_chk:
    raise SystemExit(f"STEP 0 FAILED, mismatches: {bad_chk}")
print("STEP 0 OK. chi2 144.6 is:", step0["chi2_144.6_is"])
S("Step0: mean vel D/S/I (km/s)", f"{means['Direct']:.2f}/{means['Semi-Indirect']:.2f}/{means['Indirect']:.2f}", "raw mean pulse velocity by arrangement")
S("Step0: chi2 = 144.6 is the", step0["chi2_144.6_is"] + f" (3x2 Doubtful-vs-other chi2 = {chi_32:.1f})", "which table the submitted chi-square belongs to")
S("Step0: delta_indirect exact / delta_semi exact", f"{off_exact['Indirect']:.4f} / {off_exact['Semi-Indirect']:.4f}", "column-centred offsets to direct datum")

# ======================================================================
# STEP 1  allocation table
# ======================================================================
cnt = pd.crosstab(m["col"], m["method"]).reindex(index=COLS, columns=MORD).fillna(0).astype(int)
info = m.groupby("col").agg(priority=("priority", "first"), nface=("face", "nunique"), face1=("face", "first")).reindex(COLS)
info["face"] = np.where(info["nface"] == 2, "both", info["face1"])
hasD, hasS, hasI = cnt["Direct"] > 0, cnt["Semi-Indirect"] > 0, cnt["Indirect"] > 0
alloc = {
 "n_columns": NC, "n_with_direct": int(hasD.sum()), "n_with_semi": int(hasS.sum()), "n_with_indirect": int(hasI.sum()),
 "n_with_direct_and_indirect": int((hasD & hasI).sum()), "n_direct_plus_semi_only": int((hasD & hasS & ~hasI).sum()),
 "n_indirect_plus_semi_only": int((hasI & hasS & ~hasD).sum()), "n_indirect_only": int((hasI & ~hasS & ~hasD).sum()),
 "n_direct_only": int((hasD & ~hasS & ~hasI).sum()), "n_semi_only": int((hasS & ~hasD & ~hasI).sum()),
 "n_with_exactly_two_arrangements": int(((hasD.astype(int) + hasS + hasI) == 2).sum()),
 "face_counts_by_column": info["face"].value_counts().to_dict(),
 "points_per_column_min_max": [int(cnt.sum(1).min()), int(cnt.sum(1).max())],
}
R["step1_allocation"] = alloc
def tx(s): return str(s).replace(" : ", ":").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")
rows = []
for c in COLS:
    rows.append(f"{tx(c)} & {info.loc[c,'priority']} & {info.loc[c,'face']} & {cnt.loc[c,'Direct']} & {cnt.loc[c,'Semi-Indirect']} & {cnt.loc[c,'Indirect']} & {cnt.loc[c].sum()} \\\\")
tot = cnt.sum()
(TAB / "allocation.tex").write_text(
 "\\begin{tabular}{lcccccc}\n\\toprule\nColumn & Priority & Face & Direct & Semi-direct & Indirect & Total \\\\\n\\midrule\n" +
 "\n".join(rows) + f"\n\\midrule\nAll columns & & & {tot['Direct']} & {tot['Semi-Indirect']} & {tot['Indirect']} & {tot.sum()} \\\\\n\\bottomrule\n\\end{{tabular}}\n")
S("Step1: columns with Direct / Semi / Indirect", f"{alloc['n_with_direct']} / {alloc['n_with_semi']} / {alloc['n_with_indirect']}", "of 26 columns")
S("Step1: columns with both Direct and Indirect", alloc["n_with_direct_and_indirect"], "expected 0")
S("Step1: Direct+Semi / Indirect+Semi / Indirect only", f"{alloc['n_direct_plus_semi_only']} / {alloc['n_indirect_plus_semi_only']} / {alloc['n_indirect_only']}", "column allocation classes")

# ======================================================================
# STEP 2  clustered inference
# ======================================================================
mm = smf.ols('vel ~ C(method, Treatment(reference="Direct")) + C(col)', m)
fe_t = mm.fit(cov_type="cluster", cov_kwds={"groups": m["colid"].values}, use_t=True)
fe_z = mm.fit(cov_type="cluster", cov_kwds={"groups": m["colid"].values}, use_t=False)
fe_h = mm.fit()
def coef(res, name): return float(res.params[name])
nI = 'C(method, Treatment(reference="Direct"))[T.Indirect]'; nS = 'C(method, Treatment(reference="Direct"))[T.Semi-Indirect]'
def fe_row(res):
    c = res.conf_int(0.05)
    return {k: {"coef": r3(res.params[n]), "se": r3(res.bse[n]), "ci95": [r3(c.loc[n, 0]), r3(c.loc[n, 1])]} for k, n in (("Indirect_vs_Direct", nI), ("Semi_vs_Direct", nS))}
FE = {"cluster_robust_t_df25": fe_row(fe_t), "cluster_robust_normal": fe_row(fe_z), "iid_ols_se": fe_row(fe_h)}
# Indirect vs Semi contrast
cI, cS = fe_t.params[nI], fe_t.params[nS]
Lc = np.zeros(len(fe_t.params)); Lc[list(fe_t.params.index).index(nI)] = 1; Lc[list(fe_t.params.index).index(nS)] = -1
tt = fe_t.t_test(Lc); FE["cluster_robust_t_df25"]["Indirect_vs_Semi"] = {"coef": r3(tt.effect[0]), "se": r3(tt.sd[0][0]), "ci95": [r3(tt.conf_int()[0][0]), r3(tt.conf_int()[0][1])]}
FE["n_clusters"] = NC; FE["r2"] = r3(fe_t.rsquared)
# FE-derived corrections: add -coef to reach direct datum
FE_DELTA = {"Direct": 0.0, "Semi-Indirect": -float(cS), "Indirect": -float(cI)}
FE["delta_to_direct_datum"] = {k: r3(v) for k, v in FE_DELTA.items()}
mix = smf.mixedlm('vel ~ C(method, Treatment(reference="Direct"))', m, groups=m["col"]).fit(reml=True)
cm_ = mix.conf_int(0.05)
MX = {k: {"coef": r3(mix.params[n]), "se": r3(mix.bse[n]), "ci95": [r3(cm_.loc[n, 0]), r3(cm_.loc[n, 1])]} for k, n in (("Indirect_vs_Direct", nI), ("Semi_vs_Direct", nS))}
MX["group_variance"] = r3(float(mix.cov_re.iloc[0, 0])); MX["residual_variance"] = r3(float(mix.scale))
MX["icc_conditional_on_arrangement"] = r3(float(mix.cov_re.iloc[0, 0]) / (float(mix.cov_re.iloc[0, 0]) + float(mix.scale)))
MX["intercept_direct"] = r3(mix.params["Intercept"]); MX["converged"] = bool(mix.converged)
R["step2a_column_fixed_effects"] = FE; R["step2a_mixed_model"] = MX
print("FE:", FE["cluster_robust_t_df25"]); print("MIX:", MX)
S("Step2a: FE Indirect vs Direct (km/s), cluster-robust t CI", f"{FE['cluster_robust_t_df25']['Indirect_vs_Direct']['coef']} {FE['cluster_robust_t_df25']['Indirect_vs_Direct']['ci95']}", "within-column additive estimate; negative = indirect reads lower")
S("Step2a: FE Semi vs Direct", f"{FE['cluster_robust_t_df25']['Semi_vs_Direct']['coef']} {FE['cluster_robust_t_df25']['Semi_vs_Direct']['ci95']}", "")
S("Step2a: FE Indirect vs Direct, normal CI", f"{FE['cluster_robust_normal']['Indirect_vs_Direct']['ci95']}", "z-based cluster-robust CI")
S("Step2a: Mixed Indirect vs Direct", f"{MX['Indirect_vs_Direct']['coef']} {MX['Indirect_vs_Direct']['ci95']}", "random intercept by column, REML")
S("Step2a: Mixed Semi vs Direct", f"{MX['Semi_vs_Direct']['coef']} {MX['Semi_vs_Direct']['ci95']}", "")
S("Step2a: Mixed group variance / residual variance", f"{MX['group_variance']} / {MX['residual_variance']}", "REML variance components")

# 2b within-column contrasts
cmean = m.pivot_table(index="col", columns="method", values="vel", aggfunc="mean").reindex(COLS)[MORD]
d_sd = (cmean["Semi-Indirect"] - cmean["Direct"]).values        # S-D (nan if absent)
d_si = (cmean["Semi-Indirect"] - cmean["Indirect"]).values      # S-I
d_ds = -d_sd                                                    # D-S
def within_stats(ix):
    a, b, c = d_sd[ix], d_si[ix], d_ds[ix]
    sd, si = np.nanmean(a), np.nanmean(b)
    return sd, si, np.nanmean(c) + si
sd0, si0, ch0 = within_stats(np.arange(NC))
rng = np.random.default_rng(SEED)
bw = np.array([within_stats(rng.integers(0, NC, NC)) for _ in range(10000)])
W = {"n_columns_semi_and_direct": int(np.isfinite(d_sd).sum()), "n_columns_semi_and_indirect": int(np.isfinite(d_si).sum()),
     "mean_semi_minus_direct": {"est": r3(sd0), "ci95": ci(bw[:, 0])},
     "mean_semi_minus_indirect": {"est": r3(si0), "ci95": ci(bw[:, 1])},
     "chained_direct_equivalent_indirect_offset": {"est": r3(ch0), "ci95": ci(bw[:, 2]),
        "note": "(S-I) + (D-S) using column-level means; equals the amount to ADD to indirect readings"}}
R["step2b_within_column_contrasts"] = W
S("Step2b: mean(semi-direct) over 12 columns [boot CI]", f"{W['mean_semi_minus_direct']['est']} {W['mean_semi_minus_direct']['ci95']}", "within-column")
S("Step2b: mean(semi-indirect) over 12 columns [boot CI]", f"{W['mean_semi_minus_indirect']['est']} {W['mean_semi_minus_indirect']['ci95']}", "within-column")
S("Step2b: chained indirect offset [boot CI]", f"{W['chained_direct_equivalent_indirect_offset']['est']} {W['chained_direct_equivalent_indirect_offset']['ci95']}", "(S-I)+(D-S)")

# 2c cluster bootstrap
meth_code = m["method"].map({k: i for i, k in enumerate(MORD)}).values
vcc = m["vel_cc"].values; bad = m["dbl"].values
col_rows = [np.where(m["colid"].values == c)[0] for c in range(NC)]
def boot_stats(idx):
    v, g, b, cc = y[idx], meth_code[idx], bad[idx], vcc[idx]
    nD, nS_, nI_ = [(g == k).sum() for k in range(3)]
    out = {}
    if nD == 0 or nI_ == 0 or nS_ == 0: return None
    mD, mS, mI = v[g == 0].mean(), v[g == 1].mean(), v[g == 2].mean()
    out["raw_gap"] = mD - mI
    r = sps.rankdata(v); n = len(v)
    H = sum((g == k).sum() * (r[g == k].mean() - (n + 1) / 2) ** 2 for k in range(3)) * 12 / (n * (n + 1))
    out["eps2"] = H / (n - 1)
    bD, bI = b[g == 0].sum(), b[g == 2].sum()
    out["rr_haldane"] = ((bI + .5) / (nI_ + .5)) / ((bD + .5) / (nD + .5))
    out["rr_raw"] = (bI / nI_) / (bD / nD) if bD > 0 else np.inf
    out["bad_direct_zero"] = float(bD == 0)
    out["cc_D"], out["cc_S"], out["cc_I"] = cc[g == 0].mean(), cc[g == 1].mean(), cc[g == 2].mean()
    out["delta_ind"] = out["cc_D"] - out["cc_I"]; out["delta_semi"] = out["cc_D"] - out["cc_S"]
    return out
def run_boot(kind, B):
    rg = np.random.default_rng(SEED); res = []
    for _ in range(B):
        if kind == "cluster":
            idx = np.concatenate([col_rows[c] for c in rg.integers(0, NC, NC)])
        else:
            idx = rg.integers(0, N, N)
        s = boot_stats(idx)
        if s: res.append(s)
    return pd.DataFrame(res)
bc = run_boot("cluster", 10000); bp = run_boot("point", 2000)
obs = boot_stats(np.arange(N))
def summ(df, k, finite=False):
    a = df[k].values; a = a[np.isfinite(a)] if finite else a
    return ci(a)
C2 = {}
for k, lab in (("raw_gap", "raw_direct_minus_indirect"), ("eps2", "epsilon_squared"), ("rr_haldane", "RR_doubtful_indirect_vs_direct_haldane"),
               ("cc_D", "colcentred_Direct"), ("cc_S", "colcentred_Semi"), ("cc_I", "colcentred_Indirect"), ("delta_ind", "delta_indirect_colcentred"), ("delta_semi", "delta_semi_colcentred")):
    C2[lab] = {"estimate": r3(obs[k]), "cluster_boot_ci95": summ(bc, k), "point_boot_ci95": summ(bp, k)}
    w_c = np.diff(summ(bc, k))[0]; w_p = np.diff(summ(bp, k))[0]
    C2[lab]["ci_width_ratio_cluster_over_point"] = r3(w_c / w_p) if w_p else None
C2["RR_doubtful_indirect_vs_direct_raw_finite_only"] = {"estimate": r3(obs["rr_raw"]), "cluster_boot_ci95": summ(bc, "rr_raw", True), "point_boot_ci95": summ(bp, "rr_raw", True),
    "frac_reps_with_zero_direct_doubtful_cluster": r3(bc["bad_direct_zero"].mean()), "frac_reps_with_zero_direct_doubtful_point": r3(bp["bad_direct_zero"].mean())}
C2["reps"] = {"cluster": len(bc), "point": len(bp)}
R["step2c_bootstrap"] = C2
for lab in ("raw_direct_minus_indirect", "epsilon_squared", "RR_doubtful_indirect_vs_direct_haldane", "colcentred_Direct", "colcentred_Semi", "colcentred_Indirect", "delta_indirect_colcentred"):
    S(f"Step2c: {lab}", f"{C2[lab]['estimate']}; cluster CI {C2[lab]['cluster_boot_ci95']}; point CI {C2[lab]['point_boot_ci95']}", "10000 column-resamples vs 2000 point-resamples")
NOTES["step2d"] = ("No column was measured with both direct and indirect arrangements (0 of 26). The column-centred indirect mean is computed against a column mean that already "
 "contains the arrangement mix of that column (Indirect points sit with Semi-direct points or alone; Direct points sit with Semi-direct points), so the column-centred estimate (delta_indirect = 0.237) "
 "is attenuated relative to the within-column contrast. The column fixed-effects estimate uses only within-column differences and is identified through the semi-direct arrangement as a bridge "
 "(additivity assumption: the arrangement offset is the same in every column): indirect vs semi within 12 columns, semi vs direct within 12 other columns, chained.")
R["step2d_notes"] = NOTES["step2d"]

# ======================================================================
# STEP 3  variance decomposition
# ======================================================================
two = m[m["col"].isin(cnt.index[(hasD.astype(int) + hasS + hasI) == 2])].copy()
def R2sub(df, f):
    r = smf.ols(f, df).fit(); return r.rsquared
r2_col_two = R2sub(two, "vel ~ C(col)"); r2_arr_two = R2sub(two, "vel ~ C(method)"); r2_both_two = R2sub(two, "vel ~ C(method) + C(col)")
ftest = sm.stats.anova_lm(smf.ols("vel ~ C(col)", two).fit(), smf.ols("vel ~ C(method) + C(col)", two).fit())
ssb = sum(len(g) * (g.mean() - y.mean()) ** 2 for _, g in m.groupby("col")["vel"])
V = {"R2_column_only": r3(R2s["column"]), "R2_arrangement_only": r3(R2s["arrangement"]), "R2_arrangement_plus_column": r3(R2s["arrangement+column"]),
     "increment_arrangement_beyond_column": r3(R2s["arrangement+column"] - R2s["column"]), "increment_column_beyond_arrangement": r3(R2s["arrangement+column"] - R2s["arrangement"]),
     "ICC_between_column_SS_share": r3(ssb / SST),
     "shared_variance_arrangement_and_column": r3(R2s["arrangement"] + R2s["column"] - R2s["arrangement+column"]),
     "partial_eta2_arrangement_given_column": r3((1 - R2s["column"]) and (R2s["arrangement+column"] - R2s["column"]) / (1 - R2s["column"])),
     "two_arrangement_columns_only": {"n_columns": int(two["col"].nunique()), "n_points": len(two), "R2_column": r3(r2_col_two), "R2_arrangement": r3(r2_arr_two),
        "R2_arr_plus_col": r3(r2_both_two), "increment_arrangement_beyond_column": r3(r2_both_two - r2_col_two),
        "partial_eta2_arrangement_given_column": r3((r2_both_two - r2_col_two) / (1 - r2_col_two)),
        "F_arrangement_given_column": r3(ftest["F"].iloc[1]), "p": float(f"{ftest['Pr(>F)'].iloc[1]:.3g}")}}
R["step3_variance"] = V
S("Step3: R2 column / arrangement / both", f"{V['R2_column_only']} / {V['R2_arrangement_only']} / {V['R2_arrangement_plus_column']}", "")
S("Step3: increment of arrangement beyond column / column beyond arrangement", f"{V['increment_arrangement_beyond_column']} / {V['increment_column_beyond_arrangement']}", "")
S("Step3: ICC (between-column SS share)", V["ICC_between_column_SS_share"], "")
S("Step3: within two-arrangement columns, arrangement increment beyond column", f"{V['two_arrangement_columns_only']['increment_arrangement_beyond_column']} (partial eta2 {V['two_arrangement_columns_only']['partial_eta2_arrangement_given_column']}, F={V['two_arrangement_columns_only']['F_arrangement_given_column']})", f"{two['col'].nunique()} columns")

# ======================================================================
# STEP 4  contingency and risk
# ======================================================================
def counts(df, col="quality_flag"):
    pass
def dbl_by_method(q):
    q = pd.Series(q, index=m.index); return {k: int(((m["method"] == k) & (q == "Doubtful")).sum()) for k in MORD}
nM = {k: int((m["method"] == k).sum()) for k in MORD}
dI, dD = int(t32.loc["Indirect", 1]), int(t32.loc["Direct", 1])
tab22 = np.array([[dI, nM["Indirect"] - dI], [dD, nM["Direct"] - dD]])
T = sm.stats.Table2x2(tab22)
rr = (dI / nM["Indirect"]) / (dD / nM["Direct"])
lo_w, hi_w = T.riskratio_confint()
sc = confint_proportions_2indep(dI, nM["Indirect"], dD, nM["Direct"], method="score", compare="ratio")
orr = sps.contingency.odds_ratio(tab22, kind="conditional"); orci = orr.confidence_interval(0.95)
fis = sps.fisher_exact(tab22)
RR4 = {"counts": {"indirect_doubtful": dI, "indirect_n": nM["Indirect"], "direct_doubtful": dD, "direct_n": nM["Direct"]},
       "risk_indirect": r3(dI / nM["Indirect"]), "risk_direct": r3(dD / nM["Direct"]), "RR": r3(rr),
       "RR_wald_log_ci95": [r3(lo_w), r3(hi_w)], "RR_score_ci95_farrington_manning": [r3(sc[0]), r3(sc[1])],
       "OR_conditional_exact": r3(orr.statistic), "OR_exact_ci95": [r3(orci.low), r3(orci.high)], "fisher_exact_p": float(f"{fis[1]:.3g}"),
       "RR_cluster_bootstrap_haldane": C2["RR_doubtful_indirect_vs_direct_haldane"]["cluster_boot_ci95"],
       "RR_point_bootstrap_haldane": C2["RR_doubtful_indirect_vs_direct_haldane"]["point_boot_ci95"]}
resid = {}
for name, dl in (("delta_0.237_-0.055", DELTA), ("FE", FE_DELTA)):
    q = grade(y + m["method"].map(dl).values); d = dbl_by_method(q)
    rrr = (d["Indirect"] / nM["Indirect"]) / (d["Direct"] / nM["Direct"]) if d["Direct"] else np.nan
    resid[name] = {"delta": {k: r3(v) for k, v in dl.items()}, "doubtful_by_arrangement": d, "residual_RR_indirect_vs_direct": r3(rrr),
                   "residual_RR_indirect_vs_semi": r3((d["Indirect"] / nM["Indirect"]) / (d["Semi-Indirect"] / nM["Semi-Indirect"])) if d["Semi-Indirect"] else None}
RR4["residual_after_correction"] = resid
R["step4_risk"] = RR4
S("Step4: RR Doubtful indirect vs direct", f"{RR4['RR']}; Wald {RR4['RR_wald_log_ci95']}; score {RR4['RR_score_ci95_farrington_manning']}; cluster boot (Haldane) {RR4['RR_cluster_bootstrap_haldane']}", f"{dI}/{nM['Indirect']} vs {dD}/{nM['Direct']}")
S("Step4: Fisher exact p / exact OR CI", f"{RR4['fisher_exact_p']} / OR {RR4['OR_conditional_exact']} {RR4['OR_exact_ci95']}", "")
S("Step4: residual RR after correction (0.237)", f"{resid['delta_0.237_-0.055']['residual_RR_indirect_vs_direct']} (Doubtful {resid['delta_0.237_-0.055']['doubtful_by_arrangement']})", "")
S("Step4: residual RR after FE correction", f"{resid['FE']['residual_RR_indirect_vs_direct']} (Doubtful {resid['FE']['doubtful_by_arrangement']})", "FE delta_ind=+0.85 lifts every indirect point above 3.0, so 0 Indirect Doubtful remain (RR 0)")

# ======================================================================
# STEP 5  reclassification sensitivity
# ======================================================================
SCEN = {"0.16": {"Direct": 0, "Semi-Indirect": -0.055, "Indirect": 0.16},
        "0.237": DELTA,
        "0.30": {"Direct": 0, "Semi-Indirect": -0.055, "Indirect": 0.30},
        "FE": FE_DELTA}
sens = {}
for k, dl in SCEN.items():
    vc = y + m["method"].map(dl).values; q = grade(vc)
    before = pd.crosstab(m["method"], m["quality"]).reindex(index=MORD, columns=QORD)
    after = pd.crosstab(m["method"], pd.Categorical(q, QORD)).reindex(index=MORD, columns=QORD).fillna(0).astype(int)
    q0 = m["quality"].astype(str).values
    resc_k = int(((q0 == "Doubtful") & (q != "Doubtful")).sum())
    newd = int(((q0 != "Doubtful") & (q == "Doubtful")).sum())
    trans = {}
    for a, b in itertools.product(QORD, QORD):
        n_ = int(((q0 == a) & (q == b)).sum())
        if a != b and n_: trans[f"{a}->{b}"] = {k2: int(((q0 == a) & (q == b) & (m['method'] == k2).values).sum()) for k2 in MORD}
    sens[k] = {"delta": {a: r3(b) for a, b in dl.items()}, "doubtful_before": dbl_by_method(q0), "doubtful_after": dbl_by_method(q),
               "n_doubtful_before": int((q0 == "Doubtful").sum()), "n_doubtful_after": int((q == "Doubtful").sum()),
               "reclassified_out_of_doubtful": resc_k, "newly_doubtful": newd, "n_changed_class_any": int((q0 != q).sum()),
               "class_counts_before": json.loads(before.to_json(orient="index")), "class_counts_after": json.loads(after.to_json(orient="index")),
               "excellent_before_total": int((q0 == "Excellent").sum()), "excellent_after_total": int((q == "Excellent").sum()),
               "transitions_by_arrangement": trans}
exc = sens["0.237"]["transitions_by_arrangement"]
NOTES["step5_excellent"] = ("Excellent total changes at delta=+0.237/-0.055 because (i) semi-direct points are shifted down by 0.055 km/s and those just above 4.5 km/s fall to Good "
    f"(Excellent->Good by arrangement: {exc.get('Excellent->Good')}), and (ii) indirect points are shifted up by 0.237 and those between 4.263 and 4.5 km/s rise to Excellent "
    f"(Good->Excellent by arrangement: {exc.get('Good->Excellent')}). Direct points are unchanged.")
R["step5_reclassification_sensitivity"] = sens; _mis = m[(grade(y) != m["quality"].astype(str).values)]
NOTES["step5_data_quirk"] = "Stored quality differs from the threshold rule applied to raw velocity for %d point(s): %s (so the Direct Good->Excellent transition above is this record, not a shift)." % (len(_mis), "; ".join(f"{r['col']} {r['point']} v={r['vel']} stored {r['quality']}" for _, r in _mis.iterrows()))
R["step5_note"] = NOTES["step5_excellent"] + " " + NOTES["step5_data_quirk"]
for k in SCEN:
    s = sens[k]; S(f"Step5: delta_ind={k}", f"Doubtful {s['n_doubtful_before']}->{s['n_doubtful_after']} (reclassified {s['reclassified_out_of_doubtful']}, newly Doubtful {s['newly_doubtful']}); by arr after {s['doubtful_after']}; Excellent {s['excellent_before_total']}->{s['excellent_after_total']}",
                    f"delta {s['delta']}")
S("Step5: Excellent explanation", NOTES["step5_excellent"] + " " + NOTES["step5_data_quirk"], "")
# reclass_by_arrangement.tex (delta 0.237)
b_ = sens["0.237"]["class_counts_before"]; a_ = sens["0.237"]["class_counts_after"]
rows = []
for k in MORD:
    lab = {"Direct": "Direct", "Semi-Indirect": "Semi-direct", "Indirect": "Indirect"}[k]
    rows.append(f"{lab} & " + " & ".join(f"{b_[k][q]}/{a_[k][q]}" for q in QORD) + f" & {nM[k]} \\\\")
tb = {q: sum(b_[k][q] for k in MORD) for q in QORD}; ta = {q: sum(a_[k][q] for k in MORD) for q in QORD}
(TAB / "reclass_by_arrangement.tex").write_text(
 "\\begin{tabular}{lccccc}\n\\toprule\nArrangement & Doubtful & Medium & Good & Excellent & $n$ \\\\\n\\midrule\n" + "\n".join(rows) +
 "\n\\midrule\nAll & " + " & ".join(f"{tb[q]}/{ta[q]}" for q in QORD) + f" & {N} \\\\\n\\bottomrule\n\\end{{tabular}}\n")

# ======================================================================
# STEP 6  decision rules
# ======================================================================
def rules(dl):
    q = pd.Series(grade(y + m["method"].map(dl).values), index=m.index); q0 = m["quality"].astype(str)
    out = {}
    for nm, flag_fn in (("A_doubtful", lambda s: s == "Doubtful"), ("B_doubtful_or_medium", lambda s: s.isin(["Doubtful", "Medium"]))):
        fb, fa = flag_fn(q0), flag_fn(q)
        out[nm] = {"points_before": int(fb.sum()), "points_after": int(fa.sum()),
                   "columns_before": int(m.groupby("col").apply(lambda d: fb[d.index].any()).sum()), "columns_after": int(m.groupby("col").apply(lambda d: fa[d.index].any()).sum())}
    for thr in (0.10, 0.20):
        pb = (q0 == "Doubtful").groupby(m["col"]).mean(); pa = (q == "Doubtful").groupby(m["col"]).mean()
        cb, ca = set(pb.index[pb >= thr - 1e-12]), set(pa.index[pa >= thr - 1e-12])
        out[f"C_column_ge_{int(thr*100)}pct_doubtful"] = {"columns_before": len(cb), "columns_after": len(ca),
            "points_in_flagged_columns_before": int(m["col"].isin(cb).sum()), "points_in_flagged_columns_after": int(m["col"].isin(ca).sum()),
            "doubtful_points_before": int((q0 == "Doubtful").sum()), "doubtful_points_after": int((q == "Doubtful").sum())}
    return out
R["step6_rules"] = {"delta_0.237": rules(DELTA), "FE": rules(FE_DELTA)}
for k, v in R["step6_rules"].items():
    for rn, d in v.items(): S(f"Step6 [{k}] {rn}", json.dumps(d), "before -> after")

# ======================================================================
# STEP 7  subgroups
# ======================================================================
sub = {}
for f in ("pair", "single"):
    d = m[m["face"] == f]; g = d.groupby("method")["vel"].agg(["mean", "size"])
    ccf = d.groupby("method")["vel_cc"].mean()
    sub[f] = {"n": len(d), "mean_by_arr": {k: r3(g.loc[k, "mean"]) if k in g.index else None for k in MORD}, "n_by_arr": {k: int(g.loc[k, "size"]) if k in g.index else 0 for k in MORD},
              "raw_direct_minus_indirect": r3(g.loc["Direct", "mean"] - g.loc["Indirect", "mean"]) if {"Direct", "Indirect"} <= set(g.index) else None,
              "raw_direct_minus_semi": r3(g.loc["Direct", "mean"] - g.loc["Semi-Indirect", "mean"]) if {"Direct", "Semi-Indirect"} <= set(g.index) else None,
              "raw_semi_minus_indirect": r3(g.loc["Semi-Indirect", "mean"] - g.loc["Indirect", "mean"]) if {"Semi-Indirect", "Indirect"} <= set(g.index) else None,
              "colcentred_direct_minus_indirect": r3(ccf.get("Direct", np.nan) - ccf.get("Indirect", np.nan))}
# arrangement within each face type: FE contrasts
for f in ("pair", "single"):
    d = m[m["face"] == f]
    try:
        r_ = smf.ols('vel ~ C(method, Treatment(reference="Direct")) + C(col)', d).fit(cov_type="cluster", cov_kwds={"groups": d["colid"].values})
        sub[f]["FE_indirect_vs_direct"] = r3(r_.params[nI]) if nI in r_.params else None; sub[f]["FE_semi_vs_direct"] = r3(r_.params[nS]) if nS in r_.params else None
    except Exception as e: sub[f]["FE_error"] = str(e)
sub["face_composition_by_column"] = info["face"].value_counts().to_dict()
sub["note_face_confounding"] = (f"Face type is confounded with arrangement: all {int((m['face']=='single').sum())} single-face points are Indirect (columns: {int(m.loc[m['face']=='single','col'].nunique())}); pair-face points are Direct {sub['pair']['n_by_arr']['Direct']}, Semi {sub['pair']['n_by_arr']['Semi-Indirect']}, Indirect {sub['pair']['n_by_arr']['Indirect']}. "
    "An arrangement contrast within single-face points is impossible; within pair-face the Indirect arm has only 18 points (columns: %d), so the pair-face Indirect-vs-Direct contrast is not reliable." % int(m.loc[(m['face']=='pair')&(m['method']=='Indirect'),'col'].nunique()))
sub["note_cross_section"] = "Cross-section type (e.g. square/rectangular/circular, size) is not recorded in master_dataset.csv, so arrangement contrasts cannot be stratified by cross-section."
vc237 = y + m["method"].map(DELTA).values; vcfe = y + m["method"].map(FE_DELTA).values
def rr_(a, b): return r3(np.corrcoef(a, b)[0, 1])
reb = m["rebN"].values
corr7 = {"r_vel_rebN_overall": rr_(y, reb), "r_by_arrangement": {k: rr_(y[m["method"] == k], reb[m["method"] == k]) for k in MORD},
         "r_corrected_0.237": rr_(vc237, reb), "r_corrected_FE": rr_(vcfe, reb),
         "r_corrected_by_arr_0.237": {k: rr_(vc237[m["method"] == k], reb[m["method"] == k]) for k in MORD}}
sub["rebound_correlation"] = corr7
def colcov(v):
    s = pd.Series(v).groupby(m["col"].values); return (s.std() / s.mean() * 100), s.mean()
cv_raw, mu_raw = colcov(y); cv_237, mu_237 = colcov(vc237); cv_fe, mu_fe = colcov(vcfe)
sub["column_CoV"] = {"mean_CoV_raw": r3(cv_raw.mean()), "mean_CoV_0.237": r3(cv_237.mean()), "mean_CoV_FE": r3(cv_fe.mean()),
    "corr_CoV_vs_colmean_raw": rr_(cv_raw.values, mu_raw.values), "corr_CoV_vs_colmean_0.237": rr_(cv_237.values, mu_237.values), "corr_CoV_vs_colmean_FE": rr_(cv_fe.values, mu_fe.values)}
R["step7_subgroups"] = sub
S("Step7: face pair/single raw D-I", f"{sub['pair']['raw_direct_minus_indirect']} / {sub['single']['raw_direct_minus_indirect']}", f"n pair {sub['pair']['n']}, single {sub['single']['n']}")
S("Step7: face confounding", sub["note_face_confounding"], "")
S("Step7: r(vel,rebN) overall / by arrangement (D,S,I)", f"{corr7['r_vel_rebN_overall']} / {list(corr7['r_by_arrangement'].values())}", "")
S("Step7: r(corrected vel,rebN) 0.237 / FE", f"{corr7['r_corrected_0.237']} / {corr7['r_corrected_FE']}", "")
S("Step7: corr(CoV, column mean) raw / 0.237 / FE", f"{sub['column_CoV']['corr_CoV_vs_colmean_raw']} / {sub['column_CoV']['corr_CoV_vs_colmean_0.237']} / {sub['column_CoV']['corr_CoV_vs_colmean_FE']}", "")

# ======================================================================
# STEP 8  triage on corrected velocities
# ======================================================================
def mahal(v, rebN):
    A = np.c_[v, rebN]; mu = A.mean(0); inv = np.linalg.pinv(np.cov(A.T)); d = A - mu
    md = np.sqrt(((d @ inv) * d).sum(1)); return md, np.percentile(md, 97.5)
def dbscan(X, eps=1.0, minpts=5):          # verbatim logic of upv_analysis.py
    n = len(X); D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(2))
    lab = -np.ones(n, int); cid = 0; visited = np.zeros(n, bool)
    for i in range(n):
        if visited[i]: continue
        visited[i] = True; nb = np.where(D[i] <= eps)[0]
        if len(nb) < minpts: continue
        cid += 1; lab[i] = cid; seeds = list(nb); k = 0
        while k < len(seeds):
            j = seeds[k]; k += 1
            if not visited[j]:
                visited[j] = True; nb2 = np.where(D[j] <= eps)[0]
                if len(nb2) >= minpts: seeds += list(nb2)
            if lab[j] == -1: lab[j] = cid
    return lab
ok = m[["gx", "gy"]].notna().all(1).values      # upv_analysis: F=m[[vel,rebN,gx,gy]].dropna()
def dbscan_noise(v):
    F = pd.DataFrame({"vel": v[ok], "rebN": reb[ok], "gx": m["gx"].values[ok], "gy": m["gy"].values[ok]})
    Z = ((F - F.mean()) / F.std()).values; lab = dbscan(Z)
    return set(m["key"].values[ok][lab == -1]), len(set(lab)) - (1 if -1 in lab else 0)
md0, thr0 = mahal(y, reb); fl0 = set(m["key"].values[md0 >= thr0])
noise0, ncl0 = dbscan_noise(y)
T8 = {"original": {"mahalanobis_threshold": r3(thr0), "n_flagged": len(fl0), "dbscan_noise": len(noise0), "dbscan_clusters": ncl0,
                   "dbscan_rows_used": int(ok.sum()), "note_dbscan_rows": "rows with NaN gx/gy (column 'C1', 15 points) are dropped before DBSCAN, as in upv_analysis.py"}}
for nm, vv in (("corrected_0.237", vc237), ("corrected_FE", vcfe)):
    md1, thr1 = mahal(vv, reb); fl1 = set(m["key"].values[md1 >= thr1]); noise1, ncl1 = dbscan_noise(vv)
    T8[nm] = {"mahalanobis_threshold": r3(thr1), "n_flagged": len(fl1), "overlap_with_original_12": len(fl1 & fl0), "kept_original": sorted(fl1 & fl0),
              "newly_flagged": sorted(fl1 - fl0), "dropped_from_original": sorted(fl0 - fl1),
              "dbscan_noise": len(noise1), "dbscan_clusters": ncl1, "dbscan_overlap_with_original_dbscan_noise": len(noise1 & noise0),
              "dbscan_overlap_with_original_12_mahalanobis": len(noise1 & fl0), "dbscan_noise_keys": sorted(noise1)}
T8["original_dbscan_noise_keys"] = sorted(noise0); T8["original_dbscan_overlap_with_original_12"] = len(noise0 & fl0)
T8["original_12_keys"] = sorted(fl0)
R["step8_triage"] = T8
# anomalies table
an = m.assign(md=md0)[md0 >= thr0].sort_values("md", ascending=False).copy()
an["vc"] = an["vel"] + an["method"].map(DELTA); an["qc"] = grade(an["vc"].values)
an["dbs"] = ["yes" if k in noise0 else ("n/a" if not ok[i] else "no") for k, i in zip(an["key"], an.index)]
rows = []
for _, r in an.iterrows():
    arr = {"Direct": "D", "Semi-Indirect": "S", "Indirect": "I"}[r["method"]]
    rows.append(f"{tx(r['col'])} & {r['point']} & {r['vel']:.2f} & {r['vc']:.2f} & {int(r['rebN'])} & {arr} & {r['quality']}/{r['qc']} & {r['priority']} & {r['dbs']} & {r['md']:.2f} \\\\")
(TAB / "anomalies12.tex").write_text("\\begin{tabular}{llccccccccc}\n\\toprule\nColumn & Pt & $v$ & $v_c$ & $N$ & Arr. & Quality (raw/corr.) & Prio. & DBSCAN & $d_M$ \\\\\n\\midrule\n" +
    "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
R["step8_anomaly_table"] = [{"col": r["col"], "point": r["point"], "vel": r3(r["vel"]), "vel_corr_0.237": r3(r["vc"]), "rebN": int(r["rebN"]), "arrangement": r["method"],
                             "quality_raw": str(r["quality"]), "quality_corrected": r["qc"], "priority": int(r["priority"]), "dbscan_outlier_raw": r["dbs"], "mahalanobis": r3(r["md"])} for _, r in an.iterrows()]
S("Step8: Mahalanobis raw flagged / thr", f"{len(fl0)} / {thr0:.3f}", "97.5th pct of d_M")
for nm in ("corrected_0.237", "corrected_FE"):
    t = T8[nm]; S(f"Step8: {nm} Mahalanobis flagged; overlap with original 12", f"{t['n_flagged']}; {t['overlap_with_original_12']}", f"newly flagged {len(t['newly_flagged'])}")
    S(f"Step8: {nm} DBSCAN noise; overlap with original DBSCAN noise / original 12", f"{t['dbscan_noise']}; {t['dbscan_overlap_with_original_dbscan_noise']} / {t['dbscan_overlap_with_original_12_mahalanobis']}", "")
S("Step8: original DBSCAN noise; overlap with original 12", f"{len(noise0)}; {len(noise0 & fl0)}", "")

# concern index
def concern_df(vel, qual_doubtful, w=(1/3, 1/3, 1/3), use_bad=True):
    g = pd.DataFrame({"v": vel, "col": m["col"].values, "b": qual_doubtful}).groupby("col")
    c = pd.DataFrame({"mean": g["v"].mean(), "cov": g["v"].std() / g["v"].mean() * 100, "pct_bad": g["b"].mean() * 100}).reindex(COLS)
    c["priority"] = info["priority"]
    zv = -(c["mean"] - c["mean"].mean()) / c["mean"].std(); zc = (c["cov"] - c["cov"].mean()) / c["cov"].std(); zb = (c["pct_bad"] - c["pct_bad"].mean()) / (c["pct_bad"].std() + 1e-9)
    c["z_vel"], c["z_cov"], c["z_bad"] = zv, zc, zb
    c["ci"] = (w[0] * zv + w[1] * zc + (w[2] * zb if use_bad else 0)) if use_bad else (zv + zc) / 2
    return c
c0 = concern_df(y, bad); order0 = c0["ci"].sort_values(ascending=False).index.tolist(); top5_0 = set(order0[:5]); worst0 = order0[0]
prio = c0["priority"].values.astype(float)
rg = np.random.default_rng(SEED)
PERMS = [rg.permutation(prio) for _ in range(10000)]
def sp_with_prio(ci_vals):
    rc = sps.rankdata(ci_vals); obs = sps.spearmanr(ci_vals, -prio).statistic
    cnt_ = 0
    for pp in PERMS:
        rp = sps.rankdata(-pp); rr_p = np.corrcoef(rc, rp)[0, 1]
        if abs(rr_p) >= abs(obs) - 1e-12: cnt_ += 1
    return float(obs), (cnt_ + 1) / (len(PERMS) + 1)
def eval_variant(c, label):
    rho, p = sp_with_prio(c["ci"].values); od = c["ci"].sort_values(ascending=False).index.tolist()
    return {"spearman_rho_with_priority": r3(rho), "perm_p": r3(p), "top5_overlap_with_original": len(set(od[:5]) & top5_0),
            "rank_of_original_worst_column": int(od.index(worst0) + 1), "worst_column": od[0], "top5": od[:5]}
CI = {"original": eval_variant(c0, "orig")}
CI["original"]["top5_columns_original_ranking"] = order0[:5]; CI["original"]["worst_column_original"] = worst0
vc = vc237
b237 = (grade(vc237) == "Doubtful").astype(int)
CI["V1_corrected_velocity_recomputed_doubtful"] = eval_variant(concern_df(vc, b237), "v1")
CI["V2_no_doubtful_fraction_corrected_mean_CoV"] = eval_variant(concern_df(vc, b237, use_bad=False), "v2")
CI["weights"] = {}
for w in ((0.5, 0.25, 0.25), (0.25, 0.5, 0.25), (0.25, 0.25, 0.5), (0.5, 0.5, 0.0)):
    cw = concern_df(y, bad, w=w); CI["weights"][str(w)] = eval_variant(cw, str(w))
R["step8_concern_index"] = CI
for k, v in CI.items():
    if k == "weights":
        for kk, vv in v.items(): S(f"Step8: concern weights {kk}", f"rho {vv['spearman_rho_with_priority']}, p {vv['perm_p']}, top5 overlap {vv['top5_overlap_with_original']}/5, rank of orig worst {vv['rank_of_original_worst_column']}", "")
    else: S(f"Step8: concern {k}", f"rho {v['spearman_rho_with_priority']}, p {v['perm_p']}, top5 overlap {v['top5_overlap_with_original']}/5, rank of orig worst {v['rank_of_original_worst_column']}, worst now {v['worst_column']}", "rho vs priority (priority 1 = worst)")

# ======================================================================
# STEP 9  classifier and clustering reporting
# ======================================================================
sup = m.copy()
sup["col_mean_excl"] = (sup.groupby("col")["vel"].transform("sum") - sup["vel"]) / (sup.groupby("col")["vel"].transform("size") - 1)
Xcat = pd.get_dummies(sup["method"], prefix="m"); Xface = pd.get_dummies(sup["face"], prefix="f")
Xsup = pd.concat([sup[["gx", "gy", "priority", "col_mean_excl"]], Xcat, Xface], axis=1)
Xsup = Xsup.fillna(sup[["gx", "gy"]].mean().mean()).fillna(0.0)
featnames = list(Xsup.columns); Xv = Xsup.values.astype(float); Xs = (Xv - Xv.mean(0)) / (Xv.std(0) + 1e-9)
yb = bad.astype(float)
def logreg_train(X, yy, l2=1.0, lr=0.1, epochs=4000):
    n, d = X.shape; w = np.zeros(d); b = 0.0
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-(X @ w + b))); g = p - yy; w -= lr * (X.T @ g / n + l2 * w / n); b -= lr * g.mean()
    return w, b
def pred(X, w, b): return 1 / (1 + np.exp(-(X @ w + b)))
def auc_(yy, s):
    pos, neg = s[yy == 1], s[yy == 0]
    if len(pos) == 0 or len(neg) == 0: return np.nan
    return sum((p > neg).sum() + 0.5 * (p == neg).sum() for p in pos) / (len(pos) * len(neg))
rgc = np.random.default_rng(SEED)
folds = np.array_split(rgc.permutation(N), 5); aucs = []
for f in range(5):
    te = folds[f]; tr = np.concatenate([folds[i] for i in range(5) if i != f]); w, b = logreg_train(Xs[tr], yb[tr]); aucs.append(auc_(yb[te], pred(Xs[te], w, b)))
from sklearn.model_selection import GroupKFold
gaucs = []; gfold_cols = []
for tr, te in GroupKFold(n_splits=5).split(Xs, yb, groups=m["colid"].values):
    w, b = logreg_train(Xs[tr], yb[tr]); gaucs.append(auc_(yb[te], pred(Xs[te], w, b))); gfold_cols.append(int(m["colid"].values[te].max() * 0 + len(set(m["colid"].values[te]))))
rep = []
rgg = np.random.default_rng(SEED)
for _ in range(20):
    assign = rgg.permutation(np.arange(NC) % 5); fo = assign[m["colid"].values]
    a_ = []
    for f in range(5):
        tr, te = np.where(fo != f)[0], np.where(fo == f)[0]; w, b = logreg_train(Xs[tr], yb[tr]); a_.append(auc_(yb[te], pred(Xs[te], w, b)))
    rep.append(np.nanmean(a_))
CL = {"features_logistic": featnames, "standardisation": "z-score with population SD (ddof=0)+1e-9, fitted on all 462 rows before CV (as in script)",
      "model": "numpy L2 logistic regression (l2=1.0, lr=0.1, 4000 full-batch epochs), target = Doubtful",
      "original_cv": {"type": "plain 5-fold on a random permutation of points (np.array_split of RNG.permutation), NOT grouped by column; same-column points can sit in train and test, and col_mean_excl is a leave-one-out column mean computed from the full data",
                      "seed": "numpy default_rng(42) (here a fresh stream; the submitted script drew the permutation from a shared generator after other steps)",
                      "per_fold_auc": [r3(a) for a in aucs], "mean": r3(np.nanmean(aucs)), "sd": r3(np.nanstd(aucs)), "stored_submitted_value": {"mean": 0.7038, "sd": 0.0475}},
      "group_kfold_by_column": {"per_fold_auc": [r3(a) for a in gaucs], "mean": r3(np.nanmean(gaucs)), "sd": r3(np.nanstd(gaucs)), "columns_per_test_fold": gfold_cols,
                                "note": "sklearn GroupKFold(5) on column id; deterministic"},
      "repeated_random_column_grouped_5fold_20x": {"mean": r3(np.mean(rep)), "sd_across_repeats": r3(np.std(rep))}}
# k-means silhouette (script method: Fz of vel,rebN,gx,gy dropna; own kmeans 10 restarts)
featk = ["vel", "rebN", "gx", "gy"]; Fk = m[featk].dropna(); Fz = ((Fk - Fk.mean()) / Fk.std()).values
rk = np.random.default_rng(SEED)
def kmeans(X, k, iters=200, restarts=10):
    best = None
    for _ in range(restarts):
        c = X[rk.choice(len(X), k, replace=False)]
        for _ in range(iters):
            d = ((X[:, None, :] - c[None, :, :]) ** 2).sum(2); lab = d.argmin(1)
            nc = np.array([X[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
            if np.allclose(nc, c): c = nc; break
            c = nc
        inertia = ((X - c[lab]) ** 2).sum()
        if best is None or inertia < best[0]: best = (inertia, lab, c)
    return best
Dm = np.sqrt(((Fz[:, None, :] - Fz[None, :, :]) ** 2).sum(2))
def silh(lab):
    sl = []
    for i in range(len(lab)):
        same = lab == lab[i]; same[i] = False
        if same.sum() == 0: sl.append(0); continue
        a = Dm[i, same].mean(); b = min(Dm[i, lab == j].mean() for j in set(lab) if j != lab[i]); sl.append((b - a) / max(a, b))
    return float(np.mean(sl))
sil = {k: r3(silh(kmeans(Fz, k)[1])) for k in range(2, 7)}
CL["kmeans_silhouette_k2_6"] = sil; CL["kmeans_best_k"] = int(max(sil, key=sil.get)); CL["kmeans_stored_submitted"] = {"2": 0.3138, "3": 0.2994, "4": 0.3021, "5": 0.3098, "6": 0.3103}
CL["feature_lists"] = {"PCA": featk + " (z-scored, ddof=1; rows with gx/gy NaN dropped -> %d rows)" % len(Fk) if False else f"{featk} z-scored (ddof=1), rows with NaN dropped -> {len(Fk)} rows; SVD",
    "kmeans": f"same standardised {featk}, own numpy k-means, 10 restarts, 200 iterations, seed 42", "DBSCAN": f"same standardised {featk}, eps=1.0, minpts=5 (<= eps, neighbourhood includes the point itself)",
    "Mahalanobis": "raw [vel, rebN] (not standardised), sample covariance, pinv, threshold = 97.5th empirical percentile of distances", "logistic": featnames,
    "concern_index": "per column: -z(mean vel), z(CoV of vel), z(% Doubtful), equal weights, z with ddof=1"}
# Moran
cs_ = m.groupby("col").agg(vel_mean=("vel", "mean"), gx=("gx", "first"), gy=("gy", "first")).dropna(subset=["gx", "gy"])
co = cs_[["gx", "gy"]].values; v = cs_["vel_mean"].values; n_ = len(v)
Dd = np.sqrt(((co[:, None, :] - co[None, :, :]) ** 2).sum(2)); Wm = (Dd <= 1.5).astype(float); np.fill_diagonal(Wm, 0)
z = v - v.mean(); S0 = Wm.sum(); I = (n_ / S0) * (z @ Wm @ z) / (z @ z)
rm = np.random.default_rng(SEED); Ip = np.array([(n_ / S0) * ((zp := rm.permutation(z)) @ Wm @ zp) / (zp @ zp) for _ in range(5000)])
pm = ((np.abs(Ip - Ip.mean()) >= abs(I - Ip.mean())).sum() + 1) / 5001
CL["moran"] = {"weights": "binary, w_ij=1 if Euclidean distance between (gx,gy) grid coordinates <= 1.5 (i!=j), not row-standardised; gx = bay + 0.5 for 'B' sub-bay, gy = gridline letter A..H -> 1..8; core column 'C1' excluded (no coordinates)",
               "variable": "column mean pulse velocity", "n_columns": n_, "I": r3(I), "E_I": r3(-1 / (n_ - 1)), "n_neighbour_links": int(S0), "permutations": 5000,
               "perm_p_two_sided": r3(pm), "null_2.5_97.5": ci(Ip), "null_mean": r3(Ip.mean()), "stored_submitted": {"I": 0.1023, "p": 0.26015}}
R["step9_classifier_clustering"] = CL
S("Step9: logistic per-fold AUC (original, ungrouped)", f"{CL['original_cv']['per_fold_auc']} mean {CL['original_cv']['mean']} SD {CL['original_cv']['sd']}", "submitted 0.704 +/- 0.048; folds NOT grouped by column")
S("Step9: logistic GroupKFold by column", f"{CL['group_kfold_by_column']['per_fold_auc']} mean {CL['group_kfold_by_column']['mean']} SD {CL['group_kfold_by_column']['sd']}", "")
S("Step9: repeated random column-grouped 5-fold (20x)", f"mean {CL['repeated_random_column_grouped_5fold_20x']['mean']}, SD across repeats {CL['repeated_random_column_grouped_5fold_20x']['sd_across_repeats']}", "")
S("Step9: k-means silhouette k=2..6", sil, "submitted 0.314/0.299/0.302/0.310/0.310")
S("Step9: Moran I / p / null 2.5-97.5", f"{CL['moran']['I']} / {CL['moran']['perm_p_two_sided']} / {CL['moran']['null_2.5_97.5']}", f"binary weights d<=1.5, n={n_}")

# ======================================================================
# STEP 10  Monte Carlo grading details (script 05, replicated exactly)
# ======================================================================
rm5 = np.random.default_rng(20); M = 20000
ci_off = [0.16, 0.295]; sigI = (ci_off[1] - ci_off[0]) / (2 * 1.96); sig_delta = {"Indirect": sigI, "Semi-Indirect": 0.030, "Direct": 0.0}
vcorr = y + m["method"].map(DELTA).values
P = np.zeros((N, 4))
for i, (vv, me) in enumerate(zip(vcorr, m["method"].values)):
    s = np.sqrt((0.02 * vv) ** 2 + sig_delta[me] ** 2); samp = rm5.normal(vv, s, M)
    g = np.where(samp > 4.5, 3, np.where(samp >= 3.5, 2, np.where(samp >= 3.0, 1, 0))); P[i] = np.bincount(g, minlength=4) / M
maxP = P.max(1)
MC = {"M_replicates_per_point": M, "seed": 20, "sigma_repeatability": "0.02 x corrected velocity (relative, normal)",
      "sigma_delta_indirect": r3(sigI), "sigma_delta_indirect_source": "(upper-lower)/(2*1.96) of the point-bootstrap CI [0.16, 0.295] of the indirect offset; hard-wired from deep_results.json",
      "sigma_delta_semi": 0.030, "sigma_delta_semi_propagated": "Yes, as a fixed hard-coded value 0.030 km/s (NOT estimated from the data or its bootstrap CI; the semi offset has no bootstrap CI in the pipeline); direct sigma_delta = 0",
      "combination": "per-point sd = sqrt((0.02 v_corr)^2 + sigma_delta[arrangement]^2), draws independent per point",
      "interval_method": "no interval: per-point class probabilities from M normal draws; point 'not confident' if max class probability < 0.8; no CI on probabilities (Monte-Carlo SE of a probability <= 0.0035 at M=20000)",
      "pct_not_confident_maxP_lt_0.8": r3((maxP < 0.8).mean() * 100), "pct_confident": r3((maxP >= 0.8).mean() * 100),
      "submitted_novelty_results_json_value": 18.4, "spec_value": 18.8,
      "expected_doubtful_count": r3(P[:, 0].sum()), "doubtful_prob_gt_0.5": int((P[:, 0] > 0.5).sum())}
# flip fractions: replicate script 03 stream (seed 11) and analytic exact
r11 = np.random.default_rng(11)
def OR_boot(g1, g2, nn=10000):
    a = m.loc[m["method"] == g1, "dbl"].values; b = m.loc[m["method"] == g2, "dbl"].values
    for _ in range(nn): a[r11.integers(0, len(a), len(a))]; b[r11.integers(0, len(b), len(b))]
OR_boot("Indirect", "Direct"); OR_boot("Indirect", "Semi-Indirect")          # advance the stream exactly as 03_extra_analyses.py does
g0 = grade(y)
def gradei(v): return np.where(v > 4.5, 3, np.where(v >= 3.5, 2, np.where(v >= 3.0, 1, 0)))
base_i = gradei(y); flipsim = {}
for name, fr in (("1pct", .01), ("2pct", .02), ("3pct", .03)):
    fl = np.zeros(N)
    for _ in range(2000): fl += gradei(y * (1 + r11.normal(0, fr, N))) != base_i
    fl /= 2000; flipsim[name] = {"mean_flip_prob": r3(fl.mean()), "pct_fragile_gt10": round(float((fl > 0.10).mean() * 100), 1)}
flipsim_2k_3000 = np.zeros(N)
for _ in range(3000): flipsim_2k_3000 += gradei(y * (1 + r11.normal(0, .02, N))) != base_i
flipsim_2k_3000 /= 3000
def flip_exact(fr):
    sd = fr * y; th = np.array([3.0, 3.5, 4.5]); out = np.zeros(N)
    for i in range(N):
        cdf = sps.norm.cdf(th, y[i], sd[i]); pr = np.diff(np.r_[0, cdf, 1]); out[i] = 1 - pr[base_i[i]]
    return out
flipex = {}
for name, fr in (("1pct", .01), ("2pct", .02), ("3pct", .03)):
    fx = flip_exact(fr); flipex[name] = {"mean_flip_prob": r3(fx.mean()), "pct_fragile_gt10": round(float((fx > 0.10).mean() * 100), 1)}
MC["flip_fractions"] = {"method_in_script": "script 03_extra_analyses.py: y*(1+N(0,frac)) multiplicative noise on RAW velocity, 2000 replicates per level (3000 for the 2% scatter figure), seed 11 (stream replicated incl. the preceding bootstrap draws); flips = grade differs from raw grade; fragile = flip prob > 0.10",
                        "replicated_simulation": flipsim, "analytic_exact_normal": flipex, "submitted": {"1pct": 15.6, "2pct": 29.0, "3pct": 42.2}}
R["step10_monte_carlo"] = MC
S("Step10: MC M / seed / sigma_delta (I, S)", f"{M} / 20 / {MC['sigma_delta_indirect']}, 0.030 (hard-coded)", "semi sigma_delta is a fixed assumption")
S("Step10: % points not confident (maxP<0.8)", MC["pct_not_confident_maxP_lt_0.8"], "submitted JSON 18.4; spec said 18.8")
S("Step10: % points fragile (flip>0.10) at 1/2/3% noise, simulated", f"{[flipsim[k]['pct_fragile_gt10'] for k in flipsim]}", "submitted 15.6/29.0/42.2")
S("Step10: same, analytic exact", f"{[flipex[k]['pct_fragile_gt10'] for k in flipex]}", "")
# pooled out-of-fold AUC of script 03 (seed 11 stream continues: concern permutation 20000 then idx permutation)
cs03 = m.groupby("col").agg(vel_mean=("vel", "mean"), vel_cov=("vel", lambda s: s.std() / s.mean() * 100), pct_bad=("dbl", "mean"), priority=("priority", "first"));
for _ in range(20000): r11.permutation(len(cs03))
idx03 = r11.permutation(N); fo03 = np.array_split(idx03, 5); sc03 = np.zeros(N)
for f in range(5):
    te = fo03[f]; tr = np.concatenate([fo03[i] for i in range(5) if i != f]); w, b = logreg_train(Xs[tr], yb[tr]); sc03[te] = pred(Xs[te], w, b)
CL["script03_pooled_oof_auc_rank_based"] = r3(auc_(yb, sc03)); CL["script03_pooled_oof_auc_submitted_trapz"] = 0.636
S("Step9: script-03 pooled out-of-fold AUC (rank-based) vs submitted 0.636", CL["script03_pooled_oof_auc_rank_based"], "stream replicated approximately")

# ======================================================================
# sensitivity.tex
# ======================================================================
rows = []
for k, lab in (("0.16", r"$+0.160$"), ("0.237", r"$+0.237$"), ("0.30", r"$+0.300$"), ("FE", f"FE ($+{FE_DELTA['Indirect']:.3f}$)")):
    s = sens[k]; da = s["doubtful_after"]
    rrv = (da["Indirect"] / nM["Indirect"]) / (da["Direct"] / nM["Direct"]) if da["Direct"] else float("nan")
    rows.append(f"{lab} & {s['n_doubtful_after']} & {s['reclassified_out_of_doubtful']} & {da['Direct']}/{da['Semi-Indirect']}/{da['Indirect']} & {rrv:.1f} \\\\")
tabA = "\\begin{tabular}{lcccc}\n\\toprule\n$\\delta_{\\mathrm{ind}}$ (km/s) & Doubtful after & Reclassified & Doubtful D/S/I & Residual RR \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"
rrows = []
for rn, lab in (("A_doubtful", "A: Doubtful point"), ("B_doubtful_or_medium", "B: Doubtful or Medium point"), ("C_column_ge_10pct_doubtful", "C: column $\\geq 10\\%$ Doubtful"), ("C_column_ge_20pct_doubtful", "C: column $\\geq 20\\%$ Doubtful")):
    a, b = R["step6_rules"]["delta_0.237"][rn], R["step6_rules"]["FE"][rn]
    pb = a.get("points_before", a.get("points_in_flagged_columns_before")); pa = a.get("points_after", a.get("points_in_flagged_columns_after"))
    pbF = b.get("points_after", b.get("points_in_flagged_columns_after"))
    rrows.append(f"{lab} & {pb} & {pa} & {pbF} & {a['columns_before']} & {a['columns_after']} & {b['columns_after']} \\\\")
tabB = ("\\begin{tabular}{lcccccc}\n\\toprule\nRule & \\multicolumn{3}{c}{Points flagged} & \\multicolumn{3}{c}{Columns flagged} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
        " & Raw & $+0.237$ & FE & Raw & $+0.237$ & FE \\\\\n\\midrule\n" + "\n".join(rrows) + "\n\\bottomrule\n\\end{tabular}\n")
(TAB / "sensitivity.tex").write_text(tabA + "\n% ---- decision rules (points for rule C = points lying in flagged columns) ----\n" + tabB)

# ======================================================================
# STEP 11  figures
# ======================================================================
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10,
 "axes.edgecolor": "black", "axes.linewidth": 0.9, "axes.grid": True, "grid.color": "0.85", "grid.linewidth": 0.6, "axes.axisbelow": True,
 "legend.frameon": True, "legend.edgecolor": "0.5", "text.color": "black", "axes.labelcolor": "black"})
cz = c0.sort_values("ci", ascending=True)
fig, ax = plt.subplots(figsize=(6.2, 7.4))
grad = {1: "0.2", 2: "0.55", 3: "0.85"}; hat = {1: "", 2: "////", 3: "...."}
ypos = np.arange(len(cz))
for yy_, (nm, r) in zip(ypos, cz.iterrows()):
    ax.barh(yy_, r["ci"], color=grad[int(r["priority"])], edgecolor="black", hatch=hat[int(r["priority"])], linewidth=0.8, height=0.78)
ax.set_yticks(ypos); ax.set_yticklabels([c.replace(" : ", ":") for c in cz.index], fontsize=7.5)
ax.axvline(0, color="black", lw=0.8)
ax.legend(handles=[Patch(facecolor=grad[p], edgecolor="black", hatch=hat[p], label=f"Priority {p}") for p in (1, 2, 3)], fontsize=8, loc="lower right")
ax.set_xlabel("Composite structural-concern index (z)"); ax.set_title("Data-driven column concern ranking (all 26 columns)")
ax.set_ylim(-0.7, len(cz) - 0.3); fig.savefig(FIG / "g9_concern.png"); plt.close(fig)
fig, ax = plt.subplots(figsize=(4.7, 3.7))
lv = ["1pct", "2pct", "3pct"]; vals = [flipsim[k]["pct_fragile_gt10"] for k in lv]
ax.bar(["1%", "2%", "3%"], vals, color=["0.75", "0.5", "0.25"], edgecolor="black")
for i, v_ in enumerate(vals): ax.annotate(f"{v_:.1f}%", (i, v_), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
ax.set_ylim(0, max(vals) * 1.12)
ax.set_xlabel("Assumed repeatability $\\sigma$"); ax.set_ylabel("% points fragile (flip prob. $>0.10$)"); fig.savefig(FIG / "gDb_fragility_bars.png"); plt.close(fig)
R["step11_figures"] = {"g9_concern.png": "regenerated, 26 columns, 7.5 pt labels, hatch by priority", "gDb_fragility_bars.png": "regenerated; original wrote '\\%' (LaTeX escape) inside matplotlib text (non-usetex) so literal backslashes showed; now plain %",
                       "gE_concern_priority.png": "original checked: no truncated labels, left unchanged"}
NOTES["g9_original_labels"] = "original g9 y-labels were mangled by n.replace('C1','').replace(' ','') -> e.g. '-1A:.C' (column prefix lost); replaced by full ids"

# ======================================================================
# STEP 12 outputs
# ======================================================================
R["notes"] = NOTES
R["meta"] = {"seed": SEED, "n_points": N, "n_columns": NC, "label_note": "'Semi-Indirect' in data = 'semi-direct' in the paper"}
def clean(o):
    if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return None if np.isnan(o) else (float(o) if np.isfinite(o) else str(o))
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.bool_): return bool(o)
    return o
(WORK / "revision_results.json").write_text(json.dumps(clean(R), indent=1, ensure_ascii=False))
with open(WORK / "SUMMARY.md", "w") as f:
    f.write("# Revision analysis summary (seed 42; Semi-Indirect in data = semi-direct in paper)\n\n")
    for lab, val, mean in SUM: f.write(f"- **{lab}**: {val}" + (f"  \n  _{mean}_" if mean else "") + "\n")
    f.write("\n## Notes\n\n" + "\n".join(f"- {k}: {v}" for k, v in NOTES.items()) + "\n")
print(open(WORK / "SUMMARY.md").read())
