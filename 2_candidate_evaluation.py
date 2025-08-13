#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compare SYNAPSI vs IgG Index on the same TEST set (AJCP style)

- Loads Phase-3 deployment artifact (joblib pkl)
- Recreates FE and imputes with the artifact's imputer
- Computes SYNAPSI probabilities and IgG index from raw columns
- AUCs with 95% CIs (bootstrap), ΔAUC with 95% CI, two-tailed bootstrap p
- Paired "DeLong" p via bootstrap-covariance approximation
- Threshold metrics: SYNAPSI at artifact's Youden; IgG at --igg_cutoff (default 0.70)
- Saves all results to ONE Excel file with multiple sheets (no CSV delimiter pain)
- AJCP figures: Times New Roman, 600 dpi TIFF, NO titles, NO p text on plots
"""

import argparse
import os
import warnings
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import norm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Matplotlib (AJCP style)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["savefig.format"] = "tiff"
plt.rcParams["figure.dpi"] = 600
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12

from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    accuracy_score, recall_score, precision_score, f1_score
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -------- Phase-3 feature set --------
SELECTED_FEATURES = [
    "Log_IgG_BOS", "YAS", "BOS_Protein_to_BOS_Albumin_Ratio",
    "Log_Serum_IgG", "Sodium_Potassium_Diff", "CRP",
    "Glucose_to_Protein_Ratio", "CSF_Serum_Albumin_Ratio"
]

def feature_engineering_phase3(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Oligoklonal bant" not in df.columns:
        raise ValueError("Missing target column 'Oligoklonal bant' in TEST file.")
    df = df.dropna(subset=["Oligoklonal bant"]).copy()
    df["Oligoklonal bant"] = df["Oligoklonal bant"].astype(int)

    # Drop obvious IDs/leak columns if present
    for col in ["Hasta No", "Protokol No", "BOS IgG indeksi"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # FE (same as Phase-3)
    if "BOS Total Protein" in df.columns and "BOS Albumin" in df.columns:
        df["BOS_Protein_to_BOS_Albumin_Ratio"] = df["BOS Total Protein"] / (df["BOS Albumin"] + 1e-6)
    if "BOS IgG" in df.columns:
        df["Log_IgG_BOS"] = np.log1p(df["BOS IgG"])
    if "Serum IgG" in df.columns:
        df["Log_Serum_IgG"] = np.log1p(df["Serum IgG"])
    if "BOS Sodyum" in df.columns and "BOS Potasyum" in df.columns:
        df["Sodium_Potassium_Diff"] = df["BOS Sodyum"] - df["BOS Potasyum"]
    if "BOS Glukoz" in df.columns and "BOS Total Protein" in df.columns:
        df["Glucose_to_Protein_Ratio"] = df["BOS Glukoz"] / (df["BOS Total Protein"] + 1e-6)
    if "BOS Albumin" in df.columns and "Serum Albumin" in df.columns:
        df["CSF_Serum_Albumin_Ratio"] = df["BOS Albumin"] / (df["Serum Albumin"] + 1e-6)
    if "YAŞ" in df.columns and "YAS" not in df.columns:
        df = df.rename(columns={"YAŞ": "YAS"})

    # Drop fully-empty columns (if any)
    df = df.loc[:, df.notna().any(axis=0)]

    missing = [c for c in SELECTED_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"After FE, missing required features: {missing}")

    return df

# -------- IgG index --------
def compute_igg_index(df: pd.DataFrame,
                      csf_igg_col="BOS IgG", serum_igg_col="Serum IgG",
                      csf_alb_col="BOS Albumin", serum_alb_col="Serum Albumin") -> np.ndarray:
    for c in [csf_igg_col, serum_igg_col, csf_alb_col, serum_alb_col]:
        if c not in df.columns:
            raise ValueError(f"IgG index needs column: '{c}'")

    csf_igg = df[csf_igg_col].astype(float).values
    ser_igg = df[serum_igg_col].astype(float).values
    csf_alb = df[csf_alb_col].astype(float).values
    ser_alb = df[serum_alb_col].astype(float).values

    # IgG index = (CSF IgG / Serum IgG) / (CSF Alb / Serum Alb)
    eps = 1e-12
    igg_index = (csf_igg * ser_alb) / ((ser_igg + eps) * (csf_alb + eps))
    return igg_index

# -------- Bootstrap helpers --------
def bootstrap_auc(y, scores, B=2000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    aucs = []
    n = len(y)
    idx = np.arange(n)
    for _ in range(B):
        b = rng.integers(0, n, n)
        if len(np.unique(y[b])) < 2:
            continue
        aucs.append(roc_auc_score(y[b], scores[b]))
    aucs = np.array(aucs)
    return aucs, float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

def bootstrap_auc_diff(y, s1, s2, B=2000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(y)
    for _ in range(B):
        b = rng.integers(0, n, n)
        if len(np.unique(y[b])) < 2:
            continue
        a1 = roc_auc_score(y[b], s1[b])
        a2 = roc_auc_score(y[b], s2[b])
        diffs.append(a1 - a2)
    diffs = np.array(diffs)
    prop_pos = (diffs >= 0).mean()
    p_two = 2 * min(prop_pos, 1 - prop_pos)
    return (diffs,
            float(np.mean(diffs)),
            float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)),
            float(p_two))

def delong_paired_test(y_true, s1, s2, Bcov=2000, seed=RANDOM_STATE):
    """Paired 'DeLong' p via bootstrap-covariance approximation."""
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(y_true)
    for _ in range(Bcov):
        b = rng.integers(0, n, n)
        if len(np.unique(y_true[b])) < 2:
            continue
        diffs.append(roc_auc_score(y_true[b], s1[b]) - roc_auc_score(y_true[b], s2[b]))
    diffs = np.array(diffs)
    m = diffs.mean()
    s = diffs.std(ddof=1)
    if s == 0:
        return float(m), 1.0
    z = m / (s + 1e-12)
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(m), float(p)

# -------- Metrics at threshold --------
def metrics_at(y, scores, thr: float):
    yhat = (scores >= thr).astype(int)
    return dict(
        Accuracy=accuracy_score(y, yhat),
        Sensitivity=recall_score(y, yhat, pos_label=1, zero_division=0),
        Specificity=recall_score(y, yhat, pos_label=0, zero_division=0),
        Precision_Pos=precision_score(y, yhat, pos_label=1, zero_division=0),
        Precision_Neg=precision_score(y, yhat, pos_label=0, zero_division=0),
        F1_Pos=f1_score(y, yhat, pos_label=1, zero_division=0),
        F1_Neg=f1_score(y, yhat, pos_label=0, zero_division=0),
    )

# -------- AJCP plots (NO titles, NO p text) --------
def plot_roc(y, s_syn, s_igg, out_path):
    fpr1, tpr1, _ = roc_curve(y, s_syn)
    fpr2, tpr2, _ = roc_curve(y, s_igg)
    auc1 = roc_auc_score(y, s_syn)
    auc2 = roc_auc_score(y, s_igg)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr1, tpr1, lw=2, label=f"SYNAPSI (AUC = {auc1:.3f})")
    plt.plot(fpr2, tpr2, lw=2, label=f"IgG Index (AUC = {auc2:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", frameon=False)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()

def plot_pr(y, s_syn, s_igg, out_path):
    p1, r1, _ = precision_recall_curve(y, s_syn)
    p2, r2, _ = precision_recall_curve(y, s_igg)
    a1 = auc(r1, p1); a2 = auc(r2, p2)

    plt.figure(figsize=(6, 6))
    plt.plot(r1, p1, lw=2, label=f"SYNAPSI (AUC = {a1:.3f})")
    plt.plot(r2, p2, lw=2, label=f"IgG Index (AUC = {a2:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left", frameon=False)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()

# -------- Main --------
def main():
    ap = argparse.ArgumentParser(description="Compare SYNAPSI (artifact) vs IgG index on the same TEST set.")
    ap.add_argument("--artifact", required=True, help="Path to SYNAPSI_deployment_model_AJCP.pkl")
    ap.add_argument("--test_file", required=True, help="Path to TEST Excel (same split as Phase-3 test)")
    ap.add_argument("--sheet_name", default=0, help="Excel sheet name or index")
    ap.add_argument("--outdir", default="results/SYNAPSI_vs_IgG_AJCP", help="Output directory")
    # IgG columns (override if your headers differ)
    ap.add_argument("--csf_igg_col", default="BOS IgG")
    ap.add_argument("--serum_igg_col", default="Serum IgG")
    ap.add_argument("--csf_alb_col", default="BOS Albumin")
    ap.add_argument("--serum_alb_col", default="Serum Albumin")
    # IgG decision threshold (default 0.70)
    ap.add_argument("--igg_cutoff", type=float, default=0.70)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load artifact
    art = joblib.load(args.artifact)
    model = art["model"]
    imputer = art["imputer"]
    feats = art["features"]
    thresholds = art.get("thresholds", {})
    syn_thr = float(thresholds.get("Youden_stable", 0.5))

    # 2) Read TEST & FE
    raw = pd.read_excel(args.test_file, sheet_name=args.sheet_name)
    df = feature_engineering_phase3(raw)

    # 3) Prepare X,y and impute
    y = df["Oligoklonal bant"].values.astype(int)
    X = df[feats].copy()
    X_imp = pd.DataFrame(imputer.transform(X), columns=feats)

    # 4) SYNAPSI scores
    syn_scores = model.predict_proba(X_imp)[:, 1]

    # 5) IgG index
    igg_index = compute_igg_index(
        df,
        csf_igg_col=args.csf_igg_col,
        serum_igg_col=args.serum_igg_col,
        csf_alb_col=args.csf_alb_col,
        serum_alb_col=args.serum_alb_col
    )

    # 6) AUCs
    auc_syn = roc_auc_score(y, syn_scores)
    auc_igg = roc_auc_score(y, igg_index)

    # 7) Bootstrap CIs
    _, syn_mean, syn_lo, syn_hi = bootstrap_auc(y, syn_scores, B=2000, seed=RANDOM_STATE)
    _, igg_mean, igg_lo, igg_hi = bootstrap_auc(y, igg_index, B=2000, seed=RANDOM_STATE)

    # 8) ΔAUC + bootstrap p
    _, d_mean, d_lo, d_hi, p_boot = bootstrap_auc_diff(y, syn_scores, igg_index, B=2000, seed=RANDOM_STATE)

    # 9) Paired DeLong (bootstrap-cov approx)
    _, p_delong = delong_paired_test(y, syn_scores, igg_index, Bcov=2000, seed=RANDOM_STATE)

    # 10) Threshold metrics
    m_syn = metrics_at(y, syn_scores, syn_thr)
    m_igg = metrics_at(y, igg_index, args.igg_cutoff)

    # 11) Save Excel (multiple sheets — no “single line” problems)
    xlsx_path = os.path.join(args.outdir, "syn_vs_igg_results.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame([{
            "AUC_SYN": round(auc_syn, 3),
            "AUC_SYN_CI_low": round(syn_lo, 3),
            "AUC_SYN_CI_high": round(syn_hi, 3),
            "AUC_IgG": round(auc_igg, 3),
            "AUC_IgG_CI_low": round(igg_lo, 3),
            "AUC_IgG_CI_high": round(igg_hi, 3),
            "Delta_AUC_SYN_minus_IgG": round(d_mean, 3),
            "Delta_CI_low": round(d_lo, 3),
            "Delta_CI_high": round(d_hi, 3),
            "Bootstrap_p_two_tailed": round(p_boot, 3),
            "Paired_DeLong_p": round(p_delong, 3)
        }]).to_excel(writer, sheet_name="AUC_Comparison", index=False)

        thr_df = pd.DataFrame([
            {"Model": "SYNAPSI", "Threshold": syn_thr, **{k: round(v, 3) for k, v in m_syn.items()}},
            {"Model": "IgG Index", "Threshold": args.igg_cutoff, **{k: round(v, 3) for k, v in m_igg.items()}},
        ])
        thr_df.to_excel(writer, sheet_name="Threshold_Metrics", index=False)

        pd.DataFrame({
            "y_true": y,
            "SYNAPSI_prob": syn_scores,
            "IgG_index": igg_index
        }).to_excel(writer, sheet_name="Predictions", index=False)

        meta = pd.DataFrame([{
            "artifact": args.artifact,
            "test_file": args.test_file,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "syn_threshold_used": syn_thr,
            "igg_cutoff_used": args.igg_cutoff,
            "B_bootstrap": 2000
        }])
        meta.to_excel(writer, sheet_name="Info", index=False)

    # 12) Figures (AJCP compliant)
    plot_roc(y, syn_scores, igg_index, os.path.join(args.outdir, "roc_syn_vs_igg.tiff"))
    plot_pr(y, syn_scores, igg_index, os.path.join(args.outdir, "pr_syn_vs_igg.tiff"))

    # 13) Console summary
    print("\n✅ Done")
    print(f" AUC_SYN  = {auc_syn:.3f}  [{syn_lo:.3f}, {syn_hi:.3f}]")
    print(f" AUC_IgG  = {auc_igg:.3f}  [{igg_lo:.3f}, {igg_hi:.3f}]")
    print(f" ΔAUC (SYN−IgG) = {d_mean:.3f}  [{d_lo:.3f}, {d_hi:.3f}]")
    print(f" Bootstrap p (two-tailed) = {p_boot:.3f}")
    print(f" Paired 'DeLong' p (bootstrap-cov approx) = {p_delong:.3f}")
    print(f" Excel: {xlsx_path}")
    print(" Figures: roc_syn_vs_igg.tiff, pr_syn_vs_igg.tiff")

if __name__ == "__main__":
    main()
