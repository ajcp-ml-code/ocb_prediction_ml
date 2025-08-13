#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SYNAPSI – Phase 3 (Final, AJCP style)
- Dynamic ensemble weights from dev 5-fold ROC-AUC
- Stable thresholds (median of dev 5-fold F1 & Youden)
- Single-shot hold-out TEST evaluation
- AJCP figures: Times New Roman, TIFF+PNG, no titles, panel letters (A,B,C)
- SHAP bar + beeswarm as 3-panel composites (CatBoost, XGBoost, LightGBM)
- Deployment artifact dump

Usage:
  # With pre-split files:
  python phase3_final_ajcp.py --dev_file book4_dev.xlsx --test_file book4_test.xlsx

  # Split inside (and optionally save the split):
  python phase3_final_ajcp.py --input_file book4.xlsx --save_split
"""

import argparse
import os
import warnings
import random
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, confusion_matrix, auc
)
from sklearn.calibration import calibration_curve

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import shap

# --------------------------
# Global / Style
# --------------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)

OUTPUT_DIR = "results/SYNAPSI_phase3_final_AJCP"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "shap_plots"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "confusion_matrices"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "splits"), exist_ok=True)

# (Global format'ı kaldırdık; her figürü hem .tif hem .png kaydedeceğiz)
plt.rcParams["figure.dpi"] = 600
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12

# Selected features + display mapping
SELECTED_FEATURES = [
    "Log_IgG_BOS", "YAS", "BOS_Protein_to_BOS_Albumin_Ratio",
    "Log_Serum_IgG", "Sodium_Potassium_Diff", "CRP",
    "Glucose_to_Protein_Ratio", "CSF_Serum_Albumin_Ratio"
]
FEATURE_DISPLAY = {
    "Log_IgG_BOS": "log(CSF IgG)",
    "YAS": "Age",
    "BOS_Protein_to_BOS_Albumin_Ratio": "CSF Protein/Albumin Ratio",
    "Log_Serum_IgG": "log(Serum IgG)",
    "Sodium_Potassium_Diff": "CSF Na–K Diff.",
    "CRP": "C-reactive protein",
    "Glucose_to_Protein_Ratio": "CSF Glucose/Protein",
    "CSF_Serum_Albumin_Ratio": "CSF/Serum Albumin",
}

# --------------------------
# Data & FE
# --------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    print(f"✅ Loaded: {path}  shape={df.shape}")
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Oligoklonal bant" not in df.columns:
        raise ValueError("Missing target column 'Oligoklonal bant'.")
    df = df.dropna(subset=["Oligoklonal bant"])
    df["Oligoklonal bant"] = df["Oligoklonal bant"].astype(int)

    for col in ["Hasta No", "Protokol No", "BOS IgG indeksi"]:
        if col in df.columns:
            df = df.drop(columns=[col])

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

    df = df.loc[:, df.notna().any(axis=0)]
    return df

# --------------------------
# Thresholds & Bootstrap
# --------------------------
def find_optimal_thresholds(y_true, y_prob):
    precision, recall, thr_pr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    f1_idx = np.argmax(f1[:-1])
    f1_thr = float(thr_pr[f1_idx])

    fpr, tpr, thr_roc = roc_curve(y_true, y_prob)
    yj_idx = np.argmax(tpr - fpr)
    youden_thr = float(thr_roc[yj_idx])
    return f1_thr, youden_thr

def bootstrap_metrics(y_true, y_prob, threshold, n_iterations=1000, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true); y_prob = np.array(y_prob)
    metrics = {
        "ROC-AUC": [], "PR-AUC": [], "Accuracy": [],
        "Recall_Pos": [], "Recall_Neg": [],
        "Precision_Pos": [], "Precision_Neg": [],
        "F1_Pos": [], "F1_Neg": []
    }
    for _ in range(n_iterations):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        yt = y_true[idx]; yp = y_prob[idx]
        yhat = (yp >= threshold).astype(int)

        metrics["ROC-AUC"].append(roc_auc_score(yt, yp))
        prec, rec, _ = precision_recall_curve(yt, yp)
        metrics["PR-AUC"].append(auc(rec, prec))
        metrics["Accuracy"].append(accuracy_score(yt, yhat))
        metrics["Recall_Pos"].append(recall_score(yt, yhat, pos_label=1))
        metrics["Recall_Neg"].append(recall_score(yt, yhat, pos_label=0))
        metrics["Precision_Pos"].append(precision_score(yt, yhat, pos_label=1, zero_division=0))
        metrics["Precision_Neg"].append(precision_score(yt, yhat, pos_label=0, zero_division=0))
        metrics["F1_Pos"].append(f1_score(yt, yhat, pos_label=1, zero_division=0))
        metrics["F1_Neg"].append(f1_score(yt, yhat, pos_label=0, zero_division=0))

    out = {}
    for k, vals in metrics.items():
        vals = np.array(vals)
        out[k] = {"mean": float(np.mean(vals)),
                  "95% CI": (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))}
    return out

def ci_to_string(ci_dict):
    return f"{ci_dict['mean']:.3f} ({ci_dict['95% CI'][0]:.3f}–{ci_dict['95% CI'][1]:.3f})"

# --------------------------
# Figure helpers (AJCP)
# --------------------------
def add_panel_label(ax, label):
    ax.text(0.02, 0.98, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=12, fontweight="bold")

# Combined ROC + PR + Calibration (A,B,C)
def plot_roc_pr_calibration(y_true, y_prob, tag):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    # A) ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[0].plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y_true, y_prob):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right", frameon=False)
    add_panel_label(axes[0], "A")

    # B) PR
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    axes[1].plot(rec, prec, lw=2, label=f"AUC = {auc(rec, prec):.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left", frameon=False)
    add_panel_label(axes[1], "B")

    # C) Calibration
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    axes[2].plot(prob_pred, prob_true, marker="o", lw=2, label="Model")
    axes[2].plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    axes[2].set_xlabel("Mean Predicted Probability")
    axes[2].set_ylabel("Fraction of Positives")
    axes[2].legend(frameon=False)
    add_panel_label(axes[2], "C")

    for ax in axes:
        ax.grid(False)

    fig.tight_layout()
    out_base = os.path.join(OUTPUT_DIR, "figures", f"ROC_PR_Calibration_{tag}")
    fig.savefig(out_base + ".tif", dpi=600)
    fig.savefig(out_base + ".png", dpi=600)
    plt.close(fig)
    print(f"📈 Saved: {out_base}.tif  &  {out_base}.png")

# 2-panel confusion matrices with letters
def plot_confusions(y_true, y_prob, thresholds: dict, filename_tag: str):
    names = list(thresholds.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5))
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        th = thresholds[name]
        yhat = (np.array(y_prob) >= th).astype(int)
        cm = confusion_matrix(y_true, yhat)

        im = ax.imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, str(v), ha="center", va="center", fontsize=11)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["OCB-", "OCB+"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["OCB-", "OCB+"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")

    # Önce eksenleri sıkıştır, sonra konumları al
    fig.tight_layout()

    # Panel harflerini eksen DIŞINA, sol-üst köşeye yerleştir
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for i, ax in enumerate(axes):
        pos = ax.get_position()  # figure koordinatları (0-1)
        # Sol-üstün biraz dışına: x0'dan biraz sola, y1'in biraz üzerine
        fig.text(pos.x0 - 0.02, pos.y1 + 0.01, labels[i],
                 fontsize=14, fontweight="bold", ha="left", va="bottom")

    # Harflerin kesilmemesi için üst/sol marjı biraz aç
    plt.subplots_adjust(top=0.92, left=0.10)

    out_base = os.path.join(OUTPUT_DIR, "confusion_matrices", f"confusions_{filename_tag}")
    fig.savefig(out_base + ".tif", dpi=600)
    fig.savefig(out_base + ".png", dpi=600)
    plt.close(fig)
    print(f"🧩 Saved: {out_base}.tif  &  {out_base}.png")


# SHAP composites: bar and beeswarm, each 3-panels A,B,C in CatBoost → XGBoost → LightGBM order
def shap_composites_on_dev(models_dict: dict, X_dev_imputed: pd.DataFrame, y_dev: pd.Series):
    ordered = [("CatBoost", models_dict["CatBoost"]),
               ("XGBoost", models_dict["XGBoost"]),
               ("LightGBM", models_dict["LightGBM"])]

    X_disp = X_dev_imputed.rename(columns=FEATURE_DISPLAY)

    # Fit all once
    for _, m in ordered:
        m.fit(X_dev_imputed, y_dev)

    # BAR
    fig_bar, axes_bar = plt.subplots(1, 3, figsize=(16, 5.5))
    for i, (name, m) in enumerate(ordered):
        expl = shap.TreeExplainer(m)
        sv = expl.shap_values(X_dev_imputed)
        if isinstance(sv, list):
            sv_plot = sv[1]
        elif np.ndim(sv) == 3:
            sv_plot = sv[:, :, 1]
        else:
            sv_plot = sv
        plt.sca(axes_bar[i])
        shap.summary_plot(sv_plot, X_disp, plot_type="bar", show=False, max_display=12, color=None)
        add_panel_label(axes_bar[i], ["A", "B", "C"][i])
        axes_bar[i].set_title("")

    fig_bar.tight_layout()
    out_bar_base = os.path.join(OUTPUT_DIR, "shap_plots", "SHAP_bar_AJCP")
    fig_bar.savefig(out_bar_base + ".tif", dpi=600)
    fig_bar.savefig(out_bar_base + ".png", dpi=600)
    plt.close(fig_bar)
    print(f"📊 Saved: {out_bar_base}.tif  &  {out_bar_base}.png")

    # BEESWARM
    fig_bee, axes_bee = plt.subplots(1, 3, figsize=(16, 5.5))
    for i, (name, m) in enumerate(ordered):
        expl = shap.TreeExplainer(m)
        sv = expl.shap_values(X_dev_imputed)
        if isinstance(sv, list):
            sv_plot = sv[1]
        elif np.ndim(sv) == 3:
            sv_plot = sv[:, :, 1]
        else:
            sv_plot = sv
        plt.sca(axes_bee[i])
        shap.summary_plot(sv_plot, X_disp, show=False, max_display=12, color=None)
        add_panel_label(axes_bee[i], ["A", "B", "C"][i])
        axes_bee[i].set_title("")

    fig_bee.tight_layout()
    out_bee_base = os.path.join(OUTPUT_DIR, "shap_plots", "SHAP_beeswarm_AJCP")
    fig_bee.savefig(out_bee_base + ".tif", dpi=600)
    fig_bee.savefig(out_bee_base + ".png", dpi=600)
    plt.close(fig_bee)
    print(f"🐝 Saved: {out_bee_base}.tif  &  {out_bee_base}.png")

# --------------------------
# Dynamic weights
# --------------------------
def compute_dynamic_weights(X_dev_imputed: pd.DataFrame, y_dev: pd.Series,
                            cat_params: dict, xgb_params: dict, lgb_params: dict):
    cfg = {
        "CatBoost": CatBoostClassifier(**cat_params, random_seed=RANDOM_STATE, verbose=0),
        "XGBoost":  XGBClassifier(**xgb_params, random_state=RANDOM_STATE),
        "LightGBM": LGBMClassifier(**lgb_params, random_state=RANDOM_STATE),
    }
    print("\n--- Computing dynamic weights from dev 5-fold ROC-AUC ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = {}
    for name, model in cfg.items():
        aucs = []
        for tr, va in skf.split(X_dev_imputed, y_dev):
            X_tr, X_va = X_dev_imputed.iloc[tr], X_dev_imputed.iloc[va]
            y_tr, y_va = y_dev.iloc[tr], y_dev.iloc[va]
            model.fit(X_tr, y_tr)
            prob = model.predict_proba(X_va)[:, 1]
            aucs.append(roc_auc_score(y_va, prob))
        scores[name] = float(np.mean(aucs))
        print(f"  {name}: mean ROC-AUC = {scores[name]:.3f}")

    tot = sum(scores.values())
    weights = [scores["CatBoost"]/tot, scores["XGBoost"]/tot, scores["LightGBM"]/tot]
    print(f"✅ Weights: CatBoost={weights[0]:.3f} | XGBoost={weights[1]:.3f} | LightGBM={weights[2]:.3f}")
    return weights, scores

def _test_auc(model, X, y, name):
    p = model.predict_proba(X)[:, 1]
    print(f"{name} TEST ROC-AUC = {roc_auc_score(y, p):.3f}")


# --------------------------
# Main
# --------------------------
def main(args):
    # fixed params (from Phase-2)
    best_params_catboost = {"iterations": 400, "depth": 4, "learning_rate": 0.08, "l2_leaf_reg": 3.0}
    best_params_xgboost  = {"n_estimators": 263, "max_depth": 3, "learning_rate": 0.06,
                            "subsample": 0.8, "colsample_bytree": 0.8,
                            "use_label_encoder": False, "eval_metric": "logloss"}
    best_params_lightgbm = {"n_estimators": 150, "max_depth": 4, "learning_rate": 0.05,
                            "num_leaves": 30, "subsample": 0.7, "colsample_bytree": 0.7,
                            "force_row_wise": True}

    # Inputs / split
    if args.dev_file and args.test_file:
        df_dev  = feature_engineering(load_data(args.dev_file))
        df_test = feature_engineering(load_data(args.test_file))
        for c in SELECTED_FEATURES + ["Oligoklonal bant"]:
            if c not in df_dev.columns:  raise ValueError(f"Missing in DEV: {c}")
            if c not in df_test.columns: raise ValueError(f"Missing in TEST: {c}")
        X_dev  = df_dev[SELECTED_FEATURES].copy();  y_dev  = df_dev["Oligoklonal bant"].copy()
        X_test = df_test[SELECTED_FEATURES].copy(); y_test = df_test["Oligoklonal bant"].copy()
        split_info = "Provided dev/test files"
    elif args.input_file:
        df_all = feature_engineering(load_data(args.input_file))
        for c in SELECTED_FEATURES + ["Oligoklonal bant"]:
            if c not in df_all.columns: raise ValueError(f"Missing in ALL: {c}")
        X = df_all[SELECTED_FEATURES].copy(); y = df_all["Oligoklonal bant"].copy()
        X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
        split_info = "Internal 80/20 split"
        if args.save_split:
            dev_path  = os.path.join(OUTPUT_DIR, "splits", "dev_split.xlsx")
            test_path = os.path.join(OUTPUT_DIR, "splits", "test_split.xlsx")
            df_all.loc[X_dev.index].to_excel(dev_path, index=False)
            df_all.loc[X_test.index].to_excel(test_path, index=False)
            print(f"💾 Saved splits: {dev_path}  |  {test_path}")
    else:
        raise ValueError("Provide --dev_file & --test_file OR --input_file.")

    print(f"🔀 Split mode: {split_info}  |  dev={len(y_dev)}  test={len(y_test)}")

    # Imputer from DEV only
    imputer = SimpleImputer(strategy="median")
    X_dev_imputed  = pd.DataFrame(imputer.fit_transform(X_dev),  columns=SELECTED_FEATURES)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=SELECTED_FEATURES)

    # Stable thresholds on DEV (CatBoost, 5-fold)
    print("\n--- Finding stable thresholds on DEV (CatBoost, 5-fold) ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    f1_list, youden_list = [], []
    for tr, va in skf.split(X_dev_imputed, y_dev):
        X_tr, X_va = X_dev_imputed.iloc[tr], X_dev_imputed.iloc[va]
        y_tr, y_va = y_dev.iloc[tr], y_dev.iloc[va]
        m = CatBoostClassifier(**best_params_catboost, random_seed=RANDOM_STATE, verbose=0)
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_va)[:, 1]
        f1_th, yj_th = find_optimal_thresholds(y_va, p)
        f1_list.append(f1_th); youden_list.append(yj_th)
    final_f1_th     = float(np.median(f1_list))
    final_youden_th = float(np.median(youden_list))
    print(f"✅ Stable thresholds  F1={final_f1_th:.4f} | Youden={final_youden_th:.4f}")

    # Dynamic weights (DEV 5-fold ROC-AUC)
    weights, cv_model_scores = compute_dynamic_weights(
        X_dev_imputed, y_dev,
        best_params_catboost, best_params_xgboost, best_params_lightgbm
    )

    # Train ensemble on full DEV
    cat = CatBoostClassifier(**best_params_catboost, random_seed=RANDOM_STATE, verbose=0)
    xgb = XGBClassifier(**best_params_xgboost, random_state=RANDOM_STATE)
    lgb = LGBMClassifier(**best_params_lightgbm, random_state=RANDOM_STATE)
    ensemble = VotingClassifier(
        estimators=[("CatBoost", cat), ("XGBoost", xgb), ("LightGBM", lgb)],
        voting="soft", weights=weights
    )
    ensemble.fit(X_dev_imputed, y_dev)
    # ---- Fitted klonları al ve TEST AUC yazdır ----
    # sklearn 1.x: named_estimators_ var
    if hasattr(ensemble, "named_estimators_"):
        f_cat = ensemble.named_estimators_["CatBoost"]
        f_xgb = ensemble.named_estimators_["XGBoost"]
        f_lgb = ensemble.named_estimators_["LightGBM"]
    else:
        # daha eski sürümler için güvenli yol
        names = [name for name, _ in ensemble.estimators]
        fitted_map = dict(zip(names, ensemble.estimators_))
        f_cat = fitted_map["CatBoost"]
        f_xgb = fitted_map["XGBoost"]
        f_lgb = fitted_map["LightGBM"]

    _test_auc(f_cat, X_test_imputed, y_test, "CatBoost")
    _test_auc(f_xgb, X_test_imputed, y_test, "XGBoost")
    _test_auc(f_lgb, X_test_imputed, y_test, "LightGBM")

    p_ens = ensemble.predict_proba(X_test_imputed)[:, 1]
    print(f"Ensemble TEST ROC-AUC = {roc_auc_score(y_test, p_ens):.3f}")

    # TEST evaluation (single run)
    y_prob_test = ensemble.predict_proba(X_test_imputed)[:, 1]
    roc_auc_val = roc_auc_score(y_test, y_prob_test)
    prec, rec, _ = precision_recall_curve(y_test, y_prob_test)
    pr_auc_val = auc(rec, prec)
    print(f"\n🌟 TEST  ROC-AUC={roc_auc_val:.3f}  |  PR-AUC={pr_auc_val:.3f}")

    # Bootstrap CIs at 3 thresholds
    boot_f1 = bootstrap_metrics(y_test, y_prob_test, final_f1_th)
    boot_yj = bootstrap_metrics(y_test, y_prob_test, final_youden_th)
    boot_05 = bootstrap_metrics(y_test, y_prob_test, 0.5)

    report = pd.DataFrame({
        "Threshold Type": ["F1 (stable)", "Youden (stable)", "Default 0.5"],
        "Threshold Value": [f"{final_f1_th:.4f}", f"{final_youden_th:.4f}", "0.500"],
        "ROC-AUC": [ci_to_string(boot_f1["ROC-AUC"]),
                    ci_to_string(boot_yj["ROC-AUC"]),
                    ci_to_string(boot_05["ROC-AUC"])],
        "PR-AUC": [ci_to_string(boot_f1["PR-AUC"]),
                   ci_to_string(boot_yj["PR-AUC"]),
                   ci_to_string(boot_05["PR-AUC"])],
        "Accuracy": [ci_to_string(boot_f1["Accuracy"]),
                     ci_to_string(boot_yj["Accuracy"]),
                     ci_to_string(boot_05["Accuracy"])],
        "Sensitivity (OCB+)": [ci_to_string(boot_f1["Recall_Pos"]),
                               ci_to_string(boot_yj["Recall_Pos"]),
                               ci_to_string(boot_05["Recall_Pos"])],
        "Specificity (OCB-)": [ci_to_string(boot_f1["Recall_Neg"]),
                               ci_to_string(boot_yj["Recall_Neg"]),
                               ci_to_string(boot_05["Recall_Neg"])],
        "Precision (OCB+)": [ci_to_string(boot_f1["Precision_Pos"]),
                             ci_to_string(boot_yj["Precision_Pos"]),
                             ci_to_string(boot_05["Precision_Pos"])],
        "Precision (OCB-)": [ci_to_string(boot_f1["Precision_Neg"]),
                             ci_to_string(boot_yj["Precision_Neg"]),
                             ci_to_string(boot_05["Precision_Neg"])],
        "F1 (OCB+)": [ci_to_string(boot_f1["F1_Pos"]),
                      ci_to_string(boot_yj["F1_Pos"]),
                      ci_to_string(boot_05["F1_Pos"])],
        "F1 (OCB-)": [ci_to_string(boot_f1["F1_Neg"]),
                      ci_to_string(boot_yj["F1_Neg"]),
                      ci_to_string(boot_05["F1_Neg"])],
    })
    out_xlsx = os.path.join(OUTPUT_DIR, "SYNAPSI_final_test_metrics_AJCP.xlsx")
    report.to_excel(out_xlsx, index=False)
    print("\n🔎 TEST metrics (bootstrap 95% CI):")
    print(report.to_string(index=False))
    print(f"✅ Saved: {out_xlsx}")

    # AJCP multi-panel figures (letters only because multi-panel)
    plot_roc_pr_calibration(y_test, y_prob_test, tag="TEST")
    plot_confusions(
        y_test, y_prob_test,
        thresholds={"Youden": final_youden_th, "0.5": 0.5},
        filename_tag="TEST_Youden_05"
    )
    plot_confusions(
        y_test, y_prob_test,
        thresholds={"F1": final_f1_th, "0.5": 0.5},
        filename_tag="TEST_F1_05"
    )

    # SHAP composites on DEV (CatBoost, XGBoost, LightGBM)
    shap_models = {
        "CatBoost": CatBoostClassifier(**best_params_catboost, random_seed=RANDOM_STATE, verbose=0),
        "XGBoost":  XGBClassifier(**best_params_xgboost, random_state=RANDOM_STATE),
        "LightGBM": LGBMClassifier(**best_params_lightgbm, random_state=RANDOM_STATE),
    }
    shap_composites_on_dev(shap_models, X_dev_imputed, y_dev)

    # Deployment artifact
    artifact = {
        "model": ensemble,
        "imputer": imputer,
        "features": SELECTED_FEATURES,
        "thresholds": {"F1_stable": final_f1_th, "Youden_stable": final_youden_th, "default_0.5": 0.5},
        "weights": weights,
        "cv_model_scores_dev": cv_model_scores,
        "hyperparams": {
            "catboost": best_params_catboost,
            "xgboost": best_params_xgboost,
            "lightgbm": best_params_lightgbm
        }
    }
    out_pkl = os.path.join(OUTPUT_DIR, "SYNAPSI_deployment_model.pkl")
    joblib.dump(artifact, out_pkl)
    print(f"💾 Saved deployment artifact: {out_pkl}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SYNAPSI Phase 3 – AJCP style")
    ap.add_argument("--input_file", type=str, default=None,
                    help="Single Excel (internal 80/20 split). e.g. book4.xlsx")
    ap.add_argument("--dev_file", type=str, default=None, help="Development Excel (pre-split)")
    ap.add_argument("--test_file", type=str, default=None, help="Test Excel (pre-split)")
    ap.add_argument("--save_split", action="store_true", help="Save internal split when using --input_file")
    args = ap.parse_args()
    main(args)
