import argparse
import pandas as pd
import numpy as np
import os
import warnings
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, roc_curve, precision_recall_curve, \
    auc, precision_score
from sklearn.utils import resample
import shap
import tempfile

# Karşılaştırılacak tüm modeller
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier

# --- AYARLAR ---
warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
OUTPUT_DIR = "results/Final_Comparison_with_Stats"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
plt.rcParams["savefig.format"] = "tiff";
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.family"] = "Times New Roman";
plt.rcParams["font.size"] = 12


# --- VERİ HAZIRLAMA FONKSİYONU ---
def load_and_prepare_data(file_path):
    ALLOWED_RAW_FEATURES = ['YAŞ', 'CİNSİYET', 'BOS Glukoz', 'BOS Total Protein', 'BOS Klorür', 'BOS Mikroalbumin',
                            'CRP', 'Eritrosit Sedimentasyon Hızı', 'BOS Lökosit', 'BOS Eritrosit', 'BOS Sodyum',
                            'BOS Potasyum', 'BOS Albumin', 'Serum IgG', 'Serum Albumin', 'BOS IgG', 'BOS Laktik Asit']
    TARGET_COLUMN = 'Oligoklonal bant'
    data = pd.read_excel(file_path)
    id_columns_to_drop = ['Hasta No', 'Protokol No', 'BOS IgG indeksi']
    for col in id_columns_to_drop:
        if col in data.columns: data = data.drop(columns=[col])
    all_needed_columns = ALLOWED_RAW_FEATURES + [TARGET_COLUMN]
    existing_cols = [col for col in all_needed_columns if col in data.columns]
    data = data[existing_cols]
    data = data.dropna(subset=[TARGET_COLUMN])
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)
    if 'CİNSİYET' in data.columns:
        le = LabelEncoder()
        data['CİNSİYET'] = data['CİNSİYET'].fillna('Bilinmiyor').astype(str)
        data['CİNSİYET'] = le.fit_transform(data['CİNSİYET'])
    data["BOS Protein / BOS Albumin"] = data['BOS Total Protein'] / (data['BOS Albumin'] + 1e-6)
    data["BOS Glukoz / BOS Total Protein"] = data['BOS Glukoz'] / (data['BOS Total Protein'] + 1e-6)
    data["BOS Logaritmik IgG"] = np.log1p(data['BOS IgG'])
    data["Serum Logaritmik IgG"] = np.log1p(data['Serum IgG'])
    data["BOS Albumin / Serum Albumin"] = data['BOS Albumin'] / (data['Serum Albumin'] + 1e-6)
    data["BOS IgG / Serum IgG"] = data['BOS IgG'] / (data['Serum IgG'] + 1e-6)
    data["BOS Sodyum- BOS Potasyum"] = data['BOS Sodyum'] - data['BOS Potasyum']
    data = data.rename(columns={"YAŞ": "YAS"})
    y = data[TARGET_COLUMN]
    X = data.drop(columns=[TARGET_COLUMN])
    return X, y


# --- METRİK HESAPLAMA VE GÖRSELLEŞTİRME FONKSİYONLARI ---

def format_ci_string(scores):
    mean = np.mean(scores)
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    return f"{mean:.3f} ({lower:.3f}–{upper:.3f})"


def bootstrap_on_cv_results(y_true, y_pred_prob, n_iterations=1000, return_distributions=False):
    y_true, y_pred_prob = np.array(y_true), np.array(y_pred_prob)
    metric_scores = {"ROC-AUC": [], "PR-AUC": [], "Accuracy": [], "Sensitivity": [], "Specificity": [],
                     "Precision (Positive)": [], "Precision (Negative)": [], "F1 (Positive)": [], "F1 (Negative)": []}
    for i in range(n_iterations):
        indices = np.random.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2: continue
        y_true_boot, y_pred_prob_boot = y_true[indices], y_pred_prob[indices]
        fpr, tpr, thresholds = roc_curve(y_true_boot, y_pred_prob_boot)
        optimal_threshold = thresholds[np.argmax(tpr - fpr)]
        y_pred_boot = (y_pred_prob_boot >= optimal_threshold).astype(int)
        metric_scores["ROC-AUC"].append(roc_auc_score(y_true_boot, y_pred_prob_boot))
        precision, recall, _ = precision_recall_curve(y_true_boot, y_pred_prob_boot)
        metric_scores["PR-AUC"].append(auc(recall, precision))
        metric_scores["Accuracy"].append(accuracy_score(y_true_boot, y_pred_boot))
        metric_scores["Sensitivity"].append(recall_score(y_true_boot, y_pred_boot, pos_label=1, zero_division=0))
        metric_scores["Specificity"].append(recall_score(y_true_boot, y_pred_boot, pos_label=0, zero_division=0))
        metric_scores["F1 (Positive)"].append(f1_score(y_true_boot, y_pred_boot, pos_label=1, zero_division=0))
        metric_scores["F1 (Negative)"].append(f1_score(y_true_boot, y_pred_boot, pos_label=0, zero_division=0))
        metric_scores["Precision (Positive)"].append(
            precision_score(y_true_boot, y_pred_boot, pos_label=1, zero_division=0))
        metric_scores["Precision (Negative)"].append(
            precision_score(y_true_boot, y_pred_boot, pos_label=0, zero_division=0))

    if return_distributions:
        for key in metric_scores: metric_scores[key] = np.array(metric_scores[key])
        return metric_scores

    final_results = {}
    for name, scores in metric_scores.items():
        final_results[name] = format_ci_string(scores)
    return final_results


def perform_statistical_tests(bootstrap_scores_dict):
    models = list(bootstrap_scores_dict.keys())
    p_values_data = []
    print("\n--- Pairwise Statistical Significance (p-values for ROC-AUC) ---")

    sorted_models = sorted(models, key=lambda m: np.mean(bootstrap_scores_dict[m]["ROC-AUC"]), reverse=True)

    for i in range(len(sorted_models)):
        for j in range(i + 1, len(sorted_models)):
            model1, model2 = sorted_models[i], sorted_models[j]
            scores1, scores2 = bootstrap_scores_dict[model1]["ROC-AUC"], bootstrap_scores_dict[model2]["ROC-AUC"]
            diff_scores = scores1 - scores2
            p_value = 2 * np.mean(diff_scores < 0) if np.mean(diff_scores) > 0 else 2 * np.mean(diff_scores > 0)
            p_value_str = "<.001" if p_value < 0.001 else f"{p_value:.3f}"
            print(f"{model1} vs {model2}: P = {p_value_str}")
            p_values_data.append({"Comparison": f"{model1} vs {model2}", "P-value": p_value_str})

    return pd.DataFrame(p_values_data)


def plot_shap_comparisons(models_to_plot, X, y, display_names_map):
    print("\n--- Generating Advanced Comparative SHAP Beeswarm Plots ---")
    with tempfile.TemporaryDirectory() as temp_dir:
        image_paths = []
        for name, model in models_to_plot.items():
            pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('classifier', model)])
            pipeline.fit(X, y)
            print(f"Fitted {name} on full dataset for SHAP.")

            X_imputed_transformed = pipeline.named_steps['imputer'].transform(X)
            explainer = shap.TreeExplainer(pipeline.named_steps['classifier'], X_imputed_transformed)
            shap_values = explainer.shap_values(X_imputed_transformed, check_additivity=False)

            if isinstance(shap_values, list): shap_values = shap_values[1]
            if shap_values.ndim == 3: shap_values = shap_values[:, :, 1]

            X_display = X.rename(columns=display_names_map)

            plt.figure(figsize=(8, 10))
            shap.summary_plot(shap_values, X_display, show=False, max_display=12)
            plt.tight_layout()

            temp_path = os.path.join(temp_dir, f"shap_{name}.png")
            plt.savefig(temp_path, dpi=300, bbox_inches='tight')
            image_paths.append(temp_path)
            plt.close()

        fig, axes = plt.subplots(1, len(image_paths), figsize=(22, 8))
        if len(image_paths) == 1: axes = [axes]
        subplot_labels = ['A', 'B', 'C']
        for i, (ax, img_path) in enumerate(zip(axes, image_paths)):
            img = plt.imread(img_path)
            ax.imshow(img)
            ax.axis('off')
            ax.text(0.01, 0.99, subplot_labels[i], transform=ax.transAxes, fontsize=16,
                    fontweight='bold', va='top', ha='left')

        plt.tight_layout(pad=0.1, h_pad=0, w_pad=0)
        plt.savefig(os.path.join(OUTPUT_DIR, "plots", "final_comparative_shap_summary.tiff"))
        plt.show()


# --- ANA İŞ AKIŞI ---
def screen_algorithms_and_interpret(file_path):
    X, y = load_and_prepare_data(file_path)

    models = {"Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
              "XGBoost": XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss'),
              "LightGBM": LGBMClassifier(random_state=RANDOM_STATE),
              "CatBoost": CatBoostClassifier(random_state=RANDOM_STATE, verbose=0),
              "AdaBoost": AdaBoostClassifier(random_state=RANDOM_STATE),
              "MLP Neural Net": MLPClassifier(random_state=RANDOM_STATE, max_iter=1000)}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    all_results = []
    bootstrap_distributions_dict = {}

    print("\n--- Starting 5-Fold CV to Collect Out-of-Fold Predictions ---")
    for name, model in models.items():
        pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('classifier', model)])
        y_true_all_folds, y_pred_prob_all_folds = [], []
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val, y_train, y_val = X.iloc[train_idx], X.iloc[val_idx], y.iloc[train_idx], y.iloc[val_idx]
            pipeline.fit(X_train, y_train)
            y_pred_prob_all_folds.extend(pipeline.predict_proba(X_val)[:, 1])
            y_true_all_folds.extend(y_val)

        print(f"✅ Collected predictions for {name}. Now calculating bootstrap CIs and distributions...")
        bootstrap_distributions = bootstrap_on_cv_results(y_true_all_folds, y_pred_prob_all_folds,
                                                          return_distributions=True)
        bootstrap_distributions_dict[name] = bootstrap_distributions

        formatted_results = {key: format_ci_string(scores) for key, scores in bootstrap_distributions.items()}
        formatted_results["Model"] = name
        all_results.append(formatted_results)

    # --- SONUÇLARI RAPORLAMA ---
    results_df = pd.DataFrame(all_results)
    ordered_cols = ['Model', 'ROC-AUC', 'PR-AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision (Positive)',
                    'Precision (Negative)', 'F1 (Positive)', 'F1 (Negative)']
    results_df = results_df[ordered_cols]
    results_df = results_df.sort_values(by="ROC-AUC", ascending=False,
                                        key=lambda col: col.str.split(' ').str[0].astype(float))
    print("\n--- Initial Model Screening Results (5-Fold CV with 95% CI) ---")
    print(results_df.to_string(index=False))
    excel_path = os.path.join(OUTPUT_DIR, "initial_model_screening_CI_results.xlsx")
    results_df.to_excel(excel_path, index=False)
    print(f"\n✅ Main results saved to: {excel_path}")

    p_values_df = perform_statistical_tests(bootstrap_distributions_dict)
    p_values_excel_path = os.path.join(OUTPUT_DIR, "statistical_test_p_values.xlsx")
    p_values_df.to_excel(p_values_excel_path, index=False)
    print(f"✅ P-value results saved to: {p_values_excel_path}")

    # --- GÖRSELLEŞTİRME VE YORUMLAMA ---
    print("\n--- Generating Visualizations and Interpretations ---")

    plot_df = results_df.copy()
    plot_df['Mean ROC-AUC'] = plot_df['ROC-AUC'].apply(lambda x: float(x.split(' ')[0]))
    plt.figure(figsize=(10, 7))
    sns.barplot(x="Mean ROC-AUC", y="Model", data=plot_df, palette="viridis")
    plt.xlabel("Mean ROC-AUC (from Bootstrapped CV Predictions)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", "model_screening_comparison_CI_no_title.tiff"))
    plt.show()

    sorted_model_names = results_df['Model'].tolist()
    top_models_for_shap_names = [name for name in sorted_model_names if name in ["XGBoost", "LightGBM", "CatBoost"]]
    models_to_plot_shap = {name: models[name] for name in top_models_for_shap_names}

    feature_display_names = {
        'YAS': 'Age', 'CİNSİYET': 'Sex', 'BOS Glukoz': 'CSF Glucose', 'BOS Total Protein': 'CSF Total Protein',
        'BOS Klorür': 'CSF Chloride', 'BOS Mikroalbumin': 'CSF Microalbumin', 'CRP': 'CRP',
        'Eritrosit Sedimentasyon Hızı': 'ESR', 'BOS Lökosit': 'CSF WBC', 'BOS Eritrosit': 'CSF RBC',
        'BOS Sodyum': 'CSF Sodium', 'BOS Potasyum': 'CSF Potassium', 'BOS Albumin': 'CSF Albumin',
        'Serum IgG': 'Serum IgG', 'Serum Albumin': 'Serum Albumin', 'BOS IgG': 'CSF IgG',
        'BOS Laktik Asit': 'CSF Lactic Acid', 'BOS IgG indeksi': 'CSF IgG Index',
        'BOS Protein / BOS Albumin': 'CSF Protein / Albumin Ratio',
        'BOS Glukoz / BOS Total Protein': 'CSF Glucose / Protein Ratio', 'BOS Logaritmik IgG': 'log(CSF IgG)',
        'Serum Logaritmik IgG': 'log(Serum IgG)', 'BOS Albumin / Serum Albumin': 'CSF / Serum Albumin Ratio',
        'BOS IgG / Serum IgG': 'CSF / Serum IgG Ratio', 'BOS Sodyum- BOS Potasyum': 'CSF Sodium - Potassium Diff.'
    }

    plot_shap_comparisons(models_to_plot_shap, X, y, feature_display_names)


if __name__ == '__main__':
    # Komut satırından argüman almak için bir parser oluştur
    parser = argparse.ArgumentParser(
        description="Compare ML algorithms for OCB prediction."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input Excel data file (e.g., book4.xlsx)"
    )

    args = parser.parse_args()

    # Ana fonksiyonu, komut satırından gelen dosya yolu ile çağır
    screen_algorithms_and_interpret(file_path=args.input_file)