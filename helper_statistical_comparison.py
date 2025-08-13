import pandas as pd
import numpy as np
import joblib
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample


def compare_model_vs_index(data_path, model_artifact_path, random_state=42, test_size=0.2, n_bootstraps=1000):
    """
    Compares the ROC-AUC of the trained SYNAPSI model against the conventional
    IgG Index on the same hold-out test set and calculates a p-value for the difference.
    """
    print(f"--- Comparing SYNAPSI Model vs. IgG Index ---")
    print(f"Reading data from: '{data_path}'")
    print(f"Loading model from: '{model_artifact_path}'")

    try:
        # Adım 1: Gerekli dosyaları ve verileri yükle
        artifact = joblib.load(model_artifact_path)
        ensemble_model, imputer, selected_features = artifact['model'], artifact['imputer'], artifact['features']

        data = pd.read_excel(data_path)
        if 'BOS IgG indeksi' not in data.columns:
            raise ValueError("Input file must contain the 'BOS IgG indeksi' column.")

        data = data.dropna(subset=['Oligoklonal bant', 'BOS IgG indeksi'])
        data['Oligoklonal bant'] = data['Oligoklonal bant'].astype(int)

        # Final model için gerekli olan özellikleri oluştur
        data["BOS_Protein_to_BOS_Albumin_Ratio"] = data['BOS Total Protein'] / (data['BOS Albumin'] + 1e-6)
        data["Log_IgG_BOS"] = np.log1p(data['BOS IgG'])
        data["Log_Serum_IgG"] = np.log1p(data['Serum IgG'])
        data["Sodium_Potassium_Diff"] = data['BOS Sodyum'] - data['BOS Potasyum']
        data["Glucose_to_Protein_Ratio"] = data['BOS Glukoz'] / (data['BOS Total Protein'] + 1e-6)
        data["CSF_Serum_Albumin_Ratio"] = data["BOS Albumin"] / (data["Serum Albumin"] + 1e-6)
        if "YAŞ" in data.columns: data.rename(columns={"YAŞ": "YAS"}, inplace=True)

        y = data['Oligoklonal bant']
        X_full = data[selected_features]
        igg_index_values = data['BOS IgG indeksi']

        # Adım 2: Nihai model ile birebir aynı test setini oluştur
        _, X_test, _, y_test, _, igg_index_test = train_test_split(
            X_full, y, igg_index_values, test_size=test_size, random_state=random_state, stratify=y
        )

        X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=selected_features)
        synapsi_probs = ensemble_model.predict_proba(X_test_imputed)[:, 1]

        # Adım 3: Bootstrap ile karşılaştırma yap
        bootstrap_synapsi_aucs, bootstrap_igg_index_aucs = [], []
        y_test_np, synapsi_probs_np, igg_index_test_np = np.array(y_test), np.array(synapsi_probs), np.array(
            igg_index_test)

        for i in range(n_bootstraps):
            indices = resample(np.arange(len(y_test_np)), random_state=random_state + i)
            if len(np.unique(y_test_np[indices])) < 2: continue
            bootstrap_synapsi_aucs.append(roc_auc_score(y_test_np[indices], synapsi_probs_np[indices]))
            bootstrap_igg_index_aucs.append(roc_auc_score(y_test_np[indices], igg_index_test_np[indices]))

        # Adım 4: P-değerini ve sonuçları hesapla
        diff_scores = np.array(bootstrap_synapsi_aucs) - np.array(bootstrap_igg_index_aucs)
        p_value = 2 * np.mean(diff_scores < 0) if np.mean(diff_scores) > 0 else 2 * np.mean(diff_scores > 0)
        p_value_str = "<.001" if p_value < 0.001 else f"{p_value:.3f}"

        def format_ci_string(scores):
            return f"{np.mean(scores):.3f} (95% CI: {np.percentile(scores, 2.5):.3f}–{np.percentile(scores, 97.5):.3f})"

        print("\n--- Statistical Comparison Results ---")
        print(f"SYNAPSI Model ROC-AUC: {format_ci_string(bootstrap_synapsi_aucs)}")
        print(f"IgG Index ROC-AUC    : {format_ci_string(bootstrap_igg_index_aucs)}")
        print(f"\nP-value for the difference in ROC-AUC: {p_value_str}")

        if p_value < 0.05:
            print("\nConclusion: The difference in performance is statistically significant.")
        else:
            print("\nConclusion: The difference in performance is not statistically significant.")

    except Exception as e:
        print(f"!!! ERROR: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Statistically compare the SYNAPSI model against the IgG Index."
    )
    parser.add_argument("data_file", type=str, help="Path to the Excel file with the IgG Index column.")
    parser.add_argument("model_file", type=str, help="Path to the saved SYNAPSI model artifact (.pkl file).")

    args = parser.parse_args()
    compare_model_vs_index(data_path=args.data_file, model_artifact_path=args.model_file)