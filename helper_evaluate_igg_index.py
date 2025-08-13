import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix


def evaluate_igg_index_on_test_set(file_path, random_state=42, test_size=0.2, cutoff=0.7):
    """
    Calculates the performance metrics of the conventional IgG Index on a
    hold-out test set that is consistent with the main model's test set.
    """
    print(f"--- Evaluating IgG Index on Hold-Out Test Set ---")
    print(f"Reading from: '{file_path}'")

    try:
        data = pd.read_excel(file_path)
        if 'BOS IgG indeksi' not in data.columns:
            raise ValueError("Input file must contain the 'BOS IgG indeksi' column.")

        data = data.dropna(subset=['Oligoklonal bant', 'BOS IgG indeksi'])
        data['Oligoklonal bant'] = data['Oligoklonal bant'].astype(int)

        y = data['Oligoklonal bant']
        igg_index_values = data['BOS IgG indeksi']
        X_placeholder = pd.DataFrame(index=data.index, columns=['placeholder'])

        # Recreate the exact same train/test split used for the main model
        _, _, _, y_test, _, igg_index_test = train_test_split(
            X_placeholder, y, igg_index_values,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        # Create predictions based on the standard cutoff
        y_pred_igg = (igg_index_test >= cutoff).astype(int)

        # --- Calculate Metrics ---
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_igg).ravel()
        accuracy = accuracy_score(y_test, y_pred_igg)
        sensitivity = recall_score(y_test, y_pred_igg, pos_label=1)
        specificity = recall_score(y_test, y_pred_igg, pos_label=0)
        ppv = precision_score(y_test, y_pred_igg, pos_label=1)
        npv = precision_score(y_test, y_pred_igg, pos_label=0)

        print("\n--- Final Values for Table 3 (Hold-Out Test Set) ---")
        print("\nCONFUSION MATRIX RAW NUMBERS:")
        print(f"  - Normal IgG Index & OCB Negative (TN): {tn}")
        print(f"  - Elevated IgG Index & OCB Negative (FP): {fp}")
        print(f"  - Normal IgG Index & OCB Positive (FN): {fn}")
        print(f"  - Elevated IgG Index & OCB Positive (TP): {tp}")

        print("\nCALCULATED PERFORMANCE METRICS:")
        print(f"  - Accuracy: {accuracy:.3f}")
        print(f"  - Sensitivity (Recall OCB+): {sensitivity:.3f}")
        print(f"  - Specificity (Recall OCB-): {specificity:.3f}")
        print(f"  - Positive Predictive Value (PPV): {ppv:.3f}")
        print(f"  - Negative Predictive Value (NPV): {npv:.3f}")
        print("---------------------------------------------------------")

    except Exception as e:
        print(f"!!! ERROR: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate the IgG Index on the hold-out test set."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the Excel file containing the 'BOS IgG indeksi' column (e.g., book4_with_igg_index.xlsx)."
    )

    args = parser.parse_args()
    evaluate_igg_index_on_test_set(file_path=args.input_file)