import pandas as pd
import argparse
import warnings

warnings.filterwarnings("ignore")


def prepare_igg_index(input_file, output_file):
    """
    Reads an Excel file, calculates the IgG Index, and saves the result to a new file.
    """
    print(f"--- Data Preparation Step 1: Calculating IgG Index ---")
    print(f"Reading from: '{input_file}'")

    try:
        data = pd.read_excel(input_file)
        required_columns = ['BOS IgG', 'Serum IgG', 'BOS Albumin', 'Serum Albumin']
        if not all(col in data.columns for col in required_columns):
            missing = [col for col in required_columns if col not in data.columns]
            raise ValueError(f"Required columns for calculation are missing: {missing}")

        epsilon = 1e-9
        q_igg = data['BOS IgG'] / (data['Serum IgG'] + epsilon)
        q_albumin = data['BOS Albumin'] / (data['Serum Albumin'] + epsilon)
        data['BOS IgG indeksi'] = q_igg / (q_albumin + epsilon)

        data.to_excel(output_file, index=False)

        print(f"✅ SUCCESS: New 'BOS IgG indeksi' column saved to '{output_file}'")

    except Exception as e:
        print(f"!!! ERROR: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Calculate IgG Index and create a new data file."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the original input Excel data file (e.g., book4.xlsx)."
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path for the new output Excel file (e.g., book4_with_igg_index.xlsx)."
    )

    args = parser.parse_args()
    prepare_igg_index(args.input_file, args.output_file)