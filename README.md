# A Machine Learning Model for Predicting Oligoclonal Band Positivity

This repository contains the Python scripts for the study titled "An Artificial Intelligence Model for Predicting Oligoclonal Band (OCB) Positivity Using Routine CSF and Serum Biochemical Markers".

The project is structured as a multi-stage machine learning pipeline that proceeds from a comprehensive screening of multiple algorithms to the development and validation of a final, optimized ensemble model named SYNAPSI.

## Workflow Overview

The analysis is divided into two main phases, supported by several helper scripts for data preparation and statistical analysis.

1.  **Phase 1: Comprehensive Algorithm Screening (`1_model_screening.py`)**
    * This script evaluates six different machine learning architectures on the full feature set using a 5-fold cross-validation methodology.
    * It generates a comprehensive performance report with 95% confidence intervals for all key metrics.
    * It also produces comparative SHAP summary plots to interpret the feature importance of the top-performing models.

2.  **Phase 2: Final Ensemble Model Development (`2_final_ensemble_model.py`)**
    * This script takes the insights from Phase 1 (top 3 models, top 8 features).
    * It performs hyperparameter optimization for the selected models using Optuna.
    * It constructs, trains, and evaluates the final weighted soft-voting ensemble model (SYNAPSI) on a sequestered hold-out test set.

## Requirements

To run these scripts, you need Python 3.11 and the libraries listed in the `requirements.txt` file. You can install all dependencies using pip:
```bash
pip install -r requirements.txt
```

## Usage

The scripts should be run in the following logical order. Note that file paths in the scripts may need to be adjusted based on your project structure.

### Step 1: (Optional) Prepare Data
This helper script calculates the `BOS IgG indeksi` from the raw data and saves it to a new Excel file. This is a prerequisite for the statistical comparison script.

```bash
python helper_add_igg_index.py [path/to/raw_data.xlsx] [path/to/output_data_with_index.xlsx]
```

### Step 2: Run Comprehensive Algorithm Screening
This is the main script for **Phase 1** of the study. It compares 6 ML models and generates the initial performance tables and SHAP plots.

```bash
python 1_model_screening.py [path/to/data_with_index.xlsx]
```

### Step 3: Run Final Ensemble Model Development
This is the main script for **Phase 2** of the study. It trains, optimizes, and evaluates the final SYNAPSI model.

```bash
python 2_final_ensemble_model.py [path/to/data_with_index.xlsx]
```

### Step 4: Run Helper Analyses
These scripts provide supporting evidence for the manuscript.

**A) Evaluate Baseline IgG Index Performance (for Table 3)**
This script calculates the performance metrics of the conventional IgG index on the same hold-out test set used by the final model, ensuring a fair comparison.

```bash
python helper_evaluate_igg_index.py [path/to/data_with_index.xlsx]
```

**B) Perform Statistical Comparison**
This script statistically compares the final SYNAPSI model against the IgG Index and calculates a p-value for the difference in ROC-AUC performance.

```bash
python helper_statistical_comparison.py [path/to/data_with_index.xlsx] [path/to/saved_model.pkl]
```

## Data

Please note that the clinical dataset used for this study contains sensitive patient information and cannot be made publicly available due to privacy regulations and ethical committee restrictions. The code is provided for methodological transparency and reproducibility.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
