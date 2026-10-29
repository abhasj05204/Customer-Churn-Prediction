# Customer Churn Prediction

A machine-learning project exploring customer churn in a Telco-style dataset. The repository currently contains an exploratory notebook and an initial modelling experiment. The reusable training pipeline, prediction API, automated tests, and deployment are planned work—not yet complete.

## Current status

- Dataset: 7,043 rows and 22 columns.
- The dataset contains two labels: `State` and `StateNextMonth`.
- The current notebook drops `customerID` and `StateNextMonth`, then predicts `State` (current-state classification).
- The notebook has initial EDA, feature engineering, preprocessing, four baseline classifiers, Optuna-tuned XGBoost, and test-set metrics.
- A baseline snapshot, including recorded metrics and known evaluation limitations, is in [`docs/PROJECT_BASELINE.md`](docs/PROJECT_BASELINE.md).

**Important:** The existing notebook results are an initial experiment, not a production-ready performance claim. Before retraining, the project must explicitly define the prediction horizon and verify the meaning and timing of `StateNextMonth`. The current tuning split was made after SMOTE and after preprocessing was fitted on the full training split; its Optuna validation score should not be treated as an unbiased generalization estimate. Ordinary SMOTE was applied after one-hot encoding and should be reviewed for categorical-feature validity.

## Repository structure

```text
Customer-Churn-Prediction/
├── data/
│   └── telco_churn.csv
├── notebooks/
│   └── EDA.ipynb
├── src/                  # Reusable modules to be implemented
├── models/               # Saved model artifacts (not committed by default)
├── reports/
│   └── figures/
├── tests/                # Automated tests to be added
├── docs/
│   └── PROJECT_BASELINE.md
├── requirements.txt
├── README.md
└── main.py                # Currently a dataset-loading smoke check
```

## Setup

Use a virtual environment. Python 3.12 is the recommended starting point for this project; installation should be validated on the machine that will run it.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the current checks

From the repository root, run the dataset-loading smoke check:

```bash
python main.py
```

The current `main.py` reads `data/telco_churn.csv` and prints sample rows and the dataset shape. It does **not** train a model or serve predictions yet.

To open the existing notebook, start Jupyter from the `notebooks` directory so its relative dataset path (`../data/telco_churn.csv`) resolves:

```bash
cd notebooks
jupyter lab EDA.ipynb
```

Return to the repository root before running `python main.py`.

## Recorded baseline metrics

These are the saved results for the notebook's current `State` classification task, with the existing evaluation protocol. They are recorded for reproducibility and comparison, not presented as validated next-month churn performance.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7360 | 0.5017 | 0.7834 | 0.6117 | 0.8408 |
| Random Forest | 0.7722 | 0.5734 | 0.5535 | 0.5633 | 0.8235 |
| SVM | 0.7523 | 0.5243 | 0.7219 | 0.6074 | 0.8230 |
| Baseline XGBoost | 0.7793 | 0.5827 | 0.5936 | 0.5881 | 0.8376 |
| Optuna-tuned XGBoost | 0.7842 | 0.5946 | 0.5882 | 0.5914 | 0.8395 |

## Planned engineering milestones

1. Validate target definitions, label provenance, prediction-time features, and business objective.
2. Audit data quality and feature engineering.
3. Rebuild preprocessing and model comparison with leakage-safe stratified cross-validation.
4. Compare class weighting, categorical-aware sampling, and no-resampling; tune on training data only.
5. Select the final model and decision threshold using validation data, then evaluate the chosen model on an untouched test set.
6. Add explainability, business insights, reusable training/evaluation modules, automated tests, model serialization, prediction API, frontend, and deployment.

## Notes

- Do not interpret association or feature importance as proof of causation.
- Do not select the model or tune the decision threshold based on the final test results.
- Model artifacts such as `.pkl` and `.joblib` files are ignored by Git; artifact handling will be documented when the training pipeline is implemented.


## Target audit (Step 2)

Run the reproducible label-distribution and active-cohort audit from the repository root:

~~~bash
python -m src.validate_target
python -m unittest discover -s tests -v
~~~

The audit reports the relationship between State and StateNextMonth and the candidate cohort with State == "No Churn". It is descriptive only; check [docs/TARGET_VALIDATION.md](docs/TARGET_VALIDATION.md) for assumptions that must be confirmed before retraining. The target must not be switched until label definitions, feature timestamps, and label provenance are verified.
