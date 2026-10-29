# Customer Churn Prediction

A machine-learning project exploring churn classification in a Telco-style dataset. The repository contains an exploratory notebook and an initial modelling experiment; a reusable training pipeline, API, automated suite, and deployment are planned work.

## Current dataset

The selected file is the 21-column CSV supplied for this project and committed at [data/telco_churn.csv](data/telco_churn.csv).

- Rows: 7,043
- Columns: 21
- Target column: State
- Labels: 5,174 No Churn and 1,869 Churn (26.54% Churn)
- Missing cells: 0
- Duplicate rows: 0
- Unique customer IDs: 7,043
- Dataset SHA-256: 58a435baed4902c1642facdbba81f8faa7c10d7fcec37b9c5f7077aecaf47ffd

This selected dataset does not contain StateNextMonth. The project therefore currently supports classification of the State label only. It must not be described as a verified next-month churn predictor without a time-aligned future outcome label and reliable label provenance.

## Current project status

The existing notebook includes EDA, engineered features, train/test split, preprocessing, four baseline classifiers, Optuna-tuned XGBoost, and recorded test metrics. The saved metrics are an initial baseline experiment, not production-validated performance.

See:
- [Project baseline and historical metrics](docs/PROJECT_BASELINE.md)
- [Target validation findings](docs/TARGET_VALIDATION.md)
- [Data quality audit](docs/DATA_QUALITY.md)

## Repository structure

~~~text
Customer-Churn-Prediction/
├── data/
│   └── telco_churn.csv
├── notebooks/
│   └── EDA.ipynb
├── src/
│   ├── validate_target.py
│   └── validate_data.py
├── models/               # Saved artifacts (ignored by Git by default)
├── reports/
│   └── figures/
├── tests/
│   ├── test_target_validation.py
│   └── test_data_validation.py
├── docs/
│   ├── PROJECT_BASELINE.md
│   ├── TARGET_VALIDATION.md
│   └── DATA_QUALITY.md
├── requirements.txt
├── README.md
└── main.py
~~~

## Setup

Python 3.12 is the recommended starting point. Validate dependency installation in a clean environment before treating it as a supported runtime.

### Windows PowerShell

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

### macOS / Linux

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

## Run data validation and tests

From the repository root:

~~~bash
python -m src.validate_target
python -m src.validate_data
python -m unittest discover -s tests -v
~~~

The target audit reports the label distribution, row/column counts, customer ID duplication, and whether a separate next-month target exists. It does not infer a prediction horizon from a column name.

## Run the existing notebook

Start Jupyter from the notebooks directory so its relative data path resolves:

~~~bash
cd notebooks
jupyter lab EDA.ipynb
~~~

## Recorded baseline metrics

These saved results are for the notebook's State classification task with its original evaluation protocol. The 21-column dataset now committed has the same first 21 columns/values as the prior file; the previously appended StateNextMonth column was not used as the model target.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7360 | 0.5017 | 0.7834 | 0.6117 | 0.8408 |
| Random Forest | 0.7722 | 0.5734 | 0.5535 | 0.5633 | 0.8235 |
| SVM | 0.7523 | 0.5243 | 0.7219 | 0.6074 | 0.8230 |
| Baseline XGBoost | 0.7793 | 0.5827 | 0.5936 | 0.5881 | 0.8376 |
| Optuna-tuned XGBoost | 0.7842 | 0.5946 | 0.5882 | 0.5914 | 0.8395 |

The old Optuna validation score must not be treated as an unbiased generalization estimate: the tuning split was created after SMOTE and after preprocessing was fitted on the complete training split. Ordinary SMOTE was applied after one-hot encoding; this is also under review because interpolation can create fractional categorical indicators. The test set was not oversampled, but repeated model comparisons on test metrics mean it has influenced model selection.

## Planned engineering milestones

1. Define the intended business use and confirm the semantics/timing of State.
2. Audit data quality and feature engineering.
3. Rebuild preprocessing and model comparison using leakage-safe stratified cross-validation.
4. Compare class weighting, categorical-aware resampling, and no resampling; tune on training data only.
5. Select threshold and model using validation predictions, then evaluate once on an untouched holdout test set.
6. Add explainability, business insights, reusable training/evaluation modules, automated tests, model persistence, prediction API, frontend, and deployment.

## Notes

- Do not interpret association or feature importance as proof of causation.
- Do not select a model or decision threshold using final test results.
- Current results classify the dataset's State label; they do not establish future churn prediction.
