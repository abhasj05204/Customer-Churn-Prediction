# Customer Churn Prediction

A machine-learning project exploring churn classification in a Telco-style dataset. The repository includes the exploratory notebook, reusable feature-engineering and preprocessing modules, a cross-validated baseline trainer, data audits, and unit tests. Explainability, hyperparameter optimization in the corrected pipeline, prediction API, frontend, and deployment remain planned work.

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

The notebook includes EDA and the original modelling experiment. Shared modules now provide feature engineering, leakage-safe preprocessing pipelines, binary evaluation helpers, target/data audits, and a training-only stratified cross-validation baseline runner. The selected model is chosen by mean cross-validation ROC-AUC; the holdout set is intentionally not scored during selection. The original notebook metrics are historical baseline results, not production-validated performance.

A GitHub Actions workflow (`.github/workflows/ci.yml`) has been added to install dependencies, run unit tests, and execute the dataset audits on pushes and pull requests.

See:
- [Project baseline and historical metrics](docs/PROJECT_BASELINE.md)
- [Target validation findings](docs/TARGET_VALIDATION.md)
- [Data quality audit](docs/DATA_QUALITY.md)
- [Exploratory CV experiment report](docs/CV_EXPERIMENTS.md)

## Repository structure

~~~text
Customer-Churn-Prediction/
├── data/
│   └── telco_churn.csv
├── notebooks/
│   └── EDA.ipynb
├── src/
│   ├── validate_target.py
│   ├── validate_data.py
│   ├── feature_engineering.py
│   ├── preprocess.py
│   ├── evaluate.py
│   ├── train.py
│   └── optimize.py
├── models/               # Saved artifacts (ignored by Git by default)
├── reports/
│   └── figures/
├── tests/
│   ├── test_target_validation.py
│   ├── test_data_validation.py
│   ├── test_feature_engineering.py
│   ├── test_preprocess.py
│   ├── test_evaluate.py
│   ├── test_train.py
│   └── test_optimize.py
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

## Run baseline cross-validated model comparison

From the repository root, run the baseline experiment with the default stratified 5-fold cross-validation:

~~~bash
python -m src.train --output-dir outputs
~~~

The script compares balanced Logistic Regression, balanced Random Forest, and unresampled XGBoost using training-only cross-validation, selects by mean CV ROC-AUC, fits that baseline on the training portion, and deliberately leaves holdout metrics unevaluated. It saves a local JSON report and joblib model under the chosen output directory. The model artifact is ignored by Git. The holdout is reserved for final evaluation after model and threshold selection.

## Run Optuna tuning

Optuna searches XGBoost hyperparameters using stratified cross-validation **only on the training split**. Preprocessing is fitted separately within each CV training fold, no synthetic resampling is used, and the holdout test labels are not scored.

~~~bash
python -m src.optimize --trials 30 --folds 5 --output-dir outputs
~~~

The command writes `outputs/optuna_xgboost_cv.json` and `outputs/optuna_xgboost_pipeline.joblib`. These artifacts are local and ignored by Git. The CV score is a tuning estimate, not the final test score. Compare the tuned XGBoost candidate with the cross-validated Logistic Regression baseline before final model selection.

Equivalent top-level command: `python main.py tune --trials 30`.

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
2. Audit data quality and feature engineering. (Audit implementation is present; suspicious zero-tenure charges still need a documented modeling experiment.)
3. Run and verify the cross-validated baseline in a clean environment; compare class-weighted models against no-resampling alternatives.
4. Run Optuna tuning and compare its cross-validation results with the Logistic Regression baseline; assess class-weighting and categorical-aware imbalance handling.
5. Choose a decision threshold using training-only out-of-fold/validation predictions and reserve the holdout test for final evaluation.
6. Add model explainability, business insights, and production-facing inference modules.
7. Add API, frontend, end-to-end prediction tests, monitoring hooks, and deployment.

## Notes

- Do not interpret association or feature importance as proof of causation.
- Do not select a model or decision threshold using final test results.
- Current results classify the dataset's State label; they do not establish future churn prediction.
