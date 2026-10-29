# Customer Churn Prediction

A machine-learning project exploring churn classification in a Telco-style dataset. The repository includes the EDA notebook, reusable feature engineering and preprocessing, leakage-safe cross-validation and Optuna workflows, threshold selection, SHAP explainability, a FastAPI prediction service, a React frontend, data audits, tests, and GitHub Actions. Hosted production deployment and monitoring remain to be configured.

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

This selected dataset does not contain StateNextMonth. The current task is binary classification of the supplied State label; it is not a verified next-month churn predictor. See [target validation](docs/TARGET_VALIDATION.md) for the evidence and provenance limits.

## Current project status

The notebook includes EDA and the original modelling experiment. Shared modules now provide feature engineering, leakage-safe preprocessing pipelines, binary evaluation helpers, target/data audits, and a training-only stratified cross-validation baseline runner. The selected model is chosen by mean cross-validation ROC-AUC; the holdout set is intentionally not scored during selection. The original notebook metrics are historical baseline results, not production-validated performance.

A GitHub Actions workflow (`.github/workflows/ci.yml`) installs Python dependencies, runs unit tests, and executes dataset audits. A second workflow (`.github/workflows/frontend-ci.yml`) builds the React frontend on frontend changes.

See:
- [Project baseline and historical metrics](docs/PROJECT_BASELINE.md)
- [Target validation findings](docs/TARGET_VALIDATION.md)
- [Data quality audit](docs/DATA_QUALITY.md)
- [Structural leakage and feature-timing audit](docs/LEAKAGE_AUDIT.md)
- [Feature engineering and preprocessing](docs/FEATURE_ENGINEERING.md)
- [Imbalance strategy comparison](docs/IMBALANCE_COMPARISON.md)
- [Exploratory CV experiment report](docs/CV_EXPERIMENTS.md)
- [13-step implementation status](docs/ROADMAP_STATUS.md)

## Repository structure

~~~text
Customer-Churn-Prediction/
├── data/
│   └── telco_churn.csv
├── notebooks/
│   └── EDA.ipynb
├── api/
│   ├── __init__.py
│   └── app.py
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       └── styles.css
├── src/
│   ├── __init__.py
│   ├── validate_target.py
│   ├── validate_data.py
│   ├── audit_leakage.py
│   ├── feature_engineering.py
│   ├── preprocess.py
│   ├── evaluate.py
│   ├── train.py
│   ├── optimize.py
│   ├── compare_imbalance.py
│   ├── finalize.py
│   └── explain.py
├── models/               # Saved artifacts (ignored by Git by default)
├── reports/
│   └── figures/
├── tests/
│   ├── test_target_validation.py
│   ├── test_data_validation.py
│   ├── test_leakage_audit.py
│   ├── test_feature_engineering.py
│   ├── test_preprocess.py
│   ├── test_evaluate.py
│   ├── test_train.py
│   ├── test_optimize.py
│   ├── test_imbalance_pipelines.py
│   ├── test_finalize.py
│   ├── test_explain.py
│   └── test_api.py
├── docs/
│   ├── PROJECT_BASELINE.md
│   ├── TARGET_VALIDATION.md
│   ├── DATA_QUALITY.md
│   ├── LEAKAGE_AUDIT.md
│   ├── FEATURE_ENGINEERING.md
│   ├── IMBALANCE_COMPARISON.md
│   ├── CV_EXPERIMENTS.md
│   └── ROADMAP_STATUS.md
├── requirements.txt
├── .github/workflows/
│   ├── ci.yml
│   └── frontend-ci.yml
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
python -m src.audit_leakage
python -m unittest discover -s tests -v
~~~

The target audit reports label distribution and target availability; the data audit checks schema and value quality; the leakage audit verifies known target/identifier fields are excluded from predictors and flags missing temporal provenance. None invents a future target or claims a prediction horizon that the data cannot support.

## Run baseline cross-validated model comparison

From the repository root, run the baseline experiment with the default stratified 5-fold cross-validation:

~~~bash
python -m src.train --output-dir outputs
~~~

The script compares balanced Logistic Regression, balanced Random Forest, and unresampled XGBoost using training-only cross-validation, selects by mean CV ROC-AUC, fits that baseline on the training portion, and deliberately leaves holdout metrics unevaluated. It saves a local JSON report and joblib model under the chosen output directory. The model artifact is ignored by Git. The holdout is reserved for final evaluation after model and threshold selection.

## Compare imbalance strategies

This experiment compares plain Logistic Regression, class weighting, ordinary SMOTE (experimental) and SMOTENC using the same training-only stratified CV folds. All imputing/encoding/sampling is fit only within each CV training fold; the outer holdout remains unscored.

~~~bash
python -m src.compare_imbalance --folds 5 --output-dir outputs
~~~

The report is saved to `outputs/imbalance_comparison.json`. SMOTE-after-one-hot is included only as a measured comparison; SMOTENC keeps categorical features valid through sampling. See [docs/IMBALANCE_COMPARISON.md](docs/IMBALANCE_COMPARISON.md) for the protocol.

## Run Optuna tuning

Optuna searches XGBoost hyperparameters using stratified cross-validation **only on the training split**. Preprocessing is fitted separately within each CV training fold, no synthetic resampling is used, and the holdout test labels are not scored.

~~~bash
python -m src.optimize --trials 30 --folds 5 --output-dir outputs
~~~

The command writes `outputs/optuna_xgboost_cv.json` and `outputs/optuna_xgboost_pipeline.joblib`. These artifacts are local and ignored by Git. The CV score is a tuning estimate, not the final test score. Compare the tuned XGBoost candidate with the cross-validated Logistic Regression baseline before final model selection.

Equivalent top-level command: `python main.py tune --trials 30`.

## Run final model selection, threshold tuning, and holdout evaluation

This is the end-to-end modelling evaluation command. It compares baseline models and tunes XGBoost on a model-training subset using training-only stratified CV. A separate validation split selects the decision threshold. A one-standard-error rule prefers a simpler model when its mean CV score is statistically close to the best candidate. The final pipeline is refit on all development rows, then the untouched holdout is evaluated once using the frozen threshold.

~~~bash
python -m src.finalize --trials 30 --folds 5 --output-dir outputs
~~~

The command writes `outputs/final_evaluation.json` and `outputs/final_pipeline.joblib`. Review the selected model, threshold-validation metrics, final holdout metrics, and confusion counts in the JSON report. The holdout is not used to choose the model or threshold.

Equivalent top-level command: `python main.py finalize --trials 30`.

## Run SHAP explainability

After the final pipeline has been created, generate global mean-absolute SHAP importance, a beeswarm plot, and per-row local explanations from the development partition only:

~~~bash
python -m src.explain --model outputs/final_pipeline.joblib --output-dir outputs/explainability
~~~

This command does not use the holdout rows for explanations. It writes `shap_global_importance.csv`, `shap_local_explanations.csv`, `shap_global_bar.png`, `shap_beeswarm.png`, and a JSON report. SHAP values describe model associations; they do not prove causation.

Equivalent top-level command: `python main.py explain`.

## Run the prediction API

First create a final model and threshold report using the final evaluation command above. Then start the API from the repository root:

~~~bash
uvicorn api.app:app --reload
~~~

Open `http://127.0.0.1:8000/docs` to try the interactive API. Endpoints:

- `GET /health` — checks whether a model artifact is loaded.
- `GET /model-info` — returns model/target information and selected threshold.
- `POST /predict` — predicts the State label for one customer.
- `POST /predict/batch` — predicts up to 100 customers per request.

The API validates categorical values, rejects unknown fields such as `State` and `customerID`, and uses the same inference-time feature-engineering module as model training. If the final model file is not present, the service health endpoint returns a degraded state and predictions return HTTP 503 until `python -m src.finalize --output-dir outputs` has been run.

## Run the React frontend

Use Node.js 22.12 or newer compatible with the pinned Vite release. In a second terminal, with the FastAPI backend running on port 8000:

~~~powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
~~~

The frontend is available at `http://127.0.0.1:5173`. Set `VITE_API_BASE_URL` in `frontend/.env` if the API is running elsewhere. The backend allows local Vite origins by default; configure `CHURN_CORS_ORIGINS` with comma-separated exact origins for another deployment. Do not put API keys or secrets in `VITE_*` variables because frontend variables are bundled into browser code.

The dashboard sends customer inputs to `POST /predict`, displays the returned probability and chosen threshold, and makes no prediction when the API/model is unavailable. It clearly labels the current target as `State` classification, not verified next-month churn.

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
2. Target objective is set to classification of the provided State label, with future-churn claims explicitly excluded. Data-quality audit is implemented; suspicious zero-tenure charges still need a documented modeling experiment.
3. Run and verify the cross-validated baseline in a clean environment; compare class-weighted models against no-resampling alternatives.
4. Run Optuna tuning and compare its cross-validation results with the Logistic Regression baseline; assess class-weighting and categorical-aware imbalance handling. A finalization command now uses a separate threshold-validation split and a one-standard-error model-selection rule.
5. Feature engineering and leakage-safe preprocessing have a shared whitelist/schema and fold-local transformer pipeline; see [FEATURE_ENGINEERING.md](docs/FEATURE_ENGINEERING.md).
6. Use validation predictions to choose the decision threshold and reserve the holdout test for final evaluation. The `src.finalize` orchestration implements this workflow.
7. Add business insights and production-facing inference modules. (SHAP module is implemented.)
8. Harden model/API integration with end-to-end smoke tests, add business insight reporting, configure deployment, and document monitoring/retraining.

## Notes

- Do not interpret association or feature importance as proof of causation.
- Do not select a model or decision threshold using final test results.
- Current results classify the dataset's State label; they do not establish future churn prediction.
