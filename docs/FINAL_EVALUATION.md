# Final evaluation — calibrated pipeline

## Run details

- **Calibration-aware model-validation run:** [GitHub Actions run 40943000126](https://github.com/abhasj05204/Customer-Churn-Prediction/actions/runs/40943000126)
- Commit: `0e6358d3ae5d3364b9855abeb07d8f9c164b6e3a`
- Result: **success** — dependencies installed; target/data/leakage audits, baseline CV, imbalance comparison, 30-trial Optuna, calibration-method comparison, threshold selection and final holdout evaluation all completed.
- Python 3.12.15; NumPy 2.5.3; pandas 3.0.6; SciPy 1.18.1; scikit-learn 1.9.1; imbalanced-learn 0.14.2; XGBoost 3.4.1; Optuna 5.0.0; SHAP 0.52.0; FastAPI 0.162.0.
- Dataset: 7,043 rows × 21 columns; SHA-256 `58a435baed4902c1642facdbba81f8faa7c10d7fcec37b9c5f7077aecaf47ffd`.
- Target: `State` (5,174 No Churn, 1,869 Churn). No `StateNextMonth` or documented future outcome is available.

The workflow stores a report/model artifact named `customer-churn-model-evaluation`; download it from the Actions run page. Artifacts expire 30 days after each run. A subsequent workflow adds an actual saved-model API smoke test and SHAP generation before upload.

## Baseline cross-validation

Five stratified folds, with preprocessing fitted within each fold. Values are mean ± standard deviation.

| Model | ROC-AUC | Average Precision (PR-AUC) | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Balanced Logistic Regression | 0.8460 ± 0.0105 | 0.6687 ± 0.0161 | 0.6255 ± 0.0238 | 0.5183 ± 0.0234 | 0.7895 ± 0.0362 |
| Balanced Random Forest | 0.8275 ± 0.0121 | 0.6363 ± 0.0181 | 0.5648 ± 0.0343 | 0.6649 ± 0.0308 | 0.4924 ± 0.0475 |
| XGBoost (no resampling) | 0.8326 ± 0.0125 | 0.6497 ± 0.0076 | 0.5692 ± 0.0240 | 0.6408 ± 0.0128 | 0.5139 ± 0.0445 |

## Imbalance comparison

Same Logistic Regression settings and same stratified five-fold protocol. All preprocessing/sampling is fitted only within each training fold.

| Strategy | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| No balancing | 0.8459 | 0.6628 | 0.5934 | 0.6654 | 0.5365 |
| Class weight = balanced | 0.8458 | 0.6622 | 0.6282 | 0.5188 | 0.7967 |
| Ordinary SMOTE (experimental) | 0.8442 | 0.6597 | 0.6281 | 0.5244 | 0.7833 |
| SMOTENC | 0.8406 | 0.6508 | 0.6297 | 0.5351 | 0.7652 |

Class weighting raised recall substantially relative to no balancing, with nearly identical ROC-AUC/PR-AUC. SMOTENC gave a marginally higher fold-mean F1 than class weighting but lower recall and lower ranking metrics. Ordinary SMOTE remains an experimental comparison because one-hot interpolation can create fractional categorical flags.

## Model selection

The 30-trial Optuna search achieved **0.8493 ± 0.0110** mean training-only CV ROC-AUC for tuned XGBoost. Baseline balanced Logistic Regression achieved **0.8460 ± 0.0105**. The one-standard-error rule placed both models within its eligibility cutoff, then preferred the simpler Logistic Regression baseline.

- Selected base classifier: Balanced Logistic Regression
- Probability calibration: **sigmoid**, chosen using training-only out-of-fold predictions
- Decision threshold: **0.28**, chosen on a separate threshold-validation split to maximize F1 (ties: recall, then closeness to 0.5)
- Development rows: 5,634; model-training rows: 4,225; threshold-validation rows: 1,409; final holdout rows: 1,409
- The holdout was not used to choose the estimator, calibration method or threshold.

## Calibration comparison

Calibration methods were compared using five-fold out-of-fold predictions on the model-training portion. Sigmoid/isotonic calibration was retained only if both log loss and Brier score improved over raw probabilities.

| Method | OOF Brier score | OOF log loss | OOF ECE | OOF max bin error |
|---|---:|---:|---:|---:|
| No calibration | 0.1645 | 0.4905 | 0.1476 | 0.2919 |
| Sigmoid | **0.1350** | **0.4168** | 0.0162 | 0.0573 |
| Isotonic | 0.1361 | 0.4199 | **0.0107** | **0.0484** |

Sigmoid had the best OOF log loss and Brier score, so it was selected. On the separate threshold-validation split, calibrated probabilities had Brier score 0.1358, log loss 0.4159, ECE 0.0117 and maximum bin error 0.0810. This is a meaningful improvement over the preceding uncalibrated run, whose validation ECE was 0.1524.

## Final untouched holdout — calibrated pipeline

| Metric | Result |
|---|---:|
| Accuracy | 0.7424 |
| Precision | 0.5097 |
| Recall | 0.7754 |
| F1 | 0.6151 |
| ROC-AUC | 0.8421 |
| Average Precision (PR-AUC) | 0.6365 |
| Brier score | **0.1377** |

Confusion matrix (actual rows: No Churn, Churn; predicted columns: No Churn, Churn):

~~~text
[[756, 279],
 [ 84, 290]]
~~~

This corresponds to 279 false positives and 84 false negatives in 1,409 holdout cases. Compared with the first uncalibrated run, ranking metrics are unchanged (sigmoid calibration is monotonic), while probabilities are more consistent with observed frequencies; the selected threshold changed from 0.51 to 0.28.

## Interpretation and limitations

1. The model classifies the dataset's `State` label. It does **not** establish next-month or future churn prediction.
2. Recall is 0.7754 while precision is 0.5097. Many customers flagged as Churn do not belong to the positive class in this holdout; the operating threshold should be considered in a business-cost review.
3. Calibration improves the probability estimates on these validation diagnostics, but calibration and model performance can drift on future data.
4. The zero-tenure / repeated positive TotalCharges anomaly remains flagged and has not been silently repaired.
5. SHAP outputs and saved-model API smoke testing are run in the subsequent integration workflow; their artifacts/status should be checked independently before release.
