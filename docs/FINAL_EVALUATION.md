# Pinned-environment model evaluation

## Run details

- GitHub Actions run: [Model validation and evaluation — run 40941987432](https://github.com/abhasj05204/Customer-Churn-Prediction/actions/runs/40941987432)
- Commit: `2d5d59358fe7c6975568b4ef60745cddb9638bcd`
- Result: succeeded; all steps completed, including dependency install, data audits, baseline CV, imbalance comparison, Optuna tuning, final holdout evaluation, and artifact upload.
- Artifact: `customer-churn-model-evaluation` (report JSONs and a fitted pipeline); the GitHub Actions artifact expires on **November 28, 2026**.
- Python: 3.12.15
- NumPy 2.5.3; pandas 3.0.6; SciPy 1.18.1; scikit-learn 1.9.1; imbalanced-learn 0.14.2; XGBoost 3.4.1; Optuna 5.0.0; SHAP 0.52.0; FastAPI 0.162.0.
- Dataset: 7,043 rows × 21 columns, SHA-256 `58a435baed4902c1642facdbba81f8faa7c10d7fcec37b9c5f7077aecaf47ffd`.
- Target: `State`, not a verified future/next-month outcome.

> This is the first complete pinned-environment run before training-only OOF calibration-method selection was added. The updated finalization workflow is rerunning calibration comparison separately; its results will supersede the uncalibrated probability threshold/model artifact if the method comparison selects calibration.

## Baseline cross-validation

Metrics are mean ± standard deviation across five folds on the baseline experiment's training partition.

| Model | ROC-AUC | Average Precision (PR-AUC) | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Balanced Logistic Regression | 0.8458 ± 0.0125 | 0.6622 ± 0.0140 | 0.6282 ± 0.0227 | 0.5183 ± 0.0234 | 0.7967 ± 0.0382 |
| Balanced Random Forest | 0.8263 ± 0.0111 | 0.6216 ± 0.0283 | 0.5374 ± 0.0241 | 0.6382 ± 0.0315 | 0.4642 ± 0.0215 |
| XGBoost (no resampling) | 0.8423 ± 0.0112 | 0.6521 ± 0.0256 | 0.5788 ± 0.0235 | 0.6568 ± 0.0337 | 0.5184 ± 0.0303 |

## Imbalance comparison

Same Logistic Regression settings, same stratified five-fold protocol, preprocessing and sampling fitted only inside each training fold.

| Strategy | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| No balancing | 0.8459 | 0.6628 | 0.5934 | 0.6654 | 0.5365 |
| Class weight = balanced | 0.8458 | 0.6622 | 0.6282 | 0.5188 | 0.7967 |
| Ordinary SMOTE (experimental) | 0.8442 | 0.6597 | 0.6281 | 0.5244 | 0.7833 |
| SMOTENC | 0.8406 | 0.6508 | 0.6297 | 0.5351 | 0.7652 |

In this run, class weighting gave a larger recall increase than no balancing, with similar ranking metrics. SMOTENC had a slightly higher fold-mean F1 but lower ROC-AUC/PR-AUC and recall than class weighting. Ordinary SMOTE is retained only as a comparison because post-one-hot interpolation can create fractional categorical flags.

## Model selection and threshold

The 30-trial Optuna study found tuned XGBoost mean training-only CV ROC-AUC of **0.8493 ± 0.0110**. The one-standard-error rule included both tuned XGBoost and balanced Logistic Regression, then preferred the simpler Logistic Regression baseline.

- Selected model: Balanced Logistic Regression
- Threshold selected on a separate validation split for F1: **0.51**
- Validation F1: 0.6391
- Validation precision: 0.5299
- Validation recall: 0.8048
- Holdout rows: 1,409; not used for model or threshold selection

## Final untouched holdout — first uncalibrated run

| Metric | Result |
|---|---:|
| Accuracy | 0.7395 |
| Precision | 0.5060 |
| Recall | 0.7834 |
| F1 | 0.6149 |
| ROC-AUC | 0.8421 |
| Average Precision (PR-AUC) | 0.6365 |
| Brier score | 0.1672 |

Confusion matrix (actual rows: No Churn, Churn; predicted columns: No Churn, Churn):

~~~text
[[749, 286],
 [ 81, 293]]
~~~

This represents 286 false positives and 81 false negatives in the holdout. The class decision threshold is chosen for F1, not because it represents a verified business cost optimum.

## Calibration diagnostic

On the 1,409-row threshold-validation split, the uncalibrated probabilities had:

- Brier score: 0.1658
- Log loss: 0.4937
- Expected calibration error (10 equal-width bins): 0.1524
- Maximum bin calibration error: 0.2678

The reliability bins showed predicted probabilities generally above observed positive rates. That is evidence to compare calibrated probability methods before treating individual probabilities as reliable risk estimates. This diagnostic uses validation data only; the holdout was not used to assess or select calibration.

## Interpretation and limitations

1. The model classifies `State`; the data do not establish that a probability means next-month churn.
2. Recall is relatively high at threshold 0.51, but precision is about 0.51: many flagged customers do not belong to the positive class in this dataset.
3. Ranking performance and calibration are distinct. ROC-AUC of 0.8421 does not mean an output probability such as 0.80 corresponds to an 80% event frequency.
4. Use these metrics as the first reproducible release baseline, not a production performance guarantee.
5. SHAP explanations, end-to-end API smoke tests, calibrated final artifact verification, and deployment remain separate acceptance checks.
