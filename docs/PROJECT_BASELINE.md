# Project Baseline Snapshot

This file records the repository state before target validation and robust optimization. Historical metrics are preserved for comparison; they are not a claim that the existing validation protocol is production-ready.

## Repository snapshot

- Branch: `main`
- Reviewed base commit: `783c89b89718b4bb1f27c0384266170b60625963` (`Additional changes`)
- Dataset dimensions: 7,043 rows × 22 columns
- Current-state label: `State` — 5,174 `No Churn`, 1,869 `Churn` (26.54% churn)
- Additional label: `StateNextMonth` — 4,381 `No Churn`, 2,662 `Churn` across all rows
- Current notebook drops `customerID` and `StateNextMonth`, then predicts `State`.
- Recorded split: stratified 80/20, random state 42; training set 5,634 rows, test set 1,409 rows.
- The preprocessor is fitted on the complete training split; ordinary SMOTE is then applied after one-hot encoding.

## Recorded metrics for the current `State` classification task

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7360 | 0.5017 | 0.7834 | 0.6117 | 0.8408 |
| Random Forest | 0.7722 | 0.5734 | 0.5535 | 0.5633 | 0.8235 |
| SVM | 0.7523 | 0.5243 | 0.7219 | 0.6074 | 0.8230 |
| Baseline XGBoost | 0.7793 | 0.5827 | 0.5936 | 0.5881 | 0.8376 |
| Optuna-tuned XGBoost | 0.7842 | 0.5946 | 0.5882 | 0.5914 | 0.8395 |

The tuned XGBoost confusion matrix (actual rows: No Churn, Churn; predicted columns: No Churn, Churn):

```text
[[885, 150],
 [154, 220]]
```

Optuna's best validation ROC-AUC was approximately 0.94296. This should not be considered an unbiased generalization estimate: the tuning split was created after SMOTE and after preprocessing had been fitted on the complete training split. The holdout set was not resampled, but repeated model comparison on its metrics means it has influenced model selection and should not be described as a pristine final test.

## Decisions to resolve before retraining

1. Confirm the definitions and time alignment of `State` and `StateNextMonth`.
2. Decide whether the goal is current-state classification or next-month churn prediction for customers still active.
3. Confirm which input features are available at prediction time.
4. Verify the label provenance and prediction horizon.
5. Compare class weighting, categorical-aware resampling, and no-resampling. Ordinary SMOTE after one-hot encoding can generate fractional categorical indicators.

## Implementation status at baseline

`main.py` currently loads and prints the CSV. `src/train.py`, `src/preprocess.py`, `src/feature_engineering.py`, `src/evaluate.py`, and `src/explain.py` are empty placeholders. Automated tests have not yet been implemented.
