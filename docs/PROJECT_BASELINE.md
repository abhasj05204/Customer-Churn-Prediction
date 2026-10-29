# Project Baseline Snapshot

This document preserves the original modelling results for comparison. They are not a claim that the original validation protocol is production-ready.

## Original repository snapshot

- Reviewed base commit: 783c89b89718b4bb1f27c0384266170b60625963 (Additional changes)
- Original repository CSV: 7,043 rows × 22 columns
- Columns included State and an additional StateNextMonth label.
- Original notebook dropped customerID and StateNextMonth, then predicted State.
- Stratified 80/20 split, random state 42; training set 5,634 rows, test set 1,409 rows.
- Preprocessing was fitted on the full training split; ordinary SMOTE was applied after one-hot encoding.

## Selected dataset revision

The project owner supplied the 21-column dataset and requested it as the source of truth. The repository data file has been replaced with that exact file.

- Current dataset: 7,043 rows × 21 columns.
- SHA-256: 58a435baed4902c1642facdbba81f8faa7c10d7fcec37b9c5f7077aecaf47ffd
- Target column: State
- Labels: 5,174 No Churn and 1,869 Churn.
- Missing cells: 0; duplicate rows: 0; unique customer IDs: 7,043.
- The updated file is byte-for-byte identical to the previous repository CSV after removing the final StateNextMonth column and normalizing line endings. Thus, the original features and State labels used in the saved model experiment are unchanged.
- StateNextMonth is not present in the selected dataset. The project cannot claim to train or validate future/next-month churn from this file alone.

## Recorded metrics for the State classification task

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7360 | 0.5017 | 0.7834 | 0.6117 | 0.8408 |
| Random Forest | 0.7722 | 0.5734 | 0.5535 | 0.5633 | 0.8235 |
| SVM | 0.7523 | 0.5243 | 0.7219 | 0.6074 | 0.8230 |
| Baseline XGBoost | 0.7793 | 0.5827 | 0.5936 | 0.5881 | 0.8376 |
| Optuna-tuned XGBoost | 0.7842 | 0.5946 | 0.5882 | 0.5914 | 0.8395 |

The tuned XGBoost confusion matrix (actual rows: No Churn, Churn; predicted columns: No Churn, Churn):

~~~text
[[885, 150],
 [154, 220]]
~~~

Optuna's best validation ROC-AUC was approximately 0.94296. It should not be treated as an unbiased generalization estimate: the tuning split was created after SMOTE and after preprocessing was fitted on the full training split. The holdout test data was not resampled, but repeated model comparison on its metrics means it has influenced model selection and is no longer a pristine final test.

## Open issues before production modelling

1. Confirm the operational meaning and observation time of State.
2. Future churn prediction requires an actual time-aligned outcome label and features measured before that outcome.
3. Compare class weighting, categorical-aware sampling, and no resampling. Ordinary SMOTE after one-hot encoding may generate fractional categorical indicators.
4. Rebuild evaluation with training-only cross-validation, then reserve the holdout set for the final selected model.

## Implementation status at baseline

At the original snapshot, main.py only loaded and printed the CSV. The reusable training, preprocessing, feature engineering, evaluation, and explanation modules were empty placeholders. Target-audit code and tests have since been added; a production training/inference pipeline remains planned.
