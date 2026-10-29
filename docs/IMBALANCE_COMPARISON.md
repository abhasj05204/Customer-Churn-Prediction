# Imbalance strategy comparison — Step 6

## Purpose

Compare imbalance handling without changing the estimator or data folds between strategies. All candidates use Logistic Regression with the same main hyperparameter settings and stratified CV on the development partition.

## Strategies implemented

1. **No balancing:** fold-local imputation, scaling and one-hot encoding, with plain Logistic Regression.
2. **Class weighting:** same feature pipeline and estimator, with `class_weight="balanced"`.
3. **Ordinary SMOTE (experimental):** resampling inside each training fold after preprocessing/one-hot encoding. This is included to make the limitation measurable; it can create fractional category flags and should not be treated as the default.
4. **SMOTENC:** numeric values are imputed/scaled, categorical values remain unencoded during sampling, then one-hot encoding is fitted after sampling. This preserves valid categories and all operations are fold-local.

Each fold is stratified and the validation fold is not resampled. Metrics include accuracy, precision, recall, F1, ROC-AUC and Average Precision (PR-AUC). The holdout partition is created but never scored by this experiment.

## Run

From the repository root:

~~~bash
python -m src.compare_imbalance --folds 5 --output-dir outputs
~~~

Equivalent CLI:

~~~bash
python main.py imbalance --folds 5 --output-dir outputs
~~~

This writes `outputs/imbalance_comparison.json`. Compare the CV means and fold variability before selecting an approach. The report is an internal model-selection aid, not the final holdout evaluation. The current selected target is `State`, not a verified next-month outcome.
