# Exploratory cross-validation experiments

**Status:** exploratory engineering run, not the final model evaluation.

## Protocol

- Dataset SHA-256: `58a435baed4902c1642facdbba81f8faa7c10d7fcec37b9c5f7077aecaf47ffd`
- Target: `State` (No Churn = 0, Churn = 1)
- Split: stratified 80/20 train/holdout, random state 42.
- Model selection: stratified 5-fold CV on the training portion only.
- Preprocessing: numeric imputation/scaling and categorical imputation/one-hot encoding fitted inside each fold.
- Imbalance approach: class weighting for Logistic Regression and Random Forest; no synthetic resampling for XGBoost.
- Holdout metrics: **not evaluated**.
- Auxiliary development runtime: Python 3.13.5, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0, XGBoost 3.1.3, Optuna 4.8.0. These differ from the repository's pinned dependency versions, so rerun the committed commands in the project environment before treating the numbers as a repeatable release result.

Metrics below are mean ± standard deviation across five folds.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average Precision (PR-AUC) |
|---|---:|---:|---:|---:|---:|---:|
| Balanced Logistic Regression | 0.7499 ± 0.0145 | 0.5188 ± 0.0183 | 0.7967 ± 0.0382 | 0.6282 ± 0.0227 | **0.8458 ± 0.0125** | **0.6622 ± 0.0140** |
| Balanced Random Forest | 0.7891 ± 0.0133 | 0.6400 ± 0.0355 | 0.4702 ± 0.0282 | 0.5419 ± 0.0291 | 0.8268 ± 0.0120 | 0.6197 ± 0.0315 |
| XGBoost (no resampling) | **0.7987 ± 0.0105** | **0.6497 ± 0.0316** | 0.5271 ± 0.0272 | 0.5814 ± 0.0208 | 0.8422 ± 0.0095 | 0.6563 ± 0.0210 |
| XGBoost, Optuna (30 exploratory trials) | — | — | — | — | **0.8502 ± 0.0130** | 0.6701 |

The XGBoost Optuna result is from an exploratory 30-trial run over the same parameter ranges subsequently committed to `src/optimize.py`. It is not from the final clean, pinned environment. The improvement over Logistic Regression is small relative to fold variability and does not establish that XGBoost is robustly better.

## Interpretation

- On the initial baseline experiment, balanced Logistic Regression had the highest mean CV ROC-AUC and Average Precision. It remains an important benchmark.
- XGBoost had higher fold-mean accuracy and precision, but lower recall than Logistic Regression at the default 0.5 threshold. Threshold selection must use validation/OOF predictions and business costs—not the holdout test.
- Thirty exploratory Optuna trials produced a modest ROC-AUC increase over the Logistic Regression CV mean. The gap is small relative to fold variability, so this does not yet justify declaring XGBoost the winner.
- No holdout metric has been used for these choices. The fixed holdout should be evaluated only after model and decision threshold are chosen.

## Reproduce in the repository environment

~~~bash
python -m src.train --output-dir outputs
python -m src.optimize --trials 30 --folds 5 --output-dir outputs
~~~

Compare the committed JSON reports produced by those commands. Preserve the holdout for the final evaluation step.
