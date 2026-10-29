# Implementation roadmap status

This file is the working checklist for the 13-step project plan. **Implemented code, passing automated checks, a completed model run, and a deployed application are different milestones**; statuses below do not treat them as interchangeable.

Legend: **Done** = implementation/evidence available; **Partial** = some work exists but acceptance checks remain; **Pending** = not yet verified as a deliverable.

| # | Planned step | Status | Current evidence | Remaining acceptance check |
|---:|---|---|---|---|
| 01 | Preserve current project | Done | Baseline docs/metrics, Git commit history on `main`, selected data fingerprint, pinned direct Python dependencies, CI and notebook validation | Keep changes on `main`; no separate branch created per working preference |
| 02 | Validate target/business objective | Partial | Training code and API now target the supplied `StateNextMonth` label; provenance and temporal alignment remain unverified | Document source, label-generation method, observation cut-off and future outcome window before claiming genuine next-month prediction |
| 03 | Data quality/leakage audit | Partial | Schema/data audit, target/identifier exclusion, extra-column whitelist, leakage audit/tests, CI execution | Timing leakage cannot be proven without trusted observation/outcome dates; review zero-tenure charge anomaly |
| 04 | Feature engineering/preprocessing | Done for code | Shared training/inference feature contract, fold-local impute/scale/encode pipeline, unit tests and docs | Compare sensitivity to the 11 zero-tenure/high-charge records before release |
| 05 | Baseline model comparison | Done | Pinned Python 3.12.15 five-fold baseline CV ran successfully for Logistic Regression, Random Forest and XGBoost; report is in FINAL_EVALUATION.md and the Actions artifact | Repeat only when code/data/modeling assumptions change; use holdout only at Step 9 |
| 06 | Robust cross-validation/imbalance | Done | Pinned five-fold comparison of no balancing, class weighting, ordinary SMOTE (experimental) and SMOTENC completed; scores and caveats are documented | Revisit if the feature/data contract changes |
| 07 | Optuna tuning | Done | Thirty-trial Optuna tuning completed on training-only five-fold CV in the pinned environment; tuned XGBoost mean ROC-AUC 0.8493 ± 0.0110 | Retune only within a defined training-only search protocol |
| 08 | Final model/threshold | Done | One-standard-error rule selected balanced Logistic Regression; training-only OOF calibration comparison chose sigmoid (better log loss and Brier); F1 threshold 0.28 selected on separate validation data | Calibration/threshold can be revisited only with new validation data or changed business objective |
| 09 | Final test evaluation | Done | Untouched 1,409-row holdout evaluated once; ROC-AUC 0.8421, PR-AUC 0.6365, F1 0.6151, calibrated Brier 0.1377; report and confusion matrix are documented | Do not use this holdout to tune anything further |
| 10 | Explainability/business insights | Partial | SHAP global/local module, tests and CLI; excludes final holdout from explanation sample | Run with actual fitted model; review explanations and write recommendations supported by the output |
| 11 | Reusable source code | Done for core modules | `train.py`, `preprocess.py`, `feature_engineering.py`, `evaluate.py`, `explain.py`, optimization/finalization modules; notebook delegates feature engineering to shared code | Preserve old notebook cells as labelled historical experiment; continue to use source modules as current path |
| 12 | Tests/reproducibility/docs | Partial | Python test suite passed after calibration-selection test; pinned model-validation workflow succeeded; frontend build workflow succeeded on frontend code; all .py modules contain code | Verify latest smoke/SHAP workflow and add npm lockfile for transitive frontend dependency reproducibility |
| 13 | API/frontend/deployment | Partial | FastAPI, React dashboard and CORS config implemented; trained-artifact API smoke test added to model-validation workflow and is running; frontend build CI is green | Confirm smoke test and SHAP artifacts, then configure actual hosting and production environment variables |

## Current recommended sequence

1. Verify the source and time alignment of StateNextMonth labels; document that the CSV alone cannot prove they are observed future outcomes.
2. Run the baseline and fair imbalance comparison using the committed commands.
3. Tune/choose the model and threshold, assess calibration, then produce the final untouched holdout report.
4. Generate SHAP/business insights with the final model.
5. Run the model through the API and browser frontend; add deployment configuration only after the smoke test passes.

## Reproducibility commands

~~~bash
python -m src.validate_target
python -m src.validate_data
python -m src.audit_leakage
python -m unittest discover -s tests -v
python -m src.train --output-dir outputs
python -m src.compare_imbalance --folds 5 --output-dir outputs
python -m src.optimize --trials 30 --folds 5 --output-dir outputs
python -m src.finalize --trials 30 --folds 5 --output-dir outputs
python -m src.explain --model outputs/final_pipeline.joblib --output-dir outputs/explainability
~~~
