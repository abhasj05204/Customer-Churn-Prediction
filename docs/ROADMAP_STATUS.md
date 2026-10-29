# Implementation roadmap status

This file is the working checklist for the 13-step project plan. **Implemented code, passing automated checks, a completed model run, and a deployed application are different milestones**; statuses below do not treat them as interchangeable.

Legend: **Done** = implementation/evidence available; **Partial** = some work exists but acceptance checks remain; **Pending** = not yet verified as a deliverable.

| # | Planned step | Status | Current evidence | Remaining acceptance check |
|---:|---|---|---|---|
| 01 | Preserve current project | Done | Baseline docs/metrics, Git commit history on `main`, selected data fingerprint, pinned direct Python dependencies, CI and notebook validation | Keep changes on `main`; no separate branch created per working preference |
| 02 | Validate target/business objective | Partial | Current objective documented as classification of `State`; selected 21-column dataset has no `StateNextMonth`; no future-churn claim in API/UI/docs | Exact source file/provenance and temporal semantics are unknown; future/next-month objective requires time-aligned labels |
| 03 | Data quality/leakage audit | Partial | Schema/data audit, target/identifier exclusion, extra-column whitelist, leakage audit/tests, CI execution | Timing leakage cannot be proven without trusted observation/outcome dates; review zero-tenure charge anomaly |
| 04 | Feature engineering/preprocessing | Done for code | Shared training/inference feature contract, fold-local impute/scale/encode pipeline, unit tests and docs | Compare sensitivity to the 11 zero-tenure/high-charge records before release |
| 05 | Baseline model comparison | Partial | Training-only CV runner for Logistic Regression, Random Forest and XGBoost; exploratory CV report | Run committed baseline in the pinned environment and archive JSON output |
| 06 | Robust cross-validation/imbalance | Partial | Stratified CV runners; class weighting, experimental ordinary SMOTE and SMOTENC pipelines; fold-local fit | Run full comparison against selected dataset and record reproducible metrics |
| 07 | Optuna tuning | Partial | Training-only CV tuner and parameter-range tests; exploratory result documented | Run committed tuner with pinned dependencies and save report/model artifact |
| 08 | Final model/threshold | Partial | One-standard-error model-selection rule, separate threshold split, F1 threshold search, and validation-only Brier/log-loss/ECE/reliability-bin diagnostics implemented | Run finalization in the pinned environment and review calibration diagnostics; only add a calibration layer if the validation evidence justifies it |
| 09 | Final test evaluation | Partial | Reporting code includes ROC-AUC, PR-AUC, precision, recall, F1, Brier score and confusion counts | Produce the actual holdout report from a completed final run; do not claim unobserved metrics |
| 10 | Explainability/business insights | Partial | SHAP global/local module, tests and CLI; excludes final holdout from explanation sample | Run with actual fitted model; review explanations and write recommendations supported by the output |
| 11 | Reusable source code | Done for core modules | `train.py`, `preprocess.py`, `feature_engineering.py`, `evaluate.py`, `explain.py`, optimization/finalization modules; notebook delegates feature engineering to shared code | Preserve old notebook cells as labelled historical experiment; continue to use source modules as current path |
| 12 | Tests/reproducibility/docs | Partial | Unit tests, Python CI, frontend build CI, clean notebook JSON/non-empty source checks, docs and command examples | Verify latest CI run on the newest commit; consider frontend lockfile and final clean environment rerun |
| 13 | API/frontend/deployment | Partial | FastAPI routes, React dashboard, CORS config, frontend build CI | Run final model locally, pass end-to-end prediction smoke test, configure actual hosting and production environment variables |

## Current recommended sequence

1. Finish the Step 3 feature-timing limitation using the available data; document that it cannot be inferred without timestamps.
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
