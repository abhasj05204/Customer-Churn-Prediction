# Feature engineering and preprocessing — Step 4

## Raw input contract

The single declared schema contract lives in `src/feature_engineering.py` as `REQUIRED_INPUT_COLUMNS` / `RAW_FEATURE_COLUMNS`. Training and inference use the same raw feature whitelist. `customerID`, `State`, `StateNextMonth`, and arbitrary unreviewed extra columns cannot enter the predictor frame.

The selected dataset has 7,043 rows, 21 source columns, 19 raw predictor fields and the binary target `State`.

## Engineered features

| Feature | Definition | Handling |
|---|---|---|
| `TenureGroup` | Bins `tenure` into 0–1, 1–2, 2–4, 4–5 and 5+ year bands | Created row-wise; no dataset-wide fitting |
| `AvgMonthlyCharges` | `TotalCharges / tenure` | For tenure 0, denominator is treated as undefined and this derived feature falls back to `MonthlyCharges` |
| `ServiceCount` | Number of exact `Yes` values across phone, multiple lines, security, backup, protection, support and streaming indicators | Derived from the shared feature definition |
| `SecuritySupportCount` | Number of `Yes` values across OnlineSecurity, OnlineBackup, DeviceProtection and TechSupport | Derived from the shared feature definition |

The target mapping is `No Churn → 0` and `Churn → 1`. Feature generation does not mutate the input frame. Invalid/missing target labels and non-numeric values in the three numeric charge/tenure fields raise explicit errors.

## Preprocessing

`src/preprocess.py` builds a single scikit-learn pipeline:

- Numeric fields: median imputation, then `StandardScaler`.
- Categorical fields: most-frequent imputation, then `OneHotEncoder(handle_unknown="ignore")`.
- The fitted transformer stays inside the model pipeline. During cross-validation, scikit-learn clones and fits it independently on each training fold; validation folds are transform-only.

No global scaler or encoder is fitted before cross-validation. The current baseline does not apply SMOTE. That is intentional while the categorical-aware imbalance alternatives are evaluated.

## Data-quality caveat

The data audit flags 11 rows where `tenure == 0` and `TotalCharges == 1397.475`. The feature module does not overwrite the source value. `AvgMonthlyCharges` falls back to `MonthlyCharges` for those rows, while the raw `TotalCharges` remains as recorded. A fold-local comparison of retaining/excluding that feature is still required before a production model is selected.

## Verification commands

~~~bash
python -m src.validate_data
python -m src.audit_leakage
python -m unittest discover -s tests -v
~~~

The current pipeline contract is unit tested for input immutability, target/identifier exclusion, optional target-free inference, unknown-column exclusion, numeric errors, preprocessing fit behavior, and unknown categorical values.
