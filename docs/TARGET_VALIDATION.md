# Target validation — Step 2

## Decision for this repository

**Chosen task:** binary classification of the supplied `State` label (`No Churn` vs `Churn`).

**Not claimed:** predicting whether a currently active customer will churn next month. The selected dataset does not contain `StateNextMonth`, observation timestamps, or documented future-outcome labels. A temporal prediction horizon cannot be inferred from the column names alone.

The project owner's selected CSV is the source of truth for this repository. Exact provenance of the Kaggle page/version was not recalled, so external schema comparisons below are contextual, not proof that this file is byte-for-byte the original Kaggle download.

## Selected dataset fingerprint

- File: `data/telco_churn.csv`
- SHA-256: `58a435baed4902c1642facdbba81f8faa7c10d7fcec37b9c5f7077aecaf47ffd`
- Rows: 7,043; columns: 21
- Target: `State`
- Labels: 5,174 `No Churn`, 1,869 `Churn`
- Churn label share: 26.54%
- `StateNextMonth`: absent

The selected file is byte-for-byte identical to the previously committed 22-column repository CSV after removing the final `StateNextMonth` column and normalizing line endings. The feature columns and `State` labels used by the original notebook are unchanged.

## Target semantics: evidence and limits

The values in `State` are unambiguously the supplied binary label for this project because the file contains only `No Churn` and `Churn`, and the historical notebook maps those values to 0 and 1. The dataset alone does **not** establish the exact observation time, nor prove that the features were captured before a future churn event.

A commonly distributed reference version of the IBM Telco Customer Churn dataset is published at [Kaggle: Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Reference examples for that dataset use 7,043 rows, the familiar customer/service/billing fields, and a `Churn` label. This is a plausible schema comparison, not an assertion that the project owner's exact Kaggle download has been identified. In the selected file, the label is named `State`, its values are `No Churn`/`Churn`, and `TotalCharges` is already numeric with 11 zero-tenure rows sharing the same positive charge value. Those transformations need provenance confirmation if available.

## Model and product wording

Use descriptions such as:
- “Classifies the provided customer `State` label.”
- “Outputs a model probability for the `Churn` class within this dataset.”

Do not use descriptions such as:
- “Predicts next-month churn.”
- “Forecasts future cancellation.”
- “Identifies customers who will definitely churn.”

A verified forward-looking project would require a documented observation date/cut-off, features available by that date, and an outcome label measured after that cut-off. The old `StateNextMonth` column was removed from the selected data and must not be silently used as ground truth.

## Reproducible checks

From the repository root:

~~~bash
python -m src.validate_target
python -m src.validate_data
python -m unittest discover -s tests -v
~~~

The audit reports label counts and schema facts; it does not manufacture missing future labels or infer a temporal horizon.
