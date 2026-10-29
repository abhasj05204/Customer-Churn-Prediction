# Target validation — StateNextMonth

## Current decision

**Training target:** `StateNextMonth` with labels `No Churn` and `Churn`.

The project owner supplied an updated CSV containing this column. The code now treats it as the target and explicitly excludes both `State` and `StateNextMonth` from model predictors. This resolves the code-level target selection issue.

**Important limitation:** the label's provenance is not independently verified. A CSV containing a column named `StateNextMonth` does not prove the labels were measured from actual observations one month after the feature snapshot. The model can be trained to classify the supplied label, but the project must not describe its metrics as validated real-world next-month forecasting until the data source, observation date/cut-off, label generation method, and outcome period are documented.

## Supplied dataset facts

- File expected by training: `data/telco_churn.csv`
- Expected rows: 7,043
- Expected columns: 22
- Required labels: `No Churn`, `Churn`
- Target: `StateNextMonth`
- The supplied file previously inspected has 4,381 `No Churn` and 2,662 `Churn` target labels.
- The supplied file previously inspected has no missing values and no exact duplicate rows.
- Current `State` and `StateNextMonth` are both excluded from predictors to avoid label leakage.

These are structural checks, not proof of target authenticity.

## What evidence is still required

1. Identify the source of the `StateNextMonth` values (original source, owner-generated labels, or synthetic labels).
2. If labels are derived from real records, document the feature snapshot date and the following month's outcome window.
3. Confirm that every feature is available at prediction time and predates the target outcome.
4. Define whether the intended cohort includes only customers active at the snapshot. The current file contains rows where `State` is `Churn` but `StateNextMonth` is `No Churn`; the business meaning of these transitions needs confirmation.
5. After provenance is documented, rerun the full training/evaluation workflow and report metrics as either synthetic-label classification or genuine future-outcome prediction, as appropriate.

## Reproducible checks

From the repository root:

~~~bash
python -m src.validate_target
python -m src.validate_data
python -m src.audit_leakage
python -m unittest discover -s tests -v
~~~

The audit validates schema and label values. It cannot infer or certify label provenance from column names alone.
