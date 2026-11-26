# Target validation — recorded State

## Supported target

**Training target:** `State`, with labels `No Churn` and `Churn`.

The committed dataset contains 7,043 rows and 21 columns. It includes `State` and does not include `StateNextMonth`. The CSV therefore supports classification of its recorded churn-status label only.

## Future-label limitation

A previous development iteration explored a `StateNextMonth` column. Its provenance and temporal alignment were not established, and it is absent from the selected committed dataset. The current pipeline must not require, synthesize, infer, or train against that field.

If a future dataset supplies `StateNextMonth`, the audit may report its presence as optional metadata, but it remains excluded from predictors and is not selected as the training target unless an independently documented business/data contract establishes the source snapshot, observation cut-off, label-generation method, and outcome window.

## Predictor contract

- Target: `State`
- Identifier excluded: `customerID`
- Optional future label excluded: `StateNextMonth`
- Raw predictors: the 19 customer profile, service, contract and billing fields defined in `src/feature_engineering.py`
- Encoded labels: `No Churn → 0`, `Churn → 1`

## Reproducible checks

```bash
python -m src.validate_target
python -m src.validate_data
python -m src.audit_leakage
python -m unittest discover -s tests -v
```

These checks validate schema and label values. They cannot prove temporal availability or future prediction semantics from a static CSV.
