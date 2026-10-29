# Structural leakage and feature-timing audit — Step 3

## Selected source snapshot

- File: `data/telco_churn.csv`
- SHA-256: `58a435baed4902c1642facdbba81f8faa7c10d7fcec37b9c5f7077aecaf47ffd`
- Rows × columns: 7,043 × 21
- Target: `State` with labels `No Churn` and `Churn`
- `StateNextMonth`: absent

## Structural checks implemented

The shared `engineer_inference_features` function is used to derive the actual feature frame. The audit verifies that these known identifier/target-like fields do not appear as predictor columns:

- `customerID`
- `State`
- `StateNextMonth`
- `Churn` / `ChurnLabel`
- generic `target` / `label` columns

The selected input schema contains the expected customer demographic, service, contract, payment and charge fields. Tests also supply a synthetic `StateNextMonth` column and confirm it is excluded from the predictor frame.

## Timing cannot be proven from this CSV

The dataset has no trusted observation cut-off, source event timestamp, churn event date, or outcome date. Because of this, the code reports feature-timing status as **not verifiable** rather than declaring that temporal leakage has been ruled out.

This matters especially for fields such as `TotalCharges` and `tenure`: without documentation of when they were measured relative to the label, static schema checks cannot prove they were available before a real future churn event. The current supported objective is therefore classification of the supplied `State` label—not a verified next-month forecast.

## Result and commands

The structural audit currently passes for the selected schema: known identifiers and target-like fields are excluded from the engineered predictor frame. Temporal leakage remains an explicitly documented data-provenance limitation.

~~~bash
python -m src.audit_leakage
python main.py audit
python -m unittest discover -s tests -v
~~~

The audit is run in GitHub Actions after the target and data-quality checks. It fails loudly for missing required raw predictors or invalid/missing target labels, but it does not attempt to infer time semantics or silently modify data.
