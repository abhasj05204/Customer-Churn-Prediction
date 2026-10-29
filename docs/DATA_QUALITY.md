# Data quality audit — Step 3

## Dataset fingerprint

- File: data/telco_churn.csv
- Dimensions: 7,043 rows × 21 columns
- SHA-256: 58a435baed4902c1642facdbba81f8faa7c10d7fcec37b9c5f7077aecaf47ffd
- Target: State
- Missing cells: 0
- Exact duplicate rows: 0
- Unique customer IDs: 7,043

## Checks that passed on the selected file

- All expected 21 columns are present in the expected order.
- State labels are No Churn and Churn only.
- No missing cell or exact duplicate row was found.
- customerID is unique for all 7,043 rows.
- tenure ranges from 0 to 72.
- MonthlyCharges ranges from 18.25 to 118.75.
- TotalCharges is numeric and ranges from 18.8 to 8,684.8.
- No negative value was found in the checked numeric columns.

## Anomaly requiring review

There are 11 rows where tenure is zero. All 11 have TotalCharges = 1,397.475, despite their monthly charges varying from approximately 19.70 to 80.85. The repeated value equals the median TotalCharges in this file.

This pattern is suspicious and may indicate the source data was pre-imputed, but this file alone cannot prove how or why. The audit therefore **flags but does not alter or drop these rows**. Before making modeling decisions, document the dataset source/cleaning process if possible and compare model performance with a transparent, training-fold-only treatment of this feature.

The current feature-engineering cell calculates AvgMonthlyCharges using TotalCharges / tenure and fills undefined values with MonthlyCharges. For tenure-zero rows this means AvgMonthlyCharges becomes MonthlyCharges, but the raw TotalCharges feature itself still retains the repeated value. We will test feature handling during model validation rather than silently changing the selected source file.

## Reproduce

From the repository root:

~~~bash
python -m src.validate_data
python -m unittest discover -s tests -v
~~~

The script is descriptive. It reports anomalies but does not automatically impute, remove, or modify rows.
