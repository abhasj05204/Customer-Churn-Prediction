# Target validation — Step 2

## Dataset selected for this project

The newly selected dataset is the 21-column version uploaded by the project owner. Its SHA-256 is:

~~~text
58a435baed4902c1642facdbba81f8faa7c10d7fcec37b9c5f7077aecaf47ffd
~~~

It contains 7,043 rows and 21 columns. It has the label column State and does **not** contain StateNextMonth. A byte-for-byte check confirmed that this file is exactly the previously committed 22-column CSV with the final StateNextMonth column removed and line endings normalized. The source data rows and State labels are unchanged.

## Observed data checks

- Rows: 7,043
- Columns: 21
- Unique customer IDs: 7,043
- Duplicate rows: 0
- Missing cells: 0
- State labels: 5,174 No Churn and 1,869 Churn
- Observed Churn share: 26.54%
- Tenure range: 0–72
- Tenure equal to zero: 11 rows

The uploaded file has no missing data in its current representation, including TotalCharges, which is numeric. These checks describe this file; they do not prove the dataset is free of every possible data-quality issue.

## What the current notebook predicts

The notebook maps State labels to 0/1 and trains models to classify State. The recorded historical model metrics therefore apply to the State classification task.

## What cannot be claimed from this file

This selected dataset does not have a StateNextMonth target, an explicit observation date, or documented future-outcome labels. We cannot claim that a model trained only on this file predicts churn in the next month. The previous repository version contained an extra StateNextMonth column, but that column has been removed from the selected source dataset and must not be silently used as ground truth.

The chosen target for the current data is State. To develop a genuine forward-looking churn model in the future, we would need appropriately time-aligned customer features and outcomes, together with reliable documentation/provenance. Column names alone are not evidence of prediction horizon.

## Reproducible commands

From the repository root:

~~~bash
python -m src.validate_target
python -m unittest discover -s tests -v
~~~
