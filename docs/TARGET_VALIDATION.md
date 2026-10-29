# Target validation — Step 2

## Purpose

This audit documents the observed relationship between State and StateNextMonth. It does **not** independently establish what either label means, when features were measured, or how the next-month label was created. The repository currently has no data dictionary or label-provenance notes, so the production target must not be changed based on column names alone.

## Current notebook behavior

The current notebook copies the dataset, drops customerID and StateNextMonth, and maps State to 0/1. Thus, the saved Phase 3/4 model results are for classifying State, not for predicting StateNextMonth.

## Observed dataset audit

The checked CSV has 7,043 rows and 22 columns. Its saved label counts are:

- State: 5,174 No Churn, 1,869 Churn (26.54% churn).
- StateNextMonth across all rows: 4,381 No Churn, 2,662 Churn.

Observed cross-tabulation (rows are State; columns are StateNextMonth):

| State | Next month No Churn | Next month Churn |
|---|---:|---:|
| No Churn | 4,218 | 956 |
| Churn | 163 | 1,706 |

If—and only if—State == "No Churn" means a customer is active at the prediction time and StateNextMonth really is the subsequent outcome, the candidate active cohort contains 5,174 rows: 956 have a Churn next-month label and 4,218 have a No Churn next-month label. The observed rate in that cohort is 956 / 5,174 = 18.48%.

The full-dataset StateNextMonth rate (2,662 / 7,043) should not be used as the active-customer next-month churn rate because that denominator also contains rows whose State is Churn.

## Candidate objectives

1. **Current-state classification:** predict State from features that are legitimately available at the same prediction time. This is the task represented by the existing notebook, but it may be operationally unhelpful if customers are already churned when the prediction is made.
2. **Next-month churn prediction:** restrict modelling to customers active at prediction time (candidate filter: State == "No Churn") and use StateNextMonth as the target, but only after verifying the dataset's label definitions, timestamp alignment, and generation procedure.

## Required verification before model training

Confirm the source/data dictionary or dataset-generation code for these points:

- Does State mean current customer status at time t?
- Does StateNextMonth mean the same customer's status at t+1 month?
- Were all model input features measured at or before time t, before the target outcome?
- Are rows one customer each or repeated customer-period observations? If repeated, use a time/customer-aware split as appropriate.
- Do churn labels represent actual churn, synthetic transitions, or a derived label?

Until those points are confirmed, neither target should be described as a verified future-churn target.

## Reproducible command

From the repository root:

~~~bash
python -m src.validate_target
~~~

Run the focused unit tests:

~~~bash
python -m unittest discover -s tests -v
~~~
