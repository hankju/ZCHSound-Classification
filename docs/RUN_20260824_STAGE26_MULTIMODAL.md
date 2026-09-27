# Stage26 Patient-Safe Age/Sex Multimodal Fusion

Status: completed; calibration improved and minority metrics showed a positive
but not statistically conclusive trend.

## Objective

Test whether age and sex add classification or calibration value to the fixed
Stage25 raw curriculum acoustic predictor. The design directly avoids the old
cross-fold patient leakage: a fold's demographic model may use only that fold's
train and validation patients and can never use a patient in that fold's test
set.

## Fixed design

- Acoustic baseline: the fixed two-seed AST-balanced plus BEATs-conservative
  raw curriculum pair on split seed 20268020.
- Metadata: complete `age_days` and `sex01` for all 941 patients.
- Demographic model: unweighted multinomial Logistic Regression.
- Logistic `C` is selected inside train patients only by repeated 5-fold CV.
- Candidate corrections use age or age+sex, strengths 0.10/0.25/0.50, fixed
  or entropy-adaptive weights, and either all-class Bayesian prior correction
  or abnormal-conditional correction that preserves acoustic `P(NORMAL)`.
- The same-fold validation set applies fixed Accuracy, balanced-F1,
  calibration, normal-recall, and NLL guardrails. Acoustic-only is the fallback.
- After the selection lock is serialized, the selected demographic model is
  refit on train+validation and evaluated once on the disjoint test patients.

The machine-readable pre-selection protocol is
`configs/stage26_multimodal_protocol.json`.

## Preflight

- Python compilation passed.
- Full/conditional and fixed/entropy correction synthetic checks passed.
- Validation view contains 20 validation exports and zero test exports.
- Test view is stored in a separate directory.
- Metadata is complete: 941/941 age values and 941/941 sex values.
- All source, metadata, and split hashes were recorded before selection.

## Results

The validation-only selector locked age-only entropy-adaptive corrections in
folds 0 and 1, and age+sex fixed corrections in folds 2 through 4. Folds 0-3
used the Accuracy branch and fold 4 used the Calibration branch. All selected
Logistic Regression models used `C=0.01`. The selection-lock canonical SHA256
is `f942464ff77cf1e40d0de86c45e6cd6f38281a497d4e466868ca81fda8dcb9de`.

The one-time held-out result was:

| Method | Accuracy | Macro F1 | Minority recall | Minority F1 | NORMAL recall | NLL |
|---|---:|---:|---:|---:|---:|---:|
| Acoustic curriculum | 0.831031 | 0.649904 | 0.441194 | 0.470447 | 0.971857 | 0.545301 |
| Age/sex multimodal | 0.829968 | 0.654787 | 0.460242 | 0.481423 | 0.964353 | 0.536726 |
| Delta | -0.001063 | +0.004883 | +0.019048 | +0.010977 | -0.007505 | -0.008574 |

The multimodal layer lost one net correct patient, while improving macro F1 by
0.488 percentage points, minority recall by 1.905 points, minority F1 by 1.098
points, and NLL by 0.00857. It changed 25 predictions: seven became uniquely
correct, eight became uniquely incorrect, and ten changed between two wrong
classes. Exact McNemar testing for Accuracy gave `p=1.0`.

The clearest class-specific change was PFO:

| Class | Acoustic recall | Multimodal recall | Acoustic F1 | Multimodal F1 |
|---|---:|---:|---:|---:|
| ASD | 0.436975 | 0.436975 | 0.488263 | 0.492891 |
| NORMAL | 0.971857 | 0.964353 | 0.940109 | 0.937101 |
| PDA | 0.343750 | 0.343750 | 0.423077 | 0.423077 |
| PFO | 0.542857 | 0.600000 | 0.500000 | 0.528302 |
| VSD | 0.871658 | 0.866310 | 0.898072 | 0.892562 |

A fixed 20,000-repeat paired class-stratified patient bootstrap produced:

| Delta | Estimate | 95% percentile CI |
|---|---:|---:|
| Accuracy | -0.001063 | [-0.009564, 0.006376] |
| Macro F1 | +0.004883 | [-0.009102, 0.019344] |
| Minority recall | +0.019048 | [-0.000840, 0.041737] |
| Minority F1 | +0.010977 | [-0.010872, 0.034234] |
| NORMAL recall | -0.007505 | [-0.016886, 0.000000] |
| NLL | -0.008574 | **[-0.016004, -0.000956]** |

Only the NLL interval excludes zero. The defensible interpretation is that age
and sex improve probability calibration over the fixed acoustic model and show
a useful PFO/minority-recognition trend, but do not establish a statistically
significant Accuracy or F1 improvement. The multimodal model is a secondary
calibration/balanced operating point; Stage25 acoustic remains the primary
Accuracy model.

## Leakage audit

- The selector received a directory containing 20 validation files and zero
  test files.
- Each fold selected Logistic `C` using train patients only.
- Candidate correction and strength were selected using same-fold validation
  only.
- The locked prior was refit on train+validation after serialization.
- Train+validation/test sets were disjoint in all five folds.
- Metadata-fit/test overlap was zero in every fold.
- The evaluator exactly reproduced all six Stage25 acoustic metrics before
  accepting the multimodal output.
- All 941 patients occur once in the final held-out prediction file.
- No cross-fold metadata pooling was used.

## Artifacts

- Protocol: `configs/stage26_multimodal_protocol.json`
- Selector: `code/select_stage26_multimodal_demographic_prior.py`
- Evaluator: `code/evaluate_stage26_multimodal_demographic_prior.py`
- Paired analysis: `code/analyze_stage26_multimodal.py`
- Selection lock: `results/stage26/selection_lock.json`
- Evaluation summary:
  `results/stage26/stage26_multimodal_locked_summary.json`
- Patient predictions:
  `results/stage26/stage26_multimodal_locked_predictions.csv`
- Bootstrap and subgroup analysis:
  `results/stage26/stage26_multimodal_paired_analysis.json`
