# Model Card

## Formal Models

| Model | Inputs | Accuracy | Macro F1 | Reporting role |
|---|---|---:|---:|---|
| Stage25 fixed acoustic curriculum | Heart-sound audio | 0.831031 | 0.649904 | Primary accuracy model |
| Stage26 clinical fusion | Audio probabilities, age, sex | 0.829968 | 0.654787 | Secondary multimodal model |

Stage25 averages two AST replicas and two BEATs replicas within each outer
fold. Its curriculum objective combines five-class prediction, NORMAL-versus-
CHD discrimination, four-way CHD subtype prediction, and a consistency loss.
Stage26 adds a low-capacity fold-local demographic prior to frozen Stage25
outputs. Fold-local parameters are selected before each corresponding outer
test evaluation.

## Intended Use

- Reproducible research on patient-level pediatric heart-sound classification.
- Method comparison on the documented ZCHSound 941-recording split.
- Secondary analysis of fixed recording-level probability outputs.

## Out-Of-Scope Use

- Clinical diagnosis, triage, or treatment decisions.
- Deployment as an autonomous screening device.
- Claims of performance on hospitals, devices, age ranges, or populations not
  represented by the source dataset.
- Claims that the formal split is disjoint by audio-content hash.

## Evaluation Protocol

- Five patient/recording-level outer folds, seed `20268020`.
- Each recording ID appears once in the pooled outer-test predictions.
- Train, validation, and test recording/patient IDs are disjoint within fold.
- Stage25 and Stage26 selection parameters are frozen before test evaluation.
- Canonical outputs are under `predictions/` and can be audited with the
  scripts under `scripts/`.

## Limitations

- The public 941-row source contains six byte-identical pairs under different
  IDs; four pairs cross outer folds and one pair has conflicting ASD/VSD labels.
- The experiment is not fully disjoint by audio-content hash.
- PDA and PFO have substantially less support than NORMAL.
- No independent external hospital or recording-device test set is included.
- Stage26 uses age and sex; performance and subgroup behavior should be
  examined before any new use.

## Reproducibility And Licenses

See `README.md`, `docs/REPRODUCIBILITY_STATUS.md`, `docs/DATASET.md`,
`THIRD_PARTY_NOTICES.md`, and `LICENSE`. The public audio and external AST and
BEATs base models are not redistributed.
