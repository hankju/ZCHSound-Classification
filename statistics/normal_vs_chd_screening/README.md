# NORMAL versus CHD screening

This directory contains binary screening results recomputed from the latest
formal 941-recording OOF predictions.

- Negative: `NORMAL`
- Positive: `ASD`, `PDA`, `PFO`, or `VSD`
- Binary predicted label: collapse the five-class argmax prediction into
  `NORMAL` or `CHD`
- ROC score: `1 - prob_NORMAL`

## Files

- `normal_vs_chd_metrics.csv`: pooled Accuracy, Sensitivity, Specificity, ROC
  AUC, and TN/FP/FN/TP counts for Stage25 and Stage26.
- `normal_vs_chd_predictions.csv`: recording-level binary labels and scores.
- `normal_vs_chd_summary.json`: definitions, validation status, results, input
  paths, and SHA-256 provenance.

## Reproduction

Run `code/generate_normal_vs_chd_screening.py` with the normalized Stage25 and
Stage26 recording-level prediction CSV files. The script uses only the Python
standard library and validates row count, recording-ID uniqueness, probability
ranges and sums, and reconstruction of each five-class prediction.
