# Recording-level analyses

This directory contains recording-level analyses for the latest formal acoustic
and multimodal methods and the seven matched single-backbone comparisons. All
methods use the same 941 recording IDs and fixed five-fold split (seed
`20268020`).

## Outputs

- `predictions/`: nine normalized OOF prediction CSV files with a common schema.
- `confusion_matrices_wide.csv`: one publication-ready confusion-matrix row per
  method and true class.
- `confusion_matrices_long.csv`: tidy method/true-label/predicted-label counts.
- `bootstrap_95ci.csv`: recording-level nonparametric percentile 95% confidence
  intervals for Accuracy, Macro F1, and each class F1.
- `mcnemar_all_pairwise.csv`: all 36 exact two-sided paired McNemar comparisons,
  with Holm correction across the 36 comparisons.
- `recording_level_analysis_summary.json`: settings, validation results, source
  paths, and SHA-256 provenance.

The bootstrap uses 10,000 resamples and seed `20260826`. Each resample draws 941
recordings with replacement. The same sampled indices are shared across all
methods so method-level estimates remain paired.

## Reproduction

Run `code/generate_recording_level_analyses.py` with the Stage25 prediction CSV,
Stage26 multimodal prediction CSV, seven-backbone prediction directory, fixed
split directory, and this directory as `--output-dir`. The script uses only the
Python standard library.

The known byte-identical recordings in the public dataset remain present. The
analysis is recording-ID complete but must not be described as content-
deduplicated.
