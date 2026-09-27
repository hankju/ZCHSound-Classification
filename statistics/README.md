# Stage25 vs Seven Backbones: Matched Statistical Tests

## Analysis set

- Dataset: the 941-recording public release as distributed.
- Split: patient/recording-level five-fold split, seed `20268020`.
- Methods: fixed Stage25 curriculum ensemble and seven single-backbone models.
- All eight methods contain the same 941 recording IDs and use the same fold IDs.
- Stage25 predictions are reconstructed from the frozen `curriculum_prob_*`
  columns, not the conditional-final probabilities.

The public dataset contains six byte-identical recording pairs, including one
ASD/VSD label conflict. No authoritative correction was available. These
results preserve the public release for reproducibility; the duplicate list
must be disclosed and the results must not be described as content-deduplicated.

## Primary results

Pooled OOF Stage25 performance is 83.1031% Accuracy and 64.9904% Macro F1.
The highest single-backbone pooled Accuracy is 77.1520%, shared by
ConvNeXt-Tiny and DenseNet121. The highest single-backbone pooled Macro F1 is
51.3375% from ViT-Small/16.

Randomized complete-block ANOVA uses fold as the block and method as the
treatment. Duncan and Tukey use the same ANOVA residual mean square.

| Metric | Method F(7, 28) | p | Friedman chi-square(7) | p | Kendall W |
|---|---:|---:|---:|---:|---:|
| Accuracy | 16.5247 | 2.041e-08 | 14.8050 | 0.03858 | 0.4230 |
| Macro F1 | 15.3841 | 4.397e-08 | 15.1333 | 0.03433 | 0.4324 |

Duncan multiple range testing at alpha 0.05 assigns Stage25 to group `a` and
all seven single backbones to group `b` for both metrics. Tukey HSD rejects all
seven Stage25-versus-backbone comparisons for both metrics. Exact paired
McNemar tests also reject all seven comparisons after Holm correction.

Shapiro-Wilk tests use the five fold-level observations per method. All
Accuracy tests have p > 0.05. ConvNeXt-Tiny Macro F1 has p = 0.04165; all other
Macro F1 tests have p > 0.05. Because each method has only five folds and one
normality test rejects, Friedman results are reported alongside ANOVA.

## Duncan implementation

Methods are ordered by their fold mean. For an ordered range containing `r`
means, Duncan's alpha is `1 - (1 - 0.05)^(r - 1)`. The critical range is the
studentized-range quantile multiplied by `sqrt(MSE / 5)`, where MSE and its 28
degrees of freedom come from the randomized-block ANOVA. Compact letters are
derived from the resulting pairwise significance matrix.

## Files

- `eight_method_fold_metrics.csv`: the 40 fold-level inputs used by all tests.
- `accuracy_fold_matrix.csv`, `macro_f1_fold_matrix.csv`: 5 by 8 test matrices.
- `anova_*.csv`: randomized-block ANOVA tables.
- `duncan_*_groups.csv`: publication-ready Duncan letter tables.
- `duncan_*_pairwise.csv`: Duncan critical ranges and decisions.
- `tukey_hsd_*.csv`: all 28 Tukey pairwise comparisons.
- `friedman_*.csv`: Friedman statistics, p-values, and Kendall's W.
- `shapiro_wilk_by_method.csv`: per-method fold normality tests.
- `mcnemar_stage25_vs_backbones.csv`: exact paired tests with Holm correction.
- `eight_method_pooled_metrics.csv`: pooled OOF descriptive metrics.
- `predictions/stage25_fixed_curriculum_oof_predictions.csv`: formal Stage25
  recording-level predictions reconstructed for this analysis.
- `statistical_summary.json`: machine-readable summary and SHA-256 provenance.

## Interpretation limits

Cross-validation folds share portions of their training data, so fold-level
tests should be interpreted as supporting evidence rather than independent
replications. The paired recording-level McNemar analysis provides an
additional comparison on identical recordings, but it does not remove the
known duplicate-content limitation in the public dataset.
