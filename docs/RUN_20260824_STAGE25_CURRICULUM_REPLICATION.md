# Stage25 Fixed Curriculum Replication

Status: completed; fixed confirmation gate passed.
This is the final planned acoustic experiment before manuscript closeout.

## Purpose

Replicate the Stage24 raw curriculum improvement on a second untouched
patient-level five-fold split. No model family, loss weight, seed count,
aggregation rule, or evaluation criterion may be changed using Stage25 test
predictions.

## Fixed protocol

- Patient split seed: 20268020.
- Five folds; every one of 941 patients occurs in exactly one test fold.
- Every fold has disjoint train, validation, and test patients.
- Paired flat control: AST flat plus BEATs flat.
- Fixed curriculum pair: AST balanced plus BEATs conservative.
- Two paired training seeds per model and fold, giving 40 GPU tasks.
- Each family is averaged over its two replicas; AST and BEATs are then
  combined by an unweighted arithmetic mean.
- Primary comparison: raw curriculum pair minus paired flat control.
- Confirmation requires accuracy delta >= -0.005, macro-F1 delta >= 0.01,
  and minority-F1 delta >= 0.01.
- Test probabilities are not used for selection or adaptation and are
  evaluated once after the transferred protocol lock is serialized.

The machine-readable lock is
`configs/stage25_curriculum_replication_protocol.json`.

## Preflight

- Python and SLURM syntax passed.
- Existing seed-20267020 Stage24 audit passed after parameterization.
- Task boundaries 0, 9, 10, 19, 20, 29, 30, and 39 passed dry-run mapping.
- New split audit passed: 941/941 patients appear in exactly one test fold.
- Split and training source hashes were recorded before submission.

## Results

Jobs 297345, 297346, and 297347 used a 4+4+2 H200 layout. All jobs completed
with exit code 0; the slowest elapsed time was 15 minutes 1 second. All 40
training outputs passed the fixed-manifest, patient-disjointness, probability,
metadata, and test-blind audit.

On the untouched seed-20268020 test folds, the paired flat control obtained
accuracy 0.8214665250, macro F1 0.6048342128, minority recall 0.3538165266,
minority F1 0.4085082299, normal recall 0.9793621013, and NLL 0.5390905216.

The fixed raw curriculum pair obtained accuracy 0.8310308183, macro F1
0.6499040704, minority recall 0.4411939776, minority F1 0.4704466113, normal
recall 0.9718574109, and NLL 0.5453007496. Relative to paired flat, this is:

- accuracy: +0.956 percentage points;
- macro F1: +4.507 points;
- minority recall: +8.738 points;
- minority F1: +6.194 points;
- normal recall: -0.750 points;
- NLL: +0.00621 (worse calibration).

All three predeclared confirmation gates passed. This independently replicates
the Stage24 finding and, on this split, also improves overall accuracy.

Across the two untouched split seeds 20267020 and 20268020, the unweighted
split mean changed from accuracy 0.8230605739 / macro F1 0.6052668868 /
minority F1 0.4070945494 for paired flat to accuracy 0.8278427205 / macro F1
0.6436481225 / minority F1 0.4637219675 for fixed raw curriculum. The mean
deltas are +0.478, +3.838, and +5.663 percentage points, respectively.

The curriculum result trades a small amount of normal recall and calibration
for substantially better minority recognition. It should therefore be reported
as a fixed balanced operating point, not as a probability-calibration upgrade.

## Artifacts

- Protocol SHA256: `14b91da84e031ac74234565797b2e911e64799758fd4d0a9d34008de131e42ad`.
- Selection lock canonical SHA256: `8cea675cdee574fd79d0bb686e51e5352d1c37ed26d1815fcd612f823762422b`.
- Evaluation summary file SHA256: `3987451a02c98adedc6dcbc9a67f6913769dbcefc023ff03685ee87487642ef3`.
- Machine-readable comparison:
  `results/stage25/stage24_stage25_replication_comparison.json`.
- Patient-level probabilities:
  `results/stage25/stage24_conditional_fusion_predictions.csv`.

No further acoustic model or fusion search is justified from this test view.
