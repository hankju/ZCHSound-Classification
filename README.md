# ZCHSound Heart-Sound Classification

[Model card](MODEL_CARD.md) | [Dataset and citation](docs/DATASET.md) |
[Licenses](THIRD_PARTY_NOTICES.md) | [Reproducibility status](docs/REPRODUCIBILITY_STATUS.md)

Research project by 朱恆暐 on five-class congenital heart-disease
classification from pediatric heart sounds. This repository presents the
formal acoustic ensemble (Stage25) and age/sex fusion (Stage26), with code,
fixed splits, predictions, and trained Stage25 adapter/head weights.

This repository preserves the latest formal patient-level five-fold results on
the original 941-recording public dataset:

| Method | Accuracy | Macro F1 | Role |
|---|---:|---:|---|
| Stage25 fixed acoustic curriculum | 0.831031 | 0.649904 | Primary accuracy model |
| Stage26 age/sex clinical fusion | 0.829968 | 0.654787 | Secondary balanced/calibrated model |

Stage25 is an unweighted ensemble of two AST replicas and two BEATs replicas
within each outer fold. Stage26 adds a fold-local age/sex demographic prior;
its learned parameters are selected without pooling validation labels across
outer folds.

The formal Stage25 result is the raw `curriculum` branch (`Accuracy =
0.831031`, `Macro F1 = 0.649904`). Some historical Stage24 replay output also
prints a conditional-fusion `final_acc` of approximately `0.828905`. That
auxiliary value is retained for provenance only and is not the reported
Stage25 result.

For formal reporting, use Stage25 as the primary highest-accuracy result and
Stage26 as the secondary multimodal result.

## Method

The 941 recordings are assigned to five fixed patient-level outer folds. For
each fold, training, validation, and test patients are disjoint. Development
and validation data were used to choose the complementary AST balanced and
BEATs conservative curriculum families. Their objectives combine five-class
classification with normal-versus-CHD, CHD-subtype, and consistency losses.
The model families, loss weights, seeds, and averaging rule were fixed before
the Stage25 confirmation test.

Each encoder is trained twice per fold with paired seeds. The two AST
probability vectors are averaged, as are the two BEATs vectors; the encoder
means are then averaged equally. Stage26 starts from those fixed Stage25
probabilities and fits an age/sex prior on each fold's training patients.
It chooses the demographic correction using that fold's validation patients,
then refits the selected prior on training plus validation patients before
evaluating the disjoint test patients. The test sets are used only for final
evaluation. See the [Stage25 protocol](docs/RUN_20260824_STAGE25_CURRICULUM_REPLICATION.md)
and [Stage26 protocol](docs/RUN_20260824_STAGE26_MULTIMODAL.md).

## Included

- Patient-level training, checkpoint inference, fusion, and analysis code.
- The fixed five-fold split definitions for seed `20268020`.
- Complete recording-level OOF predictions for Stage25 and Stage26.
- Stage25/Stage26 protocols, selection locks, summaries, and statistics.
- All 20 trained adapter/head checkpoints used by the primary Stage25 model:
  2 encoders x 2 replicas x 5 folds.
- Validation/test component exports and flat-control exports needed to replay
  the locked Stage25 and Stage26 fusion calculations.
- Exact environment records and SHA-256 integrity checks.
- Exact seven-backbone trainer, fixed settings, and platform-neutral 35-task runner.
- Project/third-party licenses, citation metadata, model card, and GitHub CI.

## Not Included

- Public audio files. Place them under a local dataset directory.
- The approximately 4 GB AST input cache or the BEATs fbank cache. Both can be
  recreated from the audio with the included scripts.
- The 346 MB pretrained BEATs base checkpoint. Download
  `BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt` from the official
  Microsoft/unilm BEATs release.
- Flat-control checkpoints. Their fixed recording-level probability exports
  are included, but only weights used by the best Stage25 result are retained.

The included Stage25 checkpoints contain trained LoRA/adaptor and classification
head parameters. They still require the corresponding upstream AST or BEATs
base model.

## Dataset Layout

```text
DATASET_ROOT/
  ASD/ZCH....wav
  NORMAL/ZCH....wav
  PDA/ZCH....wav
  PFO/ZCH....wav
  VSD/ZCH....wav
```

The original 941-recording release contains six known byte-identical pairs,
including one ASD/VSD conflicting-label pair. All 941 source rows are retained
because the formal Stage25/Stage26 results use the public dataset as released.
Four identical-content pairs, including the conflicting-label pair, cross
outer folds under the fixed recording-ID split. This source-data limitation is
not caused by preprocessing, but it means the release must not be described as
content-deduplicated. Read `docs/DATASET_DUPLICATE_NOTICE.md` and the
machine-readable `audits/source_dataset_duplicate_audit.csv` before using the
splits or comparing results.

The exact source paper citation is included in `CITATIONS.bib` and
`docs/DATASET.md`. The paper PDF and public audio are not redistributed.

Verify that a downloaded dataset is byte-identical to the experiment copy:

```bash
python scripts/verify_dataset.py --dataset-root "$DATASET_ROOT"
```

## Environment

The recorded training environment used Python 3.9.21, PyTorch 2.8.0+cu128,
CUDA 12.8, transformers 4.55.4, NumPy 2.0.2, SciPy 1.13.1, and
scikit-learn 1.6.1.

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r environment/requirements.txt
```

See `environment/ENVIRONMENT.md` and `environment/pip_freeze_full.txt` for the
recorded platform and complete package inventory.

## Build Input Caches

AST inputs use 4.0-second windows with 3.0-second overlap:

```bash
python code/cache_ast_input_values.py \
  --dataset-path "$DATASET_ROOT" \
  --output-root data/ast_input_values_4s_ov3 \
  --window-sec 4.0 --overlap-sec 3.0
```

BEATs inputs use the same windows:

```bash
python code/cache_beats_fbank.py \
  --dataset-path "$DATASET_ROOT" \
  --output-root data/beats_fbank_4s_ov3 \
  --checkpoint pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
  --official-code-root third_party/unilm_beats/beats \
  --window-sec 4.0 --overlap-sec 3.0
```

The expected cache manifests are preserved in `configs/cache_manifests/`.

## Retrain Stage25

The deterministic runner defines all four experimental families. Tasks `0..19`
train the two curriculum families used by the best model. Tasks `20..39` train
the paired flat controls used in the formal comparison.

```bash
python code/run_stage24_conditional_task.py \
  --task-id 0 \
  --project-root "$PWD" \
  --platform-root "$PWD" \
  --python "$(command -v python)" \
  --split-seed 20268020
```

The evaluator reports flat, curriculum, and historical conditional-fusion
branches. The expected formal Stage25 line is `curriculum_acc=0.831031`; do not
substitute the auxiliary `final_acc` for the formal result. The generated
summary is also used as the acoustic reference for Stage26, which explicitly
reads the curriculum branch.

Run the desired task IDs independently. Seeds are fixed by the runner and by
`configs/stage25_curriculum_replication_protocol.json`:

- AST: `20332100 + replica*1000 + fold`
- BEATs: `20332100 + 100000 + replica*1000 + fold`

The primary components and loss settings are:

- AST balanced: binary/subtype/consistency weights `0.50/1.00/0.10`.
- BEATs conservative: binary/subtype/consistency weights `0.25/0.50/0.05`.
- Average two replicas within each encoder, then average AST and BEATs.

## Retrain Seven Single Backbones

The formal ConvNeXt-Tiny, Swin-Tiny, DenseNet121, EfficientNet-B3,
ViT-Small/16, RegNetY-008, and ResNet50 comparisons use the same fixed
five-fold split. Their exact settings are in
`configs/backbone7_retraining_grid.tsv`.

```bash
python scripts/run_backbone_task.py --list
python scripts/run_backbone_task.py \
  --task-id 0 --dataset-root "$DATASET_ROOT"
```

Tasks `0..34` are independent and can be distributed by any scheduler. See
`docs/BASELINE_RETRAINING.md` for collection and validation instructions.

## Inference From Included Weights

Example for AST replica 1, fold 0:

```bash
python code/infer_stage25_checkpoint.py \
  --encoder ast \
  --trained-checkpoint stage24_confirmation/seed20268020/ast_curriculum_balanced/replica1/fold0/components/ast_curriculum_balanced_seed2_source/curriculum_checkpoint.pt \
  --input-root data/ast_input_values_4s_ov3 \
  --split-json configs/splits_stage24_confirm_seed20268020/cv5_tvt_fold0.json \
  --metadata-csv configs/clean_dataset_manifest_merged.csv \
  --seed 20333100 \
  --split test \
  --output-csv reproduced/ast_replica1_fold0_test.csv
```

For BEATs, change `--encoder`, the trained checkpoint and input root, and add:

```text
--beats-base-checkpoint pretrained/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
--beats-official-code-root third_party/unilm_beats/beats
```

## Replay Stage25 And Stage26 Fusion

Create physically separated validation and test views from the included
component exports:

```bash
python code/prepare_stage24_confirmation_view.py \
  --project-root "$PWD" --phase validation \
  --split-seed 20268020 --view-tag stage25_replay
python code/prepare_stage24_confirmation_view.py \
  --project-root "$PWD" --phase test \
  --split-seed 20268020 --view-tag stage25_replay
```

Regenerate the Stage25 fold-local selection lock and evaluate the separated
test view:

```bash
mkdir -p reproduced/stage25
python code/select_stage24_confirmation.py \
  --validation-root data/stage25_replay_selection \
  --split-root configs \
  --development-lock results/stage25/stage24_development_selection_lock.json \
  --output-lock reproduced/stage25/selection_lock.json \
  --split-seed 20268020
python code/evaluate_stage24_confirmation.py \
  --test-root data/stage25_replay_evaluation \
  --split-root configs \
  --development-lock results/stage25/stage24_development_selection_lock.json \
  --selection-lock reproduced/stage25/selection_lock.json \
  --output-dir reproduced/stage25/evaluation_once \
  --split-seed 20268020
```

Replay Stage26 by selecting only from the validation view and then evaluating
the frozen lock on the separated test view:

```bash
mkdir -p reproduced/stage26
python code/select_stage26_multimodal_demographic_prior.py \
  --validation-root data/stage25_replay_selection \
  --split-root configs \
  --metadata-csv configs/clean_dataset_manifest_merged.csv \
  --output-lock reproduced/stage26/selection_lock.json \
  --split-seed 20268020 --cv-seed 20262600
python code/evaluate_stage26_multimodal_demographic_prior.py \
  --test-root data/stage25_replay_evaluation \
  --split-root configs \
  --metadata-csv configs/clean_dataset_manifest_merged.csv \
  --selection-lock reproduced/stage26/selection_lock.json \
  --acoustic-reference-summary reproduced/stage25/evaluation_once/stage24_conditional_fusion_summary.json \
  --output-dir reproduced/stage26/evaluation_once
```

Regenerating locks records local paths and avoids relying on absolute paths
preserved from the original platform. Detailed method interpretation is in
`docs/RUN_20260824_STAGE26_MULTIMODAL.md`.

For reporting and statistical analysis, the normalized OOF files under
`predictions/` are the canonical inputs. They contain 941 unique recording IDs,
fold, true label, predicted label, and all five probabilities.

## Integrity Check

```bash
python scripts/verify_integrity.py
python scripts/audit_release.py
```

The equivalent repository-level command is:

```bash
make audit
```

This verifies every file listed in `SHA256SUMS`. Generated caches, downloaded
base models, and new reproduction outputs are intentionally excluded. The
release audit additionally checks 941-recording split coverage, both formal
prediction CSV files, probability reconstruction, the 20-checkpoint policy,
and GitHub's 100 MB per-object limit.

## License And Citation

Original project software is released under the root MIT `LICENSE`. Vendored
BEATs code retains Microsoft's MIT license under `third_party/unilm_beats/`.
The project license does not cover the ZCHSound audio or external pretrained
models; see `THIRD_PARTY_NOTICES.md`.

GitHub citation metadata is in `CITATION.cff`. Dataset, AST, and BEATs BibTeX
entries are in `CITATIONS.bib`.
