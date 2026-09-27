# Seven-Backbone Retraining

The seven single-backbone comparisons can be retrained from the original
941-row dataset using the exact formal split and settings preserved in
`configs/backbone7_retraining_grid.tsv`.

## Fixed Design

- Outer split: seed `20268020`, folds 0 through 4.
- Input: 4.0-second windows with 3.0-second overlap.
- Representation: log-STFT resized to 224 x 224 and repeated to three channels.
- File aggregation: mean segment logits.
- Optimizer settings: learning rate `5e-4`, weight decay `5e-2`.
- Training: up to 300 epochs, early-stopping patience 50.
- Augmentation: SpecAugment time/frequency masks 20/10 and mixup 0.4 with
  probability 0.5.
- Validation checkpoint rule: highest recording-level validation Macro F1.
- Test export: `test_file_probs.csv` from the frozen validation-selected
  checkpoint; no TTA, top-k, or temperature variant is used for the formal
  backbone comparison.

ConvNeXt-Tiny uses training seed 252. The other six methods use seed 42. All
model identifiers and remaining settings are machine-readable in the TSV.

## Run One Task

There are 35 independent tasks. Task ID is `method_index * 5 + fold`:

```bash
python scripts/run_backbone_task.py --list
python scripts/run_backbone_task.py \
  --task-id 0 \
  --dataset-root /path/to/clean_heart_sound_data
```

The runner does not require SLURM. Any scheduler, container, workstation, or
cloud task system may launch different task IDs independently. A method/fold
can also be selected explicitly:

```bash
python scripts/run_backbone_task.py \
  --method resnet50 --fold 0 \
  --dataset-root /path/to/clean_heart_sound_data
```

Run `scripts/verify_dataset.py` once before launching the grid. Timm downloads
the corresponding ImageNet-pretrained model weights on first use.

## Collect OOF Predictions

After all 35 tasks finish:

```bash
python code/collect_backbone_oof.py \
  --input-root reproduced/backbones \
  --output-dir reproduced/backbone_oof
```

The collector verifies that each method has exactly the 941 recording IDs in
the fixed five-fold split. The resulting files have the same schema and names
as the formal backbone inputs used under `statistics/`.

## Provenance

`code/nostack_filelevel_ablate.py` and `code/filelevel_mil_common.py` are the
training implementation used by the original formal backbone rerun. The new
task runner only replaces platform-specific job submission and does not change
the model, split, loss, augmentation, checkpoint, or export logic.
