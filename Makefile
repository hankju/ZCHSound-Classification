PYTHON ?= python
DATASET_ROOT ?=
TASK_ID ?= 0

.PHONY: audit dataset-audit statistics baseline-list baseline-task

audit:
	$(PYTHON) scripts/verify_integrity.py
	$(PYTHON) scripts/audit_release.py
	$(PYTHON) scripts/audit_fold_local_protocol.py

dataset-audit:
	@test -n "$(DATASET_ROOT)" || (echo "Set DATASET_ROOT=/path/to/clean_heart_sound_data" && exit 2)
	$(PYTHON) scripts/verify_dataset.py --dataset-root "$(DATASET_ROOT)"

statistics:
	$(PYTHON) code/analyze_original941_stage25_vs_backbones.py \
		--normalized-predictions-dir statistics/recording_level_analyses/predictions \
		--split-dir configs/splits_stage24_confirm_seed20268020 \
		--output-dir reproduced/statistics

baseline-list:
	$(PYTHON) scripts/run_backbone_task.py --list

baseline-task:
	@test -n "$(DATASET_ROOT)" || (echo "Set DATASET_ROOT=/path/to/clean_heart_sound_data" && exit 2)
	$(PYTHON) scripts/run_backbone_task.py --task-id $(TASK_ID) --dataset-root "$(DATASET_ROOT)"
