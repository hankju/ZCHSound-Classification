# Reproducibility status

| Requirement | Status | Evidence |
|---|---|---|
| Training code | Included | `code/train_curriculum_multitask_patient.py` and deterministic task runner |
| Inference code | Included | `code/infer_stage25_checkpoint.py`, Stage25/Stage26 evaluators |
| Trained weights | Included for primary method | 20 Stage25 curriculum adaptor/head checkpoints |
| Recording predictions | Included | Two normalized 941-row OOF CSV files under `predictions/` |
| Five-fold splits | Included | `configs/splits_stage24_confirm_seed20268020/` |
| README commands | Included | Top-level `README.md` |
| Seeds | Included | Stage25/Stage26 protocols and runner |
| Environment | Included | `environment/` |
| Component settings | Included | Protocol JSON files, selection locks, and per-component summaries |
| Source audio identity | Included | 941-path SHA-256 manifest and verifier |
| Seven backbone training | Included | Original trainer, fixed grid, 35-task runner, and OOF collector |
| Licensing and citation | Included | Root MIT license, BEATs license, notices, CFF, and BibTeX |
| Automated repository audit | Included | GitHub Actions workflow and `make audit` |

The public audio and upstream pretrained base models are not redistributed.
The included trained checkpoints are the complete fold/replica-specific
trainable states used by the primary Stage25 result, but they must be attached
to the named upstream base models. Input caches are deterministic derivatives
of the audio and can be rebuilt with the included preprocessing scripts.

The flat-control and exploratory model weights are intentionally omitted. Their
formal probability exports are retained where required for paired replay.
