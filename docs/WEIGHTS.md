# Weight policy

The repository includes only trained weights used by the primary Stage25
fixed-curriculum result:

- 10 AST curriculum checkpoints: 2 replicas x 5 folds.
- 10 BEATs curriculum checkpoints: 2 replicas x 5 folds.
- Total included trained checkpoint size: approximately 24 MiB.

Each checkpoint stores the selected trainable LoRA/adaptor and head state plus
its locked export configuration. These are not standalone full-backbone
checkpoints.

External base models:

- AST: `MIT/ast-finetuned-audioset-10-10-0.4593`, retrieved through
  Hugging Face transformers.
- BEATs: `BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt`, retrieved from the
  official Microsoft/unilm BEATs release. The preserved official code commit is
  `833df7e7832e5064a281131ee64a481afa8e5b95`.

The 20 flat-control checkpoints and unrelated exploratory checkpoints are not
included. Their probability exports are retained only where needed to replay
the fixed formal comparison. This keeps every GitHub object below 100 MB and
avoids publishing several gigabytes of unrelated experimental checkpoints.

