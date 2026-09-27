# Source Dataset Duplicate Recording Notice

## Formal reporting scope

The formal results in this repository use the original public dataset exactly
at its released size of 941 recording rows:

- Primary: Stage25 fixed acoustic curriculum, Accuracy 0.831031 and Macro F1
  0.649904.
- Secondary: Stage26 age/sex clinical fusion, Accuracy 0.829968 and Macro F1
  0.654787.

No row was removed when calculating either formal result. A separate 935-row
duplicate-removal run was performed as a sensitivity experiment, but its
removal rule is project-defined rather than an authoritative correction from
the dataset authors. It is not a formal result and is not packaged as a
replacement dataset or split.

## Observed source-data issue

The downloaded public ZCHSound release contains six pairs of byte-identical WAV
files under different recording IDs. This is a property of the source dataset,
not duplication introduced by training or preprocessing.

| First source entry | Second source entry | Labels | SHA-256 prefix | Outer-test folds |
|---|---|---|---|---|
| `NORMAL/ZCH0090.wav` | `NORMAL/ZCH0171.wav` | NORMAL / NORMAL | `add79a2966b8` | 3 / 3 |
| `NORMAL/ZCH0015.wav` | `NORMAL/ZCH0219.wav` | NORMAL / NORMAL | `c7c393f13c18` | 1 / 2 |
| `NORMAL/ZCH0268.wav` | `NORMAL/ZCH0366.wav` | NORMAL / NORMAL | `50963b87e491` | 4 / 2 |
| `NORMAL/ZCH0182.wav` | `NORMAL/ZCH0401.wav` | NORMAL / NORMAL | `51ec25406463` | 4 / 4 |
| `NORMAL/ZCH0465.wav` | `NORMAL/ZCH0515.wav` | NORMAL / NORMAL | `e2982ece4a31` | 0 / 3 |
| `ASD/ZCH0545.wav` | `VSD/ZCH0802.wav` | ASD / VSD | `343d1c65e528` | 1 / 2 |

The final pair has identical audio content but conflicting ASD and VSD labels.
The repository does not assert which label is correct. Users should ask the
dataset authors for authoritative provenance or corrected annotation.

## Consequence for the fixed split

Every one of the 941 recording IDs occurs exactly once as outer-test data, and
train/validation/test recording IDs are disjoint within each fold. However,
four duplicate-content pairs cross outer folds. Consequently, the formal
split is recording-ID/patient-ID disjoint according to the published IDs, but
it is not fully disjoint by audio-content hash. This limitation applies equally
to all methods evaluated with these fixed splits and should be disclosed when
reporting the results.

The complete hashes and fold assignments are preserved in
`audits/source_dataset_duplicate_audit.csv`. Future work should rerun all
methods using an author-verified correction or a content-grouped split; such a
rerun must be reported as a different dataset/split version rather than mixed
with the formal 941-row results.
