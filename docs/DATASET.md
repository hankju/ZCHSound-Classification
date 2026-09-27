# Dataset Source And Version

## Source

This project uses the clean, high-quality 941-recording subset of ZCHSound.
Audio files are not redistributed in this repository.

- Official project: http://zchsound.ncrcch.org.cn/
- Original-author GitHub repository:
  https://github.com/WeiJieOvO/ZCHSound-Dataset
- Dataset paper DOI: https://doi.org/10.1109/TBME.2023.3348800
- Classes used here: ASD, NORMAL, PDA, PFO, and VSD.

The source paper reports 941 participants and 941 audio recordings in the
high-quality subset. Therefore, each recording in this 941-recording release
corresponds to one participant, which is the basis for treating the fixed
recording-ID split as a patient-level split. The separate byte-identical-file
issue described below remains a source-data limitation and is disclosed
separately from the published one-recording-per-participant dataset design.

Users must obtain the dataset from the official source and comply with the
dataset's own access and usage terms. The root project LICENSE covers this
repository's original software, not the downloaded clinical audio.

## Required Citation

The dataset should be cited both in manuscripts and in software-derived work:

Weijie Jia, Yunyan Wang, Renwei Chen, Jingjing Ye, Die Li, Fei Yin, Jin Yu,
Jiajia Chen, Qiang Shu, and Weize Xu, "ZCHSound: Open-Source ZJU Paediatric
Heart Sound Database With Congenital Heart Disease," IEEE Transactions on
Biomedical Engineering, vol. 71, no. 8, pp. 2278-2286, 2024,
doi: 10.1109/TBME.2023.3348800.

The BibTeX entry is available in `CITATIONS.bib`. The source paper PDF is not
redistributed by this repository.

## Exact Local Version

The formal Stage25 and Stage26 results use all 941 rows from the downloaded
clean subset. `configs/dataset_audio_sha256.csv` records the relative path,
byte size, and SHA-256 digest of every WAV file used by the experiments.

After placing the files under a local dataset root, verify them with:

```bash
python scripts/verify_dataset.py --dataset-root /path/to/clean_heart_sound_data
```

This detects missing, additional, or byte-different WAV files. It also makes
the six source duplicate pairs independently reproducible without publishing
the audio itself. Read `docs/DATASET_DUPLICATE_NOTICE.md` before interpreting
the formal metrics.

## Expected Layout

```text
DATASET_ROOT/
  ASD/ZCH....wav
  NORMAL/ZCH....wav
  PDA/ZCH....wav
  PFO/ZCH....wav
  VSD/ZCH....wav
```
