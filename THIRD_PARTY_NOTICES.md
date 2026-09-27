# Third-Party Notices

The root MIT License applies to original project software and documentation.
It does not relicense the ZCHSound audio dataset, upstream pretrained models,
or vendored third-party code.

## Microsoft BEATs

This repository vendors the BEATs implementation from Microsoft UniLM commit
`833df7e7832e5064a281131ee64a481afa8e5b95`:

https://github.com/microsoft/unilm/tree/833df7e7832e5064a281131ee64a481afa8e5b95/beats

BEATs is distributed under the Microsoft MIT License preserved at
`third_party/unilm_beats/LICENSE`. The upstream pretrained checkpoint is not
redistributed here.

## Audio Spectrogram Transformer

The external model `MIT/ast-finetuned-audioset-10-10-0.4593` is downloaded from
Hugging Face and is identified there as BSD-3-Clause. It is not redistributed
in this repository. Users must review and comply with the current upstream
model terms:

https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593

## ZCHSound Dataset

The public audio is not included. The project LICENSE does not grant rights to
the dataset. Obtain it from its official source and follow its own usage terms:

http://zchsound.ncrcch.org.cn/

Dataset citation and version-identification details are in `docs/DATASET.md`.

## Python Dependencies

PyTorch, Transformers, timm, NumPy, SciPy, scikit-learn, librosa, and the other
packages listed under `environment/` retain their respective upstream licenses.
Installing a dependency does not place it under this project's MIT License.
