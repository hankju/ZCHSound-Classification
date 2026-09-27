# Recorded Environment

- Python: 3.9.21
- Operating system: Red Hat Enterprise Linux 9.6, x86_64
- PyTorch: 2.8.0+cu128
- Torchvision: 0.23.0+cu128
- CUDA runtime bundled with PyTorch: 12.8
- cuDNN reported by PyTorch: 9.10.2
- PyTorch Lightning: 2.6.0
- timm: 1.0.19
- NumPy: 2.0.2
- scikit-learn: 1.6.1

The full source environment is recorded in `pip_freeze_full.txt`. A byte-identical
environment is not required, but use the pinned versions first when reproducing results.

Suggested setup on a CUDA-capable machine:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r environment/requirements.txt
```

If the new platform provides a compatible PyTorch container, install only packages that
are missing from `requirements.txt`. Do not copy the old `.venv`; it contains
platform-specific CUDA libraries and accounts for most of the old project's disk usage.
