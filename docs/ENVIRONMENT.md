# Environment

## Inference (exactly as scored)

| | |
|---|---|
| Hardware | Kaggle notebook, 2× NVIDIA T4 16 GB, 4 vCPU, 29 GB RAM (the pipeline asserts two T4 devices) |
| Image | `gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f` (Python 3.13, original pinning) |
| Extra packages | offline wheels inside the weights datasets: `timm-1.0.29`, `pydicom-3.0.2`, `opencv_python_headless-5.0.0.93` (installed by the stages into a private target directory) |
| Internet | off |
| Time | ≈ 4.5–5 h on the hidden test set on a normal day (Sixway ≈ 2.3 h, H6 ≈ 2.3 h); the 9 h limit is enforced by `--time-budget-hours` |

## Training (historical runs that produced the deployed checkpoints)

| | CoAtNet models (Sixway) | H6 |
|---|---|---|
| GPU | 1× RTX 5090 32 GB (rented), BF16 | 1× RTX 5090 32 GB per arm |
| Python | 3.11/3.12 venv | 3.12.3 |
| torch / CUDA | 2.7.1+cu128 / 12.8 | 2.7.1+cu128, torchvision 0.22.1 |
| timm | 1.0.29 | 1.0.29 |
| other | pandas 2.3.2, scikit-learn 1.7.2, safetensors 0.8.0, pydicom 3.0.2 | numpy 2.2.6, opencv-python-headless 4.12.0.88, Pillow 11.0.0, psutil 7.0.0, safetensors 0.8.0, PyYAML 6.0.2 |
| time | ≈ 5.5 h per full fit (20 epochs × ≈ 16.5 min), ≈ 4.5 h per fold model | 16 epochs per arm |

Sources of these numbers: `training/sources/joint/assets/FINAL_TRAINING_CONTRACT.json` (`execution.runtime`),
`training/sources/h6/assets/proofs/ENVIRONMENT_VERIFIED.json`. `requirements.txt` pins the training environment.

## Data preparation

- cache336 and the extension cache: Kaggle CPU notebooks (competition data mounted, internet off), see `data_prep/README.md`.
- H6 v4 caches: Python 3.11 venv from `data_prep/h6_v4/rsna/requirements.txt` (adds pylibjpeg for compressed DICOM).

## CPU smoke test

`tests/cpu_smoke.py` was run on Python 3.13.17, torch 2.7.1+cpu, timm 1.0.29, numpy 2.5.3, pandas 3.0.5,
scikit-learn 1.9.1, pydicom 3.0.2 (no GPU). Result: `docs/verification/CPU_SMOKE.json`.
