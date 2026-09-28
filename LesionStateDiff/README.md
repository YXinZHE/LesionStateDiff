# LesionStateDiff

Official implementation package for **LesionStateDiff: Spatial-Semantic
Diffusion for Controllable IVOCT Lesion Resynthesis**.

The method contains two components:

- **LSDM** (Lesion-Aware Spatial Diffusion-State Modeling): constructs a
  spatial diffusion-time map and applies forward/reverse diffusion only to
  lesion pixels while preserving background and lumen states.
- **PSC** (Pathology-Semantic Conditioning): maps the complete five-class
  segmentation to spatial semantic features and injects them into the U-Net
  middle block through a zero-initialized 1x1 projection.

This release contains the method, training, sampling, checkpoint, manifest,
and unit-test code. It intentionally excludes patient data, generated images,
training logs, and model weights.

## Input contract

IVOCT images are grayscale `750 x 750` images. Training pads them to
`768 x 768`. Masks use the following fixed class indices:

| Value | Class |
|---:|---|
| 0 | background |
| 1 | LM (lumen) |
| 2 | FC (fibrous cap) |
| 3 | LC (lipid core) |
| 4 | VV (vasa vasorum) |

For RGBA masks, the channel order is strictly `R=LM, G=FC, B=LC, A=VV`.

The denoiser input is:

```text
[x_tau, hard_masked_source, segmentation, normalized_tau]
shape = [B, 4, 768, 768]
```

PSC additionally receives the complete segmentation as a semantic condition;
it does not increase the four-channel input.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[test]
pytest -q
```

The experiments were executed with Python 3.12.3, PyTorch 2.12.1+cu130,
diffusers 0.39.0, NumPy 2.4.6, Pillow 12.2.0, and PyYAML 6.0.3.

## Repository layout

```text
src/lesionstatediff/
  region_time.py                    # LSDM equations and DDIM update
  class_semantic_encoder.py         # zero-initialized PSC projection
  region_time_b3_semantic.py        # PSC-injected U-Net and sampler
  region_time_dataset.py            # image/mask loading
  unified_four_class_dataset.py     # manifest schema and samplers
  *_checkpoint.py                   # strict checkpoint loading/saving
scripts/
  build_manifest.py
  train_lsdm.py
  train_lsdm_refine.py
  train_lsdm_constant.py
  train_psc.py
  train_psc_refine.py
  generate.py
tests/
configs/
docs/
```

## Reproduction sequence

The released final model follows this sequence:

1. Train the inherited four-class conditional diffusion baseline.
2. Train LSDM with `train_lsdm.py`, then its two refinement scripts.
3. Train PSC with `train_psc.py`.
4. Refine PSC with `train_psc_refine.py`.
5. Audit and generate with `generate.py`.

The exact learning-rate lineage is recorded in
[`configs/training_schedule.yaml`](configs/training_schedule.yaml). Full CLI
examples are in [`docs/TRAINING.md`](docs/TRAINING.md).

