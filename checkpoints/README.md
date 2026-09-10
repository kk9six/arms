---
license: cc-by-nc-sa-4.0
language:
- en
tags:
- text-to-motion
- motion-generation
- human-motion-generation
- human-human-interaction
- streaming-generation
- pytorch
- arxiv:2607.05733
---

# ARMS

Pretrained checkpoints for **ARMS: Anchor–Relational Motion Streaming for Seamless Solo–Social Motion Transitions**.

- [Project page](https://hkliu.com/arms/)
- [Paper](https://arxiv.org/abs/2607.05733)
- [Source code](https://github.com/kk9six/arms)

## Checkpoint layout

| Directory | Purpose | Required VAE |
|---|---|---|
| `dit_without-shape/` | Unified HumanML3D + InterHuman model for single-person, two-person, and solo–social transition demos | `vae_without-shape/` |
| `dit_interhuman/` | InterHuman block-streaming model | `vae_with-shape/` |
| `dit_interhuman_full-window/` | InterHuman non-streaming/full-window model | `vae_with-shape/` |
| `dit_interx/` | InterX block-streaming model | `vae_interx/` |
| `dit_hml3d/` | HumanML3D-272 single-person block-streaming model | `vae_hml3d/` |
| `vae_without-shape/` | Joint HumanML3D + InterHuman causal motion VAE using neutral, zero-shape SMPL-H processing | — |
| `vae_with-shape/` | InterHuman causal motion VAE used by both InterHuman DiTs | — |
| `vae_interx/` | InterX causal motion VAE | — |
| `vae_hml3d/` | HumanML3D-272 causal motion VAE | — |

Each directory contains:

```text
config.yaml       # resolved training configuration
run.log           # original training log
model/latest.pth  # released model state
```

The DiT files load the EMA parameters stored under `net_ema`. The VAE files load the parameters stored under `net`. The files use PyTorch checkpoint serialization and should only be loaded from a trusted source.

## Download

Install the Hugging Face CLI, authenticate if the repository is gated, and download directly into the source repository's expected location:

```bash
hf auth login
hf download kksix/arms --local-dir checkpoints
```

The resulting layout should begin with:

```text
arms/
├── checkpoints/
│   ├── dit_hml3d/
│   ├── dit_interhuman/
│   ├── dit_interhuman_full-window/
│   ├── dit_interx/
│   ├── dit_without-shape/
│   ├── vae_hml3d/
│   ├── vae_interx/
│   ├── vae_with-shape/
│   ├── vae_without-shape/
│   └── ...

└── ...
```

## Data, models, and redistribution

This repository contains ARMS model weights only. Datasets, SMPL-family body models, and evaluator checkpoints have their own licenses and access terms and must be obtained from their respective sources. Downloading or using the ARMS checkpoints does not grant rights to those external assets.

## Citation
```bibtex
@inproceedings{arms2026,
  title     = {ARMS: Anchor--Relational Motion Streaming for Seamless Solo--Social Motion Transitions},
  author    = {Liu, Huakun and Yu, Qing and Fujiwara, Kent and Uchiyama, Hideaki and Kiyokawa, Kiyoshi},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
