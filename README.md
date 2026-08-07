<div align="center">

### Exploiting Local Flatness for Efficient Out-of-Distribution Detection

[Seonghwan Park](https://shpark97.github.io/)<sup>1,2</sup> · [Hyunji Jung](https://scholar.google.com/citations?user=hl4jLS4gDRUC&hl=ko)<sup>2</sup> · [Dongyeop Lee](https://dongyeoplee2.github.io/)<sup>2</sup> · [Namhoon Lee](https://namhoonlee.github.io/)<sup>2</sup>

<sup>1</sup>KETI &nbsp;·&nbsp; <sup>2</sup>POSTECH

**ECCV 2026**

[![arXiv](https://img.shields.io/badge/arXiv-2606.29952-b31b1b.svg)](https://arxiv.org/abs/2606.29952)
[![Conference](https://img.shields.io/badge/ECCV-2026-4b8bbe.svg)](https://eccv.ecva.net/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

**Fold** is a lightweight post-hoc OOD detector built on a simple observation: *OOD inputs sit in sharper regions of the loss landscape than ID inputs*. Fold measures this sharpness with the trace of the **feature Hessian** of the energy loss, computed on partially normalized features — no retraining, no OOD data, and roughly the cost of a standard forward pass.

**AutoFold** goes one step further and calibrates the normalization exponent α **fully self-supervised**: it synthesizes pseudo-OOD samples by masking the ground-truth logit of ID validation samples, so no external data of any kind is needed.

### ✨ Highlights

- **Post-hoc & plug-and-play** — works directly on any pre-trained classifier
- **Efficient** — near-zero setup time; inference at forward-pass latency
- **No OOD data** — AutoFold tunes itself using only ID samples
- **Strong & stable** — best average AUROC/FPR95 across CIFAR-10/100, ImageNet-200, and ImageNet-1K, and consistent gains across architectures

### 🔍 How it works

Computing loss-landscape curvature in parameter space is prohibitively expensive for modern networks, so Fold measures it in feature space instead: the curvature of the energy loss at the logits is transported through the classifier onto the penultimate features, capturing sharpness along class-discriminative directions. Since large feature magnitudes saturate the softmax and hide this curvature gap, features are partially normalized before scoring. The trace of the resulting feature Hessian is the OOD score — flat for ID samples, sharp for OOD. AutoFold removes the last remaining knob (the normalization strength) without touching any OOD data: it masks the ground-truth logit of ID validation samples to simulate unknown-class inputs, then picks the setting that best separates them from intact samples.

## Methods

| Paper name | Postprocessor | Description |
|---|---|---|
| Fold       | `fold`           | feature Hessian trace + partial feature normalization |
| Fold-R     | `fold_react`     | Fold + ReAct activation clipping |
| Fold-A     | `fold_ash`       | Fold + ASH activation shaping |
| AutoFold   | `autofold`       | Fold with self-supervised α calibration (ID logit masking) |
| AutoFold-R | `autofold_react` | AutoFold + ReAct |
| AutoFold-A | `autofold_ash`   | AutoFold + ASH |

## 🚀 Getting Started

### Installation

```bash
conda create -n fold python=3.10 -y
conda activate fold

# install PyTorch matching your CUDA version first, e.g.
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

git clone https://github.com/shpark97/Fold.git
cd Fold
pip install -e .
pip install libmr
```

### Data & Checkpoints

Benchmark data can be fetched with the OpenOOD downloading script (or automatically by the evaluator on first use):

```bash
python scripts/download/download.py \
    --contents 'datasets' 'checkpoints' \
    --datasets 'ood_v1.5' \
    --checkpoints 'ood_v1.5' \
    --save_dir './data' './results' \
    --dataset_mode 'benchmark'
```

Pre-trained classifiers (identical to the OpenOOD model zoo):

- **CIFAR-10 / CIFAR-100** — ResNet-18, 3 training runs (`s0`, `s1`, `s2`)
- **ImageNet-200** — ResNet-18, 3 training runs
- **ImageNet-1K** — torchvision ResNet-50 (downloaded automatically with `--tvs-pretrained`)

After downloading, checkpoints are expected at, e.g.,
`./results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt`.
ImageNet-1K training images must be obtained from the [official website](https://image-net.org/) and placed under `./data/images_largescale/imagenet_1k`.

## 📊 Evaluation

Run everything (4 ID datasets × 6 methods):

```bash
bash eval_fold.sh
```

Or individual runs:

```bash
# CIFAR-10 / CIFAR-100 / ImageNet-200
python scripts/eval_ood.py \
    --id-data cifar10 \
    --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
    --postprocessor fold \
    --batch-size 256 \
    --save-csv

# ImageNet-1K (torchvision ResNet-50)
python scripts/eval_ood_imagenet.py \
    --tvs-pretrained \
    --arch resnet50 \
    --postprocessor autofold \
    --batch-size 256 \
    --save-csv
```

Results are printed and (with `--save-csv`) written to `<root>/ood/<postprocessor>.csv`, averaged over the training runs for CIFAR/ImageNet-200.

### Other ImageNet-1K backbones

Beyond ResNet-50, the ImageNet-1K benchmark can be evaluated with the other torchvision backbones used in the paper (Table 5): RegNet-Y-16GF, DenseNet-121, Wide-ResNet-50-2, and ResNeXt-50-32x4d. Pretrained weights are downloaded automatically; no extra checkpoints are needed.

```bash
# via the runner (ARCH: resnet50 | regnet | densenet-121 | wrn50 | resnext50)
ARCH=regnet bash eval_fold.sh imagenet fold

# or directly
python scripts/eval_ood_imagenet.py \
    --tvs-pretrained \
    --arch densenet-121 \
    --postprocessor fold \
    --batch-size 256 \
    --save-csv
```

Results are written to `./results/imagenet_<arch>_tvsv1_base_default/ood/`.

## 📝 Citation

```bibtex
@inproceedings{park2026fold,
  title     = {Exploiting Local Flatness for Efficient Out-of-Distribution Detection},
  author    = {Park, Seonghwan and Jung, Hyunji and Lee, Dongyeop and Lee, Namhoon},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

Please also consider citing [OpenOOD](https://github.com/Jingkang50/OpenOOD) if you use this benchmark setup.

## 🙏 Acknowledgments

This codebase builds on the excellent [OpenOOD](https://github.com/Jingkang50/OpenOOD) benchmark. We thank its authors and contributors.

## License

MIT, following OpenOOD (see [LICENSE](LICENSE)).
