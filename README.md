## Overview
This repository contains the self-supervised-learning challenge submission for
**DAI3004: Learning Vision Intelligence** at Hanyang University ERICA.
We pre-train a Masked Autoencoder (MAE, ViT-B) on the STL-10 unlabeled split
(100k images) entirely from scratch, then evaluate the frozen encoder with the
course's fixed linear-probing protocol on STL-10 and CIFAR-10, under the strict
compute budget of a single NVIDIA RTX 4080 (24GB) GPU and a wall-clock training
time of approximately 72 hours per run.

The pipeline has three stages: (1) MAE pre-training on STL-10 unlabeled,
(2) frozen-encoder feature extraction, and (3) the fixed linear-probe
evaluation. The probe recipe (SGD lr=0.1, cosine, 100 epochs) is fixed for every
submission so the comparison stays fair.

## Structure
```
teamllm/
├── src/
│   ├── __init__.py
│   ├── models_mae.py
│   ├── engine_pretrain.py
│   ├── misc.py
│   ├── pos_embed.py
│   └── lr_sched.py
├── train.py
├── extract_features.py
├── evaluate.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Results
Linear-probe Top-1 accuracy of the frozen MAE encoder (mean-pooled patch tokens),
reported across three pre-training seeds:

| Seed           | STL-10 Top-1     | CIFAR-10 Top-1   |
|----------------|-----------------:|-----------------:|
| 0              | _TBD_            | _TBD_            |
| 42             | _TBD_            | _TBD_            |
| 2026           | _TBD_            | _TBD_            |
| **Mean ± Std** | **_TBD_ ± _TBD_**| **_TBD_ ± _TBD_**|

> `Seed` is the pre-training seed (`train.py --seed`). Fill in the `Top-1
> Accuracy` printed by `evaluate.py` for each seed, then report `Mean ± Std`
> (e.g. `76.54 ± 0.31`). The probe recipe is fixed: SGD lr=0.1,
> momentum=0.9, weight_decay=0.0, cosine annealing, 100 epochs, batch size 128,
> final-epoch evaluation.

## Installation
```
git clone https://github.com/gimyeonjik/teamllm.git

cd teamllm

pip install -r requirements.txt
```

(Optional) Download STL-10 and CIFAR-10 into `./data`.
```
python3 -c "import torchvision.datasets as d; d.STL10('./data', split='unlabeled', download=True); d.CIFAR10('./data', train=True, download=True)"
```

## Train
```
# Stage 1: MAE pre-training on STL-10 unlabeled (100k). Set the seed you want to N.
python3 train.py --seed {N} --epochs 900 --batch_size 64 --data_path ./data --output_dir ./checkpoints/seed{N}
```

## Feature Extraction
```
# Stage 2: frozen encoder -> (N, D) features + labels for STL-10 and CIFAR-10
python3 extract_features.py --ckpt ./checkpoints/seed{N}/checkpoint-899.pth --data_root ./data --out_dir ./features/seed{N}
```

## Evaluation
```
# Stage 3: fixed linear probe on the extracted features
python3 evaluate.py \
  --stl10-train-features   ./features/seed{N}/stl10_train_features.npy \
  --stl10-train-labels     ./features/seed{N}/stl10_train_labels.npy \
  --stl10-test-features    ./features/seed{N}/stl10_test_features.npy \
  --stl10-test-labels      ./features/seed{N}/stl10_test_labels.npy \
  --cifar10-train-features ./features/seed{N}/cifar10_train_features.npy \
  --cifar10-train-labels   ./features/seed{N}/cifar10_train_labels.npy \
  --cifar10-test-features  ./features/seed{N}/cifar10_test_features.npy \
  --cifar10-test-labels    ./features/seed{N}/cifar10_test_labels.npy
```
