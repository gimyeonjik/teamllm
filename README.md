# MAE — Self-Supervised Learning Challenge (STL-10 → CIFAR-10 / STL-10 Linear Probing)

한양대학교 시각지능학습 SSL 챌린지 제출물.
**STL-10 unlabeled 100k 만으로 MAE (Masked Autoencoder) 를 from scratch 학습**하고,
frozen encoder 의 feature 를 챌린지 제공 `evaluate.py` (고정 linear probing) 로 평가한다.

[facebookresearch/mae](https://github.com/facebookresearch/mae) 기반, 코드 내 `# [우리 수정]` 주석으로 모든 변경점 표기.

## 챌린지 규칙 준수

- ✅ SSL pretraining 은 **STL-10 unlabeled split (100k) 만** 사용
- ✅ **From scratch** — 외부 pretrained weight 일절 사용 안 함
- ✅ 평가 = **Linear probing only**, `evaluate.py` 무수정 (md5 `84acc8e4…`)
- ✅ Test-time adaptation 없음 (deterministic feature 추출)
- ✅ 단일 GPU 학습 지원 (gradient accumulation 으로 effective batch 유지)

## Structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE                  # Apache 2.0 (원본 facebookresearch/mae)
├── train.py                 # MAE pretraining 진입점 (단일/멀티 GPU)
├── extract_features.py      # frozen encoder → linear-probe 용 feature (N, D) 추출
├── evaluate.py              # 챌린지 제공 고정 linear probe (수정 금지)
└── src/
    ├── __init__.py
    ├── models_mae.py        # MAE 모델 (ViT-B/patch6, decoder 8×512)
    ├── engine_pretrain.py   # 학습 루프 (grad accum, U-MAE uniformity 옵션)
    ├── misc.py              # DDP/AMP/체크포인트 헬퍼
    ├── pos_embed.py         # 2D sin-cos positional embedding
    └── lr_sched.py          # warmup + cosine LR
```

## 모델 / 학습 Recipe

| 항목 | 값 | 비고 |
|------|-----|------|
| Encoder | ViT-Base (depth 12, dim 768, heads 12) | |
| Decoder | depth 8, dim 512, heads 16 | 원논문 그대로 |
| Input / Patch | 96 × 96 / **6** → 16×16 = **256 tokens** | STL native 해상도, 원논문 224/16 의 196 token regime 유지 |
| **Mask ratio** | **0.85** | 원논문 0.75 에서 상향 |
| Recon target | per-patch normalized pixels (`--norm_pix_loss`) | 원논문 그대로 |
| Optimizer | AdamW, betas (0.9, 0.95), wd 0.05 | 원논문 그대로 |
| LR | blr 1.5e-4 → `lr = blr × eff_batch / 256` 자동 | 원논문 linear scaling |
| Warmup / Schedule | 40 epoch / cosine | 원논문 그대로 |
| Effective batch | 4096 (gradient accumulation 으로 달성) | 원논문 그대로 |
| Augmentation | RandomResizedCrop(0.2, 1.0) + HFlip | 원논문 그대로 (minimal) |
| Normalization | STL-10 stats (0.4467, 0.4398, 0.4066) | |
| Precision | AMP FP16 | |

## 재현 방법

### 1. 환경
```bash
pip install -r requirements.txt
# 검증 환경: torch 2.7.1+cu118, torchvision 0.22.1, timm 1.0.26, numpy 1.26.4
```

### 2. 데이터 준비 (torchvision 자동 다운로드, ~3GB)
```bash
mkdir -p data && python -c "
import torchvision.datasets as d
d.STL10('data', split='unlabeled', download=True)
d.STL10('data', split='train', download=True)
d.STL10('data', split='test', download=True)
d.CIFAR10('data', train=True, download=True)
d.CIFAR10('data', train=False, download=True)
"
```

### 3. Pretraining

**단일 GPU (챌린지 공식 환경, 3일 budget)**
```bash
torchrun --nproc_per_node=1 train.py \
  --batch_size 64 --accum_iter 64 \
  --model mae_vit_base_patch6 --input_size 96 --norm_pix_loss \
  --epochs 400 --warmup_epochs 40 \
  --data_path data --output_dir ./checkpoints --log_dir ""
# effective batch = 64 × 64 × 1 = 4096, lr = 2.4e-3 자동, mask_ratio = 0.85 (기본값)
```

**Multi-GPU (개발용 빠른 학습, 예: 8 GPU)**
```bash
torchrun --nproc_per_node=8 train.py \
  --batch_size 64 --accum_iter 8 \
  --model mae_vit_base_patch6 --input_size 96 --norm_pix_loss \
  --epochs 800 --warmup_epochs 40 \
  --data_path data --output_dir ./checkpoints --log_dir ""
# effective batch = 64 × 8 × 8 = 4096
```

체크포인트는 100 epoch 마다 `checkpoints/checkpoint-{N}.pth` 저장 (~1.3GB/개).
Resume: `--resume checkpoints/checkpoint-{N}.pth` (warmup 재발생 없음 — LR 은 epoch 로 직접 계산).

### 4. Feature 추출 (frozen encoder, mean-pool)
```bash
python extract_features.py \
  --ckpt checkpoints/checkpoint-399.pth \
  --out_dir features --data_root data \
  --pool mean --batch_size 256
```
- STL-10: 96 native / CIFAR-10: 32→96 bicubic resize (동일 256-token grid 유지)
- 출력: `features/{stl10,cifar10}_{train,test}_{features,labels}.npy`

### 5. Linear Probing 평가 (챌린지 고정 protocol)
```bash
python evaluate.py \
  --stl10-train-features  features/stl10_train_features.npy \
  --stl10-train-labels    features/stl10_train_labels.npy \
  --stl10-test-features   features/stl10_test_features.npy \
  --stl10-test-labels     features/stl10_test_labels.npy \
  --cifar10-train-features features/cifar10_train_features.npy \
  --cifar10-train-labels   features/cifar10_train_labels.npy \
  --cifar10-test-features  features/cifar10_test_features.npy \
  --cifar10-test-labels    features/cifar10_test_labels.npy
```

## 원본 (facebookresearch/mae) 대비 변경 사항

### Recipe 적응 (STL-10 / 96px)
- 모델: `mae_vit_base_patch6` builder 신규 (patch 16 → **6**, 96px 에서 256 tokens)
- 데이터: ImageFolder(ImageNet) → `STL10(split='unlabeled')`, ImageNet stats → STL-10 stats
- **Mask ratio: 0.75 → 0.85**
- 체크포인트 저장 주기: 20 → 100 epoch

### 호환성 패치 (timm 1.0 / torch 2.7 / numpy ≥1.24)
- `qk_scale` 인자 제거, `torch._six` → `torch`, `np.float` → `np.float64`
- `timm==0.3.2` assert 제거, `add_weight_decay` 로컬 구현, tensorboard optional
- `torch.load(..., weights_only=False)` (torch 2.6+ 기본값 변경 대응)

### 구조 변경
- `main_pretrain.py` → `train.py`, `util/` → `src/` (flat module)
- `extract_features.py` 신규 (challenge eval 인터페이스)
- (옵션) `--uniformity_coef λ` — U-MAE 의 CLS uniformity 규제 (기본 0.0 = vanilla MAE)

## License

원본 [facebookresearch/mae](https://github.com/facebookresearch/mae) 의 Apache 2.0 (`LICENSE`).
