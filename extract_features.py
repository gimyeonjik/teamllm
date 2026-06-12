"""
[우리 추가] Frozen MAE encoder → linear-probe 용 feature (N, D) 추출.

pretrain 으로 학습한 MAE encoder 를 freeze 하고, STL-10 / CIFAR-10 의
labeled split 을 통과시켜 evaluate.py 가 받는 (N, D) feature + (N,) label 을 저장한다.

해상도 처리:
  - STL-10  : 이미 96x96 → 그대로
  - CIFAR-10: 32x32 → 96x96 resize (pretrain 과 동일 token grid 256 유지)

Feature pooling:
  - mean : encoder 출력 patch token 평균 (CLS 제외)  [기본, MAE linear probe 관행]
  - cls  : CLS token

출력:
  {out_dir}/stl10_train_features.npy   (5000, D)
  {out_dir}/stl10_train_labels.npy     (5000,)
  {out_dir}/stl10_test_features.npy    (8000, D)
  {out_dir}/stl10_test_labels.npy      (8000,)
  {out_dir}/cifar10_train_features.npy (50000, D)
  {out_dir}/cifar10_train_labels.npy   (50000,)
  {out_dir}/cifar10_test_features.npy  (10000, D)
  {out_dir}/cifar10_test_labels.npy    (10000,)
"""
import argparse
import os

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import models_mae

# 정규화 통계 (pretrain 과 일관). 각 데이터셋 고유 통계 사용.
STL10_MEAN, STL10_STD = (0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)
CIFAR10_MEAN, CIFAR10_STD = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def build_transform(dataset: str, img_size: int = 96):
    """eval transform: augmentation 없음. CIFAR 만 96 으로 resize."""
    if dataset == "stl10":
        ops = [T.ToTensor(), T.Normalize(STL10_MEAN, STL10_STD)]
    else:  # cifar10
        ops = [T.Resize(img_size, interpolation=T.InterpolationMode.BICUBIC),
               T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    return T.Compose(ops)


def build_dataset(dataset: str, split: str, data_root: str, img_size: int):
    tf = build_transform(dataset, img_size)
    if dataset == "stl10":
        # STL-10: linear probe 는 labeled 'train'(5k) / 'test'(8k)
        return datasets.STL10(data_root, split=split, download=False, transform=tf)
    else:
        train = (split == "train")
        return datasets.CIFAR10(data_root, train=train, download=False, transform=tf)


@torch.no_grad()
def extract(model, loader, device, pool: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    feats, labels = [], []
    for images, target in tqdm(loader, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        # mask_ratio=0 → 모든 256 token 유지. 반환 x: [B, 1+256, D]
        with torch.cuda.amp.autocast():
            latent, _, _ = model.forward_encoder(images, mask_ratio=0.0)
        if pool == "cls":
            f = latent[:, 0]                       # CLS token
        else:
            f = latent[:, 1:, :].mean(dim=1)       # patch token mean-pool
        feats.append(f.float().cpu())
        labels.append(target.clone())
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()


def main():
    p = argparse.ArgumentParser("MAE feature extraction for linear probing")
    p.add_argument("--ckpt", required=True, help="pretrain checkpoint (.pt) 경로")
    p.add_argument("--model", default="mae_vit_base_patch6")
    p.add_argument("--input_size", type=int, default=96)
    p.add_argument("--data_root", default="data")
    p.add_argument("--out_dir", default="features")
    p.add_argument("--pool", choices=["mean", "cls"], default="mean")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 모델 빌드 + checkpoint 로드 (encoder 만 필요하지만 전체 로드 후 freeze)
    model = models_mae.__dict__[args.model](norm_pix_loss=True, img_size=args.input_size).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)  # ckpt 에 argparse.Namespace 포함
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    print(f"[load] {args.ckpt}  pool={args.pool}")

    jobs = [
        ("stl10", "train"), ("stl10", "test"),
        ("cifar10", "train"), ("cifar10", "test"),
    ]
    for dataset, split in jobs:
        ds = build_dataset(dataset, split, args.data_root, args.input_size)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        feats, labels = extract(model, loader, device, args.pool)
        fpath = os.path.join(args.out_dir, f"{dataset}_{split}_features.npy")
        lpath = os.path.join(args.out_dir, f"{dataset}_{split}_labels.npy")
        np.save(fpath, feats)
        np.save(lpath, labels)
        print(f"[save] {dataset}/{split}: feats {feats.shape} → {fpath}")

    print("\n완료. evaluate.py 실행 예:")
    print(f"  python evaluate.py \\")
    print(f"    --stl10-train-features {args.out_dir}/stl10_train_features.npy \\")
    print(f"    --stl10-train-labels   {args.out_dir}/stl10_train_labels.npy \\")
    print(f"    --stl10-test-features  {args.out_dir}/stl10_test_features.npy \\")
    print(f"    --stl10-test-labels    {args.out_dir}/stl10_test_labels.npy \\")
    print(f"    --cifar10-train-features {args.out_dir}/cifar10_train_features.npy \\")
    print(f"    --cifar10-train-labels   {args.out_dir}/cifar10_train_labels.npy \\")
    print(f"    --cifar10-test-features  {args.out_dir}/cifar10_test_features.npy \\")
    print(f"    --cifar10-test-labels    {args.out_dir}/cifar10_test_labels.npy")


if __name__ == "__main__":
    main()
