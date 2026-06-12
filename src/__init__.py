"""
  models_mae.py      MAE 모델 (ViT-B encoder + decoder, patch 6 / 96px / 256 tokens)
  engine_pretrain.py 1-epoch 학습 루프 (grad accum, U-MAE uniformity 옵션 포함)
  misc.py            DDP 헬퍼, NativeScaler, 로깅, 체크포인트 저장/복원
  pos_embed.py       2D sin-cos positional embedding
  lr_sched.py        warmup + cosine LR schedule
"""
