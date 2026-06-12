# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F

import src.misc as misc
import src.lr_sched as lr_sched


class GatherLayer(torch.autograd.Function):
    """[우리 추가][U-MAE] backward 를 지원하는 all_gather (zhangq327/U-MAE loss_func.py 와 동일 역할).
    uniformity 를 전체 effective micro-batch (모든 GPU) 기준으로 계산하기 위함."""

    @staticmethod
    def forward(ctx, x):
        out = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(out, x)
        return tuple(out)

    @staticmethod
    def backward(ctx, *grads):
        return grads[dist.get_rank()].clone()


def uniformity_loss(features):
    """[우리 추가][U-MAE 공식 구현 그대로] L_unif = mean(cos_sim(z_i, z_j)^2).
    features: [B, D] (CLS token). DDP 면 전 GPU gather 후 계산."""
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        features = torch.cat(GatherLayer.apply(features), dim=0)
    features = F.normalize(features, dim=-1)
    sim = features @ features.T
    return sim.pow(2).mean()


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        # [우리 수정][U-MAE] uniformity_coef > 0 이면 CLS uniformity 규제 추가 (How Mask Matters, NeurIPS 2022)
        unif_coef = getattr(args, 'uniformity_coef', 0.0)
        with torch.cuda.amp.autocast():
            if unif_coef > 0:
                loss_recon, _, _, cls_feat = model(samples, mask_ratio=args.mask_ratio, return_cls=True)
                loss_unif = uniformity_loss(cls_feat)
                loss = loss_recon + unif_coef * loss_unif
            else:
                loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        if unif_coef > 0:  # [우리 추가][U-MAE] 두 항 분리 모니터링 (붕괴/규제 작동 확인용)
            metric_logger.update(loss_recon=loss_recon.item(), loss_unif=loss_unif.item())

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}