import time
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import cv2, os

__all__ = ['get_optimizer', 'get_loss_fn', 'save_imgs', 'get_lr_scheduler', 'test_fps']


# -----------------------------
# Optimizer
# -----------------------------
def _iter_trainable(params):
    for p in params:
        if p is None:
            continue
        if hasattr(p, "requires_grad"):
            if p.requires_grad:
                yield p
        else:
            # 万一传进来的是奇怪对象，直接略过
            continue

def get_optimizer(args, model_or_params):
    """
    兼容两种调用方式：
      - get_optimizer(args, model)
      - get_optimizer(args, model.parameters())
    并且只优化 requires_grad=True 的参数。
    """
    params = model_or_params.parameters() if hasattr(model_or_params, "parameters") else model_or_params
    params = list(_iter_trainable(params))

    opt_name = str(getattr(args, "optimizer", "adamw")).lower()
    lr = float(args.lr)
    wd = float(getattr(args, "weight_decay", 0.0))

    if opt_name == 'sgd':
        return SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    elif opt_name == 'adamw':
        return AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=wd)
    elif opt_name == 'adam':
        return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


# -----------------------------
# Loss
# -----------------------------
def get_loss_fn(loss_fn_name):
    class _CEFocalLoss(nn.Module):
        def __init__(self, alpha=0.80, gamma=2.0, label_smoothing=0.1,
                     reduction='mean', ce_weight=0.7):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.label_smoothing = label_smoothing
            self.reduction = reduction
            self.ce_weight = ce_weight

        def forward(self, inputs, targets):
            if targets.dim() == 4:
                targets = torch.squeeze(targets, dim=1)
            targets = targets.long()

            ce_loss = F.cross_entropy(
                inputs, targets,
                reduction=self.reduction,
                label_smoothing=self.label_smoothing
            )
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
            combined = self.ce_weight * ce_loss + (1 - self.ce_weight) * focal_loss

            if self.reduction == 'mean':
                return combined.mean()
            elif self.reduction == 'sum':
                return combined.sum()
            else:
                return combined  # 'none'

    class _CEDiceLoss(nn.Module):
        def __init__(self, ce_weight=1.0, dice_weight=1.0,
                     label_smoothing=0.1, reduction='mean', ignore_index=255):
            super().__init__()
            self.label_smoothing = 0.0
            self.reduction = reduction
            self.ce_weight = ce_weight
            self.ignore_index = ignore_index

        def forward(self, inputs, targets):
            targets = targets.long()
            if targets.dim() == 4:
                targets = torch.squeeze(targets, dim=1)
            if inputs.shape[-1] != targets.shape[-1]:
                inputs = F.interpolate(inputs, size=targets.shape[1:], mode='bilinear', align_corners=True)

            ce_loss = F.cross_entropy(input=inputs, target=targets,
                                      label_smoothing=0.1,
                                      ignore_index=self.ignore_index,
                                      reduction=self.reduction)
            prob_class1 = torch.softmax(inputs, dim=1)[:, 1, :, :].contiguous()
            # 这里按二分类标签 0/1 计算 dice；若标签是 {0,1} 的 long，需转 float
            tgt_float = (targets == 1).float()
            inter = (prob_class1 * tgt_float).sum()
            eps = 1e-5
            dice = (2.0 * inter + eps) / (prob_class1.sum() + tgt_float.sum() + eps)
            return 0.3 * (1.0 - dice) + 0.7 * ce_loss

    def _cross_entropy(output, target, weight=None, reduction='mean', ignore_index=255):
        if target.dim() == 4:
            target = torch.squeeze(target, dim=1)
        if output.shape[-1] != target.shape[-1]:
            output = F.interpolate(output, size=target.shape[1:], mode='bilinear', align_corners=True)

        return F.cross_entropy(input=output, target=target,
                               weight=weight,
                               label_smoothing=0.1,
                               ignore_index=ignore_index,
                               reduction=reduction)

    def _bce_dice_loss(inputs, targets):
        bce = F.binary_cross_entropy(inputs, targets)
        inter = (inputs * targets).sum()
        eps = 1e-5
        dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
        return bce + 1 - dice

    if loss_fn_name == 'ce':
        return _cross_entropy
    elif loss_fn_name == 'cefocal':
        return _CEFocalLoss()
    elif loss_fn_name == 'bcedice':
        return _bce_dice_loss
    elif loss_fn_name == 'cedice':
        return _CEDiceLoss()
    else:
        raise NotImplementedError(f"Unknown loss: {loss_fn_name}")


# -----------------------------
# LR Scheduler
# -----------------------------
class MyLambdaLR(LambdaLR):
    def __init__(self, optimizer, total_iters, lr_mode='poly', lr_factor=1.0,
                 min_lr=1e-6, phase_ratio=0.67, last_epoch=-1):
        self.total_iters = max(1, int(total_iters))
        self.lr_mode = lr_mode
        self.lr_factor = lr_factor
        self.min_lr = min_lr
        self.phase_ratio = phase_ratio
        self.base_lr = optimizer.param_groups[0]['lr']

        if lr_mode == 'poly':
            self.poly_factor1 = 1.3
            self.phase1_steps = int(self.total_iters * phase_ratio)
            self.phase2_steps = max(1, self.total_iters - self.phase1_steps)
            self.phase1_end_lr = self.base_lr * (1 - phase_ratio) ** self.poly_factor1
        elif lr_mode == 'linear':
            self.phase1_steps = int(self.total_iters * phase_ratio)
            self.phase2_steps = max(1, self.total_iters - self.phase1_steps)
            self.phase1_end_lr = self.base_lr * 0.1

        super().__init__(optimizer=optimizer, lr_lambda=self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, current_step):
        current_step = min(max(0, current_step), self.total_iters)
        if self.lr_mode == 'poly':
            lr = self.base_lr * (1 - current_step / self.total_iters) ** 1.3
        elif self.lr_mode == 'linear':
            if current_step < self.phase1_steps:
                phase1_progress = current_step / max(1, self.phase1_steps)
                lr = self.base_lr * (1 - phase1_progress) ** 1.3
            else:
                decay = (self.phase1_end_lr - self.min_lr) / max(1, self.phase2_steps)
                lr = self.phase1_end_lr - (current_step - self.phase1_steps) * decay
                lr = max(lr * self.lr_factor, self.min_lr)
        else:
            raise ValueError(f"Unsupported mode: {self.lr_mode}")

        # 返回 scale 因子
        return lr / max(1e-12, self.base_lrs[0])

def get_lr_scheduler(args, optimizer):
    return MyLambdaLR(
        optimizer,
        total_iters=args.total_iter,
        lr_mode=args.lr_mode,
        lr_factor=getattr(args, "lr_factor", 1.0),
        min_lr=float(args.lr) * 1e-2,
        phase_ratio=0.7,
    )


# -----------------------------
# Save images (robust)
# -----------------------------
def save_imgs(path, vis_list):
    path = Path(path)
    (path / 'pred').mkdir(parents=True, exist_ok=True)
    (path / 'label').mkdir(parents=True, exist_ok=True)

    dtype = np.dtype([
        ('img_name', 'U50'),
        ('f1', float),
        ('pred', np.float32, (1, 256, 256)),
        ('label', np.int64, (1, 256, 256))])

    data_tuples = []
    for d in vis_list:
        img_name = d['img_name']
        f1 = float(d['f1'])
        pred = d['pred'].squeeze(0).cpu().numpy()
        label = d['label'].squeeze(0).cpu().numpy()
        data_tuples.append((img_name, f1, pred, label))

    vis_imgs = np.array(data_tuples, dtype=dtype)
    vis_imgs = vis_imgs[vis_imgs['f1'].argsort()[::-1]]
    for i in range(len(vis_imgs)):
        mask = vis_imgs[i]['pred'].squeeze(0)
        label = vis_imgs[i]['label'].squeeze(0)
        file_name = f"{str(i).zfill(4)}_{vis_imgs[i]['img_name']}"
        cv2.imwrite(str(path / 'pred' / file_name), (mask * 255).astype(np.uint8))
        cv2.imwrite(str(path / 'label' / file_name), (label * 255).astype(np.uint8))


# -----------------------------
# FPS test (DDP/DP-safe)
# -----------------------------
def _unwrap(model):
    return model.module if hasattr(model, "module") else model

def test_fps(model, dual_input=False, device='cuda',
             warmup=20, iters=200, use_amp=True, use_channels_last=True):
    """
    - 自动解包 DDP/DP 外壳
    - 自适配双输入签名：优先尝试 model((x1,x2))，失败再试 model(x1,x2)
    - 只在单进程下调用；多卡时请在 rank0 调用
    """
    m = _unwrap(model)
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'

    m = m.to(device).eval()

    # 构造输入
    H = W = 256
    for H, W in zip([128, 256, 512, 1024], [128, 256, 512, 1024]):
      if dual_input:
        x1 = torch.randn(1, 3, H, W, device=device)
        x2 = torch.randn(1, 3, H, W, device=device)
      else:
          x = torch.randn(1, 3, H, W, device=device)
  
      # 可选 channels_last
      if use_channels_last:
          m = m.to(memory_format=torch.channels_last)
          if dual_input:
              x1 = x1.to(memory_format=torch.channels_last)
              x2 = x2.to(memory_format=torch.channels_last)
          else:
              x = x.to(memory_format=torch.channels_last)
  
      # 半精度推理（仅 CUDA 生效）
      if device == 'cuda':
          m = m.half()
          if dual_input:
              x1 = x1.half(); x2 = x2.half()
          else:
              x = x.half()
  
      @torch.no_grad()
      def forward_once():
          if dual_input:
              # 先按你训练时的用法尝试：model((pre, post))
              try:
                  return m((x1, x2))
              except Exception:
                  return m(x1, x2)
          else:
              return m(x)
  
      # 预热
      with torch.no_grad():
          for _ in range(warmup):
              with torch.cuda.amp.autocast(enabled=(use_amp and device=='cuda'), dtype=torch.float16):
                  _ = forward_once()
      if device == 'cuda':
          torch.cuda.synchronize()
  
      # 计时
      times = []
      with torch.no_grad():
          for _ in range(iters):
              t0 = time.time()
              with torch.cuda.amp.autocast(enabled=(use_amp and device=='cuda'), dtype=torch.float16):
                  _ = forward_once()
              if device == 'cuda':
                  torch.cuda.synchronize()
              times.append((time.time() - t0) * 1000)
  
      avg_ms = sum(times) / len(times)
      fps = 1000.0 / avg_ms
      print(f"[{H}x{W}, dual={dual_input}] {avg_ms:.2f} ms | {fps:.2f} FPS (FP16={use_amp and device=='cuda'}, ch_last={use_channels_last})")
    return fps, avg_ms
