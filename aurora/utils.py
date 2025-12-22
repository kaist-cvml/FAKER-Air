
from __future__ import annotations

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import os, sys
import numpy as np
import random

import math
import json
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, Sequence
import torch
import torch.nn.functional as F

# ---------- Enhanced metric computation for both pollutants ----------
def compute_metrics(pred, label, pollutant_name="PM2.5", threshold=35):
    """
    Compute metrics for a specific pollutant with appropriate threshold.
    
    Args:
        pred: Predicted values
        label: Ground truth values
        pollutant_name: Name of the pollutant ('PM2.5' or 'PM10')
        threshold: Threshold for binary classification
    """
    # Convert to binary masks
    pred_binary = pred > threshold
    label_binary = label > threshold

    acc = np.mean(pred_binary == label_binary)
    f1 = f1_score(label_binary, pred_binary)
    precision = precision_score(label_binary, pred_binary)
    recall = recall_score(label_binary, pred_binary)
    
    # Compute confusion matrix: rows = GT, columns = Prediction
    cm = confusion_matrix(label_binary, pred_binary, labels=[0, 1])
    # Normalize row-wise (each row sums to 1)
    cm_normalized = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-6)

    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]
    false_alarm_rate = FP / (FP + TN) if (FP + TN) > 0 else 0
    detection_rate = TP / (TP + FN) if (TP + FN) > 0 else 0

    ret = {
        "pollutant": pollutant_name,
        "threshold": threshold,
        "accuracy": round(acc * 100, 2),
        "f1": round(f1 * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "false_alarm_rate": round(false_alarm_rate * 100, 2),
        "detection_rate": round(detection_rate * 100, 2),
        "confusion_matrix": cm_normalized,
        "confusion_matrix_raw": cm,
    }
    return ret

AQI_BINS_DEFAULT = {
    "PM2.5": [15, 35, 75],      # (좋음,보통,나쁨,아주나쁨) 경계: [t1, t2, t3]
    "PM10" : [30, 80, 150],
    "O3"   : [0.03, 0.09, 0.15],
}

AQI_CLASS_NAMES = ["good", "moderate", "bad", "very_bad"]

def _to_4classes(x: np.ndarray, bins3: list[float]) -> np.ndarray:
    """
    bins3 = [t1, t2, t3]
      class 0: x < t1
      class 1: t1 <= x < t2
      class 2: t2 <= x < t3
      class 3: x >= t3
    """
    t1, t2, t3 = bins3
    return np.digitize(x, bins=[t1, t2, t3], right=False)  # returns 0..3

def compute_multiclass_metrics(
    pred: np.ndarray,
    label: np.ndarray,
    pollutant_name: str = "PM2.5",
    bins: list[float] | None = None,
    class_names: list[str] = AQI_CLASS_NAMES,
) -> dict:
    """
    (좋음, 보통, 나쁨, 아주 나쁨) 4단계 다중분류 지표를 계산합니다.

    Returns:
      {
        "pollutant": "...",
        "bins": [t1, t2, t3],
        "overall": { "accuracy": ..., "f1_macro": ..., "f1_weighted": ..., "f1_micro": ... },
        "per_class": {
            "good":      {"support": n, "precision": ..., "recall": ..., "f1": ..., "far": ..., "acc_ovr": ...},
            "moderate":  {...}, ...
        },
        "confusion_matrix_raw": (4x4 ndarray, rows=GT, cols=Pred),
        "confusion_matrix_row_norm": (행 정규화 4x4)
      }
    """
    key = pollutant_name.upper().replace(" ", "")
    if bins is None:
        bins = AQI_BINS_DEFAULT.get(key, AQI_BINS_DEFAULT["PM2.5"])

    pred = np.asarray(pred).ravel()
    label = np.asarray(label).ravel()

    y_true = _to_4classes(label, bins)
    y_pred = _to_4classes(pred,  bins)

    # 4x4 혼동행렬
    labels = [0, 1, 2, 3]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_row = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-6)

    N = cm.sum()
    correct = np.trace(cm)
    overall_acc = correct / (N + 1e-12)

    # micro/macro/weighted F1
    f1_macro    = f1_score(y_true, y_pred, average="macro", labels=labels)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", labels=labels)
    f1_micro    = f1_score(y_true, y_pred, average="micro", labels=labels)  # multi-class에선 accuracy와 동일

    per_class = {}
    for i, name in enumerate(class_names):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = N - TP - FN - FP

        prec   = TP / (TP + FP + 1e-12)
        rec    = TP / (TP + FN + 1e-12)
        f1     = 2 * TP / (2 * TP + FP + FN + 1e-12)
        far    = FP / (FP + TN + 1e-12)        # False Alarm Rate (one-vs-rest)
        acc_i  = (TP + TN) / (N + 1e-12)       # one-vs-rest accuracy
        supp   = cm[i, :].sum()                # GT support

        per_class[name] = {
            "support": int(supp),
            "precision": round(prec * 100, 2),
            "recall":    round(rec  * 100, 2),
            "f1":        round(f1   * 100, 2),
            "far":       round(far  * 100, 2),
            "acc_ovr":   round(acc_i* 100, 2),
        }

    return {
        "pollutant": pollutant_name,
        "bins": bins,
        "overall": {
            "accuracy":     round(overall_acc * 100, 2),
            "f1_macro":     round(f1_macro    * 100, 2),
            "f1_weighted":  round(f1_weighted * 100, 2),
            "f1_micro":     round(f1_micro    * 100, 2),
        },
        "per_class": per_class,
        "confusion_matrix_raw": cm,
        "confusion_matrix_row_norm": cm_row,
    }

def build_step_weights(T: int,
                       mode: str = "fixed",
                       base: float = 0.75,
                       gamma: float = 0.9,
                       normalize: bool = True,
                       device: torch.device | None = None) -> torch.Tensor:
    """
    Build a length-T vector of step weights. By default it reproduces the
    current behaviour (fixed: w0=1.0, others=base).
    The vector is optionally normalized to sum to T (keeps loss scale stable).
    """
    assert T >= 1
    if T == 1:
        w = torch.tensor([1.0], device=device)
    elif mode == "fixed":
        w = torch.tensor([1.0] + [base] * (T - 1), device=device)
    elif mode == "linear":
        # linearly decay from 1.0 to base
        w = torch.linspace(1.0, base, steps=T, device=device)
    elif mode == "exp":
        # exponential decay: 1, gamma, gamma^2, ...
        w = torch.tensor([gamma**i for i in range(T)], device=device)
        w[0] = 1.0
    elif mode == "sigmoid":
        # S-curve front-load (k controls steepness via gamma)
        k = max(1e-3, gamma)
        t = torch.linspace(-2.0, 2.0, steps=T, device=device)
        s = torch.sigmoid(k * t)
        # map to [base, 1.0] with s[0] ~ base, s[-1] ~ 1.0
        s = (s - s.min()) / (s.max() - s.min() + 1e-8)
        w = base + (1.0 - base) * s
    else:
        raise ValueError(f"Unknown rollout weight mode: {mode}")

    if normalize and w.sum() > 0:
        w = w * (T / w.sum())
    return w


@dataclass
class DynamicLossScheduler:
    """
    Month/case aware multipliers that gently re-weight loss groups.
    All multipliers default to 1.0 (no behaviour change).
    """
    month_multipliers: Dict[int, float]  # e.g., {12:1.2, 1:1.2, 2:1.1}
    severity_alpha: float = 0.0          # 0 disables case-severity boost
    severity_threshold: float = 0.05     # baseline exceedance fraction
    pattern_warmup_epochs: int = 0
    pattern_max_scale: float = 1.0

    def _month_factor(self, times: Tuple) -> float:
        if not times:
            return 1.0
        vals = []
        for t in times:
            m = getattr(t, "month", None)
            vals.append(self.month_multipliers.get(int(m), 1.0) if m is not None else 1.0)
        return float(sum(vals) / max(1, len(vals)))

    def _severity_factor(self, tgt_batch) -> float:
        """Boost PM group when exceedance is widespread (softly, capped)."""
        if self.severity_alpha <= 0.0:
            return 1.0
        boost = 1.0

        def frac_exceed(var: str, thr: float) -> float:
            if var not in tgt_batch.surf_vars:
                return 0.0
            g = tgt_batch.surf_vars[var][:, 0]  # [B,H,W], first step
            return float((g > thr).float().mean().item())

        f25 = frac_exceed("pm2p5", 35.0)
        f10 = frac_exceed("pm10", 80.0)
        f = max(f25, f10)
        if f > self.severity_threshold:
            # normalize to [0,1] above threshold
            x = min(1.0, (f - self.severity_threshold) / max(1e-6, 1.0 - self.severity_threshold))
            boost = 1.0 + self.severity_alpha * x
        return boost

    def pattern_scale(self, epoch: int) -> float:
        if self.pattern_warmup_epochs <= 0:
            return self.pattern_max_scale
        x = max(0.0, min(1.0, epoch / float(self.pattern_warmup_epochs)))
        return (x * x) * self.pattern_max_scale   # quadratic warmup

    def group_multipliers(self, tgt_i, epoch: int) -> Dict[str, float]:
        """
        Return per-group multipliers for this mini-batch/step.
        Keys: 'pm', 'obssec', 'cmaq', 'era5', 'pattern_scale'
        """
        month_k = self._month_factor(getattr(tgt_i.metadata, "time", ()))
        sev_k   = self._severity_factor(tgt_i)
        pm_k    = month_k * sev_k
        return {
            "pm": pm_k,
            "obssec": 1.0,     # stay conservative (no change unless you want it)
            "cmaq": 1.0,
            "era5": 1.0,
            "pattern_scale": self.pattern_scale(epoch),
        }

def parse_month_weights(s: str) -> Dict[int, float]:
    """
    Parse JSON or empty string. Example:
    s = '{"12":1.2,"1":1.2,"2":1.1}'
    """
    if not s:
        return {}
    d = json.loads(s)
    return {int(k): float(v) for k, v in d.items()}

EPS = 1e-8

def _to_B1HW(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is [B,1,H,W]. Accepts [B,T,H,W] or [B,H,W]."""
    if x.dim() == 4:
        # assume [B,T,H,W] -> use first time step by default
        return x[:, :1]
    elif x.dim() == 3:
        return x.unsqueeze(1)
    elif x.dim() == 5:
        raise ValueError("Expected [B,T,H,W] or [B,H,W], got 5D.")
    return x  # [B,1,H,W]

def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean over H,W with optional mask [B,1,H,W] -> scalar."""
    if mask is None:
        return x.mean()
    num = (x * mask).sum(dim=(-1, -2, -3))
    den = mask.sum(dim=(-1, -2, -3)).clamp_min(1.0)
    return (num / den).mean()

def multiscale_mse(pred: torch.Tensor, target: torch.Tensor,
                   mask: torch.Tensor | None = None,
                   scales: Iterable[int] = (1, 2, 4)) -> torch.Tensor:
    """
    Multi-scale (blur/downsample) MSE using average pooling as a cheap blur.
    Larger scales give shift tolerance.
    """
    p = _to_B1HW(pred); g = _to_B1HW(target)
    m = _to_B1HW(mask) if mask is not None else None

    loss, wsum = 0.0, 0.0
    for s in scales:
        if s <= 1:
            ps, gs = p, g
            ms = m
        else:
            ps = F.avg_pool2d(p, kernel_size=s, stride=s, padding=0)
            gs = F.avg_pool2d(g, kernel_size=s, stride=s, padding=0)
            ms = None if m is None else F.avg_pool2d(m, kernel_size=s, stride=s, padding=0).clamp_max(1.0)
        l = _masked_mean((ps - gs) ** 2, ms)
        w = 1.0 / s  # emphasize coarse scales mildly
        loss += w * l
        wsum += w
    return loss / (wsum + EPS)

def _translate2d(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """
    Translate by (dy,dx) with border replication. x: [B,1,H,W]
    """
    B, C, H, W = x.shape
    pad = (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0))  # (left,right,top,bottom)
    x_pad = F.pad(x, pad, mode="replicate")
    y = x_pad[:, :, max(-dy, 0):max(-dy, 0)+H, max(-dx, 0):max(-dx, 0)+W]
    return y

def soft_shift_mse(pred: torch.Tensor, target: torch.Tensor,
                   mask: torch.Tensor | None = None,
                   radius: int = 1, temperature: float = 0.5) -> torch.Tensor:
    """
    Soft-min over small integer shifts: loss = softmin_{|Δ|<=r} MSE(pred, shift(target,Δ))
    Using log-sum-exp as a smooth min.
    """
    p = _to_B1HW(pred); g = _to_B1HW(target)
    m = _to_B1HW(mask) if mask is not None else None

    losses = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            gs = _translate2d(g, dy, dx)
            ms = None if m is None else _translate2d(m, dy, dx)
            l = _masked_mean((p - gs) ** 2, ms)  # scalar
            losses.append(l.unsqueeze(0))
    L = torch.cat(losses, dim=0)  # [K], scalar per-batch already averaged

    # softmin: -T * logsumexp(-L/T)
    softmin = -temperature * torch.logsumexp(-L / temperature, dim=0)
    return softmin

def gradient_loss(pred: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    First-order gradient (finite differences) loss to match shapes/edges.
    """
    p = _to_B1HW(pred); g = _to_B1HW(target)
    m = _to_B1HW(mask) if mask is not None else None

    def grads(x):
        dx = x[..., :, 1:] - x[..., :, :-1]
        dy = x[..., 1:, :] - x[..., :-1, :]
        return dx, dy

    px, py = grads(p)
    gx, gy = grads(g)

    if m is None:
        return ((px - gx) ** 2).mean() + ((py - gy) ** 2).mean()

    # build interior masks for differences
    mx = m[..., :, 1:] * m[..., :, :-1]
    my = m[..., 1:, :] * m[..., :-1, :]
    lx = _masked_mean((px - gx) ** 2, mx)
    ly = _masked_mean((py - gy) ** 2, my)
    return lx + ly

def soft_iou_tolerant(pred: torch.Tensor, target: torch.Tensor,
                      threshold: float, tau: float = 2.0,
                      radius: int = 1, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Soft IoU with tolerance:
      1) make soft masks via sigmoid((x - thr)/tau)
      2) allow ±radius pixels by dilating the *other* mask (max-pool)
      3) symmetric (pred vs dilated target, and target vs dilated pred) and average.
    Returns IoU in [0,1]. Use (1 - IoU) as a loss.
    """
    p = _to_B1HW(pred); g = _to_B1HW(target)
    m = _to_B1HW(mask) if mask is not None else None

    p_soft = torch.sigmoid((p - threshold) / max(tau, 1e-6))
    g_soft = torch.sigmoid((g - threshold) / max(tau, 1e-6))

    pad = radius
    pool = (2 * radius + 1)
    g_dil = F.max_pool2d(g_soft, kernel_size=pool, stride=1, padding=pad)
    p_dil = F.max_pool2d(p_soft, kernel_size=pool, stride=1, padding=pad)

    # fuzzy set AND/OR approximations
    inter1 = torch.minimum(p_soft, g_dil)
    union1 = torch.maximum(p_soft, g_dil)
    inter2 = torch.minimum(g_soft, p_dil)
    union2 = torch.maximum(g_soft, p_dil)

    iou1 = _masked_mean(inter1, m) / (_masked_mean(union1, m) + EPS)
    iou2 = _masked_mean(inter2, m) / (_masked_mean(union2, m) + EPS)
    return 0.5 * (iou1 + iou2)

def pattern_aware_loss(pred: torch.Tensor, target: torch.Tensor,
                       mask: torch.Tensor | None = None,
                       ms_scales: Tuple[int, ...] = (1, 2, 4),
                       shift_radius: int = 1, shift_temp: float = 0.5,
                       grad_w: float = 0.0,
                       iou_threshold: float | None = None,
                       iou_tau: float = 2.0, iou_radius: int = 1,
                       weights=(1.0, 0.5, 0.5, 0.0)) -> torch.Tensor:
    """
    Composite loss:
      L = w_mse*MSE + w_ms*MS-MSE + w_shift*SoftShiftMSE + w_grad*GradLoss + w_iou*(1-SoftIoU)
    Set w_iou>0 only if iou_threshold is provided.
    """
    w_mse, w_ms, w_shift, w_iou = weights
    p = _to_B1HW(pred); g = _to_B1HW(target)
    m = _to_B1HW(mask) if mask is not None else None

    base_mse = _masked_mean((p - g) ** 2, m)
    ms_loss  = multiscale_mse(p, g, m, scales=ms_scales)
    shift    = soft_shift_mse(p, g, m, radius=shift_radius, temperature=shift_temp)
    total = w_mse*base_mse + w_ms*ms_loss + w_shift*shift

    if grad_w > 0.0:
        total = total + grad_w * gradient_loss(p, g, m)

    if w_iou > 0.0 and iou_threshold is not None:
        iou = soft_iou_tolerant(p, g, threshold=iou_threshold, tau=iou_tau, radius=iou_radius, mask=m)
        total = total + w_iou * (1.0 - iou)

    return total
