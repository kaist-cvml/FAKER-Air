# aurora/rl/reward.py
from __future__ import annotations
from typing import Dict, List, Literal
import torch
import torch.nn.functional as F
from aurora import Batch
import numpy as np
from aurora.utils import compute_metrics

POLLUTANTS = ("pm2p5", "pm10")
THRESHOLDS = {"pm2p5": 35.0, "pm10": 80.0}
CLASS_BOUNDS = {
    "pm10":  [0.0, 30.0, 80.0, 150.0, 1_000.0],   # µg m^-3
    "pm2p5": [0.0, 15.0, 35.0,  75.0,   800.0],   # µg m^-3
    # "o3":  [0.0, 0.03, 0.090, 0.150, 0.300],    # ppb (optional)
}
CLASS_SCALES = {"pm10": 1.0, "pm2p5": 1.0}

def _mask_from_gt(gt_frame: torch.Tensor) -> torch.Tensor:
    """Return a [B,1,H,W] boolean mask built from ground-truth non-zero pixels."""
    return (gt_frame > 0).unsqueeze(1)

def _randn_like_with_gen(mu: torch.Tensor, rng: torch.Generator | None) -> torch.Tensor:
    """
    Draw N(0,1) noise with an optional Generator.
    - Uses .normal_ to support the `generator` argument on both CPU/CUDA.
    - Allocates the output on the same device/dtype as `mu`.
    """
    out = torch.empty_like(mu)
    if rng is None:
        return out.normal_(mean=0.0, std=1.0)               # no generator
    return out.normal_(mean=0.0, std=1.0, generator=rng)    # with generator

def _normalize_step_weights(T: int, step_weights: torch.Tensor | None, device) -> torch.Tensor:
    if step_weights is None:
        return torch.ones(T, device=device, dtype=torch.float32)
    sw = step_weights.to(device=device, dtype=torch.float32)
    if sw.numel() != T:
        sw = sw[:T] if sw.numel() > T else torch.cat([sw, torch.ones(T - sw.numel(), device=device)])
    return sw
    
def mse_over_rollout(
    preds: List[Batch],
    tgt: Batch,
    weights: Dict[str, float] = {"pm2p5": 1.0, "pm10": 1.0},
    step_weights: torch.Tensor | None = None,   # <-- FIX: add
) -> torch.Tensor:
    dev = preds[0].surf_vars[next(iter(preds[0].surf_vars))].device
    num = torch.tensor(0.0, device=dev)
    den = torch.tensor(0.0, device=dev)

    T = min(len(preds), tgt.surf_vars[next(iter(tgt.surf_vars))].shape[1])

    # <-- FIX: normalize and crop step weights
    if step_weights is None:
        sw = torch.ones(T, device=dev)
    else:
        sw = step_weights[:T].to(device=dev, dtype=torch.float32)

    for t in range(T):
        wt = sw[t].clamp_min(0.0)
        gt_t = tgt.slice_time(t)
        pr_t = preds[t]
        for var, w in weights.items():
            if var not in pr_t.surf_vars or var not in gt_t.surf_vars:
                continue
            g = gt_t.surf_vars[var][:, 0]     # [B,H,W]
            p = pr_t .surf_vars[var][:, 0]
            m = (g > 0).unsqueeze(1)          # [B,1,H,W]
            if m.any():
                diff2 = ((p - g) ** 2).unsqueeze(1) * m
                mse   = diff2.sum() / m.sum().clamp_min(1.0)
                thr   = THRESHOLDS.get(var, 1.0)
                mse_dimless = mse / (thr * thr)
                num = num + wt * float(w) * mse_dimless
                den = den + wt * float(w)
    return num / den.clamp_min(1.0)


# ------------------- MSE → Reward Shaping ------------------- #
RewardShape = Literal["neg", "inv", "one_minus", "exp", "log", "rel_impr"]

def _shape_dimless_mse_to_reward(
    mse_dimless: torch.Tensor,
    shape: RewardShape = "exp",
    ref_mse: torch.Tensor | None = None,
    tau: float = 1.0,
) -> torch.Tensor:
    """
    mse_dimless: 무차원 MSE(스칼라 텐서)
    shape:
      - 'neg'       : -mse
      - 'inv'       : 1 / (1 + mse)
      - 'one_minus' : 1 - clamp(mse, 0, 1)
      - 'exp'       : exp(-mse / tau)
      - 'log'       : -log(mse + eps)
      - 'rel_impr'  : clamp( (ref - mse) / (ref + eps), -1, 1 )
    """
    eps = 1e-6
    m = mse_dimless

    if shape == "neg":
        return -m
    elif shape == "inv":
        return 1.0 / (1.0 + m)
    elif shape == "one_minus":
        return 1.0 - torch.clamp(m, 0.0, 1.0)
    elif shape == "exp":
        tau = max(1e-6, float(tau))
        return torch.exp(-m / tau)
    elif shape == "log":
        return -torch.log(m + eps)
    elif shape == "rel_impr":
        if ref_mse is None:
            # 기준이 없으면 보수적으로 'inv'로 대체
            return 1.0 / (1.0 + m)
        r = (ref_mse - m) / (ref_mse + eps)
        return torch.clamp(r, -1.0, 1.0)
    else:
        raise ValueError(f"Unknown reward shape: {shape}")

# ------------------- F1 Reward ------------------- #
def f1_reward_over_rollout(
    preds: List[Batch],
    tgt: Batch,
    weights: Dict[str, float] = {"pm2p5": 1.0, "pm10": 1.0},
    step_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    클래스(threshold) 기반 이진 분류 F1을 rollout/변수 가중 평균으로 계산.
    """
    # 안전한 device/T 유도
    first_var = next(iter(preds[0].surf_vars))
    device = preds[0].surf_vars[first_var].device

    num = torch.tensor(0.0, device=device)
    den = torch.tensor(0.0, device=device)

    T = min(len(preds), tgt.surf_vars[first_var].shape[1])
    sw = _normalize_step_weights(T, step_weights, device)

    eps = 1e-6
    for t in range(T):
        wt = sw[t].clamp_min(0.0)
        gt_t = tgt.slice_time(t)
        pr_t = preds[t]

        for var, w in weights.items():
            if var not in pr_t.surf_vars or var not in gt_t.surf_vars:
                continue

            g = gt_t.surf_vars[var][:, 0]  # [B,H,W]
            p = pr_t.surf_vars[var][:, 0]
            m = (g > 0)
            if not m.any():
                continue

            thr = THRESHOLDS.get(var, 1.0)
            gt_pos   = (g[m] >= thr)
            pred_pos = (p[m] >= thr)

            tp = (gt_pos & pred_pos).sum().float()
            fp = ((~gt_pos) & pred_pos).sum().float()
            fn = (gt_pos & (~pred_pos)).sum().float()

            f1 = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
            num = num + wt * float(w) * f1
            den = den + wt * float(w)

    return num / den.clamp_min(1e-6)

def mse_reward_over_rollout(
    preds: List[Batch],
    tgt: Batch,
    weights: Dict[str, float] = {"pm2p5": 1.0, "pm10": 1.0},
    *,
    shape: RewardShape = "exp",
    ref_mse: torch.Tensor | None = None,
    tau: float = 1.0,
    step_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    1) 무차원 MSE를 rollout 전체에서 평균 (step_weights 반영)
    2) 지정된 방법(shape)으로 보상으로 변환
    """
    mse_dimless = mse_over_rollout(preds, tgt, weights, step_weights=step_weights)
    return _shape_dimless_mse_to_reward(mse_dimless, shape=shape, ref_mse=ref_mse, tau=tau)

def _edges_from_bounds(bounds: list[float], device, dtype) -> torch.Tensor:
    assert len(bounds) == 5
    return torch.tensor(bounds[1:-1], device=device, dtype=dtype)

def _to_class(x: torch.Tensor, bounds: list[float], scale: float) -> torch.Tensor:
    x_scaled = x * float(scale)
    edges = _edges_from_bounds(bounds, x.device, x.dtype)
    return torch.bucketize(x_scaled, edges)


def _cls_reward_step(
    pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor,
    bounds: list[float], scale: float,
    coarse_w: float, exact_w: float, fa_penalty: float,
    exact_mult_per_cls: torch.Tensor | None = None,
    coarse_mult_per_cls: torch.Tensor | None = None,  
    exact_bonus_per_cls: torch.Tensor | None = None,  
) -> torch.Tensor:
    if not mask.any():
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    m = mask.squeeze(1)
    gt_cls   = _to_class(gt,   bounds, scale)
    pred_cls = _to_class(pred, bounds, scale)
    gt_high, pred_high = (gt_cls >= 2), (pred_cls >= 2)
    coarse_ok = (gt_high == pred_high)
    exact_ok  = (gt_cls == pred_cls)
    false_alarm = (pred_high & (~gt_high))

    # 1) coarse / exact 항을 분리
    base_coarse = coarse_w * coarse_ok.float()
    base_exact  = exact_w  * exact_ok.float()

    # 2) 클래스별 multiplier (gt 기준 인덱싱)
    #    gt_cls: [B,H,W], per_cls: [4] → 인덱싱으로 픽셀별 계수 생성
    idx = gt_cls.clamp_min(0)  # 안전하게
    if coarse_mult_per_cls is not None:
        cm = coarse_mult_per_cls.to(device=pred.device, dtype=pred.dtype)
        cm = cm[idx.clamp_max(cm.numel()-1)]
        base_coarse = base_coarse * cm
    if exact_mult_per_cls is not None:
        em = exact_mult_per_cls.to(device=pred.device, dtype=pred.dtype)
        em = em[idx.clamp_max(em.numel()-1)]
        base_exact = base_exact * em

    # 3) (옵션) exact 정답 보너스 가산
    if exact_bonus_per_cls is not None:
        eb = exact_bonus_per_cls.to(device=pred.device, dtype=pred.dtype)
        eb = eb[idx.clamp_max(eb.numel()-1)]
        base_exact = base_exact + eb * exact_ok.float()

    r_pix = (base_coarse + base_exact) - fa_penalty * false_alarm.float()
    return (r_pix[m]).mean()

def cls_reward_over_rollout(
    preds: List[Batch], tgt: Batch, *,
    weights: Dict[str, float] = {"pm2p5": 1.0, "pm10": 1.0},
    bounds_overrides: Dict[str, list[float]] | None = None,
    scale_overrides: Dict[str, float] | None = None,
    coarse_w: float = 0.5, exact_w: float  = 0.5, fa_penalty: float = 0.0,
    step_weights: torch.Tensor | None = None,
    exact_mult_overrides: Dict[str, list[float]] | None = None,
    coarse_mult_overrides: Dict[str, list[float]] | None = None,
    exact_bonus_overrides: Dict[str, list[float]] | None = None,
) -> torch.Tensor:
    device = preds[0].surf_vars["pm2p5"].device
    num = torch.tensor(0.0, device=device)
    den = torch.tensor(0.0, device=device)

    T = min(len(preds), tgt.surf_vars["pm2p5"].shape[1])
    sw = _normalize_step_weights(T, step_weights, device)

    for t in range(T):
        wt = sw[t].clamp_min(0.0)
        gt_t = tgt.slice_time(t)
        pr_t = preds[t]
        for var, w in weights.items():
            if var not in pr_t.surf_vars or var not in gt_t.surf_vars:
                continue
            g = gt_t.surf_vars[var][:, 0]
            p = pr_t.surf_vars[var][:, 0]
            m = (g > 0).unsqueeze(1)
            if not m.any():
                continue

            bounds = (bounds_overrides.get(var) if bounds_overrides and var in bounds_overrides
                      else CLASS_BOUNDS.get(var))
            assert bounds is not None, f"No class bounds for '{var}'."
            scale = (scale_overrides.get(var) if scale_overrides and var in scale_overrides
                     else CLASS_SCALES.get(var, 1.0))

            # per-class 벡터 준비 (없으면 None 유지 → 기본 동작)
            exact_mult_t  = None
            coarse_mult_t = None
            exact_bonus_t = None
            if exact_mult_overrides and var in exact_mult_overrides:
                assert len(exact_mult_overrides[var]) == 4, "expect 4 values for classes 0..3"
                exact_mult_t = torch.tensor(exact_mult_overrides[var], device=device, dtype=p.dtype)
            if coarse_mult_overrides and var in coarse_mult_overrides:
                assert len(coarse_mult_overrides[var]) == 4, "expect 4 values for classes 0..3"
                coarse_mult_t = torch.tensor(coarse_mult_overrides[var], device=device, dtype=p.dtype)
            if exact_bonus_overrides and var in exact_bonus_overrides:
                assert len(exact_bonus_overrides[var]) == 4, "expect 4 values for classes 0..3"
                exact_bonus_t = torch.tensor(exact_bonus_overrides[var], device=device, dtype=p.dtype)

            r_var = _cls_reward_step(
                p, g, m, bounds, scale, coarse_w, exact_w, fa_penalty,
                exact_mult_per_cls=exact_mult_t,
                coarse_mult_per_cls=coarse_mult_t,
                exact_bonus_per_cls=exact_bonus_t,
            )
            num = num + wt * float(w) * r_var
            den = den + wt * float(w)
    return num / den.clamp_min(1e-6)

def hybrid_reward_over_rollout(
    preds: List[Batch], tgt: Batch,
    mse_w: float = 1.0, cls_w: float = 0.15, recall_w: float = 0.1, *,
    mse_shape: RewardShape = "rel_impr",
    ref_mse: torch.Tensor | None = None,
    tau: float = 0.5,
    weights: Dict[str, float] = {"pm2p5": 1.0, "pm10": 0.5},
    bounds_overrides=None, scale_overrides=None,
    coarse_w: float = 0.45, exact_w: float = 0.55, fa_penalty: float = 0.15,
    recall_var: str = "pm2p5", recall_thr: float = 35.0, recall_avg: str = "macro",
    step_weights: torch.Tensor | None = None,
    # --- NEW: 분류 보상에 전달 ---
    exact_mult_overrides: Dict[str, list[float]] | None = None,
    coarse_mult_overrides: Dict[str, list[float]] | None = None,
    exact_bonus_overrides: Dict[str, list[float]] | None = None,
) -> torch.Tensor:
    rmse = mse_reward_over_rollout(
        preds, tgt, weights, shape=mse_shape, ref_mse=ref_mse, tau=tau, step_weights=step_weights
    )
    rcls = cls_reward_over_rollout(
        preds, tgt, weights=weights,
        bounds_overrides=bounds_overrides, scale_overrides=scale_overrides,
        coarse_w=coarse_w, exact_w=exact_w, fa_penalty=fa_penalty,
        step_weights=step_weights,
        exact_mult_overrides=exact_mult_overrides,
        coarse_mult_overrides=coarse_mult_overrides,
        exact_bonus_overrides=exact_bonus_overrides,
    )
    rrec = recall_reward_over_rollout(
        preds, tgt, var=recall_var, thr=recall_thr, w=1.0, average=recall_avg, step_weights=step_weights
    )
    return float(mse_w) * rmse + float(cls_w) * rcls + float(recall_w) * rrec

# ------------------- Recall / FAR (제약·게이팅) ------------------- #
def recall_reward_over_rollout(
    preds: List[Batch], tgt: Batch,
    var: str = "pm2p5", thr: float = 35.0, w: float = 1.0,
    average: str = "macro",
    step_weights: torch.Tensor | None = None,   # <<< 추가
) -> torch.Tensor:
    dev = preds[0].surf_vars[var].device
    num = torch.tensor(0.0, device=dev)
    den = torch.tensor(1e-6, device=dev)
    T = min(len(preds), tgt.surf_vars[var].shape[1])
    sw = _normalize_step_weights(T, step_weights, dev)

    for t in range(T):
        wt = sw[t].clamp_min(0.0)
        g = tgt.slice_time(t).surf_vars[var][:, 0]
        p = preds[t].surf_vars[var][:, 0]
        m = (g > 0)
        if not m.any():
            continue
        gt_pos   = (g[m] >= thr)
        pred_pos = (p[m] >= thr)
        tp = (gt_pos & pred_pos).sum().float()
        P  = gt_pos.sum().float().clamp_min(1.0)
        if average == "micro":
            num = num + wt * tp
            den = den + wt * P
        else:
            num = num + wt * (tp / P)
            den = den + wt
    return float(w) * (num / den)

def far_metric_over_rollout(
    preds: List[Batch], tgt: Batch,
    var: str = "pm2p5", thr: float = 35.0, w: float = 1.0,
    step_weights: torch.Tensor | None = None,   # <<< 추가
) -> torch.Tensor:
    device = preds[0].surf_vars[var].device
    num = torch.tensor(0.0, device=device)
    den = torch.tensor(1e-6, device=device)
    T = min(len(preds), tgt.surf_vars[var].shape[1])
    sw = _normalize_step_weights(T, step_weights, device)

    for t in range(T):
        wt = sw[t].clamp_min(0.0)
        g = tgt.slice_time(t).surf_vars[var][:, 0]
        p = preds[t].surf_vars[var][:, 0]
        m = (g > 0)
        if m.any():
            gt_pos   = (g[m] >= thr)
            pred_pos = (p[m] >= thr)
            neg      = (~gt_pos)
            if neg.any():
                fp    = (pred_pos & neg).float().sum()
                n_neg = neg.float().sum().clamp_min(1.0)
                num   = num + wt * (fp / n_neg)
                den   = den + wt
    return w * (num / den)