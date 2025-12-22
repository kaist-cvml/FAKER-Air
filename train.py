#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --------------------------------------------------------------------------- #
# 0.  Libraries & configuration
# --------------------------------------------------------------------------- #
from __future__ import annotations
import argparse, os, sys, json, logging, random
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt

from aurora import Aurora, AuroraAirPollution, Batch
from aurora.utils import compute_metrics, pattern_aware_loss, build_step_weights, parse_month_weights, DynamicLossScheduler
from aurora.dataloader import (
    WeatherDataset, aurora_collate_fn,
    collate_batches, CONC_VARS, M2D_VARS, M3D_VARS, _meta_update
)
from aurora.normalisation import unnormalise_surf_var, normalise_surf_var, SURF_STATS
# ------------------------------------------------------------------
# Bind the station‑aware roll‑out so that both models use it
# ------------------------------------------------------------------
from aurora.rollout import rollout as station_rollout
Aurora.rollout = station_rollout
AuroraAirPollution.rollout = station_rollout

from contextlib import contextmanager
import contextlib

sys.path.append(os.getcwd())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# --------------------------------------------------------------------------- #
# 1.  Helper – make H, W multiples of 3
# --------------------------------------------------------------------------- #
def pad_or_crop(arr: np.ndarray, mul: int = 3) -> np.ndarray:
    """
    Reflect‑pad or centre‑crop *arr* (…, H, W) so that H and W are divisible
    by *mul* (3 for 3×3 patch embedding).
    """
    *rest, H, W = arr.shape
    pad_h = (mul - H % mul) % mul
    pad_w = (mul - W % mul) % mul
    top, bot = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    if pad_h or pad_w:
        arr = np.pad(arr, (*[(0, 0)] * len(rest), (top, bot), (left, right)),
                     mode="reflect")
    # crop (rarely needed if input dims are multiples of mul already)
    *_, H2, W2 = arr.shape
    crop_top = max(0, (H2 % mul) // 2)
    crop_left = max(0, (W2 % mul) // 2)
    return arr[..., crop_top:H2 - crop_top, crop_left:W2 - crop_left]

# ------------- collate function (skips None) ------------------------------ #
def _safe_collate(batch: List[Any]):
    batch = [b for b in batch if b is not None and b[0] is not None]
    if not batch:
        return None
    return collate_batches(batch)

def make_loader(ds: Dataset, bs: int, dist_on: bool, shuffle: bool) -> DataLoader:
    sampler = DistributedSampler(ds, shuffle=shuffle) if dist_on else None
    return DataLoader(
        ds, batch_size=bs, sampler=sampler,
        shuffle=shuffle and sampler is None,
        num_workers=4, pin_memory=True, drop_last=False,
        collate_fn=_safe_collate,
    )

def first_non_none_batch(loader):
    for b in loader:
        if b is not None:
            return b
    raise RuntimeError("All batches are None. Check data availability/paths.")

def get_dataloader(
    batch, data_dir, npz_path,
    train_start_date, train_end_date,
    test_start_date, test_end_date,
    cmaq_root, flow_root, sources, rollout,
    use_masking=False, mask_ratio=0.5,
    use_cmaq_conc=True, use_cmaq_m2d=True, use_cmaq_m3d=True,
    use_wind_prompt=False, use_cutmix=False,
    use_cmaq_pm_only=False, use_hybrid_target=False, use_flow=False,
    # --- NEW: CAMS-related ---
    cams_root: str | None = None,
    cams_pm_only: bool = True,
    hybrid_source: str = "cmaq",
):
    train_dataset = WeatherDataset(
        train_start_date, train_end_date, data_dir, npz_path,
        cmaq_root=cmaq_root, flow_root=flow_root, sources=sources, horizon=rollout,
        use_masking=use_masking, mask_ratio=mask_ratio,
        use_wind_prompt=use_wind_prompt, use_cutmix=use_cutmix,
        use_cmaq_conc=use_cmaq_conc, use_cmaq_m2d=use_cmaq_m2d, use_cmaq_m3d=use_cmaq_m3d,
        use_cmaq_pm_only=use_cmaq_pm_only,
        use_hybrid_target=use_hybrid_target,
        use_flow=use_flow,
        # --- pass CAMS knobs to dataset ---
        cams_dir=(cams_root if cams_root else None),
        # use_cams_pm_only=cams_pm_only,
        hybrid_target_source=hybrid_source,
    )
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch, sampler=train_sampler,
                                  collate_fn=aurora_collate_fn)

    eval_dataset = WeatherDataset(
        test_start_date, test_end_date, data_dir, npz_path,
        cmaq_root=cmaq_root, flow_root=flow_root, sources=sources, horizon=rollout,
        use_masking=False,                # never augment val/test
        use_wind_prompt=use_wind_prompt,  # keep feature flags aligned
        use_cmaq_conc=use_cmaq_conc, use_cmaq_m2d=use_cmaq_m2d, use_cmaq_m3d=use_cmaq_m3d,
        use_cutmix=use_cutmix,
        use_cmaq_pm_only=use_cmaq_pm_only,
        # NOTE: keep validation target = pure OBS for fair metrics
        # use_hybrid_target=use_hybrid_target,
        use_flow=use_flow,
        cams_dir=(cams_root if cams_root else None),
        # use_cams_pm_only=cams_pm_only,
        # hybrid_target_source=hybrid_source,
    )
    eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, sampler=eval_sampler,
                                 collate_fn=aurora_collate_fn)
    return train_dataloader, eval_dataloader



# --------------------------------------------------------------------------- #
# 3.  Loss – masked MSE on pollutants
# --------------------------------------------------------------------------- #
POLLUTANTS = ("pm2p5", "pm10", "tcso2", "tcno2", "o3", "tcco")

def masked_mse(pred: torch.Tensor,
               tgt : torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """
    pred / tgt / mask have identical shape  [B, T, H, W]
    The mask contains 1 for valid grid‑points, 0 otherwise.
    """
    diff2 = (pred - tgt).pow(2)
    diff2 = diff2 * mask                     # zero‑out invalid cells
    # avoid division by zero
    denom = torch.clamp(mask.sum(), min=1.0)
    return diff2.sum() / denom

MAIN_POLLUTANTS = {"pm2p5", "pm10"}             # ← weight 1.0  (baseline)
# secondary surf‑level vars (CMAQ or others)
SECONDARY_SURF   = {                            # weight 0.10
    "tcso2", "tcno2", "o3", "tcco",
}
# every surf‑level variable that is not in MAIN_POLLUTANTS ∪ SECONDARY_SURF
DEFAULT_SURF_W   = 0.5
# CMAQ chemistry / met tokens live in .atmos_vars
# you can override specific variables here if needed
ATMOS_VAR_WEIGHT = 0.5                         # global default
SURF_WEIGHTS  = {v:1.00 for v in MAIN_POLLUTANTS}
SURF_WEIGHTS |= {v:0.1 for v in SECONDARY_SURF}       # union / update


# --------------------------------------------------------------------------- #
# 2) loss_fn
# --------------------------------------------------------------------------- #
def loss_fn(
    pb: "Batch",
    tb: "Batch",
    w_pm: float = 2.0,
    w_obssec: float = 0.1,
    w_cmaq_total: float = 1.0,
    w_era5_total: float = 0.5,
    # --- NEW: CAMS auxiliary total weight (0 disables) ---
    w_cams_total: float = 0.0,
    # --- PM2.5 AQI level weights ---
    w_pm25_good: float = 0.1,
    w_pm25_moderate: float = 0.1,
    w_pm25_bad: float = 0.2,
    w_pm25_very_bad: float = 0.1,
    # ------------------------------------
    cmaq2d_frac: float = 0.5,
    use_pattern_aware_loss: bool = False,
    return_details: bool = False,
    pattern_scale: float = 1.0,
    use_flow: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """
    Computes the weighted multi-group loss. Optionally returns a dict of
    per-component losses both *raw* (pre-weight) and *weighted* (post-group-weight).

    The PM group keeps the original semantics: pm2p5 and pm10 are aggregated
    by a mean and a single group weight (w_pm) is applied to that mean. For
    logging, we also compute per-variable contributions (pm2p5/pm10) such that
    pm25_w + pm10_w == w_pm * mean(pm25, pm10) even when only one is present.

    Returns
    -------
    total_loss : torch.Tensor
        Weighted average: (sum(group_weight * group_mean)) / (sum(group_weight))
    details (optional) : dict
        {
          "raw":      {"pm25": ..., "pm10": ..., "obssec": ..., "cmaq2d": ..., "cmaq3d": ...},
          "weighted": {"pm25": ..., "pm10": ..., "obssec": ..., "cmaq2d": ..., "cmaq3d": ...},
          "den":      torch.scalar (sum of active group-weights used in the denominator)
        }
    """

    def _align_time(a, b):
        if a.shape[1] != b.shape[1]:
            T = min(a.shape[1], b.shape[1])
            return a[:, :T], b[:, :T]
        return a, b

    device = next(iter(pb.surf_vars.values())).device
    def zero():
        return torch.tensor(0.0, device=device)

    # Teacher-forcing mask built from PMs
    mask_tf = ((tb.surf_vars.get("pm2p5", torch.tensor(0., device=device)) > 0) |
               (tb.surf_vars.get("pm10" , torch.tensor(0., device=device)) > 0)).float()

    # Collect per-variable losses
    pm25_losses, pm10_losses = [], []
    pm25_origin_losses, pm10_origin_losses = [], []
    cmaq_pm_losses, obssec_losses, cmaq2d_losses, cmaq3d_losses = [], [], [], []
    era5_surf_losses, era5_atmos_losses = [], []
    cams_pm_losses = []
    # --- ADD dict for PM2.5 level losses ---
    pm25_level_losses = {
        'good': [], 'moderate': [], 'bad': [], 'very_bad': []
    }
    pm25_origin_level_losses = {
        'good': [], 'moderate': [], 'bad': [], 'very_bad': []
    }
    # -----------------------------------------

    # Surface variables (obs + possible *_cmaq 2D tokens)
    for v in set(pb.surf_vars).intersection(tb.surf_vars):
        p, g = _align_time(pb.surf_vars[v], tb.surf_vars[v])
        if v == 'wind_prompt':
            continue
        if v == "pm2p5":
            base_mask = mask_tf if mask_tf.shape[1] == p.shape[1] else mask_tf[:, :p.shape[1]]
            mse_term = masked_mse(p, g, base_mask)
            if use_pattern_aware_loss:
                patt_term = pattern_aware_loss(
                    p, g, mask=base_mask,
                    ms_scales=(1,2,4),
                    shift_radius=1, shift_temp=0.5,
                    grad_w=0.1,                     
                    iou_threshold=35.0,              
                    iou_tau=2.0, iou_radius=1,
                    weights=(1.0, 0.5, 0.5, 0.5)     # (w_mse, w_ms, w_shift, w_iou)
                )
                pm25_losses.append(mse_term + patt_term * patt_term)
            else:
                pm25_losses.append(mse_term)

            # --- START of PM2.5 level-wise loss calculation ---
            # Using VIS_VARS['pm2p5']['bounds'] = [0, 15, 35, 75, 800]
            bounds = [0, 15, 35, 75] # Define AQI bounds for PM2.5
            
            # 1. Good (0 <= gt < 15)
            lo, scale = SURF_STATS['pm2p5']
            bounds_phys = torch.tensor([0., 15., 35., 75.], device=device)
            bounds_norm = (bounds_phys - lo) / scale

            mask_good     = base_mask * (g >= bounds_norm[0]) * (g < bounds_norm[1])
            mask_moderate = base_mask * (g >= bounds_norm[1]) * (g < bounds_norm[2])
            mask_bad      = base_mask * (g >= bounds_norm[2]) * (g < bounds_norm[3])
            mask_very_bad     = base_mask * (g >= bounds_norm[3])
            if mask_good.any():
                pm25_level_losses['good'].append(masked_mse(p, g, mask_good))
            
            # 2. Moderate (15 <= gt < 35)
            if mask_moderate.any():
                pm25_level_losses['moderate'].append(masked_mse(p, g, mask_moderate))

            # 3. Bad (35 <= gt < 75)
            if mask_bad.any():
                pm25_level_losses['bad'].append(masked_mse(p, g, mask_bad))
            
            # 4. Very Bad (gt >= 75)
            if mask_very_bad.any():
                pm25_level_losses['very_bad'].append(masked_mse(p, g, mask_very_bad))
            # --- END of PM2.5 level-wise loss calculation ---

        elif v == "pm10":
            mm = mask_tf if mask_tf.shape[1] == p.shape[1] else mask_tf[:, :p.shape[1]]
            mse_term = masked_mse(p, g, mm)
            if use_pattern_aware_loss:
                patt_term = pattern_aware_loss(
                    p, g, mask=mm,
                    ms_scales=(1,2,4),
                    shift_radius=1, shift_temp=0.5,
                    grad_w=0.1,
                    iou_threshold=80.0, iou_tau=3.0, iou_radius=1,
                    weights=(1.0, 0.5, 0.5, 0.5)
                )
                pm10_losses.append(mse_term + patt_term * patt_term)
            else:
                pm10_losses.append(mse_term)

        elif v == "pm2p5_origin":
            base_mask = mask_tf if mask_tf.shape[1] == p.shape[1] else mask_tf[:, :p.shape[1]]
            mse_term = masked_mse(p, g, base_mask)
            if use_pattern_aware_loss:
                patt_term = pattern_aware_loss(
                    p, g, mask=base_mask,
                    ms_scales=(1,2,4),
                    shift_radius=1, shift_temp=0.5,
                    grad_w=0.1,                     
                    iou_threshold=35.0,              
                    iou_tau=2.0, iou_radius=1,
                    weights=(1.0, 0.5, 0.5, 0.5)     # (w_mse, w_ms, w_shift, w_iou)
                )
                pm25_origin_losses.append(mse_term + patt_term * patt_term)
            else:
                pm25_origin_losses.append(mse_term)

            # --- START of PM2.5 level-wise loss calculation ---
            # Using VIS_VARS['pm2p5']['bounds'] = [0, 15, 35, 75, 800]
            bounds = [0, 15, 35, 75] # Define AQI bounds for PM2.5
            
            # 1. Good (0 <= gt < 15)
            lo, scale = SURF_STATS['pm2p5']
            bounds_phys = torch.tensor([0., 15., 35., 75.], device=device)
            bounds_norm = (bounds_phys - lo) / scale

            mask_good     = base_mask * (g >= bounds_norm[0]) * (g < bounds_norm[1])
            mask_moderate = base_mask * (g >= bounds_norm[1]) * (g < bounds_norm[2])
            mask_bad      = base_mask * (g >= bounds_norm[2]) * (g < bounds_norm[3])
            mask_very_bad     = base_mask * (g >= bounds_norm[3])
            if mask_good.any():
                pm25_origin_level_losses['good'].append(masked_mse(p, g, mask_good))
            
            # 2. Moderate (15 <= gt < 35)
            if mask_moderate.any():
                pm25_origin_level_losses['moderate'].append(masked_mse(p, g, mask_moderate))

            # 3. Bad (35 <= gt < 75)
            if mask_bad.any():
                pm25_origin_level_losses['bad'].append(masked_mse(p, g, mask_bad))
            
            # 4. Very Bad (gt >= 75)
            if mask_very_bad.any():
                pm25_origin_level_losses['very_bad'].append(masked_mse(p, g, mask_very_bad))
            # --- END of PM2.5 level-wise loss calculation ---

        elif v == "pm10_origin":
            mm = mask_tf if mask_tf.shape[1] == p.shape[1] else mask_tf[:, :p.shape[1]]
            mse_term = masked_mse(p, g, mm)
            if use_pattern_aware_loss:
                patt_term = pattern_aware_loss(
                    p, g, mask=mm,
                    ms_scales=(1,2,4),
                    shift_radius=1, shift_temp=0.5,
                    grad_w=0.1,
                    iou_threshold=80.0, iou_tau=3.0, iou_radius=1,
                    weights=(1.0, 0.5, 0.5, 0.5)
                )
                pm10_origin_losses.append(mse_term + patt_term * patt_term)
            else:
                pm10_origin_losses.append(mse_term)
                
        elif v in {"tcso2", "tcno2", "o3", "tcco"}:
            obssec_losses.append(F.mse_loss(p, g))
        elif v.endswith("_cmaq"):
            # If the variable is pm2.5_cmaq or pm10_cmaq, add it to the special list
            if v in {"pm2p5_cmaq", "pm10_cmaq"}:
                cmaq_pm_losses.append(F.mse_loss(p, g))
            # Otherwise, add it to the general CMAQ 2D list
            else:
                cmaq2d_losses.append(F.mse_loss(p, g))
        elif v.endswith("_cams"):
            if v in {"pm2p5_cams", "pm10_cams"}:
                cams_pm_losses.append(F.mse_loss(p, g))
        else:
            era5_surf_losses.append(F.mse_loss(p, g))

    # Atmospheric variables
    for v in set(pb.atmos_vars).intersection(tb.atmos_vars):
        p, g = _align_time(pb.atmos_vars[v], tb.atmos_vars[v])
        # --- 💡 3. MODIFY CMAQ VARIABLE HANDLING (continued for 3D) ---
        if v.endswith("_cmaq"):
            if v in {"pm2p5_cmaq", "pm10_cmaq"}:
                cmaq_pm_losses.append(F.mse_loss(p, g))
            else:
                cmaq3d_losses.append(F.mse_loss(p, g))
        # -----------------------------------------------------------

    # Group means (raw)
    # --- CALCULATE WEIGHTED AVERAGE for PM2.5 ---
    L_pm25_good = torch.stack(pm25_level_losses['good']).mean() if pm25_level_losses['good'] else zero()
    L_pm25_moderate = torch.stack(pm25_level_losses['moderate']).mean() if pm25_level_losses['moderate'] else zero()
    L_pm25_bad = torch.stack(pm25_level_losses['bad']).mean() if pm25_level_losses['bad'] else zero()
    L_pm25_very_bad = torch.stack(pm25_level_losses['very_bad']).mean() if pm25_level_losses['very_bad'] else zero()
    L_cmaq_pm = torch.stack(cmaq_pm_losses).mean() if cmaq_pm_losses else zero()
    L_cams_pm = torch.stack(cams_pm_losses).mean() if cams_pm_losses else zero()

    if use_flow:
        L_pm25_origin_good = torch.stack(pm25_origin_level_losses['good']).mean() if pm25_origin_level_losses['good'] else zero()
        L_pm25_origin_moderate = torch.stack(pm25_origin_level_losses['moderate']).mean() if pm25_origin_level_losses['moderate'] else zero()
        L_pm25_origin_bad = torch.stack(pm25_origin_level_losses['bad']).mean() if pm25_origin_level_losses['bad'] else zero()
        L_pm25_origin_very_bad = torch.stack(pm25_origin_level_losses['very_bad']).mean() if pm25_origin_level_losses['very_bad'] else zero()

    # Calculate the weighted sum of PM2.5 losses
    pm25_num = (w_pm25_good * L_pm25_good + 
                w_pm25_moderate * L_pm25_moderate + 
                w_pm25_bad * L_pm25_bad + 
                w_pm25_very_bad * L_pm25_very_bad)
    pm25_den = (w_pm25_good if pm25_level_losses['good'] else 0) + \
               (w_pm25_moderate if pm25_level_losses['moderate'] else 0) + \
               (w_pm25_bad if pm25_level_losses['bad'] else 0) + \
               (w_pm25_very_bad if pm25_level_losses['very_bad'] else 0)
    
    L_pm25 = pm25_num / torch.clamp(torch.tensor(pm25_den, device=device), min=1e-9)

    if use_flow:
        pm25_origin_num = (w_pm25_good * L_pm25_origin_good + 
                    w_pm25_moderate * L_pm25_origin_moderate + 
                    w_pm25_bad * L_pm25_origin_bad + 
                    w_pm25_very_bad * L_pm25_origin_very_bad)
        pm25_origin_den = (w_pm25_good if pm25_origin_level_losses['good'] else 0) + \
                (w_pm25_moderate if pm25_origin_level_losses['moderate'] else 0) + \
                (w_pm25_bad if pm25_origin_level_losses['bad'] else 0) + \
                (w_pm25_very_bad if pm25_origin_level_losses['very_bad'] else 0)
        
        L_origin_pm25 = pm25_origin_num / torch.clamp(torch.tensor(pm25_origin_den, device=device), min=1e-9)
    # ----------------------------------------------
    L_pm10 = torch.stack(pm10_losses).mean() if pm10_losses else zero()
    n_pm   = int(bool(pm25_losses)) + int(bool(pm10_losses))
    L_pm_group = ((L_pm25 + L_pm10) / n_pm) if n_pm > 0 else zero()

    if use_flow:
        L_origin_pm10 = torch.stack(pm10_origin_losses).mean() if pm10_origin_losses else zero()
        n_origin_pm   = int(bool(pm25_origin_losses)) + int(bool(pm10_origin_losses))
        L_origin_pm_group = ((L_origin_pm25 + L_origin_pm10) / n_origin_pm) if n_origin_pm > 0 else zero()

    # ERA5
    L_era5_surf = torch.stack(era5_surf_losses).mean() if era5_surf_losses else zero()
    L_era5_atmos = torch.stack(era5_atmos_losses).mean() if era5_atmos_losses else zero()


    L_obs   = torch.stack(obssec_losses).mean() if obssec_losses else zero()
    L_c2d   = torch.stack(cmaq2d_losses).mean() if cmaq2d_losses else zero()
    L_c3d   = torch.stack(cmaq3d_losses).mean() if cmaq3d_losses else zero()

    # Group weights
    w_c2d = w_cmaq_total * cmaq2d_frac
    w_c3d = w_cmaq_total * (1.0 - cmaq2d_frac)
    w_era5 = w_era5_total 

    # Numerator / Denominator for the weighted average
    num = torch.tensor(0.0, device=device)
    den = torch.tensor(0.0, device=device)

    if n_pm > 0:
        num = num + w_pm * L_pm_group
        den = den + w_pm
    if obssec_losses:
        num = num + w_obssec * L_obs
        den = den + w_obssec
    if cmaq2d_losses:
        num = num + w_c2d * L_c2d
        den = den + w_c2d
    if cmaq3d_losses:
        num = num + w_c3d * L_c3d
        den = den + w_c3d
    if era5_surf_losses:
        num = num + w_era5 * L_era5_surf
        den = den + w_era5
    if era5_atmos_losses:
        num = num + w_era5 * L_era5_atmos
        den = den + w_era5
    if cmaq_pm_losses:
        num += w_cmaq_total * 10 * L_cmaq_pm
        den += w_cmaq_total * 10
    if cams_pm_losses:
        num += w_cams_total * 10 * L_cams_pm
        den += w_cams_total * 10

    if den.item() == 0.0:
        raise RuntimeError("No overlapping variables between prediction and target batches.")

    if use_flow:
        if n_origin_pm > 0:
            num = num + w_pm * L_origin_pm_group
            den = den + w_pm

        for v in set(pb.flow_vars).intersection(tb.flow_vars):
            if "mask" in v: continue
            if "pm10" in v: mask = tb.flow_vars["maskpm10"]
            else: mask = tb.flow_vars["maskpm2p5"]
            p, g = _align_time(pb.flow_vars[v], tb.flow_vars[v])
            if pb.flow_vars[v].shape != tb.flow_vars[v].shape:
                print(pb.flow_vars[v].shape, tb.flow_vars[v].shape, v)
                raise 0
                
            loss = F.mse_loss(p*mask, g*mask)
            if "pm10x" in v: flow10x_losses = [loss]
            elif "pm10y" in v: flow10y_losses = [loss]
            elif "pm2p5x" in v: flow25x_losses = [loss]
            elif "pm2p5y" in v: flow25y_losses = [loss]

        flow10x_losses_torch = torch.stack(flow10x_losses).mean() if flow10x_losses else zero()
        flow10y_losses_torch = torch.stack(flow10y_losses).mean() if flow10y_losses else zero()
        flow25x_losses_torch = torch.stack(flow25x_losses).mean() if flow25x_losses else zero()
        flow25y_losses_torch = torch.stack(flow25y_losses).mean() if flow25y_losses else zero()
        num = num + flow10x_losses_torch + flow10y_losses_torch + flow25x_losses_torch + flow25y_losses_torch

    total = num / den

    if not return_details:
        return total

    # Build a consistent breakdown for logging
    # PM per-var weighted contributions are split so that:
    #   pm25_w + pm10_w == w_pm * L_pm_group
    w_pm_each = (w_pm / n_pm) if n_pm > 0 else 0.0
    details = {
        "raw": {
            "pm25":  L_pm25,
            "pm10":  L_pm10,
            "obssec": L_obs,
            "cmaq2d": L_c2d,
            "cmaq3d": L_c3d,
            # --- NEW: expose raw CAMS PM loss (zero if none) ---
            "cams_pm": L_cams_pm,
        },
        "weighted": {
            "pm25":   (w_pm_each * L_pm25) if n_pm > 0 else zero(),
            "pm10":   (w_pm_each * L_pm10) if n_pm > 0 else zero(),
            "obssec": (w_obssec * L_obs)  if obssec_losses else zero(),
            "cmaq2d": (w_c2d * L_c2d)     if cmaq2d_losses else zero(),
            "cmaq3d": (w_c3d * L_c3d)     if cmaq3d_losses else zero(),
            # --- NEW: CAMS PM weighted contribution ---
            "cams_pm": (w_cams_total * 10 * L_cams_pm) if cams_pm_losses else zero(),
        },
        "den": den,
    }
    if use_flow:
        details["raw"].update({"pm2p5_origin":L_origin_pm25,
                               "pm10_origin":L_origin_pm10,
                               "flowpm10x": flow10x_losses_torch,
                               "flowpm10y": flow10y_losses_torch,
                               "flowpm2p5x": flow25x_losses_torch,
                               "flowpm2p5y": flow25y_losses_torch,
                               })
        details["weighted"].update({"pm2p5_origin":L_origin_pm25,
                               "pm10_origin":L_origin_pm10,
                               "flowpm10x": flow10x_losses_torch,
                               "flowpm10y": flow10y_losses_torch,
                               "flowpm2p5x": flow25x_losses_torch,
                               "flowpm2p5y": flow25y_losses_torch,
                               })

    return total, details


def gather_metrics(y_true_dict, y_pred_dict, thresh=35.0):
    """
    Combine per-batch numpy arrays and return metrics per pollutant.

    y_true_dict / y_pred_dict: dict[var] -> list[np.ndarray] (any length)
    returns:                     dict[var] -> dict[str,float]
    """
    out = {}
    for var in ("pm2p5", "pm10"):
        if not y_true_dict[var]:           # nothing gathered for this var
            continue
        yt = np.concatenate(y_true_dict[var])
        yp = np.concatenate(y_pred_dict[var])
        if var == "pm10":
            thresh=80.0
        out[var] = compute_metrics(yp, yt, var.upper(), thresh)
    return out

def _table(md: dict[str, dict[str, float]],
           indent: str = "\t\t") -> str:
    """
    Pretty-print per-pollutant metrics as a fixed-width
    Markdown table, prefixed with an indent.

    md = {"pm2p5": {...}, "pm10": {...}}
    """
    # --------------- column meta ---------------- #
    cols          = list(md.keys())                # ["pm2p5", "pm10", …]
    nice_headers  = [c.upper() for c in cols]
    rows_config   = [("F1",      "f1"),
                     ("Acc",     "accuracy"),
                     ("FAR",     "false_alarm_rate"),
                     ("DR",      "detection_rate"),
                     ("Recall",  "recall")]

    # --------------- compute widths ------------- #
    col_w = [len(h) for h in nice_headers]         # start with header width
    for i, c in enumerate(cols):                   # widen if value is longer
        for _, key in rows_config:
            col_w[i] = max(col_w[i], len(f"{md[c][key]:.3f}"))

    # --------------- helpers -------------------- #
    def fmt_cell(txt: str, i: int, align: str = ">") -> str:
        """Pad a cell to col_w[i] (right-align numbers, center headers)."""
        w = col_w[i]
        if align == "center":
            return txt.center(w)
        return f"{txt:{align}{w}}"

    # --------------- build table ---------------- #
    header = (
        f"| {'Metric':<6} | " +
        " | ".join(fmt_cell(h, i, 'center') for i, h in enumerate(nice_headers)) +
        " |"
    )
    sep = (
        "|" + "-" * 8 + "|" +
        "|".join("-" * (col_w[i] + 2) for i in range(len(cols))) +
        "|"
    )
    body_lines = []
    for nice, key in rows_config:
        vals = " | ".join(fmt_cell(f"{md[c][key]:.3f}", i) for i, c in enumerate(cols))
        body_lines.append(f"| {nice:<6} | {vals} |")

    # --------------- indent & return ------------ #
    lines = [header, sep] + body_lines
    return indent + f"\n{indent}".join(lines)

# --------------------------------------------------------------------------- #
# 4.  Validation
# --------------------------------------------------------------------------- #
def validate(
    loader: DataLoader,
    model,
    *,
    use_rollout_cutmix: bool = False,
    obs_path: str | None = None,
    cmaq_root: str | None = None,
    use_flow: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Returns metrics dict exactly like train_one_epoch:
    {"pm2p5": {...}, "pm10": {...}}
    """
    model.eval()
    y_true, y_pred = defaultdict(list), defaultdict(list)

    for_list = ["pm2p5", "pm10"]
    if use_flow:
        for_list.append("pm2p5_origin")
        for_list.append("pm10_origin")

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            x, tgt = (b.to("cuda") for b in batch)
            if use_flow:
                tgt.surf_vars['pm2p5_origin'] = tgt.surf_vars['pm2p5']
                tgt.surf_vars['pm10_origin'] = tgt.surf_vars['pm10']

            steps_h = tgt.surf_vars["pm2p5"].shape[1]
            core = model.module if isinstance(model, DDP) else model
            if steps_h > 1:
                pred_seq = list(core.rollout(
                    x, steps=steps_h,
                    use_rollout_cutmix=use_rollout_cutmix,
                    obs_path=obs_path, cmaq_root=cmaq_root
                ))
                pred     = Batch.concat_time(pred_seq)
            else:
                pred = core(x)

            for var in for_list:
                for t in range(steps_h):
                    p_var = pred.surf_vars[var][:, t]
                    g_var = tgt .surf_vars[var][:, t]
                    mask  = g_var > 0
                    if mask.any():
                        y_true[var].append(g_var[mask].cpu().numpy())
                        y_pred[var].append(p_var[mask].cpu().numpy())

    metrics = {}
    for var in for_list:
        if y_true[var]:
            yt = np.concatenate(y_true[var])
            yp = np.concatenate(y_pred[var])
            threshold = 35 if "pm2p5" in var else 80
            metrics[var] = compute_metrics(yp, yt, var.upper(), threshold)
        else:                                          # no valid pixels
            metrics[var] = {k: 0.0 for k in (
                "f1", "accuracy", "false_alarm_rate",
                "detection_rate", "recall")}
    return metrics

# --------------------------------------------------------------------------- #
# 5.  Training helpers
# --------------------------------------------------------------------------- #
def save_ckpt(ep, model, opt, scheduler, loss, best_f1, path: Path):
    sd = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save({"epoch": ep, "model_state_dict": sd,
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_f1_score": best_f1,
                "loss": loss}, path)
    logging.info(f"[ckpt] saved → {path}")

def train_one_epoch(loader, model, opt, scaler, epoch,
                    accum, writer, amp, rollout_weight,
                    w_pm, w_obssec, w_cmaq, w_era5, cmaq2d_frac,
                    # --- existing ---
                    use_cutmix: bool = False,
                    use_flow: bool = False,
                    obs_path: str | None = None,
                    cmaq_root: str | None = None,
                    # --- NEW: CAMS total weight ---
                    w_cams: float = 0.0,
                    # --- PM2.5 AQI-level weights (기존) ---
                    w_pm25_good: float = 0.0,
                    w_pm25_moderate: float = 0.0,
                    w_pm25_bad: float = 0.0,
                    w_pm25_very_bad: float = 0.0,
                    scheduler=None,
                    use_pattern_aware_loss=False,
                    rollout_weight_mode: str = "fixed",
                    rollout_gamma: float = 0.9,
                    dyn_sched: DynamicLossScheduler | None = None,
                    ):

    model.train()
    total, steps = 0.0, 0
    dev = next(model.parameters()).device
    y_true_dict = defaultdict(list)
    y_pred_dict = defaultdict(list)

    # Running sums of per-component *normalized* batch losses (comparable to Train/BatchLoss)
    comp_keys = ["pm25", "pm10", "obssec", "cmaq2d", "cmaq3d", "cams_pm"]  # <- NEW key
    if use_flow:
        comp_keys.append("pm2p5_origin")
        comp_keys.append("pm10_origin")
        comp_keys.append("flowpm2p5x")
        comp_keys.append("flowpm2p5y")
        comp_keys.append("flowpm10x")
        comp_keys.append("flowpm10y")
    epoch_comp_sum = {k: 0.0 for k in comp_keys}

    pbar = tqdm(loader, desc=f"Epoch {epoch}", ncols=110,
                disable=dist.get_rank() != 0)

    for batch in pbar:
        if batch is None:
            continue
        x, tgt = batch
        x, tgt = x.to("cuda"), tgt.to("cuda")
        if use_flow:
            tgt.surf_vars['pm2p5_origin'] = tgt.surf_vars['pm2p5']
            tgt.surf_vars['pm10_origin'] = tgt.surf_vars['pm10']

        sync_now = ((steps + 1) % accum == 0)
        ctx = (
            model.no_sync()
            if (hasattr(model, "no_sync") and not sync_now)
            else contextlib.nullcontext()
        )

        # Will hold per-component normalized contributions for this batch
        batch_comp_norm = {k: torch.tensor(0.0, device=dev) for k in comp_keys}

        with ctx:
            with torch.amp.autocast(device_type="cuda", enabled=amp):
                # (1) rollout
                steps_h = tgt.surf_vars["pm2p5"].shape[1]
                if steps_h > 1:
                    core     = model.module if isinstance(model, DDP) else model
                    # --- START: Pass the new arguments to the rollout function ---
                    pred_seq = list(core.rollout(
                        x, 
                        steps=steps_h,
                        use_rollout_cutmix=use_cutmix,
                        obs_path=obs_path,
                        cmaq_root=cmaq_root
                    ))
                    # --- END: Pass the new arguments ---
                    pred     = Batch.concat_time(pred_seq)
                else:
                    pred = model(x)

                # (2) loss + component breakdowns
                if steps_h == 1:
                    mult = dyn_sched.group_multipliers(tgt, epoch) if dyn_sched else {
                        "pm":1.0, "obssec":1.0, "cmaq":1.0, "era5":1.0, "pattern_scale":1.0
                    }
                    loss_raw, det = loss_fn(
                        pred, tgt,
                        w_pm=w_pm, w_obssec=w_obssec,
                        w_era5_total=w_era5,
                        w_cmaq_total=w_cmaq, cmaq2d_frac=cmaq2d_frac,
                        w_cams_total = w_cams,
                        return_details=True,
                        w_pm25_good=w_pm25_good,
                        w_pm25_moderate=w_pm25_moderate,
                        w_pm25_bad=w_pm25_bad,
                        w_pm25_very_bad=w_pm25_very_bad,
                        use_pattern_aware_loss=use_pattern_aware_loss,
                        pattern_scale = mult["pattern_scale"],
                        use_flow=use_flow,
                    )
                    # Normalize each component by the same denominator used in total loss
                    den = torch.clamp(det["den"], min=1e-12)
                    for k in comp_keys:
                        # component contribution to the final batch loss (no accum division)
                        batch_comp_norm[k] = det["weighted"][k] / den
                    loss = loss_raw / accum

                else:
                    step_w = build_step_weights(
                        steps_h,
                        mode=rollout_weight_mode,
                        base=rollout_weight,
                        gamma=rollout_gamma,
                        device=dev
                    )
                    total_loss_wsum = torch.tensor(0.0, device=dev)
                    total_wsum = float(step_w.sum().item())
                    comp_norm_wsum = {k: torch.tensor(0.0, device=dev) for k in comp_keys}

                    for i in range(steps_h):
                        tgt_i  = tgt.slice_time(i)
                        pred_i = pred.slice_time(i)

                        mult = dyn_sched.group_multipliers(tgt_i, epoch) if dyn_sched else {
                            "pm":1.0, "obssec":1.0, "cmaq":1.0, "era5":1.0, "pattern_scale":1.0
                        }

                        li, det_i = loss_fn(
                            pred_i, tgt_i,
                            w_pm = w_pm * mult["pm"],
                            w_obssec = w_obssec * mult["obssec"],
                            w_era5_total = w_era5 * mult["era5"],
                            w_cmaq_total = w_cmaq * mult["cmaq"],
                            cmaq2d_frac = cmaq2d_frac,
                            w_cams_total = w_cams,
                            w_pm25_good = w_pm25_good,
                            w_pm25_moderate = w_pm25_moderate,
                            w_pm25_bad = w_pm25_bad,
                            w_pm25_very_bad = w_pm25_very_bad,
                            use_pattern_aware_loss = use_pattern_aware_loss,
                            pattern_scale = mult["pattern_scale"],
                            return_details=True,
                            use_flow=use_flow,
                        )

                        wi = step_w[i]
                        total_loss_wsum = total_loss_wsum + wi * li

                        den_i = torch.clamp(det_i["den"], min=1e-12)
                        for k in comp_keys:
                            comp_norm_wsum[k] = comp_norm_wsum[k] + wi * (det_i["weighted"][k] / den_i)

                    loss_raw = total_loss_wsum / max(total_wsum, 1e-12)
                    for k in comp_keys:
                        batch_comp_norm[k] = comp_norm_wsum[k] / max(total_wsum, 1e-12)
                    loss = loss_raw / accum

            # backward / optimizer step
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (steps + 1) % accum == 0:
                if scaler is not None:
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                scheduler.step()
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()

        # Aggregate totals for epoch averages
        total += loss.item() * accum
        steps += 1
        for k in comp_keys:
            epoch_comp_sum[k] += batch_comp_norm[k].item()

        # Quick per-batch logging (every 50 steps)
        if writer and dist.get_rank() == 0 and steps % 50 == 0:
            # Note: Train/BatchLoss logs the raw (pre-accum) value, so we do the same
            global_step = epoch * len(loader) + steps
            writer.add_scalar("Train/BatchLoss", loss.item() * accum, global_step)
            writer.add_scalar("Train/BatchLoss/PM25",     batch_comp_norm["pm25"].item(),   global_step)
            writer.add_scalar("Train/BatchLoss/PM10",     batch_comp_norm["pm10"].item(),   global_step)
            writer.add_scalar("Train/BatchLoss/SurfOther",batch_comp_norm["obssec"].item(), global_step)
            writer.add_scalar("Train/BatchLoss/CMAQ2D",   batch_comp_norm["cmaq2d"].item(), global_step)
            writer.add_scalar("Train/BatchLoss/CMAQ3D",   batch_comp_norm["cmaq3d"].item(), global_step)
            writer.add_scalar("Train/BatchLoss/CAMS_PM",  batch_comp_norm["cams_pm"].item(),global_step)

        # Metrics collection (unchanged)
        for var in POLLUTANTS:
            g = tgt.surf_vars[var][:, 0]
            p = pred.surf_vars[var][:, 0]
            m = g > 0
            if m.any():
                y_true_dict[var].append(g[m].cpu().numpy())
                y_pred_dict[var].append(p[m].detach().cpu().numpy())

    # Compute epoch-level metrics (unchanged)
    metrics_all = gather_metrics(y_true_dict, y_pred_dict)

    # TensorBoard logging: epoch averages for component losses
    if writer and dist.get_rank() == 0:
        writer.add_scalar("Train/EpochLoss", total / max(1, steps), epoch)
        for tag, key in (("PM25","pm25"),("PM10","pm10"),
                         ("SurfOther","obssec"),("CMAQ2D","cmaq2d"),("CMAQ3D","cmaq3d"),
                         ("CAMS_PM","cams_pm")):           # <- NEW
            writer.add_scalar(f"Train/EpochLoss/{tag}", epoch_comp_sum[key] / max(1, steps), epoch)
        # Also write the usual detection metrics
        for v, mm in metrics_all.items():
            t = v.upper()
            writer.add_scalar(f"Train/{t}/F1",        mm["f1"],               epoch)
            writer.add_scalar(f"Train/{t}/Accuracy",  mm["accuracy"],         epoch)
            writer.add_scalar(f"Train/{t}/FAR",       mm["false_alarm_rate"], epoch)
            writer.add_scalar(f"Train/{t}/DR",        mm["detection_rate"],   epoch)
            writer.add_scalar(f"Train/{t}/Recall",    mm["recall"],           epoch)

    if dist.get_rank() == 0:
        txt = " | ".join(
            f"{v.upper()} F1={mm['f1']:.3f} Acc={mm['accuracy']:.3f}"
            for v, mm in metrics_all.items()
        )
        logging.info(f"[epoch {epoch}] TRAIN loss={total/max(1,steps):.4e} | {txt}")

    return total / max(1, steps), metrics_all


# --------------------------------------------------------------------------- #
# 6.  Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser("Aurora Air‑Pollution training (obs)")
    # dates
    ap.add_argument("--train-start", default="2019-01-01")
    ap.add_argument("--train-end",   default="2019-12-31")
    ap.add_argument("--val-start",   default="2020-01-01")
    ap.add_argument("--val-end",     default="2020-12-31")
    # paths
    ap.add_argument("--obs-root", default="./data/obs_npz_27km")
    ap.add_argument("--cmaq-root",
                default="./data/obs_npz",
                help="root folder containing hourly *_cmaq.npy files")
    ap.add_argument("--cams-root", default="./data/cams/",
            help="root folder containing daily CAMS files (*-cams-surface-level.nc / *-cams-atmospheric.nc). "
                 "Leave empty to use <ERA5 root>/../cams")
    ap.add_argument("--flow-root",
                default="./data/cmaq_flow",
                help="root folder containing hourly *_cmaq.npy files")
    ap.add_argument('--data-dir',    default='./data/era5')
    ap.add_argument("--ckpt-dir",    default="./checkpoints")
    ap.add_argument("--exp-name",    default="obs_only_3x3")
    # hyper‑params
    ap.add_argument("--model", default="pollution")
    ap.add_argument("--data-sources", default="obs", help="comma-separated list ⇒ obs, cmaq, era5 …")
    ap.add_argument("--rollout-steps", type=int, default=1, help="usig rollout loss if > 1")
    ap.add_argument("--rollout-weight", type=float, default=0.75)
    ap.add_argument("--use-masking", action="store_true", help="Enable masking augmentation.")
    ap.add_argument("--mask-ratio", type=float, default=0.5, help="Ratio of values to mask in rollout augmentation.")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--accum-steps", type=int, default=8)
    ap.add_argument("--lr-special",  type=float, default=3e-4)
    ap.add_argument("--lr-rest",     type=float, default=1e-5)
    ap.add_argument("--use-lora",    action="store_true")
    ap.add_argument("--resume",      action="store_true")
    ap.add_argument('--use_cmaq',    action='store_true', help="Train with CMAQ variables as extra surface tokens.")
    ap.add_argument('--use_wind_prompt',    action='store_true', help="Train with wind visual prompt.")
    ap.add_argument('--use_cutmix',    action='store_true', help="Train with cutmix with CMAQ.")
    ap.add_argument("--pattern_aware_loss",      action="store_true")
    ap.add_argument("--use_flow",      action="store_true")
    ap.add_argument('--w-era5', type=float, default=0.5, help='ERA5 total weight') 
    ap.add_argument('--use_cmaq_pm_only', action="store_true",  help="use only PM2.5 and PM10 of CMAQ")
    ap.add_argument('--use_hybrid_target', action="store_true",  help="use GT as OBS+CMAQ")
    ap.add_argument('--w-pm', type=float, default=1.0)
    ap.add_argument('--w-obssec', type=float, default=0.03)
    ap.add_argument('--w-cmaq', type=float, default=1.0, help='CMAQ total weight')
    ap.add_argument('--w-cams', type=float, default=0.0,
                help='CAMS total weight for auxiliary loss; 0 disables CAMS loss.')
    ap.add_argument('--hybrid-source', choices=['cmaq','cams','auto'], default='cmaq',
                help="Which source fills missing OBS when --use_hybrid_target is on.")
    ap.add_argument('--cmaq2d-frac', type=float, default=0.5)
    ap.add_argument('--w-pm25-good', type=float, default=0.01, help='Weight for PM2.5 "good" range loss')
    ap.add_argument('--w-pm25-moderate', type=float, default=0.2, help='Weight for PM2.5 "moderate" range loss')
    ap.add_argument('--w-pm25-bad', type=float, default=0.2, help='Weight for PM2.5 "bad" range loss')
    ap.add_argument('--w-pm25-very-bad', type=float, default=0.01, help='Weight for PM2.5 "very bad" range loss')

    ap.add_argument('--rollout-weight-mode', choices=['fixed','linear','exp','sigmoid'],
                 default='linear', help='How to weight rollout steps.')
    ap.add_argument('--rollout-gamma', type=float, default=0.9,
                    help='Shape parameter for exp/sigmoid step weighting.')

    ap.add_argument('--month-weights', type=str, default='',
                    help='JSON mapping month->mult, e.g. {"12":1.2,"1":1.2,"2":1.1}')
    ap.add_argument('--severity-alpha', type=float, default=0.0,
                    help='0 disables severity-based boost; typical 0.2~0.5.')
    ap.add_argument('--severity-threshold', type=float, default=0.05,
                    help='Exceedance fraction baseline for severity boost.')
    ap.add_argument('--pattern-warmup-epochs', type=int, default=0,
                    help='0 disables; else warm-up pattern loss over this many epochs.')
    ap.add_argument('--pattern-max-scale', type=float, default=1.0,
                    help='Cap for pattern loss scaling after warmup.')

    amp_group = ap.add_mutually_exclusive_group()
    amp_group.add_argument("--amp",    dest="amp", action="store_true",  help="enable AMP (fp16)")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false", help="disable AMP (fp32)")
    cams_pm_group = ap.add_mutually_exclusive_group()
    cams_pm_group.add_argument('--cams-pm-only', dest='cams_pm_only', action='store_true',
                            help='Use only PM2.5/PM10 from CAMS (default).')
    cams_pm_group.add_argument('--cams-full', dest='cams_pm_only', action='store_false',
                            help='(Reserved) Allow non-PM CAMS vars if dataset provides them.')
    ap.set_defaults(cams_pm_only=True)
    ap.set_defaults(amp=False)
    args = ap.parse_args()

    # distributed init ------------------------------------------------------ #
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    is_master = dist.get_rank() == 0

    # reproducibility
    seed = 42 + dist.get_rank()
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logging.info(f"args:\n{args}") 
    sources = [s.strip().lower() for s in args.data_sources.split(',')]
    logging.info(f"[INFO] use inputs: {sources}")     # ex) ['obs', 'cmaq']
    # data loaders ---------------------------------------------------------- #
    train_ld, val_ld = get_dataloader(
        args.batch,
        args.data_dir,
        args.obs_root,
        args.train_start,
        args.train_end,
        args.val_start,
        args.val_end,
        cmaq_root=args.cmaq_root,
        flow_root=args.flow_root,
        sources=tuple(sources),
        rollout=args.rollout_steps,
        use_masking=args.use_masking, 
        mask_ratio=args.mask_ratio, 
        use_wind_prompt=args.use_wind_prompt,
        use_cutmix=args.use_cutmix,
        use_cmaq_pm_only=args.use_cmaq_pm_only,
        use_hybrid_target=args.use_hybrid_target,
        use_flow=args.use_flow,
        # --- NEW: CAMS-related ---
        cams_root=(args.cams_root if args.cams_root else None),
        cams_pm_only=args.cams_pm_only,
        hybrid_source=args.hybrid_source,
    )


    # model ----------------------------------------------------------------- #
    in0, tgt0 = first_non_none_batch(train_ld)

    # Derive the token lists from what the dataset actually provides.
    surf_vars = tuple(sorted(in0.surf_vars.keys()))
    real_atmos_vars = tuple(sorted(in0.atmos_vars.keys()))
    static_vars = tuple(sorted(in0.static_vars.keys()))

    logging.info(f"► surf vars (from batch): {surf_vars}")
    logging.info(f"► atmos vars (from batch): {real_atmos_vars}")
    logging.info(f"► static vars (from batch): {static_vars}") 

    # Detect Z (number of vertical levels) from any atmos var, if present
    if real_atmos_vars:
        any_key = real_atmos_vars[0]
        Z_detected = in0.atmos_vars[any_key].shape[2]  # [B, T, Z, H, W]
    else:
        Z_detected = 2

    # Sanity checks
    if real_atmos_vars:
        assert all(v.shape[2] == Z_detected for v in in0.atmos_vars.values()), \
            f"Mixed Z in atmos_vars: {[ (k, v.shape) for k,v in in0.atmos_vars.items() ]}"
        assert len(in0.metadata.atmos_levels) in (0, Z_detected), \
            f"Metadata.atmos_levels length {len(in0.metadata.atmos_levels)} != Z={Z_detected}"

    logging.info(f"[INIT] Detected CMAQ vertical levels (Z) = {Z_detected}")

    latent_levels = Z_detected  # use 1 for 1-level CMAQ, 5 for 5-level CMAQ, etc.

    # ------------------------------------------------------------------ #
    # Choose a window size that is compatible with the current data.
    #   · vertical window  = 1     (always divides latent_levels)
    #   · height  window   = GCD(H//patch,  4) or 1
    #   · width   window   = GCD(W//patch, 12) or 1
    # This keeps memory low while satisfying Swin‑3D requirements.
    # ------------------------------------------------------------------ #
    import math
    H, W          = in0.spatial_shape
    patch_sz      = 2                       # same as you pass to Aurora
    h_tokens      = H // patch_sz
    w_tokens      = W // patch_sz
    win_h         = math.gcd(h_tokens, 8) or 1
    win_w         = math.gcd(w_tokens, 12) or 1
    window_size   = (1, win_h, win_w)       # (levels_axis, H, W)
    logging.info(f"window_size chosen = {window_size}")

    if args.model == "pollution":
        logging.info(f"MODEL: aurora-0.4-air-pollution.ckpt, AMP: {args.amp}")
        model = AuroraAirPollution(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=real_atmos_vars,
            use_lora=args.use_lora,
            patch_size=patch_sz,
            latent_levels=latent_levels,
            window_size=window_size,
        )
        # model.load_checkpoint("microsoft/aurora", "aurora-0.4-air-pollution.ckpt", strict=False)
    else:
        logging.info(f"MODEL: aurora-0.25-pretrained.ckpt, AMP: {args.amp}")
        model = Aurora(
            use_lora=args.use_lora,
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=real_atmos_vars,
            patch_size=patch_sz,
            latent_levels=latent_levels,
            window_size=window_size,
            use_flow=args.use_flow,
        )
        # model.load_checkpoint("microsoft/aurora", "aurora-0.25-pretrained.ckpt", strict=False)
    model = model.cuda(local_rank)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        static_graph=True, 
        gradient_as_bucket_view=True,
    )

    # optimizer – higher LR for pollutant tokens --------------------------- #
    new_vars = ["pm2p5", "pm10", "tcso2", "tcno2", "o3", "tcco"]
    special, rest = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(v in n for v in new_vars) or "_cmaq" in n or "_cams" in n:
            special.append(p)
            new_vars.append(n)
        else:
            rest.append(p)
    opt = torch.optim.Adam([
        {"params": special, "lr": args.lr_special},
        {"params": rest,    "lr": args.lr_rest},
    ])
    logging.info(f"special variables = {new_vars}")
    total_steps = args.epochs * len(train_ld)
    scheduler = OneCycleLR(
        opt,
        max_lr=[args.lr_special, args.lr_rest],
        total_steps=total_steps,
        pct_start=0.1,      # Use 30% of steps to ramp up LR
        div_factor=10,      # Initial LR will be max_lr / 10
        final_div_factor=1e4 # Final LR will be max_lr / 10000
    )
    scaler = torch.amp.GradScaler() if args.amp else None

    # ---------- checkpoint & logging ----------------------------------------- #
    ckpt_root = (
        Path(args.ckpt_dir)
        / f"Train:{args.train_start}-{args.train_end}"
        f"_Test:{args.val_start}-{args.val_end}"
        / args.exp_name
    )
    ckpt_root.mkdir(parents=True, exist_ok=True)

    if is_master:
        fh = logging.FileHandler(ckpt_root / "train.log", mode="a")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logging.getLogger().addHandler(fh)

    start_epoch = 0
    last_ckpt   = ckpt_root / "last.pth"
    if args.resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location="cpu")
        model.load_state_dict(ck["model_state_dict"], strict=False)
        opt.load_state_dict(ck["optimizer_state_dict"])
        if "best_f1_score" in ck:
            best_pm25_f1 = ck["best_f1_score"]
            logging.info(f"[ckpt] resumed best_pm25_f1: {best_pm25_f1:.4f}")
        if "scheduler_state_dict" in ck:
            scheduler.load_state_dict(ck["scheduler_state_dict"])
            logging.info("[ckpt] resumed scheduler state")
        start_epoch = ck["epoch"] + 1
        logging.info(f"[ckpt] resumed from {last_ckpt} (epoch {start_epoch})")

    writer = SummaryWriter(str(ckpt_root / "tb")) if is_master else None

    # ───────────────── train loop ────────────────────────────────────────────
    best_pm25_f1 = 0.0
    month_w = parse_month_weights(args.month_weights)
    dyn_sched = DynamicLossScheduler(
        month_multipliers = month_w,
        severity_alpha = args.severity_alpha,
        severity_threshold = args.severity_threshold,
        pattern_warmup_epochs = args.pattern_warmup_epochs,
        pattern_max_scale = args.pattern_max_scale
    )

    for epoch in range(start_epoch, args.epochs):
        train_ld.sampler.set_epoch(epoch)

        tr_loss, tr_metrics = train_one_epoch(
            train_ld, model, opt, scaler,
            epoch, args.accum_steps, writer, args.amp, args.rollout_weight,
            args.w_pm, args.w_obssec, args.w_cmaq, args.w_era5, args.cmaq2d_frac,
            # --- START: Pass CutMix related arguments ---
            use_cutmix=args.use_cutmix,
            use_flow=args.use_flow,
            obs_path=args.obs_root,
            cmaq_root=args.cmaq_root,
            w_cams=args.w_cams,
            # --- END: Pass CutMix related arguments ---
            w_pm25_good=args.w_pm25_good,
            w_pm25_moderate=args.w_pm25_moderate,
            w_pm25_bad=args.w_pm25_bad,
            w_pm25_very_bad=args.w_pm25_very_bad,
            scheduler=scheduler,
            use_pattern_aware_loss=args.pattern_aware_loss,
            rollout_weight_mode=args.rollout_weight_mode,
            rollout_gamma=args.rollout_gamma,
            dyn_sched=dyn_sched,
        )

        # ── TensorBoard & pretty log (TRAIN) ──────────────────────────────
        if is_master:
            writer.add_scalar("Train/EpochLoss", tr_loss, epoch)
            for var, mm in tr_metrics.items():
                tag = var.upper()
                writer.add_scalar(f"Train/{tag}/F1",       mm["f1"],               epoch)
                writer.add_scalar(f"Train/{tag}/Accuracy", mm["accuracy"],         epoch)
                writer.add_scalar(f"Train/{tag}/FAR",      mm["false_alarm_rate"], epoch)
                writer.add_scalar(f"Train/{tag}/DR",       mm["detection_rate"],   epoch)
                writer.add_scalar(f"Train/{tag}/Recall",   mm["recall"],           epoch)

            logging.info(f"[epoch {epoch}] TRAIN loss={tr_loss:.4e}\n" + _table(tr_metrics))

        # ────────────── validation ───────────────────────────────────────────
        # after creating eval_sampler
        if isinstance(val_ld.sampler, DistributedSampler):
            val_ld.sampler.set_epoch(epoch)
        val_metrics = validate(
            val_ld, model,
            use_rollout_cutmix=args.use_cutmix,
            obs_path=args.obs_root,
            cmaq_root=args.cmaq_root,
            use_flow=args.use_flow,
        )

        if is_master:
            # -------- TensorBoard (VAL) --------
            for var, mm in val_metrics.items():
                tag = var.upper()
                writer.add_scalar(f"Val/{tag}/F1",       mm["f1"],               epoch)
                writer.add_scalar(f"Val/{tag}/Accuracy", mm["accuracy"],         epoch)
                writer.add_scalar(f"Val/{tag}/FAR",      mm["false_alarm_rate"], epoch)
                writer.add_scalar(f"Val/{tag}/DR",       mm["detection_rate"],   epoch)
                writer.add_scalar(f"Val/{tag}/Recall",   mm["recall"],           epoch)

            # -------- pretty markdown table --------
            logging.info(f"[epoch {epoch}] VAL\n" + _table(val_metrics))

        pm25_f1 = val_metrics.get("pm2p5", {}).get("f1", 0.0)
        if pm25_f1 > best_pm25_f1 and is_master:
            best_pm25_f1 = pm25_f1
            save_ckpt(epoch, model, opt, scheduler, tr_loss, pm25_f1, ckpt_root / "best_pm25.pth")

        if is_master:
            save_ckpt(epoch, model, opt, scheduler, tr_loss, pm25_f1, last_ckpt)

    if writer:
        writer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
