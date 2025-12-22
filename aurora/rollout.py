"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""
from __future__ import annotations
import dataclasses
from typing import Generator

import torch
from torch import Tensor

from aurora.batch import Batch
from aurora.model.aurora import Aurora
from aurora.normalisation import normalise_surf_var, SURF_STATS

__all__ = ["rollout"]


# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #
def _last_n_frames(x: Tensor, n: int) -> Tensor:
    """Keep only the last *n* frames along time‑dimension (dim=1)."""
    return x[:, -n:]


def _apply_station_mask(pred: Tensor, mask_const: Tensor) -> Tensor:
    return pred * mask_const      # (B,1,H,W)  –  non‑station → 0

# Add these imports at the top of aurora/rollout.py
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from scipy.ndimage import zoom

# This list is needed to find the variable index in CMAQ files.
CONC_VARS = [
    "SO4_25", "NH4_25", "NO3_25", "ORG_25", "EC_25", "MISC_25",
    "PM2P5", "tcso2", "tcco", "tcno2", "O3", "NO", "NOx",
    "SO4_10", "NH4_10", "NO3_10", "ORG_10", "EC_10", "MISC_10",
    "PM10", "ISOPRENE", "OLES", "AROS", "ALKS",
]

def _load_obs_for_ts(ts: datetime, npz_path: str) -> np.ndarray | None:
    """Loads the observation data for a single timestamp."""
    try:
        date_nodash = ts.strftime('%Y%m%d')
        hour = ts.strftime('%H')
        file_path = Path(npz_path) / f"{date_nodash}{hour}_obs.npz"
        with np.load(file_path) as data:
            return data['obs_channels'] # Shape: (C, H, W)
    except FileNotFoundError:
        return None

def _load_cmaq_for_ts(
    ts: datetime, 
    cmaq_root: str,
    target_h: int | None = None,  # Add target height
    target_w: int | None = None   # Add target width
) -> dict[str, np.ndarray] | None:
    """Loads the CMAQ concentration data for a single timestamp."""
    try:
        date_str, hh = ts.strftime("%Y%m%d"), ts.strftime("%H")
        yy, mm, dd = date_str[:4], date_str[4:6], date_str[6:]
        
        base = Path(cmaq_root) / yy / mm / dd / "NIER_27_01"
        conc_f = base / f"{date_str}_x_conc.npy"
        
        if not conc_f.exists():
            return None

        conc_full = np.load(conc_f, mmap_mode="r")
        
        hour_to_idx = {"00": 0, "06": 1, "12": 2, "18": 3}
        time_idx = hour_to_idx[hh]
        
        # Slice for the specific time. Assume Z-level is present.
        if conc_full.ndim == 5:
            conc_t = conc_full[time_idx, :, 0, :, :]   # (V,H,W) surface
        elif conc_full.ndim == 4:
            conc_t = conc_full[time_idx, :, :, :]
        else:
            return None
        
        # --- START OF MODIFICATION: RESIZING LOGIC ---
        if target_h is not None and target_w is not None:
            orig_h, orig_w = conc_t.shape[-2:]
            if (orig_h, orig_w) != (target_h, target_w):
                zoom_h = target_h / orig_h
                zoom_w = target_w / orig_w
                # The shape is (V, H, W), so zoom factors are (1, zoom_h, zoom_w)
                conc_t = zoom(conc_t, (1, zoom_h, zoom_w), order=1)
        # --- END OF MODIFICATION ---

        # --- IMPORTANT: match dataloader orientation ---
        # conc_t = np.flip(conc_t, axis=-2).copy()
        return {'conc': conc_t.astype(np.float32)}

    except Exception:
        return None

def build_next_input_from_pred_and_cmaq(
    pred_vars: dict[str, torch.Tensor],
    obs_data: np.ndarray | None,
    cmaq_data: dict[str, np.ndarray] | None,
    device: torch.device
) -> dict[str, torch.Tensor]:
    """
    Build the next-step input from the previous prediction (at station pixels)
    and CMAQ (elsewhere). OBS is used ONLY to locate station pixels (mask).
    """
    # Map surface keys to CMAQ species keys (order in CONC_VARS is fixed)
    var_map = {
        'pm2p5': ('PM2P5', 0),
        'pm10' : ('PM10' , 1),
        'tcso2': ('tcso2', 2),
        'tcno2': ('tcno2', 3),
        'o3'   : ('O3'   , 4),
        'tcco' : ('tcco' , 5),
    }

    # Prepare OBS mask (True where a station reports a value)
    obs_mask_by_ch: dict[str, torch.Tensor] = {}
    if obs_data is not None:
        obs_t = torch.from_numpy(obs_data).to(device)  # shape (C,H,W)
        for surf_key, (_, obs_idx) in var_map.items():
            # obs > 0 indicates station location on this grid
            m = (obs_t[obs_idx] > 0).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            obs_mask_by_ch[surf_key] = m
    else:
        # No OBS → no station mask (behave as if there are no stations)
        for surf_key in var_map.keys():
            obs_mask_by_ch[surf_key] = None

    # Prepare CMAQ slices (physical) for each variable
    cmaq_phys_by_ch: dict[str, torch.Tensor] = {}
    if cmaq_data is not None and 'conc' in cmaq_data:
        cmaq_t = torch.from_numpy(cmaq_data['conc']).to(device)  # (V,H,W)
        for surf_key, (cmaq_name, _) in var_map.items():
            try:
                j = CONC_VARS.index(cmaq_name)  # species index
            except ValueError:
                continue
            # raw CMAQ at surface level already extracted upstream into cmaq_data['conc'][j]
            cmaq_raw = cmaq_t[j].unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            cmaq_phys_by_ch[surf_key] = cmaq_raw
    # else: CMAQ missing → will fall back to pure prediction at station pixels and zeros elsewhere

    mixed: dict[str, torch.Tensor] = {}
    # Use prediction at stations, CMAQ elsewhere (all in physical space)
    for surf_key in pred_vars.keys():
        pred_slice = pred_vars[surf_key]  # [B,1,H,W], already normalized
        B = pred_slice.shape[0]

        # default: if CMAQ for this variable is missing, keep prediction everywhere
        bg = cmaq_phys_by_ch.get(surf_key, None)
        if bg is None:
            mixed[surf_key] = pred_slice
            continue

        # station mask for this channel (may be None)
        m = obs_mask_by_ch.get(surf_key, None)
        if m is None:
            # No stations known → use CMAQ everywhere
            mixed[surf_key] = bg.expand_as(pred_slice)
            continue

        # Broadcast mask and background to batch size
        m_b = m.expand(B, -1, -1, -1)                  # [B,1,H,W]
        bg_b = bg.expand_as(pred_slice)                # [B,1,H,W]
        mixed[surf_key] = torch.where(m_b, pred_slice, bg_b)

    # Keep any variables that are not in var_map as-is
    return mixed

# --------------------------------------------------------------------- #
# main roll‑out
# --------------------------------------------------------------------- #
# --- Find and modify the rollout function in aurora/rollout.py ---

def rollout(
    model: Aurora, 
    batch: Batch, 
    steps: int,
    # --- 💡 ADDED ARGUMENTS START 💡 ---
    use_rollout_cutmix: bool = False,
    obs_path: str | None = None,
    cmaq_root: str | None = None
    # --- 💡 ADDED ARGUMENTS END 💡 ---
) -> Generator[Batch, None, None]:
    """
    Performs an autoregressive rollout of the model.
    
    Args:
        model: The Aurora model.
        batch: The initial input batch.
        steps: The number of rollout steps to perform.
        use_rollout_cutmix: If True, apply CutMix at each rollout step.
        obs_path: Path to the root directory of observation (.npz) files.
        cmaq_root: Path to the root directory of CMAQ data.
    """
    # ... (the first few lines of the function are the same)
    p = next(model.parameters())
    batch = (model.batch_transform_hook(batch)
             .type(p.dtype)
             .crop(model.patch_size)
             .to(p.device))

    T_MAX = model.max_history_size
    target_h, target_w = batch.spatial_shape

    # The original `station_mask` is not needed for CutMix logic,
    # but we'll keep it for the original behavior.
    station_mask = {
        k: (v != 0).any(dim=1, keepdim=True).float()
        for k, v in batch.surf_vars.items()
    }

    for i in range(steps): # Use an index `i` to track the step
        batch = dataclasses.replace(
            batch,
            surf_vars={k: _last_n_frames(v, T_MAX) for k, v in batch.surf_vars.items()},
            atmos_vars={k: _last_n_frames(v, T_MAX) for k, v in batch.atmos_vars.items()},
        )

        pred = model.forward(batch)
        yield pred

        pred_detached = dataclasses.replace(
            pred,
            surf_vars={k: v.detach() for k, v in pred.surf_vars.items()},
            atmos_vars={k: v.detach() for k, v in pred.atmos_vars.items()},
        )
        
        # --- 💡 MODIFIED LOGIC START 💡 ---
        next_input_surf_vars = pred_detached.surf_vars

        if use_rollout_cutmix and (obs_path is not None) and (cmaq_root is not None):
            last_known_time = batch.metadata.time[-1]
            pred_time = last_known_time + timedelta(hours=6)

            # Load OBS only to get the station mask at pred_time
            obs_for_pred_time = _load_obs_for_ts(pred_time, obs_path)    # (C,H,W) or None
            cmaq_for_pred_time = _load_cmaq_for_ts(
                pred_time, 
                cmaq_root, 
                target_h=target_h, 
                target_w=target_w
            ) # {'conc':(V,H,W)} or None

            next_input_surf_vars = build_next_input_from_pred_and_cmaq(
                pred_detached.surf_vars,
                obs_for_pred_time,
                cmaq_for_pred_time,
                p.device
            )
            #  next_input_surf_vars = pred_detached.surf_vars
        else:
            # Legacy fallback: keep prediction and zero-out non-stations (not recommended)
            next_input_surf_vars = {
                k: _apply_station_mask(v, station_mask[k])
                for k, v in pred_detached.surf_vars.items()
            }
        # --- 💡 MODIFIED LOGIC END 💡 ---

        next_surf = {
            k: torch.cat(
                [batch.surf_vars[k][:, 1:], v], dim=1
            ) for k, v in next_input_surf_vars.items()
        }

        next_atmos = {
            k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
            for k, v in pred_detached.atmos_vars.items()
        }

        batch = dataclasses.replace(
            batch,
            surf_vars=next_surf,
            atmos_vars=next_atmos,
            metadata=pred.metadata
        )