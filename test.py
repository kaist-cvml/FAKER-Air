# ---------- lib ----------
import argparse
import os, sys
import numpy as np
import random
import logging
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from torch.utils.data._utils.collate import default_collate

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from aurora import Aurora, AuroraAirPollution, Batch
from aurora import rollout as aurora_rollout
from aurora.dataloader import (
    WeatherDataset, aurora_collate_fn, make_lat_lon,
    collate_batches, CONC_VARS, M2D_VARS, M3D_VARS, _meta_update
)
from aurora.utils import compute_metrics, compute_multiclass_metrics

from pathlib import Path
import io, imageio, calendar, dataclasses
from datetime import datetime
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from PIL import Image
from scipy.ndimage import zoom

sys.path.append(os.path.join(os.getcwd(), 'aurora'))

# Set up logging.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from matplotlib.colors import ListedColormap, BoundaryNorm

CONC_VARS = [
    "SO4_25", "NH4_25", "NO3_25", "ORG_25", "EC_25", "MISC_25",
    "PM2P5", "tcso2", "tcco", "tcno2", "O3", "NO", "NOx",
    "SO4_10", "NH4_10", "NO3_10", "ORG_10", "EC_10", "MISC_10",
    "PM10", "ISOPRENE", "OLES", "AROS", "ALKS",
]

AQI_COLORS = ["#0066ff",   # good      (blue)
              "#27ae60",   # moderate  (green)
              "#f1c40f",   # bad       (yellow)
              "#c0392b"]   # very bad  (red)

VIS_VARS = dict(
    pm10 = dict(
        scale=1e-9,
        bounds=[0, 30, 80, 150, 1_000],            # µg m⁻³
        unit="µg m⁻³",
    ),
    pm2p5 = dict(
        scale=1e-9,
        bounds=[0, 15, 35, 75, 800],
        unit="µg m⁻³",
    ),
    o3   = dict(                                   
        scale=1.0,                                 
        bounds=[0, 0.03, 0.090, 0.150, 0.300],
        unit="ppb",
    ),
)

_METRIC_STR = ("F1={f1:.2f}, Acc={accuracy:.2f}, "
               "Precision={precision:.2f}, Recall={recall:.2f}, "
               "FAR={false_alarm_rate:.2f}, DR={detection_rate:.2f}, "
               "CSI={csi:.2f}, ETS={ets:.2f}, TSS={tss:.2f}")

def _fmt(m: dict) -> str:
    # Safe formatting with defaults so missing keys won't crash
    defaults = {
        "f1": np.nan, "accuracy": np.nan, "precision": np.nan, "recall": np.nan,
        "false_alarm_rate": np.nan, "detection_rate": np.nan,
        "csi": np.nan, "ets": np.nan, "tss": np.nan
    }
    z = {**defaults, **(m or {})}
    return _METRIC_STR.format(**z)

def _load_obs(var: str, ts: datetime, npz_root: str) -> np.ndarray:
    """Read `<YYYYMMDDHH>_obs.npz` and return the channel that matches *var*."""
    file = Path(npz_root) / f"{ts:%Y%m%d%H}_obs.npz"
    with np.load(file) as npz:
        # order: pm2p5, pm10, so2, no2, o3, co   (see dataset code)
        CHIDX = dict(pm2p5=0, pm10=1, so2=2, no2=3, o3=4, co=5)[var]
        return npz["obs_channels"][CHIDX]

def regrid_to_latlon(
    data2d: np.ndarray,
    lats_2d: np.ndarray,
    lons_2d: np.ndarray,
    H: int | None = None,
    W: int | None = None,
    lon_range: tuple[float, float] | None = None,
    lat_range: tuple[float, float] | None = None,
    fill_nearest: bool = True,
):
    """
    Interpolate a curvilinear (H,W) field onto a regular lon/lat grid (rectangular).

    Args
    ----
    data2d      : (H,W) values on a curvilinear grid.
    lats_2d     : (H,W) latitudes of the source grid (cell centers).
    lons_2d     : (H,W) longitudes of the source grid (cell centers).
    H, W        : target grid shape; if None, use source shape.
    lon_range   : (min_lon, max_lon) for the target grid; default=min/max of source.
    lat_range   : (min_lat, max_lat) for the target grid; default=min/max of source.
    fill_nearest: fill NaNs left by 'linear' interpolation with a 'nearest' pass.

    Returns
    -------
    lon_g, lat_g, vals : (H,W) regular lon/lat mesh and the interpolated values.
    """
    data2d  = np.asarray(data2d, dtype=float)
    lats_2d = np.asarray(lats_2d, dtype=float)
    lons_2d = _wrap_lon_180(np.asarray(lons_2d, dtype=float))

    if H is None or W is None:
        H, W = data2d.shape

    if lon_range is None:
        lon_min, lon_max = float(np.nanmin(lons_2d)), float(np.nanmax(lons_2d))
    else:
        lon_min, lon_max = lon_range

    if lat_range is None:
        lat_min, lat_max = float(np.nanmin(lats_2d)), float(np.nanmax(lats_2d))
    else:
        lat_min, lat_max = lat_range

    lon_t = np.linspace(lon_min, lon_max, W)
    lat_t = np.linspace(lat_min, lat_max, H)
    lon_g, lat_g = np.meshgrid(lon_t, lat_t)

    # First, linear interpolation (smooth)
    vals = griddata(
        (lons_2d.ravel(), lats_2d.ravel()),
        data2d.ravel(),
        (lon_g, lat_g),
        method="linear"
    )

    # Optionally fill voids outside the convex hull using nearest neighbor
    if fill_nearest and np.isnan(vals).any():
        nearest = griddata(
            (lons_2d.ravel(), lats_2d.ravel()),
            data2d.ravel(),
            (lon_g, lat_g),
            method="nearest"
        )
        vals[np.isnan(vals)] = nearest[np.isnan(vals)]

    return lon_g, lat_g, vals

def nearest_on_curvilinear(stn_lat, stn_lon, lats2d, lons2d):
    tree = cKDTree(np.c_[lats2d.ravel(), lons2d.ravel()])
    dist, idx = tree.query(np.c_[stn_lat, stn_lon])
    r, c = np.unravel_index(idx, lats2d.shape)
    return r, c

def load_cmaq_for_rollout(
    timestamps: Tuple[datetime, ...], 
    variable_name: str, 
    cmaq_root: str,
    target_h: int,  # Add target height as a required argument
    target_w: int,  # Add target width as a required argument
) -> torch.Tensor | None:
    """
    # Loads CMAQ data for a given sequence of timestamps and a specific variable.
    #
    # Args:
    #     timestamps: A tuple of datetime objects for the rollout horizon.
    #     variable_name: The name of the variable to load (e.g., 'pm2p5').
    #     cmaq_root: The root directory of the CMAQ dataset.
    #
    # Returns:
    #     A torch.Tensor of shape (T, H, W) containing the CMAQ data,
    #     or None if data loading fails completely.
    """
    # --- Map user-friendly variable names to the names in CONC_VARS ---
    var_map = {'pm2p5': 'PM2P5', 'pm10': 'PM10', 'o3': 'O3', 'tcso2':'tcso2', 'tcno2':'tcno2', 'tcco':'tcco'}
    cmaq_var_name = var_map.get(variable_name.lower())
    if not cmaq_var_name:
        logging.warning(f"Variable '{variable_name}' not mapped for CMAQ loading.")
        return None
    
    try:
        var_idx = CONC_VARS.index(cmaq_var_name)
    except ValueError:
        logging.warning(f"CMAQ variable '{cmaq_var_name}' not found in CONC_VARS list.")
        return None

    cmaq_sequence = []
    hour_to_idx = {"00": 0, "06": 1, "12": 2, "18": 3}

    # --- Loop through each timestamp in the rollout horizon ---
    for ts in timestamps:
        try:
            date_str = ts.strftime("%Y%m%d")
            yy, mm, dd = ts.strftime("%Y"), ts.strftime("%m"), ts.strftime("%d")
            
            cmaq_file = Path(cmaq_root) / yy / mm / dd / "NIER_27_01" / f"{date_str}_x_conc.npy"

            if not cmaq_file.exists():
                # Append a tensor of NaNs if a file is missing for proper alignment
                cmaq_sequence.append(torch.full((target_h, target_w), float('nan'))) # Assuming H,W = 128,174
                continue

            cmaq_data_full = np.load(cmaq_file, mmap_mode="r")
            time_idx = hour_to_idx[ts.strftime("%H")]
            
            # --- Slice data for the correct time, variable, and surface level ---
            if cmaq_data_full.ndim == 5:  # Shape: (T, V, Z, H, W)
                cmaq_slice = cmaq_data_full[time_idx, var_idx, 0, :, :]
            elif cmaq_data_full.ndim == 4:  # Shape: (T, V, H, W)
                cmaq_slice = cmaq_data_full[time_idx, var_idx, :, :]
            else:
                cmaq_slice = np.full((128, 174), float('nan'))
            
            # Resize the CMAQ slice to match the target OBS shape.
            orig_h, orig_w = cmaq_slice.shape
            if (orig_h, orig_w) != (target_h, target_w):
                zoom_h = target_h / orig_h
                zoom_w = target_w / orig_w
                # The shape of cmaq_slice is (H, W), so zoom factors are simple.
                cmaq_slice = zoom(cmaq_slice, (zoom_h, zoom_w), order=1)
            # Flip the y-axis (H dimension, which is axis -2) to match the dataloader's preprocessing.
            # .copy() is used for safety with memory-mapped arrays.
            # cmaq_slice = np.flip(cmaq_slice, axis=-2).copy()
            # =======================================================

            cmaq_sequence.append(torch.from_numpy(cmaq_slice.astype(np.float32)))

        except Exception as e:
            logging.error(f"Failed to load or process CMAQ for {ts}: {e}")
            cmaq_sequence.append(torch.full((target_h, target_w), float('nan')))

    if not cmaq_sequence:
        return None
    
    # --- Stack the sequence into a single tensor of shape (T, H, W) ---
    return torch.stack(cmaq_sequence, dim=0)

def load_cmaq_for_batch(
    timestamps: Tuple[datetime, ...], 
    variable_name: str, 
    cmaq_root: str,
    target_shape: Tuple[int, int] = (128, 174) # Set your model's output H, W
) -> torch.Tensor | None:
    """
    Loads CMAQ data for a given set of timestamps and a specific variable.

    Args:
        timestamps: A tuple of datetime objects from the batch metadata.
        variable_name: The name of the variable to load (e.g., 'pm2p5').
        cmaq_root: The root directory of your CMAQ dataset.
        target_shape: The (H, W) shape to which the data should conform.

    Returns:
        A torch.Tensor of shape (T, H, W) containing the CMAQ data,
        or None if data for any timestamp could not be loaded.
    """
    # --- Map friendly variable names to the names in CONC_VARS ---
    var_map = {'pm2p5': 'PM2P5', 'pm10': 'PM10', 'o3': 'O3'}
    cmaq_var_name = var_map.get(variable_name.lower())
    if not cmaq_var_name:
        print(f"[WARN] Variable '{variable_name}' is not mapped for CMAQ loading.")
        return None
    
    try:
        var_idx = CONC_VARS.index(cmaq_var_name)
    except ValueError:
        print(f"[WARN] CMAQ variable '{cmaq_var_name}' not found in CONC_VARS list.")
        return None

    cmaq_sequence = []
    hour_to_idx = {"00": 0, "06": 1, "12": 2, "18": 3}

    # --- Loop through each timestamp in the batch's target horizon ---
    for ts in timestamps:
        try:
            date_str = ts.strftime("%Y%m%d")
            yy, mm, dd = ts.strftime("%Y"), ts.strftime("%m"), ts.strftime("%d")
            
            cmaq_file = Path(cmaq_root) / yy / mm / dd / "NIER_27_01" / f"{date_str}_x_conc.npy"

            if not cmaq_file.exists():
                print(f"[WARN] CMAQ file not found: {cmaq_file}")
                # Append a tensor of NaNs if a file is missing
                cmaq_sequence.append(torch.full(target_shape, float('nan')))
                continue

            cmaq_data_full = np.load(cmaq_file, mmap_mode="r")
            time_idx = hour_to_idx[ts.strftime("%H")]
            
            # --- Slice the data for the correct time and variable ---
            # Assumes surface level (Z=0) if there is a vertical dimension
            if cmaq_data_full.ndim == 5:  # Shape: (T, V, Z, H, W)
                cmaq_slice = cmaq_data_full[time_idx, var_idx, 0, :, :]
            elif cmaq_data_full.ndim == 4:  # Shape: (T, V, H, W)
                cmaq_slice = cmaq_data_full[time_idx, var_idx, :, :]
            else:
                print(f"[WARN] Unexpected CMAQ data shape: {cmaq_data_full.shape}")
                cmaq_slice = np.full(target_shape, float('nan'))
            
            # NOTE: Add resizing/interpolation here if CMAQ grid differs from model output
            # For example, using scipy.ndimage.zoom if needed.
            # if cmaq_slice.shape != target_shape:
            #     ... resize logic ...

            cmaq_sequence.append(torch.from_numpy(cmaq_slice.astype(np.float32)))

        except Exception as e:
            print(f"[ERROR] Failed to load or process CMAQ for {ts}: {e}")
            cmaq_sequence.append(torch.full(target_shape, float('nan')))

    # --- Stack the sequence into a single tensor ---
    if not cmaq_sequence:
        return None
    
    # Shape: (T, H, W) where T is the prediction horizon
    return torch.stack(cmaq_sequence, dim=0)

# ----------------------------------------------------------------------------

def _save_gif(
    frames: list[np.ndarray], 
    out_path: Path, 
    fps: int = 1,               # Add fps parameter (e.g., 2 frames per second)
    upscale: bool = True,       # Add parameter to control upscaling
    target_height: int = 1080   # Add parameter for target resolution
) -> None:
    """
    Saves a list of image frames as a GIF, with options for controlling speed and resolution.
    
    Args:
        frames: A list of numpy arrays, where each array is an image frame.
        out_path: The file path to save the GIF.
        fps: Frames per second for the GIF animation.
        upscale: If True, upscale frames to the target_height.
        target_height: The target height in pixels for upscaling.
    """
    # This line is not strictly necessary and can consume a lot of memory for long GIFs.
    # imageio.mimsave can handle a list of frames directly.
    # frames = np.array(frames) 

    if upscale and frames[0].shape[0] < target_height:
        H = target_height
        # Use a generator expression for slightly better memory efficiency during the list creation
        resized_frames = [
            np.array(Image.fromarray(f).resize(
                (round(H * f.shape[1] / f.shape[0]), H), 
                Image.BICUBIC
            )) for f in frames
        ]
        imageio.mimsave(out_path, resized_frames, fps=fps)
    else:
        # If not upscaling, save the original frames
        imageio.mimsave(out_path, frames, fps=fps)
        
    logging.info(f"  ↳ saved GIF → {out_path} (fps={fps})")

def _nearest_grid_indices(stn_lats, stn_lons, grid_lat_1d, grid_lon_1d):
    """
    stn_lats, stn_lons: (N,) 
    grid_lat_1d: (H,) 
    grid_lon_1d: (W,) 
    return: (rows, cols)
    """
    lat_idx = np.abs(grid_lat_1d[None, :] - stn_lats[:, None]).argmin(axis=1)
    lon_idx = np.abs(grid_lon_1d[None, :] - stn_lons[:, None]).argmin(axis=1)
    return lat_idx, lon_idx

def centers_to_edges_2d(lon_c, lat_c):
    """(H, W) 중심좌표 → (H+1, W+1) 모서리좌표 근사 생성"""
    lon_c = np.asarray(lon_c, dtype=float); lat_c = np.asarray(lat_c, dtype=float)
    H, W = lon_c.shape
    lon_e = np.empty((H+1, W+1)); lat_e = np.empty((H+1, W+1))

    # 내부 셀 모서리 = 인접 4개 중심 평균
    lon_e[1:-1, 1:-1] = 0.25*(lon_c[:-1,:-1] + lon_c[1:,:-1] + lon_c[:-1,1:] + lon_c[1:,1:])
    lat_e[1:-1, 1:-1] = 0.25*(lat_c[:-1,:-1] + lat_c[1:,:-1] + lat_c[:-1,1:] + lat_c[1:,1:])

    # 경계는 외삽
    lon_e[0, 1:-1]  = 2*lon_c[0, :-1]  - lon_e[1, 1:-1]
    lon_e[-1,1:-1]  = 2*lon_c[-1,:-1]  - lon_e[-2,1:-1]
    lon_e[1:-1, 0]  = 2*lon_c[:-1, 0]  - lon_e[1:-1,1]
    lon_e[1:-1,-1]  = 2*lon_c[:-1,-1]  - lon_e[1:-1,-2]

    lat_e[0, 1:-1]  = 2*lat_c[0, :-1]  - lat_e[1, 1:-1]
    lat_e[-1,1:-1]  = 2*lat_c[-1,:-1]  - lat_e[-2,1:-1]
    lat_e[1:-1, 0]  = 2*lat_c[:-1, 0]  - lat_e[1:-1,1]
    lat_e[1:-1,-1]  = 2*lat_c[:-1,-1]  - lat_e[1:-1,-2]

    # 네 귀퉁이
    lon_e[0,0]   = 2*lon_c[0,0]     - lon_e[1,1]
    lon_e[0,-1]  = 2*lon_c[0,-1]    - lon_e[1,-2]
    lon_e[-1,0]  = 2*lon_c[-1,0]    - lon_e[-2,1]
    lon_e[-1,-1] = 2*lon_c[-1,-1]   - lon_e[-2,-2]

    return _wrap_lon_180(lon_e), lat_e

def nearest_on_curvilinear(stn_lat, stn_lon, lats2d, lons2d):
    """
    2D 곡선 그리드(lats2d, lons2d)에서
    관측소 좌표(stn_lat, stn_lon)와 가장 가까운 격자의
    행(r), 열(c) 인덱스를 찾습니다.
    """
    # 2D 좌표를 1D로 풀어서 KDTree 생성
    tree = cKDTree(np.c_[lats2d.ravel(), lons2d.ravel()])
    
    # 관측소 좌표(N, 2)에 대해 가장 가까운 점 검색
    dist, idx = tree.query(np.c_[stn_lat, stn_lon])
    
    # 1D 인덱스(idx)를 2D 인덱스(r, c)로 변환
    r, c = np.unravel_index(idx, lats2d.shape)
    return r, c

# --- skill scores (from 2x2 confusion matrix) -----------------------------
def _skill_from_cm(cm: np.ndarray) -> dict:
    """
    Compute CSI, ETS, TSS (and bias) from a 2x2 confusion matrix:
    cm = [[TN, FP],
          [FN, TP]]
    Returns floats; uses np.nan when undefined.
    """
    if cm is None or np.asarray(cm).shape != (2, 2):
        return {"csi": np.nan, "ets": np.nan, "tss": np.nan, "bias": np.nan}

    cm = np.asarray(cm, dtype=float)
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    n = tn + fp + fn + tp

    # CSI (== IoU for event mask)
    den_csi = tp + fp + fn
    csi = (tp / den_csi) if den_csi > 0 else np.nan

    # ETS (equitable threat score; remove hits due to chance)
    he = ((tp + fp) * (tp + fn)) / n if n > 0 else 0.0
    den_ets = (tp + fp + fn - he)
    ets = ((tp - he) / den_ets) if den_ets > 0 else np.nan

    # TSS (a.k.a. Peirce Skill Score) = TPR - FPR
    tpr_den = tp + fn
    fpr_den = fp + tn
    tpr = (tp / tpr_den) if tpr_den > 0 else np.nan
    fpr = (fp / fpr_den) if fpr_den > 0 else np.nan
    tss = (tpr - fpr) if (not np.isnan(tpr) and not np.isnan(fpr)) else np.nan

    # Frequency Bias (over/under-forecasting of event area)
    bias_den = tp + fn
    bias = ((tp + fp) / bias_den) if bias_den > 0 else np.nan

    return {"csi": csi, "ets": ets, "tss": tss, "bias": bias}


def attach_skill_scores(m: dict) -> dict:
    """
    Attach CSI/ETS/TSS/Bias into a metric dict produced by `compute_metrics`.
    Expects `confusion_matrix_raw` in m.
    """
    if not isinstance(m, dict) or "confusion_matrix_raw" not in m:
        return m
    skills = _skill_from_cm(m["confusion_matrix_raw"])
    m.update(skills)
    return m

def plot_2x2_pixel_and_points(
    input_map, gt_map, pred_map,
    stations_lat, stations_lon,
    lons_2d, lats_2d,
    cfg, title="",
    show_pred_on_top_right=False
):
    """
    # Creates a 2x2 visualization with a shared custom color bar on the right
    # and returns it as a numpy array for GIF creation.
    #
    # Args:
    #     input_map, gt_map, pred_map: (H,W) arrays in physical units.
    #     stations_lat/lon: (N,) arrays of station coordinates.
    #     grid_lat_1d, grid_lon_1d: 1D grid coordinates from make_lat_lon.
    #     cfg: Configuration dictionary from VIS_VARS (e.g., VIS_VARS['pm2p5']).
    #     title: Title for the entire figure.
    #     show_pred_on_top_right: If True, the prediction map is shown top-right; otherwise, GT is shown.
    """
    # --- START: Custom Colormap Generation Logic ---
    bounds = cfg["bounds"]

    # Define the start and end colors for each segment of the color bar
    color_pairs = [
        ['#2795EF', '#1976D2'],  # Blue range for "Good"
        ['#8FDB91', '#248929'],  # Green range for "Moderate"
        ['#EBE082', '#F3B517'],  # Yellow/Orange range for "Bad"
        ['#F88967', '#D81E1E']   # Red range for "Very Bad"
    ]

    all_colors = []
    vmin, vmax = bounds[0], bounds[-1]
    
    n_colors_total = 256 # Total number of discrete colors in the final colormap

    # Create a gradient for each segment defined by the bounds
    for i in range(len(bounds) - 1):
        start_val, end_val = bounds[i], bounds[i+1]
        
        if i < len(color_pairs):
            start_color, end_color = color_pairs[i]
        else: # Fallback for any extra segments
            start_color, end_color = ['#FFFFFF', '#000000']

        # Determine how many colors this segment gets, proportional to its value range
        segment_range = end_val - start_val
        num_segment_colors = max(1, int(np.round((segment_range / (vmax - vmin)) * n_colors_total)))

        # Linearly interpolate between the start and end colors for this segment
        segment_gradient = np.linspace(0, 1, num_segment_colors)
        colors_for_segment = [
            tuple(
                np.array(plt.cm.colors.to_rgba(start_color)) * (1 - t) + 
                np.array(plt.cm.colors.to_rgba(end_color)) * t
            ) for t in segment_gradient
        ]
        all_colors.extend(colors_for_segment)

    cmap = ListedColormap(all_colors)
    norm = Normalize(vmin=vmin, vmax=vmax)
    ticks = bounds
    # --- END: Custom Colormap Generation Logic ---

    fig = plt.figure(figsize=(12, 9), dpi=150, constrained_layout=True)
    
    # [수정] 모든 projection을 PlateCarree로 통일합니다.
    projection = ccrs.PlateCarree()
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05], height_ratios=[1, 1])

    # [수정] build_native_proj() 호출 제거 및 projection=projection으로 설정
    ax11 = fig.add_subplot(gs[0, 0], projection=projection)
    ax12 = fig.add_subplot(gs[0, 1], projection=projection)
    ax21 = fig.add_subplot(gs[1, 0], projection=projection)
    ax22 = fig.add_subplot(gs[1, 1], projection=projection)
    cax  = fig.add_subplot(gs[:, 2])

    top_right_map = pred_map if show_pred_on_top_right else gt_map
    im1 = ax11.pcolormesh(lons_2d, lats_2d, input_map, norm=norm, cmap=cmap,
                          transform=ccrs.PlateCarree(), shading="auto")
    im2 = ax12.pcolormesh(lons_2d, lats_2d, top_right_map, norm=norm, cmap=cmap,
                          transform=ccrs.PlateCarree(), shading="auto")
    
    # [수정] 2D Curvilinear 그리드에서 가장 가까운 격자점을 찾도록
    r_idx, c_idx = nearest_on_curvilinear(stations_lat, stations_lon, lats_2d, lons_2d)
    gt_pts   = gt_map[r_idx, c_idx]
    pred_pts = pred_map[r_idx, c_idx]

    ax21.scatter(stations_lon, stations_lat, c=gt_pts,   s=15, cmap=cmap, norm=norm, edgecolors="k", linewidths=0.2, transform=ccrs.PlateCarree())
    ax22.scatter(stations_lon, stations_lat, c=pred_pts, s=15, cmap=cmap, norm=norm, edgecolors="k", linewidths=0.2, transform=ccrs.PlateCarree())
    
    map_axes = [ax11, ax12, ax21, ax22]
    
    # [수정] plot_extent를 데이터 범위(97~170)보다 약간 넓게 설정
    plot_extent=[100, 155, 20, 52] # [West, East, South, North]

    for ax in map_axes:
        ax.set_extent(plot_extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor='black')
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
        ax.add_feature(cfeature.LAND, edgecolor='lightgray', facecolor='lightgray', zorder=-1)
        ax.add_feature(cfeature.OCEAN, facecolor='aliceblue', zorder=-1)

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels, gl.right_labels = False, False
        gl.xlabel_style, gl.ylabel_style = {'size': 8}, {'size': 8}

    cb = fig.colorbar(im2, cax=cax, ticks=ticks)
    cb.set_label(cfg["unit"])
    
    ax11.set_title("Input (OBS⊕CMAQ)")
    ax12.set_title("GT (OBS⊕CMAQ, t=s)" if not show_pred_on_top_right else "Prediction (Aurora, t=s)")
    ax21.set_title("Stations – GT")
    ax22.set_title("Stations – Prediction")
    fig.suptitle(title, fontsize=14)
    
    with io.BytesIO() as buf:
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frame_as_array = imageio.v2.imread(buf)
        
    return frame_as_array

# --- NEW: GRID_INFO에서 Lambert Conformal 파라미터 읽기 -----------------------
def _get_first(ds, keys, default=None):
    """Dataset 변수 또는 전역 속성에서 첫 번째로 발견되는 키의 스칼라 값을 반환."""
    for k in keys:
        if k in ds:                # 변수
            try:    return float(ds[k].values)
            except: pass
        if k in ds.attrs:          # 전역 속성
            try:    return float(ds.attrs[k])
            except: pass
    return default

def _wrap_lon_180(lon2d: np.ndarray) -> np.ndarray:
    """Ensure longitudes are in [-180, 180] for PlateCarree."""
    lon = np.asarray(lon2d, dtype=np.float64)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    return lon

def build_lcc_from_gridinfo(grid_info_path: str) -> ccrs.LambertConformal:
    """
    GRID_INFO_27km.nc에서 LCC 파라미터를 읽어 Cartopy LambertConformal을 생성.
    필수 항목이 없으면 LAT/LON로부터 안전한 기본값을 유도.
    """
    with xr.open_dataset(grid_info_path) as ds:
        # 기본 중심 경·위도는 격자 평균으로 안전하게 추정
        try:
            _lat = ds["LAT"].values
            _lon = ds["LON"].values
            cen_lat_fallback = float(np.nanmean(_lat))
            cen_lon_fallback = float(np.nanmean(_lon))
            # 표준위도 기본값은 위도 분포의 사분위 수로 설정
            q25, q75 = np.nanpercentile(_lat, [25, 75])
            std1_fallback, std2_fallback = float(q25), float(q75)
        except Exception:
            cen_lat_fallback, cen_lon_fallback = 35.0, 125.0
            std1_fallback, std2_fallback     = 25.0, 47.0

        stdlat1 = _get_first(ds, ["TRUELAT1", "STDLAT1"], default=std1_fallback)
        stdlat2 = _get_first(ds, ["TRUELAT2", "STDLAT2"], default=std2_fallback)
        if stdlat2 is None:  # 단일 표준위도 케이스
            stdlat2 = stdlat1

        central_longitude = _get_first(ds, ["STAND_LON", "CENT_LON"], default=cen_lon_fallback)
        central_latitude  = _get_first(ds, ["CENT_LAT", "MOAD_CEN_LAT"], default=cen_lat_fallback)

    return ccrs.LambertConformal(
        central_longitude=central_longitude,
        central_latitude=central_latitude,
        standard_parallels=(stdlat1, stdlat2),
    )

def flip_to_match_lat(grid: np.ndarray, lat2d: np.ndarray) -> np.ndarray:
    """
    lat2d의 0행이 남쪽(평균 위도가 더 작음)이라면,
    imshow 기준으로는 origin='lower'가 깔끔.
    pcolormesh에서 grid와 lat2d의 행이 '서로 반대'라면 grid만 뒤집어 맞춥니다.
    """
    # lat2d가 북→남(맨 위가 북) 이면서 grid가 남→북이라면 grid 뒤집기
    top_lat    = np.nanmean(lat2d[0, :])
    bottom_lat = np.nanmean(lat2d[-1,:])
    # 관례적으로: top_lat > bottom_lat  → 맨 위가 북쪽
    # 만약 예측 grid가 lat2d와 반대라면 grid[::-1, :]로 뒤집어 정렬
    # (반대 여부는 데이터 사양에 따라 다르니, 필요 시 토글로 사용)
    return grid[::-1, :]

def shape_for_degree_grid(lats2d, lons2d, dlat=0.25, dlon=0.25):
    lon_min, lon_max = float(np.nanmin(lons2d)), float(np.nanmax(lons2d))
    lat_min, lat_max = float(np.nanmin(lats2d)), float(np.nanmax(lats2d))
    H = int(round((lat_max - lat_min) / dlat)) + 1
    W = int(round((lon_max - lon_min) / dlon)) + 1
    return H, W

def make_panel_obs_vs_aurora(
    left_title: str,
    right_title: str,
    left_gt_grid : np.ndarray,         # (H,W) – OBS-on-grid for top scatter colors
    right_gt_grid: np.ndarray,         # (H,W)
    left_pred_grid: np.ndarray,        # (H,W)
    right_pred_grid: np.ndarray,       # (H,W)
    left_bg_grid  : np.ndarray,        # (H,W)
    right_bg_grid : np.ndarray,        # (H,W)
    cfg: dict,
    suptitle: str,
    lons_2d: np.ndarray,               # (H,W) source lon centers (curvilinear)
    lats_2d: np.ndarray,               # (H,W) source lat centers (curvilinear)
    plot_extent: list[float],          # [W, E, S, N]
    dpi: int = 220,
    flip_y: bool = False,              # keep False for curvilinear grids
    save_png_to: str | Path | None = None,
    rectify_to_regular: bool = False,  # NEW: make rectangular by regridding
    rect_shape: tuple[int, int] | None = None,  # e.g., (300, 400). Default: source shape
    target_res_deg: float | None = 0.25,
) -> np.ndarray:
    """
    Build a 2×2 panel. If 'rectify_to_regular' is True, background and prediction are
    interpolated to a regular lon/lat grid (rectangular cells) before plotting.
    """
    # ----- color config -----
    bounds = cfg["bounds"]; unit = cfg["unit"]
    vmin, vmax = bounds[0], bounds[-1]
    # segmented gradient colormap
    color_pairs = [['#2795EF','#1976D2'], ['#8FDB91','#248929'], ['#EBE082','#F3B517'], ['#F88967','#D81E1E']]
    cols=[]; n_total=256
    for i in range(len(bounds)-1):
        s = np.array(plt.cm.colors.to_rgba(color_pairs[i][0] if i < len(color_pairs) else '#FFFFFF'))
        e = np.array(plt.cm.colors.to_rgba(color_pairs[i][1] if i < len(color_pairs) else '#000000'))
        n_seg = max(1, int(round((bounds[i+1]-bounds[i])/(vmax-vmin)*n_total)))
        for t in np.linspace(0,1,n_seg): cols.append(tuple(s*(1-t)+e*t))
    cmap = ListedColormap(cols); norm = Normalize(vmin=vmin, vmax=vmax); ticks=bounds

    # ----- (optional) vertical flip of values only -----
    if flip_y:
        left_gt_grid    = np.flipud(left_gt_grid)
        right_gt_grid   = np.flipud(right_gt_grid)
        left_pred_grid  = np.flipud(left_pred_grid)
        right_pred_grid = np.flipud(right_pred_grid)
        left_bg_grid    = np.flipud(left_bg_grid)
        right_bg_grid   = np.flipud(right_bg_grid)

    # ----- prepare lon/lat for plotting -----
    lon_c = _wrap_lon_180(lons_2d); lat_c = lats_2d
    Hs, Ws = lat_c.shape
    Hr, Wr = rect_shape if rect_shape is not None else (Hs, Ws)

    if rectify_to_regular:
        west,east,south,north = plot_extent
        if target_res_deg is not None and rect_shape is None:
            Wr = int(round((east - west) / target_res_deg)) + 1
            Hr = int(round((north - south) / target_res_deg)) + 1
        else:
            Hr, Wr = rect_shape if rect_shape is not None else (Hs, Ws)

        lon_rng = (west, east)    # ← 반드시 지정
        lat_rng = (south, north)

        lonL, latL, left_bg_grid   = regrid_to_latlon(left_bg_grid,   lat_c, lon_c, Hr, Wr, lon_rng, lat_rng)
        _,    _,    left_pred_grid = regrid_to_latlon(left_pred_grid, lat_c, lon_c, Hr, Wr, lon_rng, lat_rng)
        lonR, latR, right_bg_grid  = regrid_to_latlon(right_bg_grid,  lat_c, lon_c, Hr, Wr, lon_rng, lat_rng)
        _,    _,    right_pred_grid= regrid_to_latlon(right_pred_grid,lat_c, lon_c, Hr, Wr, lon_rng, lat_rng)
    else:
        lonL, latL = lon_c, lat_c
        lonR, latR = lon_c, lat_c

    # valid pixels for top-row scatter (from GT mask)
    L_valid = left_gt_grid  > 0
    R_valid = right_gt_grid > 0
    L_obs_lon, L_obs_lat, L_obs_val = lon_c[L_valid], lat_c[L_valid], left_gt_grid[L_valid]
    R_obs_lon, R_obs_lat, R_obs_val = lon_c[R_valid], lat_c[R_valid], right_gt_grid[R_valid]

    # ----- figure/layout -----
    fig = plt.figure(figsize=(12, 7), dpi=dpi)
    gs  = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05], height_ratios=[1, 1], wspace=0.25, hspace=0.22)
    ax_obs_L  = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax_obs_R  = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    ax_pred_L = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    ax_pred_R = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
    c_ax      = fig.add_subplot(gs[:, 2])

    def _style(ax, scale='10m'):  # '10m' | '50m' | '110m'
        ax.set_extent(plot_extent, crs=ccrs.PlateCarree())

        # 고해상도 네추럴어스 피처 구성
        land    = cfeature.LAND.with_scale(scale)
        ocean   = cfeature.OCEAN.with_scale(scale)
        coast   = cfeature.COASTLINE.with_scale(scale)
        borders = cfeature.BORDERS.with_scale(scale)
        lakes   = cfeature.LAKES.with_scale(scale)

        # 배경 채우기
        ax.add_feature(land,  facecolor='#f2f2f2', edgecolor='none', zorder=0)
        ax.add_feature(ocean, facecolor='#e9f2fb', edgecolor='none', zorder=0)
        ax.add_feature(lakes, facecolor='#e9f2fb', edgecolor='none', zorder=1)

        # 경계선/해안선
        ax.add_feature(coast,   linewidth=0.8, edgecolor='black', zorder=6)
        ax.add_feature(borders, linewidth=0.6, edgecolor='black', linestyle=':', zorder=6)

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                        linewidth=0.5, color='gray', alpha=0.5, linestyle='--', zorder=6)
        gl.top_labels = False; gl.right_labels = False
        gl.xlabel_style = {'size': 8}; gl.ylabel_style = {'size': 8}

    for ax in (ax_obs_L, ax_obs_R, ax_pred_L, ax_pred_R): _style(ax, scale='10m')

    # ----- row 1: “stations” (scatter from valid GT pixels) -----
    ax_obs_L.scatter(L_obs_lon, L_obs_lat, c=L_obs_val, cmap=cmap, norm=norm,
                     s=10, edgecolor='gray', linewidth=0.3,
                     transform=ccrs.PlateCarree(), zorder=7)
    ax_obs_R.scatter(R_obs_lon, R_obs_lat, c=R_obs_val, cmap=cmap, norm=norm,
                     s=10, edgecolor='gray', linewidth=0.3,
                     transform=ccrs.PlateCarree(), zorder=7)

    # ----- row 2: background + prediction on a rectangular grid -----
    ax_pred_L.pcolormesh(lonL, latL, left_bg_grid,  cmap=cmap, norm=norm,
                         shading='auto', transform=ccrs.PlateCarree(), zorder=1, rasterized=True)
    imL = ax_pred_L.pcolormesh(lonL, latL, left_pred_grid, cmap=cmap, norm=norm,
                               shading='auto', transform=ccrs.PlateCarree(), zorder=3, rasterized=True)
    ax_pred_R.pcolormesh(lonR, latR, right_bg_grid, cmap=cmap, norm=norm,
                         shading='auto', transform=ccrs.PlateCarree(), zorder=1, rasterized=True)
    imR = ax_pred_R.pcolormesh(lonR, latR, right_pred_grid, cmap=cmap, norm=norm,
                               shading='auto', transform=ccrs.PlateCarree(), zorder=3, rasterized=True)

    cb = fig.colorbar(imR, cax=c_ax, ticks=ticks); cb.set_label(unit, fontsize=9)
    ax_obs_L.set_title(f"{left_title}  (Stations from GT)")
    ax_obs_R.set_title(f"{right_title} (Stations from GT)")
    ax_pred_L.set_title(f"{left_title}  – Background + Prediction")
    ax_pred_R.set_title(f"{right_title} – Background + Prediction")
    if suptitle: fig.suptitle(suptitle, fontsize=13)

    if save_png_to is not None:
        fig.savefig(save_png_to, dpi=dpi, bbox_inches="tight")

    with io.BytesIO() as buf:
        fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig); buf.seek(0)
        return imageio.v2.imread(buf)


# ─────────────────────────────────────────────────────────────
# make_panel_obs_vs_aurora 호출을 위한 유틸
# ─────────────────────────────────────────────────────────────
def _wrap_lon_180(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return np.where(a > 180.0, a - 360.0, a)

def _resize_to(arr: np.ndarray, H: int, W: int) -> np.ndarray:
    """(h,w) array를 (H,W)로 bilinear 리샘플."""
    if arr.shape == (H, W):
        return arr
    zh, zw = H / arr.shape[0], W / arr.shape[1]
    return zoom(arr, (zh, zw), order=1)

# ─────────────────────────────────────────────────────────────
# OBS×Pred 2×2 패널을 월 단위로 GIF/PNG 생성
# ─────────────────────────────────────────────────────────────
def write_rollout_gifs_with_obs(
        init_batch: "Batch",
        preds: list["Batch"],
        npz_root: str,
        vis_root: Path,
        cmaq_root: str,
        grid_info_path: str,
        tb_writer: "SummaryWriter | None" = None,
        lead_hours_to_show: list[int] = [12, 24, 48, 72, 96, 120],
        save_png_also: bool = True,
        png_every: int = 1,
        flip_y: bool = True,
        plot_extent: list[float] = [100, 155, 20, 52],
        dpi: int = 220,
    ) -> None:
    """
    Create one 2x2 panel per pair of lead hours supplied in `lead_hours_to_show`.
    If the list length is odd, the last item will be paired with itself.
    """
    t0 = init_batch.metadata.time[0]
    vis_root.mkdir(parents=True, exist_ok=True)
    logging.info(f"[vis] creating OBS vs Aurora panels for {t0:%Y-%m-%d} …")

    # --- load grid lat/lon once ---
    try:
        gi = xr.open_dataset(grid_info_path)
        latv = gi["LAT"].values
        lonv = gi["LON"].values
        lats_2d_full = latv[0, 0, :, :] if latv.ndim == 4 else np.squeeze(latv)
        lons_2d_full = lonv[0, 0, :, :] if lonv.ndim == 4 else np.squeeze(lonv)
        gi.close()
    except Exception as e:
        logging.error(f"[vis] GRID_INFO read failed: {e}")
        return

    # Normalize longitudes to [-180, 180] for PlateCarree
    lons_2d_full = _wrap_lon_180(lons_2d_full)
    Hc, Wc = lats_2d_full.shape
    rect_H, rect_W = shape_for_degree_grid(lats_2d_full, lons_2d_full, dlat=0.3, dlon=0.3)


    # --- build (index, hour) pairs from lead_hours_to_show ---
    def _hour_to_step(h: int) -> int:
        # model predicts every 6 h, step index is (h/6) - 1
        return (h // 6) - 1

    pairs: list[tuple[tuple[int,int], tuple[int,int]]] = []
    hours = list(lead_hours_to_show)
    if len(hours) == 0:
        logging.warning("[vis] no lead hours given")
        return

    # group into pairs: (h[i], h[i+1]); last odd -> pair with itself
    i = 0
    while i < len(hours):
        hL = hours[i]
        kL = _hour_to_step(hL)
        if not (0 <= kL < len(preds)):
            logging.warning(f"[vis] skip hour {hL}h (step idx {kL} out of range)")
            i += 1
            continue

        if i + 1 < len(hours):
            hR = hours[i+1]
            kR = _hour_to_step(hR)
            if not (0 <= kR < len(preds)):
                logging.warning(f"[vis] skip hour {hR}h (step idx {kR} out of range); pairing {hL}h with itself")
                kR, hR = kL, hL
        else:
            # odd length: pair last with itself
            hR, kR = hL, kL

        pairs.append(((kL, hL), (kR, hR)))
        i += 2

    if not pairs:
        logging.warning(f"[vis] no valid lead hours in {lead_hours_to_show}")
        return

    # --- iterate variables and all pairs; make a panel per pair ---
    for var, cfg in VIS_VARS.items():
        if var.upper() != "PM2P5":
            continue
        for ((kL, hL), (kR, hR)) in pairs:
            # times for CMAQ/OBS loading
            tsL = preds[kL].metadata.time[0]
            tsR = preds[kR].metadata.time[0]

            # background for both times
            cams_seq = load_cmaq_for_rollout((tsL, tsR), var, cmaq_root, target_h=Hc, target_w=Wc)
            if cams_seq is None or cams_seq.shape[0] != 2:
                logging.warning(f"[vis] CAMS not available for {var} @ {t0:%Y-%m-%d} / +{hL}h,+{hR}h")
                continue
            bgL = cams_seq[0].cpu().numpy()
            bgR = cams_seq[1].cpu().numpy()

            # GT (OBS on grid)
            try:
                gtL = _load_obs(var, tsL, npz_root)  # (Hc, Wc)
                gtR = _load_obs(var, tsR, npz_root)
            except Exception as e:
                logging.warning(f"[vis] OBS npz missing for {var}: {e}")
                continue

            # predictions → resize to CAMS grid if needed
            try:
                predL = preds[kL].surf_vars[var][0, 0].detach().cpu().numpy()
                predR = preds[kR].surf_vars[var][0, 0].detach().cpu().numpy()
            except KeyError:
                logging.warning(f"[vis] variable '{var}' not found in prediction tensors; skip.")
                continue

            if predL.shape != (Hc, Wc):
                predL = _resize_to(predL, Hc, Wc)
            if predR.shape != (Hc, Wc):
                predR = _resize_to(predR, Hc, Wc)

            # build one panel for the current pair
            frame = make_panel_obs_vs_aurora(
                left_title      = f"+{hL}h",
                right_title     = f"+{hR}h",
                left_gt_grid    = gtL,
                right_gt_grid   = gtR,
                left_pred_grid  = predL,
                right_pred_grid = predR,
                left_bg_grid    = bgL,
                right_bg_grid   = bgR,
                cfg             = cfg,
                suptitle        = f"{var.upper()}  {t0:%Y-%m-%d %HZ}",
                lons_2d         = lons_2d_full,
                lats_2d         = lats_2d_full,
                plot_extent     = plot_extent,
                dpi             = dpi,
                flip_y          = True,
                rectify_to_regular = True,     # keep rectangular output
                target_res_deg=0.25,
            )

            # save a PNG per pair (+hLh +hRh)
            out_png = vis_root / f"{var}_{t0:%Y%m%d%HZ}_panel_{hL}h_{hR}h.png"
            Image.fromarray(frame).save(out_png)
            logging.info(f"[vis] saved {out_png}")

            # optional: also push to TensorBoard with the hours in the tag
            if tb_writer is not None:
                tag_root = f"Panel/OBS_vs_Aurora/{var.upper()}/{t0:%Y%m%d}/+{hL}h_{hR}h"
                img = torch.from_numpy(frame[..., :3]).permute(2, 0, 1)
                tb_writer.add_image(tag_root, img, global_step=0)


# ---------- dataloader helper ----------
def get_test_dataloader(batch, data_dir, npz_path, test_start_date, test_end_date, distributed, cmaq_root, flow_root, sources, horizon, use_wind_prompt=False, use_cutmix=False, use_cmaq_pm_only=False, use_hybrid_target=False, use_flow=False,):
    """Create test dataloader for inference."""
    test_dataset = WeatherDataset(test_start_date, test_end_date, data_dir, npz_path, cmaq_root=cmaq_root, flow_root=flow_root, sources=sources, horizon=horizon, use_wind_prompt=use_wind_prompt, use_cutmix=use_cutmix, use_cmaq_pm_only=use_cmaq_pm_only, use_hybrid_target=use_hybrid_target, use_flow=use_flow)
    
    if distributed:
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch, sampler=test_sampler, collate_fn=aurora_collate_fn)
    else:
        test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False, collate_fn=aurora_collate_fn)
    
    return test_dataloader


def load_checkpoint(model, checkpoint_path, device):
    """Load model checkpoint with DDP prefix handling."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', 0.0)
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', 0.0)
    else:
        # Assume the checkpoint is just the state dict
        state_dict = checkpoint
        epoch = 0
        loss = 0.0
    
    # Remove "module." prefix if present (from DDP training)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove "module." prefix (7 characters)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    
    # Load the cleaned state dict
    try:
        model.load_state_dict(new_state_dict, strict=False)
        logging.info(f"Checkpoint loaded successfully (epoch: {epoch}, loss: {loss:.4f})")
    except Exception as e:
        logging.error(f"Error loading state dict: {e}")
        # Try with strict=False if there are still issues
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        if missing_keys:
            logging.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys: {unexpected_keys}")
    
    return model, epoch


def get_month_from_date_string(date_str):
    """Extract month from date string in format YYYY-MM-DD."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").month
    except:
        return None


def inference_and_evaluate(test_dataloader, model, writer=None, save_dir=None, test_start_date=None):
    """
    Run inference on test dataset and compute metrics for both PM2.5 and PM10.
    Returns overall metrics and monthly breakdown for both pollutants.
    """
    model.eval()
    
    # Store predictions and targets for overall metrics - separate for each pollutant
    all_data = {
        'PM2.5': {'pred': [], 'target': []},
        'PM10': {'pred': [], 'target': []}
    }
    
    # Store monthly data for analysis
    monthly_data = {
        'PM2.5': {},
        'PM10': {}
    }
    
    # Store visualization data
    viz_samples = []
    
    logging.info("Starting inference on test dataset for PM2.5 and PM10...")
    
    # Create progress bar
    try:
        total_batches = len(test_dataloader)
    except:
        total_batches = None
    
    if total_batches:
        pbar = tqdm(test_dataloader, total=total_batches, desc="Standard Inference", unit="batch", ncols=100)
    else:
        pbar = tqdm(test_dataloader, desc="Standard Inference", unit="batch", ncols=100)
    
    processed_samples = 0
    
    with torch.no_grad():
        for i, batch in enumerate(pbar):

            if batch is None or batch[0] is None:
                continue

            input, target = batch
            input = input.to("cuda")
            target = target.to("cuda")
            
            # Forward pass
            pred = model.forward(input)
            
            # Process each sample in the batch
            B = target.surf_vars["pm2p5"].shape[0]
            for b in range(B):
                # Process PM2.5
                pm2p5_mask = (target.surf_vars["pm2p5"][b, 0] > 0)
                if torch.count_nonzero(pm2p5_mask) > 0:
                    target_pm2p5 = target.surf_vars["pm2p5"][b, 0][pm2p5_mask].detach().cpu().numpy()
                    
                    # --- START OF CORRECTION for Standard Mode ---
                    # pred_norm_pm2p5   = pred.surf_vars["pm2p5"][b, 0]
                    # pred_unnorm_pm2p5 = unnormalise_surf_var(
                    #     pred_norm_pm2p5.unsqueeze(0).unsqueeze(0), 'pm2p5'
                    # ).squeeze()
                    # pred_pm2p5 = pred_unnorm_pm2p5[pm2p5_mask].detach().cpu().numpy()
                    pred_pm2p5 = pred.surf_vars["pm2p5"][b, 0][pm2p5_mask].detach().cpu().numpy()
                    # --- END OF CORRECTION for Standard Mode ---
                    
                    all_data['PM2.5']['target'].append(target_pm2p5)
                    all_data['PM2.5']['pred'].append(pred_pm2p5)
                
                # Process PM10
                pm10_mask  = (target.surf_vars["pm10"][b, 0]  > 0)
                if torch.count_nonzero(pm10_mask) > 0:
                    target_pm10 = target.surf_vars["pm10"][b, 0][pm10_mask].detach().cpu().numpy()
                    
                    # --- START OF CORRECTION for Standard Mode (PM10) ---
                    # pred_norm_pm10   = pred.surf_vars["pm10"][b, 0]
                    # pred_unnorm_pm10 = unnormalise_surf_var(
                    #     pred_norm_pm10.unsqueeze(0).unsqueeze(0), 'pm10'
                    # ).squeeze()
                    # pred_pm10 = pred_unnorm_pm10[pm10_mask].detach().cpu().numpy()
                    pred_pm10 =pred.surf_vars["pm10"][b, 0][pm10_mask].detach().cpu().numpy()
                    # --- END OF CORRECTION for Standard Mode (PM10) ---

                    all_data['PM10']['target'].append(target_pm10)
                    all_data['PM10']['pred'].append(pred_pm10)
                
                # Monthly data collection
                try:
                    if test_start_date:
                        sample_date = datetime.strptime(test_start_date, "%Y-%m-%d") + timedelta(days=i + b)
                        month = sample_date.month
                        month_name = calendar.month_name[month]
                        
                        # PM2.5 monthly data
                        if torch.count_nonzero(pm2p5_mask) > 0:
                            if month_name not in monthly_data['PM2.5']:
                                monthly_data['PM2.5'][month_name] = {
                                    'predictions': [], 'targets': [], 'month_num': month
                                }
                            monthly_data['PM2.5'][month_name]['predictions'].append(pred_pm2p5)
                            monthly_data['PM2.5'][month_name]['targets'].append(target_pm2p5)
                        
                        # PM10 monthly data
                        if torch.count_nonzero(pm10_mask) > 0:
                            if month_name not in monthly_data['PM10']:
                                monthly_data['PM10'][month_name] = {
                                    'predictions': [], 'targets': [], 'month_num': month
                                }
                            monthly_data['PM10'][month_name]['predictions'].append(pred_pm10)
                            monthly_data['PM10'][month_name]['targets'].append(target_pm10)
                except:
                    pass
            
            # Save first few samples for visualization
            if len(viz_samples) < 5:
                pm2p5_mask = (target.surf_vars["pm2p5"][0, 0] != 0)
                pm10_mask = (target.surf_vars["pm10"][0, 0] != 0)
                viz_samples.append({
                    'input_pm2p5': input.surf_vars["pm2p5"][0, 0].detach().cpu().numpy(),
                    'pred_pm2p5': pred.surf_vars["pm2p5"][0, 0].detach().cpu().numpy(),
                    'target_pm2p5': target.surf_vars["pm2p5"][0, 0].detach().cpu().numpy(),
                    'mask_pm2p5': pm2p5_mask.cpu().numpy(),
                    'input_pm10': input.surf_vars["pm10"][0, 0].detach().cpu().numpy(),
                    'pred_pm10': pred.surf_vars["pm10"][0, 0].detach().cpu().numpy(),
                    'target_pm10': target.surf_vars["pm10"][0, 0].detach().cpu().numpy(),
                    'mask_pm10': pm10_mask.cpu().numpy()
                })
            
            processed_samples += 1
            
            # Update progress bar
            if processed_samples % 50 == 0:
                pbar.set_postfix({'Samples': processed_samples})
    
    pbar.close()
    
    # Compute overall metrics for both pollutants
    overall_metrics = {}
    # --- NEW: 4‑class containers ------------------------------------
    overall_metrics_4cls = {}                    
    monthly_metrics_4cls = {'PM2.5': {}, 'PM10': {}}
    
    for pollutant in ['PM2.5', 'PM10']:
        if all_data[pollutant]['target'] and all_data[pollutant]['pred']:
            all_target = np.concatenate(all_data[pollutant]['target'])
            all_pred = np.concatenate(all_data[pollutant]['pred'])
            
            threshold = 35 if pollutant == 'PM2.5' else 80
            overall_metrics[pollutant] = compute_metrics(all_pred, all_target, pollutant, threshold)
            overall_metrics[pollutant] = attach_skill_scores(overall_metrics[pollutant])

            # --- NEW: 4‑class overall --------------------------------
            overall_metrics_4cls[pollutant] = compute_multiclass_metrics(
                all_pred, all_target, pollutant_name=pollutant
            )
            # -----------------------------------------------------------
            
            logging.info(f"Overall {pollutant} Test Metrics:")
            for key, value in overall_metrics[pollutant].items():
                if key not in ['confusion_matrix', 'confusion_matrix_raw', 'pollutant', 'threshold']:
                    logging.info(f"  {key}: {value}")
    
    # Compute monthly metrics for both pollutants
    monthly_metrics = {'PM2.5': {}, 'PM10': {}}
    
    print("\nComputing monthly metrics...")
    
    for pollutant in ['PM2.5', 'PM10']:
        threshold = 35 if pollutant == 'PM2.5' else 80
        
        for month_name, data in tqdm(monthly_data[pollutant].items(), 
                                   desc=f"{pollutant} Monthly Metrics", unit="month"):
            if data['predictions'] and data['targets']:
                month_pred = np.concatenate(data['predictions'])
                month_target = np.concatenate(data['targets'])
                monthly_metrics[pollutant][month_name] = compute_metrics(month_pred, month_target, pollutant, threshold)
                monthly_metrics[pollutant][month_name] = attach_skill_scores(monthly_metrics[pollutant][month_name])
                monthly_metrics[pollutant][month_name]['month_num'] = data['month_num']

                 # --- NEW: 4‑class monthly ------------------------------
                monthly_metrics_4cls[pollutant][month_name] = compute_multiclass_metrics(
                    month_pred, month_target, pollutant_name=pollutant
                )
                monthly_metrics_4cls[pollutant][month_name]['month_num'] = data['month_num']
                # -------------------------------------------------------
                
                logging.info(f"{month_name} {pollutant} Metrics:")
                for key, value in monthly_metrics[pollutant][month_name].items():
                    if key not in ['confusion_matrix', 'confusion_matrix_raw', 'pollutant', 'threshold', 'month_num']:
                        logging.info(f"  {key}: {value}")
    
    # Log to TensorBoard if writer is provided
    if writer is not None:
        log_metrics_to_tensorboard(writer, overall_metrics, monthly_metrics, viz_samples,
                                   overall_metrics_4cls, monthly_metrics_4cls)
    
    # Save results if save_dir is provided
    if save_dir is not None:
        save_test_results(save_dir, overall_metrics, monthly_metrics, viz_samples,
                          overall_metrics_4cls, monthly_metrics_4cls) 
    
    return overall_metrics, monthly_metrics, overall_metrics_4cls, monthly_metrics_4cls


def log_metrics_to_tensorboard(writer, overall_metrics, monthly_metrics, viz_samples,
                               overall_metrics_4cls=None, monthly_metrics_4cls=None):
    """Log metrics for both PM2.5 and PM10 to TensorBoard."""
    
    # Log overall metrics for both pollutants
    for pollutant in ['PM2.5', 'PM10']:
        if pollutant in overall_metrics:
            metrics = overall_metrics[pollutant]
            writer.add_scalar(f"Test/{pollutant}_Overall_Accuracy", metrics["accuracy"], 0)
            writer.add_scalar(f"Test/{pollutant}_Overall_F1", metrics["f1"], 0)
            writer.add_scalar(f"Test/{pollutant}_Overall_Precision", metrics["precision"], 0)
            writer.add_scalar(f"Test/{pollutant}_Overall_Recall", metrics["recall"], 0)
            writer.add_scalar(f"Test/{pollutant}_Overall_False_Alarm_Rate", metrics["false_alarm_rate"], 0)
            writer.add_scalar(f"Test/{pollutant}_Overall_Detection_Rate", metrics["detection_rate"], 0)
            writer.add_scalar(f"Test/{pollutant}_Overall_CSI", metrics.get("csi", 0.0), 0)
            writer.add_scalar(f"Test/{pollutant}_Overall_ETS", metrics.get("ets", 0.0), 0)
            writer.add_scalar(f"Test/{pollutant}_Overall_TSS", metrics.get("tss", 0.0), 0)
    
    # Log monthly metrics for both pollutants
    for pollutant in ['PM2.5', 'PM10']:
        if pollutant in monthly_metrics:
            for month_name, metrics in monthly_metrics[pollutant].items():
                month_num = metrics['month_num']
                writer.add_scalar(f"Test_Monthly/{pollutant}_Accuracy", metrics["accuracy"], month_num)
                writer.add_scalar(f"Test_Monthly/{pollutant}_F1", metrics["f1"], month_num)
                writer.add_scalar(f"Test_Monthly/{pollutant}_Precision", metrics["precision"], month_num)
                writer.add_scalar(f"Test_Monthly/{pollutant}_Recall", metrics["recall"], month_num)
                writer.add_scalar(f"Test_Monthly/{pollutant}_False_Alarm_Rate", metrics["false_alarm_rate"], month_num)
                writer.add_scalar(f"Test_Monthly/{pollutant}_Detection_Rate", metrics["detection_rate"], month_num)
                writer.add_scalar(f"Test_Monthly/{pollutant}_CSI", metrics.get("csi", 0.0), month_num)
                writer.add_scalar(f"Test_Monthly/{pollutant}_ETS", metrics.get("ets", 0.0), month_num)
                writer.add_scalar(f"Test_Monthly/{pollutant}_TSS", metrics.get("tss", 0.0), month_num)
    
    # Create confusion matrix visualizations for both pollutants
    for pollutant in ['PM2.5', 'PM10']:
        if pollutant in overall_metrics:
            create_confusion_matrix_plot(writer, overall_metrics[pollutant]["confusion_matrix_raw"], 
                                       f"Test/{pollutant}_Overall_Confusion_Matrix", 0)
    
    # Create sample visualizations for both pollutants
    for i, sample in enumerate(viz_samples):
        create_dual_pollutant_visualization(writer, sample, f"Test/Sample_{i+1}", i)
    
    if overall_metrics_4cls:
        for pollutant, m4 in overall_metrics_4cls.items():
            writer.add_scalar(f"Test/{pollutant}_4C/Overall_Accuracy",    m4["overall"]["accuracy"], 0)
            writer.add_scalar(f"Test/{pollutant}_4C/Overall_F1_macro",    m4["overall"]["f1_macro"], 0)
            writer.add_scalar(f"Test/{pollutant}_4C/Overall_F1_weighted", m4["overall"]["f1_weighted"], 0)
            writer.add_scalar(f"Test/{pollutant}_4C/Overall_F1_micro",    m4["overall"]["f1_micro"], 0)
            # per-class
            for cname, cm in m4["per_class"].items():
                base = f"Test/{pollutant}_4C/PerClass/{cname}"
                writer.add_scalar(base + "/F1",     cm["f1"],     0)
                writer.add_scalar(base + "/Recall", cm["recall"], 0)
                writer.add_scalar(base + "/Precision", cm["precision"], 0)
                writer.add_scalar(base + "/FAR",    cm["far"],    0)
                writer.add_scalar(base + "/Acc_OVR", cm["acc_ovr"], 0)

    # --- NEW: 4‑class monthly ----------------------------------------
    if monthly_metrics_4cls:
        for pollutant, mon_dict in monthly_metrics_4cls.items():
            for month_name, m4 in mon_dict.items():
                month_num = m4['month_num']
                writer.add_scalar(f"Test_Monthly/{pollutant}_4C/Accuracy",    m4["overall"]["accuracy"],    month_num)
                writer.add_scalar(f"Test_Monthly/{pollutant}_4C/F1_macro",    m4["overall"]["f1_macro"],    month_num)
                writer.add_scalar(f"Test_Monthly/{pollutant}_4C/F1_weighted", m4["overall"]["f1_weighted"], month_num)
                writer.add_scalar(f"Test_Monthly/{pollutant}_4C/F1_micro",    m4["overall"]["f1_micro"],    month_num)
                for cname, cm in m4["per_class"].items():
                    base = f"Test_Monthly/{pollutant}_4C/PerClass/{cname}"
                    writer.add_scalar(base + "/F1",       cm["f1"],     month_num)
                    writer.add_scalar(base + "/Recall",   cm["recall"], month_num)
                    writer.add_scalar(base + "/Precision",cm["precision"], month_num)
                    writer.add_scalar(base + "/FAR",      cm["far"],    month_num)
                    writer.add_scalar(base + "/Acc_OVR",  cm["acc_ovr"], month_num)


def create_confusion_matrix_plot(writer, cm, tag, step):
    """Create confusion matrix plot for TensorBoard."""
    # Normalize confusion matrix
    cm_norm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-6)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    classes_row = ['GT: Negative', 'GT: Positive']
    classes_col = ['Pred: Negative', 'Pred: Positive']
    ax.set(xticks=np.arange(len(classes_col)),
           yticks=np.arange(len(classes_row)),
           xticklabels=classes_col,
           yticklabels=classes_row,
           title=f'Confusion Matrix (Normalized)\nRaw counts: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = '.3f'
    thresh = cm_norm.max() / 2.
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f'{format(cm_norm[i, j], fmt)}\n({cm[i, j]})',
                   ha="center", va="center",
                   color="white" if cm_norm[i, j] > thresh else "black")
    
    fig.tight_layout()
    writer.add_figure(tag, fig, step)
    plt.close(fig)


def create_dual_pollutant_visualization(writer, sample, tag, step):
    """Create prediction vs ground truth visualization for both PM2.5 and PM10."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # PM2.5 visualizations (top row)
    im1 = axes[0, 0].imshow(sample['input_pm2p5'], vmin=-50, vmax=50, cmap="viridis")
    axes[0, 0].set_title("PM2.5 Input Observations")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    plt.colorbar(im1, ax=axes[0, 0])
    
    pred_pm2p5_filled = np.where(sample['mask_pm2p5'], sample['pred_pm2p5'], sample['target_pm2p5'])
    im2 = axes[0, 1].imshow(pred_pm2p5_filled, vmin=-50, vmax=50, cmap="viridis")
    axes[0, 1].set_title("PM2.5 Prediction")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].imshow(sample['target_pm2p5'], vmin=-50, vmax=50, cmap="viridis")
    axes[0, 2].set_title("PM2.5 Ground Truth")
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])
    plt.colorbar(im3, ax=axes[0, 2])
    
    # PM10 visualizations (bottom row)
    im4 = axes[1, 0].imshow(sample['input_pm10'], vmin=-50, vmax=100, cmap="plasma")
    axes[1, 0].set_title("PM10 Input Observations")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    plt.colorbar(im4, ax=axes[1, 0])
    
    pred_pm10_filled = np.where(sample['mask_pm10'], sample['pred_pm10'], sample['target_pm10'])
    im5 = axes[1, 1].imshow(pred_pm10_filled, vmin=-50, vmax=100, cmap="plasma")
    axes[1, 1].set_title("PM10 Prediction")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    plt.colorbar(im5, ax=axes[1, 1])
    
    im6 = axes[1, 2].imshow(sample['target_pm10'], vmin=-50, vmax=100, cmap="plasma")
    axes[1, 2].set_title("PM10 Ground Truth")
    axes[1, 2].set_xticks([])
    axes[1, 2].set_yticks([])
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    writer.add_figure(tag, fig, step)
    plt.close(fig)


def save_test_results(save_dir, overall_metrics, monthly_metrics, viz_samples,
                      overall_metrics_4cls=None, monthly_metrics_4cls=None):
    """Save test results for both PM2.5 and PM10 to files."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save overall metrics for both pollutants
    with open(os.path.join(save_dir, "overall_metrics_all_pollutants.txt"), "w") as f:
        f.write("Overall Test Metrics for PM2.5 and PM10:\n")
        f.write("=" * 50 + "\n")
        
        for pollutant in ['PM2.5', 'PM10']:
            if pollutant in overall_metrics:
                f.write(f"\n{pollutant} Metrics (Threshold: {overall_metrics[pollutant]['threshold']} μg/m³):\n")
                f.write("-" * 30 + "\n")
                for key, value in overall_metrics[pollutant].items():
                    if key not in ['confusion_matrix', 'confusion_matrix_raw', 'pollutant', 'threshold']:
                        f.write(f"  {key}: {value}\n")
                
                cm = overall_metrics[pollutant]['confusion_matrix_raw']
                f.write(f"  Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}\n")
    
    # Save monthly metrics for both pollutants
    with open(os.path.join(save_dir, "monthly_metrics_all_pollutants.txt"), "w") as f:
        f.write("Monthly Test Metrics for PM2.5 and PM10:\n")
        f.write("=" * 50 + "\n")
        
        for pollutant in ['PM2.5', 'PM10']:
            if pollutant in monthly_metrics:
                f.write(f"\n{pollutant} Monthly Metrics:\n")
                f.write("=" * 30 + "\n")
                
                for month_name, metrics in monthly_metrics[pollutant].items():
                    f.write(f"\n{month_name}:\n")
                    f.write("-" * 15 + "\n")
                    for key, value in metrics.items():
                        if key not in ['confusion_matrix', 'confusion_matrix_raw', 'pollutant', 'threshold', 'month_num']:
                            f.write(f"  {key}: {value}\n")
                    
                    cm = metrics['confusion_matrix_raw']
                    f.write(f"  Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}\n")
    
    # --- NEW: 4‑class overall 저장 -----------------------------------
    if overall_metrics_4cls:
        with open(os.path.join(save_dir, "overall_metrics_4class.txt"), "w") as f:
            f.write("Overall 4-Class Metrics (good / moderate / bad / very_bad)\n")
            f.write("=" * 60 + "\n")
            for pollutant, m4 in overall_metrics_4cls.items():
                f.write(f"\n[{pollutant}] bins (t1,t2,t3) = {m4['bins']}\n")
                f.write(f"  Accuracy : {m4['overall']['accuracy']}\n")
                f.write(f"  F1_macro: {m4['overall']['f1_macro']} | "
                        f"F1_weighted: {m4['overall']['f1_weighted']} | "
                        f"F1_micro: {m4['overall']['f1_micro']}\n")
                f.write("  Per-class:\n")
                for cname, cm in m4["per_class"].items():
                    f.write(f"    - {cname:<10} "
                            f"F1={cm['f1']}, Recall={cm['recall']}, "
                            f"Precision={cm['precision']}, FAR={cm['far']}, Acc_OVR={cm['acc_ovr']}, "
                            f"Support={cm['support']}\n")

    # --- NEW: 4‑class monthly 저장 -----------------------------------
    if monthly_metrics_4cls:
        with open(os.path.join(save_dir, "monthly_metrics_4class.txt"), "w") as f:
            f.write("Monthly 4-Class Metrics\n")
            f.write("=" * 60 + "\n")
            for pollutant, mon_dict in monthly_metrics_4cls.items():
                f.write(f"\n[{pollutant}]\n")
                for month_name, m4 in mon_dict.items():
                    f.write(f"\n{month_name} (bins={m4['bins']})\n")
                    f.write(f"  Accuracy : {m4['overall']['accuracy']}\n")
                    f.write(f"  F1_macro: {m4['overall']['f1_macro']} | "
                            f"F1_weighted: {m4['overall']['f1_weighted']} | "
                            f"F1_micro: {m4['overall']['f1_micro']}\n")
                    f.write("  Per-class:\n")
                    for cname, cm in m4["per_class"].items():
                        f.write(f"    - {cname:<10} "
                                f"F1={cm['f1']}, Recall={cm['recall']}, "
                                f"Precision={cm['precision']}, FAR={cm['far']}, Acc_OVR={cm['acc_ovr']}, "
                                f"Support={cm['support']}\n")


# Add rollout prediction function
def rollout_prediction(
    model: torch.nn.Module,
    initial_batch: Batch,
    target_hours: int = 72,
    step_hours:   int = 6,
    use_rollout_cutmix: bool = False,
    obs_path:     str = None,
    cmaq_root:    str = None,
) -> list[Batch]:
    """
    Perform a multi‑step roll‑out and return *all* one‑step predictions
    as a list whose length equals `target_hours // step_hours`.
    """
    core     = model.module if isinstance(model, DDP) else model
    n_steps  = target_hours // step_hours
    device   = next(core.parameters()).device

    with torch.no_grad():
        preds = list(aurora_rollout(core, initial_batch.to(device), steps=n_steps,
                                    use_rollout_cutmix=use_rollout_cutmix,
                                    obs_path=obs_path,
                                    cmaq_root=cmaq_root))
    return preds


# --------------------------------------------------------------------------- #
#  Roll‑out evaluation (multi‑step)                                           #
# --------------------------------------------------------------------------- #
def rollout_inference_and_evaluate(
        test_loader       : DataLoader,
        model,
        rollout_hours     : int        = 72,
        writer            : SummaryWriter | None = None,
        save_dir          : str | Path | None    = None,
        test_start_date   : str | None = None,
        npz_path          : str | None = None,
        checkpoint_path   : str | None = None,
        step_hours        : int        = 6,
        use_rollout_cutmix: bool = False,
        cmaq_root         : str | Path | None = None,
) -> tuple[dict, dict, dict, dict, dict]:
    """
    Args
    ----
    test_loader   : DataLoader that yields (x, y) where y has ≥ horizon timesteps
    model         : Aurora / AuroraAirPollution (DDP or general)
    rollout_hours : total hours to predict (k * step_hours)
    writer        : TensorBoard writer (optional)
    save_dir      : folder to dump txt (optional)
    Returns
    -------
    step_results            – dict[pollutant][step_key] -> metrics
    monthly_metrics         – dict[pollutant][month]    -> metrics
    overall_rollout_metrics – dict[pollutant]           -> metrics
    """
    device = next(model.parameters()).device
    model.eval()

    # ------------------------------------------------------------------ #
    # 1. bookkeeping
    # ------------------------------------------------------------------ #
    num_steps  = rollout_hours // step_hours
    step_keys  = [f"step_{i+1}_+{(i+1)*step_hours}h" for i in range(num_steps)]
    pollutants = ["PM2.5", "PM10"]

    step_metrics   = {p:{k:{"pred":[], "tgt":[]} for k in step_keys} for p in pollutants}
    monthly_buffer = {p:{} for p in pollutants}   # raw 
    # ─── NEW: abs‑change accumulator ───────────────────────────────────
    #   index k  ↔  Δ(step k  →  k+1)  ,  k ∈ [0 … num_steps‑2]
    change_stats = {
        p: dict(
            gt_sum   = np.zeros(num_steps-1, dtype=np.float64),
            pred_sum = np.zeros(num_steps-1, dtype=np.float64),
            count    = np.zeros(num_steps-1, dtype=np.int64),
        ) for p in pollutants
    }
    viz_samples    : list[dict] = []

    # ────────────────────────────────────────────────────────────────────
    total_batches = len(test_loader) if hasattr(test_loader, "__len__") else None
    pbar = tqdm(test_loader, total=total_batches, desc="Roll‑out", ncols=110)


    for batch_idx, batch in enumerate(pbar):
        if batch is None or batch[0] is None:
            continue

        x0, tgt = batch
        x0, tgt = x0.to(device), tgt.to(device)

        # ① roll‑out
        preds = rollout_prediction(model, x0,
                                   target_hours=rollout_hours,
                                   step_hours=step_hours,
                                   use_rollout_cutmix=use_rollout_cutmix,
                                   obs_path=npz_path,
                                   cmaq_root=cmaq_root)      # len == num_steps

        # (option) save GIF
        if npz_path and checkpoint_path:
            ckpt_tag = "_".join(Path(checkpoint_path).with_suffix("").parts[-2:])
            vis_root = Path("vis") / ckpt_tag
            # write_rollout_gifs_with_obs(x0, preds, npz_path, vis_root, cmaq_root, '/fsx/mid_term/aurora/docs/GRID_INFO_27km.nc', writer)

        B , H , W = tgt.surf_vars["pm2p5"].shape[0:3]
        tgt_max_T = tgt.surf_vars["pm2p5"].shape[1]            # GT horizon

        for b in range(B):
            for s, step_key in enumerate(step_keys):
                if s >= len(preds):                break
                if s >= tgt_max_T:                 continue 

                pred_b = preds[s]                  # Batch(T=1)
                # --------- PM2.5 -------------------------------------------------
                g25 = tgt.surf_vars["pm2p5"][b, s]
                # p25 = unnormalise_surf_var(
                #         pred_b.surf_vars["pm2p5"][b, 0].unsqueeze(0).unsqueeze(0), 'pm2p5'
                #     ).squeeze()
                p25 = pred_b.surf_vars["pm2p5"][b, 0] 
                mask25 = g25 != 0
                if mask25.any():
                    step_metrics["PM2.5"][step_key]["tgt"].append(g25[mask25].cpu().numpy())
                    step_metrics["PM2.5"][step_key]["pred"].append(p25[mask25].cpu().numpy())

                # --------- PM10 --------------------------------------------------
                g10 = tgt.surf_vars["pm10"][b, s]
                # p10 = unnormalise_surf_var(
                #         pred_b.surf_vars["pm10"][b, 0].unsqueeze(0).unsqueeze(0), 'pm10'
                #     ).squeeze()
                p10 = pred_b.surf_vars["pm10"][b, 0]
                mask10 = g10 != 0
                if mask10.any():
                    # --- START OF CORRECTION for Rollout Mode (PM10) ---
                    step_metrics["PM10"][step_key]["tgt"].append(g10[mask10].cpu().numpy())
                    step_metrics["PM10"][step_key]["pred"].append(p10[mask10].cpu().numpy())
                    # --- END OF CORRECTION for Rollout Mode (PM10) ---

                # --------- monthly buffer ---------------------------------------------
                if test_start_date:
                    init_dt  = datetime.strptime(test_start_date, "%Y-%m-%d")
                    cur_time = preds[s].metadata.time[0]
                    mon_name = calendar.month_name[cur_time.month]
                    for p_lbl, _mask in [("PM2.5", mask25), ("PM10", mask10)]:
                        if _mask.any():
                            mbuf = monthly_buffer[p_lbl].setdefault(mon_name, {"pred": [], "tgt": [], "mon": cur_time.month})
                            if p_lbl == "PM2.5":
                                mbuf["tgt"].append(g25[_mask].detach().cpu().numpy())
                                mbuf["pred"].append(p25[_mask].detach().cpu().numpy())
                            else:
                                mbuf["tgt"].append(g10[_mask].detach().cpu().numpy())
                                mbuf["pred"].append(p10[_mask].detach().cpu().numpy())

        # (option) vis
        # -------- guard: build sample only if we have ≥4 rollout steps ----------
        if len(viz_samples) < 3 and len(preds) >= 4:
            try:
                viz_samples.append({
                    # inputs
                    "input_00h_pm2p5": x0.surf_vars["pm2p5"][0, 0].cpu().numpy(),
                    "input_06h_pm2p5": x0.surf_vars["pm2p5"][0, 1].cpu().numpy(),
                    "input_00h_pm10" : x0.surf_vars["pm10"][0, 0].cpu().numpy(),
                    "input_06h_pm10" : x0.surf_vars["pm10"][0, 1].cpu().numpy(),
                    # single GT frame (+12 h)
                    "target_12h_pm2p5": tgt.surf_vars["pm2p5"][0, 0].cpu().numpy(),
                    "target_12h_pm10" : tgt.surf_vars["pm10"][0, 0].cpu().numpy(),
                    # first four rollout predictions
                    "predictions_pm2p5": [p.surf_vars["pm2p5"][0, 0].cpu().numpy() for p in preds[:4]],
                    "predictions_pm10" : [p.surf_vars["pm10"][0, 0].cpu().numpy() for p in preds[:4]],
                    "times"            : [p.metadata.time[0]                     for p in preds[:4]],
                })
            except KeyError as e:
                logging.warning(f"[viz] sample skipped – missing key {e}")

    pbar.close()

    # ------------------------------------------------------------------ #
    # 3. stepwise, metrics
    # ------------------------------------------------------------------ #
    def _compute(buf: dict, thr: int) -> dict:
        """Return zeros when buffer is empty to avoid ValueError."""
        if not buf["pred"] or not buf["tgt"]:
            return {
                "accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0,
                "false_alarm_rate": 0.0, "detection_rate": 0.0,
                "confusion_matrix_raw": np.zeros((2,2), dtype=int),
                "threshold": thr, "pollutant": buf.get("label",""),
                # ensure skill keys exist even when empty
                "csi": 0.0, "ets": 0.0, "tss": 0.0
            }
        y_pred = np.concatenate(buf["pred"])
        y_true = np.concatenate(buf["tgt"])
        m = compute_metrics(y_pred, y_true, buf.get("label",""), thr)
        return attach_skill_scores(m)

    thresholds = {"PM2.5":35, "PM10":80}
    step_results = {p:{} for p in pollutants}

    for p in pollutants:
        thr = thresholds[p]
        for k in step_keys:
            step_results[p][k] = _compute(step_metrics[p][k], thr)

    # ------------------------------------------------------------------ #
    # 4. rollout, metrics
    # ------------------------------------------------------------------ #
    overall_rollout_metrics = {}
    monthly_metrics         = {p:{} for p in pollutants}

    # --- NEW: 4‑class containers ----------------------------------------
    overall_rollout_metrics_4cls = {}
    monthly_metrics_4cls         = {p:{} for p in pollutants}
    # --------------------------------------------------------------------

    for p in pollutants:
        thr = thresholds[p]
        all_pred = [arr for k in step_keys for arr in step_metrics[p][k]["pred"]]
        all_tgt  = [arr for k in step_keys for arr in step_metrics[p][k]["tgt" ]]
        if not all_pred:
            overall_rollout_metrics[p] = _compute({"pred":[], "tgt":[], "label":p}, thr)
            # --- NEW: 4‑class  ----------------------------------
            overall_rollout_metrics_4cls[p] = compute_multiclass_metrics(
                np.array([]), np.array([]), pollutant_name=p
            )
            continue

        overall_rollout_metrics[p] = compute_metrics(np.concatenate(all_pred),
                                                    np.concatenate(all_tgt ), p, thr)
        overall_rollout_metrics[p] = attach_skill_scores(overall_rollout_metrics[p])
        # --- NEW: 4‑class overall ----------------------------------------
        overall_rollout_metrics_4cls[p] = compute_multiclass_metrics(
            np.concatenate(all_pred), np.concatenate(all_tgt), pollutant_name=p
        )

        for mon, buf in monthly_buffer[p].items():
            monthly_metrics[p][mon] = _compute(buf | {"label":p}, thr)
            monthly_metrics[p][mon]["month_num"] = buf["mon"]
            # --- NEW: 4‑class monthly ------------------------------------
            monthly_metrics_4cls[p][mon] = compute_multiclass_metrics(
                np.concatenate(buf["pred"]), np.concatenate(buf["tgt"]), pollutant_name=p
            )
            monthly_metrics_4cls[p][mon]["month_num"] = buf["mon"]

    # ------------------------------------------------------------------ #
    # 5. TensorBoard
    # ------------------------------------------------------------------ #
    # 5. TensorBoard / Save (with 4-class)
    if writer:
        log_rollout_metrics_to_tensorboard(
            writer, step_results, viz_samples,
            monthly_metrics, overall_rollout_metrics,
            overall_rollout_metrics_4cls, monthly_metrics_4cls
        )
    if save_dir:
        save_rollout_results(
            save_dir, step_results, viz_samples,
            monthly_metrics, overall_rollout_metrics,
            overall_rollout_metrics_4cls, monthly_metrics_4cls
        )

    return (step_results,
            monthly_metrics,
            overall_rollout_metrics,
            overall_rollout_metrics_4cls,
            monthly_metrics_4cls)



def rollout_prediction_with_progress(
    model: torch.nn.Module,
    initial_batch: Batch,
    target_hours: int = 72,
    step_hours:   int = 6,
) -> list[Batch]:
    """Same as above, but shows a progress‑bar in the terminal."""
    core    = model.module if isinstance(model, DDP) else model
    n_steps = target_hours // step_hours
    device  = next(core.parameters()).device

    preds, pbar = [], tqdm(range(n_steps), desc="Roll‑out", unit="step", ncols=80)
    with torch.no_grad():
        for pred in aurora_rollout(core, initial_batch.to(device), steps=n_steps):
            preds.append(pred)
            pbar.update(1)
    pbar.close()
    return preds


def log_rollout_metrics_to_tensorboard(
    writer,
    step_results,
    viz_samples,
    monthly_metrics=None,
    overall_rollout_metrics=None,
    overall_rollout_metrics_4cls=None,
    monthly_metrics_4cls=None,
):
    """TensorBoard logging for rollout (2-class + 4-class)."""
    if writer is None:
        return

    # step-wise (2-class)
    for pollutant, steps in step_results.items():
        for step_key, m in steps.items():
            try:
                parts = step_key.split('_'); step_idx = int(parts[1]) if len(parts) > 2 else 0
            except Exception:
                step_idx = 0
            for name in ["accuracy","f1","precision","recall","false_alarm_rate","detection_rate",
             "csi","ets","tss"]:
                writer.add_scalar(f"Rollout/{pollutant}/{step_key}/{name}", m.get(name, 0.0), step_idx)
            if "confusion_matrix_raw" in m:
                create_confusion_matrix_plot(writer, m["confusion_matrix_raw"],
                                             f"Rollout/{pollutant}/{step_key}/Confusion_Matrix", step_idx)

    # overall (2-class)
    if overall_rollout_metrics:
        for pollutant, m in overall_rollout_metrics.items():
            for name in ["accuracy","f1","precision","recall","false_alarm_rate","detection_rate",
             "csi","ets","tss"]:
                writer.add_scalar(f"Rollout/{pollutant}/Overall/{name}", m.get(name, 0.0), 0)
            if "confusion_matrix_raw" in m:
                create_confusion_matrix_plot(writer, m["confusion_matrix_raw"],
                                             f"Rollout/{pollutant}/Overall/Confusion_Matrix", 0)

    # monthly (2-class)
    if monthly_metrics:
        for pollutant, months in monthly_metrics.items():
            for mon_name, m in months.items():
                mon = m.get("month_num", 0)
                for name in ["accuracy","f1","precision","recall","false_alarm_rate","detection_rate",
             "csi","ets","tss"]:
                    writer.add_scalar(f"Rollout_Monthly/{pollutant}/{name}", m.get(name, 0.0), mon)
                if "confusion_matrix_raw" in m:
                    create_confusion_matrix_plot(writer, m["confusion_matrix_raw"],
                                                 f"Rollout_Monthly/{pollutant}/{mon_name}/Confusion_Matrix", mon)

    # overall (4-class)
    if overall_rollout_metrics_4cls:
        for pollutant, m4 in overall_rollout_metrics_4cls.items():
            writer.add_scalar(f"Rollout/{pollutant}_4C/Overall_Accuracy",    m4["overall"]["accuracy"], 0)
            writer.add_scalar(f"Rollout/{pollutant}_4C/Overall_F1_macro",    m4["overall"]["f1_macro"], 0)
            writer.add_scalar(f"Rollout/{pollutant}_4C/Overall_F1_weighted", m4["overall"]["f1_weighted"], 0)
            writer.add_scalar(f"Rollout/{pollutant}_4C/Overall_F1_micro",    m4["overall"]["f1_micro"], 0)
            for cname, cm in m4["per_class"].items():
                base = f"Rollout/{pollutant}_4C/PerClass/{cname}"
                writer.add_scalar(base + "/F1",        cm["f1"],     0)
                writer.add_scalar(base + "/Recall",    cm["recall"], 0)
                writer.add_scalar(base + "/Precision", cm["precision"], 0)
                writer.add_scalar(base + "/FAR",       cm["far"],    0)
                writer.add_scalar(base + "/Acc_OVR",   cm["acc_ovr"], 0)

    # monthly (4-class)
    if monthly_metrics_4cls:
        for pollutant, mon_dict in monthly_metrics_4cls.items():
            for month_name, m4 in mon_dict.items():
                month_num = m4["month_num"]
                writer.add_scalar(f"Rollout_Monthly/{pollutant}_4C/Accuracy",    m4["overall"]["accuracy"],    month_num)
                writer.add_scalar(f"Rollout_Monthly/{pollutant}_4C/F1_macro",    m4["overall"]["f1_macro"],    month_num)
                writer.add_scalar(f"Rollout_Monthly/{pollutant}_4C/F1_weighted", m4["overall"]["f1_weighted"], month_num)
                writer.add_scalar(f"Rollout_Monthly/{pollutant}_4C/F1_micro",    m4["overall"]["f1_micro"],    month_num)
                for cname, cm in m4["per_class"].items():
                    base = f"Rollout_Monthly/{pollutant}_4C/PerClass/{cname}"
                    writer.add_scalar(base + "/F1",        cm["f1"],     month_num)
                    writer.add_scalar(base + "/Recall",    cm["recall"], month_num)
                    writer.add_scalar(base + "/Precision", cm["precision"], month_num)
                    writer.add_scalar(base + "/FAR",       cm["far"],    month_num)
                    writer.add_scalar(base + "/Acc_OVR",   cm["acc_ovr"], month_num)

    # rollout progression visuals
    _required = {
        "input_00h_pm2p5","input_06h_pm2p5","target_12h_pm2p5",
        "input_00h_pm10","input_06h_pm10","target_12h_pm10",
        "predictions_pm2p5","predictions_pm10","times"
    }
    for i, sample in enumerate(viz_samples):
        if not _required.issubset(sample.keys()):
            missing = sorted(_required - sample.keys())
            logging.warning(f"[viz] sample {i} skipped – missing {missing}")
            continue
        create_dual_pollutant_rollout_visualization(writer, sample, f"Rollout/Sample_{i+1}", i)


def create_dual_pollutant_rollout_visualization(writer, sample, tag, step):
    """Create rollout progression visualization for both PM2.5 and PM10."""
    
    fig, axes = plt.subplots(4, 3, figsize=(18, 24))
    
    # PM2.5 input timesteps (row 0)
    im1 = axes[0, 0].imshow(sample['input_00h_pm2p5'], vmin=-50, vmax=50, cmap="viridis")
    axes[0, 0].set_title("PM2.5 Input: 00h")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(sample['input_06h_pm2p5'], vmin=-50, vmax=50, cmap="viridis")
    axes[0, 1].set_title("PM2.5 Input: 06h")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])
    plt.colorbar(im2, ax=axes[0, 1])
    
    im3 = axes[0, 2].imshow(sample['target_12h_pm2p5'], vmin=-50, vmax=50, cmap="viridis")
    axes[0, 2].set_title("PM2.5 Target: 12h")
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])
    plt.colorbar(im3, ax=axes[0, 2])
    
    # PM2.5 predictions (row 1)
    for k, (pred, t) in enumerate(zip(sample['predictions_pm2p5'], sample['times'])):
        if k < 3:
            im = axes[1, k].imshow(pred, vmin=-50, vmax=50, cmap="viridis")
            axes[1, k].set_title(f"PM2.5 Pred: {t.strftime('%H')}h (+{6*(k+1)}h)")
            axes[1, k].set_xticks([])
            axes[1, k].set_yticks([])
            plt.colorbar(im, ax=axes[1, k])
    
    # PM10 input timesteps (row 2)
    im4 = axes[2, 0].imshow(sample['input_00h_pm10'], vmin=-50, vmax=100, cmap="plasma")
    axes[2, 0].set_title("PM10 Input: 00h")
    axes[2, 0].set_xticks([])
    axes[2, 0].set_yticks([])
    plt.colorbar(im4, ax=axes[2, 0])
    
    im5 = axes[2, 1].imshow(sample['input_06h_pm10'], vmin=-50, vmax=100, cmap="plasma")
    axes[2, 1].set_title("PM10 Input: 06h")
    axes[2, 1].set_xticks([])
    axes[2, 1].set_yticks([])
    plt.colorbar(im5, ax=axes[2, 1])
    
    im6 = axes[2, 2].imshow(sample['target_12h_pm10'], vmin=-50, vmax=100, cmap="plasma")
    axes[2, 2].set_title("PM10 Target: 12h")
    axes[2, 2].set_xticks([])
    axes[2, 2].set_yticks([])
    plt.colorbar(im6, ax=axes[2, 2])
    
    # PM10 predictions (row 3)
    for k, (pred, t) in enumerate(zip(sample['predictions_pm10'], sample['times'])):
        if k < 3:
            im = axes[3, k].imshow(pred, vmin=-50, vmax=100, cmap="plasma")
            axes[3, k].set_title(f"PM10 Pred: {t.strftime('%H')}h (+{6*(k+1)}h)")
            axes[3, k].set_xticks([])
            axes[3, k].set_yticks([])
            plt.colorbar(im, ax=axes[3, k])
    
    plt.tight_layout()
    writer.add_figure(tag, fig, step)
    plt.close(fig)


def save_rollout_results(
    save_dir,
    step_results,
    viz_samples,
    monthly_metrics=None,
    overall_rollout_metrics=None,
    overall_rollout_metrics_4cls=None,   # ← 추가
    monthly_metrics_4cls=None            # ← 추가
):
    """Save rollout test results for both PM2.5 and PM10 to files."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save overall rollout metrics if available (ADD THIS SECTION)
    if overall_rollout_metrics:
        with open(os.path.join(save_dir, "rollout_overall_metrics_all_pollutants.txt"), "w") as f:
            f.write("Overall Rollout Prediction Metrics for PM2.5 and PM10:\n")
            f.write("=" * 60 + "\n")
            
            for pollutant in ['PM2.5', 'PM10']:
                if pollutant in overall_rollout_metrics:
                    f.write(f"\n{pollutant} Overall Rollout Metrics (Threshold: {overall_rollout_metrics[pollutant]['threshold']} μg/m³):\n")
                    f.write("-" * 40 + "\n")
                    for key, value in overall_rollout_metrics[pollutant].items():
                        if key not in ['confusion_matrix', 'confusion_matrix_raw', 'pollutant', 'threshold']:
                            f.write(f"  {key}: {value}\n")
                    
                    cm = overall_rollout_metrics[pollutant]['confusion_matrix_raw']
                    f.write(f"  Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}\n")
    
    # Save step-wise metrics for both pollutants
    with open(os.path.join(save_dir, "rollout_metrics_all_pollutants.txt"), "w") as f:
        f.write("Rollout Prediction Metrics for PM2.5 and PM10:\n")
        f.write("=" * 60 + "\n")
        
        for pollutant in ['PM2.5', 'PM10']:
            if pollutant in step_results:
                f.write(f"\n{pollutant} Rollout Metrics:\n")
                f.write("=" * 30 + "\n")
                
                for step_key, metrics in step_results[pollutant].items():
                    f.write(f"\n{step_key}:\n")
                    f.write("-" * 20 + "\n")
                    for key, value in metrics.items():
                        if key not in ['confusion_matrix', 'confusion_matrix_raw', 'pollutant', 'threshold']:
                            f.write(f"  {key}: {value}\n")
                    
                    cm = metrics['confusion_matrix_raw']
                    f.write(f"  Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}\n")
    
    # Save monthly metrics if available
    if monthly_metrics:
        with open(os.path.join(save_dir, "rollout_monthly_metrics_all_pollutants.txt"), "w") as f:
            f.write("Monthly Rollout Prediction Metrics for PM2.5 and PM10:\n")
            f.write("=" * 60 + "\n")
            
            for pollutant in ['PM2.5', 'PM10']:
                if pollutant in monthly_metrics:
                    f.write(f"\n{pollutant} Monthly Rollout Metrics:\n")
                    f.write("=" * 30 + "\n")
                    
                    for month_name, metrics in monthly_metrics[pollutant].items():
                        f.write(f"\n{month_name}:\n")
                        f.write("-" * 20 + "\n")
                        for key, value in metrics.items():
                            if key not in ['confusion_matrix', 'confusion_matrix_raw', 'pollutant', 'threshold', 'month_num']:
                                f.write(f"  {key}: {value}\n")
                        
                        cm = metrics['confusion_matrix_raw']
                        f.write(f"  Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}\n")
    
    # ---- NEW: 4‑class overall 저장 -----------------------------------
    if overall_rollout_metrics_4cls:
        with open(os.path.join(save_dir, "rollout_overall_metrics_4class.txt"), "w") as f:
            f.write("Overall 4-Class Metrics (good / moderate / bad / very_bad)\n")
            f.write("=" * 60 + "\n")
            for pollutant, m4 in overall_rollout_metrics_4cls.items():
                f.write(f"\n[{pollutant}] bins (t1,t2,t3) = {m4['bins']}\n")
                f.write(f"  Accuracy : {m4['overall']['accuracy']}\n")
                f.write(f"  F1_macro: {m4['overall']['f1_macro']} | "
                        f"F1_weighted: {m4['overall']['f1_weighted']} | "
                        f"F1_micro: {m4['overall']['f1_micro']}\n")
                f.write("  Per-class:\n")
                for cname, cm in m4["per_class"].items():
                    f.write(f"    - {cname:<10} "
                            f"F1={cm['f1']}, Recall={cm['recall']}, "
                            f"Precision={cm['precision']}, FAR={cm['far']}, "
                            f"Acc_OVR={cm['acc_ovr']}, Support={cm['support']}\n")

    # ---- NEW: 4‑class monthly 저장 -----------------------------------
    if monthly_metrics_4cls:
        with open(os.path.join(save_dir, "rollout_monthly_metrics_4class.txt"), "w") as f:
            f.write("Monthly 4-Class Metrics\n")
            f.write("=" * 60 + "\n")
            for pollutant, mon_dict in monthly_metrics_4cls.items():
                f.write(f"\n[{pollutant}]\n")
                for month_name, m4 in mon_dict.items():
                    f.write(f"\n{month_name} (bins={m4['bins']})\n")
                    f.write(f"  Accuracy : {m4['overall']['accuracy']}\n")
                    f.write(f"  F1_macro: {m4['overall']['f1_macro']} | "
                            f"F1_weighted: {m4['overall']['f1_weighted']} | "
                            f"F1_micro: {m4['overall']['f1_micro']}\n")
                    f.write("  Per-class:\n")
                    for cname, cm in m4["per_class"].items():
                        f.write(f"    - {cname:<10} "
                                f"F1={cm['f1']}, Recall={cm['recall']}, "
                                f"Precision={cm['precision']}, FAR={cm['far']}, "
                                f"Acc_OVR={cm['acc_ovr']}, Support={cm['support']}\n")

def main():
    parser = argparse.ArgumentParser(description='Test script for Aurora / Aurora‑Air‑Pollution')
    parser.add_argument('--test-start-date', default='2019-01-01', help='Test start date (YYYY-MM-DD)')
    parser.add_argument('--test-end-date', default='2019-12-31', help='Test end date (YYYY-MM-DD)')
    parser.add_argument('--data-dir', default='/fsx/mid_term/fine_dust/era5', help='Data directory')
    parser.add_argument('--npz-path', default="/fsx/mid_term/fine_dust/23-air-pollution/obs_npz", help='NPZ file path')
    parser.add_argument('--checkpoint-path', default='/fsx/mid_term/aurora/checkpoints/Train:2020-01-01-2020-12-31_Test:2019-01-01-2019-12-31/obs_with_aurora25_patchsz2_pretrained_accstep4_rollout2/best_pm2p5.pth', help='Path to model checkpoint')
    parser.add_argument('--batch', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--results-dir', default=None, help='Directory to save test results')
    parser.add_argument('--tensorboard-dir', default='./runs/', help='TensorBoard log directory (optional)')
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--rollout-hours', type=int, default=72, help='Hours to predict ahead using rollout (default: 72)')
    parser.add_argument('--mode', choices=['standard', 'rollout'], default='standard', help='Inference mode: standard (6h) or rollout (multi-step)')

    parser.add_argument('--model', choices=['aurora', 'pollution'], default='aurora',
                        help='aurora : generic pre‑trained,  pollution : air‑pollution head')
    parser.add_argument('--data-sources', default='obs', help='comma‑separated list ⇒ obs, cmaq, obs,cmaq …')
    parser.add_argument("--cmaq-root",
                default="/fsx/mid_term/fine_dust/cmaq_npy",
                help="root folder containing hourly *_cmaq.npy files")
    parser.add_argument("--flow-root",
                default="/fsx/mid_term/fine_dust/cmaq_flow",
                help="root folder containing hourly *_cmaq.npy files")
    parser.add_argument("--use-wind-prompt", action="store_true", help="Enable wind visual prompt.")
    parser.add_argument('--use_cutmix',    action='store_true', help="Train with cutmix with CMAQ.")
    parser.add_argument('--use_cmaq_pm_only', action="store_true",  help="use only PM2.5 and PM10 of CMAQ")
    parser.add_argument('--use_hybrid_target', action="store_true",  help="use GT as OBS+CMAQ")
    parser.add_argument("--use_flow",      action="store_true")

    args = parser.parse_args()
    
    # Initialize distributed training if specified
    distributed = args.distributed
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        if torch.cuda.device_count() > 1:
            dist.init_process_group(backend='nccl')
        is_master = (local_rank == 0)
    else:
        local_rank = 0
        is_master = True
    
    # Fix seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create results directory
    if is_master and args.results_dir is not None:
        os.makedirs(args.results_dir, exist_ok=True)
    
    # Load test dataloader
    sources = [s.strip().lower() for s in args.data_sources.split(',')]
    test_horizon = args.rollout_hours // 6 if args.mode == "rollout" else 1
    logging.info(f"[INFO] use inputs: {sources}")
    test_dataloader = get_test_dataloader(
        args.batch,
        args.data_dir,
        args.npz_path,
        args.test_start_date,
        args.test_end_date,
        distributed=distributed,
        cmaq_root=args.cmaq_root,
        flow_root=args.flow_root,
        sources=tuple(sources),
        horizon=test_horizon,
        use_wind_prompt=args.use_wind_prompt,
        use_cutmix=args.use_cutmix,
        use_cmaq_pm_only=args.use_cmaq_pm_only,
        # use_hybrid_target=args.use_hybrid_target,
        use_flow=args.use_flow,
    )
    
    # Initialize model
    logging.info("Fetching a sample batch to configure the model...")
    in0, tgt0 = None, None
    for batch in test_dataloader:
        if batch is not None and batch[0] is not None:
            in0, tgt0 = batch
            break
    if in0 is None:
        raise RuntimeError("Could not retrieve a valid batch from the test dataloader.")

    # Dynamically derive variable lists from the actual data batch
    surf_vars = tuple(sorted(in0.surf_vars.keys()))
    static_vars = tuple(sorted(in0.static_vars.keys()))
    atmos_vars = tuple(sorted(in0.atmos_vars.keys()))

    logging.info(f"► Surface variables from data: {surf_vars}")
    logging.info(f"► Static variables from data: {static_vars}")
    logging.info(f"► Atmospheric variables from data: {atmos_vars}")
    
    # Detect Z (number of vertical levels) from any atmos var, if present
    if atmos_vars:
        any_key = atmos_vars[0]
        Z_detected = in0.atmos_vars[any_key].shape[2]  # [B, T, Z, H, W]
    else:
        # Z_detected = 5 # A sensible default if no atmospheric variables are present
        Z_detected = 2

    # ------------------------------------------------------------------ #
    # Choose a window size that is compatible with the current data.
    #   · vertical window  = 1     (always divides latent_levels)
    #   · height  window   = GCD(H//patch,  8) or 1
    #   · width   window   = GCD(W//patch, 12) or 1
    # This keeps memory low while satisfying Swin‑3D requirements.
    # ------------------------------------------------------------------ #
    import math
    H, W = in0.spatial_shape
    patch_sz      = 2                       # same as you pass to Aurora
    h_tokens      = H // patch_sz
    w_tokens      = W // patch_sz
    win_h         = math.gcd(h_tokens, 8) or 1
    win_w         = math.gcd(w_tokens, 12) or 1
    window_size   = (1, win_h, win_w)       # (levels_axis, H, W)
    logging.info(f"window_size chosen = {window_size}")

    if args.model == "pollution":
        logging.info(f"MODEL: aurora-0.4-air-pollution.ckpt")
        model = AuroraAirPollution(
            surf_vars=surf_vars,
            static_vars=static_vars, # Pass the discovered static vars
            atmos_vars=atmos_vars,   # Pass the discovered atmos vars
            patch_size=patch_sz,
            latent_levels=Z_detected,
            window_size=window_size
        )
    else:
        logging.info(f"MODEL: aurora-0.25-pretrained.ckpt")
        model = Aurora(
            surf_vars=surf_vars,
            static_vars=static_vars, # Pass the discovered static vars
            atmos_vars=atmos_vars,   # Pass the discovered atmos vars
            patch_size=patch_sz,
            latent_levels=Z_detected,
            window_size=window_size,
            use_flow=args.use_flow,
        )
    
    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model, checkpoint_epoch = load_checkpoint(model, args.checkpoint_path, device)
    
    # Wrap model in DDP if using distributed training
    if distributed and torch.cuda.device_count() > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
    
    # Setup TensorBoard writer if specified
    writer = None
    if is_master and args.tensorboard_dir:
        mode_suffix = f"_rollout_{args.rollout_hours}h" if args.mode == 'rollout' else "_standard"
        run_dir = os.path.join(args.tensorboard_dir, 
                              "test_" + datetime.now().strftime("%Y%m%d-%H%M%S") + 
                              f"_TEST:{args.test_start_date}~{args.test_end_date}" + mode_suffix + "_pm2p5_PM10")
        os.makedirs(run_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=run_dir)
        if args.results_dir is None:
            args.results_dir = run_dir
        logging.info(f"TensorBoard logs will be saved to: {run_dir}")
    
    # Run inference based on mode
    if args.mode == 'rollout':
        logging.info(f"Starting rollout evaluation for {args.rollout_hours} hours from {args.test_start_date} to {args.test_end_date}")
        logging.info(f"Using checkpoint from epoch {checkpoint_epoch}")
        
        (step_results, monthly_metrics, overall_rollout_metrics,
         overall_rollout_metrics_4cls, monthly_metrics_4cls) = rollout_inference_and_evaluate(
            test_dataloader, 
            model, 
            args.rollout_hours,
            writer if is_master else None,
            args.results_dir if is_master else None,
            args.test_start_date,
            npz_path=args.npz_path,
            checkpoint_path=args.checkpoint_path,
            use_rollout_cutmix=args.use_cutmix,
            cmaq_root=args.cmaq_root,
        )
        
        # Print rollout summary for both pollutants
        if is_master:
            print("\n" + "="*80)
            print("ROLLOUT EVALUATION SUMMARY - PM2.5 and PM10")
            print("="*80)
            print(f"Test Period: {args.test_start_date} to {args.test_end_date}")
            print(f"Rollout Hours: {args.rollout_hours}")
            print(f"Checkpoint: {args.checkpoint_path} (epoch {checkpoint_epoch})")
            
            # Add overall rollout metrics display
            for pollutant in ['PM2.5', 'PM10']:
                if pollutant in overall_rollout_metrics:
                    print(f"\n{pollutant} Overall Rollout Metrics (Threshold: {overall_rollout_metrics[pollutant]['threshold']} μg/m³):")
                    for key, value in overall_rollout_metrics[pollutant].items():
                        if key not in ['confusion_matrix', 'confusion_matrix_raw', 'pollutant', 'threshold']:
                            print(f"  {key}: {value}")
            
            for pollutant in ['PM2.5', 'PM10']:
                if pollutant in step_results:
                    print(f"\n{pollutant} Step-wise Results:")
                    for step_key, metrics in step_results[pollutant].items():
                        print(f"  {step_key}: {_fmt(metrics)}")
                    
                    if pollutant in monthly_metrics:
                        print(f"\n{pollutant} Monthly Results:")
                        for month_name, metrics in monthly_metrics[pollutant].items():
                            print(f"  {month_name}: {_fmt(metrics)}")
            
            # 4-class rollout overall
            for pollutant in ['PM2.5', 'PM10']:
                if overall_rollout_metrics_4cls and pollutant in overall_rollout_metrics_4cls:
                    m4 = overall_rollout_metrics_4cls[pollutant]
                    print(f"\n{pollutant} Overall Rollout (4‑class, bins={m4['bins']}):")
                    print(f"  Acc={m4['overall']['accuracy']}, "
                          f"F1_macro={m4['overall']['f1_macro']}, "
                          f"F1_weighted={m4['overall']['f1_weighted']}, "
                          f"F1_micro={m4['overall']['f1_micro']}")
                    for cname, cm in m4['per_class'].items():
                        print(f"    [{cname}] F1={cm['f1']}, Recall={cm['recall']}, "
                              f"Precision={cm['precision']}, FAR={cm['far']}, "
                              f"Acc_OVR={cm['acc_ovr']}")
            # 4-class rollout monthly
            for pollutant in ['PM2.5', 'PM10']:
                if monthly_metrics_4cls and pollutant in monthly_metrics_4cls:
                    print(f"\n{pollutant} Monthly Rollout 4‑class Results:")
                    for month_name, m4 in monthly_metrics_4cls[pollutant].items():
                        print(f"  {month_name}: Acc={m4['overall']['accuracy']}, "
                              f"F1_macro={m4['overall']['f1_macro']}, "
                              f"F1_weighted={m4['overall']['f1_weighted']}, "
                              f"F1_micro={m4['overall']['f1_micro']}")

            
            print(f"\nResults saved to: {args.results_dir}")
    
    else:
        # Standard 6-hour evaluation for both pollutants
        logging.info(f"Starting standard evaluation from {args.test_start_date} to {args.test_end_date}")
        logging.info(f"Using checkpoint from epoch {checkpoint_epoch}")
        
        overall_metrics, monthly_metrics, overall_metrics_4cls, monthly_metrics_4cls = inference_and_evaluate(
            test_dataloader, model, writer if is_master else None,
            args.results_dir if is_master else None, args.test_start_date
        )
        
        # Print standard summary for both pollutants
        if is_master:
            print("\n" + "="*80)
            print("STANDARD EVALUATION SUMMARY - PM2.5 and PM10")
            print("="*80)
            print(f"Test Period: {args.test_start_date} to {args.test_end_date}")
            print(f"Checkpoint: {args.checkpoint_path} (epoch {checkpoint_epoch})")

            for pollutant in ['PM2.5', 'PM10']:
                if pollutant in overall_metrics:
                    print(f"\n{pollutant} Overall (binary @ {overall_metrics[pollutant]['threshold']} μg/m³):")
                    for key, value in overall_metrics[pollutant].items():
                        if key not in ['confusion_matrix', 'confusion_matrix_raw', 'pollutant', 'threshold']:
                            print(f"  {key}: {value}")

                if pollutant in overall_metrics_4cls:
                    m4 = overall_metrics_4cls[pollutant]
                    print(f"\n{pollutant} Overall (4‑class, bins={m4['bins']}):")
                    print(f"  Acc={m4['overall']['accuracy']}, "
                        f"F1_macro={m4['overall']['f1_macro']}, "
                        f"F1_weighted={m4['overall']['f1_weighted']}, "
                        f"F1_micro={m4['overall']['f1_micro']}")
                    for cname, cm in m4["per_class"].items():
                        print(f"    [{cname}] F1={cm['f1']}, Recall={cm['recall']}, "
                            f"Precision={cm['precision']}, FAR={cm['far']}, Acc_OVR={cm['acc_ovr']}")

            # ---- 4‑class 콘솔 출력 -----------------------------------
            for pollutant in ['PM2.5', 'PM10']:
                if overall_metrics_4cls and pollutant in overall_metrics_4cls:
                    m4 = overall_metrics_4cls[pollutant]
                    print(f"\n{pollutant} Overall (4‑class, bins={m4['bins']}):")
                    print(f"  Acc={m4['overall']['accuracy']}, "
                            f"F1_macro={m4['overall']['f1_macro']}, "
                            f"F1_weighted={m4['overall']['f1_weighted']}, "
                            f"F1_micro={m4['overall']['f1_micro']}")
                    for cname, cm in m4["per_class"].items():
                        print(f"    [{cname}] F1={cm['f1']}, Recall={cm['recall']}, "
                                f"Precision={cm['precision']}, FAR={cm['far']}, Acc_OVR={cm['acc_ovr']}, "
                                f"Support={cm['support']}")

            for pollutant in ['PM2.5', 'PM10']:
                if pollutant in monthly_metrics_4cls:
                    print(f"\n{pollutant} Monthly 4‑class Results:")
                    for month_name, m4 in monthly_metrics_4cls[pollutant].items():
                        print(f"  {month_name}: Acc={m4['overall']['accuracy']}, "
                                f"F1_macro={m4['overall']['f1_macro']}, "
                                f"F1_weighted={m4['overall']['f1_weighted']}, "
                                f"F1_micro={m4['overall']['f1_micro']}")
                        
            print(f"\nResults saved to: {args.results_dir}")
    
    # Cleanup
    if writer:
        writer.close()
    
    if distributed and torch.cuda.device_count() > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
