import torch
import numpy as np
import random
import os
from torch.utils.data import Dataset
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
from aurora import Batch, Metadata
from scipy.ndimage import zoom
import calendar
from dataclasses import is_dataclass, replace as dc_replace
# --- Add libraries for visualization ---
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.interpolate import RegularGridInterpolator
import io
from typing import Optional, Tuple, Dict, Any

HOUR2IDX = {"00": 0, "06": 1, "12": 2, "18": 3}

# Variable order inside raw CMAQ arrays (fixed by preprocessing) -------------
CONC_VARS = [
    "SO4_25", "NH4_25", "NO3_25", "ORG_25", "EC_25", "MISC_25",
    "PM2P5", "tcso2", "tcco", "tcno2", "O3", "NO", "NOx",
    "SO4_10", "NH4_10", "NO3_10", "ORG_10", "EC_10", "MISC_10",
    "PM10", "ISOPRENE", "OLES", "AROS", "ALKS",
]  # 24
M2D_VARS = [
    "SOIT2", "SOIT1", "TEMPG", "PRSFC", "WSPD10", "USTAR", "GLW", "HFX",
    "PBL", "RADYNI", "WSTAR", "RN", "RC", "MOLI", "RSTOMI", "Q2", "WDIR10",
    "TEMP2", "GSW", "RGRND", "RH", "CFRAC", "CLDT", "CLDB", "WBAR",
    "SNOCOV", "VEG", "LAI", "SOIM1", "SOIM2", "SLTYP", "WR", "LH",
]  # 33
M3D_VARS = ["UWIND", "WWIND", "VWIND", "QC", "TA", "QR", "PRES", "DENS"]  # 8
_CMAQ_LEVELS = ("sfc", "925", "850", "700", "500")   # order we will keep
_CMAQ_PLEV = (1013, 925, 850, 700, 500)

# ------------------------- East Asia 0.40° bounding box --------------------- #
# We use this box to crop CAMS to OBS domain (matches your default OBS grid).
_EA_SOUTH, _EA_NORTH = 19.6, 47.6
_EA_WEST,  _EA_EAST  = 104.8, 159.6

# ------------------------------------------------------------------ #
# Default 0.40° air-pollution grid:   47.6-19.6 / 104.8-159.6
# Default 0.25° ERA5 grid-over-Korea: 49.165-19.415 / 104.643-160.393
# ------------------------------------------------------------------ #
def make_lat_lon(H: int, W: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create a 1-D lat / lon vector that exactly matches an ERA5 or air-pollution
    tile. Pass the bounding box that belongs to your dataset.

    # ---- 0.40° air-pollution (69 × 138) ----
    lat, lon = make_lat_lon(69, 138)

    # ---- 0.25° ERA5 (120 × 224) ----
    lat, lon = make_lat_lon(120, 224)

    # ------- CMAQ (128 × 174) -------
    """
    if H == 120 and W == 224:           # ERA5 0.25°
        south, north = 19.415, 49.165
        west , east  = 104.643, 160.393
    elif H == 128 and W == 174:         # CMAQ 27 km grid
        south, north = 13.0, 52.0
        west,  east  = 97.0, 170.0
    else:                               # default to 0.40° air-pollution grid (OBS/CAMS EA crop)
        south, north = _EA_SOUTH, _EA_NORTH
        west , east  = _EA_WEST , _EA_EAST

    lat = torch.linspace(north, south, H, dtype=torch.float32)
    lon = torch.linspace(west , east , W, dtype=torch.float32)
    return lat, lon

def _normalize_lons_to_360(lon1d: np.ndarray) -> np.ndarray:
    """Convert [-180,180] longitudes to [0,360) for robust slicing."""
    lon1d = np.asarray(lon1d)
    if lon1d.min() < 0:
        lon1d = (lon1d + 360.0) % 360.0
    return lon1d

def _box_to_slices(lat1d: np.ndarray, lon1d: np.ndarray,
                   south: float, west: float, north: float, east: float) -> tuple[slice, slice]:
    """
    Given global 1D lat/lon and a [S,W,N,E] box, return 2D ROI slices.
    Handles both ascending/descending lat and 0..360 vs -180..180 lon.
    """
    lat = np.asarray(lat1d)
    lon = _normalize_lons_to_360(np.asarray(lon1d))
    W_, E_ = west % 360.0, east % 360.0

    # Latitude indices (support both orders)
    if lat[0] > lat[-1]:  # descending (N -> S)
        lat_mask = (lat <= north) & (lat >= south)
    else:                  # ascending (S -> N)
        lat_mask = (lat >= south) & (lat <= north)
    lat_idx = np.where(lat_mask)[0]
    if lat_idx.size == 0:
        raise ValueError("Requested lat box is outside CAMS latitude range.")
    j0, j1 = lat_idx[0], lat_idx[-1] + 1

    # Longitude indices on [0,360)
    lon = lon % 360.0
    if W_ <= E_:
        lon_mask = (lon >= W_) & (lon <= E_)
    else:  # dateline crossing (not expected here but kept for safety)
        lon_mask = (lon >= W_) | (lon <= E_)
    lon_idx = np.where(lon_mask)[0]
    if lon_idx.size == 0:
        raise ValueError("Requested lon box is outside CAMS longitude range.")
    i0, i1 = lon_idx[0], lon_idx[-1] + 1

    return slice(j0, j1), slice(i0, i1)

def _infer_to_ug_scale_from_units(units: str) -> float:
    """Return multiplier to convert to μg m^-3 based on a CF-style units string."""
    u = (units or "").lower()
    if "kg" in u:
        return 1e9   # kg m^-3 → μg m^-3
    return 1.0       # already μg m^-3 (or unknown) → pass-through

def _safe_zoom2d(arr2d: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Resize a 2D array to (H_tgt, W_tgt) using bilinear interpolation."""
    h, w = arr2d.shape
    th, tw = target_hw
    if (h, w) == (th, tw):
        return arr2d
    return zoom(arr2d, (th / h, tw / w), order=1)

def wind_to_image(u_wind, v_wind, H, W, density=1.5, linewidth_factor=1.8, vmax_speed=20.0):
    """
    Converts u and v wind components into a visual prompt using a streamplot.
    - Linewidth represents wind speed.
    - A gradient along each streamline indicates flow direction.
    - Handles out-of-bounds coordinates to prevent errors.
    """
    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    y_coords, x_coords = np.mgrid[0:H, 0:W]
    speed = np.sqrt(u_wind**2 + v_wind**2)

    strm = ax.streamplot(x_coords, y_coords, u_wind, -v_wind,
                         linewidth=0, # Calculate paths only
                         density=density)
    
    segments = strm.lines.get_segments()
    
    lines = []
    line_widths = []
    colors = []
    
    cmap = plt.get_cmap('plasma') # A vibrant colormap that shows gradient well in grayscale
    
    speed_interpolator = RegularGridInterpolator((np.arange(H), np.arange(W)), speed)

    for line in segments:
        if len(line) < 2:
            continue
            
        gradient = np.linspace(0.0, 1.0, len(line))
        
        line_points = np.array(line)
        
        # --- START OF ERROR FIX ---
        # Clip coordinates to be within the valid grid bounds [0, H-1] and [0, W-1]
        line_points[:, 0] = np.clip(line_points[:, 0], 0, W - 1)
        line_points[:, 1] = np.clip(line_points[:, 1], 0, H - 1)
        # --- END OF ERROR FIX ---

        # Interpolator expects (y, x) coordinates, so we swap columns
        segment_speeds = speed_interpolator(line_points[:-1, ::-1])
        
        norm_speed = np.clip(segment_speeds / vmax_speed, 0, 1)
        widths = (0.7 + 2.0 * norm_speed) * linewidth_factor # Increased base linewidth

        points = line_points.reshape(-1, 1, 2)
        segments_for_lc = np.concatenate([points[:-1], points[1:]], axis=1)
        
        lines.extend(segments_for_lc)
        line_widths.extend(widths)
        colors.extend(cmap(gradient[:-1]))

    line_collection = LineCollection(lines, colors=colors, linewidths=line_widths, antialiased=True)
    ax.add_collection(line_collection)
    
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    with io.BytesIO() as buf:
        fig.savefig(buf, format='png', facecolor='black', edgecolor='none', transparent=True)
        buf.seek(0)
        img = plt.imread(buf)
    
    plt.close(fig)
    
    # Convert RGBA to a single grayscale channel using luminosity
    wind_prompt = (0.299 * img[:H, :W, 0] + 0.587 * img[:H, :W, 1] + 0.114 * img[:H, :W, 2])
    return (wind_prompt * 255).astype(np.uint8)

def wind_to_image_arrow(u_wind, v_wind, H, W, density=1.5, linewidth=1.2):
    """
    Converts u and v wind components into a visual prompt using a streamplot
    with a gradient along each streamline to clearly indicate flow direction.
    """
    fig, ax = plt.subplots(figsize=(W/100, H/100), dpi=100)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    y, x = np.mgrid[0:H, 0:W]
    
    # Generate the streamplot data but don't render it yet
    strm = ax.streamplot(x, y, u_wind, -v_wind,
                         linewidth=0, # Set linewidth to 0 to avoid drawing
                         density=density)
    
    # --- START OF MODIFICATION ---
    # Get the line segments from the streamplot object
    segments = strm.lines.get_segments()
    
    # Create a new LineCollection with custom gradient colors
    cmap = plt.get_cmap('viridis') 

    lines = []
    colors = []
    for line in segments:
        gradient = np.linspace(0.0, 1.0, len(line))
        points = np.array([line[:-1], line[1:]]).transpose((1, 0, 2))
        lines.extend(points)
        colors.extend(cmap(gradient[:-1]))

    line_collection = LineCollection(lines, colors=colors, linewidths=linewidth, antialiased=True)
    ax.add_collection(line_collection)
    # --- END OF MODIFICATION ---

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    with io.BytesIO() as buf:
        fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), edgecolor='none', transparent=True)
        buf.seek(0)
        img = plt.imread(buf)
    
    plt.close(fig)
    
    wind_prompt = (0.299 * img[:H, :W, 0] + 0.587 * img[:H, :W, 1] + 0.114 * img[:H, :W, 2])
    wind_prompt = (wind_prompt * 255).astype(np.uint8)
    return wind_prompt

class WeatherDataset(Dataset):
    def __init__(
        self,
        start_date='2023-01-01',
        end_date='2023-01-05',
        data_dir="./data/era5/",
        npz_path='./data/obs_hourly_era5',
        handle_leap_year='replace_mar1',
        cmaq_root='',
        cams_dir='',
        flow_root='',
        sources=('obs',),
        horizon=1,
        use_masking=False,
        mask_ratio=0.5,
        use_wind_prompt=False,
        use_cutmix=False,
        use_cmaq_conc=True,
        use_cmaq_m2d=True,
        use_cmaq_m3d=True,
        use_cmaq_pm_only: bool = False,
        use_hybrid_target: bool = False,
        hybrid_target_source: str = "auto",   # NEW: {"auto","cmaq","cams"} for hybrid fallback
        use_flow=False,
    ):
        """
        handle_leap_year options:
        - 'skip': Skip Feb 29th entirely
        - 'replace_feb28': Replace Feb 29th with Feb 28th
        - 'replace_mar1': Replace Feb 29th with Mar 1st
        - 'keep': Keep Feb 29th (may cause file not found errors)

        horizon : Number of 6‑hour steps to predict after the last input
                  e.g. horizon=3 predicts +6h, +12h, +18h.

        sources may include any of: {"obs","era5","cmaq","cams"}.
        If "cams" is present, the loader will add:
          - Inputs:  pm2p5_cams, pm10_cams (two time steps)
          - Targets: pm2p5_cams, pm10_cams (horizon sequence)
        CAMS PM are converted to μg m^-3 if the file units are kg m^-3.
        """
        self.data_dir = Path(data_dir)
        # Respect explicit cams_dir if provided; otherwise derive beside ERA5 root
        self.cams_dir = Path(cams_dir) if cams_dir else Path(str(self.data_dir).replace('era5', 'cams'))
        self.npz_path = npz_path
        self.handle_leap_year = handle_leap_year
        self.cmaq_root = Path(cmaq_root)
        self.flow_root = Path(flow_root)
        self.sources   = set(s.lower() for s in sources)

        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date   = datetime.strptime(end_date, "%Y-%m-%d")
        self.horizon = int(horizon)
        self.use_masking = use_masking
        self.mask_ratio = mask_ratio
        self.use_wind_prompt = use_wind_prompt
        self.use_cutmix = use_cutmix

        self.use_cmaq_conc = bool(use_cmaq_conc)
        self.use_cmaq_m2d  = bool(use_cmaq_m2d)
        self.use_cmaq_m3d  = bool(use_cmaq_m3d)
        self.use_cmaq_pm_only = bool(use_cmaq_pm_only)
        self.use_hybrid_target = bool(use_hybrid_target)
        self.hybrid_target_source = str(hybrid_target_source).lower()
        assert self.hybrid_target_source in {"auto","cmaq","cams"}, \
            "hybrid_target_source must be one of {'auto','cmaq','cams'}"
        self.use_flow = bool(use_flow)

        # Generate dates with leap-year handling
        self.dates = self._generate_safe_dates()

        # Build the sample list
        self.training_samples = []
        for i, date in enumerate(self.dates):
            date_str = date.strftime("%Y-%m-%d")
            self.training_samples.append({
                "type": "same",
                "date": date_str,
                "input_hours": ["00", "06"],
                "target_hour": "12"
            })
            self.training_samples.append({
                "type": "same",
                "date": date_str,
                "input_hours": ["06", "12"],
                "target_hour": "18"
            })
            if i < len(self.dates) - 1:
                next_date_str = self.dates[i + 1].strftime("%Y-%m-%d")
                self.training_samples.append({
                    "type": "cross",
                    "date": date_str,
                    "next_date": next_date_str,
                    "input_hours": ["12", "18"],
                    "target_hour": "00"
                })
                self.training_samples.append({
                    "type": "cross",
                    "date": date_str,
                    "next_date": next_date_str,
                    "input_hours": ["18", "00"],
                    "target_hour": "06"
                })
        self.training_samples = [s for s in self.training_samples
                                 if self._has_all_inputs(s)]

    # ------------------------------- File checks ----------------------------- #
    def _cmaq_files_exist(self, ts: str) -> bool:
        date_str, hh = ts[:8], ts[8:]
        yy, mm, dd = date_str[:4], date_str[4:6], date_str[6:]
        base = self.cmaq_root / yy / mm / dd / "NIER_27_01"
        return (base / f"{date_str}_x_conc.npy").exists() \
        and (base / f"{date_str}_x_metcro2d.npy").exists() \
        and (base / f"{date_str}_x_metcro3d.npy").exists()

    def _era5_files_exist(self, date_str: str) -> bool:
        static_path = self.data_dir / f"{date_str}-static.nc"
        surf_path   = self.data_dir / f"{date_str}-surface-level.nc"
        atmos_path  = self.data_dir / f"{date_str}-atmospheric.nc"
        return static_path.exists() and surf_path.exists() and atmos_path.exists()

    def _cams_surface_exists(self, date_str: str) -> bool:
        """Check CAMS surface file for this date. We only need surface for PM."""
        f = self.cams_dir / f"{date_str}-cams-surface-level.nc"
        return f.exists()

    # ----------------------------- Availability gate ------------------------ #
    def _has_all_inputs(self, s: dict) -> bool:
        try:
            if s["type"] == "same":
                d = s["date"]; h0, h1 = s["input_hours"]; ht = s["target_hour"]
                # obs
                if not Path(f"{self.npz_path}/{d.replace('-','')}{h0}_obs.npz").exists(): return False
                if not Path(f"{self.npz_path}/{d.replace('-','')}{h1}_obs.npz").exists(): return False
                # obs target sequence (horizon)
                for off in range(self.horizon):
                    dd, hh = self._shift(d, ht, off)
                    if not Path(f"{self.npz_path}/{dd.replace('-','')}{hh}_obs.npz").exists(): return False

                # cmaq
                if "cmaq" in self.sources:
                    ts0 = f"{d.replace('-','')}{h0}"
                    ts1 = f"{d.replace('-','')}{h1}"
                    if not self._cmaq_files_exist(ts0): return False
                    if not self._cmaq_files_exist(ts1): return False
                    for off in range(self.horizon):
                        dd, hh = self._shift(d, ht, off)
                        if not self._cmaq_files_exist(f"{dd.replace('-','')}{hh}"): return False
                # era5
                if "era5" in self.sources:
                    if not self._era5_files_exist(d): return False
                # cams
                if "cams" in self.sources:
                    # inputs are in day d; targets may spill to later days → check each day we touch
                    if not self._cams_surface_exists(d): return False
                    for off in range(self.horizon):
                        dd, _ = self._shift(d, ht, off)
                        if not self._cams_surface_exists(dd): return False

                return True

            else:
                d0, d1 = s["date"], s["next_date"]
                h0, h1 = s["input_hours"]; ht = s["target_hour"]
                if not Path(f"{self.npz_path}/{d0.replace('-','')}{h0}_obs.npz").exists(): return False
                if not Path(f"{self.npz_path}/{d1.replace('-','')}{h1}_obs.npz").exists(): return False
                for off in range(self.horizon):
                    dd, hh = self._shift(d1, ht, off)
                    if not Path(f"{self.npz_path}/{dd.replace('-','')}{hh}_obs.npz").exists(): return False
                if "cmaq" in self.sources:
                    ts0 = f"{d0.replace('-','')}{h0}"
                    ts1 = f"{d1.replace('-','')}{h1}"
                    if not self._cmaq_files_exist(ts0): return False
                    if not self._cmaq_files_exist(ts1): return False
                    for off in range(self.horizon):
                        dd, hh = self._shift(d1, ht, off)
                        if not self._cmaq_files_exist(f"{dd.replace('-','')}{hh}"): return False
                if "era5" in self.sources:
                    if not self._era5_files_exist(d0): return False
                    if not self._era5_files_exist(d1): return False
                if "cams" in self.sources:
                    # cross: inputs might be split across d0 and d1; targets start on d1
                    if not self._cams_surface_exists(d0): return False
                    if not self._cams_surface_exists(d1): return False
                    for off in range(self.horizon):
                        dd, _ = self._shift(d1, ht, off)
                        if not self._cams_surface_exists(dd): return False

                return True
        except Exception:
            return False

    # ------------------------------- ERA5 I/O -------------------------------- #
    def _load_data_for_date(self, date_str):
        static_path = self.data_dir / f"{date_str}-static.nc"
        surf_path   = self.data_dir / f"{date_str}-surface-level.nc"
        atmos_path  = self.data_dir / f"{date_str}-atmospheric.nc"
        
        # Check if the three ERA5 files exist
        if not (static_path.exists() and surf_path.exists() and atmos_path.exists()):
            return None, None, None
        
        # Try to open them (handle the case of a corrupted netcdf)
        try:
            static_ds = xr.open_dataset(static_path, engine="netcdf4")
            surf_ds   = xr.open_dataset(surf_path, engine="netcdf4")
            atmos_ds  = xr.open_dataset(atmos_path, engine="netcdf4")
        except OSError:
            return None, None, None
        
        return static_ds, surf_ds, atmos_ds

    # ------------------------------- OBS I/O --------------------------------- #
    def _load_obs_with_date(self, date_str, hour):
        try:
            date_nodash = date_str.replace('-', '')
            date_hour = f"{date_nodash}{hour}"
            npz_path = f"{self.npz_path}/{date_hour}_obs.npz"
            obs = np.load(npz_path)
            return obs['obs_channels']
        except:
            return None

    # ------------------------------- Time utils ------------------------------ #
    def _slice_time(self, arr: np.ndarray, hh: str) -> np.ndarray:
        """arr: (5,T?,Z?,H,W) → (…,H,W)  00 / 06 / 12 / 18 / 24(hh)"""
        h2i = {"00": 0, "06": 1, "12": 2, "18": 3, "24": 4}
        return arr[h2i[hh]]
    
    def _cmaq_plev(self, Z: int) -> tuple[int, ...]:
        """Return a tuple of pressure levels that matches Z."""
        base = (1013, 925, 850, 700, 500)
        if Z <= len(base):
            return base[:Z]
        return tuple([base[0]] * Z)

    def _shift(self, date_str: str, hh: str, offset: int) -> tuple[str, str]:
        """Translate (date, hour, offset in 6 h) → new (date, hour)."""
        t0 = datetime.strptime(f"{date_str} {hh}", "%Y-%m-%d %H")
        t  = t0 + timedelta(hours=6 * offset)
        return t.strftime("%Y-%m-%d"), t.strftime("%H")

    def _generate_safe_dates(self):
        """Generate dates with proper leap year handling."""
        all_dates = []
        current_date = self.start_date
        
        while current_date <= self.end_date:
            if current_date.month == 2 and current_date.day == 29:
                if self.handle_leap_year == 'skip':
                    print(f"Skipping leap day: {current_date.strftime('%Y-%m-%d')}")
                    current_date += timedelta(days=1)
                    continue
                elif self.handle_leap_year == 'replace_feb28':
                    adjusted_date = current_date.replace(day=28)
                    all_dates.append(adjusted_date)
                    print(f"Replaced {current_date.strftime('%Y-%m-%d')} with {adjusted_date.strftime('%Y-%m-%d')}")
                elif self.handle_leap_year == 'replace_mar1':
                    adjusted_date = current_date.replace(month=3, day=1)
                    all_dates.append(adjusted_date)
                    print(f"Replaced {current_date.strftime('%Y-%m-%d')} with {adjusted_date.strftime('%Y-%m-%d')}")
                elif self.handle_leap_year == 'keep':
                    all_dates.append(current_date)
                    print(f"Warning: Keeping leap day {current_date.strftime('%Y-%m-%d')} - ensure data files exist")
            else:
                all_dates.append(current_date)
            
            current_date += timedelta(days=1)
        
        return all_dates
    
    def __len__(self):
        return len(self.training_samples)

    # ------------------------------- CMAQ I/O -------------------------------- #
    def _load_cmaq(self, ts: str, target_h: int | None = None, target_w: int | None = None) -> dict | None:
        """
        Robustly load CMAQ arrays and keep the true number of vertical levels (Z).
        Returns:
        {
            "conc":  (24*Zc, H, W)  float32,
            "m2d":   (33, H, W)     float32,
            "m3d":   (8*Zm, H, W)   float32,
            "Z_conc": int,
            "Z_m3d":  int,
            "plev":   tuple[int,...]
        }
        """
        date_str, hh = ts[:8], ts[8:]
        yy, mm, dd = date_str[:4], date_str[4:6], date_str[6:]

        for grid in ("NIER_27_01",):
            base = self.cmaq_root / yy / mm / dd / grid
            conc_f = base / f"{date_str}_x_conc.npy"
            m2d_f  = base / f"{date_str}_x_metcro2d.npy"
            m3d_f  = base / f"{date_str}_x_metcro3d.npy"
            if not (conc_f.exists() and m2d_f.exists() and m3d_f.exists()):
                continue

            try:
                conc = np.load(conc_f, mmap_mode="r")
                m2d  = np.load(m2d_f , mmap_mode="r")
                m3d  = np.load(m3d_f , mmap_mode="r")
            except Exception as e:
                print(f"[WARN] failed reading {base}: {e}")
                return None

            conc_t = self._slice_time(conc, hh)   # -> (n_spec, Zc, H, W)
            m2d_t  = self._slice_time(m2d,  hh)   # -> (33, 1, H, W) or (33, H, W)
            m3d_t  = self._slice_time(m3d,  hh)   # -> (8, Zm, H, W)

            # --- Resizing to match OBS H/W if requested ---
            if target_h is not None and target_w is not None:
                orig_h, orig_w = conc_t.shape[-2:]
                if (orig_h, orig_w) != (target_h, target_w):
                    zoom_h = target_h / orig_h
                    zoom_w = target_w / orig_w
                    conc_t = zoom(conc_t, (1, 1, zoom_h, zoom_w), order=1)
                    m3d_t  = zoom(m3d_t,  (1, 1, zoom_h, zoom_w), order=1)
                    if m2d_t.ndim == 4:
                        m2d_t = zoom(m2d_t, (1, 1, zoom_h, zoom_w), order=1)
                    elif m2d_t.ndim == 3:
                        m2d_t = zoom(m2d_t, (1, zoom_h, zoom_w), order=1)

            assert conc_t.ndim == 4, f"unexpected conc shape: {conc_t.shape}"
            n_spec, Zc, H_new, W_new = conc_t.shape
            if m2d_t.ndim == 4:
                m2d_t = m2d_t.squeeze(1)
            n_m3d, Zm, _, _ = m3d_t.shape
            
            conc_flat = conc_t.reshape(n_spec * Zc, H_new, W_new).astype(np.float32)
            m3d_flat  = m3d_t.reshape(n_m3d * Zm, H_new, W_new).astype(np.float32)
            m2d_t     = m2d_t.astype(np.float32)
            
            Z_for_meta = Zc if Zc > 0 else Zm
            plev = self._cmaq_plev(Z_for_meta)

            return {
                "conc": conc_flat, "m2d": m2d_t, "m3d": m3d_flat,
                "Z_conc": Zc, "Z_m3d": Zm, "plev": plev
            }

        return None

    def _load_cmaq_sequence(self, start_date: str, start_hh: str,
                            target_h: int | None = None, target_w: int | None = None) -> dict | None:
        """Stack CMAQ sequences across horizon keeping Z info."""
        conc_seq, m2d_seq, m3d_seq = [], [], []
        Zc, Zm, plev = None, None, None

        for off in range(self.horizon):
            d, h = self._shift(start_date, start_hh, off)
            ts   = f"{d.replace('-', '')}{h}"
            c    = self._load_cmaq(ts, target_h=target_h, target_w=target_w)
            if c is None:
                return None
            if Zc is None:
                Zc, Zm, plev = c["Z_conc"], c["Z_m3d"], c["plev"]
            else:
                assert (Zc, Zm, plev) == (c["Z_conc"], c["Z_m3d"], c["plev"]), "CMAQ Z/plev changed within horizon"
            conc_seq.append(c["conc"])
            m2d_seq .append(c["m2d"])
            m3d_seq .append(c["m3d"])

        return dict(
            conc=np.stack(conc_seq, axis=0),   # (T, 24*Zc, H, W)
            m2d =np.stack(m2d_seq , axis=0),   # (T, 33,     H, W)
            m3d =np.stack(m3d_seq , axis=0),   # (T, 8*Zm,   H, W)
            Z_conc=Zc, Z_m3d=Zm, plev=plev
        )

    # ------------------------------- CAMS I/O -------------------------------- #
    def _open_cams_surface(self, date_str: str) -> Optional[xr.Dataset]:
        fn = self.cams_dir / f"{date_str}-cams-surface-level.nc"
        if not fn.exists():
            return None
        # Try to decode CF/timedelta so that time/valid_time are materialized
        for decode_td in (True, False):
            try:
                ds = xr.open_dataset(fn, engine="netcdf4", decode_timedelta=decode_td)
                break
            except Exception as e:
                if decode_td:
                    # try again without decode_timedelta
                    continue
                print(f"[WARN] Failed to open CAMS surface: {fn} ({e})")
                return None
        # If forecast_period exists, select analysis slice (0) but keep time axis
        # (some files are [forecast_reference_time, time, forecast_period, lat, lon])
        if "forecast_period" in ds.dims or "forecast_period" in ds.coords:
            try:
                ds = ds.isel(forecast_period=0)
            except Exception:
                pass
        return ds

    def _cams_nearest_time_index(self, ds: xr.Dataset, ts: datetime) -> int:
        """
        Return index of the coordinate closest to ts. Supports:
        - time (datetime64 or numeric with units)
        - valid_time (datetime64)
        - forecast_reference_time + forecast_period (datetime64 + timedelta / numeric with units)
        """
        ts64 = np.datetime64(ts, "s")

        def _numeric_with_units_to_times(var):
            units = (var.attrs.get("units", "") or "").lower()
            vals = np.asarray(var.values)
            if "since" in units:
                base = units.split(" since ")[1]
                base64 = np.datetime64(base)
            else:
                # If no explicit base, assume seconds since UNIX epoch
                base64 = np.datetime64("1970-01-01T00:00:00")
            if "hour" in units:
                td = (vals.astype(np.float64) * 3600.0).astype("timedelta64[s]")
            elif "minute" in units:
                td = (vals.astype(np.float64) * 60.0).astype("timedelta64[s]")
            elif "second" in units:
                try:
                    td = vals.astype("timedelta64[s]")
                except Exception:
                    td = (vals.astype(np.float64)).astype("timedelta64[s]")
            else:
                # fallback: hours
                td = (vals.astype(np.float64) * 3600.0).astype("timedelta64[s]")
            return base64 + td

        # 0) Prefer a proper datetime64 'time'
        if "time" in ds.coords:
            tvar = ds["time"]
            try:
                times = np.asarray(tvar.values).astype("datetime64[s]")
            except Exception:
                times = _numeric_with_units_to_times(tvar)
            return int(np.argmin(np.abs(times - ts64)))

        # 1) valid_time as full datetimes
        if "valid_time" in ds.coords:
            vt = np.asarray(ds["valid_time"].values).astype("datetime64[s]")
            return int(np.argmin(np.abs(vt - ts64)))

        # 2) Reconstruct from forecast_reference_time + time/forecast_period
        fr_key = "forecast_reference_time" if "forecast_reference_time" in ds.coords else None
        if fr_key is not None:
            fr = np.asarray(ds[fr_key].values)
            # handle scalar vs vector
            if np.ndim(fr) == 0:
                fr = np.array([fr])
            try:
                fr_dt = fr.astype("datetime64[s]")
            except Exception:
                # some files store numeric with units on FR too
                fr_dt = _numeric_with_units_to_times(ds[fr_key])

            # candidate lead coordinate names
            lead_key = "time" if "time" in ds.coords else ("forecast_period" if "forecast_period" in ds.coords else None)
            if lead_key is not None:
                lead = ds[lead_key]
                try:
                    lead_td = lead.values.astype("timedelta64[s]")
                except Exception:
                    # numeric with units
                    lead_td = _numeric_with_units_to_times(lead) - np.datetime64("1970-01-01T00:00:00")
                # broadcast: common CAMS layout is [forecast_reference_time, time, lat, lon]
                # assume single FR after isel(forecast_period=0); if multiple FRs, choose the last
                fr_sel = fr_dt[-1] if fr_dt.size > 1 else fr_dt[0]
                times = fr_sel + lead_td.astype("timedelta64[s]")
                times = np.asarray(times).astype("datetime64[s]")
                return int(np.argmin(np.abs(times - ts64)))

        # 3) As a last resort: any 1D datetime-like coord
        for cand in ("analysis_time", "step", "init_time"):
            if cand in ds.coords:
                try:
                    tt = np.asarray(ds[cand].values).astype("datetime64[s]")
                    return int(np.argmin(np.abs(tt - ts64)))
                except Exception:
                    pass

        raise ValueError("Cannot resolve CAMS time index: no usable time coordinate found.")


    def _cams_crop_and_resize(self, ds: xr.Dataset, arr2d: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        lat_name = "latitude" if "latitude" in ds.coords else ("lat" if "lat" in ds.coords else None)
        lon_name = "longitude" if "longitude" in ds.coords else ("lon" if "lon" in ds.coords else None)
        if lat_name is None or lon_name is None:
            raise ValueError("CAMS dataset has no latitude/longitude coordinates.")
        lat1d = np.asarray(ds[lat_name].values)
        lon1d = np.asarray(ds[lon_name].values)
        roi = _box_to_slices(lat1d, lon1d, _EA_SOUTH, _EA_WEST, _EA_NORTH, _EA_EAST)
        cropped = np.asarray(arr2d)[roi]
        return _safe_zoom2d(cropped, (target_h, target_w))

    def _load_cams_slice(self, date_str: str, hour: str, target_h: int, target_w: int) -> Optional[Dict[str, np.ndarray]]:
        ds = self._open_cams_surface(date_str)
        if ds is None:
            return None
        try:
            idx = self._cams_nearest_time_index(ds, datetime.strptime(f"{date_str} {hour}", "%Y-%m-%d %H"))
            out = {}
            for key in ("pm2p5", "pm10"):
                if key not in ds.variables:
                    print(f"[WARN] CAMS surface missing variable '{key}' on {date_str} (skip).")
                    ds.close(); return None
                da = ds[key].isel(time=idx) if "time" in ds[key].coords else ds[key].isel(valid_time=idx) \
                    if "valid_time" in ds[key].coords else ds[key].isel(step=idx) if "step" in ds[key].coords else ds[key]
                scale = _infer_to_ug_scale_from_units(da.attrs.get("units", ""))
                arr = np.asarray(da.values)
                arr = self._cams_crop_and_resize(ds, arr, target_h, target_w)
                out[key] = arr.astype(np.float32) * float(scale)
            ds.close()
            return out
        except Exception as e:
            # print(f"[WARN] Failed to load CAMS slice {date_str} {hour}: {e}")
            try: ds.close()
            except Exception: pass
            return None

    def _load_cams_sequence(self, start_date: str, start_hh: str,
                            target_h: int, target_w: int) -> Optional[Dict[str, np.ndarray]]:
        """
        Build horizon sequences for CAMS pm2p5/pm10 aligned to (start_date, start_hh).
        Returns dict with arrays of shape (T, H, W).
        """
        seq25, seq10 = [], []
        for off in range(self.horizon):
            d, h = self._shift(start_date, start_hh, off)
            sl = self._load_cams_slice(d, h, target_h=target_h, target_w=target_w)
            if sl is None:
                return None
            seq25.append(sl["pm2p5"])
            seq10.append(sl["pm10"])
        return {"pm2p5": np.stack(seq25, axis=0), "pm10": np.stack(seq10, axis=0)}

    # ----------------------------- FLOW (unchanged) -------------------------- #
    def _is_leap_year(self, year):
        return calendar.isleap(year)

    def _load_flow_with_date(self, date, hour):
        year, month, day = date.split("-")
        try:
            flow_path = self.flow_root / f"{year}{month}{day}{hour}"
            flow_pm25_data = np.load(f"{flow_path}_pm25_flow.npy").transpose(2, 0, 1) / 10
            flow_pm25_mask = np.load(f"{flow_path}_pm25_mask.npy")[None, None, ...]
            flow_pm10_data = np.load(f"{flow_path}_pm10_flow.npy").transpose(2, 0, 1) / 10
            flow_pm10_mask = np.load(f"{flow_path}_pm10_mask.npy")[None, None, ...]

            return flow_pm25_data, flow_pm25_mask, flow_pm10_data, flow_pm10_mask
        except:
            return None

    # ============================= __getitem__ =============================== #
    def __getitem__(self, idx):
        sample = self.training_samples[idx]
        date = sample["date"]
        h2i = HOUR2IDX

        if sample["type"] == "same":
            # ------------------- Load OBS -------------------------------------
            obs_input0 = self._load_obs_with_date(date, sample["input_hours"][0])
            obs_input1 = self._load_obs_with_date(date, sample["input_hours"][1])
            obs_target_seq = self._load_target_sequence(date, sample["target_hour"])
            if obs_input0 is None or obs_input1 is None or obs_target_seq is None:
                return None, None
            C, H, W = obs_input0.shape
            lat, lon = make_lat_lon(H, W)

            if "obs" in self.sources:
                obs_concat = np.stack([obs_input0, obs_input1], axis=1)  # (C,2,H,W)

                # --- Masking augmentation (OBS inputs only) -------------------
                if self.use_masking and random.random() < 0.5:
                    pollutant_indices = [0, 1, 2, 3, 4, 5]  # pm2p5, pm10, so2, no2, o3, co
                    input_pollutants = obs_concat[pollutant_indices, :]
                    mask = torch.rand(input_pollutants.shape) >= self.mask_ratio
                    input_pollutants = torch.from_numpy(input_pollutants).float() * mask.float()
                    obs_concat[pollutant_indices, :] = input_pollutants.numpy()

                if obs_target_seq is None:
                    return None, None

            # --- CutMix with CMAQ before assembly (optional) -----------------
            if self.use_cutmix:
                ts0 = f"{date.replace('-', '')}{sample['input_hours'][0]}"
                ts1 = f"{date.replace('-', '')}{sample['input_hours'][1]}"
                c0  = self._load_cmaq(ts0, target_h=H, target_w=W)
                c1  = self._load_cmaq(ts1, target_h=H, target_w=W)
                if c0 is not None and c1 is not None:
                    Zc = max(1, int(c0["Z_conc"]))
                    obs_mask_t0 = obs_concat[0, 0] > 0
                    obs_mask_t1 = obs_concat[0, 1] > 0
                    CMAQ_VAR_MAP = {0:'PM2P5', 1:'PM10', 2:'tcso2', 3:'tcno2', 4:'O3', 5:'tcco'}
                    for obs_idx, cmaq_var_name in CMAQ_VAR_MAP.items():
                        try:
                            var_j = CONC_VARS.index(cmaq_var_name)
                            start = var_j * Zc
                            cmaq_t0 = c0['conc'][start]
                            cmaq_t1 = c1['conc'][start]
                            obs_concat[obs_idx, 0] = np.where(obs_mask_t0, obs_concat[obs_idx, 0], cmaq_t0)
                            obs_concat[obs_idx, 1] = np.where(obs_mask_t1, obs_concat[obs_idx, 1], cmaq_t1)
                        except Exception as e:
                            print(f"[WARN] CutMix skip {cmaq_var_name}: {e}")

            # --- Flow fields (optional) --------------------------------------
            if self.use_flow:
                now_flow_pm25_data, now_flow_pm25_mask, now_flow_pm10_data, now_flow_pm10_mask = self._load_flow_with_date(date, sample["input_hours"][0])
                next_flow_pm25_data, next_flow_pm25_mask, next_flow_pm10_data, next_flow_pm10_mask = self._load_flow_with_date(date, sample["input_hours"][1])
                flow_vars_input = {
                    "flowpm2p5x": torch.from_numpy(now_flow_pm25_data[0][None, None, ...]),
                    "flowpm2p5y": torch.from_numpy(now_flow_pm25_data[1][None, None, ...]),
                    "maskpm2p5": torch.from_numpy(now_flow_pm25_mask),
                    "flowpm10x": torch.from_numpy(now_flow_pm10_data[0][None, None, ...]),
                    "flowpm10y": torch.from_numpy(now_flow_pm10_data[1][None, None, ...]),
                    "maskpm10": torch.from_numpy(now_flow_pm10_mask),
                }
                flow_vars_target = {
                    "flowpm2p5x": torch.from_numpy(next_flow_pm25_data[0][None, None, ...]),
                    "flowpm2p5y": torch.from_numpy(next_flow_pm25_data[1][None, None, ...]),
                    "maskpm2p5": torch.from_numpy(next_flow_pm25_mask),
                    "flowpm10x": torch.from_numpy(next_flow_pm10_data[0][None, None, ...]),
                    "flowpm10y": torch.from_numpy(next_flow_pm10_data[1][None, None, ...]),
                    "maskpm10": torch.from_numpy(next_flow_pm10_mask),
                }

            # ----------------- Assemble surface targets from OBS --------------
            surf_vars_input = {
                "pm2p5": torch.from_numpy(obs_concat[0][None]),
                "pm10":  torch.from_numpy(obs_concat[1][None]),
                "tcso2": torch.from_numpy(obs_concat[2][None]),
                "tcno2": torch.from_numpy(obs_concat[3][None]),
                "o3":    torch.from_numpy(obs_concat[4][None]),
                "tcco":  torch.from_numpy(obs_concat[5][None]),
            }
            surf_vars_target = {
                "pm2p5": torch.from_numpy(obs_target_seq[0][None]),
                "pm10" : torch.from_numpy(obs_target_seq[1][None]),
                "tcso2": torch.from_numpy(obs_target_seq[2][None]),
                "tcno2": torch.from_numpy(obs_target_seq[3][None]),
                "o3"   : torch.from_numpy(obs_target_seq[4][None]),
                "tcco" : torch.from_numpy(obs_target_seq[5][None]),
            }
            atmos_vars_input = {}
            atmos_vars_target = {}
            static_vars = {}
            atmos_levels = tuple()
            target_time = datetime.strptime(f"{date} {sample['target_hour']}", "%Y-%m-%d %H")

            # ----------------- Hybrid target (CMAQ/CAMS fallback) -------------
            target_date = date  # for "same" samples
            if self.use_hybrid_target:
                prefer = self.hybrid_target_source
                used = None
                # Prefer CMAQ if requested/available; otherwise CAMS
                if (prefer in {"auto","cmaq"}) and ("cmaq" in self.sources):
                    c_tgt = self._load_cmaq_sequence(target_date, sample["target_hour"], target_h=H, target_w=W)
                    if c_tgt is not None:
                        used = "cmaq"
                        Zc = max(1, int(c_tgt.get("Z_conc", 1)))
                        PM_VARS_MAP = {'pm2p5': 'PM2P5', 'pm10': 'PM10'}
                        for key, cmaq_name in PM_VARS_MAP.items():
                            try:
                                var_j = CONC_VARS.index(cmaq_name)
                                start = var_j * Zc
                                cmaq_target_seq = c_tgt['conc'][:, start]  # (T,H,W)
                                obs_idx = {'pm2p5': 0, 'pm10': 1}[key]
                                obs_seq = obs_target_seq[obs_idx]         # (T,H,W)
                                obs_mask_seq = (obs_seq > 0)
                                hybrid_target_seq = np.where(obs_mask_seq, obs_seq, cmaq_target_seq)
                                surf_vars_target[key] = torch.from_numpy(hybrid_target_seq[None, ...]).float()
                            except Exception as e:
                                print(f"[WARN] Hybrid (CMAQ) fail {key}: {e}")
                if used is None and ((prefer in {"auto","cams"}) and ("cams" in self.sources)):
                    cams_tgt = self._load_cams_sequence(target_date, sample["target_hour"], target_h=H, target_w=W)
                    if cams_tgt is not None:
                        used = "cams"
                        for key in ("pm2p5", "pm10"):
                            try:
                                obs_idx = 0 if key == "pm2p5" else 1
                                obs_seq = obs_target_seq[obs_idx]       # (T,H,W)
                                cams_seq = cams_tgt[key]                 # (T,H,W)
                                obs_mask_seq = (obs_seq > 0)
                                hybrid_target_seq = np.where(obs_mask_seq, obs_seq, cams_seq)
                                surf_vars_target[key] = torch.from_numpy(hybrid_target_seq[None, ...]).float()
                            except Exception as e:
                                print(f"[WARN] Hybrid (CAMS) fail {key}: {e}")

            # ----------------- ERA5 (optional, unchanged) ----------------------
            if "era5" in self.sources:
                static_ds, surf_ds, atmos_ds = self._load_data_for_date(date)
                if static_ds is None or surf_ds is None or atmos_ds is None:
                    return None, None

                def resize_array(arr, target_h, target_w):
                    original_h, original_w = arr.shape[-2], arr.shape[-1]
                    if original_h == target_h and original_w == target_w:
                        return arr
                    zoom_h = target_h / original_h
                    zoom_w = target_w / original_w
                    zoom_factors = [1] * (arr.ndim - 2) + [zoom_h, zoom_w]
                    return zoom(arr, zoom_factors, order=1)

                t0, t1 = sample["input_hours"]
                idx0, idx1 = h2i[t0], h2i[t1]

                surf_vars_input.update({
                    "2t":  torch.from_numpy(resize_array(surf_ds["t2m"].values[[idx0, idx1]], H, W))[None],
                    "10u": torch.from_numpy(resize_array(surf_ds["u10"].values[[idx0, idx1]], H, W))[None],
                    "10v": torch.from_numpy(resize_array(surf_ds["v10"].values[[idx0, idx1]], H, W))[None],
                    "msl": torch.from_numpy(resize_array(surf_ds["msl"].values[[idx0, idx1]], H, W))[None],
                })
                atmos_vars_input.update({
                    "t": torch.from_numpy(resize_array(atmos_ds["t"].values[[idx0, idx1]], H, W))[None],
                    "u": torch.from_numpy(resize_array(atmos_ds["u"].values[[idx0, idx1]], H, W))[None],
                    "v": torch.from_numpy(resize_array(atmos_ds["v"].values[[idx0, idx1]], H, W))[None],
                    "q": torch.from_numpy(resize_array(atmos_ds["q"].values[[idx0, idx1]], H, W))[None],
                    "z": torch.from_numpy(resize_array(atmos_ds["z"].values[[idx0, idx1]], H, W))[None],
                })
                static_vars.update({
                    "z":   torch.from_numpy(resize_array(static_ds["z"].values[0], H, W)),
                    "slt": torch.from_numpy(resize_array(static_ds["slt"].values[0], H, W)),
                    "lsm": torch.from_numpy(resize_array(static_ds["lsm"].values[0], H, W)),
                })

                if self.use_wind_prompt:
                    u0_resized = resize_array(surf_ds["u10"].values[idx0], H, W)
                    v0_resized = resize_array(surf_ds["v10"].values[idx0], H, W)
                    u1_resized = resize_array(surf_ds["u10"].values[idx1], H, W)
                    v1_resized = resize_array(surf_ds["v10"].values[idx1], H, W)
                    wind_prompt_t0 = wind_to_image(u0_resized, v0_resized, H, W)
                    wind_prompt_t1 = wind_to_image(u1_resized, v1_resized, H, W)
                    wind_prompts = np.stack([wind_prompt_t0, wind_prompt_t1], axis=0) / 255.0
                    surf_vars_input['wind_prompt'] = torch.from_numpy(wind_prompts[np.newaxis, ...]).float()

                idx_target = h2i[sample["target_hour"]]
                target_slice = slice(idx_target, idx_target + self.horizon)
                surf_vars_target.update({
                    "2t":  torch.from_numpy(resize_array(surf_ds["t2m"].values[target_slice], H, W))[None],
                    "10u": torch.from_numpy(resize_array(surf_ds["u10"].values[target_slice], H, W))[None],
                    "10v": torch.from_numpy(resize_array(surf_ds["v10"].values[target_slice], H, W))[None],
                    "msl": torch.from_numpy(resize_array(surf_ds["msl"].values[target_slice], H, W))[None],
                })
                atmos_vars_target.update({
                    "t": torch.from_numpy(resize_array(atmos_ds["t"].values[target_slice], H, W))[None],
                    "u": torch.from_numpy(resize_array(atmos_ds["u"].values[target_slice], H, W))[None],
                    "v": torch.from_numpy(resize_array(atmos_ds["v"].values[target_slice], H, W))[None],
                    "q": torch.from_numpy(resize_array(atmos_ds["q"].values[target_slice], H, W))[None],
                    "z": torch.from_numpy(resize_array(atmos_ds["z"].values[target_slice], H, W))[None],
                })
                atmos_levels = tuple(int(level) for level in atmos_ds.pressure_level.values)

            # ----------------- CMAQ (optional, unchanged semantics) -----------
            if "cmaq" in self.sources or self.use_cutmix:
                t0, t1 = sample["input_hours"]
                ts0 = f"{date.replace('-', '')}{t0}"
                ts1 = f"{date.replace('-', '')}{t1}"
                c0 = self._load_cmaq(ts0, target_h=H, target_w=W)
                c1 = self._load_cmaq(ts1, target_h=H, target_w=W)
                if c0 is None or c1 is None:
                    return None, None
                
                Zc, Zm, plev = c0["Z_conc"], c0["Z_m3d"], c0["plev"]
                assert (Zc, Zm, plev) == (c1["Z_conc"], c1["Z_m3d"], c1["plev"]), "Mismatched CMAQ Z between input hours"

                if "cmaq" in self.sources:
                    conc_vars_to_use = CONC_VARS
                    m2d_vars_to_use = M2D_VARS
                    m3d_vars_to_use = M3D_VARS
                    if self.use_cmaq_pm_only:
                        conc_vars_to_use = ["PM2P5", "PM10"]
                        m2d_vars_to_use = []
                        m3d_vars_to_use = []

                    if self.use_cmaq_m2d:
                        for i, name in enumerate(m2d_vars_to_use):
                            surf_vars_input[f"{name.lower()}_cmaq"] = torch.from_numpy(
                                np.stack([c0["m2d"][M2D_VARS.index(name)], c1["m2d"][M2D_VARS.index(name)]])[None]
                            )

                    if self.use_cmaq_conc and Zc > 0:
                        for base in conc_vars_to_use:
                            original_j = CONC_VARS.index(base)
                            start = original_j * Zc
                            if Zc == 1:
                                sname = f"{base.lower()}_cmaq"
                                v0 = c0["conc"][start]
                                v1 = c1["conc"][start]
                                surf_vars_input[sname] = torch.from_numpy(np.stack([v0, v1])[None])
                            else:
                                lev0 = c0["conc"][start:start+Zc]
                                lev1 = c1["conc"][start:start+Zc]
                                stacked = np.stack([lev0, lev1], axis=0)
                                atmos_vars_input[f"{base.lower()}_cmaq"] = torch.from_numpy(stacked[None])

                    if self.use_cmaq_m3d and Zm > 0:
                        for base in m3d_vars_to_use:
                            original_j = M3D_VARS.index(base)
                            start = original_j * Zm
                            lev0 = c0["m3d"][start:start+Zm]
                            lev1 = c1["m3d"][start:start+Zm]
                            stacked = np.stack([lev0, lev1], axis=0)
                            atmos_vars_input[f"{base.lower()}_cmaq"] = torch.from_numpy(stacked[None])

                    conc_as_3d = self.use_cmaq_conc and (Zc > 1)
                    met_as_3d  = self.use_cmaq_m3d  and (Zm > 0)
                    if met_as_3d:
                        atmos_levels = self._cmaq_plev(Zm)
                    elif conc_as_3d:
                        atmos_levels = self._cmaq_plev(Zc)

                    c_tgt = self._load_cmaq_sequence(date, sample["target_hour"], target_h=H, target_w=W)
                    if c_tgt is None:
                        return None, None
                    Zc_t, Zm_t, plev_t = c_tgt["Z_conc"], c_tgt["Z_m3d"], c_tgt["plev"]
                    assert (Zc, Zm, plev) == (Zc_t, Zm_t, plev_t), "Input/target CMAQ Z mismatch"

                    if self.use_cmaq_m2d:
                        for i, name in enumerate(M2D_VARS):
                            key = f"{name.lower()}_cmaq"
                            arr = c_tgt["m2d"][:, i].astype(np.float32)
                            surf_vars_target[key] = torch.from_numpy(arr)[None]

                    if self.use_cmaq_conc and Zc > 0:
                        if Zc == 1:
                            for j, base in enumerate(CONC_VARS):
                                start = j * Zc
                                key = f"{base.lower()}_cmaq"
                                arr = c_tgt["conc"][:, start]
                                surf_vars_target[key] = torch.from_numpy(arr)[None]
                        else:
                            for j, base in enumerate(CONC_VARS):
                                start = j * Zc
                                conc_all = c_tgt["conc"][:, start:start+Zc]
                                atmos_vars_target[f"{base.lower()}_cmaq"] = torch.from_numpy(conc_all[None])

                    if self.use_cmaq_m3d and Zm > 0:
                        for j, base in enumerate(M3D_VARS):
                            start = j * Zm
                            m3d_all = c_tgt["m3d"][:, start:start+Zm]
                            atmos_vars_target[f"{base.lower()}_cmaq"] = torch.from_numpy(m3d_all[None])

            # ----------------- CAMS (NEW) ------------------------------------
            if "cams" in self.sources:
                t0, t1 = sample["input_hours"]
                cam0 = self._load_cams_slice(date, t0, target_h=H, target_w=W)
                cam1 = self._load_cams_slice(date, t1, target_h=H, target_w=W)
                if (cam0 is None) or (cam1 is None):
                    return None, None
                # Inputs: add pm2p5_cams / pm10_cams with 2 time steps
                surf_vars_input.update({
                    "pm2p5_cams": torch.from_numpy(np.stack([cam0["pm2p5"], cam1["pm2p5"]], axis=0)[None]),
                    "pm10_cams":  torch.from_numpy(np.stack([cam0["pm10"] , cam1["pm10"] ], axis=0)[None]),
                })
                # Targets: add horizon sequences
                cams_tgt = self._load_cams_sequence(date, sample["target_hour"], target_h=H, target_w=W)
                if cams_tgt is None:
                    return None, None
                surf_vars_target.update({
                    "pm2p5_cams": torch.from_numpy(cams_tgt["pm2p5"][None]),
                    "pm10_cams":  torch.from_numpy(cams_tgt["pm10" ][None]),
                })

            # ----------------- Metadata & return ------------------------------
            tgt_times = tuple(
                datetime.strptime(
                    f"{self._shift(date, sample['target_hour'], off)[0]} "
                    f"{self._shift(date, sample['target_hour'], off)[1]}",
                    "%Y-%m-%d %H")
                for off in range(self.horizon)
            )
            in_times = tuple(
                datetime.strptime(f"{date} {hh}", "%Y-%m-%d %H")
                for hh in sample["input_hours"]
            )
            meta_in  = Metadata(lat=lat, lon=lon, time=in_times, atmos_levels=atmos_levels)
            meta_tgt = Metadata(lat=lat, lon=lon, time=tgt_times, atmos_levels=atmos_levels)

            batch = Batch(surf_vars=surf_vars_input, static_vars=static_vars, atmos_vars=atmos_vars_input, metadata=meta_in)
            target_batch = Batch(surf_vars=surf_vars_target, static_vars=static_vars, atmos_vars=atmos_vars_target, metadata=meta_tgt)

            if self.use_flow:
                batch.flow_vars = flow_vars_input
                target_batch.flow_vars = flow_vars_target

            return batch, target_batch

        # ============================ CROSS samples ===========================
        elif sample["type"] == "cross":
            t0, t1 = sample["input_hours"]
            curr_date = sample["date"]
            next_date = sample["next_date"]

            # Choose correct calendar day for second input hour
            date0 = curr_date
            date1 = curr_date if HOUR2IDX[t1] > HOUR2IDX[t0] else next_date

            # ------------------- Load OBS -------------------------------------
            obs_input0 = self._load_obs_with_date(date0, t0)
            obs_input1 = self._load_obs_with_date(date1, t1)
            if obs_input0 is None or obs_input1 is None:
                return None, None
            C, H, W = obs_input0.shape
            lat, lon = make_lat_lon(H, W)

            if "obs" in self.sources:
                obs_concat = np.stack([obs_input0, obs_input1], axis=1)
                obs_target_seq = self._load_target_sequence(next_date, sample["target_hour"])
                if self.use_masking and random.random() < 0.5:
                    pollutant_indices = [0, 1, 2, 3, 4, 5]
                    input_pollutants = obs_concat[pollutant_indices, :]
                    mask = torch.rand(input_pollutants.shape) >= self.mask_ratio
                    input_pollutants = torch.from_numpy(input_pollutants).float() * mask.float()
                    obs_concat[pollutant_indices, :] = input_pollutants.numpy()

                if obs_target_seq is None:
                    return None, None

            # --- CutMix with CMAQ before assembly (optional) -----------------
            if self.use_cutmix:
                ts0 = f"{date0.replace('-', '')}{t0}"
                ts1 = f"{date1.replace('-', '')}{t1}"
                c0  = self._load_cmaq(ts0, target_h=H, target_w=W)
                c1  = self._load_cmaq(ts1, target_h=H, target_w=W)
                if c0 is not None and c1 is not None:
                    Zc = max(1, int(c0["Z_conc"]))
                    obs_mask_t0 = obs_concat[0, 0] > 0
                    obs_mask_t1 = obs_concat[0, 1] > 0
                    CMAQ_VAR_MAP = {0:'PM2P5', 1:'PM10', 2:'tcso2', 3:'tcno2', 4:'O3', 5:'tcco'}
                    for obs_idx, cmaq_var_name in CMAQ_VAR_MAP.items():
                        try:
                            var_j = CONC_VARS.index(cmaq_var_name)
                            start = var_j * Zc
                            cmaq_t0 = c0['conc'][start]
                            cmaq_t1 = c1['conc'][start]
                            obs_concat[obs_idx, 0] = np.where(obs_mask_t0, obs_concat[obs_idx, 0], cmaq_t0)
                            obs_concat[obs_idx, 1] = np.where(obs_mask_t1, obs_concat[obs_idx, 1], cmaq_t1)
                        except Exception as e:
                            print(f"[WARN] CutMix skip {cmaq_var_name}: {e}")

            if self.use_flow:
                now_flow_pm25_data, now_flow_pm25_mask, now_flow_pm10_data, now_flow_pm10_mask = self._load_flow_with_date(date0, t0)
                next_flow_pm25_data, next_flow_pm25_mask, next_flow_pm10_data, next_flow_pm10_mask = self._load_flow_with_date(date1, t1)
                flow_vars_input = {
                    "flowpm2p5x": torch.from_numpy(now_flow_pm25_data[0][None, None, ...]),
                    "flowpm2p5y": torch.from_numpy(now_flow_pm25_data[1][None, None, ...]),
                    "maskpm2p5": torch.from_numpy(now_flow_pm25_mask),
                    "flowpm10x": torch.from_numpy(now_flow_pm10_data[0][None, None, ...]),
                    "flowpm10y": torch.from_numpy(now_flow_pm10_data[1][None, None, ...]),
                    "maskpm10": torch.from_numpy(now_flow_pm10_mask),
                }
                flow_vars_target = {
                    "flowpm2p5x": torch.from_numpy(next_flow_pm25_data[0][None, None, ...]),
                    "flowpm2p5y": torch.from_numpy(next_flow_pm25_data[1][None, None, ...]),
                    "maskpm2p5": torch.from_numpy(next_flow_pm25_mask),
                    "flowpm10x": torch.from_numpy(next_flow_pm10_data[0][None, None, ...]),
                    "flowpm10y": torch.from_numpy(next_flow_pm10_data[1][None, None, ...]),
                    "maskpm10": torch.from_numpy(next_flow_pm10_mask),
                }

            surf_vars_input = {
                "pm2p5": torch.from_numpy(obs_concat[0][None]),
                "pm10":  torch.from_numpy(obs_concat[1][None]),
                "tcso2": torch.from_numpy(obs_concat[2][None]),
                "tcno2": torch.from_numpy(obs_concat[3][None]),
                "o3":    torch.from_numpy(obs_concat[4][None]),
                "tcco":  torch.from_numpy(obs_concat[5][None]),
            }

            surf_vars_target = {
                "pm2p5": torch.from_numpy(obs_target_seq[0][None]),
                "pm10" : torch.from_numpy(obs_target_seq[1][None]),
                "tcso2": torch.from_numpy(obs_target_seq[2][None]),
                "tcno2": torch.from_numpy(obs_target_seq[3][None]),
                "o3"   : torch.from_numpy(obs_target_seq[4][None]),
                "tcco" : torch.from_numpy(obs_target_seq[5][None]),
            }

            atmos_vars_input = {}
            atmos_vars_target = {}
            static_vars = {}
            atmos_levels = tuple()
            target_time = datetime.strptime(f"{next_date} {sample['target_hour']}", "%Y-%m-%d %H")

            # ----------------- Hybrid target (CMAQ/CAMS fallback) -------------
            target_date = next_date  # for "cross" samples
            if self.use_hybrid_target:
                prefer = self.hybrid_target_source
                used = None
                if (prefer in {"auto","cmaq"}) and ("cmaq" in self.sources):
                    c_tgt = self._load_cmaq_sequence(target_date, sample["target_hour"], target_h=H, target_w=W)
                    if c_tgt is not None:
                        used = "cmaq"
                        Zc = max(1, int(c_tgt.get("Z_conc", 1)))
                        PM_VARS_MAP = {'pm2p5': 'PM2P5', 'pm10': 'PM10'}
                        for key, cmaq_name in PM_VARS_MAP.items():
                            try:
                                var_j = CONC_VARS.index(cmaq_name)
                                start = var_j * Zc
                                cmaq_target_seq = c_tgt['conc'][:, start]
                                obs_idx = {'pm2p5': 0, 'pm10': 1}[key]
                                obs_seq = obs_target_seq[obs_idx]
                                obs_mask_seq = (obs_seq > 0)
                                hybrid_target_seq = np.where(obs_mask_seq, obs_seq, cmaq_target_seq)
                                surf_vars_target[key] = torch.from_numpy(hybrid_target_seq[None, ...]).float()
                            except Exception as e:
                                print(f"[WARN] Hybrid (CMAQ) fail {key}: {e}")
                if used is None and ((prefer in {"auto","cams"}) and ("cams" in self.sources)):
                    cams_tgt = self._load_cams_sequence(target_date, sample["target_hour"], target_h=H, target_w=W)
                    if cams_tgt is not None:
                        used = "cams"
                        for key in ("pm2p5", "pm10"):
                            try:
                                obs_idx = 0 if key == "pm2p5" else 1
                                obs_seq = obs_target_seq[obs_idx]
                                cams_seq = cams_tgt[key]
                                obs_mask_seq = (obs_seq > 0)
                                hybrid_target_seq = np.where(obs_mask_seq, obs_seq, cams_seq)
                                surf_vars_target[key] = torch.from_numpy(hybrid_target_seq[None, ...]).float()
                            except Exception as e:
                                print(f"[WARN] Hybrid (CAMS) fail {key}: {e}")

            # ----------------- ERA5 (optional, unchanged) ----------------------
            if "era5" in self.sources:
                static_ds0, surf_ds0, atmos_ds0 = self._load_data_for_date(date0)
                static_ds1, surf_ds1, atmos_ds1 = self._load_data_for_date(date1)
                static_ds_tgt, surf_ds_tgt, atmos_ds_tgt = self._load_data_for_date(next_date)
                if any(ds is None for ds in [static_ds0, surf_ds0, atmos_ds0, static_ds1, surf_ds1, atmos_ds1, static_ds_tgt, surf_ds_tgt, atmos_ds_tgt]):
                    return None, None

                def resize_array(arr, target_h, target_w):
                    original_h, original_w = arr.shape[-2], arr.shape[-1]
                    if original_h == target_h and original_w == target_w:
                        return arr
                    zoom_h = target_h / original_h
                    zoom_w = target_w / original_w
                    zoom_factors = [1] * (arr.ndim - 2) + [zoom_h, zoom_w]
                    return zoom(arr, zoom_factors, order=1)

                idx0, idx1 = h2i[t0], h2i[t1]
                try:
                    surf_vars_input.update({
                        "2t":  torch.from_numpy(resize_array(np.stack([surf_ds0["t2m"].values[idx0], surf_ds1["t2m"].values[idx1]]), H, W))[None],
                        "10u": torch.from_numpy(resize_array(np.stack([surf_ds0["u10"].values[idx0], surf_ds1["u10"].values[idx1]]), H, W))[None],
                        "10v": torch.from_numpy(resize_array(np.stack([surf_ds0["v10"].values[idx0], surf_ds1["v10"].values[idx1]]), H, W))[None],
                        "msl": torch.from_numpy(resize_array(np.stack([surf_ds0["msl"].values[idx0], surf_ds1["msl"].values[idx1]]), H, W))[None],
                    })
                    atmos_vars_input.update({
                        "t": torch.from_numpy(resize_array(np.stack([atmos_ds0["t"].values[idx0], atmos_ds1["t"].values[idx1]]), H, W))[None],
                        "u": torch.from_numpy(resize_array(np.stack([atmos_ds0["u"].values[idx0], atmos_ds1["u"].values[idx1]]), H, W))[None],
                        "v": torch.from_numpy(resize_array(np.stack([atmos_ds0["v"].values[idx0], atmos_ds1["v"].values[idx1]]), H, W))[None],
                        "q": torch.from_numpy(resize_array(np.stack([atmos_ds0["q"].values[idx0], atmos_ds1["q"].values[idx1]]), H, W))[None],
                        "z": torch.from_numpy(resize_array(np.stack([atmos_ds0["z"].values[idx0], atmos_ds1["z"].values[idx1]]), H, W))[None],
                    })
                except ValueError:
                    print(f"Skipping problematic sample due to shape mismatch. Dates: {date0}, {date1}")
                    return None, None

                static_vars.update({
                    "z":   torch.from_numpy(resize_array(static_ds_tgt["z"].values[0], H, W)),
                    "slt": torch.from_numpy(resize_array(static_ds_tgt["slt"].values[0], H, W)),
                    "lsm": torch.from_numpy(resize_array(static_ds_tgt["lsm"].values[0], H, W)),
                })

                if self.use_wind_prompt:
                    u0_resized = resize_array(surf_ds0["u10"].values[idx0], H, W)
                    v0_resized = resize_array(surf_ds0["v10"].values[idx0], H, W)
                    u1_resized = resize_array(surf_ds1["u10"].values[idx1], H, W)
                    v1_resized = resize_array(surf_ds1["v10"].values[idx1], H, W)
                    wind_prompt_t0 = wind_to_image(u0_resized, v0_resized, H, W)
                    wind_prompt_t1 = wind_to_image(u1_resized, v1_resized, H, W)
                    wind_prompts = np.stack([wind_prompt_t0, wind_prompt_t1], axis=0) / 255.0
                    surf_vars_input['wind_prompt'] = torch.from_numpy(wind_prompts[np.newaxis, ...]).float()

                idx_target = h2i[sample["target_hour"]]
                target_slice = slice(idx_target, idx_target + self.horizon)
                surf_vars_target.update({
                    "2t":  torch.from_numpy(resize_array(surf_ds_tgt["t2m"].values[target_slice], H, W))[None],
                    "10u": torch.from_numpy(resize_array(surf_ds_tgt["u10"].values[target_slice], H, W))[None],
                    "10v": torch.from_numpy(resize_array(surf_ds_tgt["v10"].values[target_slice], H, W))[None],
                    "msl": torch.from_numpy(resize_array(surf_ds_tgt["msl"].values[target_slice], H, W))[None],
                })
                atmos_vars_target.update({
                    "u": torch.from_numpy(resize_array(atmos_ds_tgt["u"].values[target_slice], H, W))[None],
                    "t": torch.from_numpy(resize_array(atmos_ds_tgt["t"].values[target_slice], H, W))[None],
                    "v": torch.from_numpy(resize_array(atmos_ds_tgt["v"].values[target_slice], H, W))[None],
                    "q": torch.from_numpy(resize_array(atmos_ds_tgt["q"].values[target_slice], H, W))[None],
                    "z": torch.from_numpy(resize_array(atmos_ds_tgt["z"].values[target_slice], H, W))[None],
                })
                atmos_levels = tuple(int(level) for level in atmos_ds_tgt.pressure_level.values)

            # ----------------- CMAQ (optional, unchanged) ---------------------
            if "cmaq" in self.sources or self.use_cutmix:
                ts0 = f"{date0.replace('-', '')}{t0}"
                ts1 = f"{date1.replace('-', '')}{t1}"
                c0 = self._load_cmaq(ts0, target_h=H, target_w=W)
                c1 = self._load_cmaq(ts1, target_h=H, target_w=W)
                if c0 is None or c1 is None:
                    return None, None
                Zc, Zm, plev = c0["Z_conc"], c0["Z_m3d"], c0["plev"]
                assert (Zc, Zm, plev) == (c1["Z_conc"], c1["Z_m3d"], c1["plev"]), "Mismatched CMAQ Z between input hours"

            if "cmaq" in self.sources:
                conc_vars_to_use = CONC_VARS
                m2d_vars_to_use = M2D_VARS
                m3d_vars_to_use = M3D_VARS
                if self.use_cmaq_pm_only:
                    conc_vars_to_use = ["PM2P5", "PM10"]
                    m2d_vars_to_use = []
                    m3d_vars_to_use = []

                if self.use_cmaq_m2d:
                    for i, name in enumerate(m2d_vars_to_use):
                        surf_vars_input[f"{name.lower()}_cmaq"] = torch.from_numpy(
                            np.stack([c0["m2d"][M2D_VARS.index(name)], c1["m2d"][M2D_VARS.index(name)]])[None]
                        )

                if self.use_cmaq_conc and Zc > 0:
                    for base in conc_vars_to_use:
                        original_j = CONC_VARS.index(base)
                        start = original_j * Zc
                        if Zc == 1:
                            sname = f"{base.lower()}_cmaq"
                            v0 = c0["conc"][start]
                            v1 = c1["conc"][start]
                            surf_vars_input[sname] = torch.from_numpy(np.stack([v0, v1])[None])
                        else:
                            lev0 = c0["conc"][start:start+Zc]
                            lev1 = c1["conc"][start:start+Zc]
                            stacked = np.stack([lev0, lev1], axis=0)
                            atmos_vars_input[f"{base.lower()}_cmaq"] = torch.from_numpy(stacked[None])

                if self.use_cmaq_m3d and Zm > 0:
                    for base in m3d_vars_to_use:
                        original_j = M3D_VARS.index(base)
                        start = original_j * Zm
                        lev0 = c0["m3d"][start:start+Zm]
                        lev1 = c1["m3d"][start:start+Zm]
                        stacked = np.stack([lev0, lev1], axis=0)
                        atmos_vars_input[f"{base.lower()}_cmaq"] = torch.from_numpy(stacked[None])

                conc_as_3d = self.use_cmaq_conc and (Zc > 1)
                met_as_3d  = self.use_cmaq_m3d  and (Zm > 0)
                if met_as_3d:
                    atmos_levels = self._cmaq_plev(Zm)
                elif conc_as_3d:
                    atmos_levels = self._cmaq_plev(Zc)

                c_tgt = self._load_cmaq_sequence(date0, sample["target_hour"], target_h=H, target_w=W)
                if c_tgt is None:
                    return None, None
                Zc_t, Zm_t, plev_t = c_tgt["Z_conc"], c_tgt["Z_m3d"], c_tgt["plev"]
                assert (Zc, Zm, plev) == (Zc_t, Zm_t, plev_t), "Input/target CMAQ Z mismatch"

                if self.use_cmaq_m2d:
                    for i, name in enumerate(M2D_VARS):
                        key = f"{name.lower()}_cmaq"
                        arr = c_tgt["m2d"][:, i].astype(np.float32)
                        surf_vars_target[key] = torch.from_numpy(arr)[None]

                if self.use_cmaq_conc and Zc > 0:
                    if Zc == 1:
                        for j, base in enumerate(CONC_VARS):
                            start = j * Zc
                            key = f"{base.lower()}_cmaq"
                            arr = c_tgt["conc"][:, start]
                            surf_vars_target[key] = torch.from_numpy(arr)[None]
                    else:
                        for j, base in enumerate(CONC_VARS):
                            start = j * Zc
                            conc_all = c_tgt["conc"][:, start:start+Zc]
                            atmos_vars_target[f"{base.lower()}_cmaq"] = torch.from_numpy(conc_all[None])

                if self.use_cmaq_m3d and Zm > 0:
                    for j, base in enumerate(M3D_VARS):
                        start = j * Zm
                        m3d_all = c_tgt["m3d"][:, start:start+Zm]
                        atmos_vars_target[f"{base.lower()}_cmaq"] = torch.from_numpy(m3d_all[None])

            # ----------------- CAMS (NEW) ------------------------------------
            if "cams" in self.sources:
                cam0 = self._load_cams_slice(date0, t0, target_h=H, target_w=W)
                cam1 = self._load_cams_slice(date1, t1, target_h=H, target_w=W)
                if (cam0 is None) or (cam1 is None):
                    return None, None
                surf_vars_input.update({
                    "pm2p5_cams": torch.from_numpy(np.stack([cam0["pm2p5"], cam1["pm2p5"]], axis=0)[None]),
                    "pm10_cams":  torch.from_numpy(np.stack([cam0["pm10"] , cam1["pm10"] ], axis=0)[None]),
                })
                cams_tgt = self._load_cams_sequence(next_date, sample["target_hour"], target_h=H, target_w=W)
                if cams_tgt is None:
                    return None, None
                surf_vars_target.update({
                    "pm2p5_cams": torch.from_numpy(cams_tgt["pm2p5"][None]),
                    "pm10_cams":  torch.from_numpy(cams_tgt["pm10" ][None]),
                })

            # ----------------- Assemble & return ------------------------------
            in_times = (
                datetime.strptime(f"{date0} {t0}", "%Y-%m-%d %H"),
                datetime.strptime(f"{date1} {t1}", "%Y-%m-%d %H"),
            )
            meta_in  = Metadata(lat=lat, lon=lon, time=in_times, atmos_levels=atmos_levels)
            tgt_times = tuple(
                datetime.strptime(
                    f"{self._shift(next_date, sample['target_hour'], off)[0]} "
                    f"{self._shift(next_date, sample['target_hour'], off)[1]}",
                    "%Y-%m-%d %H")
                for off in range(self.horizon)
            )
            meta_tgt = Metadata(lat=lat, lon=lon, time=tgt_times, atmos_levels=atmos_levels)

            batch = Batch(surf_vars=surf_vars_input, static_vars=static_vars, atmos_vars=atmos_vars_input, metadata=meta_in)
            target_batch = Batch(surf_vars=surf_vars_target, static_vars=static_vars, atmos_vars=atmos_vars_target, metadata=meta_tgt)

            if self.use_flow:
                batch.flow_vars = flow_vars_input
                target_batch.flow_vars = flow_vars_target

            return batch, target_batch

    # --------------------------- Target sequence helper ---------------------- #
    def _load_target_sequence(self,
                              start_date: str,
                              start_hh  : str) -> np.ndarray | None:
        """
        Load <horizon> consecutive OBS tensors as ndarray (C, horizon, H, W).
        Returns None if any step is missing.
        """
        seq = []
        for off in range(self.horizon):
            d, h = self._shift(start_date, start_hh, off)
            x    = self._load_obs_with_date(d, h)
            if x is None:
                return None
            seq.append(x)
        return np.stack(seq, axis=1)

# ================================ Collate =================================== #
def collate_batches(batch_list):
    # Filter out any Nones
    filtered = [b for b in batch_list if b is not None and b[0] is not None]

    if len(filtered) == 0:
        return None

    inputs = [b[0] for b in filtered]
    targets = [b[1] for b in filtered]

    if getattr(inputs[0], "flow_vars", None) is None:
        use_flow = False
    else:
        use_flow = True
        collated_flow_vars = {
            k: torch.cat([b.flow_vars[k] for b in inputs], dim=0)
            for k in inputs[0].flow_vars
        }

    # Then do the usual stacking
    collated_surf_vars = {
        k: torch.cat([b.surf_vars[k] for b in inputs], dim=0)
        for k in inputs[0].surf_vars
    }
    collated_static_vars = inputs[0].static_vars
    collated_atmos_vars = {
        k: torch.cat([b.atmos_vars[k] for b in inputs], dim=0)
        for k in inputs[0].atmos_vars
    }

    # ---- INPUT metadata --------------------------------------------------
    meta0 = inputs[0].metadata
    collated_time_in = tuple(b.metadata.time[0] for b in inputs)   # length == B
    new_metadata = type(meta0)(
        lat=meta0.lat,
        lon=meta0.lon,
        time=collated_time_in,
        atmos_levels=meta0.atmos_levels,
        rollout_step=meta0.rollout_step,
    )

    # ---- TARGET metadata -------------------------------------------------
    meta0t = targets[0].metadata
    new_metadata_t = type(meta0t)(
        lat=meta0t.lat,
        lon=meta0t.lon,
        time=meta0t.time,               # keep full horizon tuple
        atmos_levels=meta0t.atmos_levels,
        rollout_step=meta0t.rollout_step,
    )

    collated_inputs = inputs[0].__class__(
        surf_vars=collated_surf_vars,
        static_vars=collated_static_vars,
        atmos_vars=collated_atmos_vars,
        metadata=new_metadata,
    )
    if use_flow:
        collated_inputs.flow_vars = collated_flow_vars

    # Do the same for targets
    collated_surf_vars_t = {
        k: torch.cat([b.surf_vars[k] for b in targets], dim=0)
        for k in targets[0].surf_vars
    }
    if use_flow:
        collated_flow_vars_t = {
            k: torch.cat([b.flow_vars[k] for b in targets], dim=0)
            for k in targets[0].flow_vars
        }

    collated_static_vars_t = targets[0].static_vars
    collated_atmos_vars_t = {
        k: torch.cat([b.atmos_vars[k] for b in targets], dim=0)
        for k in targets[0].atmos_vars
    }
    meta0t = targets[0].metadata
    new_metadata_t  = type(meta0t)(
        lat=meta0t.lat,
        lon=meta0t.lon,
        time=meta0t.time,
        atmos_levels=meta0t.atmos_levels,
        rollout_step=meta0t.rollout_step,
    )
    collated_targets = targets[0].__class__(
        surf_vars=collated_surf_vars_t,
        static_vars=collated_static_vars_t,
        atmos_vars=collated_atmos_vars_t,
        metadata=new_metadata_t,
    )
    if use_flow:
        collated_targets.flow_vars = collated_flow_vars_t

    return (collated_inputs, collated_targets)


def aurora_collate_fn(batch):
    filtered = [b for b in batch if b is not None and b[0] is not None]
    if len(filtered) == 0:
        return None
    return collate_batches(filtered)

def _meta_update(meta, **kwargs):
    if hasattr(meta, "_replace"):              # named‑tuple
        return meta._replace(**kwargs)
    elif is_dataclass(meta):                   # dataclass (Aurora ≥ 0.4)
        return dc_replace(meta, **kwargs)
    raise TypeError(type(meta))

def _slice_time_batch(batch: "Batch", idx: int) -> "Batch":
    """Return a shallow copy with every T dimension cut to length 1 at *idx*."""
    surf  = {k: v[:, idx:idx + 1] for k, v in batch.surf_vars.items()}
    atmos = {k: v[:, idx:idx + 1] for k, v in batch.atmos_vars.items()}
    meta  = _meta_update(batch.metadata, time=(batch.metadata.time[idx],))
    return Batch(
        surf_vars=surf,
        atmos_vars=atmos,
        static_vars=batch.static_vars,
        metadata=meta,
    )

def _concat_time(b_list: list["Batch"]) -> "Batch":
    surf = {k: torch.cat([b.surf_vars[k]  for b in b_list], dim=1)
            for k in b_list[0].surf_vars}
    atmos= {k: torch.cat([b.atmos_vars[k] for b in b_list], dim=1)
            for k in b_list[0].atmos_vars}
    meta = _meta_update(
        b_list[0].metadata,
        time=tuple(t for b in b_list for t in b.metadata.time)
    )
    return Batch(surf_vars=surf,
                 atmos_vars=atmos,
                 static_vars=b_list[0].static_vars,
                 metadata=meta)

Batch.concat_time = staticmethod(_concat_time)
Batch.slice_time = _slice_time_batch

def visualise_obs(batch):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    bounds = [0, 15, 35, 75, 1000]
    color_pairs = [
        ['#2795EF', '#1976D2'],
        ['#8FDB91', '#248929'],
        ['#EBE082', '#F3B517'],
        ['#F88967', '#D81E1E'],
    ]
    vmin, vmax = bounds[0], bounds[-1]
    eps = 1e-12
    stops = []
    for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
        c0, c1 = color_pairs[i]
        x0 = (lo - vmin) / (vmax - vmin)
        x1 = (hi - vmin) / (vmax - vmin)
        stops.append((x0, c0))
        stops.append((max(x0, x1 - eps), c1))
        stops.append((x1, color_pairs[i+1][0] if i < len(color_pairs)-1 else c1))
    cmap = mcolors.LinearSegmentedColormap.from_list("pm_custom", stops)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    units = "µg m$^{-3}$"
    
    data_obs = batch.surf_vars['pm2p5'][0].detach().cpu().numpy()
    data_cmaq = batch.surf_vars.get('pm2p5_cmaq', None)
    plt.imshow(data_obs[0], cmap=cmap, norm=norm)
    plt.savefig("123123_00.png")
    plt.close()
    plt.imshow(data_obs[1], cmap=cmap, norm=norm)
    plt.savefig("123123_01.png")
    plt.close()
    if data_cmaq is not None:
        data_cmaq = data_cmaq[0].detach().cpu().numpy()
        plt.imshow(data_cmaq[0], cmap=cmap, norm=norm)
        plt.savefig("123123_10.png")
        plt.close()
        plt.imshow(data_cmaq[1], cmap=cmap, norm=norm)
        plt.savefig("123123_11.png")
        plt.close()
    
    raise 0
