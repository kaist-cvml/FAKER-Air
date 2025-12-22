# aurora/rl/grpo_train.py
from __future__ import annotations
import argparse, os, random, logging, time, math
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from aurora import Aurora, AuroraAirPollution, Batch
from aurora.dataloader import WeatherDataset, aurora_collate_fn
from aurora.utils import build_step_weights
from aurora.rl.grpo import (
    SamplerCfg, GRPOCfg,
    rollout_with_sampling_and_logprob, rollout_deterministic
)
from aurora.rl.reward import (
    mse_over_rollout, mse_reward_over_rollout, f1_reward_over_rollout, cls_reward_over_rollout, hybrid_reward_over_rollout,
    far_metric_over_rollout, recall_reward_over_rollout,
)
from tqdm import tqdm
from collections import deque

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -------------------------- #
# Utilities for diagnostics  #
# -------------------------- #
def _collapse_group_scalar(x: torch.Tensor) -> torch.Tensor:
    """
    x: shape [G] or [G, ...] -> returns shape [G] (mean over trailing dims).
    Accepts list of tensors as well.
    """
    if isinstance(x, list):
        x = torch.stack(x)
    x = x.float()
    if x.dim() == 0:
        x = x[None]
    if x.dim() > 1:
        x = x.view(x.shape[0], -1).mean(dim=1)
    return x

def _global_grad_norm(model: torch.nn.Module) -> float:
    """Compute global L2 grad norm across all parameters (NaN-safe)."""
    sq_sum = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            if torch.isfinite(g).all():
                sq_sum += float(torch.sum(g * g).item())
    return float(sq_sum ** 0.5)

def _param_norm(model: torch.nn.Module) -> float:
    """Compute global L2 param norm (for scale reference)."""
    sq_sum = 0.0
    for p in model.parameters():
        if p is not None:
            w = p.detach()
            if torch.isfinite(w).all():
                sq_sum += float(torch.sum(w * w).item())
    return float(sq_sum ** 0.5)

def _get_lr(optimizer: torch.optim.Optimizer) -> float:
    return optimizer.param_groups[0].get("lr", 0.0)

def _count_params(model: torch.nn.Module) -> tuple[int, int]:
    """Return (total_params, trainable_params)."""
    m = model.module if isinstance(model, DDP) else model
    total = sum(p.numel() for p in m.parameters())
    train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, train

def make_loader(ds, batch, shuffle, distributed, drop_last=False, *,
                pin_memory=False, num_workers=2, prefetch_factor=2, persistent_workers=False):
    samp = DistributedSampler(ds, shuffle=shuffle) if distributed else None
    return DataLoader(
        ds, batch_size=batch, sampler=samp,
        shuffle=shuffle and samp is None,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=drop_last,
        collate_fn=aurora_collate_fn,
    )

def save_ckpt(path: Path, model, opt, epoch: int, best_metric: float, extra: dict | None = None):
    """
    Save a GRPO checkpoint with enough info to resume or analyze later.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    sd = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    payload = {
        "epoch": epoch,
        "model_state_dict": sd,
        "optimizer_state_dict": opt.state_dict(),
        "best_metric": best_metric,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    logging.info(f"[ckpt] saved → {path}")

def build_models(args, in0):
    surf_vars = tuple(sorted(in0.surf_vars.keys()))
    static_vars = tuple(sorted(in0.static_vars.keys()))
    atmos_vars = tuple(sorted(in0.atmos_vars.keys()))
    # Detect Z (number of vertical levels) from any atmos var, if present
    if atmos_vars:
        any_key = atmos_vars[0]
        Z_detected = in0.atmos_vars[any_key].shape[2]  # [B, T, Z, H, W]
    else:
        Z_detected = 2  # fallback

    # ---------------- Window size heuristic for Swin‑3D ---------------- #
    import math
    H, W = in0.spatial_shape
    patch_sz      = 2
    h_tokens      = H // patch_sz
    w_tokens      = W // patch_sz
    win_h         = math.gcd(h_tokens, 8) or 1
    win_w         = math.gcd(w_tokens, 12) or 1
    window_size   = (1, win_h, win_w)
    logging.info(f"window_size chosen = {window_size}")

    if args.model == "pollution":
        policy = AuroraAirPollution(
            surf_vars=surf_vars, static_vars=static_vars, atmos_vars=atmos_vars,
            patch_size=patch_sz, latent_levels=Z_detected, window_size=window_size,
            use_lora=args.use_lora,
            drop_rate=args.drop_rate, drop_path=args.drop_path,   # <--- NEW
        )
        ref = AuroraAirPollution(
            surf_vars=surf_vars, static_vars=static_vars, atmos_vars=atmos_vars,
            patch_size=patch_sz, latent_levels=Z_detected, window_size=window_size,
            use_lora=args.use_lora,
            drop_rate=0.0, drop_path=0.0,   # <--- NEW
        )
    else:
        policy = Aurora(
            surf_vars=surf_vars, static_vars=static_vars, atmos_vars=atmos_vars,
            patch_size=patch_sz, latent_levels=Z_detected, window_size=window_size,
            drop_rate=args.drop_rate, drop_path=args.drop_path,   # <--- NEW
        )
        ref = Aurora(
            surf_vars=surf_vars, static_vars=static_vars, atmos_vars=atmos_vars,
            patch_size=patch_sz, latent_levels=Z_detected, window_size=window_size,
            drop_rate=0.0, drop_path=0.0,   # <--- NEW
        )
    return policy, ref

def load_state_dict_flex(model: torch.nn.Module, ckpt_path: str, strict: bool = False):
    obj = torch.load(ckpt_path, map_location="cpu")  # only trusted files
    if isinstance(obj, dict) and ("model_state_dict" in obj or "model" in obj):
        state = obj.get("model_state_dict", obj.get("model"))
    else:
        state = obj

    clean = {}
    for k, v in state.items():
        if k.startswith("module."):
            clean[k[7:]] = v
        else:
            clean[k] = v

    missing, unexpected = model.load_state_dict(clean, strict=strict)
    logging.info(f"[ckpt] loaded '{ckpt_path}' (missing={len(missing)}, unexpected={len(unexpected)})")

class PIKLController:
    """Proportional-Integral controller for KL to tightly track a target."""
    def __init__(self, beta_init: float, target_kl: float,
                 beta_min: float, beta_max: float,
                 kp: float = 0.08, ki: float = 0.02, decay: float = 0.95):
        self.beta = float(beta_init)
        self.target = float(target_kl)
        self.beta_min, self.beta_max = float(beta_min), float(beta_max)
        self.kp, self.ki, self.decay = float(kp), float(ki), float(decay)
        self.err_i = 0.0

    def update(self, kl_ema: float) -> float:
        err = float(kl_ema - self.target)
        self.err_i = self.decay * self.err_i + err
        mult = math.exp(self.kp * err + self.ki * self.err_i)
        self.beta = float(np.clip(self.beta * mult, self.beta_min, self.beta_max))
        return self.beta

class PIConstraint:
    """
    mode='min'  → metric >= target (ex. Recall)
    mode='max'  → metric <= target (ex. FAR)
    """
    def __init__(self, init: float, target: float, kp: float, ki: float,
                 lo: float = 0.0, hi: float = 3.0, decay: float = 0.95, mode: str = "min"):
        assert mode in ("min", "max")
        self.lmbd  = float(init)
        self.tgt   = float(target)
        self.kp    = float(kp)
        self.ki    = float(ki)
        self.lo    = float(lo)
        self.hi    = float(hi)
        self.decay = float(decay)
        self.mode  = mode
        self.err_i = 0.0

    def update(self, metric: float) -> float:
        metric = float(metric)
        if self.mode == "min":   # metric >= target
            err = self.tgt - metric
        else:                    # metric <= target
            err = metric - self.tgt
        self.err_i = self.decay * self.err_i + err
        mult = math.exp(self.kp * err + self.ki * self.err_i)
        self.lmbd = float(np.clip(self.lmbd * mult, self.lo, self.hi))
        return self.lmbd


class RolloutCurriculum:
    """Increase rollout_steps gradually for stability."""
    def __init__(self, base: int, max_steps: int, step_inc: int = 2, every_epochs: int = 1):
        self.base, self.max, self.inc, self.every = base, max_steps, step_inc, every_epochs
    def steps_for_epoch(self, ep: int) -> int:
        k = (ep // self.every)
        return int(min(self.max, self.base + k * self.inc))

def centered_rank(x: torch.Tensor) -> torch.Tensor:
    # returns values in [-1, 1], deterministic order-insensitive
    ranks = torch.argsort(torch.argsort(x))
    y = ranks.float() / max(1, x.numel() - 1)
    return 2.0 * (y - 0.5)
    
def main():
    ap = argparse.ArgumentParser("Aurora + GRPO")

    # Data & model
    ap.add_argument("--train-start", default="2020-01-01")
    ap.add_argument("--train-end",   default="2020-12-31")
    ap.add_argument("--val-start",   default="2021-01-01")
    ap.add_argument("--val-end",     default="2021-12-31")
    ap.add_argument("--data-dir",    default="./data/era5")
    ap.add_argument("--npz-path",    default="./data/obs_npz_27km")
    ap.add_argument("--cmaq-root",   default="./data/cmaq_only_npy")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--model", choices=["aurora","pollution"], default="pollution")
    ap.add_argument("--data-sources", default="obs", help="comma-separated list ⇒ obs, cmaq, era5 …")
    ap.add_argument("--use-lora", action="store_true")
    ap.add_argument("--hybrid-target", action="store_true",
                help="Use OBS where available, otherwise CMAQ for reward targets (OBS⊕CMAQ).")
    ap.add_argument("--use-masking", action="store_true", help="Enable masking augmentation.")

    # Training schedule
    ap.add_argument("--epochs-sft", type=int, default=1)
    ap.add_argument("--epochs-grpo", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=0.0)

    # GRPO / sampling / reward
    ap.add_argument("--beta-kl", type=float, default=1e-3)
    ap.add_argument("--beta-kl-min", type=float, default=5e-6)
    ap.add_argument("--beta-kl-max", type=float, default=1e-1)
    ap.add_argument("--target-kl", type=float, default=15.0, help="Target KL for step-wise EMA controller.")
    ap.add_argument("--kl-every-step", action="store_true",
                    help="Apply KL on every rollout step (overrides first/last).")
    ap.add_argument("--kl-last-step-only", action="store_true",
                    help="Apply KL only on the last rollout step.")
    ap.add_argument("--kl-ema-alpha", type=float, default=0.05, help="EMA smoothing for step-wise KL.")
    ap.add_argument("--reward-ema-alpha", type=float, default=0.05, help="EMA smoothing for reward mean/std.")
    ap.add_argument("--adv-ema-coeff", type=float, default=0.5,
                    help="Blend coeff for EMA-normalized advantage (0→group-only, 1→EMA-only).")

    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--rollout-steps", type=int, default=12)
    ap.add_argument("--sigma-pm25", type=float, default=3.0)
    ap.add_argument("--sigma-pm10", type=float, default=6.0)
    ap.add_argument("--use-cutmix", action="store_true")
    ap.add_argument("--reward", choices=["mse", "f1", "cls", "hybrid"], default="mse")

    ap.add_argument("--cls-exact-mult-pm25",  type=str, default="",
                    help="ex: '1,1,2,3'")
    ap.add_argument("--cls-exact-mult-pm10",  type=str, default="")
    ap.add_argument("--cls-coarse-mult-pm25", type=str, default="",
                    help="ex: '1,1,1,1'")
    ap.add_argument("--cls-coarse-mult-pm10", type=str, default="")
    ap.add_argument("--cls-exact-bonus-pm25", type=str, default="",
                    help="ex: '0,0,0.0,0.2'")
    ap.add_argument("--cls-exact-bonus-pm10", type=str, default="")

    # ----- Constraints (Recipe 2) -----
    ap.add_argument("--recall-floor",      type=float, default=0.60)
    ap.add_argument("--far-target",        type=float, default=0.18)
    ap.add_argument("--lambda-rec-init",   type=float, default=0.20)
    ap.add_argument("--lambda-far-init",   type=float, default=0.20)
    ap.add_argument("--lambda-min",        type=float, default=0.0)
    ap.add_argument("--lambda-rec-max",    type=float, default=3.0)
    ap.add_argument("--lambda-far-max",    type=float, default=3.0)
    ap.add_argument("--pi-rec-kp",         type=float, default=0.05)
    ap.add_argument("--pi-rec-ki",         type=float, default=0.01)
    ap.add_argument("--pi-far-kp",         type=float, default=0.05)
    ap.add_argument("--pi-far-ki",         type=float, default=0.01)
    ap.add_argument("--metric-ema-alpha",  type=float, default=0.05)

    # ----- Curriculum gating (Recipe 3) -----
    ap.add_argument("--rollout-gating", action="store_true")
    ap.add_argument("--gate-steps",     type=int,   default=200)
    ap.add_argument("--gate-inc",       type=int,   default=1)

    # tiered-class reward hyper-parameters
    ap.add_argument("--cls-coarse", type=float, default=0.5,
                    help="+α reward for correct high/low side (good+moderate vs bad+verybad).")
    ap.add_argument("--cls-exact", type=float, default=0.5,
                    help="+β reward for exact class match.")
    ap.add_argument("--cls-fa-penalty", type=float, default=0.0,
                    help="−γ penalty when predicting high while GT is low (false alarm).")

    # optional: override class bounds via CLI (comma-separated, 5 endpoints)
    ap.add_argument("--cls-bounds-pm25", type=str, default="0,15,35,75,800",
                    help="5 endpoints for PM2.5 classes (good,moderate,bad,verybad).")
    ap.add_argument("--cls-bounds-pm10", type=str, default="0,30,80,150,1000",
                    help="5 endpoints for PM10 classes (good,moderate,bad,verybad).")
    ap.add_argument("--cls-scale-pm25", type=float, default=1.0,
                    help="Scale multiplier applied to PM2.5 values before binning.")
    ap.add_argument("--cls-scale-pm10", type=float, default=1.0,
                    help="Scale multiplier applied to PM10 values before binning.")
    ap.add_argument("--reward-temp", type=float, default=1.0,
                    help="Temperature τ for exp reward shaping (used when --reward-shape=exp).")
    
    ap.add_argument("--rollout-curriculum", action="store_true")
    ap.add_argument("--rollout-base", type=int, default=1)
    ap.add_argument("--rollout-inc", type=int, default=1)
    ap.add_argument("--rollout-every", type=int, default=1)
    ap.add_argument("--sigma-decay", type=float, default=0.98)
    ap.add_argument("--kp", type=float, default=0.08)
    ap.add_argument("--ki", type=float, default=0.02)

    # Checkpoints & distributed
    ap.add_argument("--ref-ckpt",  default=None, help="Reference (frozen) weights for KL.")
    ap.add_argument("--init-ckpt", default=None, help="Initial policy weights.")
    ap.add_argument("--ckpt-dir",  default="./checkpoints_grpo", help="Where to save GRPO checkpoints")
    ap.add_argument("--exp-name",  default="grpo_run", help="Experiment name for checkpoint subfolder")
    ap.add_argument("--tensorboard", default="./runs/grpo")
    ap.add_argument("--distributed", action="store_true")
    ap.add_argument("--pin-memory", action="store_true", default=False)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--prefetch-factor", type=int, default=2)
    ap.add_argument("--persistent-workers", action="store_true")

    # Logprob / KL weights per variable
    ap.add_argument("--logprob-vars", type=str, default="pm2p5,pm10",
                    help="Comma-separated list of variables used in log-prob/KL.")
    ap.add_argument("--logprob-w-pm25", type=float, default=1.0)
    ap.add_argument("--logprob-w-pm10", type=float, default=0.5)
    ap.add_argument("--kl-w-pm25",      type=float, default=1.0)
    ap.add_argument("--kl-w-pm10",      type=float, default=0.5)

    # Reward variable weights
    ap.add_argument("--reward-w-pm25",  type=float, default=1.0)
    ap.add_argument("--reward-w-pm10",  type=float, default=0.5)
    ap.add_argument("--step-weight-mode", choices=["uniform","linear","exp","sigmoid"], default="linear")
    ap.add_argument("--step-weight-base", type=float, default=0.75)  
    ap.add_argument("--step-weight-gamma", type=float, default=0.9)

    # --- in ArgumentParser setup ---
    ap.add_argument("--drop-rate", type=float, default=0.0, help="Module dropout prob.")
    ap.add_argument("--drop-path", type=float, default=0.0, help="Stochastic depth prob.")
    ap.add_argument("--dropout-in-rollout", action="store_true",
                    help="Keep policy in train() so dropout is active during rollouts (MC dropout).")

    # Sampling variance control
    ap.add_argument("--antithetic", action="store_true",
                    help="Use antithetic sampling (±ε) within each group.")
    ap.add_argument("--common-noise", action="store_true",
                    help="Use common random numbers: share ε stream across group members.")

    # AMP & shaping
    ap.add_argument("--amp", action="store_true", help="Use autocast (bf16 if supported).")
    ap.add_argument("--reward-shape",
                    choices=["neg", "inv", "one_minus", "exp", "log", "rel_impr"],
                    default="exp", help="Shaping for dimensionless MSE → reward.")

    # Logging controls
    ap.add_argument("--log-every", type=int, default=1, help="Scalar log interval (steps).")
    ap.add_argument("--hist-every", type=int, default=200, help="Histogram log interval (steps).")
    ap.add_argument("--val-iters", type=int, default=0, help="Validation batches per epoch (0=skip).")
    ap.add_argument("--profile-mem", action="store_true", help="Log CUDA memory usage periodically.")

    args = ap.parse_args()

    # DDP init
    if args.distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        is_master = (dist.get_rank() == 0)
    else:
        local_rank = 0; is_master = True

    # Seeding
    seed = 42
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    def _parse_bounds(s: str) -> list[float]:
        return [float(x) for x in s.split(",")]
    
    def _parse_vec4_opt(s: str) -> list[float] | None:
        s = s.strip()
        if not s:
            return None
        vals = [float(x) for x in s.split(",")]
        assert len(vals) == 4, "Expected 4 comma-separated floats"
        return vals

    cls_bounds = {
        "pm2p5": _parse_bounds(args.cls_bounds_pm25),
        "pm10" : _parse_bounds(args.cls_bounds_pm10),
    }
    cls_scales = {"pm2p5": args.cls_scale_pm25, "pm10": args.cls_scale_pm10}

    # per-class 조절(옵션)
    cls_exact_mult = {
        "pm2p5": _parse_vec4_opt(args.cls_exact_mult_pm25),
        "pm10" : _parse_vec4_opt(args.cls_exact_mult_pm10),
    }
    cls_coarse_mult = {
        "pm2p5": _parse_vec4_opt(args.cls_coarse_mult_pm25),
        "pm10" : _parse_vec4_opt(args.cls_coarse_mult_pm10),
    }
    cls_exact_bonus = {
        "pm2p5": _parse_vec4_opt(args.cls_exact_bonus_pm25),
        "pm10" : _parse_vec4_opt(args.cls_exact_bonus_pm10),
    }

    cls_exact_mult  = {k: v for k, v in cls_exact_mult.items()  if v is not None}
    cls_coarse_mult = {k: v for k, v in cls_coarse_mult.items() if v is not None}
    cls_exact_bonus = {k: v for k, v in cls_exact_bonus.items() if v is not None}

    # Derive logging paths from ckpt/exp if desired
    if args.ref_ckpt:
        try:
            args.exp_name = f"{args.ref_ckpt.split('/')[-3]}/{args.ref_ckpt.split('/')[-2]}_{args.exp_name}"
        except Exception:
            pass
    args.tensorboard = str(Path(args.ckpt_dir) / args.exp_name)

    logging.info(f"args:\n{args}")
    sources = [s.strip().lower() for s in args.data_sources.split(',')]
    logging.info(f"[INFO] use inputs: {sources}")

    # Datasets & loaders
    train_ds = WeatherDataset(
        args.train_start, args.train_end, args.data_dir, args.npz_path,
        cmaq_root=args.cmaq_root, sources=sources, horizon=args.rollout_steps,
        use_cutmix=args.use_cutmix, use_cmaq_pm_only=True,
        use_hybrid_target=args.hybrid_target,  # <-- NEW,
        use_masking=args.use_masking,
    )
    val_ds   = WeatherDataset(
        args.val_start, args.val_end, args.data_dir, args.npz_path,
        cmaq_root=args.cmaq_root, sources=sources, horizon=args.rollout_steps,
        use_cutmix=args.use_cutmix, use_cmaq_pm_only=True,
        # use_hybrid_target=args.hybrid_target  # <-- NEW (reward/VAL metrics also use hybrid GT)
    )

    train_ld = make_loader(
        train_ds, args.batch, shuffle=True, distributed=args.distributed, drop_last=True,
        pin_memory=args.pin_memory, num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor, persistent_workers=args.persistent_workers
    )
    val_ld = make_loader(
        val_ds, 1, shuffle=False, distributed=args.distributed, drop_last=False,
        pin_memory=args.pin_memory, num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor, persistent_workers=args.persistent_workers
    )

    current_train_batch = args.batch

    in0, tgt0 = next(b for b in train_ld if b is not None)
    policy, ref = build_models(args, in0)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    if args.init_ckpt:
        load_state_dict_flex(policy, args.init_ckpt, strict=False)
    if args.ref_ckpt:
        load_state_dict_flex(ref, args.ref_ckpt, strict=False)
    policy = policy.to(device)
    ref    = ref.to(device)
    for p in ref.parameters(): p.requires_grad_(False)  # freeze reference
    ref.eval()

    if args.distributed:
        policy = DDP(policy, device_ids=[local_rank], output_device=local_rank,
                     static_graph=True, gradient_as_bucket_view=True)

    opt = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95), eps=1e-8
    )
    writer = SummaryWriter(args.tensorboard) if is_master else None
    if is_master and writer:
        writer.add_text("config/args", str(args))
        tot_params, train_params = _count_params(policy)
        writer.add_scalar("model/params_total", tot_params, 0)
        writer.add_scalar("model/params_trainable", train_params, 0)
        writer.add_scalar("sched/target_kl", float(args.target_kl), 0)
        writer.add_scalar("sched/adv_ema_coeff", float(args.adv_ema_coeff), 0)
        logging.info(f"[model] params total={tot_params:,} trainable={train_params:,}")

    logprob_vars = tuple([s.strip() for s in args.logprob_vars.split(",") if s.strip()])

    sampler = SamplerCfg(
        sigma={"pm2p5": args.sigma_pm25, "pm10": args.sigma_pm10},
        vars_for_logprob=logprob_vars,
        var_logprob_weight={"pm2p5": args.logprob_w_pm25, "pm10": args.logprob_w_pm10},
        var_kl_weight={"pm2p5": args.kl_w_pm25, "pm10": args.kl_w_pm10},
        kl_on_first_step_only=not (args.kl_every_step or args.kl_last_step_only),
        kl_on_last_step_only=args.kl_last_step_only,
    )

    grpo_cfg = GRPOCfg(
        group_size=args.group_size,
        beta_kl=args.beta_kl,
        rollout_steps=args.rollout_steps,
        use_rollout_cutmix=args.use_cutmix,
        obs_path=args.npz_path,
        cmaq_root=args.cmaq_root,
        use_amp=args.amp,
    )

    kl_ctrl = PIKLController(
    beta_init=grpo_cfg.beta_kl, target_kl=args.target_kl,
    beta_min=args.beta_kl_min, beta_max=args.beta_kl_max,
    kp=args.kp, ki=args.ki
    )
    # Constraint controllers (Recall↑, FAR↓)
    rec_ctrl = PIConstraint(
        init=args.lambda_rec_init, target=args.recall_floor,
        kp=args.pi_rec_kp, ki=args.pi_rec_ki,
        lo=args.lambda_min, hi=args.lambda_rec_max, mode="min"
    )
    far_ctrl = PIConstraint(
        init=args.lambda_far_init, target=args.far_target,
        kp=args.pi_far_kp, ki=args.pi_far_ki,
        lo=args.lambda_min, hi=args.lambda_far_max, mode="max"
    )

    # Metric EMAs + gating window
    rec_ema, far_ema = None, None
    rec_window = deque(maxlen=args.gate_steps)

    curric = RolloutCurriculum(
        base=args.rollout_base, max_steps=args.rollout_steps,
        step_inc=args.rollout_inc, every_epochs=args.rollout_every
    ) if args.rollout_curriculum else None

    reward_weights = {"pm2p5": args.reward_w_pm25, "pm10": args.reward_w_pm10}

    # ------------------  Phase 1: SFT (optional) ------------------ #
    global_step = 0
    for ep in range(args.epochs_sft):
        policy.train()
        total = 0.0; steps = 0
        pbar = tqdm(train_ld, desc=f"SFT {ep+1}/{args.epochs_sft}", ncols=120, disable=not is_master)
        t_last = time.time()
        for batch in pbar:
            if batch is None:
                if args.distributed:
                    dist.barrier()   # make all ranks wait here => keep step alignment
                continue
            x, y = (b.to(device) for b in batch)

            ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                   if getattr(args, "amp", False) and torch.cuda.is_bf16_supported()
                   else torch.cuda.amp.autocast(enabled=False))
            with ctx:
                pred = policy(x)

            loss = 0.0
            for var in ("pm2p5","pm10"):
                g = y.surf_vars[var][:,0]; p = pred.surf_vars[var][:,0]
                m = (g>0)
                if m.any():
                    diff2 = (p[m]-g[m]).pow(2).mean()
                    loss = loss + diff2

            opt.zero_grad()
            loss.backward()

            gn_before = _global_grad_norm(policy.module if isinstance(policy, DDP) else policy)
            clip_ret = torch.nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
            opt.step()
            gn_after = _global_grad_norm(policy.module if isinstance(policy, DDP) else policy)

            total += float(loss); steps += 1; global_step += 1

            now = time.time()
            steps_per_sec = 1.0 / max(1e-6, (now - t_last))
            t_last = now

            if is_master and writer and (global_step % args.log_every == 0):
                writer.add_scalar("SFT/step_loss", float(loss), global_step)
                writer.add_scalar("SFT/grad_norm/before_clip", float(gn_before), global_step)
                writer.add_scalar("SFT/grad_norm/after_clip",  float(gn_after),  global_step)
                writer.add_scalar("SFT/param_norm", _param_norm(policy.module if isinstance(policy, DDP) else policy), global_step)
                writer.add_scalar("SFT/lr", _get_lr(opt), global_step)
                writer.add_scalar("SFT/steps_per_sec", steps_per_sec, global_step)
                if args.profile_mem and torch.cuda.is_available():
                    dev = torch.device("cuda", local_rank)
                    writer.add_scalar("sys/cuda_mem_alloc_GB", torch.cuda.memory_allocated(dev)/1e9, global_step)
                    writer.add_scalar("sys/cuda_mem_rsrv_GB",  torch.cuda.memory_reserved(dev)/1e9,  global_step)

            if is_master and steps % 10 == 0:
                pbar.set_postfix(loss=f"{total/max(1,steps):.4f}")
        if is_master and writer:
            writer.add_scalar("SFT/epoch_avg_loss", total/max(1,steps), ep)
        logging.info(f"[SFT] epoch {ep} loss={total/max(1,steps):.4f}")

    # ------------------  Phase 2: GRPO ------------------ #
    best_reward = -float("inf")
    ckpt_root = Path(args.ckpt_dir) / args.exp_name

    for ep in range(args.epochs_grpo):
        policy.train()

        # ---- (A) 이번 에폭의 rollout_steps를 먼저 확정 (curriculum) ----
        if curric is not None:
            grpo_cfg.rollout_steps = curric.steps_for_epoch(ep)

        # ---- (B) rollout_steps에 따른 배치 크기 결정 ----
        if grpo_cfg.rollout_steps == 1 and args.group_size <= 4:
            desired_batch = args.batch * 4
        elif grpo_cfg.rollout_steps == 2 and args.group_size <= 4:
            desired_batch = args.batch * 2
        else:
            desired_batch = args.batch

        # ---- (C) 배치 크기가 바뀌면 DataLoader 재생성 ----
        if desired_batch != current_train_batch:
            train_ld = make_loader(
                train_ds, desired_batch, shuffle=True, distributed=args.distributed, drop_last=True,
                pin_memory=args.pin_memory, num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor, persistent_workers=args.persistent_workers
            )
            current_train_batch = desired_batch
            logging.info(
                f"[curriculum] rollout_steps={grpo_cfg.rollout_steps} → train batch_size={current_train_batch}"
            )

        # DDP sampler 에폭 동기화 (로더가 바뀌었을 수 있으니 여기서 호출)
        if args.distributed and isinstance(train_ld.sampler, DistributedSampler):
            train_ld.sampler.set_epoch(ep)

        # 선택: 스케줄/상태 로깅
        if is_master and writer:
            writer.add_scalar("sched/rollout_steps", grpo_cfg.rollout_steps, ep)
            writer.add_scalar("sched/batch_size",    current_train_batch,   ep)

        ep_loss, ep_reward = 0.0, 0.0; n_steps = 0
        epoch_kl_sum = 0.0; epoch_kl_cnt = 0

        logging.info(f"[sched] epoch={ep} beta_kl={grpo_cfg.beta_kl:.4g}")
        logging.info(f"[sched] epoch={ep} sigma_pm25={sampler.sigma['pm2p5']:.3g} sigma_pm10={sampler.sigma['pm10']:.3g}")

        if is_master and writer:
            writer.add_scalar("sched/beta_kl", float(grpo_cfg.beta_kl), ep)
            writer.add_scalar("sampler/sigma_pm2p5", float(sampler.sigma["pm2p5"]), ep)
            writer.add_scalar("sampler/sigma_pm10",  float(sampler.sigma["pm10"]),  ep)

        # --- EMA trackers and KL controller targets for this epoch --- #
        kl_ema = None
        r_ema_mean, r_ema_var = 0.0, 1e-6

        pbar = tqdm(train_ld, desc=f"GRPO {ep+1}/{args.epochs_grpo}", ncols=120, disable=not is_master)
        t_last = time.time()

        if curric is not None:
            grpo_cfg.rollout_steps = curric.steps_for_epoch(ep)
        sampler.sigma["pm2p5"] *= args.sigma_decay
        sampler.sigma["pm10"]  *= args.sigma_decay
        logging.info(f"[sched] epoch={ep} rollout_steps={grpo_cfg.rollout_steps} "
                    f"sigma_pm25={sampler.sigma['pm2p5']:.2f} sigma_pm10={sampler.sigma['pm10']:.2f}")
        
        mode_for_build = "fixed" if args.step_weight_mode == "uniform" else args.step_weight_mode  # uniform→fixed 매핑
        Tw = int(grpo_cfg.rollout_steps)
        step_w = build_step_weights(
            Tw, mode=mode_for_build, base=args.step_weight_base, gamma=args.step_weight_gamma, device=device
        ).float()

        for batch in pbar:
            if args.distributed:
                none_flag = torch.tensor(0 if batch is not None else 1,
                                        device=device, dtype=torch.int32)
                dist.all_reduce(none_flag, op=dist.ReduceOp.SUM)
                if int(none_flag.item()) > 0:
                    continue
            else:
                if batch is None:
                    continue
            x, y = (b.to(device) for b in batch)

            # --- Decide handles and modes for rollout (policy/ref) ---
            m_policy = policy.module if isinstance(policy, DDP) else policy
            m_ref    = ref  # ref is not wrapped with DDP
            # Reference must be deterministic
            m_ref.eval()
            # Policy: toggle MC-dropout during rollout via flag
            prev_train_mode = m_policy.training
            if args.dropout_in_rollout:
                m_policy.train()   # Dropout/DropPath ON during rollout sampling
            else:
                m_policy.eval()    # Dropout/DropPath OFF during rollout sampling

            # Reference baseline for relative-improvement shaping (compute once per step)
            ref_mse_scalar = None
            if args.reward_shape == "rel_impr":
                # m_ref is in eval(), no dropout => deterministic baseline
                ref_preds = rollout_deterministic(m_ref, x, grpo_cfg)
                ref_mse_scalar = mse_over_rollout(ref_preds, y, weights=reward_weights)

            # (Do NOT compute the shaped debug metric here; we’ll do it later after we have preds_for_debug)
            logprobs, kls, rewards = [], [], []
            per_var_aux_sum = {}
            preds_for_debug = None

            # Common random numbers seed (shared ε stream across group)
            base_seed = None
            if args.common_noise:
                if args.distributed:
                    if dist.get_rank() == 0:
                        base_seed_t = torch.randint(0, 2**31-1, (1,), device=device, dtype=torch.int64)
                    else:
                        base_seed_t = torch.zeros(1, device=device, dtype=torch.int64)
                    dist.broadcast(base_seed_t, src=0)
                    base_seed = int(base_seed_t.item())
                else:
                    base_seed = int(torch.randint(0, 2**31-1, (1,)).item())
            else:
                base_seed = int(torch.randint(0, 2**31-1, (1,)).item())

            for g in range(args.group_size):
                grpo_cfg.noise_seed  = base_seed + g
                grpo_cfg.group_index = g
                grpo_cfg.antithetic  = bool(args.antithetic)
                grpo_cfg.common_noise= bool(args.common_noise)

                if args.common_noise:
                    with torch.random.fork_rng(devices=[device], enabled=True):
                        torch.manual_seed(base_seed + g)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(base_seed + g)
                        preds, logp, kl, aux = rollout_with_sampling_and_logprob(
                            m_policy, m_ref, x, y, sampler, grpo_cfg, step_weights=step_w
                        )
                else:
                    preds, logp, kl, aux = rollout_with_sampling_and_logprob(
                        m_policy, m_ref, x, y, sampler, grpo_cfg, step_weights=step_w
                    )
                if g == 0:
                    preds_for_debug = preds

                if isinstance(logp, torch.Tensor):
                    logp = logp.float().reshape(-1).mean()
                if isinstance(kl, torch.Tensor):
                    kl = kl.float().reshape(-1).mean()
                    
                # Reward shaping (with temperature support in reward.py)
                # --- (1) base reward ---
                if args.reward == "mse":
                    r = mse_reward_over_rollout(
                        preds, y, weights=reward_weights,
                        shape=args.reward_shape, ref_mse=ref_mse_scalar, tau=args.reward_temp,
                        step_weights=step_w,                                  # <-- FIX: apply
                    )
                elif args.reward == "f1":
                    r = torch.tensor(
                        f1_reward_over_rollout(preds, y, weights=reward_weights),
                        device=device, dtype=torch.float32
                    )
                elif args.reward == "cls":
                    r = cls_reward_over_rollout(
                        preds, y, weights=reward_weights,
                        bounds_overrides=cls_bounds, scale_overrides=cls_scales,
                        coarse_w=args.cls_coarse, exact_w=args.cls_exact, fa_penalty=args.cls_fa_penalty,
                        step_weights=step_w,                                  # <-- FIX: apply
                        exact_mult_overrides=cls_exact_mult,
                        coarse_mult_overrides=cls_coarse_mult,
                        exact_bonus_overrides=cls_exact_bonus,
                    )
                elif args.reward == "hybrid":
                    r = hybrid_reward_over_rollout(
                        preds, y, mse_w=1.0, cls_w=0.15,
                        mse_shape=args.reward_shape, ref_mse=ref_mse_scalar, tau=args.reward_temp,
                        weights=reward_weights, bounds_overrides=cls_bounds, scale_overrides=cls_scales,
                        coarse_w=args.cls_coarse, exact_w=args.cls_exact, fa_penalty=args.cls_fa_penalty,
                        step_weights=step_w,                                  # <-- FIX: apply
                        exact_mult_overrides=cls_exact_mult,
                        coarse_mult_overrides=cls_coarse_mult,
                        exact_bonus_overrides=cls_exact_bonus,
                    )
                else:
                    raise ValueError(f"Unknown reward type: {args.reward}")

                # --- (2) Recipe 2 ---
                lambda_rec_t = torch.as_tensor(rec_ctrl.lmbd,  device=r.device, dtype=r.dtype)
                lambda_far_t = torch.as_tensor(far_ctrl.lmbd,  device=r.device, dtype=r.dtype)
                rec_floor_t  = torch.as_tensor(args.recall_floor, device=r.device, dtype=r.dtype)
                far_tgt_t    = torch.as_tensor(args.far_target,   device=r.device, dtype=r.dtype)

                rec_metric = recall_reward_over_rollout(preds, y, var="pm2p5", thr=35.0, w=1.0,
                                        step_weights=step_w)
                far_metric = far_metric_over_rollout   (preds, y, var="pm2p5", thr=35.0, w=1.0,
                                                        step_weights=step_w)

                rec_margin = rec_metric - rec_floor_t          
                far_excess = torch.clamp(far_metric - far_tgt_t, min=0)

                r = r + lambda_rec_t * rec_margin - lambda_far_t * far_excess

                # append
                rewards.append(r)
                logprobs.append(logp)
                kls.append(kl)

                # Aggregate auxiliary stats
                for var, d in aux["per_var"].items():
                    if var not in per_var_aux_sum:
                        per_var_aux_sum[var] = {"mask":0.0, "lp":0.0, "kl":0.0, "cnt":0, "sigma": d["sigma"]}
                    per_var_aux_sum[var]["mask"] += float(d["mask_frac"])
                    per_var_aux_sum[var]["lp"]   += float(d["lp_mean"])
                    if not (np.isnan(d["kl_mean"]) or np.isinf(d["kl_mean"])):  # guard NaN/Inf
                        per_var_aux_sum[var]["kl"] += float(d["kl_mean"])
                    per_var_aux_sum[var]["cnt"]  += 1

            R    = _collapse_group_scalar(torch.stack(rewards))
            logP = _collapse_group_scalar(torch.stack(logprobs))
            KL   = _collapse_group_scalar(torch.stack(kls))

            assert R.dim()==logP.dim()==KL.dim()==1, (R.shape, logP.shape, KL.shape)
            assert R.shape[0]==logP.shape[0]==KL.shape[0]==args.group_size

            # 1) EMA baseline (for A_ema)
            R_mean = R.mean().detach()
            r_alpha = float(args.reward_ema_alpha)
            r_ema_mean = (1.0 - r_alpha) * r_ema_mean + r_alpha * float(R_mean.item())
            r_diff = float(R_mean.item()) - r_ema_mean
            r_ema_var  = (1.0 - r_alpha) * r_ema_var + r_alpha * (r_diff * r_diff)
            r_ema_std  = (r_ema_var ** 0.5) + 1e-6
            A_ema = (R - R_mean.new_tensor(r_ema_mean)) / R_mean.new_tensor(r_ema_std)

            # 2) Two candidate advantages (for logging / robustness)
            A_group = (R - R.mean()) / R.std(unbiased=False).clamp_min(1e-6)  # for diagnostics
            A_rank  = centered_rank(R.detach())                                # robust for small G

            # 3) Advantage used for the update (rank + small EMA blend)
            c = float(args.adv_ema_coeff)  # e.g., 0.1
            A_used = (1.0 - c) * A_rank + c * A_ema

            # Step-wise KL EMA & controller
            kl_alpha = float(args.kl_ema_alpha)
            kl_step = float(KL.mean().item())
            if kl_ema is None: kl_ema = kl_step
            else:              kl_ema = (1.0 - kl_alpha)*kl_ema + kl_alpha*kl_step

            grpo_cfg.beta_kl = kl_ctrl.update(kl_ema)

            # NaN/Inf guards
            if not torch.isfinite(logP).all() or not torch.isfinite(KL).all():
                logging.warning(f"[nan-guard] non-finite logP/KL at step={n_steps} (epoch {ep})")

            epoch_kl_sum += KL.mean().item(); epoch_kl_cnt += 1

            # Restore training mode for the optimization step
            m_policy.train()

            # Policy gradient loss (detach advantages)
            loss = -(A_used.detach() * logP).mean() + grpo_cfg.beta_kl * KL.mean()

            opt.zero_grad()
            loss.backward()

            # Grad diagnostics and step
            gn_before = _global_grad_norm(policy.module if isinstance(policy, DDP) else policy)
            clip_ret  = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            gn_after  = _global_grad_norm(policy.module if isinstance(policy, DDP) else policy)

            # Accounting
            ep_loss   += float(loss)
            ep_reward += float(R.mean().detach().cpu().item())
            n_steps   += 1
            global_step += 1

            # Throughput & memory logs
            now = time.time()
            steps_per_sec = 1.0 / max(1e-6, (now - t_last))
            t_last = now

            # ---------------------- step-level logging ---------------------- #
            if preds_for_debug is not None:
                rec_now = float(recall_reward_over_rollout(
                    preds_for_debug, y, var="pm2p5", thr=35.0, w=1.0, step_weights=step_w  # <<< 추가
                ).item())
                far_now = float(far_metric_over_rollout(
                    preds_for_debug, y, var="pm2p5", thr=35.0, w=1.0, step_weights=step_w  # <<< 추가
                ).item())
                a = float(args.metric_ema_alpha)

                rec_ema = rec_now if rec_ema is None else (1.0 - a) * rec_ema + a * rec_now
                far_ema = far_now if far_ema is None else (1.0 - a) * far_ema + a * far_now
                rec_window.append(rec_now)

                # PI
                rec_ctrl.update(rec_ema)
                far_ctrl.update(far_ema)

                if is_master and writer:
                    writer.add_scalar("constraint/recall_step", rec_now, global_step)
                    writer.add_scalar("constraint/far_step",    far_now, global_step)
                    writer.add_scalar("constraint/recall_ema",  float(rec_ema), global_step)
                    writer.add_scalar("constraint/far_ema",     float(far_ema), global_step)
                    writer.add_scalar("constraint/lambda_rec",  float(rec_ctrl.lmbd), global_step)
                    writer.add_scalar("constraint/lambda_far",  float(far_ctrl.lmbd), global_step)

            if is_master and writer and (global_step % args.log_every == 0):
                writer.add_scalar("GRPO/step_loss",   float(loss),                global_step)
                writer.add_scalar("GRPO/step_reward", float(R.mean().item()),     global_step)
                writer.add_scalar("GRPO/step_logp",   float(logP.mean().item()),  global_step)
                writer.add_scalar("GRPO/step_kl",     float(KL.mean().item()),    global_step)
                writer.add_scalar("GRPO/adv_std_group", float(A_group.std(unbiased=False).item()), global_step)
                writer.add_scalar("GRPO/adv_std_rank",  float(A_rank.std(unbiased=False).item()),  global_step)
                writer.add_scalar("GRPO/adv_std_ema",   float(A_ema.std(unbiased=False).item()),   global_step)
                writer.add_scalar("sched/kl_ema", float(kl_ema), global_step)

                # grads/params/lr
                writer.add_scalar("GRPO/grad_norm/before_clip", float(gn_before), global_step)
                writer.add_scalar("GRPO/grad_norm/after_clip",  float(gn_after),  global_step)
                writer.add_scalar("GRPO/grad_norm/preclip_return", float(clip_ret), global_step)
                writer.add_scalar("GRPO/param_norm", _param_norm(policy.module if isinstance(policy, DDP) else policy), global_step)
                writer.add_scalar("GRPO/lr", _get_lr(opt), global_step)

                # schedule + sampler state
                writer.add_scalar("sched/beta_kl_step", float(grpo_cfg.beta_kl), global_step)
                for var, s in sampler.sigma.items():
                    writer.add_scalar(f"sampler/sigma_{var}_step", float(s), global_step)

                # per-variable aux (avg across group)
                for var, d in per_var_aux_sum.items():
                    cnt = max(1, d["cnt"])
                    writer.add_scalar(f"aux/mask_frac/{var}", d["mask"]/cnt, global_step)
                    writer.add_scalar(f"aux/logprob_mean/{var}", d["lp"]/cnt, global_step)
                    if d["kl"] != 0.0:
                        writer.add_scalar(f"aux/kl_mean/{var}", d["kl"]/cnt, global_step)
                    writer.add_scalar(f"aux/sigma/{var}", d["sigma"], global_step)

                # Optional diagnostics with the first rollout in the group
                if preds_for_debug is not None and is_master and writer and (global_step % args.log_every == 0):
                    m25 = mse_over_rollout(preds_for_debug, y, {"pm2p5": 1.0, "pm10": 0.0}, step_weights=step_w)  # <-- FIX
                    m10 = mse_over_rollout(preds_for_debug, y, {"pm2p5": 0.0, "pm10": 1.0}, step_weights=step_w)  # <-- FIX
                    writer.add_scalar("metric/mse_dimless_pm2p5", float(m25.item()), global_step)
                    writer.add_scalar("metric/mse_dimless_pm10",  float(m10.item()),  global_step)

                    ref_mse_for_log = ref_mse_scalar if args.reward_shape == "rel_impr" else None
                    shaped = float(mse_reward_over_rollout(
                        preds_for_debug, y, weights=reward_weights,
                        shape=args.reward_shape, ref_mse=ref_mse_for_log, tau=args.reward_temp,
                        step_weights=step_w,                                  # <-- FIX
                    ).item())
                    writer.add_scalar("metric/reward_shaped_sample", shaped, global_step)

                    if args.reward == "cls":
                        r_debug = cls_reward_over_rollout(
                            preds_for_debug, y, weights=reward_weights,
                            bounds_overrides=cls_bounds, scale_overrides=cls_scales,
                            coarse_w=args.cls_coarse, exact_w=args.cls_exact, fa_penalty=args.cls_fa_penalty,
                            step_weights=step_w,
                            exact_mult_overrides=cls_exact_mult,      # <<< 추가
                            coarse_mult_overrides=cls_coarse_mult,    # <<< 추가
                            exact_bonus_overrides=cls_exact_bonus,    # <<< 추가
                        )
                        writer.add_scalar("metric/reward_cls_sample", float(r_debug.item()), global_step)

                writer.add_scalar("sys/steps_per_sec", steps_per_sec, global_step)
                writer.add_scalar("sys/seconds_per_step", 1.0 / max(1e-9, steps_per_sec), global_step)
                if args.profile_mem and torch.cuda.is_available():
                    dev = torch.device("cuda", local_rank)
                    writer.add_scalar("sys/cuda_mem_alloc_GB", torch.cuda.memory_allocated(dev)/1e9, global_step)
                    writer.add_scalar("sys/cuda_mem_rsrv_GB",  torch.cuda.memory_reserved(dev)/1e9,  global_step)

            # Histograms (sparser)
            if is_master and writer and (global_step % args.hist_every == 0):
                writer.add_histogram("hist/R",   R.detach().cpu(),  global_step)
                writer.add_histogram("hist/logP",logP.detach().cpu(),global_step)
                writer.add_histogram("hist/KL",  KL.detach().cpu(),  global_step)
                writer.add_histogram("hist/A_used", A_used.detach().cpu(), global_step)

            if is_master and n_steps % 5 == 0:
                pbar.set_postfix(loss=f"{ep_loss/max(1,n_steps):.4f}",
                                 reward=f"{ep_reward/max(1,n_steps):.4f}",
                                 logp=f"{logP.mean().item():.3e}",
                                 kl=f"{KL.mean().item():.3e}")

        # ---- Curriculum Gating (Recipe 3) ----
        if args.rollout_gating:
            rec_recent = float(rec_ema) if rec_ema is not None else 0.0
            if is_master and writer:
                writer.add_scalar("gate/recall_recent_mean", rec_recent, ep)
                writer.add_scalar("gate/rollout_steps", grpo_cfg.rollout_steps, ep)
            if rec_recent >= float(args.recall_floor):
                if grpo_cfg.rollout_steps < args.rollout_steps:
                    grpo_cfg.rollout_steps = min(args.rollout_steps, grpo_cfg.rollout_steps + args.gate_inc)
                    logging.info(f"[gate] recall_ok ({rec_recent:.3f} >= {args.recall_floor:.2f}) → "
                                f"rollout_steps → {grpo_cfg.rollout_steps}")
                    T = grpo_cfg.rollout_steps
            else:
                # 조건 실패 → β_KL, σ, λ_rec 순차 보정
                old_beta = grpo_cfg.beta_kl
                grpo_cfg.beta_kl = min(grpo_cfg.beta_kl * 1.15, args.beta_kl_max)
                sampler.sigma["pm2p5"] = min(sampler.sigma["pm2p5"] * 1.05, 6.0)
                rec_ctrl.lmbd          = min(rec_ctrl.lmbd * 1.20, args.lambda_rec_max)

                logging.info(f"[gate] recall_low ({rec_recent:.3f} < {args.recall_floor:.2f}) → "
                            f"beta_kl {old_beta:.3g}→{grpo_cfg.beta_kl:.3g}, "
                            f"sigma_pm25→{sampler.sigma['pm2p5']:.2f}, "
                            f"lambda_rec→{rec_ctrl.lmbd:.3f}")

            rec_window.clear()


        # -------- epoch averages & controller (optional smoothing) -------- #
        avg_loss   = ep_loss   / max(1, n_steps)
        avg_reward = ep_reward / max(1, n_steps)
        avg_kl     = (epoch_kl_sum / max(1, epoch_kl_cnt)) if epoch_kl_cnt > 0 else 0.0

        if args.distributed:
            t_loss, t_reward, t_kl = (torch.tensor(avg_loss, device=device),
                                      torch.tensor(avg_reward, device=device),
                                      torch.tensor(avg_kl, device=device))
            dist.all_reduce(t_loss,   op=dist.ReduceOp.AVG)
            dist.all_reduce(t_reward, op=dist.ReduceOp.AVG)
            dist.all_reduce(t_kl,     op=dist.ReduceOp.AVG)
            avg_loss, avg_reward, avg_kl = float(t_loss), float(t_reward), float(t_kl)

        if is_master and writer:
            writer.add_scalar("GRPO/epoch_avg_loss",   avg_loss,   ep)
            writer.add_scalar("GRPO/epoch_avg_reward", avg_reward, ep)
            writer.add_scalar("GRPO/epoch_avg_kl",     avg_kl,     ep)

        # Epoch-level KL controller (kept for additional smoothing)
        # lo, hi = 0.8 * args.target_kl, 1.2 * args.target_kl
        # if avg_kl > hi:
        #     grpo_cfg.beta_kl = min(grpo_cfg.beta_kl * 1.5, args.beta_kl_max)
        # elif avg_kl < lo:
        #     grpo_cfg.beta_kl = max(grpo_cfg.beta_kl * 0.7, args.beta_kl_min)
        if is_master:
            logging.info(f"[sched] epoch={ep} avg_kl={avg_kl:.3f} → beta_kl={grpo_cfg.beta_kl:.4g}")
            if writer:
                writer.add_scalar("sched/beta_kl_post_epoch", float(grpo_cfg.beta_kl), ep)

        # Optional lightweight validation (deterministic)
        if is_master and writer and args.val_iters > 0:
            policy.eval()
            val_mse_list, val_iter = [], 0
            with torch.no_grad():
                for vb in val_ld:
                    if vb is None: continue
                    vx, vy = (b.to(device) for b in vb)
                    vpreds = rollout_deterministic(policy.module if isinstance(policy, DDP) else policy, vx, grpo_cfg)
                    vm = mse_over_rollout(vpreds, vy, reward_weights)
                    val_mse_list.append(float(vm.item()))
                    val_iter += 1
                    if val_iter >= args.val_iters: break
            if val_mse_list:
                writer.add_scalar("VAL/mse_dimless_mean", float(np.mean(val_mse_list)), ep)
                writer.add_scalar("VAL/mse_dimless_std",  float(np.std(val_mse_list)),  ep)

        # Checkpointing
        if is_master:
            save_ckpt(ckpt_root / "last_policy.pth", policy, opt, ep, best_reward,
                      extra={"beta_kl": grpo_cfg.beta_kl, "sigma": sampler.sigma})
            if avg_reward > best_reward:
                best_reward = avg_reward
                save_ckpt(ckpt_root / "best_policy.pth", policy, opt, ep, best_reward,
                          extra={"beta_kl": grpo_cfg.beta_kl, "sigma": sampler.sigma})

        logging.info(f"[GRPO] epoch {ep} loss={ep_loss/max(1,n_steps):.4f} reward={ep_reward/max(1,n_steps):.4f}")
        if args.distributed:
            dist.barrier()

    if writer: writer.close()
    if args.distributed: dist.destroy_process_group()

if __name__ == "__main__":
    main()