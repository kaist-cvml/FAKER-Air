# aurora/rl/grpo.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import dataclasses
import torch
import math
from aurora import Batch
from aurora.model.aurora import Aurora
from aurora.rollout import (_load_obs_for_ts, _load_cmaq_for_ts, build_next_input_from_pred_and_cmaq)

@dataclass
class SamplerCfg:
    sigma: Dict[str, float] = dataclasses.field(
        default_factory=lambda: {"pm2p5": 3.0, "pm10": 6.0}
    )
    vars_for_logprob: Tuple[str, ...] = ("pm2p5", "pm10")
    var_logprob_weight: Dict[str, float] = dataclasses.field(
        default_factory=lambda: {"pm2p5": 1.0, "pm10": 0.5}
    )
    var_kl_weight: Dict[str, float] = dataclasses.field(
        default_factory=lambda: {"pm2p5": 1.0, "pm10": 0.5}
    )
    kl_on_first_step_only: bool = True
    kl_on_last_step_only:  bool = False

@dataclasses.dataclass
class GRPOCfg:
    group_size: int = 4
    beta_kl: float = 1e-3
    rollout_steps: int = 12
    use_rollout_cutmix: bool = True
    obs_path: str | None = None
    cmaq_root: str | None = None
    use_amp: bool = False   # AMP inside rollout
    # NEW: sampling controls
    common_noise: bool = False       # enable common random numbers across ranks
    antithetic: bool = False         # use eps and -eps
    noise_seed: int | None = None    # seed for current rollout call
    group_index: int = 0             # which member of the group (0..G-1)

def _gaussian_logprob(a, mu, sigma, mask):
    # log N(a | mu, sigma^2 I) averaged on valid pixels
    var_t = torch.tensor(sigma**2, device=mu.device, dtype=mu.dtype)
    const = torch.log(var_t * (2.0 * math.pi))
    lp = -0.5 * (((a - mu) ** 2) / var_t + const)
    if mask is not None:
        lp = lp * mask
        return lp.sum() / mask.sum().clamp_min(1.0)
    return lp.mean()

def _kl_gaussian_same_sigma(mu_p, mu_ref, sigma, mask):
    # KL[N(mu_p, sigma^2) || N(mu_ref, sigma^2)] = (mu_ref - mu_p)^2 / (2 sigma^2)
    var_t = torch.tensor(sigma**2, device=mu_p.device, dtype=mu_p.dtype)
    kl = (mu_ref - mu_p) ** 2 / (2.0 * var_t)
    if mask is not None:
        kl = kl * mask
        return kl.sum() / mask.sum().clamp_min(1.0)
    return kl.mean()

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

@torch.no_grad()
def _prepare_batch_like_forward(model: Aurora, batch: Batch) -> Batch:
    p = next(model.parameters())
    return (model.batch_transform_hook(batch).type(p.dtype).crop(model.patch_size).to(p.device))

def _normalize_step_weights(T: int, step_weights: Optional[torch.Tensor], device) -> Optional[torch.Tensor]:
    if step_weights is None:
        return None
    sw = step_weights.to(device=device, dtype=torch.float32)
    if sw.numel() != T:
        sw = sw[:T] if sw.numel() > T else torch.cat([sw, torch.ones(T - sw.numel(), device=device)])
    return sw

def rollout_with_sampling_and_logprob(
    policy: Aurora,
    ref: Aurora | None,
    init_batch: Batch,
    tgt: Batch,
    sampler: 'SamplerCfg',
    cfg: GRPOCfg,
    *,
    rng_seed: int | None = None,
    noise_scale: float = 1.0,
    step_weights: torch.Tensor | None = None,
):
    """
    Returns:
        preds: List[Batch] over rollout steps
        logprob_total: scalar mean log-prob averaged over variables/steps (weighted)
        kl_total: scalar mean KL averaged over variables/steps (weighted)
        aux_stats: dict with per-variable mask fraction, lp/kl means, and sigmas
    """
    device = next(policy.parameters()).device
    batch = _prepare_batch_like_forward(policy, init_batch)
    T_MAX = policy.max_history_size
    H, W = batch.spatial_shape

    preds: List[Batch] = []
    logprob_total = torch.tensor(0.0, device=device)
    kl_total      = torch.tensor(0.0, device=device)
    sum_w_lp      = torch.tensor(0.0, device=device)
    sum_w_kl      = torch.tensor(0.0, device=device)

    # detailed accumulators
    per_var_acc = {
        var: {"lp_sum": 0.0, "lp_cnt": 0, "kl_sum": 0.0, "kl_cnt": 0, "mask_sum": 0.0, "mask_cnt": 0}
        for var in sampler.vars_for_logprob
    }

    # Unified AMP context
    if cfg.use_amp and torch.cuda.is_available():
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        amp_ctx = torch.cuda.amp.autocast(enabled=False)

    # Common RNG for this rollout (optional)
    gen = None
    if cfg.noise_seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(cfg.noise_seed))

    # Build step weights (shape: [T])
    T = int(cfg.rollout_steps)
    if step_weights is None:
        step_w = torch.ones(T, device=device, dtype=torch.float32)
    else:
        step_w = step_weights.to(device=device, dtype=torch.float32)
        if step_w.numel() != T:
            step_w = step_w[:T] if step_w.numel() > T else torch.cat(
                [step_w, torch.ones(T - step_w.numel(), device=device, dtype=torch.float32)]
            )

    for step in range(cfg.rollout_steps):
        # keep only last T_MAX frames
        batch = dataclasses.replace(
            batch,
            surf_vars={k: v[:, -T_MAX:] for k, v in batch.surf_vars.items()},
            atmos_vars={k: v[:, -T_MAX:] for k, v in batch.atmos_vars.items()},
        )

        # policy forward
        with amp_ctx:
            pred_mu: Batch = policy.forward(batch)

        # reference forward (if we might use it)
        if ref is not None and ((not sampler.kl_on_first_step_only) or step == 0 or sampler.kl_on_last_step_only):
            with torch.no_grad():
                with amp_ctx:
                    mu_ref: Batch = ref.forward(batch)
        else:
            mu_ref = None

        sampled_surf = {}
        for var in pred_mu.surf_vars:
            mu = pred_mu.surf_vars[var]  # [B,1,H,W]
            if (var in sampler.vars_for_logprob) and (var in sampler.sigma) and (sampler.sigma[var] > 0.0):
                sigma = float(sampler.sigma[var])

                # sample action
                eps = torch.empty_like(mu)
                if gen is not None:
                    eps.normal_(mean=0.0, std=1.0, generator=gen)
                else:
                    eps.normal_(mean=0.0, std=1.0)

                # antithetic for odd group members
                if cfg.antithetic and (cfg.group_index % 2 == 1):
                    eps = -eps

                a = mu + sigma * eps

                # GT mask at this step
                t_idx = step if step < tgt.surf_vars.get(var, mu).shape[1] else (tgt.surf_vars.get(var, mu).shape[1] - 1)
                gt_step = tgt.slice_time(t_idx)
                gvar    = gt_step.surf_vars.get(var, torch.zeros_like(mu[:, 0]))
                mask    = (gvar[:, 0] > 0).unsqueeze(1)  # [B,1,H,W], bool

                a_const = a.detach()

                # log-prob (mean over valid pixels), weighted per-variable
                w_log   = float(sampler.var_logprob_weight.get(var, 1.0))
                lp_mean = _gaussian_logprob(a_const, mu, sigma, mask)
                logprob_total = logprob_total + w_log * lp_mean
                sum_w_lp      = sum_w_lp + w_log

                # decide whether to apply KL on this step
                take_kl = False
                if mu_ref is not None:
                    if not sampler.kl_on_first_step_only and not sampler.kl_on_last_step_only:
                        take_kl = True  # every step
                    elif sampler.kl_on_first_step_only and step == 0:
                        take_kl = True
                    elif sampler.kl_on_last_step_only and step == (cfg.rollout_steps - 1):
                        take_kl = True

                # step weight for this step
                wt = step_w[step].clamp_min(0.0)

                if take_kl:
                    w_kl   = float(sampler.var_kl_weight.get(var, 1.0))
                    mu_r   = mu_ref.surf_vars[var]
                    kl_mean = _kl_gaussian_same_sigma(mu, mu_r, sigma, mask)
                    kl_total = kl_total + wt * w_kl * kl_mean
                    sum_w_kl = sum_w_kl + wt * w_kl

                    if var in per_var_acc:
                        per_var_acc[var]["kl_sum"] += float(kl_mean.item())
                        per_var_acc[var]["kl_cnt"] += 1

                if var in per_var_acc:
                    per_var_acc[var]["lp_sum"]   += float(lp_mean.item())
                    per_var_acc[var]["lp_cnt"]   += 1
                    per_var_acc[var]["mask_sum"] += float(mask.float().mean().item())
                    per_var_acc[var]["mask_cnt"] += 1

                sampled_surf[var] = a_const
            else:
                sampled_surf[var] = mu.detach()

        pred_sampled = dataclasses.replace(pred_mu, surf_vars=sampled_surf)
        preds.append(pred_sampled)

        # build next input (with optional cutmix)
        next_ts  = batch.metadata.time[-1] + policy.timestep
        obs_t    = _load_obs_for_ts(next_ts, cfg.obs_path) if (cfg.use_rollout_cutmix and cfg.obs_path) else None
        cmaq_t   = _load_cmaq_for_ts(next_ts, cfg.cmaq_root, H, W) if (cfg.use_rollout_cutmix and cfg.cmaq_root) else None
        mixed_in = build_next_input_from_pred_and_cmaq(
            pred_vars=pred_sampled.surf_vars, obs_data=obs_t, cmaq_data=cmaq_t, device=device
        ) if cfg.use_rollout_cutmix else pred_sampled.surf_vars

        next_surf = {k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1) for k, v in mixed_in.items()}
        next_atms = {k: torch.cat([batch.atmos_vars[k][:, 1:], v.detach()], dim=1) for k, v in pred_mu.atmos_vars.items()}
        batch = dataclasses.replace(batch, surf_vars=next_surf, atmos_vars=next_atms, metadata=pred_mu.metadata)

    # normalize aggregates (NaN-safe when sums are zero)
    logprob_total = logprob_total / sum_w_lp.clamp_min(1e-6)
    kl_total      = kl_total      / sum_w_kl.clamp_min(1e-6)

    # auxiliary summary
    per_var_out = {}
    for var, acc in per_var_acc.items():
        per_var_out[var] = {
            "mask_frac": (acc["mask_sum"] / max(1, acc["mask_cnt"])),
            "lp_mean":   (acc["lp_sum"]   / max(1, acc["lp_cnt"])),
            "kl_mean":   (acc["kl_sum"]   / max(1, acc["kl_cnt"])) if acc["kl_cnt"] > 0 else float("nan"),
            "sigma":     float(sampler.sigma.get(var, 0.0)),
        }
    aux_stats = {
        "lp_mean_all": float(logprob_total.item()),
        "kl_mean_all": float(kl_total.item()) if sum_w_kl.item() > 0 else float("nan"),
        "per_var": per_var_out,
        "steps": cfg.rollout_steps,
    }
    return preds, logprob_total, kl_total, aux_stats


@torch.no_grad()
def rollout_deterministic(
    model: Aurora,
    init_batch: Batch,
    cfg: GRPOCfg,
) -> List[Batch]:
    """Deterministic rollout that always uses the model mean (no sampling) for surf vars."""
    device = next(model.parameters()).device
    batch = _prepare_batch_like_forward(model, init_batch)
    T_MAX = model.max_history_size
    H, W = batch.spatial_shape

    preds: List[Batch] = []

    if getattr(cfg, "use_amp", False) and torch.cuda.is_available():
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        amp_ctx = torch.cuda.amp.autocast(enabled=False)

    for step in range(cfg.rollout_steps):
        batch = dataclasses.replace(
            batch,
            surf_vars={k: v[:, -T_MAX:] for k, v in batch.surf_vars.items()},
            atmos_vars={k: v[:, -T_MAX:] for k, v in batch.atmos_vars.items()},
        )
        with amp_ctx:
            mu: Batch = model.forward(batch)

        det_surf = {k: v.detach() for k, v in mu.surf_vars.items()}
        pred_det = dataclasses.replace(mu, surf_vars=det_surf)
        preds.append(pred_det)

        next_ts = batch.metadata.time[-1] + model.timestep
        obs_t  = _load_obs_for_ts(next_ts, cfg.obs_path) if (cfg.use_rollout_cutmix and cfg.obs_path) else None
        cmaq_t = _load_cmaq_for_ts(next_ts, cfg.cmaq_root, H, W) if (cfg.use_rollout_cutmix and cfg.cmaq_root) else None
        mixed_in = build_next_input_from_pred_and_cmaq(
            pred_vars=pred_det.surf_vars, obs_data=obs_t, cmaq_data=cmaq_t, device=device
        ) if cfg.use_rollout_cutmix else pred_det.surf_vars

        next_surf = {k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1) for k, v in mixed_in.items()}
        next_atms = {k: torch.cat([batch.atmos_vars[k][:, 1:], v.detach()], dim=1) for k, v in mu.atmos_vars.items()}
        batch = dataclasses.replace(batch, surf_vars=next_surf, atmos_vars=next_atms, metadata=mu.metadata)
    return preds