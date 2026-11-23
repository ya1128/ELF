from typing import Optional

import torch
import torch.nn.functional as F


# ============================================
# Noise Schedulers (how to compute z from x0 and noise)
# ============================================

def add_noise(x0, noise, t, config, cond_seq_mask=None):
    """Flow-matching interpolation z = t*x0 + (1-t)*noise*scale, preserving cond tokens."""
    t_expanded = t.reshape(-1, 1, 1)
    z = t_expanded * x0 + (1 - t_expanded) * noise * config.denoiser_noise_scale
    if cond_seq_mask is not None:
        z = cond_seq_mask * x0 + (1 - cond_seq_mask) * z
    return z


# ============================================
# Time Schedulers (how to sample t)
# ============================================

def sample_timesteps(
    batch_size: int,
    P_mean: float = -0.8,
    P_std: float = 0.8,
    time_schedule: str = 'logit_normal',
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
):
    """Sample timesteps using various time schedules.

    Args:
        batch_size: Number of samples
        P_mean: Mean for logit-normal distribution
        P_std: Std for logit-normal distribution
        time_schedule: 'logit_normal' or 'uniform'

    Returns:
        Sampled timesteps in [0, 1]
    """
    if time_schedule == 'logit_normal':
        z = torch.randn((batch_size,), dtype=dtype, device=device) * P_std + P_mean
        return torch.sigmoid(z)
    if time_schedule == 'uniform':
        return torch.rand((batch_size,), dtype=dtype, device=device)
    raise ValueError(f"Unknown time_schedule: {time_schedule}")

# 把 [0,1] 切成 64 步采样的时间网格,返回值是 n_steps + 1 个时间点(64+1=65)
def get_sampling_steps(
    n_steps: int, time_schedule: str = "logit_normal",
    P_mean: float = -0.8, P_std: float = 0.8,
    device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a length-(n_steps+1) tensor of t values in [0, 1] for a sampling run.

    - "uniform": evenly-spaced linspace from 0 to 1 (deterministic). 均匀切分
    - "logit_normal": sorted logit-normal samples with 0 / 1 endpoints (random). 先生成 63 个随机数，套 sigmoid，压到 (0,1) 区间
    """
    if time_schedule == "uniform":
        return torch.linspace(0.0, 1.0, n_steps + 1, dtype=dtype, device=device)
    if time_schedule == "logit_normal":
        steps = sample_timesteps(
            batch_size=n_steps - 1,
            P_mean=P_mean, P_std=P_std, time_schedule=time_schedule,
            device=device, dtype=dtype,
        )
        steps = torch.sort(steps).values  # 随机 t 排序，采样要按时间递增走：t=0 是噪声，t 往 1 靠近是目标 clean latent
        endpoints_lo = torch.zeros((1,), dtype=dtype, device=steps.device)
        endpoints_hi = torch.ones((1,), dtype=dtype, device=steps.device)
        return torch.cat([endpoints_lo, steps, endpoints_hi], dim=0)
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


# ============================================
# CFG Scale Sampling (how to sample cfg scale)
# ============================================

def sample_cfg_scale(batch_size, cfg_min=0.0, cfg_max=3.0,
                     dtype=torch.float32, device=None):
    """Sample CFG scale from log-uniform distribution in [cfg_min, cfg_max]."""
    u = torch.rand((batch_size,), dtype=dtype, device=device)
    a = float(1.0 + cfg_min)
    b = float(1.0 + cfg_max)
    log_ratio = torch.tensor(b / a, dtype=dtype, device=u.device).log()
    return a * torch.exp(u * log_ratio) - 1.0


# ============================================
# Conditioning helpers (preserve clean tokens during sampling) 把条件段位置“钉回”干净条件嵌入，模型每一步去噪都会更新，需要保证条件段 cond_seq 不变
# ============================================
# 凡是 mask > 0 的位置，输出用 cond_seq；凡是 mask <= 0 的位置，输出用 z_updated
def restore_cond(z_updated, cond_seq, cond_seq_mask):
    """Restore clean conditioning tokens in z after a denoising step."""
    mask = cond_seq_mask
    target_ndim = max(z_updated.dim(), cond_seq.dim())
    while mask.dim() < target_ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask > 0, cond_seq, z_updated)


def restore_vx(v, x, cond_seq, cond_seq_mask):
    """Restore cond positions: x -> clean cond_seq, v -> 0 (cond tokens don't move)."""
    if cond_seq is not None:
        x = restore_cond(x, cond_seq, cond_seq_mask)
        v = restore_cond(v, torch.zeros_like(cond_seq), cond_seq_mask)
    return v, x


# ============================================
# Flow-matching forward passes (with optional self-cond / CFG)
# ============================================

def net_out_to_v_x(net_out, z, t, t_eps=5e-2):
    """Convert x_pred network output to v and x.

    When the model returns a tuple (denoised_output, decoder_logits),
    decoder logits are discarded here (used separately in training).
    """
    if isinstance(net_out, tuple):
        net_out = net_out[0]
    t_reshaped = t.reshape(-1, 1, 1)
    x = net_out  # x_pred：模型认为当前 z_t 去噪后的 clean latent
    denom = torch.clamp(1.0 - t_reshaped, min=t_eps)  # 计算分母：1 - t 最小值不小于 t_eps=0.05
    v = (x - z) / denom  # 从模型预测的 clean latent 转换回 ODE 更新需要的速度
    return v, x

# self-conditioning 的 forward 过程
def _forward_sample_self_cond(
    model, z, t_batch, x_pred_prev, config,
    self_cond_cfg_scale, cond_seq, cond_seq_mask,
):
    """Forward pass with self-conditioning."""
    t_eps = config.t_eps
    self_cond_prob = config.self_cond_prob
    # 恢复条件段
    def _restore(v, x):
        return restore_vx(v, x, cond_seq=cond_seq, cond_seq_mask=cond_seq_mask)

    if config.num_self_cond_cfg_tokens > 0:
        if x_pred_prev is None:
            x_pred_prev = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)  # 如果上一步预测不存在，就初始化一个 x_pred_prev，通常是 x_pred_prev 存在且为 x_pred
        z_input_cond = torch.cat([z, x_pred_prev], dim=-1)  # z = 上一轮 clean prediction x_pred_prev 沿最后一个 embedding 维拼接后送进模型
        self_cond_scale_batch = torch.full((z.shape[0],), float(self_cond_cfg_scale),
                                           dtype=z.dtype, device=z.device)
        net_out_cond = model(z_input_cond, t_batch, deterministic=True,
                             self_cond_cfg_scale=self_cond_scale_batch)  # 调用 ELF 模型
        v_cond, x_cond = net_out_to_v_x(net_out_cond, z, t_batch, t_eps)  # x_cond：模型预测的 clean latent；v_cond：由 x_cond 和当前 z 反推出的速度
        return _restore(v_cond, x_cond)  # 条件段恢复：v_cond[cond]=0；x_cond[cond]=cond_seq

    # No self-conditioning 模型没有 self-conditioning 历史输入，只有当前 z
    if self_cond_prob == 0:
        net_out = model(z, t_batch, deterministic=True)
        v, x = net_out_to_v_x(net_out, z, t_batch, t_eps)
        return _restore(v, x)

    # Combined unconditional and conditional forward pass self-cond CFG 分支
    v_uncond = x_uncond = None
    if self_cond_cfg_scale != 1 or x_pred_prev is None:
        z_uncond = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
        z_input_uncond = torch.cat([z, z_uncond], dim=-1)
        net_out_uncond = model(z_input_uncond, t_batch, deterministic=True)
        v_uncond, x_uncond = net_out_to_v_x(net_out_uncond, z, t_batch, t_eps)
        v_uncond, x_uncond = _restore(v_uncond, x_uncond)
        if self_cond_cfg_scale == 0.0 or x_pred_prev is None:
            return v_uncond, x_uncond

    z_input_cond = torch.cat([z, x_pred_prev], dim=-1)
    net_out_cond = model(z_input_cond, t_batch, deterministic=True)
    v_cond, x_cond = net_out_to_v_x(net_out_cond, z, t_batch, t_eps)
    v_cond, x_cond = _restore(v_cond, x_cond)
    if self_cond_cfg_scale == 1:
        return v_cond, x_cond

    v_out = v_uncond + self_cond_cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + self_cond_cfg_scale * (x_cond - x_uncond)
    return _restore(v_out, x_out)

# 做 CFG 的函数，cfg_scale=2 作用到速度场（先算 conditional；继续算 unconditional；CFG 外推）
def _forward_sample(
    model, z, t_batch, x_pred_prev, config,
    cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask,
):
    """Forward pass with optional self-conditioning and CFG. 计算有条件”的速度和 clean prediction""" 
    v_cond, x_cond = _forward_sample_self_cond(
        model, z, t_batch, x_pred_prev, config,
        self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    # CFG 尺度为 1 时，不做无条件分支
    if cfg_scale == 1.0:
        return v_cond, x_cond

    # Unconditional forward: zero out cond prefix, no self-cond state, no restore
    z_uncond = restore_cond(z, torch.zeros_like(z), cond_seq_mask)  # mask > 0 的位置用第二个参数；mask <= 0 的位置用第一个参数；无条件版本：条件段被擦成零，生成段的噪声状态保留
    x_pred_prev_uncond = (
        None if x_pred_prev is None
        else restore_cond(x_pred_prev, torch.zeros_like(x_pred_prev), cond_seq_mask)
    )
    v_uncond, x_uncond = _forward_sample_self_cond(
        model, z_uncond, t_batch, x_pred_prev_uncond, config,
        self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=torch.zeros_like(cond_seq), cond_seq_mask=cond_seq_mask,
    )
    # classifier-free guidance 的核心公式：展开 v_out = (1 - cfg_scale) * v_uncond + cfg_scale * v_cond
    v_out = v_uncond + cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + cfg_scale * (x_cond - x_uncond)
    return restore_vx(v_out, x_out, cond_seq, cond_seq_mask)

# 一个 ODE Euler 步，把噪声往前推进一小步。 x_pred：模型对干净表示的预测；v_pred：根据预测换算出的更新方向
def _ode_step(
    model, z, t, t_next, x_pred_prev,
    config, cfg_scale, self_cond_cfg_scale,
    cond_seq, cond_seq_mask,
):
    """Single ODE (Euler) step for sampling."""
    t_batch = torch.full((z.shape[0],), float(t), dtype=z.dtype, device=z.device)  # 把标量 t 变成 batch 张量，所有样本用同一个 t
    v_pred, x_pred = _forward_sample(
        model=model, z=z, t_batch=t_batch, x_pred_prev=x_pred_prev,
        config=config, cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    return z + (t_next - t) * v_pred, x_pred  # z_next = z + h * v_pred（h = t_next - t）


def _sde_step(
    model, z, t, t_next, x_pred_prev,
    config, cfg_scale, self_cond_cfg_scale,
    cond_seq, cond_seq_mask, gamma, generator,
):
    """Per-step SDE-style sampler with hybrid (t-and-step) noise scaling.

    t_back = t * (1 - gamma * h), where h = t_next - t. alpha = 1 - gamma*h is the
    signal-preservation fraction, constant in t. gamma=0 degenerates to a plain ODE step.
    Uniform-N-step equivalence with old multiplicative gamma_old: gamma_hybrid = gamma_old * N.
    """
    h = float(t_next - t)
    alpha = max(0.0, min(1.0, 1.0 - gamma * h))
    t_back = alpha * float(t)
    if z.is_cuda:
        eps = torch.randn(z.shape, dtype=z.dtype, device=z.device) * config.denoiser_noise_scale
    else:
        eps = torch.randn(z.shape, generator=generator, dtype=z.dtype) * config.denoiser_noise_scale
    z_back = restore_cond(alpha * z + (1.0 - alpha) * eps, cond_seq, cond_seq_mask)
    t_batch = torch.full((z.shape[0],), t_back, dtype=z.dtype, device=z.device)
    v_pred, x_pred = _forward_sample(
        model=model, z=z_back, t_batch=t_batch, x_pred_prev=x_pred_prev,
        config=config, cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    return z_back + (t_next - t_back) * v_pred, x_pred
