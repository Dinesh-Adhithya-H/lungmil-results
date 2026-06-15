"""Loss functions: hinge, Cox-Breslow, survival contrastive, NT-Xent, SupCon."""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


# ── Classification ────────────────────────────────────────────────────────────

def hinge_loss(logit: torch.Tensor, target: torch.Tensor,
               cw: Tuple[float, float]) -> torch.Tensor:
    """Weighted hinge loss.  target ∈ {0,1};  cw = (w_neg, w_pos)."""
    y = 2.0 * target - 1.0
    w = torch.where(target > 0.5,
                    logit.new_full((), cw[1]),
                    logit.new_full((), cw[0]))
    return (w * torch.clamp(1.0 - y * logit, min=0.0)).mean()


def compute_class_weights(records) -> Tuple[float, float]:
    """Balanced class weights via sklearn, capped at 20× to avoid extreme weighting."""
    import numpy as np
    from sklearn.utils.class_weight import compute_class_weight
    labels = [r["label"] for r in records if r.get("label") in (0, 1)]
    if not labels or len(set(labels)) < 2:
        return (1.0, 1.0)
    y = np.array(labels)
    weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y)
    w_neg, w_pos = float(min(weights[0], 20.0)), float(min(weights[1], 20.0))
    return (w_neg, w_pos)


# ── Survival ──────────────────────────────────────────────────────────────────

def cox_breslow_loss(cox_buffer) -> Optional[torch.Tensor]:
    """
    Breslow approximation of Cox partial negative log-likelihood.
    cox_buffer: list of (hazard_tensor, time_float, event_float)
    Returns scalar loss or None if no events.
    """
    if not cox_buffer:
        return None
    hazards = torch.stack([h for h, t, e in cox_buffer])
    dev     = hazards.device
    times   = torch.tensor([t for h, t, e in cox_buffer], dtype=torch.float32, device=dev)
    events  = torch.tensor([e for h, t, e in cox_buffer], dtype=torch.float32, device=dev)
    if events.sum() == 0:
        return None
    order   = torch.argsort(times)
    h_s     = hazards[order]
    e_s     = events[order]
    h_max   = h_s.max().detach()
    exp_h   = torch.exp(h_s - h_max)
    suffix_exp = torch.cumsum(exp_h.flip(0), dim=0).flip(0)
    log_risk   = torch.log(suffix_exp + 1e-9) + h_max
    nll = (log_risk - h_s) * e_s
    return nll.sum() / e_s.sum().clamp(min=1)


def c_index(hazards, times, events) -> float:
    """Harrell's C-index via lifelines.utils.concordance_index."""
    from lifelines.utils import concordance_index as _ci
    try:
        return float(_ci(times, [-h for h in hazards], events))
    except Exception:
        return 0.5


def surv_con_loss(
    z:      torch.Tensor,
    ttes:   torch.Tensor,
    events: torch.Tensor,
    tau:      float = 0.1,
    tau_time: float = 90.0,
) -> Optional[torch.Tensor]:
    """
    Soft temporal supervised contrastive loss for survival.
    Positive weight w_ij = exp(-|T_i - T_j| / tau_time) if both δ=1, else 0.
    Censored samples appear only in the denominator.
    """
    N = z.shape[0]
    if N < 2:
        return None
    dev = z.device
    sim = z @ z.T / tau

    T      = ttes.float()
    ev     = events.float()
    dT     = (T.unsqueeze(1) - T.unsqueeze(0)).abs()
    w_time = torch.exp(-dT / tau_time)
    ev_mat = ev.unsqueeze(1) * ev.unsqueeze(0)
    W      = w_time * ev_mat
    W.fill_diagonal_(0.0)

    self_mask = torch.eye(N, dtype=torch.bool, device=dev)
    sim_max   = sim.detach().max(dim=1, keepdim=True).values
    exp_sim   = torch.exp(sim - sim_max) * (~self_mask)
    log_denom = torch.log(exp_sim.sum(dim=1) + 1e-8) + sim_max.squeeze(1)

    anchor_mask = ev > 0.5
    if anchor_mask.sum() == 0:
        return None
    pos_weight_sum = W[anchor_mask].sum(dim=1)
    valid = pos_weight_sum > 1e-6
    if valid.sum() == 0:
        return None

    log_num  = sim[anchor_mask] - sim_max[anchor_mask]
    per_pair = W[anchor_mask] * (log_num - log_denom[anchor_mask].unsqueeze(1))
    per_anch = -per_pair.sum(dim=1) / pos_weight_sum.clamp(min=1e-8)
    return per_anch[valid].mean()


def surv_rank_loss(
    hazards:   torch.Tensor,
    ttes:      torch.Tensor,
    events:    torch.Tensor,
    max_pairs: int = 2048,
) -> Optional[torch.Tensor]:
    """
    Pairwise ranking loss (Luck et al. / DeepHit-style):
    L = -Σ_{i,j: T_i < T_j, δ_i=1} log σ(h_i − h_j)
    """
    N = len(hazards)
    if N < 2:
        return None
    T  = ttes.float()
    ev = events.float()
    Ti = T.unsqueeze(1); Tj = T.unsqueeze(0)
    di = ev.unsqueeze(1)
    mask = (Ti < Tj) & (di > 0.5)
    idx  = mask.nonzero(as_tuple=False)
    if idx.shape[0] == 0:
        return None
    if idx.shape[0] > max_pairs:
        perm = torch.randperm(idx.shape[0], device=hazards.device)[:max_pairs]
        idx  = idx[perm]
    hi = hazards[idx[:, 0]]
    hj = hazards[idx[:, 1]]
    return -torch.log(torch.sigmoid(hi - hj) + 1e-8).mean()


# ── Contrastive ───────────────────────────────────────────────────────────────

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor,
                 tau: float) -> Optional[torch.Tensor]:
    """NT-Xent (SimCLR) for N paired augmented views."""
    N = z1.shape[0]
    if N < 2:
        return None
    z   = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / tau
    pos_idx = torch.cat([torch.arange(N, 2*N, device=z.device),
                         torch.arange(0, N,   device=z.device)])
    self_mask = torch.eye(2*N, dtype=torch.bool, device=z.device)
    sim_max   = sim.detach().max(dim=1, keepdim=True).values
    exp_sim   = torch.exp(sim - sim_max) * (~self_mask).float()
    log_denom = torch.log(exp_sim.sum(dim=1) + 1e-8)
    pos_sim   = sim[torch.arange(2*N, device=z.device), pos_idx]
    return -(pos_sim - sim_max.squeeze(1) - log_denom).mean()


def temporal_ordered_clr_loss(
    z:              torch.Tensor,
    disease_times:  torch.Tensor,
    tau_temp:       float = 0.15,
    tau_time:       float = 180.0,
    uniform_floor:  float = 0.01,
) -> torch.Tensor:
    """
    Temporal-ordered contrastive loss with symmetric soft targets + uniform floor.
    z: (B, D) L2-normalised;  disease_times: (B,) signed days.
    """
    if z.shape[0] < 2:
        return z.new_tensor(0.0)
    B   = z.shape[0]
    sim = (z @ z.T) / tau_temp
    dt  = (disease_times[:, None] - disease_times[None, :]).abs()
    target = torch.exp(-dt / tau_time) + uniform_floor
    target.fill_diagonal_(0.0)
    target = target / target.sum(dim=1, keepdim=True).clamp(min=1e-8)
    loss_fwd = -(target * F.log_softmax(sim, dim=1)).sum(dim=1)
    loss_bwd = -(target * F.log_softmax(sim, dim=0)).sum(dim=0)
    return (0.5 * (loss_fwd + loss_bwd)).mean()


def batch_supcon_loss(
    buffer:                  List[Tuple[torch.Tensor, int, str, str]],
    tau:                     float,
    cw:                      Tuple[float, float],
    min_multimodal_stems:    int = 1,
) -> Optional[torch.Tensor]:
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).
    buffer: list of (L2-normalised vector, label, stem, mod).
    Positives: same stem + diff modality OR same label + diff stem.
    """
    stem_counts: dict = {}
    for _, _, s, _ in buffer:
        stem_counts[s] = stem_counts.get(s, 0) + 1
    if sum(1 for c in stem_counts.values() if c >= 2) < min_multimodal_stems:
        return None
    B = len(buffer)
    if B < 2:
        return None
    zs     = torch.stack([b[0] for b in buffer])
    dev    = zs.device
    labels = torch.tensor([b[1] for b in buffer], dtype=torch.long, device=dev)
    stems  = [b[2] for b in buffer]
    mods   = [b[3] for b in buffer]
    stem_vocab = {s: i for i, s in enumerate(dict.fromkeys(stems))}
    mod_vocab  = {m: i for i, m in enumerate(dict.fromkeys(mods))}
    stem_ids   = torch.tensor([stem_vocab[s] for s in stems], dtype=torch.long, device=dev)
    mod_ids    = torch.tensor([mod_vocab[m]  for m in mods],  dtype=torch.long, device=dev)
    self_mask  = torch.eye(B, dtype=torch.bool, device=dev)
    same_label = labels.unsqueeze(0) == labels.unsqueeze(1)
    same_stem  = stem_ids.unsqueeze(0) == stem_ids.unsqueeze(1)
    diff_mod   = mod_ids.unsqueeze(0)  != mod_ids.unsqueeze(1)
    pos_mask   = ((same_stem & diff_mod) | same_label) & ~self_mask
    sims       = torch.matmul(zs, zs.T) / tau
    exp_sims   = torch.exp(sims - sims.detach().max(dim=1, keepdim=True).values)
    exp_sims   = exp_sims.masked_fill(self_mask, 0.0)
    has_grad   = torch.tensor([b[0].requires_grad for b in buffer],
                               dtype=torch.bool, device=dev)
    anchor_mask = has_grad & pos_mask.any(dim=1)
    if not anchor_mask.any():
        return None
    cw_sum  = cw[0] + cw[1] + 1e-8
    weights = torch.where(labels == 1,
                          torch.full((B,), cw[1] / cw_sum, device=dev),
                          torch.full((B,), cw[0] / cw_sum, device=dev))
    denom     = exp_sims.sum(dim=1, keepdim=True) + 1e-8
    log_probs = torch.log(exp_sims / denom + 1e-8)
    pos_f     = pos_mask.float()
    pos_count = pos_f.sum(dim=1).clamp(min=1)
    per_anc   = -(log_probs * pos_f).sum(dim=1) / pos_count
    return (per_anc * weights)[anchor_mask].mean()


def attention_transfer_loss(
    alpha_self:  torch.Tensor,
    alpha_cross: torch.Tensor,
    tau:   float = 0.5,
    top_k: int   = 50,
) -> torch.Tensor:
    """Attention Transfer (Zagoruyko & Komodakis, ICLR 2017) for MIL.
    KL divergence between teacher-guided and student gate distributions."""
    N      = alpha_self.shape[0]
    k      = min(top_k, N)
    top_idx   = alpha_cross.topk(k, dim=0).indices
    p_target  = F.softmax(alpha_cross[top_idx] / tau, dim=0)
    p_student = F.softmax(alpha_self[top_idx]  / tau, dim=0)
    return F.kl_div(p_student.log(), p_target.detach(), reduction="batchmean")


def crd_loss_fn(r_cross: torch.Tensor,
                r_teacher: torch.Tensor) -> torch.Tensor:
    """CRD-style cosine distance (He et al., ICLR 2020). Loss = 1 − cos_sim ∈ [0,2]."""
    r_c = F.normalize(r_cross.float(),   dim=0)
    r_t = F.normalize(r_teacher.float(), dim=0).detach()
    return 1.0 - (r_c * r_t).sum()
