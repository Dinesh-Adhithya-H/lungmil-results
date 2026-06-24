"""
Phase 2 multimodal fusion models.

Two-phase design — Phase 2 purpose
-------------------------------------
Phase 2 takes the Phase 1 summary tokens (ABMIL-pooled patch representations)
from all available modalities and **fuses** them to produce a joint prediction.

Rather than learning representations from scratch, Phase 2 starts from
already-predictive per-modality encoders (trained in Phase 1) and only learns
how to *combine* them.  This avoids the common failure mode where one dominant
modality overwhelms the others during end-to-end joint training.

Fusion variants
---------------
  early  : All patches from all modalities concatenated → DualGatedPool
  late   : Per-modality ABMIL summaries → learnable weighted combination
  middle : Per-modality ABMIL summaries → cross-modal transformer → pool
  slot   : SetTransformerMIL — PMA seed compression per modality → SAB cross-modal → ABMIL

Classes exported
----------------
EarlyFusionMIL
LateFusionMIL
MiddleFusionMIL
SetTransformerMIL
DualGatedPool
MultiTaskHead
"""

import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import (
    GatedAttentionEncoder,
    ModalFFNEncoder,
    ProjectionHead,
    FFN,
    CrossModalTransformer,
    PMA,
    SAB,
    TemporalSAB,
)

# ── Constants (mirror train_mm_abmil_v7.py) ───────────────────────────────────
HIDDEN_DIM = 256
DROPOUT    = 0.4

P2_MAX_PATCHES  = 2048
P2_MAX_HE_BLOCK = 1024
P1_CLR_PROJ_DIM = 128

TASK_SPEC = {
    "acr_cls":  {"type": "cls",  "label_key": "label",
                 "tte_key": None,             "ev_key": None},
    "acr_surv": {"type": "surv", "label_key": None,
                 "tte_key": "tte_next_acr",   "ev_key": "event_next_acr"},
    "clad":     {"type": "surv", "label_key": None,
                 "tte_key": "clad_time",       "ev_key": "clad_event"},
    "death":    {"type": "surv", "label_key": None,
                 "tte_key": "death_time",      "ev_key": "death_event"},
}


# ── Shared pooling helpers ─────────────────────────────────────────────────────

def _abmil_pool(h, att_V, att_U, att_w):
    gate  = att_V(h) * att_U(h)
    alpha = F.softmax(att_w(gate), dim=0)
    return (alpha * h).sum(0)

def _pool(use_cls, tokens, cls_token, cls_attn_mod, cls_norm,
          att_V, att_U, att_w, device):
    if use_cls:
        cls = cls_token.to(device)
        seq = torch.cat([cls, tokens], dim=0).unsqueeze(0)
        out, _ = cls_attn_mod(seq, seq, seq)
        return cls_norm(out.squeeze(0)[0])
    return _abmil_pool(tokens, att_V, att_U, att_w)


# ── Phase 1 checkpoint loaders ────────────────────────────────────────────────

def _load_p1_encoder(p1_dir: Path, mod: str,
                     trainable: bool = True,
                     use_spatial: bool = False):
    from .phase1 import SingleModalMIL
    from mil.data.registry import _feat_dim

    ckpt_path = p1_dir / mod / "best_model.pt"
    if not ckpt_path.exists():
        # Fallback: pick best available checkpoint using stored val_bacc history.
        # This allows Phase 2 to run even when Phase 1 was interrupted before
        # the final best_model.pt was written (e.g. job killed mid-training).
        ckpt_dir = p1_dir / mod / "checkpoints"
        ckpts = sorted(ckpt_dir.glob("ep*.pt")) if ckpt_dir.exists() else []
        assert ckpts, f"Missing Phase 1 checkpoint: {ckpt_path}"
        best_b, best_cp = -1.0, ckpts[-1]
        for cp in ckpts:
            try:
                data = torch.load(cp, map_location="cpu", weights_only=False)
                hist = data.get("history", {}).get("val_bacc", [])
                b = max(hist) if hist else -1.0
                if b > best_b:
                    best_b, best_cp = b, cp
                del data
            except Exception:
                pass
        print(f"  [warn] {mod}/best_model.pt missing; "
              f"using {best_cp.name} (val_bacc≈{best_b:.4f})")
        ckpt_path = best_cp

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = state["model"] if isinstance(state, dict) and "model" in state else state
    # Detect actual feat_dim from checkpoint to handle legacy format changes
    # (e.g. Clinical was 107 raw features before switching to 408 one-hot tokens).
    w = state.get("encoder.backbone.0.weight")
    actual_feat_dim = int(w.shape[1]) if w is not None else _feat_dim(mod)
    if actual_feat_dim != _feat_dim(mod):
        print(f"  [info] {mod}: checkpoint feat_dim={actual_feat_dim} "
              f"(registry={_feat_dim(mod)}) — using checkpoint dim")
    # Reconstruct with same use_spatial flag so pos_enc buffers exist if needed
    base  = SingleModalMIL(actual_feat_dim, HIDDEN_DIM, DROPOUT, P1_CLR_PROJ_DIM,
                            use_cross_attn=False,
                            use_spatial=(use_spatial and mod == "HE"))
    base.load_state_dict(state, strict=False)   # pos_enc has no params; backbone loads fine
    enc   = base.encoder
    del state, base
    for p in enc.parameters():
        p.requires_grad = trainable
    return enc

def _load_p1_proj_head(p1_dir: Path, mod: str,
                       frozen: bool = True) -> ProjectionHead:
    ckpt_path = p1_dir / mod / "best_model.pt"
    if not ckpt_path.exists():
        ckpt_dir = p1_dir / mod / "checkpoints"
        ckpts = sorted(ckpt_dir.glob("ep*.pt")) if ckpt_dir.exists() else []
        assert ckpts, f"Missing Phase 1 checkpoint: {ckpt_path}"
        best_b, best_cp = -1.0, ckpts[-1]
        for cp in ckpts:
            try:
                data = torch.load(cp, map_location="cpu", weights_only=False)
                hist = data.get("history", {}).get("val_bacc", [])
                b = max(hist) if hist else -1.0
                if b > best_b:
                    best_b, best_cp = b, cp
                del data
            except Exception:
                pass
        print(f"  [warn] {mod}/best_model.pt missing; "
              f"using {best_cp.name} (val_bacc≈{best_b:.4f})")
        ckpt_path = best_cp
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = state["model"] if isinstance(state, dict) and "model" in state else state
    ph    = ProjectionHead(HIDDEN_DIM, P1_CLR_PROJ_DIM)
    ph_state = {k[len("proj_head."):]: v
                for k, v in state.items() if k.startswith("proj_head.")}
    if ph_state: ph.load_state_dict(ph_state)
    else: print(f"  [warn] No proj_head in checkpoint for {mod} — random init")
    del state
    for p in ph.parameters(): p.requires_grad = not frozen
    return ph


# ── Phase 2 task heads ────────────────────────────────────────────────────────

class DualGatedPool(nn.Module):
    """
    Two independent gated-attention ABMIL pools on the same token set.
    Used by early/late/middle fusion so each task pools differently
    from shared features without competing through a single bottleneck.

    cls pathway:  A_cls = softmax(w_cls · (V_cls(x) ⊙ U_cls(x)))
                  r_cls = A_cls^T x  → logit
    surv pathway: A_surv = softmax(w_surv · (V_surv(x) ⊙ U_surv(x)))
                  r_surv = A_surv^T x → hazard
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.cls_V    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.cls_U    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.cls_w    = nn.Linear(hidden_dim, 1, bias=False)
        self.cls_norm = nn.LayerNorm(hidden_dim)
        self.cls_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

        self.surv_V    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.surv_U    = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.surv_w    = nn.Linear(hidden_dim, 1, bias=False)
        self.surv_norm = nn.LayerNorm(hidden_dim)
        self.hazard_head = nn.Linear(hidden_dim, 1, bias=True)
        nn.init.normal_(self.hazard_head.weight, 0.0, 0.01)
        nn.init.zeros_(self.hazard_head.bias)

    def forward(self, x: torch.Tensor, device: torch.device):
        """x: (N, H) — N tokens (patches / modality summaries / cross-modal slots)"""
        A_cls  = torch.softmax(self.cls_w(self.cls_V(x) * self.cls_U(x)),  dim=0)  # (N,1)
        r_cls  = self.cls_norm((A_cls  * x).sum(0))                                 # (H,)
        logit  = self.cls_head(r_cls).squeeze()

        A_surv = torch.softmax(self.surv_w(self.surv_V(x) * self.surv_U(x)), dim=0)
        r_surv = self.surv_norm((A_surv * x).sum(0))
        hazard = self.hazard_head(r_surv).squeeze()

        return logit, hazard, r_cls, r_surv


class MultiTaskHead(nn.Module):
    """
    Generalised N-task head: each task gets a learnable query token
    that cross-attends to all K*M slots → scalar output + representation.
    Output: dict {task_name: (scalar_output, rep_H)}
    """
    def __init__(self, tasks: List[str], hidden_dim: int,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.task_names = tasks
        self.queries = nn.ParameterDict({
            t: nn.Parameter(torch.zeros(1, 1, hidden_dim)) for t in tasks
        })
        for t in tasks:
            nn.init.normal_(self.queries[t], std=0.02)
        self.attns = nn.ModuleDict({
            t: nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
            for t in tasks
        })
        self.norms = nn.ModuleDict({t: nn.LayerNorm(hidden_dim) for t in tasks})
        heads_d = {}
        for t in tasks:
            if TASK_SPEC[t]["type"] == "cls":
                heads_d[t] = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
            else:
                h = nn.Linear(hidden_dim, 1, bias=True)
                nn.init.normal_(h.weight, 0.0, 0.01)
                nn.init.zeros_(h.bias)
                heads_d[t] = h
        self.heads = nn.ModuleDict(heads_d)

    def forward(self, tokens: torch.Tensor, device: torch.device) -> Dict[str, tuple]:
        kv = tokens.unsqueeze(0)   # (1, T, H)
        out: Dict[str, tuple] = {}
        for t in self.task_names:
            q = self.queries[t].to(device)
            r, _ = self.attns[t](q, kv, kv)
            r = self.norms[t](r).squeeze(0).squeeze(0)   # (H,)
            out[t] = (self.heads[t](r).squeeze(), r)
        return out


# ── Fusion variant 1: EarlyFusionMIL ─────────────────────────────────────────

class EarlyFusionMIL(nn.Module):
    """All patches → concat → two separate gated-ABMIL pools (cls / surv)."""
    def __init__(self, encoders, proj_heads, hidden_dim=256, dropout=0.4,
                 modal_dropout=0.3, max_patches_per_mod=P2_MAX_PATCHES,
                 use_cls=False, proj_dim=128, tasks=None):
        super().__init__()
        self.encoders      = nn.ModuleDict(encoders)
        self.proj_heads    = nn.ModuleDict(proj_heads)
        self.modal_dropout = modal_dropout
        self.max_patches   = max_patches_per_mod
        self.use_cls       = use_cls
        _tasks = tasks if tasks else ["acr_cls", "acr_surv"]
        if _tasks == ["acr_cls", "acr_surv"]:
            self.task_head = DualGatedPool(hidden_dim, dropout=dropout)
        else:
            self.task_head = MultiTaskHead(_tasks, hidden_dim, dropout=dropout)

    def forward(self, bags: dict, device: torch.device):
        he_coords = bags.get("HE_coords")
        present_mods = [m for m in self.encoders if bags.get(m) is not None]
        if not present_mods:
            return torch.tensor(0.0, device=device, requires_grad=True)
        # Modal dropout — always keep at least 1 modality
        if self.training:
            kept = [m for m in present_mods if random.random() >= self.modal_dropout]
            if not kept:
                kept = [random.choice(present_mods)]
            present_mods = kept
        # Balance token budget across present modalities
        budget_per_mod = max(1, self.max_patches // len(present_mods))
        patches = []
        for mod in present_mods:
            enc = self.encoders[mod]
            t = bags[mod].to(device, non_blocking=True)
            if t.shape[0] > budget_per_mod:
                t = t[torch.randperm(t.shape[0], device=device)[:budget_per_mod]]
            crds = he_coords if mod == "HE" else None
            patches.append(enc.encode_patches(t, coords=crds))
        H_all = torch.cat(patches, dim=0)  # (N_total, H) balanced across mods
        return self.task_head(H_all, device)


# ── Fusion variant 2: LateFusionMIL ──────────────────────────────────────────

class LateFusionMIL(nn.Module):
    """
    True late fusion: per-modality ABMIL → per-modality cls/surv heads →
    combine decisions with learnable softmax weights.
    Each modality votes independently; the combination is learned.
    When tasks != ["acr_cls","acr_surv"], uses MultiTaskHead over stacked modality reps.
    """
    def __init__(self, encoders, proj_heads, hidden_dim=256, dropout=0.4,
                 modal_dropout=0.3, proj_dim=128, tasks=None):
        super().__init__()
        self.encoders      = nn.ModuleDict(encoders)
        self.modal_dropout = modal_dropout
        _tasks = tasks if tasks else ["acr_cls", "acr_surv"]
        self._use_legacy   = (_tasks == ["acr_cls", "acr_surv"])
        if self._use_legacy:
            self.cls_heads  = nn.ModuleDict({
                m: nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
                for m in encoders})
            self.surv_heads = nn.ModuleDict({
                m: nn.Linear(hidden_dim, 1, bias=True)
                for m in encoders})
            for m in encoders:
                nn.init.normal_(self.surv_heads[m].weight, 0.0, 0.01)
                nn.init.zeros_(self.surv_heads[m].bias)
            self.log_weights = nn.Parameter(torch.zeros(len(encoders)))
            self.mod_index   = {m: i for i, m in enumerate(encoders)}
        else:
            # MultiTaskHead attends to stacked per-modality reps (M, H)
            self.task_head = MultiTaskHead(_tasks, hidden_dim, dropout=dropout)

    def forward(self, bags: dict, device: torch.device):
        he_coords = bags.get("HE_coords")
        reps: dict = {}
        for mod, enc in self.encoders.items():
            t = bags.get(mod)
            if t is None: continue
            if self.training and random.random() < self.modal_dropout: continue
            crds = he_coords if mod == "HE" else None
            rep, _, _ = enc(t.to(device, non_blocking=True), coords=crds)
            reps[mod] = rep
        if not reps:
            return torch.tensor(0.0, device=device, requires_grad=True)

        if not self._use_legacy:
            # MultiTaskHead over stacked modality reps (M, H)
            tokens = torch.stack(list(reps.values()), dim=0)  # (M, H)
            return self.task_head(tokens, device)

        # Legacy: per-modality heads + weighted combination
        cls_logits: dict = {}; surv_hazards: dict = {}; indices: list = []
        for mod, rep in reps.items():
            cls_logits[mod]   = self.cls_heads[mod](rep).squeeze()
            surv_hazards[mod] = self.surv_heads[mod](rep).squeeze()
            indices.append(self.mod_index[mod])

        r_cls  = torch.stack(list(reps.values())).mean(0)
        r_surv = r_cls

        if len(reps) == 1:
            return (list(cls_logits.values())[0],
                    list(surv_hazards.values())[0],
                    r_cls, r_surv)

        idx     = torch.tensor(indices, device=device)
        weights = F.softmax(self.log_weights[idx], dim=0)
        logit   = (weights * torch.stack(list(cls_logits.values()))).sum()
        hazard  = torch.stack(list(surv_hazards.values())).mean()
        return logit, hazard, r_cls, r_surv


# ── Fusion variant 3: MiddleFusionMIL ────────────────────────────────────────

class MiddleFusionMIL(nn.Module):
    """ABMIL per mod → summaries → cross-modal transformer → dual gated-ABMIL per task."""
    def __init__(self, encoders, proj_heads, hidden_dim=256, n_heads=4,
                 n_layers=2, dropout=0.1, modal_dropout=0.3, use_cls=False,
                 use_recon=False, proj_dim=128, tasks=None):
        super().__init__()
        self.encoders      = nn.ModuleDict(encoders)
        self.proj_heads    = nn.ModuleDict(proj_heads)
        self.modal_dropout = modal_dropout
        self.use_recon     = use_recon
        self.transformer   = nn.ModuleList([nn.ModuleDict({
            "attn": nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True),
            "norm": nn.LayerNorm(hidden_dim), "ffn": FFN(hidden_dim, dropout),
        }) for _ in range(n_layers)])
        _tasks = tasks if tasks else ["acr_cls", "acr_surv"]
        if _tasks == ["acr_cls", "acr_surv"]:
            self.task_head = DualGatedPool(hidden_dim, dropout=dropout)
        else:
            self.task_head = MultiTaskHead(_tasks, hidden_dim, n_heads=n_heads, dropout=dropout)
        if use_recon:
            self.recon_decoders = nn.ModuleDict({
                m: nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, hidden_dim))
                for m in encoders})

    def forward(self, bags: dict, device: torch.device):
        he_coords = bags.get("HE_coords")
        summary_dict: Dict[str, torch.Tensor] = {}
        for mod, enc in self.encoders.items():
            t = bags.get(mod)
            if t is None: continue
            if self.training and random.random() < self.modal_dropout: continue
            crds = he_coords if mod == "HE" else None
            rep, _, _ = enc(t.to(device, non_blocking=True), coords=crds)
            summary_dict[mod] = rep
        if not summary_dict:
            return torch.tensor(0.0, device=device, requires_grad=True)
        x = torch.stack(list(summary_dict.values()), dim=0).unsqueeze(0)  # (1, M, H)
        if len(summary_dict) >= 2:
            for L in self.transformer:
                a, _ = L["attn"](x, x, x)
                x    = L["ffn"](L["norm"](x + a))
        tokens = x.squeeze(0)  # (M, H)
        if self.use_recon and summary_dict:
            m = random.choice(list(summary_dict.keys()))
            self._last_recon = F.mse_loss(self.recon_decoders[m](tokens.mean(0)),
                                           summary_dict[m].detach())
        else:
            self._last_recon = None
        return self.task_head(tokens, device)


class SetTransformerMIL(nn.Module):
    """
    Multimodal MIL via Set Transformer seed compression.

    Architecture
    ------------
    Stage 1  Per-modality ModalFFNEncoder: (N, feat_dim) → (N, H).

    Stage 2  Per-modality PMA (Pooling by Multihead Attention):
               K learned seed vectors attend to N patch tokens via standard
               cross-attention → (K, H) per modality.
             Standard softmax over N patches per seed — no GRU, no competitive
             routing, no iterative refinement.  Seeds stay diverse because they
             have distinct learned query weights and independent gradient paths;
             there is no shared GRU hidden state to force convergence.

    Stage 3  Concatenate seeds from all present modalities → (M*K, H).
             SAB (self-attention) lets seeds from different modalities exchange
             information for cross-modal interaction.

    Stage 4  Per-task gated ABMIL over M*K tokens → (H,) task representation.

    Stage 5  Per-task cls / survival head.
    """

    def __init__(self, encoders, hidden_dim: int = 256,
                 n_seeds: int = 16, n_pma_layers: int = 2,
                 n_sab_layers: int = 1, n_heads: int = 4,
                 dropout: float = 0.1, modal_dropout: float = 0.3,
                 max_he_patches: int = P2_MAX_HE_BLOCK,
                 tasks: Optional[List[str]] = None):
        super().__init__()
        self.encoders       = nn.ModuleDict(encoders)
        self.modal_dropout  = modal_dropout
        self.max_he_patches = max_he_patches
        self.n_seeds        = n_seeds
        _tasks = tasks if tasks is not None else ["acr_cls", "acr_surv", "clad", "death"]
        self.task_names     = _tasks

        # Per-modality PMA: compress N patches → K seeds
        self.pma = nn.ModuleDict({
            mod: PMA(hidden_dim, n_seeds, n_heads, n_pma_layers, dropout)
            for mod in encoders
        })

        # Cross-modal SAB: seeds from all modalities self-attend
        self.sab = nn.ModuleList([SAB(hidden_dim, n_heads, dropout)
                                   for _ in range(n_sab_layers)])

        # Per-task gated ABMIL over M*K tokens
        self.abmil_V = nn.ModuleDict({
            t: nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())    for t in _tasks})
        self.abmil_U = nn.ModuleDict({
            t: nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid()) for t in _tasks})
        self.abmil_w = nn.ModuleDict({
            t: nn.Linear(hidden_dim, 1, bias=False)                           for t in _tasks})

        # Per-task output heads
        heads: dict = {}
        for t in _tasks:
            if TASK_SPEC[t]["type"] == "cls":
                heads[t] = nn.Linear(hidden_dim, 1)
            else:
                h = nn.Linear(hidden_dim, 1, bias=True)
                nn.init.normal_(h.weight, 0.0, 0.01)
                nn.init.zeros_(h.bias)
                heads[t] = h
        self.heads = nn.ModuleDict(heads)

    def forward(self, bags: dict, device: torch.device) -> dict:
        he_coords = bags.get("HE_coords")

        # Stage 1+2: per-modality encode + compress to K seeds
        mod_seeds: List[torch.Tensor] = []
        for mod, enc in self.encoders.items():
            t = bags.get(mod)
            if t is None:
                continue
            if self.training and random.random() < self.modal_dropout:
                continue
            t = t.to(device, non_blocking=True)
            if mod == "HE" and t.shape[0] > self.max_he_patches:
                idx = torch.randperm(t.shape[0], device=device)[:self.max_he_patches]
                t = t[idx]
            crds = he_coords if mod == "HE" else None
            h = enc.encode_patches(t, coords=crds)   # (N, H)
            s = self.pma[mod](h)                      # (K, H)
            mod_seeds.append(s)

        if not mod_seeds:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Stage 3: concatenate all modality seeds + cross-modal self-attention
        tokens = torch.cat(mod_seeds, dim=0)          # (M*K, H)
        for layer in self.sab:
            tokens = layer(tokens)

        # Stage 4+5: per-task gated ABMIL → head
        out: dict = {}
        for task in self.task_names:
            gate  = self.abmil_V[task](tokens) * self.abmil_U[task](tokens)  # (M*K, H)
            alpha = torch.softmax(self.abmil_w[task](gate), dim=0)            # (M*K, 1)
            rep   = (alpha * tokens).sum(0)                                   # (H,)
            out[task] = (self.heads[task](rep).squeeze(), rep)

        return out


class LongitudinalMIL(nn.Module):
    """
    Temporal extension of SetTransformerMIL for longitudinal biopsy sequences.

    Architecture
    ------------
    For a patient with T biopsies at days [d_0 < d_1 < ... < d_{T-1}]:

    1. Per biopsy t, per modality m:
       patches → ModalFFNEncoder → PMA → (K, H) seeds

    2. Concatenate all biopsies' seeds: (T*M*K, H), track days per token.

    3. TemporalSAB (causal + ALiBi): tokens attend to history; recent biopsies
       down-weighted by temporal distance from the query token.

    4. Per-task recency-weighted ABMIL:
       alpha_i ∝ exp( w·tanh(V·h_i) − |gamma| * |days_i − anchor| / sigma )
       anchor = last biopsy day for patient-level tasks (acr_surv, clad)
               = current biopsy day for per-biopsy tasks (death, acr_cls)

    Task supervision
    ----------------
    acr_cls  : per biopsy where acr_grade known → BCE loss
    acr_surv : patient-level (acr_days fixed) → one Cox per patient
    clad     : gap-time per biopsy → T Cox contributions per patient
    death    : gap-time per biopsy → T Cox contributions per patient
    """

    def __init__(self, encoders, hidden_dim: int = 256,
                 n_seeds: int = 16, n_pma_layers: int = 2,
                 n_sab_layers: int = 1, n_heads: int = 4,
                 dropout: float = 0.1, modal_dropout: float = 0.3,
                 max_he_patches: int = 2048,
                 tasks: Optional[List[str]] = None):
        super().__init__()
        self.encoders       = nn.ModuleDict(encoders)
        self.modal_dropout  = modal_dropout
        self.max_he_patches = max_he_patches
        self.n_seeds        = n_seeds
        _tasks = tasks if tasks is not None else ["acr_cls", "acr_surv", "clad", "death"]
        self.task_names     = _tasks

        self.pma = nn.ModuleDict({
            mod: PMA(hidden_dim, n_seeds, n_heads, n_pma_layers, dropout)
            for mod in encoders
        })
        self.temporal_sab = TemporalSAB(hidden_dim, n_heads, dropout, n_sab_layers)

        self.abmil_V = nn.ModuleDict({
            t: nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
            for t in _tasks})
        self.abmil_U = nn.ModuleDict({
            t: nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
            for t in _tasks})
        self.abmil_w = nn.ModuleDict({
            t: nn.Linear(hidden_dim, 1, bias=False)
            for t in _tasks})

        # Per-task recency decay (learned scalar)
        self.recency_gamma = nn.ParameterDict({
            t: nn.Parameter(torch.ones(1)) for t in _tasks})

        heads: dict = {}
        for t in _tasks:
            if TASK_SPEC[t]["type"] == "cls":
                heads[t] = nn.Linear(hidden_dim, 1)
            else:
                h = nn.Linear(hidden_dim, 1, bias=True)
                nn.init.normal_(h.weight, 0.0, 0.01)
                nn.init.zeros_(h.bias)
                heads[t] = h
        self.heads = nn.ModuleDict(heads)

    def _abmil_rep(self, task: str, tokens: torch.Tensor,
                   days: torch.Tensor, anchor_day: float) -> torch.Tensor:
        """Recency-weighted ABMIL anchored at anchor_day."""
        gate  = self.abmil_V[task](tokens) * self.abmil_U[task](tokens)  # (N, H)
        raw   = self.abmil_w[task](gate).squeeze(-1)                      # (N,)
        sigma = (days.max() - days.min() + 1.0).clamp(min=1.0)
        bias  = -self.recency_gamma[task].abs() * (days - anchor_day).abs() / sigma
        alpha = torch.softmax(raw + bias, dim=0)                          # (N,)
        return (alpha.unsqueeze(1) * tokens).sum(0)                       # (H,)

    def forward(self, patient_data: dict, device: torch.device) -> dict:
        """
        patient_data:
          'bags_list' : list of T dicts, each {mod_name: tensor or None}
          'days'      : list of T floats (days from first biopsy)
          'records'   : list of T per-biopsy label dicts

        Returns task-keyed dict:
          'acr_surv' : (hazard, rep, acr_t, acr_e)     — patient-level
          'clad'     : [(hazard, t_val, e_val), ...]    — per biopsy
          'death'    : [(hazard, t_val, e_val), ...]    — per biopsy
          'acr_cls'  : [(logit, label), ...]            — labeled biopsies only
        """
        import math as _math

        days_list = patient_data["days"]
        bags_list = patient_data["bags_list"]
        records   = patient_data["records"]
        T         = len(days_list)

        all_seeds: List[torch.Tensor] = []
        all_days:  List[torch.Tensor] = []
        biopsy_ends: List[int]        = []  # token count cumsum after each biopsy

        running_total = 0
        for t_idx in range(T):
            bags  = bags_list[t_idx]
            d_val = float(days_list[t_idx])
            mod_seeds = []
            for mod, enc in self.encoders.items():
                feat = bags.get(mod)
                if feat is None:
                    continue
                if self.training and random.random() < self.modal_dropout:
                    continue
                feat = feat.to(device, non_blocking=True)
                if mod == "HE" and feat.shape[0] > self.max_he_patches:
                    idx  = torch.randperm(feat.shape[0], device=device)[:self.max_he_patches]
                    feat = feat[idx]
                h = enc.encode_patches(feat)    # (N, H)
                s = self.pma[mod](h)             # (K, H)
                mod_seeds.append(s)

            if not mod_seeds:
                biopsy_ends.append(running_total)
                continue

            biopsy_seeds = torch.cat(mod_seeds, dim=0)   # (n_present * K, H)
            n_new        = biopsy_seeds.shape[0]
            all_seeds.append(biopsy_seeds)
            all_days.append(torch.full((n_new,), d_val, dtype=torch.float32, device=device))
            running_total += n_new
            biopsy_ends.append(running_total)

        if not all_seeds:
            dummy = torch.tensor(0.0, device=device, requires_grad=True)
            return {t: dummy for t in self.task_names}

        tokens   = torch.cat(all_seeds, dim=0)    # (total_tokens, H)
        days_tok = torch.cat(all_days,  dim=0)    # (total_tokens,)

        tokens = self.temporal_sab(tokens, days_tok)

        out: dict = {}

        # ACR survival: patient-level, anchor at last biopsy
        if "acr_surv" in self.task_names:
            anchor = days_tok[-1].item()
            rep    = self._abmil_rep("acr_surv", tokens, days_tok, anchor)
            hazard = self.heads["acr_surv"](rep).squeeze()
            acr_t  = next(
                (float(r.get("acr_days", float("nan"))) for r in records
                 if not _math.isnan(float(r.get("acr_days", float("nan"))))),
                float("nan"))
            acr_e  = next(
                (float(r.get("acr_status", float("nan"))) for r in records
                 if not _math.isnan(float(r.get("acr_status", float("nan"))))),
                float("nan"))
            out["acr_surv"] = (hazard, rep, acr_t, acr_e)

        # CLAD and Death: per-biopsy gap-time, causal context
        for task_name, t_key, e_key in [
            ("clad",  "clad_time",  "clad_event"),
            ("death", "death_time", "death_event"),
        ]:
            if task_name not in self.task_names:
                continue
            biopsy_hazards = []
            for t_idx, (rec, end_idx) in enumerate(zip(records, biopsy_ends)):
                if end_idx == 0:
                    continue
                t_val = float(rec.get(t_key, float("nan")))
                e_val = float(rec.get(e_key, float("nan")))
                if _math.isnan(t_val) or t_val < 0:
                    continue
                tok_t  = tokens[:end_idx]
                days_t = days_tok[:end_idx]
                anchor = float(days_list[t_idx])
                rep    = self._abmil_rep(task_name, tok_t, days_t, anchor)
                hazard = self.heads[task_name](rep).squeeze()
                e_safe = float(e_val) if not _math.isnan(float(e_val)) else 0.0
                biopsy_hazards.append((hazard, t_val, e_safe))
            out[task_name] = biopsy_hazards

        # ACR classification: per labeled biopsy, causal context
        if "acr_cls" in self.task_names:
            cls_out = []
            for t_idx, (rec, end_idx) in enumerate(zip(records, biopsy_ends)):
                label = rec.get("label")
                if label is None or end_idx == 0:
                    continue
                tok_t  = tokens[:end_idx]
                days_t = days_tok[:end_idx]
                anchor = float(days_list[t_idx])
                rep    = self._abmil_rep("acr_cls", tok_t, days_t, anchor)
                logit  = self.heads["acr_cls"](rep).squeeze()
                cls_out.append((logit, label))
            out["acr_cls"] = cls_out

        return out
