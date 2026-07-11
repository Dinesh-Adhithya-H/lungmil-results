"""
interpretable_mil/model.py
==========================
Architecture: H [N,D] -> assign [N,K] -> histogram [K] -> head -> logits

Head options
------------
  linear    : logit = w^T h  (each prototype independently predicts)
  mlp       : logit = MLP(h)
  quadratic : logit = w^T h + h^T W h   ← recommended
              W[i,j] = interaction weight between proto i and proto j.
              Contextualized attention:
                α_k = (w_k + Σ_j W[k,j]*h_j) * h_k
              α_k is high ONLY when proto k is present AND its interaction
              partners are also present — solves the co-occurrence problem.

The quadratic head is the key contribution: a linear head cannot distinguish
"proto A alone" from "proto A with proto B" because the attention score h[A]
is the same in both cases. The quadratic interaction term W[A,B]*h_A*h_B is
zero when either prototype is absent, making the attention score fully
conditional on co-occurrence.

Regularisation: L1 penalty on W (off-diagonal) encourages sparse interactions
— only a few prototype pairs dominate, making rules directly readable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MILOutput:
    logits:       torch.Tensor              # [1, n_classes] or [1] for surv
    probs:        torch.Tensor              # [1, n_classes] or [1] for surv
    histogram:    torch.Tensor              # [K]  normalised histogram
    assign:       torch.Tensor              # [N, K] per-patch assignments
    attn_weights: Optional[torch.Tensor] = None  # [K] contextualized scores (quadratic only)


# ─────────────────────────────────────────────────────────────────────────────
# Prototype assigner
# ─────────────────────────────────────────────────────────────────────────────

class PrototypeAssigner(nn.Module):
    def __init__(self, in_dim, K, temp_init=0.1, temp_min=0.02, temp_max=0.5):
        super().__init__()
        self.K = K; self.temp_min = temp_min; self.temp_max = temp_max
        self.prototypes = nn.Parameter(torch.zeros(K, in_dim))
        self.log_temp   = nn.Parameter(torch.tensor(temp_init).log())

    @torch.no_grad()
    def init_from_instances(self, H):
        N = H.shape[0]
        idx = torch.randperm(N, device=H.device)[:self.K] if N >= self.K \
              else torch.randint(0, N, (self.K,), device=H.device)
        self.prototypes.data.copy_(F.normalize(H[idx].float(), dim=-1))

    def temperature(self):
        return self.log_temp.exp().clamp(self.temp_min, self.temp_max)

    def forward(self, H):
        H_n = F.normalize(H.float(), dim=-1)
        P_n = F.normalize(self.prototypes, dim=-1)
        sim = H_n @ P_n.T
        return F.softmax(sim / self.temperature(), dim=-1)  # [N, K]


# ─────────────────────────────────────────────────────────────────────────────
# Heads
# ─────────────────────────────────────────────────────────────────────────────

class LinearHead(nn.Module):
    def __init__(self, K, n_classes):
        super().__init__()
        self.fc = nn.Linear(K, n_classes)

    def forward(self, h):
        return self.fc(h.unsqueeze(0))   # [1, n_classes]

    def attn_weights(self, h):
        """Individual prototype importances: w_k * h_k  (no interactions)."""
        w = self.fc.weight.T   # [K, n_classes]
        return (w * h.unsqueeze(-1)).squeeze(-1) if w.shape[1] == 1 \
               else w[:, 1] * h   # [K] — weight toward positive class


class MLPHead(nn.Module):
    def __init__(self, K, n_classes, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(K, hidden), nn.ReLU(),
                                  nn.Linear(hidden, n_classes))

    def forward(self, h):
        return self.net(h.unsqueeze(0))   # [1, n_classes]

    def attn_weights(self, h):
        return None   # MLP not directly interpretable per-prototype


class QuadraticHead(nn.Module):
    """
    logit = w^T h  +  h^T W_off h  +  bias

    W is constrained symmetric and zero-diagonal (off-diagonal only):
        W_off = triu(W, 1) + triu(W, 1)^T

    This ensures W[i,i] = 0 (no self-interaction) and W[i,j] = W[j,i].

    Gradient w.r.t. h_k (= effective weight of proto k given context h):
        dlogit/dh_k = w_k + 2 * Σ_{j≠k} W_off[k,j] * h_j
                    = w_k + (W_off + W_off^T)[k,:] @ h
                    = w_k + 2 * W_off[k,:] @ h   (symmetric)

    Contextualized attention:
        α_k = (dlogit/dh_k) * h_k

    α_k is high only when proto k is present (h_k > 0) AND the effective
    weight is high (either via individual weight w_k or interaction W_off[k,j]*h_j).

    For survival (n_classes=1): output is hazard score, no bias.
    """
    def __init__(self, K: int, n_classes: int = 2, l1_reg: float = 1e-3,
                 task: str = "cls"):
        super().__init__()
        self.K         = K
        self.n_classes = n_classes
        self.l1_reg    = l1_reg
        self.task      = task

        # Individual weights: one per prototype per output
        n_out = 1 if task == "surv" or n_classes == 1 else n_classes
        self.w    = nn.Parameter(torch.zeros(K, n_out))
        # Upper-triangular interaction matrix (off-diagonal terms only)
        self.W_ut = nn.Parameter(torch.zeros(K, K) * 0.01)
        if task == "cls":
            self.bias = nn.Parameter(torch.zeros(n_out))
        else:
            self.register_parameter("bias", None)

        nn.init.normal_(self.w,    std=0.01)
        nn.init.normal_(self.W_ut, std=0.001)

    def _W_off(self):
        """Symmetric zero-diagonal interaction matrix from upper triangle."""
        triu = torch.triu(self.W_ut, diagonal=1)
        return triu + triu.T   # [K, K] symmetric, zero diagonal

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [K] → [1, n_classes] or [1] for surv."""
        W_off  = self._W_off()
        linear = h @ self.w                    # [n_out]
        quad   = (h @ W_off @ h).unsqueeze(0)  # [1]  scalar interaction term
        if self.n_classes > 1 and self.task == "cls":
            # Broadcast interaction equally to all classes (or learn per-class W if needed)
            logit = linear + quad + (self.bias if self.bias is not None else 0)
            return logit.unsqueeze(0)   # [1, n_classes]
        else:
            logit = (linear + quad).squeeze()
            if self.bias is not None:
                logit = logit + self.bias.squeeze()
            return logit.unsqueeze(0)   # [1]

    def attn_weights(self, h: torch.Tensor) -> torch.Tensor:
        """
        Contextualized attention: α_k = (w_k + 2*W_off[k,:]*h) * h_k

        α_k is high ONLY when:
          1. h_k > 0  (prototype k is present), AND
          2. effective_weight_k > 0  (either individually predictive,
             OR an interaction partner j is also present: W[k,j]*h_j > 0)

        When proto A is alone (h_B=0): α_A = (w_A + W[A,B]*0) * h_A = w_A*h_A
        When proto A+B co-occur:        α_A = (w_A + W[A,B]*h_B) * h_A  ← larger

        Returns [K] — one score per prototype, for positive class (or hazard).
        """
        W_off = self._W_off()
        # effective weight for positive class / hazard (last column or only column)
        w_pos = self.w[:, -1]            # [K]
        effective_w = w_pos + 2.0 * (W_off @ h)   # [K]
        return effective_w * h           # [K]

    def interaction_rules(self, top_k: int = 10) -> List[dict]:
        """
        Extract the top-k prototype pair interactions as human-readable rules.

        Returns list of dicts sorted by |W[i,j]|:
            {proto_i, proto_j, weight, rule_str}
        Positive weight → co-occurrence promotes disease.
        Negative weight → co-occurrence suppresses disease.
        """
        W_off = self._W_off().detach().cpu()
        # Upper triangle only (symmetric, no double-counting)
        rows, cols = torch.triu_indices(self.K, self.K, offset=1)
        vals = W_off[rows, cols]
        order = vals.abs().argsort(descending=True)

        rules = []
        for idx in order[:top_k]:
            i, j = int(rows[idx]), int(cols[idx])
            w = float(vals[idx])
            direction = "→ disease (label=1)" if w > 0 else "→ healthy (label=0)"
            rules.append({
                "proto_i": i, "proto_j": j,
                "weight":  round(w, 4),
                "rule":    f"proto_{i} AND proto_{j}  {direction}",
            })
        return rules

    def individual_weights(self) -> torch.Tensor:
        """Returns [K] individual weights for positive class / hazard."""
        return self.w[:, -1].detach().cpu()

    def explain_patient(self, h: torch.Tensor) -> List[dict]:
        """
        Full per-prototype breakdown for one patient.

        For each prototype k, returns:
          presence        : h_k                           (how much is it there?)
          individual      : w_k * h_k                    (effect if alone)
          interaction_sum : (Σ_j W[k,j]*h_j) * h_k      (effect of co-occurrences)
          net_contrib     : α_k = (w_k + 2*W[k,:]*h)*h_k (total contextual contribution)
          role            : "driver" | "suppressor" | "neutral" | "absent"

        This makes ALL logical patterns readable:
          - AND(A,B)→+1: both show  role="driver" with interaction_sum > 0
          - A AND NOT B: A shows role="driver"; when B present, A flips to "suppressor"
          - XOR: individual>0 for both, but interaction_sum<0 when both present
        """
        W_off = self._W_off()
        w_pos = self.w[:, -1]
        h_cpu = h.detach().cpu()
        W_cpu = W_off.detach().cpu()
        w_cpu = w_pos.detach().cpu()

        records = []
        for k in range(self.K):
            pres  = float(h_cpu[k])
            indiv = float(w_cpu[k] * pres)
            inter = float((W_cpu[k] @ h_cpu - W_cpu[k, k] * pres) * pres)  # off-diag
            net   = indiv + 2.0 * inter   # = α_k
            if pres < 0.01:
                role = "absent"
            elif abs(net) < 0.01:
                role = "neutral"
            elif net > 0:
                role = "driver"
            else:
                role = "suppressor"
            # Which other prototypes are driving the interaction?
            partners = sorted(
                [(j, float(W_cpu[k, j] * h_cpu[j]))
                 for j in range(self.K) if j != k and abs(W_cpu[k, j] * h_cpu[j]) > 0.01],
                key=lambda x: -abs(x[1]))[:3]
            records.append({
                "proto":          k,
                "presence":       round(pres,  4),
                "individual":     round(indiv, 4),
                "interaction_sum":round(inter, 4),
                "net_contrib":    round(net,   4),
                "role":           role,
                "top_partners":   partners,
            })
        return sorted(records, key=lambda r: -abs(r["net_contrib"]))

    def l1_loss(self) -> torch.Tensor:
        """Sparsity regularisation on off-diagonal interaction weights."""
        return self.l1_reg * torch.triu(self.W_ut, diagonal=1).abs().sum()


class CoxQuadraticHead(QuadraticHead):
    """QuadraticHead specialised for survival (no bias, single hazard output)."""
    def __init__(self, K: int, l1_reg: float = 1e-3):
        super().__init__(K, n_classes=1, l1_reg=l1_reg, task="surv")


# ─────────────────────────────────────────────────────────────────────────────
# InterpretableMIL
# ─────────────────────────────────────────────────────────────────────────────

class InterpretableMIL(nn.Module):
    """
    Prototype-histogram MIL with optional quadratic interaction head.

    head options:
      "linear"    — original, no interaction modelling
      "mlp"       — non-linear, not interpretable per-prototype
      "quadratic" — recommended: explicit pairwise interaction weights
                    + fully interpretable contextualized attention scores

    For task="surv": always uses a single-output hazard head.
    For task="cls":  uses n_classes outputs.
    """
    def __init__(self, in_dim: int = 1024, K: int = 16, n_classes: int = 2,
                 head: str = "quadratic", mlp_hidden: int = 32,
                 proj_dim: Optional[int] = None, temp_init: float = 0.1,
                 task: str = "cls", l1_reg: float = 1e-3):
        super().__init__()
        assert head in ("linear", "mlp", "quadratic"), \
            f"head must be linear|mlp|quadratic, got {head!r}"
        self.task = task; self.K = K; self.n_classes = n_classes; self.in_dim = in_dim

        if proj_dim is not None:
            self.proj     = nn.Sequential(nn.Linear(in_dim, proj_dim),
                                           nn.LayerNorm(proj_dim), nn.GELU())
            assign_dim = proj_dim
        else:
            self.proj     = nn.Identity()
            assign_dim = in_dim

        self.assigner = PrototypeAssigner(assign_dim, K, temp_init=temp_init)

        if task == "surv":
            if head == "quadratic":
                self.head = CoxQuadraticHead(K, l1_reg=l1_reg)
            else:
                # Fallback: linear Cox
                self.head = _CoxLinear(K)
        else:
            if head == "quadratic":
                self.head = QuadraticHead(K, n_classes, l1_reg=l1_reg, task="cls")
            elif head == "linear":
                self.head = LinearHead(K, n_classes)
            else:
                self.head = MLPHead(K, n_classes, hidden=mlp_hidden)

    def init_prototypes(self, H: torch.Tensor):
        with torch.no_grad():
            H_proj = self.proj(H.float())
        self.assigner.init_from_instances(H_proj)

    def forward(self, H: torch.Tensor) -> MILOutput:
        """H: [N, in_dim]"""
        H_proj = self.proj(H.float())
        assign = self.assigner(H_proj)        # [N, K]
        h      = assign.mean(dim=0)           # [K] histogram

        logits = self.head(h)                 # [1, n_classes] or [1]
        probs  = F.softmax(logits, dim=-1) if self.task == "cls" else logits

        # Contextualized attention weights (quadratic / linear heads only)
        attn = None
        if hasattr(self.head, "attn_weights"):
            attn = self.head.attn_weights(h)  # [K]

        return MILOutput(logits=logits, probs=probs,
                         histogram=h, assign=assign, attn_weights=attn)

    def interaction_rules(self, top_k: int = 10):
        """Human-readable prototype interaction rules (quadratic head only)."""
        if isinstance(self.head, QuadraticHead):
            return self.head.interaction_rules(top_k)
        return []

    def prototype_weights(self) -> Optional[torch.Tensor]:
        """[K] individual weights toward positive class (all head types)."""
        if isinstance(self.head, QuadraticHead):
            return self.head.individual_weights()
        if hasattr(self.head, "fc"):
            w = self.head.fc.weight   # [n_classes, K] or [1, K]
            return w[-1].detach().cpu() if w.shape[0] > 1 else w[0].detach().cpu()
        return None

    def regularization_loss(self) -> torch.Tensor:
        """L1 regularisation on interaction weights (quadratic head only)."""
        if isinstance(self.head, QuadraticHead):
            return self.head.l1_loss()
        return torch.tensor(0.0)

    def explain_patient(self, H: torch.Tensor) -> List[dict]:
        """
        Per-prototype explanation for one patient.
        Runs forward pass, then decomposes prediction into:
          individual contribution  (prototype alone)
          interaction contribution (co-occurrence effect)
          net contribution         (combined, with sign)
          role: driver | suppressor | neutral | absent

        Works for ALL logical patterns (AND, OR, XOR, A AND NOT B, ...).
        Only available for quadratic head.
        """
        if not isinstance(self.head, QuadraticHead):
            raise ValueError("explain_patient requires head='quadratic'")
        with torch.no_grad():
            out = self.forward(H)
        return self.head.explain_patient(out.histogram)


class _CoxLinear(nn.Module):
    """Linear Cox head (fallback when head!='quadratic' and task='surv')."""
    def __init__(self, K):
        super().__init__()
        self.fc = nn.Linear(K, 1, bias=False)
        nn.init.normal_(self.fc.weight, 0.0, 0.01)

    def forward(self, h):
        return self.fc(h.unsqueeze(0)).squeeze(-1)   # [1]

    def attn_weights(self, h):
        return self.fc.weight.squeeze() * h          # [K]
