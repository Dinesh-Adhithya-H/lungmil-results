"""
pma_gmm_experiment.py — PMA GMM clustering (Set Transformer paper, Section 5.2).

The paper's key insight: train seeds with GMM negative log-likelihood, not MSE.
  loss = -mean_i  log  Σ_k (1/K) N(x_i | μ_k, σ²I)
Seeds become cluster MEANS in embedding space — pulled by the data distribution.

Architecture (paper): ISAB × 2 → PMA_K → GMM NLL
FFN encoder (pre-trained, frozen) maps raw patches → 256-dim embeddings.
ISAB + PMA + σ are trained from scratch with GMM loss.

Compare variants: softmax (b=0) vs b-cos (b=1,2,4,8) in all attention ops.
"""
from __future__ import annotations
import argparse, json, math, sys
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from mil.data.splits   import build_splits_multitask
from mil.data.registry import _feat_key
from mil.models.builders import build_model_v8

SAMPLES_DIR = Path("/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples")
SPLITS_CSV  = Path("/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")
RESULTS_DIR = _ROOT / "results/mm_abmil_v8"

TARGET_MODS  = ["HE", "CT"]
MAX_PATCHES  = 512
HIDDEN_DIM   = 256
N_SEEDS      = 16
N_INDUCERS   = 64   # inducing points per ISAB layer
N_HEADS      = 4
TRAIN_EPOCHS = 50
LR           = 1e-3
B_VALUES     = [1, 2, 4, 8]


# ── Building blocks ────────────────────────────────────────────────────────────

class FFN(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net  = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(),
                                   nn.Linear(dim * 2, dim))
        self.norm = nn.LayerNorm(dim)
    def forward(self, x): return self.norm(x + self.net(x))


class MAB(nn.Module):
    """Multi-head Attention Block: MAB(X,Y) = LN(H + rFF(H)), H = LN(X + MHA(X,Y,Y)).
    b=0 → standard softmax;  b>0 → b-cos: ReLU(cos(q,k))^b / sum.
    """
    def __init__(self, dim: int, n_heads: int = 4, b: float = 0.0):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads
        self.b        = b
        self.proj_q   = nn.Linear(dim, dim, bias=False)
        self.proj_k   = nn.Linear(dim, dim, bias=False)
        self.proj_v   = nn.Linear(dim, dim, bias=False)
        self.proj_o   = nn.Linear(dim, dim, bias=False)
        self.norm1    = nn.LayerNorm(dim)
        self.ffn      = FFN(dim)

    def _attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """q: (nh, Nq, hd)  k/v: (nh, Nk, hd)  →  (nh, Nq, hd)"""
        if self.b == 0:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            w = F.softmax(scores, dim=-1)
        else:
            q_n = F.normalize(q, dim=-1)
            k_n = F.normalize(k, dim=-1)
            raw = F.relu(q_n @ k_n.transpose(-2, -1)).pow(self.b)
            w   = raw / (raw.sum(-1, keepdim=True) + 1e-9)
        return w @ v

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x: (Nx, D) queries  y: (Ny, D) keys/values"""
        Nx, D = x.shape
        Ny    = y.shape[0]
        nh, hd = self.n_heads, self.head_dim
        q = self.proj_q(x).view(Nx, nh, hd).transpose(0, 1)   # (nh, Nx, hd)
        k = self.proj_k(y).view(Ny, nh, hd).transpose(0, 1)
        v = self.proj_v(y).view(Ny, nh, hd).transpose(0, 1)
        out = self._attn(q, k, v).transpose(0, 1).contiguous().view(Nx, D)
        out = self.proj_o(out)
        h   = self.norm1(x + out)
        return self.ffn(h)


class ISAB(nn.Module):
    """Induced Set Attention Block: ISAB(X) = MAB(X, MAB(I, X))
    Uses m learnable inducing points for O(Nm) complexity instead of O(N²).
    """
    def __init__(self, dim: int, n_heads: int = 4, m: int = 64, b: float = 0.0):
        super().__init__()
        self.ind  = nn.Parameter(torch.randn(m, dim) * 0.02)
        self.mab1 = MAB(dim, n_heads, b)   # inducing points attend to input
        self.mab2 = MAB(dim, n_heads, b)   # input attends to compressed inducing

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        i_prime = self.mab1(self.ind, x)   # (m, D)  inducing → input
        return self.mab2(x, i_prime)       # (N, D)  input → inducing


class PMA(nn.Module):
    """Pooling by Multihead Attention: PMA_K(X) = MAB(S, rFF(X)).
    K learnable seed vectors attend to rFF(X) → K cluster representations.
    """
    def __init__(self, dim: int, K: int = 16, n_heads: int = 4, b: float = 0.0):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(K, dim) * 0.02)
        self.rff   = FFN(dim)                  # rFF applied to X first (paper)
        self.mab   = MAB(dim, n_heads, b)      # seeds attend to rFF(X)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, D) → seeds: (K, D)"""
        kv = self.rff(x)
        return self.mab(self.seeds, kv)        # (K, D)


class GMMClusterModel(nn.Module):
    """Set Transformer clustering model.
    Architecture: ISAB × 2 → PMA_K
    Loss: GMM negative log-likelihood with isotropic Gaussian components.
    """
    def __init__(self, dim: int = 256, K: int = 16,
                 n_heads: int = 4, m: int = 64, b: float = 0.0):
        super().__init__()
        self.K         = K
        self.dim       = dim
        self.isab1     = ISAB(dim, n_heads, m, b)
        self.isab2     = ISAB(dim, n_heads, m, b)
        self.pma       = PMA(dim, K, n_heads, b)
        # Shared learnable log-σ (scalar) — starts at σ=1
        self.log_sigma = nn.Parameter(torch.zeros(1))

    def cluster_means(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, D) → μ: (K, D)"""
        h = self.isab1(x)
        h = self.isab2(h)
        return self.pma(h)

    def gmm_nll(self, x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """GMM negative log-likelihood.
        x: (N, D)  mu: (K, D)
        Assumes uniform mixture weights π_k = 1/K, isotropic Gaussian N(μ_k, σ²I).
        """
        sigma = self.log_sigma.exp().clamp(min=1e-4)
        D     = self.dim

        # log p(x_i | k) = -D/2 log(2π) - D log(σ) - ||x_i - μ_k||² / (2σ²)
        diff   = x.unsqueeze(1) - mu.unsqueeze(0)        # (N, K, D)
        sq_dist = diff.pow(2).sum(-1)                     # (N, K)
        log_p_xk = (- 0.5 * sq_dist / sigma ** 2
                    - D * self.log_sigma
                    - 0.5 * D * math.log(2 * math.pi))   # (N, K)

        # log p(x_i) = log Σ_k (1/K) p(x_i | k) = logsumexp - log(K)
        log_p_x = torch.logsumexp(log_p_xk - math.log(self.K), dim=-1)  # (N,)
        return -log_p_x.mean()

    def soft_assign(self, x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """Soft cluster responsibilities.  x: (N,D)  mu: (K,D)  →  (N,K)"""
        sigma    = self.log_sigma.exp().clamp(min=1e-4)
        sq_dist  = (x.unsqueeze(1) - mu.unsqueeze(0)).pow(2).sum(-1)  # (N,K)
        log_resp = -0.5 * sq_dist / sigma ** 2
        return F.softmax(log_resp, dim=-1)

    def hard_assign(self, x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """Cosine-nearest cluster (same as MSE nearest when embeddings are L2-normalized).
        x: (N,D)  mu: (K,D)  →  (N,) int64
        """
        x_n  = F.normalize(x, dim=-1)
        mu_n = F.normalize(mu, dim=-1)
        return (x_n @ mu_n.T).argmax(dim=-1)


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_raw_pt(stem: str):
    path = SAMPLES_DIR / f"{stem}.pt"
    if not path.exists():
        return {}
    data = torch.load(path, map_location="cpu", weights_only=False)
    inp  = data.get("inputs", {})
    bic  = data.get("bag_instance_cluster_ids", {})
    out  = {}
    for mod in TARGET_MODS:
        key = _feat_key(mod)
        t   = inp.get(key)
        if t is None or not isinstance(t, torch.Tensor) or t.numel() == 0:
            continue
        if t.dtype == torch.float16: t = t.float()
        if t.dim() == 1: t = t.unsqueeze(0)
        cids = bic.get(key)
        out[mod] = {"patches": t, "cluster_ids": cids}
    return out


def sample_patches(t: torch.Tensor, cids, max_n: int, device):
    if t.shape[0] > max_n:
        idx  = torch.randperm(t.shape[0])[:max_n]
        t    = t[idx]
        if cids is not None and isinstance(cids, torch.Tensor):
            cids = cids[idx]
    return t.to(device), cids


# ── Metrics ───────────────────────────────────────────────────────────────────

def inter_seed_cosine(mu: torch.Tensor) -> float:
    K   = mu.shape[0]
    if K < 2: return 0.0
    s_n = F.normalize(mu, dim=-1)
    g   = (s_n @ s_n.T).fill_diagonal_(0)
    return g.sum().item() / (K * (K - 1))


def nmi_score(a, b) -> float:
    try:
        from sklearn.metrics import normalized_mutual_info_score
        return float(normalized_mutual_info_score(a, b, average_method="arithmetic"))
    except Exception:
        return float("nan")


# ── UMAP ──────────────────────────────────────────────────────────────────────

def fit_umap(X: np.ndarray) -> np.ndarray:
    from umap import UMAP
    return UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                metric="cosine", random_state=42).fit_transform(X)


def plot_gmm_umap(all_h: np.ndarray, all_cids: np.ndarray,
                  all_seeds: np.ndarray,
                  run_label: str, out_path: Path):
    coords  = fit_umap(np.vstack([all_h, all_seeds]))
    n_h     = len(all_h)
    umap_h  = coords[:n_h]
    umap_s  = coords[n_h:]

    has_cids   = all_cids is not None and len(all_cids) > 0
    n_clusters = (int(all_cids.max()) + 1) if has_cids else 1

    fig, ax = plt.subplots(figsize=(7, 6))
    if has_cids:
        sc = ax.scatter(umap_h[:, 0], umap_h[:, 1],
                        c=all_cids, cmap=plt.cm.get_cmap("tab20", n_clusters),
                        s=3, alpha=0.4, linewidths=0, rasterized=True,
                        vmin=0, vmax=n_clusters - 1)
        plt.colorbar(sc, ax=ax, label="Annotated cluster ID", shrink=0.7)
    else:
        ax.scatter(umap_h[:, 0], umap_h[:, 1], s=3, alpha=0.4, color="#888", rasterized=True)

    ax.scatter(umap_s[:, 0], umap_s[:, 1],
               c=np.arange(len(umap_s)), cmap="Set1",
               s=200, marker="*", edgecolors="k", linewidths=0.8,
               zorder=5, label="Seeds (cluster means)")
    for i, (x, y) in enumerate(umap_s):
        ax.text(x, y, str(i), fontsize=6, ha="center", va="center",
                color="white", fontweight="bold", zorder=6)

    ax.set_title(f"GMM clustering — {run_label}", fontsize=9, fontweight="bold")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_path.with_suffix(f".{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  UMAP → {out_path}.pdf + .png")


# ── Load pre-trained FFN encoder ──────────────────────────────────────────────

def load_ffn_encoder(split: int, fold: int, device):
    ckpt = RESULTS_DIR / f"phase2/split{split}_fold{fold}/set_mil_mt_mega/model_set_mil_mt_final.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model = build_model_v8(variant="set_mil_mt", task="mega")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    encoders = {mod: deepcopy(model.encoders[mod]).eval().to(device) for mod in TARGET_MODS}
    for enc in encoders.values():
        for p in enc.parameters():
            p.requires_grad_(False)
    return encoders


@torch.no_grad()
def embed_patches(ffn_enc, patches: torch.Tensor) -> torch.Tensor:
    return ffn_enc.encode_patches(patches)   # (N, 256), L2-normalized


# ── Training ─────────────────────────────────────────────────────────────────

def train_gmm(model: GMMClusterModel,
              ffn_enc,
              train_stems: List[str],
              mod: str,
              device,
              epochs: int = TRAIN_EPOCHS) -> List[float]:
    opt   = Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    curve = []
    for ep in range(epochs):
        model.train()
        ep_nlls = []
        for stem in np.random.permutation(train_stems):
            raw = load_raw_pt(stem)
            if mod not in raw: continue
            patches, _ = sample_patches(raw[mod]["patches"], None, MAX_PATCHES, device)
            h          = embed_patches(ffn_enc, patches)   # (N, 256), no grad
            opt.zero_grad()
            mu   = model.cluster_means(h)
            loss = model.gmm_nll(h, mu)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_nlls.append(loss.item())
        mean_nll = float(np.mean(ep_nlls)) if ep_nlls else float("nan")
        curve.append(mean_nll)
        print(f"    ep {ep+1:3d}/{epochs}  NLL={mean_nll:.4f}  σ={model.log_sigma.exp().item():.4f}", flush=True)
    return curve


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_gmm(model: GMMClusterModel,
                 ffn_enc,
                 test_stems: List[str],
                 mod: str,
                 device,
                 n_umap_patients: int = 20):
    model.eval()
    nll_list, icos_list, nmi_list = [], [], []
    umap_h, umap_cids, umap_seeds = [], [], []

    with torch.no_grad():
        for si, stem in enumerate(test_stems):
            raw = load_raw_pt(stem)
            if mod not in raw: continue
            patches, cids = sample_patches(raw[mod]["patches"], raw[mod]["cluster_ids"],
                                           MAX_PATCHES, device)
            h  = embed_patches(ffn_enc, patches)
            mu = model.cluster_means(h)

            nll_list.append(model.gmm_nll(h, mu).item())
            icos_list.append(inter_seed_cosine(mu))

            if cids is not None and isinstance(cids, torch.Tensor) and cids.shape[0] == h.shape[0]:
                hard = model.hard_assign(h, mu).cpu().numpy()
                nmi_list.append(nmi_score(hard, cids.long().cpu().numpy()))

            if si < n_umap_patients:
                umap_h.append(h.cpu().numpy())
                umap_seeds.append(mu.cpu().numpy())
                umap_cids.append(cids.long().cpu().numpy() if (cids is not None) else None)

    umap_data = None
    if umap_h:
        # Use first patient's seeds (representative)
        rep_seeds = umap_seeds[0]
        all_h     = np.vstack(umap_h)
        # Merge cids: use zeros for patients without annotation
        merged_cids = []
        for arr, h in zip(umap_cids, umap_h):
            merged_cids.append(arr if arr is not None else np.zeros(h.shape[0], dtype=np.int64))
        all_cids = np.concatenate(merged_cids)
        has_any  = any(c is not None for c in umap_cids)
        umap_data = {"h": all_h, "cids": all_cids if has_any else None,
                     "seeds": rep_seeds}

    return {"nll": nll_list, "inter_cos": icos_list, "nmi": nmi_list, "umap_data": umap_data}


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",   type=int, default=0)
    p.add_argument("--fold",    type=int, default=0)
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--n-test",  type=int, default=100)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  split={args.split}  fold={args.fold}")

    encoders = load_ffn_encoder(args.split, args.fold, device)
    print("FFN encoders loaded (frozen).")

    splits      = build_splits_multitask(str(SAMPLES_DIR), str(SPLITS_CSV),
                                         args.fold, split=args.split)
    train_stems = [r["stem"] for r in splits["train"]][:args.n_train]
    test_stems  = [r["stem"] for r in splits["test"]][:args.n_test]
    print(f"Train: {len(train_stems)}  Test: {len(test_stems)}")

    out_dir = RESULTS_DIR / f"analysis/pma_gmm/split{args.split}_fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs    = [("softmax", 0.0)] + [(f"bcos_b{b}", float(b)) for b in B_VALUES]
    summary = {}
    curves  = {}

    for mod in TARGET_MODS:
        print(f"\n{'='*60}  {mod}  {'='*60}")
        ffn = encoders[mod]

        for run_name, b in runs:
            run_label = f"{mod}_{run_name}"
            print(f"\n  --- {run_label}  (b={b}) ---")

            model = GMMClusterModel(dim=HIDDEN_DIM, K=N_SEEDS,
                                    n_heads=N_HEADS, m=N_INDUCERS, b=b).to(device)

            curve  = train_gmm(model, ffn, train_stems, mod, device, TRAIN_EPOCHS)
            result = evaluate_gmm(model, ffn, test_stems, mod, device)

            curves[run_label] = curve
            summary[run_label] = {
                "nll_mean":       float(np.mean(result["nll"])),
                "nll_std":        float(np.std(result["nll"])),
                "inter_cos_mean": float(np.mean(result["inter_cos"])),
                "inter_cos_std":  float(np.std(result["inter_cos"])),
                "nmi_mean":       float(np.nanmean(result["nmi"])) if result["nmi"] else None,
                "nmi_std":        float(np.nanstd(result["nmi"])) if result["nmi"] else None,
                "n_nmi_patients": len(result["nmi"]),
            }

            s = summary[run_label]
            nmi_str = f"{s['nmi_mean']:.4f}±{s['nmi_std']:.4f}" if s["nmi_mean"] is not None else "n/a"
            print(f"    TEST  NLL={s['nll_mean']:.4f}  inter_cos={s['inter_cos_mean']:.4f}  NMI={nmi_str}")

            torch.save(model.state_dict(), out_dir / f"model_{run_label}.pt")

            ud = result.get("umap_data")
            if ud is not None:
                plot_gmm_umap(ud["h"], ud["cids"], ud["seeds"],
                              run_label, out_dir / f"umap_{run_label}")

    with open(out_dir / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"GMM Clustering  split={args.split}  K={N_SEEDS}  ISAB×2 → PMA")
    print(f"{'='*100}")
    hdr = f"{'Run label':40s}  {'NLL':>10s}  {'Inter-cos':>10s}  {'NMI':>16s}  {'#NMI':>6s}"
    print(hdr); print("-" * len(hdr))
    for run_label, s in summary.items():
        nmi_str = (f"{s['nmi_mean']:.4f}±{s['nmi_std']:.4f}"
                   if s["nmi_mean"] is not None else "n/a")
        print(f"{run_label:40s}  {s['nll_mean']:>10.4f}  "
              f"{s['inter_cos_mean']:>10.4f}  {nmi_str:>16s}  {s['n_nmi_patients']:>6d}")
    print(f"{'='*100}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    all_variants = ["softmax"] + [f"bcos_b{b}" for b in B_VALUES]
    x_ticks      = np.arange(len(all_variants))
    mod_colors   = {"HE": "#E53935", "CT": "#43A047"}

    # Training NLL curves
    fig, axes = plt.subplots(1, len(TARGET_MODS), figsize=(7 * len(TARGET_MODS), 4))
    cmap_c    = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_variants)))
    for mi, (mod, ax) in enumerate(zip(TARGET_MODS, axes)):
        for ci, var in enumerate(all_variants):
            key   = f"{mod}_{var}"
            curve = curves.get(key, [])
            ax.plot(curve, label=var, color=cmap_c[ci])
        ax.set_title(f"{mod} NLL training curve", fontsize=9)
        ax.set_xlabel("Epoch"); ax.set_ylabel("GMM NLL")
        ax.legend(fontsize=7)
    fig.suptitle("GMM NLL training (lower = seeds explain patches better)", fontweight="bold")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"training_curves.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Test metrics
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    fig2.suptitle(f"GMM clustering quality  split={args.split}  K={N_SEEDS}", fontweight="bold")
    metrics = [("nll_mean",       "nll_std",       "GMM NLL ↓"),
               ("inter_cos_mean", "inter_cos_std",  "Inter-seed cosine ↓"),
               ("nmi_mean",       "nmi_std",        "NMI vs annotated ↑")]
    for ax, (mk, sk, label) in zip(axes2, metrics):
        for mod in TARGET_MODS:
            vals = [summary.get(f"{mod}_{v}", {}).get(mk) or 0.0 for v in all_variants]
            errs = [summary.get(f"{mod}_{v}", {}).get(sk) or 0.0 for v in all_variants]
            ax.errorbar(x_ticks, vals, yerr=errs, label=mod,
                        color=mod_colors[mod], marker="o", capsize=3)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(all_variants, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(label, fontsize=9); ax.set_title(label, fontsize=9)
        ax.legend(fontsize=9)
    fig2.tight_layout()
    for ext in ("pdf", "png"):
        fig2.savefig(out_dir / f"metrics_comparison.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"\nAll outputs → {out_dir}/\nDone.")


if __name__ == "__main__":
    main()
