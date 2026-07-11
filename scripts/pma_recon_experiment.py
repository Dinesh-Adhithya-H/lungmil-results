"""
pma_recon_experiment.py — PMA seed clustering quality: b-cos reconstruction experiment.

Architecture (Set Transformer paper, Lee et al. 2019):
  Encoder (frozen): h = ModalFFNEncoder(patches)            (N, H)
                    s = PMA(h)  =  MAB(seeds, rFF(h))       (K, H)

  Decoder (trained): x̂ = MAB_bcos(h, s)  — patches attend to seeds
                         MAB(X, Y) = LN(H + rFF(H)),  H = LN(X + Attn(X, Y, Y))
                         b-cos Attn: weights = ReLU(cos(q,k))^b / sum  (0 = standard softmax)

  Metrics:
    recon_mse     — ||h - x̂||²  (lower = seeds represent patch space well)
    nmi_annotated — NMI(argmax_attn, bag_instance_cluster_ids)  vs. 43 annotated clusters
    inter_seed_cos — mean pairwise cosine sim between K seeds (1 = collapse)

  b sweeps: [0, 1, 2, 4, 8]  (0 = softmax, higher = sharper b-cos)

Run via:  sbatch scripts/submit_pma_recon.sh
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Dict, List, Optional

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

TARGET_MODS = ["HE", "CT"]
MAX_PATCHES = 512
HIDDEN_DIM  = 256
B_VALUES     = [1, 2, 4, 8]   # b-cos sharpness; softmax is a separate variant
TRAIN_EPOCHS = 20
LR           = 1e-3


# ── FFN residual block (Set Transformer rFF) ────────────────────────────────
class FFN(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net  = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                                   nn.Dropout(dropout), nn.Linear(dim * 2, dim))
        self.norm = nn.LayerNorm(dim)
    def forward(self, x): return self.norm(x + self.net(x))


# ── B-cos MAB encoder (PMA replacement for experiment only) ──────────────────
class BcosPMA(nn.Module):
    """
    PMA with b-cos cross-attention (encoder, trained in experiment only).
    MAB(seeds, rFF(x)) where attention is ReLU(cos(q,k))^b or softmax.
    Mirrors the BcosMABDecoder but seeds are queries and patches are KV.
    """
    def __init__(self, hidden_dim: int, n_seeds: int, n_heads: int = 4,
                 dropout: float = 0.1, b: float = 1.0):
        super().__init__()
        self.b        = b
        self.n_heads  = n_heads
        self.head_dim = hidden_dim // n_heads
        assert hidden_dim % n_heads == 0
        self.seeds = nn.Parameter(torch.empty(1, n_seeds, hidden_dim))
        nn.init.trunc_normal_(self.seeds, std=0.02)
        # Paper: PMA_k(X) = MAB(S, rFF(X)) — rFF applied directly to X, no pre-norm
        self.rff   = FFN(hidden_dim, dropout)
        self.proj_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out    = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm1  = nn.LayerNorm(hidden_dim)
        self.ffn    = FFN(hidden_dim, dropout)

    def _attn_weights(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """q: (K,H)  k: (N,H)  →  weights (K,N)"""
        if self.b == 0:
            return F.softmax(q @ k.T / (self.head_dim ** 0.5), dim=-1)
        q_n = F.normalize(q, dim=-1)
        k_n = F.normalize(k, dim=-1)
        raw = F.relu(q_n @ k_n.T).pow(self.b)
        return raw / (raw.sum(-1, keepdim=True) + 1e-9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, H) → seeds: (K, H)"""
        N = x.shape[0]
        K = self.seeds.shape[1]
        nh = self.n_heads
        hd = self.head_dim

        kv = self.rff(x)          # rFF(x) directly — Set Transformer: PMA_k(X) = MAB(S, rFF(X))
        s  = self.seeds.squeeze(0)               # (K, H)

        q = self.proj_q(s).view(K, nh, hd)
        k = self.proj_k(kv).view(N, nh, hd)
        v = self.proj_v(kv).view(N, nh, hd)

        head_outs = []
        for h in range(nh):
            w = self._attn_weights(q[:, h], k[:, h])   # (K, N)
            head_outs.append(w @ v[:, h])               # (K, hd)

        out  = self.out(torch.cat(head_outs, dim=-1))  # (K, H)
        h_v  = self.norm1(s + out)
        return self.ffn(h_v)                            # (K, H)


# ── B-cos MAB decoder ───────────────────────────────────────────────────────
class BcosMABDecoder(nn.Module):
    """
    MAB(X, Y) from Set Transformer paper, using b-cos or softmax attention.
      H   = LayerNorm(X + Attn(X, Y, Y))
      out = LayerNorm(H + rFF(H))

    Here X = patches (queries), Y = seeds (keys/values).
    b = 0 → standard scaled dot-product softmax
    b > 0 → b-cos: weights = ReLU(cosine(q,k))^b, renormalised
    """
    def __init__(self, hidden_dim: int, n_heads: int = 4,
                 dropout: float = 0.1, b: float = 0.0):
        super().__init__()
        self.b = b
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        assert hidden_dim % n_heads == 0
        self.proj_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.proj_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out    = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm1  = nn.LayerNorm(hidden_dim)
        self.ffn    = FFN(hidden_dim, dropout)

    def _attn(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """q: (N,H)  k: (K,H)  →  weights (N,K)"""
        if self.b == 0:
            return F.softmax(q @ k.T / (self.head_dim ** 0.5), dim=-1)
        q_n = F.normalize(q, dim=-1)
        k_n = F.normalize(k, dim=-1)
        cos = q_n @ k_n.T                             # (N, K) in [-1, 1]
        raw = F.relu(cos).pow(self.b)                 # clip negatives, sharpen
        return raw / (raw.sum(-1, keepdim=True) + 1e-9)

    def forward(self, x: torch.Tensor, seeds: torch.Tensor):
        """x: (N,H) patches   seeds: (K,H)  →  x_hat (N,H),  attn_weights (N,K)"""
        N, H = x.shape
        K    = seeds.shape[0]
        nh   = self.n_heads
        hd   = self.head_dim

        # Multi-head projection
        q = self.proj_q(x).view(N, nh, hd)       # (N, nh, hd)
        k = self.proj_k(seeds).view(K, nh, hd)    # (K, nh, hd)
        v = self.proj_v(seeds).view(K, nh, hd)    # (K, nh, hd)

        # Per-head attention, then average weights for interpretability
        head_outs = []
        attn_all  = []
        for h in range(nh):
            w = self._attn(q[:, h], k[:, h])        # (N, K) per head
            attn_all.append(w)
            head_outs.append(w @ v[:, h])            # (N, hd)

        out   = self.out(torch.cat(head_outs, dim=-1))        # (N, H)
        h_vec = self.norm1(x + out)                           # MAB inner residual
        x_hat = self.ffn(h_vec)                               # MAB outer rFF
        attn_mean = torch.stack(attn_all, dim=0).mean(0)      # (N, K) averaged heads
        return x_hat, attn_mean


# ── Data helpers ─────────────────────────────────────────────────────────────
def load_raw_pt(stem: str):
    """Return (patches, cluster_ids) per modality from .pt file."""
    path = SAMPLES_DIR / f"{stem}.pt"
    if not path.exists():
        return {}
    data = torch.load(path, map_location="cpu", weights_only=False)
    inp  = data.get("inputs", {})
    bic  = data.get("bag_instance_cluster_ids", {})
    out  = {}
    for mod in TARGET_MODS:
        key = _feat_key(mod)   # e.g. "HE_cells"
        t   = inp.get(key)
        if t is None or not isinstance(t, torch.Tensor) or t.numel() == 0:
            continue
        if t.dtype == torch.float16: t = t.float()
        if t.dim() == 1: t = t.unsqueeze(0)
        cids = bic.get(key)
        out[mod] = {"patches": t, "cluster_ids": cids}
    return out


def sample_patches(t: torch.Tensor, cids, max_n: int, device):
    """Subsample patches and cluster_ids with the same indices."""
    if t.shape[0] > max_n:
        idx  = torch.randperm(t.shape[0])[:max_n]
        t    = t[idx]
        if cids is not None and isinstance(cids, torch.Tensor):
            cids = cids[idx]
    return t.to(device), cids


# ── Metrics ──────────────────────────────────────────────────────────────────
def inter_seed_cosine(seeds: torch.Tensor) -> float:
    K   = seeds.shape[0]
    if K < 2: return 0.0
    s_n = F.normalize(seeds, dim=-1)
    g   = (s_n @ s_n.T).fill_diagonal_(0)
    return g.sum().item() / (K * (K - 1))


def nmi_score(labels_a, labels_b) -> float:
    """Normalised Mutual Information between two integer label arrays."""
    try:
        from sklearn.metrics import normalized_mutual_info_score
        return float(normalized_mutual_info_score(labels_a, labels_b,
                                                   average_method="arithmetic"))
    except Exception:
        return float("nan")


# ── UMAP helpers ──────────────────────────────────────────────────────────────
def fit_umap(X: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.1) -> np.ndarray:
    from umap import UMAP
    return UMAP(n_components=2, n_neighbors=n_neighbors,
                min_dist=min_dist, metric="cosine",
                random_state=42).fit_transform(X)


def plot_seed_umap(all_h: np.ndarray, all_cids: np.ndarray,
                   all_seeds: np.ndarray,
                   run_label: str, out_path: Path):
    """UMAP of patch embeddings coloured by annotated cluster IDs,
    seeds overlaid as large stars. One panel per variant."""
    coords = fit_umap(np.vstack([all_h, all_seeds]))
    n_h    = len(all_h)
    umap_h = coords[:n_h]
    umap_s = coords[n_h:]

    n_clusters = int(all_cids.max()) + 1
    cmap       = plt.cm.get_cmap("tab20", n_clusters)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(umap_h[:, 0], umap_h[:, 1],
                    c=all_cids, cmap=cmap,
                    s=3, alpha=0.4, linewidths=0, rasterized=True)
    ax.scatter(umap_s[:, 0], umap_s[:, 1],
               c=np.arange(len(umap_s)), cmap="Set1",
               s=180, marker="*", edgecolors="k", linewidths=0.8,
               zorder=5, label="Seeds")
    for i, (x, y) in enumerate(umap_s):
        ax.text(x, y, str(i), fontsize=6, ha="center", va="center",
                color="white", fontweight="bold", zorder=6)
    plt.colorbar(sc, ax=ax, label="Annotated cluster ID", shrink=0.7)
    ax.set_title(run_label, fontsize=9, fontweight="bold")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_path.with_suffix(f".{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  UMAP → {out_path}.pdf + .png")


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(split: int, fold: int, device):
    """Load set_mil_mt from checkpoint. All params trainable (warm start)."""
    ckpt = RESULTS_DIR / f"phase2/split{split}_fold{fold}/set_mil_mt_mega/model_set_mil_mt_final.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    model = build_model_v8(variant="set_mil_mt", task="mega")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    return model.to(device)


# ── Training ──────────────────────────────────────────────────────────────────
def get_components(model, bcos_pma: Optional[nn.Module], mod: str):
    """Return (ffn_encoder, pma) to use for this run. Both fully trainable."""
    ffn = model.encoders[mod]
    pma = bcos_pma if bcos_pma is not None else model.pma[mod]
    return ffn, pma


def encode(ffn, pma, patches: torch.Tensor) -> tuple:
    """FFN encode + PMA compress. Returns (h, seeds)."""
    h     = ffn.encode_patches(patches)
    seeds = pma(h)
    return h, seeds


def train_decoder(decoder: BcosMABDecoder,
                  ffn, pma,
                  train_stems: List[str],
                  mod: str,
                  device,
                  epochs: int = TRAIN_EPOCHS) -> List[float]:
    """Train FFN encoder + PMA + decoder jointly to minimise ||h - x̂||²."""
    params = list(ffn.parameters()) + list(pma.parameters()) + list(decoder.parameters())
    opt    = Adam(params, lr=LR, weight_decay=1e-4)
    loss_curve = []
    for ep in range(epochs):
        ep_losses = []
        idxs = np.random.permutation(len(train_stems))
        for i in idxs:
            stem = train_stems[i]
            raw  = load_raw_pt(stem)
            if mod not in raw: continue
            patches, _ = sample_patches(raw[mod]["patches"], raw[mod]["cluster_ids"], MAX_PATCHES, device)
            ffn.train(); pma.train(); decoder.train()
            opt.zero_grad()
            h, seeds = encode(ffn, pma, patches)
            x_hat, _ = decoder(h, seeds)
            loss = F.mse_loss(x_hat, h)
            loss.backward()
            opt.step()
            ep_losses.append(loss.item())
        mean_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        loss_curve.append(mean_loss)
        print(f"    [train] ep {ep+1:02d}/{epochs}  loss={mean_loss:.5f}", flush=True)
    return loss_curve


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_decoder(decoder: BcosMABDecoder, ffn, pma,
                     test_stems: List[str], mod: str, device,
                     n_umap_patients: int = 20):
    """Returns dict with mse, inter_seed_cos, nmi, and umap data."""
    ffn.eval(); pma.eval(); decoder.eval()
    mse_list, icos_list, nmi_list = [], [], []
    umap_h, umap_cids, umap_seeds = [], [], []

    with torch.no_grad():
        for si, stem in enumerate(test_stems):
            raw = load_raw_pt(stem)
            if mod not in raw: continue
            patches, cids = sample_patches(raw[mod]["patches"], raw[mod]["cluster_ids"], MAX_PATCHES, device)
            h, seeds      = encode(ffn, pma, patches)
            x_hat, attn   = decoder(h, seeds)

            mse_list.append(F.mse_loss(x_hat, h).item())
            icos_list.append(inter_seed_cosine(seeds))

            if cids is not None and isinstance(cids, torch.Tensor):
                h_n         = F.normalize(h, dim=-1)
                s_n         = F.normalize(seeds, dim=-1)
                hard_assign = (h_n @ s_n.T).argmax(dim=-1).cpu()
                nmi_list.append(nmi_score(hard_assign.numpy(), cids.long().numpy()))

            # Collect for UMAP (first n_umap_patients with valid cids)
            if si < n_umap_patients and cids is not None:
                umap_h.append(h.cpu().numpy())
                umap_cids.append(cids.long().cpu().numpy())
                umap_seeds.append(seeds.cpu().numpy())

    umap_data = None
    if umap_h:
        umap_data = {
            "h":    np.vstack(umap_h),
            "cids": np.concatenate(umap_cids),
            "seeds": np.vstack(umap_seeds),   # (n_patients*K, H) — all seeds pooled
        }
    return {"mse": mse_list, "inter_cos": icos_list, "nmi": nmi_list,
            "umap_data": umap_data}


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",      type=int, default=0)
    p.add_argument("--fold",       type=int, default=0)
    p.add_argument("--n-train",    type=int, default=200,
                   help="Max training stems (0 = all)")
    p.add_argument("--n-test",     type=int, default=100,
                   help="Max test stems (0 = all)")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  split={args.split}  fold={args.fold}")

    model = load_model(args.split, args.fold, device)
    print(f"Model loaded. K={model.n_seeds}  mods={list(model.encoders.keys())}")

    splits  = build_splits_multitask(str(SAMPLES_DIR), str(SPLITS_CSV),
                                     args.fold, split=args.split)
    train_stems = [r["stem"] for r in splits["train"]]
    test_stems  = [r["stem"] for r in splits["test"]]
    if args.n_train > 0: train_stems = train_stems[:args.n_train]
    if args.n_test  > 0: test_stems  = test_stems[:args.n_test]
    print(f"Train: {len(train_stems)}  Test: {len(test_stems)}")

    out_dir = RESULTS_DIR / f"analysis/pma_recon/split{args.split}_fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Run experiment ────────────────────────────────────────────────────────
    # Variant A: softmax in both encoder (PMA) and decoder — baseline
    # Variant B: b-cos (b=1,2,4,8) in both encoder (PMA) and decoder
    import copy
    all_results = {}   # {run_label: {mse, inter_cos, nmi}}
    all_curves  = {}   # {run_label: loss_curve}

    for mod in TARGET_MODS:
        print(f"\n{'='*55}  {mod}  {'='*55}")

        # Build run list: (run_label, b_enc, b_dec)
        runs = [("softmax", 0, 0)] + [(f"bcos_b{b}", b, b) for b in B_VALUES]

        for run_label_base, b_enc, b_dec in runs:
            run_label = f"{mod}_{run_label_base}"
            print(f"\n  --- {run_label}  (enc b={b_enc}, dec b={b_dec}) ---")

            ffn     = copy.deepcopy(model.encoders[mod]).to(device)
            pma     = BcosPMA(HIDDEN_DIM, n_seeds=model.n_seeds,
                              n_heads=4, dropout=0.1, b=b_enc).to(device)
            decoder = BcosMABDecoder(HIDDEN_DIM, n_heads=4, dropout=0.1, b=b_dec).to(device)

            curve  = train_decoder(decoder, ffn, pma, train_stems, mod, device)
            result = evaluate_decoder(decoder, ffn, pma, test_stems, mod, device)

            all_results[run_label] = result
            all_curves[run_label]  = curve

            nmi_v = np.nanmean(result["nmi"]) if result["nmi"] else float("nan")
            print(f"    TEST  mse={np.mean(result['mse']):.5f}  "
                  f"inter_cos={np.mean(result['inter_cos']):.4f}  "
                  f"nmi={nmi_v:.4f}")

            torch.save(decoder.state_dict(), out_dir / f"decoder_{run_label}.pt")
            torch.save(pma.state_dict(), out_dir / f"pma_{run_label}.pt")

            # UMAP: patches coloured by annotated cluster, seeds as stars
            ud = result.get("umap_data")
            if ud is not None:
                K       = model.n_seeds
                s_mat   = ud["seeds"].reshape(-1, K, HIDDEN_DIM)   # (n_pat, K, H)
                rep_seeds = s_mat.mean(axis=0)                     # (K, H) mean prototype
                plot_seed_umap(ud["h"], ud["cids"], rep_seeds,
                               run_label,
                               out_dir / f"umap_{run_label}")

    # ── Save raw results ──────────────────────────────────────────────────────
    summary = {}
    for run_label, r in all_results.items():
        summary[run_label] = {
            "mse_mean":       float(np.mean(r["mse"])),
            "mse_std":        float(np.std(r["mse"])),
            "inter_cos_mean": float(np.mean(r["inter_cos"])),
            "inter_cos_std":  float(np.std(r["inter_cos"])),
            "nmi_mean":       float(np.nanmean(r["nmi"])) if r["nmi"] else None,
            "nmi_std":        float(np.nanstd(r["nmi"])) if r["nmi"] else None,
        }
    with open(out_dir / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"PMA Seed Clustering Quality  split={args.split}  K={model.n_seeds}")
    print(f"{'='*100}")
    hdr = f"{'Run label':45s}  {'MSE':>10s}  {'Inter-cos':>10s}  {'NMI (annotated)':>16s}"
    print(hdr); print("-" * len(hdr))
    for run_label, s in summary.items():
        nmi_str = (f"{s['nmi_mean']:.4f}±{s['nmi_std']:.4f}"
                   if s["nmi_mean"] is not None else "n/a")
        print(f"{run_label:45s}  "
              f"{s['mse_mean']:>10.5f}  "
              f"{s['inter_cos_mean']:>10.4f}  "
              f"{nmi_str:>16s}")
    print(f"{'='*100}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    all_variants = ["softmax"] + [f"bcos_b{b}" for b in B_VALUES]
    x_ticks      = np.arange(len(all_variants))
    mod_colors   = {"HE": "#E53935", "CT": "#43A047"}
    mod_ls       = {"HE": "-", "CT": "--"}

    # Figure 1: Training loss curves
    fig1, axes1 = plt.subplots(1, len(TARGET_MODS),
                                figsize=(7 * len(TARGET_MODS), 4), squeeze=False)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_variants)))
    for mi, mod in enumerate(TARGET_MODS):
        ax = axes1[0][mi]
        for ci, var in enumerate(all_variants):
            key   = f"{mod}_{var}"
            curve = all_curves.get(key, [])
            ax.plot(curve, label=var, color=cmap[ci])
        ax.set_title(f"{mod} training loss", fontsize=9)
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE")
        ax.legend(fontsize=7)
    fig1.suptitle("Training loss curves (encoder + decoder unfrozen)", fontweight="bold")
    fig1.tight_layout()
    for ext in ("pdf", "png"):
        fig1.savefig(out_dir / f"training_curves.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: Test metrics across variants, one line per modality
    metric_keys = [("mse_mean",       "Reconstruction MSE ↓"),
                   ("inter_cos_mean", "Inter-seed cosine ↓"),
                   ("nmi_mean",       "NMI vs annotated clusters ↑")]
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))
    fig2.suptitle(f"PMA seed quality  split={args.split}  K={model.n_seeds}",
                   fontweight="bold")
    for ax, (mkey, mlabel) in zip(axes2, metric_keys):
        for mod in TARGET_MODS:
            vals = [summary.get(f"{mod}_{v}", {}).get(mkey) or 0.0 for v in all_variants]
            errs = [summary.get(f"{mod}_{v}", {}).get(mkey.replace("mean","std")) or 0.0
                    for v in all_variants]
            ax.errorbar(x_ticks, vals, yerr=errs, label=mod,
                        color=mod_colors[mod], linestyle=mod_ls[mod],
                        marker="o", capsize=3)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(all_variants, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(mlabel, fontsize=9); ax.set_title(mlabel, fontsize=9)
        ax.legend(fontsize=9)
    fig2.tight_layout()
    for ext in ("pdf", "png"):
        fig2.savefig(out_dir / f"metrics_comparison.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"\nFigures → {out_dir}/\n")


if __name__ == "__main__":
    main()
