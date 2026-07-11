"""
umap_slots.py — UMAP of slot representations to visualise attention collapse

Compares:
  • Init slots   : model.shared_slots (K=128, H=256) — before any patient
  • Post-attn slots per modality (K, H) per patient

If the model collapsed to mean pooling, all K post-attention slots from
every patient land on the same point in UMAP space (all K slots identical).
The init slots should be spread out (random init).

Usage
-----
  python3 interpretability/umap_slots.py \\
      --split 1 --fold 0 --p2-tag alt_shared_combined \\
      --split-set test --n-patients 80

SLURM: see submit_umap_slots.sh
"""

from __future__ import annotations
import argparse, gc, sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from mil.data.loader   import preload_bags
from mil.data.splits   import build_splits_multitask
from mil.data.registry import MODALITIES
from mil.models.builders import build_model_v8

SAMPLES_DIR = "/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
SPLITS_CSV  = "/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
RESULTS_DIR = Path("/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MOD_COLORS = {"HE": "#E53935", "BAL": "#1E88E5", "CT": "#43A047", "Clinical": "#8E24AA"}
MODS       = ["HE", "BAL", "CT", "Clinical"]


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(results_dir, split, fold, p2_tag, slot_k=128):
    fold_tag  = f"split{split}_fold{fold}"
    ckpt_path = results_dir / "phase2" / fold_tag / f"slot_mega_{p2_tag}" / "model_slot_final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Not found: {ckpt_path}")
    model = build_model_v8(variant="slot", slot_k=slot_k, task="mega")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    print(f"  Loaded {ckpt_path.parent.name}  K={slot_k}")
    return model


# ── per-patient slot extraction ───────────────────────────────────────────────

@torch.no_grad()
def extract_slot_reps(model, bags):
    """
    Returns dict: mod → (K, H) slot representation after MHASlotAttn.
    Uses explicit need_weights replay (same as analyze_sankey_clean.py).
    """
    reps = {}
    for mod, enc in model.encoders.items():
        t = bags.get(mod)
        if t is None:
            continue
        t = t.to(DEVICE, non_blocking=True)
        if mod == "HE" and t.shape[0] > model.max_he_patches:
            t = t[:model.max_he_patches]
        h = enc.encode_patches(t)                          # (N, H)
        h_norm = F.normalize(h, dim=-1)
        kv     = h_norm.unsqueeze(0)                       # (1, N, H)
        sa     = model.slot_attns[mod]
        slots  = model.shared_slots.clone()                # (K, H)
        for _ in range(sa.n_iters):
            q = sa.norm_q(slots).unsqueeze(0)
            out, _ = sa.mha(q, kv, kv, need_weights=False)
            slots = slots + out.squeeze(0)
            slots = slots + sa.mlp(slots)
        reps[mod] = slots.cpu().numpy()                    # (K, H)
    return reps


# ── UMAP / t-SNE ─────────────────────────────────────────────────────────────

def fit_reducer(X, method="umap", n_components=2, seed=42):
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=seed,
                                n_neighbors=15, min_dist=0.1)
            return reducer.fit_transform(X)
        except ImportError:
            print("  umap-learn not found, falling back to t-SNE")
            method = "tsne"
    from sklearn.manifold import TSNE
    return TSNE(n_components=n_components, random_state=seed,
                perplexity=min(30, max(5, X.shape[0] // 5)),
                n_iter=1000).fit_transform(X)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_umap_collapse(init_slots, post_slots_by_mod, out_dir, method="umap"):
    """
    Two-row figure:
      Row 1: Init slots vs post-attention slots (one column per modality)
             Points coloured by SLOT INDEX — if collapsed all 128 colours pile up.
      Row 2: Post-attention slots only, coloured by PATIENT INDEX
             — if collapsed each patient's 128 slots are one dot.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    K = init_slots.shape[0]
    slot_cmap = cm.get_cmap("tab20", K)

    present_mods = [m for m in MODS if m in post_slots_by_mod and
                    len(post_slots_by_mod[m]) > 0]
    n_mods = len(present_mods)
    if n_mods == 0:
        print("  No post-attention slots available — nothing to plot"); return

    # ── per-modality figures ──────────────────────────────────────────────────
    for mod in present_mods:
        post_list = post_slots_by_mod[mod]      # list of (K, H) arrays, one per patient
        N_pat = len(post_list)
        post_all = np.concatenate(post_list, axis=0)    # (N_pat*K, H)
        pat_idx  = np.repeat(np.arange(N_pat), K)       # (N_pat*K,)
        slot_idx = np.tile(np.arange(K), N_pat)         # (N_pat*K,)

        # Stack init + post for joint UMAP
        combined = np.concatenate([init_slots, post_all], axis=0)  # (K + N_pat*K, H)
        src_label = np.array(["init"] * K + ["post"] * len(post_all))

        print(f"  Fitting {method} for {mod}  "
              f"(init K={K}, post {N_pat}×K={len(post_all)})  dim={combined.shape[1]}")
        emb = fit_reducer(combined, method=method)

        emb_init = emb[:K]
        emb_post = emb[K:]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"{mod} — slot representations (K={K}, {N_pat} patients)\n"
                     f"{'UMAP' if method=='umap' else 't-SNE'}  |  "
                     f"Collapse diagnosis: if all post-attn slots in one cluster → mean pooling",
                     fontsize=11, fontweight="bold")

        # Panel 1: init slots coloured by slot index
        ax = axes[0]
        ax.scatter(emb_init[:, 0], emb_init[:, 1],
                   c=np.arange(K), cmap="tab20", s=22, alpha=0.9, linewidths=0)
        ax.set_title(f"Init slots (K={K})\n[coloured by slot index]", fontsize=9)
        ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2"); ax.axis("equal")
        ax.set_xticks([]); ax.set_yticks([])

        # Panel 2: post-attn slots coloured by slot index
        # Collapsed → all same colour per slot but all slots at same location
        ax = axes[1]
        ax.scatter(emb_post[:, 0], emb_post[:, 1],
                   c=slot_idx, cmap="tab20", s=8, alpha=0.35, linewidths=0)
        ax.set_title(f"Post-attn slots — {N_pat} patients\n[coloured by slot index]", fontsize=9)
        ax.set_xlabel("dim 1"); ax.axis("equal")
        ax.set_xticks([]); ax.set_yticks([])

        # Panel 3: post-attn slots coloured by patient
        # Collapsed → each patient's 128 slots form a single point
        ax = axes[2]
        pat_cmap = cm.get_cmap("tab20", N_pat)
        ax.scatter(emb_post[:, 0], emb_post[:, 1],
                   c=pat_idx, cmap="tab20", s=8, alpha=0.35, linewidths=0)
        ax.set_title(f"Post-attn slots — {N_pat} patients\n[coloured by patient]", fontsize=9)
        ax.set_xlabel("dim 1"); ax.axis("equal")
        ax.set_xticks([]); ax.set_yticks([])

        fig.tight_layout()
        p = out_dir / f"umap_slots_{mod}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  → {p}")

    # ── combined overview: one panel per modality (init+post overlay) ─────────
    fig, axes = plt.subplots(1, n_mods, figsize=(5 * n_mods, 5))
    if n_mods == 1: axes = [axes]
    fig.suptitle("Init (×) vs post-attention slots (·) — collapse check\n"
                 "Collapsed: post-attn dots pile into one blob far from init crosses",
                 fontsize=11, fontweight="bold")

    for ax, mod in zip(axes, present_mods):
        post_list = post_slots_by_mod[mod]
        N_pat = len(post_list)
        post_all = np.concatenate(post_list, axis=0)
        combined = np.concatenate([init_slots, post_all], axis=0)
        emb = fit_reducer(combined, method=method)
        emb_i = emb[:K]; emb_p = emb[K:]
        pat_idx = np.repeat(np.arange(N_pat), K)
        ax.scatter(emb_p[:, 0], emb_p[:, 1], c=pat_idx, cmap="tab20",
                   s=6, alpha=0.3, linewidths=0, label="post-attn")
        ax.scatter(emb_i[:, 0], emb_i[:, 1], c="black", marker="x",
                   s=30, alpha=0.9, linewidths=1, label="init")
        ax.set_title(f"{mod}", fontsize=10, fontweight="bold",
                     color=MOD_COLORS.get(mod, "#333"))
        ax.set_xticks([]); ax.set_yticks([]); ax.axis("equal")
        ax.legend(fontsize=7, markerscale=1.4, framealpha=0.7)

    fig.tight_layout()
    p = out_dir / "umap_slots_overview.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  → {p}")

    # ── spread statistics (quantify collapse) ────────────────────────────────
    print(f"\n  Spread statistics (std of slot reps — collapsed → near zero):")
    print(f"  {'Source':20s}  {'mean_std':>10s}  {'max_std':>10s}")
    print(f"  {'init':20s}  {init_slots.std(0).mean():>10.5f}  {init_slots.std(0).max():>10.5f}")
    for mod in present_mods:
        post_list = post_slots_by_mod[mod]
        # Per-patient inter-slot std: if collapsed all K slots identical → std=0
        stds = [arr.std(0).mean() for arr in post_list]
        print(f"  {mod+' post-attn':20s}  {np.mean(stds):>10.5f}  {np.max(stds):>10.5f}  "
              f"(N_pat={len(post_list)})")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",       type=int,  default=1)
    p.add_argument("--fold",        type=int,  default=0)
    p.add_argument("--p2-tag",      default="alt_shared_combined")
    p.add_argument("--slot-k",      type=int,  default=128)
    p.add_argument("--split-set",   default="test", choices=["train","val","test","all"])
    p.add_argument("--n-patients",  type=int,  default=80,
                   help="Max patients per modality (0=all)")
    p.add_argument("--method",      default="umap", choices=["umap","tsne"])
    p.add_argument("--out-dir",     default=None)
    p.add_argument("--samples-dir", default=SAMPLES_DIR)
    p.add_argument("--splits-csv",  default=SPLITS_CSV)
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42); np.random.seed(42)

    results_dir = Path(args.results_dir)
    base    = Path(_ROOT) / "interpretability" / f"slot_shared_s{args.split}f{args.fold}_{args.p2_tag}"
    out_dir = Path(args.out_dir) if args.out_dir else base / "umap"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  umap_slots  tag={args.p2_tag}  set={args.split_set}  method={args.method}")
    print(f"  out: {out_dir}")
    print(f"{'='*60}\n")

    # Load model
    model = load_model(results_dir, args.split, args.fold, args.p2_tag, args.slot_k)

    # Init slots — single fixed tensor regardless of patient
    init_slots = model.shared_slots.detach().cpu().numpy()   # (K, H)
    print(f"  Init slots: {init_slots.shape}  std={init_slots.std(0).mean():.5f}")

    # Patient records
    splits_dict = build_splits_multitask(args.samples_dir, args.splits_csv,
                                         args.fold, split=args.split)
    if args.split_set == "all":
        records = splits_dict["train"] + splits_dict["val"] + splits_dict["test"]
    else:
        records = splits_dict[args.split_set]
    if args.n_patients > 0:
        records = records[:args.n_patients]
    print(f"  Patients to process: {len(records)}")

    # Preload bags
    all_stems = [r["stem"] for r in records]
    bag_cache = preload_bags(all_stems, args.samples_dir)

    # Collect post-attention slot reps per modality
    post_slots: Dict[str, List[np.ndarray]] = {m: [] for m in MODS}
    for i, rec in enumerate(records):
        stem  = rec["stem"]
        entry = bag_cache.get(stem, {})
        bags  = {m: entry.get(m) for m in MODALITIES}
        try:
            reps = extract_slot_reps(model, bags)
        except Exception as e:
            print(f"  [warn] {stem}: {e}"); continue
        for mod, arr in reps.items():
            post_slots[mod].append(arr)
        if (i + 1) % 20 == 0:
            print(f"  processed {i+1}/{len(records)}", flush=True)
        gc.collect()

    del model, bag_cache
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    for mod in MODS:
        n = len(post_slots[mod])
        if n > 0:
            # compute inter-slot std per patient
            stds = [arr.std(0).mean() for arr in post_slots[mod]]
            print(f"  {mod:10s}: {n} patients  mean_inter_slot_std={np.mean(stds):.6f}")

    print(f"\n  Plotting {args.method.upper()} ...")
    plot_umap_collapse(init_slots, post_slots, out_dir, method=args.method)
    print(f"\n  Done → {out_dir}\n")


if __name__ == "__main__":
    main()
