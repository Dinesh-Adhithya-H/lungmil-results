"""
Set-MIL-MT interpretability — 7-level representation extraction + visualisation.

7 levels:
  1. instance_reps    : post-ModalFFNEncoder patch embeddings  (N, 256) per modality
  2. seeds_init       : learned PMA seed vectors               (K, 256) per modality [shared weights]
  3. seeds_post_pma   : per-patient PMA output                 (K, 256) per modality
  4. sab_attn         : SAB self-attn weights                  (M*K, M*K) per task per patient
  5. abmil_attn       : per-task ABMIL attention over M*K seeds (M*K,) per task
  6. gate_vals        : TaskModalGate output                    {task: (n_mods,)} per patient
  7. final_reps       : per-task patient-level representations  (256,) per task

Outputs under interpretability/set_mil_mt_interp/split{s}_fold{f}_mega/:
  A_instance_reps.pdf          UMAP of patch embeddings per modality (label row + input-cluster row)
  B_seeds.pdf                  Init seeds (star) vs post-PMA seeds + seed→cluster affinity heatmap
  C_sab_crossmodal_attn.pdf    SAB cross-modal attention heatmap (per-seed labels, avg attn sidebar)
  D_abmil_seed_importance.pdf  ABMIL alpha per seed per task
  E_task_modal_gate.pdf        TaskModalGate weight matrix (task x modality)
  F_modality_combo_ablation.pdf BACC/CI under each modality subset
  G_final_rep_hexbin_{task}.pdf 5-panel: label / score / TTE+avgTTE / modality-combo / risk×TTE
  H_information_pathway.pdf    Cluster→seed→ABMIL→prediction weight chain per task

Usage (sbatch only -- never run Python on the login node):
  sbatch interpretability/submit_interpret_set_mil_mt.sh --split 0 [--fold 1] [--variant mega]
"""

import argparse, math, os, sys, warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.patches as mpatches
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))  # interpretability/
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mil.models.builders import build_model_v8
from mil.data.registry import MODALITIES, _pres_col
from mil.data.loader import preload_bags
from mil.data.splits import build_splits_multitask

# ── Shared constants, colours, and utilities ──────────────────────────────────
from shared import (
    SPLITS_CSV, SAMPLES_DIR, RESULTS_ROOT, HE_CLUSTER_MAP,
    MOD_ORDER, MOD_COLORS, TASK_COLORS, TASK_LABELS,
    HE_BIO_MAP, HE_BIO_COLORS, bio_label,
    savefig as _savefig_shared,
    umap_embed as _umap_embed,
    seed_cluster_mass as _seed_cluster_mass,
    sorted_cluster_order as _sorted_cluster_order,
    sort_seeds_by_diversity as _sort_seeds_by_diversity,
    noncollapsed_seed_mask as _noncollapsed_seed_mask,
)

OUT_ROOT     = ROOT / "interpretability" / "set_mil_mt_interp"
CMAP_HAZARD  = "RdBu_r"
CMAP_TTE     = "RdBu"
_EMPTY_COLOR = "#FFE57F"


# ── Local helpers (set_mil_mt-specific) ──────────────────────────────────────

def _uniform_lim(axes, xy):
    xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    px, py = (xmax - xmin) * 0.05, (ymax - ymin) * 0.05
    for ax in axes:
        ax.set_xlim(xmin - px, xmax + px)
        ax.set_ylim(ymin - py, ymax + py)


def _hex_panel(ax, xy, values, cmap, vmin, vmax, gridsize=35, title="", norm=None):
    kw = dict(norm=norm) if norm is not None else dict(vmin=vmin, vmax=vmax)
    im = ax.hexbin(xy[:, 0], xy[:, 1], C=values,
                   gridsize=gridsize, cmap=cmap,
                   reduce_C_function=np.nanmean, mincnt=1,
                   linewidths=0, edgecolors="none", **kw)
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    return im


K_PATCH = 8   # fallback: number of K-means clusters when pre-computed IDs unavailable

def _cluster_pool(pool_list, seed=42):
    """K-means on concatenated instance reps. Returns (km, all_vecs, labels)."""
    from sklearn.cluster import MiniBatchKMeans
    X = np.concatenate(pool_list).astype(np.float32)
    km = MiniBatchKMeans(n_clusters=K_PATCH, n_init=3, random_state=seed, batch_size=4096)
    labels = km.fit_predict(X)
    return km, X, labels


def _bar_with_collapse_mask(ax, x, vals, seed_colors, nc_mask, width=0.85):
    """Plot bars; collapsed seeds (nc_mask=False) shown as grey/dim."""
    nc_idx  = np.where(nc_mask)[0]
    col_idx = np.where(~nc_mask)[0]
    if len(nc_idx):
        ax.bar(nc_idx, vals[nc_idx],
               color=[seed_colors[i] for i in nc_idx], width=width, alpha=0.85)
    if len(col_idx):
        ax.bar(col_idx, vals[col_idx],
               color="#d0d0d0", width=width, alpha=0.25)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(split, fold, variant, device):
    vtag = "set_mil_mt"
    vdir = f"{vtag}_mega" if variant == "mega" else f"{vtag}_{variant}"
    ckpt = RESULTS_ROOT / f"split{split}_fold{fold}" / vdir / f"model_{vtag}_final.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    # Keys must match TASK_GROUPS in builders.py: clad_surv→"clad", death_surv→"death"
    task_map = {
        "mega":      ["acr_cls", "acr_surv", "clad", "death"],
        "cls":       ["acr_cls"],
        "acr_surv":  ["acr_surv"],
        "clad_surv": ["clad"],
        "death_surv":["death"],
    }
    tasks = task_map.get(variant, ["acr_cls", "acr_surv", "clad", "death"])
    task_key = variant if variant != "cls" else "cls"
    model = build_model_v8(variant=vtag, task=task_key, slot_k=16, n_cross_layers=1)
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device), tasks


# ── SAB patching helper ───────────────────────────────────────────────────────

def _make_sab_wrapper(layer, attn_list):
    """Return a replacement for SAB.forward that captures attn weights."""
    def patched(x):
        xb    = x.unsqueeze(0)
        a, w  = layer.attn(xb, xb, xb, need_weights=True, average_attn_weights=True)
        attn_list.append(w.squeeze(0).detach().cpu())
        return layer.ffn(layer.norm(xb + a)).squeeze(0)
    return patched


# ── Single-patient extraction ─────────────────────────────────────────────────

@torch.no_grad()
def extract_patient(model, bags, device, tasks):
    m = model

    # Patch SAB layers to capture attention weights
    sab_attn_list = []
    orig_sab_fwds = [(layer, layer.forward) for layer in m.sab]
    for layer, _ in orig_sab_fwds:
        layer.forward = _make_sab_wrapper(layer, sab_attn_list)

    # Hook abmil_w per task (captures raw attention logits before softmax)
    abmil_raw  = {}
    abmil_hooks = []
    for task in tasks:
        def _hook(mod_, inp, out, t=task):
            abmil_raw[t] = out.detach().cpu()
        abmil_hooks.append(m.abmil_w[task].register_forward_hook(_hook))

    # Hook task_gate
    gate_vals  = {}
    gate_hooks = []
    if m.task_gate is not None:
        def _gate_hook(mod_, inp, out):
            for t, v in out.items():
                gate_vals[t] = v.detach().cpu().numpy()
        gate_hooks.append(m.task_gate.register_forward_hook(_gate_hook))

    # Forward pass
    out_dict = m(bags, device)

    # Restore SAB and remove hooks
    for layer, orig in orig_sab_fwds:
        layer.forward = orig
    for h in abmil_hooks + gate_hooks:
        h.remove()

    if not isinstance(out_dict, dict):
        return None

    # Map SAB attn list -> per-task (use_task_gate: SAB called once per task in order)
    sab_attn = {}
    if m.use_task_gate:
        for i, task in enumerate(m.task_names):
            if i < len(sab_attn_list):
                sab_attn[task] = sab_attn_list[i].numpy()
    else:
        if sab_attn_list:
            mat = sab_attn_list[0].numpy()
            for task in tasks:
                sab_attn[task] = mat

    # ABMIL: store both raw logits and normalized alpha
    abmil_attn      = {t: torch.softmax(raw, dim=0).squeeze(-1).numpy()
                       for t, raw in abmil_raw.items()}
    abmil_raw_logits = {t: raw.squeeze(-1).numpy()
                        for t, raw in abmil_raw.items()}

    # Final reps and logits from model output tuples (logit, rep)
    final_reps = {}
    logits_out = {}
    for task, val in out_dict.items():
        if isinstance(val, tuple):
            logit, rep = val
            logits_out[task] = float(logit.cpu().item()) if logit.ndim == 0 \
                               else float(logit.cpu().numpy().flat[0])
            final_reps[task] = rep.cpu().numpy()

    # Instance reps + PMA (second lightweight forward through encoders only)
    inst_reps  = {}
    inst_keys  = {}   # proj_k(instance) — actual key vectors used in attention
    seeds_post = {}
    pma_attn   = {}   # normalized attention weights (K, N)
    pma_bcos   = {}   # raw B-cos logits relu(q·k)^b (K, N) — use for cluster affinity
    pma_idx    = {}   # which patch indices were actually used (after max_patches truncation)
    present_mods = []
    for mod, enc in m.encoders.items():
        t = bags.get(mod)
        if t is None:
            continue
        t = t.to(device)
        used_idx = None
        if t.shape[0] > m.max_he_patches:
            used_idx = torch.randperm(t.shape[0], device=device)[:m.max_he_patches]
            t = t[used_idx]
        crds = bags.get("HE_coords") if mod == "HE" else None
        h    = enc.encode_patches(t, coords=crds)
        inst_reps[mod] = h.cpu().numpy()
        pma_idx[mod]   = used_idx.cpu().numpy() if used_idx is not None else None

        s, aw, logits_tuple = m.pma[mod](h, return_attn=True, return_logits=True)
        mod_idx = torch.tensor(m._mod_idx[mod], device=device)
        seeds_post[mod] = (s + m.modal_embed(mod_idx)).cpu().numpy()
        pma_attn[mod]   = aw.cpu().numpy()
        pma_bcos[mod]   = logits_tuple[2].cpu().numpy()  # relu(q·k)^b

        # Per-patient key vectors: proj_k(instance) — actual space queried against
        inst_keys[mod] = m.pma[mod].proj_k(h).detach().cpu().numpy()  # (N, H)
        present_mods.append(mod)

    return {
        "inst_reps":         inst_reps,
        "inst_keys":         inst_keys,
        "seeds_post":        seeds_post,
        "pma_attn":          pma_attn,
        "pma_bcos":          pma_bcos,
        "pma_idx":           pma_idx,
        "sab_attn":          sab_attn,
        "abmil_attn":        abmil_attn,
        "abmil_raw_logits":  abmil_raw_logits,
        "gate_vals":         gate_vals,
        "final_reps":        final_reps,
        "logits":            logits_out,
        "present_mods":      present_mods,
    }


# ── Extraction loop ───────────────────────────────────────────────────────────

def extract_all(split, fold, variant, device, max_samples=None):
    print(f"[extract] split={split} fold={fold} variant={variant}")
    model, tasks = load_model(split, fold, variant, device)
    splits       = build_splits_multitask(SAMPLES_DIR, SPLITS_CSV, fold=fold, split=split)
    recs         = splits["test"]
    if max_samples:
        recs = recs[:max_samples]

    bag_cache = preload_bags([r["stem"] for r in recs], SAMPLES_DIR, n_workers=8)

    _MOD_PT_KEY = {"HE": "HE_cells", "BAL": "BAL_cells", "CT": "CT_cells"}
    samples_path = Path(SAMPLES_DIR)

    results = []
    for i, rec in enumerate(recs):
        bags = bag_cache.get(rec["stem"])
        if bags is None:
            continue
        try:
            out = extract_patient(model, bags, device, tasks)
        except Exception as e:
            print(f"  [skip] {rec['stem']}: {e}")
            continue
        if out is None:
            continue

        # Load pre-computed cluster IDs + names + clinical feature names from raw .pt
        cluster_ids   = {}
        cluster_names = {}
        cfn           = []
        try:
            raw_pt = torch.load(samples_path / f"{rec['stem']}.pt",
                                map_location="cpu", weights_only=False)
            for mod, key in _MOD_PT_KEY.items():
                ids = raw_pt.get("bag_instance_cluster_ids", {}).get(key)
                if isinstance(ids, torch.Tensor) and ids.numel() > 0:
                    cluster_ids[mod] = ids.numpy()
                nms = raw_pt.get("bag_cluster_names", {}).get(key)
                if isinstance(nms, list) and nms:
                    cluster_names[mod] = nms
            cfn = raw_pt.get("clinical_feature_names") or []
        except Exception:
            pass

        out["stem"]                  = rec["stem"]
        out["patient_id"]            = rec.get("patient_id", rec["stem"])
        out["anchor_dt"]             = rec.get("anchor_dt", None)
        out["label"]                 = rec.get("label")
        out["event_acr"]             = rec.get("event_next_acr", float("nan"))
        out["tte_acr"]               = rec.get("tte_next_acr", float("nan"))
        out["event_clad"]            = rec.get("clad_event", float("nan"))
        out["tte_clad"]              = rec.get("clad_time", float("nan"))
        out["event_death"]           = rec.get("death_event", float("nan"))
        out["tte_death"]             = rec.get("death_time", float("nan"))
        # Clinical: no pre-computed clusters — each row is a named feature directly
        if cfn:
            n_clin = len(cfn)
            cluster_ids["Clinical"]   = np.arange(n_clin)
            cluster_names["Clinical"] = list(cfn)

        out["cluster_ids"]           = cluster_ids
        out["cluster_names"]         = cluster_names
        out["clinical_feature_names"] = cfn
        results.append(out)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(recs)} done")

    # Raw seed parameters (for reference)
    seeds_init = {mod: model.pma[mod].seeds.detach().cpu().numpy()
                  for mod in model.encoders}
    # Actual query vectors: proj_q(normalize(seed)) — what seeds look like to instances
    with torch.no_grad():
        seeds_init_q = {}
        for mod in model.encoders:
            pma = model.pma[mod]
            s_n = F.normalize(pma.seeds, dim=-1)
            seeds_init_q[mod] = pma.proj_q(s_n).cpu().numpy()   # (K, H)

    return results, seeds_init, seeds_init_q, tasks


def extract_all_splits(variant, device, max_samples=None):
    """
    Pool test patients from all 5 splits (fold 0 model per split).
    Each sample appears as a test patient exactly once.
    Returns pooled results (~4200 patients), plus seeds from split 0.
    Panels B and F are not valid on pooled data (different models per split).
    """
    all_results = []
    seeds_init = seeds_init_q = tasks = None
    for split in range(5):
        try:
            res, si, siq, t = extract_all(split, fold=0, variant=variant,
                                          device=device, max_samples=max_samples)
        except FileNotFoundError as e:
            print(f"  [skip split{split}] {e}")
            continue
        # Tag each result with its split so we can trace back if needed
        for r in res:
            r["_split"] = split
        all_results.extend(res)
        if seeds_init is None:   # use split 0's seeds for reference
            seeds_init, seeds_init_q, tasks = si, siq, t
        print(f"  [all_splits] split{split}: {len(res)} patients  total={len(all_results)}")
    return all_results, seeds_init, seeds_init_q, tasks


# ── Panel A: instance rep UMAPs ───────────────────────────────────────────────

def panel_A(results, out_dir, split, fold, metric="euclidean"):
    rng = np.random.default_rng(42)
    inst_pool  = {mod: [] for mod in MOD_ORDER}
    inst_labs  = {mod: [] for mod in MOD_ORDER}
    inst_clus  = {mod: [] for mod in MOD_ORDER}  # pre-computed cluster IDs
    cnames_ref = {}  # first seen cluster name list per modality

    for res in results:
        lab  = res["label"]
        pidx = res.get("pma_idx", {})  # patch indices used (if truncated)
        for mod, h in res["inst_reps"].items():
            n   = min(len(h), 150)
            idx = rng.choice(len(h), n, replace=False)
            inst_pool[mod].append(h[idx])
            inst_labs[mod].extend([lab] * n)
            # Map back to original patch index if truncated
            c_ids = res.get("cluster_ids", {}).get(mod)
            orig_idx = pidx.get(mod)
            if c_ids is not None:
                # if patches were truncated, c_ids matches truncated set
                if orig_idx is not None and len(c_ids) == len(orig_idx):
                    inst_clus[mod].extend(c_ids[idx].tolist())
                elif len(c_ids) == len(h):
                    inst_clus[mod].extend(c_ids[idx].tolist())
                else:
                    inst_clus[mod].extend([-1] * n)
            else:
                inst_clus[mod].extend([-1] * n)
            nms = res.get("cluster_names", {}).get(mod)
            if nms and mod not in cnames_ref:
                cnames_ref[mod] = nms

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    metric_note = f"UMAP metric: {metric}" + (" ← correct for L2-norm outputs" if metric == "cosine" else " ← Euclidean (default)")
    fig.suptitle(f"A — Instance reps (post-ModalFFNEncoder) | split{split}_fold{fold}\n{metric_note}",
                 fontsize=11, fontweight="bold")

    for col, mod in enumerate(MOD_ORDER):
        pool_list = inst_pool[mod]
        ax_lab = axes[0, col]
        ax_clu = axes[1, col]

        if not pool_list or sum(len(p) for p in pool_list) < 20:
            for ax in (ax_lab, ax_clu):
                ax.text(0.5, 0.5, f"{mod}\n(no data)", ha="center", va="center",
                        transform=ax.transAxes)
            continue

        pool = np.concatenate(pool_list)
        xy   = _umap_embed(pool, metric=metric)
        labs = np.array([(1 if l == 1 else (0 if l == 0 else -1)) for l in inst_labs[mod]])

        # row 1: ACR label coloring
        ax = ax_lab
        ax.scatter(xy[labs == 0,  0], xy[labs == 0,  1], s=1.5, c="#1E88E5", alpha=0.4, label="ACR-")
        ax.scatter(xy[labs == 1,  0], xy[labs == 1,  1], s=1.5, c="#E53935", alpha=0.5, label="ACR+")
        if (labs == -1).any():
            ax.scatter(xy[labs == -1, 0], xy[labs == -1, 1], s=1.5, c="#bbb", alpha=0.3)
        ax.set_title(f"{mod} — by label", fontsize=9, color=MOD_COLORS[mod], fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.legend(markerscale=4, fontsize=7, framealpha=0.6)

        # row 2: pre-computed cluster coloring (with interpretable names)
        ax = ax_clu
        cl_ids = np.array(inst_clus[mod])
        nms    = cnames_ref.get(mod, [])
        valid  = cl_ids >= 0
        if valid.sum() > 20:
            unique_ids = np.unique(cl_ids[valid])
            n_uniq = len(unique_ids)
            id_to_pos = {cid: i for i, cid in enumerate(unique_ids)}
            pos_ids = np.array([id_to_pos.get(c, 0) for c in cl_ids])
            try:
                cmap_cl = matplotlib.colormaps.get_cmap("tab20").resampled(max(n_uniq, 2))
            except AttributeError:
                cmap_cl = plt.cm.get_cmap("tab20", max(n_uniq, 2))
            ax.scatter(xy[valid, 0], xy[valid, 1], c=pos_ids[valid], s=1.5, alpha=0.5,
                       cmap=cmap_cl, vmin=0, vmax=n_uniq - 1)
            # legend: show up to 12 most common clusters with name
            from collections import Counter
            top_ids = [cid for cid, _ in Counter(cl_ids[valid].tolist()).most_common(12)]
            handles = []
            for cid in top_ids:
                lbl = nms[cid] if nms and cid < len(nms) else str(cid)
                pos = id_to_pos.get(cid, 0)
                handles.append(plt.Line2D([0],[0], marker='o', color='w',
                                          markerfacecolor=cmap_cl(pos / max(n_uniq-1, 1)),
                                          markersize=5, label=lbl))
            ax.legend(handles=handles, fontsize=5, ncol=2, framealpha=0.6, loc="lower left")
        else:
            ax.text(0.5, 0.5, f"{mod}\n(no cluster IDs)", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title(f"{mod} — by cluster", fontsize=9, color=MOD_COLORS[mod], fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    axes[0, 0].set_ylabel("By label", fontsize=9)
    axes[1, 0].set_ylabel("By cluster (pre-computed)", fontsize=9)
    fig.tight_layout()
    suffix = f"_{metric}" if metric != "euclidean" else ""
    fig.savefig(out_dir / f"A_instance_reps{suffix}.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / f"A_instance_reps{suffix}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  A ({metric}) done")


# ── Panel B: seeds init vs post-PMA + seed-to-cluster affinity ───────────────

def panel_B(results, seeds_init, seeds_init_q, out_dir, split, fold):
    if seeds_init is None or seeds_init_q is None:
        print("  B skipped (seeds_init not available — use single-split mode, not --json-only)")
        return
    seeds_post_pool  = {mod: [] for mod in MOD_ORDER}
    inst_keys_pool   = {mod: [] for mod in MOD_ORDER}  # proj_k(instance) per patient
    seed_cluster_aff = {mod: [] for mod in MOD_ORDER}  # list of (K_seeds, K_clusters)
    cnames_ref       = {}  # first seen cluster names per modality
    max_cluster_id   = {mod: 0 for mod in MOD_ORDER}

    for res in results:
        for mod, s in res["seeds_post"].items():
            seeds_post_pool[mod].append(s)
        for mod, ik in res.get("inst_keys", {}).items():
            n = min(len(ik), 150)
            idx = np.random.default_rng(42).choice(len(ik), n, replace=False)
            inst_keys_pool[mod].append(ik[idx])
        for mod, h in res["inst_reps"].items():
            # Use B-cos scores (relu(q·k)^b) — more interpretable than normalized weights
            bcos = res.get("pma_bcos", {}).get(mod)
            if bcos is None:
                bcos = res.get("pma_attn", {}).get(mod)  # fallback
            if bcos is None or bcos.ndim != 2 or bcos.shape[1] != len(h):
                continue
            # Use pre-computed cluster IDs
            c_ids = res.get("cluster_ids", {}).get(mod)
            if c_ids is not None and len(c_ids) == len(h):
                cl = c_ids
                nms = res.get("cluster_names", {}).get(mod, [])
                if nms and mod not in cnames_ref:
                    cnames_ref[mod] = nms
                k_clus = int(cl.max()) + 1 if len(cl) > 0 else K_PATCH
                max_cluster_id[mod] = max(max_cluster_id[mod], k_clus)
                aff = _seed_cluster_mass(bcos, cl, k_clus)
            else:
                # Fallback: K-means
                from sklearn.cluster import MiniBatchKMeans
                km  = MiniBatchKMeans(n_clusters=K_PATCH, n_init=3, random_state=42, batch_size=4096)
                cl  = km.fit_predict(h.astype(np.float32))
                aff = _seed_cluster_mass(bcos, cl, K_PATCH)
            seed_cluster_aff[mod].append(aff)

    fig = plt.figure(figsize=(18, 9))
    gs_outer = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.25)
    fig.suptitle(f"B — Seeds: init★ vs post-PMA (top) | Seed→cluster affinity (bottom) | split{split}_fold{fold}",
                 fontsize=10, fontweight="bold")

    for col, mod in enumerate(MOD_ORDER):
        post_pool = seeds_post_pool.get(mod, [])
        init      = seeds_init.get(mod)

        # ── top: UMAP of query vectors ★ vs instance key vectors ─────────────
        # Seeds shown as proj_q(normalize(seed)) — actual query space
        # Instances shown as proj_k(instance) — actual key space
        # Both in same H=256 projected space, so UMAP is geometrically meaningful
        ax = fig.add_subplot(gs_outer[0, col])
        init_q    = seeds_init_q.get(mod)   # (K, H) — actual query vectors
        key_pool  = inst_keys_pool.get(mod, [])
        if not key_pool or init_q is None:
            ax.text(0.5, 0.5, f"{mod}\n(no data)", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            K      = init_q.shape[0]
            keys   = np.concatenate(key_pool)               # (N_total, H)
            all_vecs = np.vstack([keys, init_q])
            xy     = _umap_embed(all_vecs, n_neighbors=15, min_dist=0.1, metric="cosine")
            xy_keys, xy_q = xy[:-K], xy[-K:]
            # Color instance cloud by B-cos-weighted density (grey, no task label)
            ax.scatter(xy_keys[:, 0], xy_keys[:, 1], s=1.5, c="#aaa", alpha=0.2, zorder=1)
            # Detect collapsed seeds from query vector cosine similarity
            q_norms = init_q / (np.linalg.norm(init_q, axis=1, keepdims=True) + 1e-8)
            q_sim   = q_norms @ q_norms.T
            np.fill_diagonal(q_sim, 0)
            q_redundant = q_sim.max(axis=1) > 0.90
            # Non-collapsed seeds: colored stars; collapsed: small grey dots (background)
            for k, (xi, yi) in enumerate(xy_q):
                if q_redundant[k]:
                    ax.scatter(xi, yi, s=25, c="#ccc", alpha=0.5, zorder=3, marker="o")
                else:
                    ax.scatter(xi, yi, s=90, c=[k], cmap="tab20",
                               vmin=0, vmax=K - 1, edgecolors="black",
                               linewidths=0.9, zorder=5, marker="*")
                    ax.text(xi, yi + 0.2, str(k), fontsize=5, ha="center", va="bottom",
                            color="black", zorder=6)
            n_uniq = int((~q_redundant).sum())
            if col == 0:
                ax.legend(handles=[
                    plt.Line2D([0],[0], marker='*', color='w', markerfacecolor='gray',
                               markersize=8, label=f'Unique seed (★, n={n_uniq})'),
                    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#ccc',
                               markersize=5, label=f'Collapsed seed (●, n={K-n_uniq})'),
                    plt.Line2D([0],[0], marker='o', color='w', markerfacecolor='#aaa',
                               markersize=4, label='Instance key (proj_k(x))'),
                ], fontsize=6, framealpha=0.7)
        ax.set_title(f"{mod} — query vs key space  ({n_uniq if key_pool else 0} unique seeds)",
                     fontsize=9, color=MOD_COLORS[mod], fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

        # ── bottom: seed→cluster B-cos affinity heatmap ─────────────────────
        ax2 = fig.add_subplot(gs_outer[1, col])
        aff_list = seed_cluster_aff.get(mod, [])
        if not aff_list:
            ax2.text(0.5, 0.5, f"{mod}\n(no pma_bcos)", ha="center", va="center",
                     transform=ax2.transAxes)
        else:
            max_k = max(a.shape[1] for a in aff_list)
            padded  = [np.pad(a, ((0,0),(0, max_k - a.shape[1]))) for a in aff_list]
            mean_aff_raw = np.stack(padded).mean(0)   # (K_seeds, max_k)
            nms = cnames_ref.get(mod, [str(c) for c in range(max_k)])

            # ── sort columns: Clinical → by attention mass (top-40 shown)
            #                  HE → bio-category; others → original order ──
            CLINICAL_TOP_N = 40
            top_n_arg = CLINICAL_TOP_N if mod == "Clinical" else None
            n_cols = mean_aff_raw.shape[1]
            nms_clipped = nms[:n_cols]   # cluster names must not exceed actual affinity columns
            col_order = _sorted_cluster_order(nms_clipped, mod, mean_aff_raw=mean_aff_raw, top_n=top_n_arg)
            nms_sorted = [nms_clipped[i] for i in col_order]
            mean_aff_col = mean_aff_raw[:, col_order]

            # ── sort rows (seeds) by hierarchical clustering for diversity view ──
            seed_order = _sort_seeds_by_diversity(mean_aff_col)
            mean_aff   = mean_aff_col[seed_order, :]

            K_s = mean_aff.shape[0]
            # seed labels after reordering
            orig_seed_labels = [f"{mod[0] if mod != 'Clinical' else 'Cl'}{k}"
                                for k in range(K_s)]
            reordered_labels = [orig_seed_labels[i] for i in seed_order]

            im = ax2.imshow(mean_aff, cmap="YlOrRd", aspect="auto",
                            vmin=0, vmax=np.percentile(mean_aff, 95))

            # ── x-axis: show ALL sub-cluster names, rotated 90° ──
            ax2.set_xticks(range(len(nms_sorted)))
            ax2.set_xticklabels(nms_sorted, fontsize=4.5, rotation=90)

            # ── for HE: color each x-tick label by biological category ──
            if mod == "HE" and HE_BIO_MAP:
                for tick, nm in zip(ax2.get_xticklabels(), nms_sorted):
                    bio = HE_BIO_MAP.get(nm, "Unknown")
                    tick.set_color(HE_BIO_COLORS.get(bio, "#333333"))

                # Draw colored spans above heatmap for macro-category groups
                bio_spans = {}  # bio_name → [start_col, end_col]
                cur_bio, start = None, 0
                for xi, nm in enumerate(nms_sorted):
                    bio = HE_BIO_MAP.get(nm, "Unknown")
                    if bio != cur_bio:
                        if cur_bio is not None:
                            bio_spans.setdefault(cur_bio, []).append((start, xi - 1))
                        cur_bio, start = bio, xi
                if cur_bio:
                    bio_spans.setdefault(cur_bio, []).append((start, len(nms_sorted) - 1))

                # Draw colored bars at the top of the plot
                ylim = ax2.get_ylim()
                bar_h = (ylim[0] - ylim[1]) * 0.06   # 6% of y-axis height
                bar_y = ylim[1] - bar_h * 1.1
                for bio, spans in bio_spans.items():
                    color = HE_BIO_COLORS.get(bio, "#bbb")
                    for xs, xe in spans:
                        ax2.add_patch(plt.Rectangle(
                            (xs - 0.5, bar_y), xe - xs + 1, bar_h,
                            color=color, transform=ax2.transData,
                            clip_on=False, zorder=5))
                # Bio-category legend (compact)
                handles = [plt.Line2D([0],[0], color=c, lw=5,
                                      label=k.replace(" with ", "\nw/ ").replace(" and ", " & "))
                           for k, c in HE_BIO_COLORS.items()
                           if k in bio_spans]
                ax2.legend(handles=handles, fontsize=4, loc="lower right",
                           framealpha=0.7, ncol=1)

            # ── y-axis: drop collapsed/redundant seeds (cosine sim > 0.90) ──
            norms     = mean_aff / (np.linalg.norm(mean_aff, axis=1, keepdims=True) + 1e-8)
            sim_mat   = norms @ norms.T
            np.fill_diagonal(sim_mat, 0)
            redundant = sim_mat.max(axis=1) > 0.90
            # Keep only unique seeds; drop duplicates entirely
            keep_rows = np.where(~redundant)[0]
            n_dropped = redundant.sum()
            if len(keep_rows) == 0:
                keep_rows = np.arange(K_s)   # safety: keep all if all collapsed
            mean_aff = mean_aff[keep_rows, :]
            reordered_labels = [reordered_labels[i] for i in keep_rows]
            K_s = mean_aff.shape[0]
            # Re-draw imshow with filtered rows
            ax2.cla()
            im = ax2.imshow(mean_aff, cmap="YlOrRd", aspect="auto",
                            vmin=0, vmax=np.percentile(mean_aff, 95))
            ax2.set_xticks(range(mean_aff.shape[1]))
            ax2.set_xticklabels(nms_sorted, fontsize=4.5, rotation=90)
            ax2.set_yticks(range(K_s))
            ax2.set_yticklabels(reordered_labels, fontsize=5)
            for tick in ax2.get_yticklabels():
                tick.set_color("#333")
            ax2.set_ylabel(
                f"Non-collapsed seeds ({K_s}/{K_s + n_dropped} unique, "
                f"{n_dropped} redundant dropped)", fontsize=6)

            if mod == "Clinical":
                shown = len(col_order)
                total = mean_aff_raw.shape[1]
                ax2.set_xlabel(
                    f"Top-{shown} of {total} clinical features — sorted by total seed attention ↓",
                    fontsize=6, color="#555")
                # Annotate total attention captured by shown vs hidden features
                shown_mass = mean_aff_col.sum()
                total_mass = mean_aff_raw.sum()
                frac = shown_mass / total_mass if total_mass > 0 else 1.0
                ax2.text(0.99, 0.01,
                         f"{frac*100:.0f}% of total\nB-cos mass shown",
                         ha="right", va="bottom", fontsize=6,
                         transform=ax2.transAxes,
                         bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
            else:
                ax2.set_xlabel("Input cluster — sorted by bio-category", fontsize=7)
            fig.colorbar(im, ax=ax2, shrink=0.7, label="B-cos attn mass")
        ax2.set_title(f"{mod} — seed→cluster (B-cos)", fontsize=8, color=MOD_COLORS[mod])

    fig.savefig(out_dir / "B_seeds.pdf", dpi=150, bbox_inches="tight"); fig.savefig(str(out_dir / "B_seeds.pdf").replace(".pdf", ".png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  B done")


# ── Panel C: SAB cross-modal attention — full seed labels, modality coloring,
#            ABMIL alpha sidebar, ACR+/high-risk vs ACR-/low-risk differential ──

def panel_C(results, tasks, out_dir, split, fold):
    """
    SAB cross-modal attention heatmaps.

    The SetTransformerMIL architecture stacks a Set Attention Block (SAB) that
    allows every seed from every modality to attend to every other seed.  This
    produces an (S × S) attention matrix per patient (S = K × n_mods, K=16 seeds
    per modality).  Panel C visualises this matrix averaged across all patients,
    and separately for high-risk vs low-risk patients.

    Outputs
    -------
    C_sab_crossmodal_attn.png
        Full S×S heatmap per task.
        - Main grid (viridis): mean SAB attention weight.  Entry (i, j) = seed i
          attending to seed j.  Diagonal blocks = self-attention within a modality;
          off-diagonal blocks = cross-modal attention.
        - Right sidebar: mean ABMIL α per seed, i.e. how much that seed ultimately
          contributes to the final prediction.
        - Bottom strip (RdBu_r): differential attention = high-risk mean − low-risk
          mean.  Red = seed pair more active in high-risk patients.

    C_sab_significant.png
        Sparse version for the paper: only the top 10% of |Δ attn| entries are
        kept; the rest are zeroed.  Rows and columns are further filtered to only
        seeds that participate in at least one significant pair, collapsing the
        full 64×64 matrix to a compact submatrix of biologically active seeds.
        Use this to identify which modality-seed pairs differ most between
        disease groups.

    Canonical alignment
        Each patient can have a different subset of modalities.  Patient matrices
        are placed into a canonical 64×64 grid (modalities at fixed offsets) so
        averaging is always apples-to-apples.  Per-cell counts track how many
        patients contributed to each cell (absent-modality cells are excluded from
        the mean, not treated as zero).

    Stratification
        acr_cls: high-risk = label 1 (ACR+);  low-risk = label 0.
        survival tasks: high-risk = logit ≥ cohort median;  low-risk = logit < median.
    """
    present_mods_main = [m for m in MOD_ORDER
                          if sum(m in r["present_mods"] for r in results) >= 5]
    K = 16
    total_seeds = len(present_mods_main) * K
    mod_abbr = {"HE": "H", "BAL": "B", "CT": "C", "Clinical": "Cl"}

    # Per-seed metadata — use "HE·s00" format for clear modality+index labelling
    seed_labels    = []
    seed_mod_col   = []   # color per seed position (by modality)
    mod_boundaries = []
    mod_spans      = []   # (start, end) per modality block
    for mod in present_mods_main:
        start = len(seed_labels)
        mod_boundaries.append(start)
        seed_labels.extend([f"{mod}·s{k:02d}" for k in range(K)])
        seed_mod_col.extend([MOD_COLORS[mod]] * K)
        mod_spans.append((start, start + K - 1, mod))

    ntasks = len(tasks)
    # Per task: one column with 3 rows (main heatmap, diff heatmap, alpha bar)
    # Scale figure so cells are large enough to read at ≥8pt tick labels
    cell_px = max(0.4, min(0.55, 16.0 / total_seeds))  # px per cell, 0.4–0.55
    panel_w = max(6.0, total_seeds * cell_px + 3.5)
    panel_h = max(7.0, total_seeds * cell_px + 4.0)
    fig = plt.figure(figsize=(panel_w * ntasks, panel_h))
    outer_gs = gridspec.GridSpec(1, ntasks, figure=fig, wspace=0.55)
    fig.suptitle(
        f"C — SAB cross-modal attention  |  split{split}_fold{fold}",
        fontsize=10, fontweight="bold")

    for ti, task in enumerate(tasks):
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2, 2,
            subplot_spec=outer_gs[ti],
            height_ratios=[4, 1],
            width_ratios=[6, 1],
            hspace=0.08, wspace=0.06)

        ax_main = fig.add_subplot(inner_gs[0, 0])
        ax_bar  = fig.add_subplot(inner_gs[0, 1], sharey=ax_main)
        ax_diff = fig.add_subplot(inner_gs[1, 0], sharex=ax_main)
        ax_bar2 = fig.add_subplot(inner_gs[1, 1])
        ax_bar2.set_visible(False)

        task_color = TASK_COLORS.get(task, "#333")

        # ── collect SAB matrices, stratified by risk/label ──
        # Use canonical alignment: place each patient's modality blocks at their
        # canonical position in the full 64×64 matrix so all patients contribute,
        # not just those with all 4 modalities.
        def _canonical_sab(mat, pres_ordered):
            can = np.zeros((total_seeds, total_seeds))
            pat_off = {}
            off = 0
            for mo in pres_ordered:
                pat_off[mo] = off
                off += K
            for mo_r in pres_ordered:
                if mo_r not in present_mods_main:
                    continue
                pr = pat_off[mo_r]
                cr = present_mods_main.index(mo_r) * K
                for mo_c in pres_ordered:
                    if mo_c not in present_mods_main:
                        continue
                    pc = pat_off[mo_c]
                    cc = present_mods_main.index(mo_c) * K
                    can[cr:cr + K, cc:cc + K] = mat[pr:pr + K, pc:pc + K]
            return can

        mats_all, mats_pos, mats_neg = [], [], []
        logits_task = [r.get("logits", {}).get(task) for r in results]
        valid_logits = [v for v in logits_task if v is not None]
        median_logit = float(np.median(valid_logits)) if valid_logits else 0.0

        for r, lg in zip(results, logits_task):
            if task not in r["sab_attn"]:
                continue
            m = r["sab_attn"][task]
            pres_ordered = [mo for mo in MOD_ORDER if mo in r.get("present_mods", set())]
            mt = _canonical_sab(m, pres_ordered)
            mats_all.append(mt)
            if task == "acr_cls":
                lbl = r.get("label")
                if lbl == 1:   mats_pos.append(mt)
                elif lbl == 0: mats_neg.append(mt)
            else:
                if lg is not None:
                    if lg >= median_logit: mats_pos.append(mt)
                    else:                  mats_neg.append(mt)

        if not mats_all:
            ax_main.set_visible(False); ax_bar.set_visible(False)
            ax_diff.set_visible(False); continue

        # Presence masks: 1 where both the row-modality and col-modality are present.
        # Use these as per-cell denominators so absent-modality zeros don't dilute the mean.
        def _pres_mask(r):
            pres_set = r.get("present_mods", set())
            pv = np.array([1.0 if mo in pres_set else 0.0
                           for mo in present_mods_main for _ in range(K)])
            return np.outer(pv, pv)

        cnt_all = np.clip(sum(_pres_mask(r) for r in results
                              if task in r["sab_attn"]), 1, None)
        cnt_pos = np.clip(sum(_pres_mask(r) for r, mt in zip(results, [None] * len(results))
                              if mt is not None), 1, None)

        tm_all  = np.stack(mats_all).sum(0) / cnt_all

        if len(mats_pos) > 2:
            cnt_p = np.clip(
                sum(_pres_mask(r) for r, lg in zip(results, logits_task)
                    if task in r["sab_attn"] and (
                        (task == "acr_cls" and r.get("label") == 1) or
                        (task != "acr_cls" and lg is not None and lg >= median_logit)
                    )), 1, None)
            tm_pos = np.stack(mats_pos).sum(0) / cnt_p
        else:
            tm_pos = tm_all

        if len(mats_neg) > 2:
            cnt_n = np.clip(
                sum(_pres_mask(r) for r, lg in zip(results, logits_task)
                    if task in r["sab_attn"] and (
                        (task == "acr_cls" and r.get("label") == 0) or
                        (task != "acr_cls" and lg is not None and lg < median_logit)
                    )), 1, None)
            tm_neg = np.stack(mats_neg).sum(0) / cnt_n
        else:
            tm_neg = tm_all

        tm_diff = tm_pos - tm_neg   # positive = more attn in high-risk/ACR+

        # ── mean ABMIL alpha per seed (canonical alignment) ──
        alpha_vecs_can = []
        for r in results:
            a = r.get("abmil_attn", {}).get(task)
            if a is None:
                continue
            pres_ordered = [mo for mo in MOD_ORDER if mo in r.get("present_mods", set())]
            can_a = np.zeros(total_seeds)
            off = 0
            for mo in pres_ordered:
                if mo in present_mods_main:
                    ci = present_mods_main.index(mo) * K
                    can_a[ci:ci + K] = a[off:off + K]
                off += K
            alpha_vecs_can.append(can_a)
        if alpha_vecs_can:
            mean_alpha = np.stack(alpha_vecs_can).mean(0)
        else:
            mean_alpha = np.ones(total_seeds) / total_seeds

        # ── main heatmap ──
        vmax = np.percentile(tm_all, 98)
        im = ax_main.imshow(tm_all, cmap="viridis", aspect="auto", vmin=0, vmax=vmax)

        # Modality-pair block outlines (thick colored rectangles)
        for xs, xe, mrow in mod_spans:
            for ys, ye, mcol in mod_spans:
                same = (mrow == mcol)
                col  = MOD_COLORS[mrow] if same else "#ffffff"
                lw   = 1.5 if same else 0.8
                rect = plt.Rectangle((xs - 0.5, ys - 0.5), xe - xs + 1, ye - ys + 1,
                                     linewidth=lw, edgecolor=col, facecolor="none", zorder=4)
                ax_main.add_patch(rect)
        # Divider lines
        for b in mod_boundaries[1:]:
            ax_main.axvline(b - 0.5, color="white", lw=1.0, alpha=0.6)
            ax_main.axhline(b - 0.5, color="white", lw=1.0, alpha=0.6)

        # All seed labels, colored by modality; font size scales with number of seeds
        tick_fs = max(7, min(9, int(130 / max(total_seeds, 1))))
        ax_main.set_xticks(range(total_seeds))
        ax_main.set_xticklabels(seed_labels, fontsize=tick_fs, rotation=90)
        ax_main.set_yticks(range(total_seeds))
        ax_main.set_yticklabels(seed_labels, fontsize=tick_fs)
        for tick, col_v in zip(ax_main.get_xticklabels(), seed_mod_col):
            tick.set_color(col_v)
        for tick, col_v in zip(ax_main.get_yticklabels(), seed_mod_col):
            tick.set_color(col_v)

        # Modality labels on top + full task name title
        TASK_FULL_C = {
            "acr_cls": "ACR Classification", "acr_surv": "ACR Survival",
            "clad": "CLAD Survival", "death": "Death Survival",
            "clad_surv": "CLAD Survival", "death_surv": "Death Survival",
        }
        ax_main.set_title(
            f"{TASK_FULL_C.get(task, task)}  (N={len(mats_all)})",
            fontsize=9, color=task_color, fontweight="bold", pad=14)
        for xs, xe, mname in mod_spans:
            ax_main.text((xs + xe) / 2, -0.10, mname, ha="center", va="top",
                         fontsize=9, color=MOD_COLORS[mname], fontweight="bold",
                         transform=ax_main.get_xaxis_transform())
        ax_main.set_xlabel("Key", fontsize=7)
        ax_main.set_ylabel("Query", fontsize=7)

        fig.colorbar(im, ax=ax_main, shrink=0.6, pad=0.01, label="SAB attention weight")

        # ── right sidebar: ABMIL alpha bars with modality color + value annotations ──
        ax_bar.barh(range(total_seeds), mean_alpha, color=seed_mod_col, alpha=0.85, height=0.85)
        # Annotate top-5 seeds by alpha
        top5 = np.argsort(mean_alpha)[-5:]
        for si in top5:
            ax_bar.text(mean_alpha[si] + 0.001, si, f"{mean_alpha[si]:.3f}",
                        va="center", fontsize=4.5, color="#333", fontweight="bold")
        ax_bar.set_xlim(0, mean_alpha.max() * 1.35)
        ax_bar.set_xlabel("Mean α", fontsize=8)
        ax_bar.tick_params(axis="both", labelsize=tick_fs)
        ax_bar.set_title("ABMIL α\n(pred. weight)", fontsize=8)
        ax_bar.invert_yaxis()
        # Modality separators on sidebar
        for b in mod_boundaries[1:]:
            ax_bar.axhline(b - 0.5, color="#888", lw=0.7, ls="--")

        # ── bottom: differential heatmap (ACR+ or high-risk) − (ACR- or low-risk) ──
        pos_label = "ACR+" if task == "acr_cls" else "High-risk"
        neg_label = "ACR−" if task == "acr_cls" else "Low-risk"
        vd = np.abs(tm_diff).max()
        im2 = ax_diff.imshow(tm_diff[np.newaxis, :, :].squeeze(0) if tm_diff.ndim == 2
                              else tm_diff,
                              cmap="RdBu_r", aspect="auto", vmin=-vd, vmax=vd)
        for b in mod_boundaries[1:]:
            ax_diff.axvline(b - 0.5, color="#555", lw=0.8)
        ax_diff.set_yticks([0])
        ax_diff.set_yticklabels([f"{pos_label}−{neg_label}"], fontsize=8)
        ax_diff.set_xticks([])
        ax_diff.set_xlabel(f"{pos_label} vs {neg_label}  "
                           f"(n+={len(mats_pos)}, n−={len(mats_neg)})", fontsize=8)
        fig.colorbar(im2, ax=ax_diff, orientation="horizontal",
                     shrink=0.6, pad=0.35, label="Δ attn")

    fig.savefig(out_dir / "C_sab_crossmodal_attn.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "C_sab_crossmodal_attn.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── Sparse heatmap: only significant cross-attentions (top 10% |Δ|) ─────
    # Collect all tm_diff matrices across tasks for a shared threshold
    sig_data = {}  # task -> (tm_all, tm_diff, mats_pos, mats_neg)
    for task in tasks:
        idx_valid = [i for i, r in enumerate(results) if task in r.get("sab_attn", {})]
        if len(idx_valid) < 10:
            continue
        mats_all, mats_pos, mats_neg = [], [], []
        logits_task = [r.get("logits", {}).get(task) for r in results]
        valid_logits = [v for v in logits_task if v is not None]
        median_logit = float(np.median(valid_logits)) if valid_logits else 0.0

        def _pres_mask_c(r):
            pres_set = r.get("present_mods", set())
            pv = np.array([1.0 if mo in pres_set else 0.0
                           for mo in present_mods_main for _ in range(K)])
            return np.outer(pv, pv)

        def _canon(mat, pres_ordered):
            can = np.zeros((total_seeds, total_seeds))
            pat_off = {}; off = 0
            for mo in pres_ordered:
                pat_off[mo] = off; off += K
            for mo_r in pres_ordered:
                if mo_r not in present_mods_main: continue
                pr = pat_off[mo_r]; cr = present_mods_main.index(mo_r) * K
                for mo_c in pres_ordered:
                    if mo_c not in present_mods_main: continue
                    pc = pat_off[mo_c]; cc = present_mods_main.index(mo_c) * K
                    can[cr:cr+K, cc:cc+K] = mat[pr:pr+K, pc:pc+K]
            return can

        for r, lg in zip(results, logits_task):
            if task not in r.get("sab_attn", {}): continue
            pres_ordered = [mo for mo in MOD_ORDER if mo in r.get("present_mods", set())]
            mt = _canon(r["sab_attn"][task], pres_ordered)
            mats_all.append(mt)
            is_pos = (task == "acr_cls" and r.get("label") == 1) or \
                     (task != "acr_cls" and lg is not None and lg >= median_logit)
            if is_pos: mats_pos.append(mt)
            else:      mats_neg.append(mt)

        cnt_all = np.clip(sum(_pres_mask_c(r) for r in results
                              if task in r.get("sab_attn", {})), 1, None)
        tm_all  = np.stack(mats_all).sum(0) / cnt_all
        if len(mats_pos) > 2 and len(mats_neg) > 2:
            cnt_p = np.clip(sum(_pres_mask_c(r) for r, lg in zip(results, logits_task)
                                if task in r.get("sab_attn", {}) and
                                ((task == "acr_cls" and r.get("label") == 1) or
                                 (task != "acr_cls" and lg is not None and lg >= median_logit))
                                ), 1, None)
            cnt_n = np.clip(sum(_pres_mask_c(r) for r, lg in zip(results, logits_task)
                                if task in r.get("sab_attn", {}) and
                                ((task == "acr_cls" and r.get("label") == 0) or
                                 (task != "acr_cls" and lg is not None and lg < median_logit))
                                ), 1, None)
            tm_pos = np.stack(mats_pos).sum(0) / cnt_p
            tm_neg = np.stack(mats_neg).sum(0) / cnt_n
            tm_diff = tm_pos - tm_neg
        else:
            tm_diff = np.zeros_like(tm_all)
        sig_data[task] = (tm_all, tm_diff, len(mats_pos), len(mats_neg))

    if sig_data:
        # Significant cross-attention plot.
        # Only seeds that participate in ≥1 significant (top-10% |Δ|) pair are shown.
        # This collapses the 64×64 full matrix to a compact submatrix of active seeds —
        # rows and columns are seeds that have meaningful differential cross-attention
        # between high-risk and low-risk patients.  Modality source is colour-coded on
        # tick labels.  Red = more attention in high-risk, Blue = more in low-risk.
        ntasks_s = len(sig_data)
        TASK_FULL_C = {"acr_cls":"ACR Classif.","acr_surv":"ACR Survival",
                       "clad_surv":"CLAD Survival","death_surv":"Death Survival",
                       "clad":"CLAD Survival","death":"Death Survival"}

        # Pre-compute active seed indices per task (union used for shared axis sizing)
        task_active = {}
        for task, (tm_all, tm_diff, np_, nn_) in sig_data.items():
            thresh = np.percentile(np.abs(tm_diff), 90)
            sig_mask = np.abs(tm_diff) >= thresh
            active = np.where(sig_mask.any(axis=0) | sig_mask.any(axis=1))[0]
            task_active[task] = active

        # Figure width scales with max active seeds across tasks
        max_active = max((len(v) for v in task_active.values()), default=8)
        cell_size = max(0.40, min(0.70, 14.0 / max(max_active, 1)))
        fig_w = max(6, max_active * cell_size + 3.5) * ntasks_s
        fig_h = max(5, max_active * cell_size + 2.5)
        fig_s, axes_s = plt.subplots(1, ntasks_s, figsize=(fig_w, fig_h), squeeze=False)

        split_lbl = "all_splits" if split < 0 else f"split{split}_fold{fold}"
        fig_s.suptitle(
            f"C — Significant SAB cross-attention  |  {split_lbl}",
            fontsize=10, fontweight="bold")

        for ti, (task, (tm_all, tm_diff, np_, nn_)) in enumerate(sig_data.items()):
            ax = axes_s[0, ti]
            active = task_active[task]
            if len(active) == 0:
                ax.text(0.5, 0.5, "no significant pairs", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
                ax.set_title(TASK_FULL_C.get(task, task), fontsize=9,
                             color=TASK_COLORS.get(task, "#333"), fontweight="bold")
                continue

            thresh = np.percentile(np.abs(tm_diff), 90)
            sparse = np.where(np.abs(tm_diff) >= thresh, tm_diff, 0.0)
            # Sub-matrix: rows and cols restricted to active seeds only
            sub = sparse[np.ix_(active, active)]
            vd = np.abs(sub).max() or 1e-8
            im = ax.imshow(sub, cmap="RdBu_r", aspect="auto", vmin=-vd, vmax=vd)

            a_lbls = [seed_labels[i] for i in active]
            a_cols = [seed_mod_col[i] for i in active]
            n_a = len(active)
            tick_fs = max(6, min(9, int(110 / max(n_a, 1))))
            ax.set_xticks(range(n_a)); ax.set_xticklabels(a_lbls, fontsize=tick_fs, rotation=90)
            ax.set_yticks(range(n_a));  ax.set_yticklabels(a_lbls, fontsize=tick_fs)
            for tick, col_v in zip(ax.get_xticklabels(), a_cols): tick.set_color(col_v)
            for tick, col_v in zip(ax.get_yticklabels(), a_cols): tick.set_color(col_v)

            ax.set_title(f"{TASK_FULL_C.get(task, task)}  (n+={np_}, n−={nn_})",
                         fontsize=9, color=TASK_COLORS.get(task, "#333"), fontweight="bold")
            ax.set_xlabel("Key", fontsize=7)
            ax.set_ylabel("Query", fontsize=7)
            fig_s.colorbar(im, ax=ax, shrink=0.7, label="Δ attn")

        fig_s.savefig(out_dir / "C_sab_significant.pdf", dpi=150, bbox_inches="tight")
        fig_s.savefig(out_dir / "C_sab_significant.png", dpi=120, bbox_inches="tight")
        plt.close(fig_s)
    print("  C done")


# ── Panel D: ABMIL seed importances — all seeds, raw logits, prediction link ──

def panel_D(results, tasks, out_dir, split, fold):
    """
    ABMIL seed importances and their link to the prediction.

    After the SAB cross-modal exchange, each seed passes through a per-task
    gated-attention ABMIL head.  The head scores every seed with a raw logit
    (pre-softmax) and then normalises all seeds jointly via a single global
    softmax to produce attention weights α.  The final task embedding is the
    α-weighted sum of seed representations.

    Outputs (D_abmil_seed_importance.png) — 3 rows per task:
    ──────────────────────────────────────────────────────
    Row 1 — Raw pre-softmax logit ± SD
        What the gating network assigns to each seed before competition.
        Higher = the network prefers this seed for aggregation.
        Seeds are ordered by modality block (HE·s00–s15, BAL·s00–s15, …).

    Row 2 — Global softmax α ± SD  (dashed = uniform 1/N reference)
        The actual weight each seed receives after global competition.
        IMPORTANT: a seed's α depends on ALL other seeds' logits, not only its
        own.  If a patient happens to have many high-logit seeds from one
        modality, seeds from another modality can receive low α even if their
        raw logits are moderate.  Do not interpret raw logit and α as equivalent.
        Seeds above the uniform line (1/N = 1/64 for 4 modalities) attract
        disproportionate attention relative to a random aggregator.

    Row 3 — Pearson r(alpha_k, final prediction logit)
        Correlation across all patients between seed k's attention weight and
        the task-specific prediction logit.
        Red bars: seed k is more attended in high-risk patients.
        Blue bars: seed k is more attended in low-risk patients.
        Top-3 positive and top-3 negative seeds are annotated by label.
        CAVEAT: alpha is confounded by softmax competition. If BAL/Clinical seeds
        dominate in disease cases, HE alpha is suppressed even if HE raw logits
        carry no disease signal — can create spurious "health" correlation for HE.

    Row 4 — Pearson r(raw logit_k, final prediction logit)  [competition-free]
        Same as row 3 but uses the pre-softmax raw logit instead of alpha.
        Raw logits are not affected by what other seeds score, so this row
        reflects the gating network's intrinsic seed preference independently
        of modality competition.
        Key comparison: if row 3 shows negative HE r but row 4 is flat/neutral,
        the HE health-correlation in row 3 is a softmax squeeze-out artefact —
        BAL/Clinical are winning attention in disease, not HE being protective.
        If both rows 3 and 4 are negative for HE, the health signal is genuine.
    """
    K = 16

    TASK_FULL = {
        "acr_cls":  "ACR Classification", "acr_surv": "ACR Survival",
        "clad":     "CLAD Survival",       "death":    "Death Survival",
        "clad_surv": "CLAD Survival",      "death_surv": "Death Survival",
    }
    ntasks = len(tasks)

    # ── Per-task, per-modality data collection using correct per-patient offsets ──
    # For each modality: separate seed arrays extracted from each patient using
    # per-patient offsets based on their present_mods (avoids mislabeling seeds
    # from one modality as another when patient coverage differs).
    task_data = {}  # task -> {mod -> {raw: (n,K), norm: (n,K), logits: (n,)}}
    for task in tasks:
        raw_by_mod  = {mo: [] for mo in MOD_ORDER}
        norm_by_mod = {mo: [] for mo in MOD_ORDER}
        log_by_mod  = {mo: [] for mo in MOD_ORDER}
        for r in results:
            pres_ordered = [m for m in MOD_ORDER if m in r.get("present_mods", set())]
            raw_r  = r.get("abmil_raw_logits", {}).get(task)
            norm_r = r.get("abmil_attn", {}).get(task)
            logit_r = r.get("logits", {}).get(task)
            if raw_r is None or norm_r is None:
                continue
            offset = 0
            for mo in pres_ordered:
                raw_by_mod[mo].append(raw_r[offset:offset + K])
                norm_by_mod[mo].append(norm_r[offset:offset + K])
                if logit_r is not None:
                    log_by_mod[mo].append(logit_r)
                offset += K
        task_data[task] = {
            "raw":  {mo: np.stack(raw_by_mod[mo])  if raw_by_mod[mo]  else np.empty((0, K)) for mo in MOD_ORDER},
            "norm": {mo: np.stack(norm_by_mod[mo]) if norm_by_mod[mo] else np.empty((0, K)) for mo in MOD_ORDER},
            "log":  {mo: np.array(log_by_mod[mo])  if log_by_mod[mo]  else np.array([])     for mo in MOD_ORDER},
        }

    # Modalities present in ≥5 patients (across all tasks, union)
    n_per_mod = {mo: max(task_data[t]["norm"][mo].shape[0] for t in tasks) for mo in MOD_ORDER}
    present_mods_main = [mo for mo in MOD_ORDER if n_per_mod[mo] >= 5]

    if not present_mods_main:
        print("  D skipped (no modality with ≥5 patients)"); return

    # Build global seed labels / colors / boundaries from present_mods_main
    seed_labels    = []
    seed_colors    = []
    mod_boundaries = []
    for mod in present_mods_main:
        mod_boundaries.append(len(seed_labels))
        seed_labels.extend([f"{mod}·s{k:02d}" for k in range(K)])
        seed_colors.extend([MOD_COLORS[mod]] * K)
    n_seeds = len(seed_labels)

    panel_w_d = max(5.0, n_seeds * 0.38 + 2.5)
    fig, axes = plt.subplots(4, ntasks, figsize=(min(panel_w_d * ntasks, 30), 17),
                             gridspec_kw={"hspace": 0.65, "wspace": 0.40,
                                          "height_ratios": [2, 2, 2, 2]})
    if ntasks == 1:
        axes = axes.reshape(4, 1)
    uniform = 1.0 / n_seeds if n_seeds > 0 else 0.0
    # Global softmax caveat: a seed's α depends on ALL other seeds' raw logits in the same
    # patient — not just its own. So a seed with a moderate raw logit can receive HIGH α
    # if the other seeds happen to have lower logits in that patient. This is NOT a bug;
    # it is the correct global-competition property of ABMIL softmax attention.
    # Reading guide: Row 1 = what the gating network scores before competition.
    #                Row 2 = weight after global competition (use this for biology).
    #                Row 3 = correlation with the final prediction logit.
    fig.suptitle(
        f"D — Seed Importances  |  split{split}_fold{fold}\n"
        "Row 1: raw pre-softmax logit  ·  "
        "Row 2: global softmax α (dashed = uniform 1/N)  ·  "
        "Row 3: Pearson r(α, logit)  ·  "
        "Row 4: Pearson r(raw logit, logit) — competition-free",
        fontsize=11, fontweight="bold")

    def _add_dividers(ax, mods, boundaries, n_s):
        for b in boundaries[1:]:
            ax.axvline(b - 0.5, color="#888", lw=1.0, ls="--")
        for mi, (b, mod) in enumerate(zip(boundaries, mods)):
            end = boundaries[mi + 1] if mi + 1 < len(boundaries) else n_s
            ax.axvspan(b - 0.5, end - 0.5, alpha=0.06, color=MOD_COLORS[mod], zorder=0)
            ylim   = ax.get_ylim()
            yspan  = ylim[1] - ylim[0]
            ax.text(b + (end - b) / 2 - 0.5, ylim[1] - 0.03 * yspan, mod,
                    ha="center", fontsize=8, color=MOD_COLORS[mod], fontweight="bold", va="top",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec=MOD_COLORS[mod], alpha=0.7, linewidth=0.8))

    def _add_xlabels(ax, lbls, n_s):
        xlbl_fs = max(5, min(8, int(120 / max(n_s, 1))))
        ax.set_xticks(np.arange(n_s))
        ax.set_xticklabels(lbls[:n_s], rotation=65, ha="right", fontsize=xlbl_fs)
        for tick, lbl in zip(ax.get_xticklabels(), lbls[:n_s]):
            tick.set_color(MOD_COLORS.get(lbl.split("·")[0], "#333"))

    # Collect global (patient-level) softmax arrays for each task using canonical alignment
    # abmil_attn is already global softmax over all present seeds concatenated
    global_norm = {task: [] for task in tasks}
    global_raw  = {task: [] for task in tasks}
    global_log  = {task: [] for task in tasks}
    for r in results:
        pres_ordered = [m for m in MOD_ORDER if m in r.get("present_mods", set())]
        for task in tasks:
            raw_r  = r.get("abmil_raw_logits", {}).get(task)
            norm_r = r.get("abmil_attn", {}).get(task)
            logit_r = r.get("logits", {}).get(task)
            if raw_r is None or norm_r is None:
                continue
            # Place into canonical n_seeds vector (zeros for absent modalities)
            can_raw  = np.zeros(n_seeds)
            can_norm = np.zeros(n_seeds)
            off = 0
            for mo in pres_ordered:
                if mo not in present_mods_main:
                    off += K; continue
                ci = present_mods_main.index(mo) * K
                can_raw[ci:ci+K]  = raw_r[off:off+K]
                can_norm[ci:ci+K] = norm_r[off:off+K]
                off += K
            global_raw[task].append(can_raw)
            global_norm[task].append(can_norm)
            if logit_r is not None:
                global_log[task].append(logit_r)

    for ti, task in enumerate(tasks):
        ax_raw       = axes[0, ti]
        ax_norm      = axes[1, ti]
        ax_corr      = axes[2, ti]
        ax_corr_raw  = axes[3, ti]   # NEW: Pearson r of raw logit (competition-free)
        task_color = TASK_COLORS.get(task, "#333")

        raw_mat  = np.stack(global_raw[task])  if global_raw[task]  else np.zeros((1, n_seeds))
        norm_mat = np.stack(global_norm[task]) if global_norm[task] else np.zeros((1, n_seeds))
        log_arr  = np.array(global_log[task])  if global_log[task]  else np.array([])

        mu_r = raw_mat.mean(0);  se_r = raw_mat.std(0)
        mu_n = norm_mat.mean(0); se_n = norm_mat.std(0)

        if mu_r.sum() == 0 and mu_n.sum() == 0:
            for ax in [ax_raw, ax_norm, ax_corr, ax_corr_raw]:
                ax.set_visible(False)
            continue

        x      = np.arange(n_seeds)
        colors = seed_colors[:n_seeds]

        # Row 0: raw logits (pre-softmax)
        ax_raw.bar(x, mu_r, color=colors, alpha=0.85, width=0.9)
        ax_raw.errorbar(x, mu_r, yerr=se_r, fmt="none", color="#444", lw=0.7, capsize=1.5)
        ax_raw.axhline(0, color="#888", lw=0.7, ls=":")
        ax_raw.set_title(f"{TASK_FULL.get(task, task)}\nRaw logit ± SD",
                         fontsize=9, color=task_color, fontweight="bold", pad=6)
        ax_raw.set_ylabel("Raw logit", fontsize=8)
        ax_raw.tick_params(axis="both", labelsize=6)
        _add_xlabels(ax_raw, seed_labels, n_seeds)
        _add_dividers(ax_raw, present_mods_main, mod_boundaries, n_seeds)

        # Row 1: global softmax α over ALL seeds — uniform reference line at 1/n_seeds
        ax_norm.bar(x, mu_n, color=colors, alpha=0.85, width=0.9)
        ax_norm.errorbar(x, mu_n, yerr=se_n, fmt="none", color="#444", lw=0.7, capsize=1.5)
        ax_norm.axhline(uniform, color="#555", lw=1.2, ls="--",
                        label=f"uniform 1/{n_seeds}={uniform:.4f}")
        ax_norm.legend(fontsize=6.5, framealpha=0.8, loc="upper right")
        ax_norm.set_title("Global softmax α ± SD", fontsize=8)
        ax_norm.set_ylabel("Mean α", fontsize=8)
        ax_norm.tick_params(axis="both", labelsize=6)
        _add_xlabels(ax_norm, seed_labels, n_seeds)
        _add_dividers(ax_norm, present_mods_main, mod_boundaries, n_seeds)

        # Row 2: Pearson r(global α_k, prediction logit) — using canonical global arrays
        corrs = []
        for k in range(n_seeds):
            col_k = norm_mat[:, k]
            if len(log_arr) == norm_mat.shape[0] and norm_mat.shape[0] > 5 \
                    and col_k.std() > 1e-8 and log_arr.std() > 1e-8:
                corrs.append(float(np.corrcoef(col_k, log_arr)[0, 1]))
            else:
                corrs.append(0.0)
        corrs = np.array(corrs)

        if corrs.any():
            bar_colors = ["#d62728" if c > 0 else "#1f77b4" for c in corrs]
            ax_corr.bar(x, corrs, color=bar_colors, alpha=0.85, width=0.9)
            ax_corr.axhline(0, color="#333", lw=0.7)
            ax_corr.set_title("Pearson r(α, risk logit)  [red=+risk]",
                              fontsize=8)
            ax_corr.set_ylabel("Pearson r", fontsize=7)
            ax_corr.tick_params(axis="both", labelsize=6)
            _add_xlabels(ax_corr, seed_labels, n_seeds)
            _add_dividers(ax_corr, present_mods_main, mod_boundaries, n_seeds)
            top_pos = np.argsort(corrs)[-3:]
            top_neg = np.argsort(corrs)[:3]
            for k in list(top_pos) + list(top_neg):
                ax_corr.text(k, corrs[k] + (0.015 if corrs[k] >= 0 else -0.015),
                             seed_labels[k], ha="center",
                             fontsize=7, va="bottom" if corrs[k] >= 0 else "top",
                             color="#333", fontweight="bold")
        else:
            ax_corr.text(0.5, 0.5, "insufficient data", ha="center", va="center",
                         transform=ax_corr.transAxes, fontsize=9)
            ax_corr.set_visible(False)

        # Row 3: Pearson r(raw logit_k, prediction logit) — NOT affected by softmax competition.
        # Compare this row with row 2 (Pearson r of α):
        #   If row 2 and row 3 agree: the correlation is real biology.
        #   If row 3 is flat/neutral but row 2 shows negative HE r: it is a softmax
        #   competition artefact — BAL/Clinical seeds dominate softmax in disease cases,
        #   squeezing HE α down even if HE raw logits show no disease preference.
        corrs_raw = []
        for k in range(n_seeds):
            col_k = raw_mat[:, k]
            if len(log_arr) == raw_mat.shape[0] and raw_mat.shape[0] > 5 \
                    and col_k.std() > 1e-8 and log_arr.std() > 1e-8:
                corrs_raw.append(float(np.corrcoef(col_k, log_arr)[0, 1]))
            else:
                corrs_raw.append(0.0)
        corrs_raw = np.array(corrs_raw)

        if corrs_raw.any():
            bar_colors_r = ["#d62728" if c > 0 else "#1f77b4" for c in corrs_raw]
            ax_corr_raw.bar(x, corrs_raw, color=bar_colors_r, alpha=0.85, width=0.9)
            ax_corr_raw.axhline(0, color="#333", lw=0.7)
            ax_corr_raw.set_title("Pearson r(raw logit, risk logit)  [competition-free]",
                                  fontsize=8)
            ax_corr_raw.set_ylabel("Pearson r", fontsize=7)
            ax_corr_raw.tick_params(axis="both", labelsize=6)
            _add_xlabels(ax_corr_raw, seed_labels, n_seeds)
            _add_dividers(ax_corr_raw, present_mods_main, mod_boundaries, n_seeds)
            top_pos = np.argsort(corrs_raw)[-3:]
            top_neg = np.argsort(corrs_raw)[:3]
            for k in list(top_pos) + list(top_neg):
                ax_corr_raw.text(k, corrs_raw[k] + (0.015 if corrs_raw[k] >= 0 else -0.015),
                                 seed_labels[k], ha="center",
                                 fontsize=7, va="bottom" if corrs_raw[k] >= 0 else "top",
                                 color="#333", fontweight="bold")
        else:
            ax_corr_raw.set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "D_abmil_seed_importance.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(str(out_dir / "D_abmil_seed_importance.pdf").replace(".pdf", ".png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  D done")


# ── Panel E: TaskModalGate weights ───────────────────────────────────────────

def panel_E(results, tasks, out_dir, split, fold):
    gate_rows = [r["gate_vals"] for r in results if r["gate_vals"]]
    if not gate_rows:
        print("  E skipped (no gate values)"); return
    n_mod = len(MOD_ORDER)
    gate_matrix = np.zeros((len(tasks), n_mod))
    for ti, task in enumerate(tasks):
        vals = np.stack([g[task] for g in gate_rows if task in g])
        if vals.shape[0] == 0:
            continue
        gate_matrix[ti, :vals.shape[1]] = vals.mean(0)

    # Auto-scale to data range so subtle task/modality differences are visible
    g_lo = gate_matrix[gate_matrix > 0].min() if (gate_matrix > 0).any() else 0.0
    g_hi = gate_matrix.max()
    spread = g_hi - g_lo
    if spread < 0.04:
        g_mid = gate_matrix.mean()
        g_lo  = max(0.0, g_mid - 0.02)
        g_hi  = min(1.0, g_mid + 0.02)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(gate_matrix, cmap="YlGn", vmin=g_lo, vmax=g_hi, aspect="auto")
    ax.set_xticks(range(n_mod)); ax.set_xticklabels(MOD_ORDER, fontsize=10)
    ax.set_yticks(range(len(tasks))); ax.set_yticklabels(tasks, fontsize=9)
    g_thresh = (g_lo + g_hi) / 2
    for ti in range(len(tasks)):
        for mi in range(n_mod):
            v = gate_matrix[ti, mi]
            ax.text(mi, ti, f"{v:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if v > g_thresh else "black")
    ax.set_title(f"E — TaskModalGate weights (mean, {len(gate_rows)} patients) | split{split}_fold{fold}",
                 fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label=f"Gate weight [{g_lo:.3f}–{g_hi:.3f}]")
    fig.tight_layout()
    fig.savefig(out_dir / "E_task_modal_gate.pdf", dpi=150, bbox_inches="tight"); fig.savefig(str(out_dir / "E_task_modal_gate.pdf").replace(".pdf", ".png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  E done")


# ── Panel F: modality combo scores ───────────────────────────────────────────

def panel_F(results, tasks, out_dir, split, fold, variant="mega"):
    """
    Unimodal ablation: load per-modality test metrics from the saved JSON and plot
    a grouped bar chart (modality × metric) versus the all-modality baseline.
    """
    import json as _json
    vdir = "set_mil_mt_mega" if variant == "mega" else f"set_mil_mt_{variant}"
    metrics_path = RESULTS_ROOT / f"split{split}_fold{fold}" / vdir / "metrics_set_mil_mt_final.json"
    if not metrics_path.exists():
        print(f"  F skipped (metrics not found: {metrics_path})"); return
    with open(metrics_path) as fh:
        mdata = _json.load(fh)

    unimod = mdata.get("unimodal_ablation", {})
    test   = mdata.get("test", {})
    if not unimod:
        print("  F skipped (no unimodal_ablation in metrics)"); return

    # Metric definitions: (label, key-in-unimod, key-in-test, chance)
    METRICS = [
        ("ACR BACC",    "bacc",           "bacc",           0.5),
        ("ACR C-index", "acr_c_index",    "c_index",        0.5),
        ("CLAD C-index","clad_c_index",   "clad_c_index",   0.5),
        ("Death C-index","death_c_index", "death_c_index",  0.5),
    ]

    mods = MOD_ORDER  # ["HE", "BAL", "CT", "Clinical"]
    n_metrics = len(METRICS)
    n_mods    = len(mods)

    # Collect bars: rows = metrics, cols = mods + "All"
    bar_data   = np.full((n_metrics, n_mods + 1), float("nan"))
    bar_ns     = np.full((n_mods + 1,), 0, dtype=int)
    for mi, mo in enumerate(mods):
        ab = unimod.get(mo, {})
        bar_ns[mi] = ab.get("n", 0)
        for ki, (_, uk, _, _) in enumerate(METRICS):
            bar_data[ki, mi] = ab.get(uk, float("nan"))
    # All-modality column
    bar_ns[-1] = len(results)
    for ki, (_, _, tk, _) in enumerate(METRICS):
        bar_data[ki, -1] = test.get(tk, float("nan"))

    colors = [MOD_COLORS.get(m, "#aaa") for m in mods] + ["#333333"]
    labels = [f"{m}\n(n={bar_ns[i]})" for i, m in enumerate(mods)] + [f"All\n(n={bar_ns[-1]})"]

    fig, axes = plt.subplots(1, n_metrics, figsize=(3.5 * n_metrics, 4.5), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    fig.suptitle(
        f"F — Unimodal ablation: per-modality vs. all-modality | split{split}_fold{fold}",
        fontsize=11, fontweight="bold")

    x = np.arange(n_mods + 1)
    bar_w = 0.65
    for ki, (ax, (metric_lbl, _, _, chance)) in enumerate(zip(axes, METRICS)):
        vals = bar_data[ki]
        bars = ax.bar(x, vals, width=bar_w, color=colors, alpha=0.82, edgecolor="white", lw=0.5)
        ax.axhline(chance, color="#aaa", lw=0.8, ls="--", zorder=0, label="Chance")
        # Annotate values
        for xi, (bar, v) in enumerate(zip(bars, vals)):
            if not np.isnan(v):
                ax.text(xi, v + 0.005, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold",
                        color=colors[xi])
        # Highlight the "All" bar with a black edge
        bars[-1].set_edgecolor("#111"); bars[-1].set_linewidth(1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(metric_lbl, fontsize=9, fontweight="bold")
        ax.set_ylabel("Score", fontsize=8)
        ylo = min(chance - 0.05, np.nanmin(vals) - 0.03)
        yhi = max(np.nanmax(vals) + 0.07, chance + 0.05)
        ax.set_ylim(max(0, ylo), min(1, yhi))
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / "F_modality_combo_ablation.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(str(out_dir / "F_modality_combo_ablation.pdf").replace(".pdf", ".png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  F done")


# ── Panel G: final rep hexbin (5 panels) ─────────────────────────────────────

def panel_G(results, tasks, out_dir, split, fold):
    """
    Final-representation UMAP with clinical overlays — one figure per task.

    Each patient's final_rep (the task-specific embedding produced by ABMIL after
    attention-weighted seed aggregation) is projected to 2D with UMAP.  The same
    2D coordinates are reused across all subplots within a task; only the
    colour/size mapping changes.

    Subplots (7 per task, saved as G_final_rep_hexbin_{task}.png):
    ──────────────────────────────────────────────────────────────
    0 — ACR label
        Points coloured red (ACR+/rejection=1) or blue (ACR−=0); grey if unknown.
        Tests whether the embedding clusters by rejection status.

    1 — Normalized risk score
        acr_cls: sigmoid P(ACR+) mapped to [0,1].
        survival tasks: percentile rank of Cox logit within full test cohort [0,1].
        Colour: red=high risk, blue=low risk.  Tests whether the prediction score
        is spatially coherent in representation space.

    2 — TTE with event markers
        Marker size ∝ TTE (shorter TTE = larger marker = more urgent).
        Colour: red=event occurred, blue=censored.  Subplot title shows median TTE.
        Tests whether patients with imminent events cluster in a specific region.

    3 — Modality combination
        Each unique combination of present modalities gets a distinct colour.
        Combos with <5 patients are collapsed to "Other".
        Tests whether the embedding is confounded by data availability rather
        than biology (a clean model should not cluster by modality combo).

    4 — KM: top vs bottom tertile
        Kaplan-Meier curves for patients in the top third vs bottom third of the
        risk score distribution.  Separation validates that the learned embedding
        captures survival-relevant information beyond simple stratification.

    5 — (reserved / empty in current layout)

    6 — CV split annotation
        Points coloured by outer cross-validation split (0–4).
        Tests whether any visible cluster is an artefact of a single split.
        A good model should have splits mixed uniformly across the UMAP.

    KDE contours
        Thin density contours (linewidths=0.4, 2 levels) are overlaid on most
        panels to guide the eye to high-density regions without hiding individual
        points (MS=8, marker size deliberately small).
    """
    ep_keys = {
        "acr_cls":    ("event_acr",   "tte_acr"),
        "acr_surv":   ("event_acr",   "tte_acr"),
        "clad_surv":  ("event_clad",  "tte_clad"),
        "death_surv": ("event_death", "tte_death"),
    }

    # Build combo → color mapping; collapse combos with < MIN_COMBO_N into "Other"
    MIN_COMBO_N = 5
    try:
        _tab20 = matplotlib.colormaps["tab20"]
    except (KeyError, AttributeError):
        _tab20 = plt.cm.get_cmap("tab20")
    raw_combos = ["+".join(sorted(r["present_mods"])) for r in results]
    from collections import Counter
    combo_counts = Counter(raw_combos)
    major_combos = sorted([c for c, n in combo_counts.items() if n >= MIN_COMBO_N],
                          key=lambda c: -combo_counts[c])
    combo_color = {c: _tab20(i % 20) for i, c in enumerate(major_combos)}
    combo_color["Other"] = "#aaaaaa"

    def _kde_contours(ax, pts, color, levels=2, alpha=0.5):
        """Overlay KDE density contours for a point cloud."""
        if len(pts) < 10:
            return
        try:
            from scipy.stats import gaussian_kde
            k = gaussian_kde(pts.T, bw_method=0.18)
            xg = np.linspace(pts[:, 0].min() - 0.5, pts[:, 0].max() + 0.5, 80)
            yg = np.linspace(pts[:, 1].min() - 0.5, pts[:, 1].max() + 0.5, 80)
            XX, YY = np.meshgrid(xg, yg)
            ZZ = k(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
            ax.contour(XX, YY, ZZ, levels=levels, colors=[color],
                       alpha=alpha, linewidths=0.3, zorder=2)
        except Exception:
            pass

    for task in tasks:
        idx_valid = [i for i, r in enumerate(results) if task in r["final_reps"]]
        if len(idx_valid) < 20:
            continue
        reps   = np.stack([results[i]["final_reps"][task] for i in idx_valid])
        xy     = _umap_embed(reps)

        ev_key, tte_key = ep_keys.get(task, ("event_acr", "tte_acr"))
        logits = np.array([results[i]["logits"].get(task, float("nan")) for i in idx_valid])
        ev     = np.array([results[i].get(ev_key,  float("nan")) for i in idx_valid])
        tte    = np.array([results[i].get(tte_key, float("nan")) for i in idx_valid])
        labs   = np.array([float(results[i]["label"]) if results[i]["label"] is not None
                            else float("nan") for i in idx_valid])
        combos_here = [raw_combos[i] for i in idx_valid]
        display_combos = [c if c in combo_color else "Other" for c in combos_here]

        # Classification: sigmoid → P(ACR+) ∈ [0,1]
        # Survival: percentile rank within this cohort ∈ [0,1]
        if task == "acr_cls":
            scores    = 1.0 / (1.0 + np.exp(-logits))
            score_lbl = "P(ACR+)"
        else:
            from scipy.stats import rankdata
            valid_log = ~np.isnan(logits)
            scores = np.full_like(logits, float("nan"))
            if valid_log.sum() > 1:
                ranks = (rankdata(logits[valid_log]) - 1) / max(valid_log.sum() - 1, 1)
                scores[valid_log] = ranks
            score_lbl = "Risk percentile (0=low, 1=high)"
        N = len(idx_valid)
        MS = 8   # smaller dots

        splits_here = np.array([results[i].get("_split", -1) for i in idx_valid])

        fig = plt.figure(figsize=(30, 4.5))
        grd = gridspec.GridSpec(1, 7, figure=fig, wspace=0.14)
        axs = [fig.add_subplot(grd[0, i]) for i in range(7)]
        split_lbl = "all_splits" if split < 0 else f"split{split}_fold{fold}"
        fig.suptitle(f"G — {task} | {split_lbl}  N={N}",
                     fontsize=10, fontweight="bold")

        # 0: label scatter with KDE contours per class
        ax = axs[0]
        ax.set_facecolor("#f9f9f9")
        m0 = labs == 0; m1 = labs == 1; mn = np.isnan(labs)
        ax.scatter(xy[m0, 0], xy[m0, 1], s=MS, c="#1E88E5", alpha=0.65,
                   edgecolors="none", label=f"ACR− (n={m0.sum()})", zorder=3)
        ax.scatter(xy[m1, 0], xy[m1, 1], s=MS, c="#E53935", alpha=0.80,
                   edgecolors="none", label=f"ACR+ (n={m1.sum()})", zorder=4)
        if mn.any():
            ax.scatter(xy[mn, 0], xy[mn, 1], s=MS * 0.4, c="#bbb", alpha=0.4, zorder=2)
        if m0.sum() >= 10:
            _kde_contours(ax, xy[m0], "#1E88E5", levels=3, alpha=0.4)
        if m1.sum() >= 10:
            _kde_contours(ax, xy[m1], "#E53935", levels=3, alpha=0.4)
        ax.set_title("ACR label", fontsize=8, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(markerscale=1.5, fontsize=7, framealpha=0.75, loc="best")

        # 1: risk score scatter (scatter is better than hexbin for N~150)
        ax = axs[1]
        ax.set_facecolor("#f9f9f9")
        valid = ~np.isnan(scores)
        if valid.sum() > 5:
            vl = np.nanpercentile(scores[valid], 2)
            vh = np.nanpercentile(scores[valid], 98)
            try:
                _cmap_h = matplotlib.colormaps[CMAP_HAZARD]
            except (KeyError, AttributeError):
                _cmap_h = plt.cm.get_cmap(CMAP_HAZARD)
            norm_s = matplotlib.colors.Normalize(vmin=vl, vmax=vh)
            sc = ax.scatter(xy[valid, 0], xy[valid, 1],
                            c=scores[valid], cmap=CMAP_HAZARD, norm=norm_s,
                            s=MS, alpha=0.80, edgecolors="none", zorder=3)
            cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
            cb.ax.tick_params(labelsize=7)
            cb.set_label(score_lbl, fontsize=7)
            # annotate top-5 highest-risk patients
            top5 = np.argsort(scores[valid])[-5:]
            for ti in top5:
                ax.annotate("▲", xy=(xy[valid][ti, 0], xy[valid][ti, 1]),
                            fontsize=6, color="#B71C1C", ha="center", va="center")
        ax.set_title(score_lbl, fontsize=8, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

        # 2: TTE scatter — events colored by TTE (red=short), censored as small grey
        ax = axs[2]
        ax.set_facecolor("#f9f9f9")
        ev_m = (~np.isnan(tte)) & (ev == 1)
        ce_m = (~np.isnan(tte)) & (ev == 0)
        if ev_m.sum() > 5:
            vl      = np.nanpercentile(tte[ev_m], 2)
            vh      = np.nanpercentile(tte[ev_m], 98)
            vcenter = float(np.nanmedian(tte[ev_m]))
            vcenter = float(np.clip(vcenter, vl + 1e-3, vh - 1e-3))
            tte_norm = matplotlib.colors.TwoSlopeNorm(vcenter=vcenter, vmin=vl, vmax=vh)
            if ce_m.any():
                ax.scatter(xy[ce_m, 0], xy[ce_m, 1], s=MS * 0.3, c="#cccccc",
                           alpha=0.5, edgecolors="none", zorder=2, label=f"Censored (n={ce_m.sum()})")
            sc2 = ax.scatter(xy[ev_m, 0], xy[ev_m, 1],
                             c=tte[ev_m], cmap=CMAP_TTE, norm=tte_norm,
                             s=MS * 1.4, alpha=0.85, edgecolors="white", linewidths=0.3,
                             zorder=4, label=f"Event (n={ev_m.sum()})")
            cb2 = fig.colorbar(sc2, ax=ax, shrink=0.75, pad=0.02)
            cb2.ax.tick_params(labelsize=7)
            cb2.set_label("TTE (days)", fontsize=7)
            # annotate 5 shortest-TTE events (most urgent)
            tte_ev = tte[ev_m]
            xy_ev  = xy[ev_m]
            top5_short = np.argsort(tte_ev)[:5]
            for ti in top5_short:
                ax.annotate(f"{tte_ev[ti]:.0f}d",
                            xy=(xy_ev[ti, 0], xy_ev[ti, 1]),
                            xytext=(4, 4), textcoords="offset points",
                            fontsize=5.5, color="#B71C1C", fontweight="bold")
            ax.legend(fontsize=6, framealpha=0.7, markerscale=1.2,
                      loc="best", handlelength=1)
        ax.set_title(f"TTE (events, med={vcenter if ev_m.sum() > 5 else 0:.0f}d)",
                     fontsize=8, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

        # 3: modality combo — major combos colored, rare → "Other" grey
        ax = axs[3]
        ax.set_facecolor("#f9f9f9")
        _MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h", "<", ">"]
        # plot "Other" first (background)
        other_mask = np.array([c == "Other" for c in display_combos])
        if other_mask.any():
            ax.scatter(xy[other_mask, 0], xy[other_mask, 1], s=MS * 0.6,
                       color="#aaaaaa", alpha=0.5, marker="o",
                       label=f"Other (n={other_mask.sum()})", zorder=2)
        for ci, combo in enumerate(major_combos):
            mask = np.array([c == combo for c in display_combos])
            if not mask.any():
                continue
            mk = _MARKERS[ci % len(_MARKERS)]
            sz = MS * 1.5 if mk in ("P", "X") else MS
            ax.scatter(xy[mask, 0], xy[mask, 1], s=sz, marker=mk,
                       color=combo_color[combo], alpha=0.82, edgecolors="none",
                       label=f"{combo} (n={mask.sum()})", zorder=3 + ci)
        ax.set_title("Modality combo", fontsize=8, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=5.5, markerscale=1.6, framealpha=0.80, loc="best",
                  ncol=1, handlelength=1.2, title="combo (n)", title_fontsize=5.5)

        # 4: risk × TTE scatter with LOWESS trend
        ax = axs[4]
        ax.set_facecolor("#f9f9f9")
        valid4 = ~(np.isnan(scores) | np.isnan(tte))
        if valid4.sum() > 5:
            colors4 = np.array(["#E53935" if e == 1 else "#1E88E5" for e in ev[valid4]])
            ax.scatter(scores[valid4], tte[valid4], c=colors4, s=MS * 0.7, alpha=0.65,
                       edgecolors="none", zorder=3)
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                lo = lowess(tte[valid4], scores[valid4], frac=0.5)
                ax.plot(lo[:, 0], lo[:, 1], color="#E65100", lw=2.0, zorder=5,
                        label="LOWESS")
            except Exception:
                pass
            ax.set_xlabel(score_lbl, fontsize=8)
            ax.set_ylabel("TTE (days)", fontsize=8)
            ax.set_title("Risk vs TTE", fontsize=9, fontweight="bold")
            ax.tick_params(labelsize=7)
            ax.legend(handles=[
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#E53935',
                           markersize=6, label=f'Event (n={(ev[valid4]==1).sum()})'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1E88E5',
                           markersize=6, label=f'Censored (n={(ev[valid4]==0).sum()})'),
                plt.Line2D([0], [0], color='#E65100', lw=2, label='LOWESS'),
            ], fontsize=6.5, framealpha=0.7)
            ax.spines[["top", "right"]].set_visible(False)

        # 5: model-stratified KM / outcome curve (top vs bottom tertile of scores)
        ax = axs[5]
        ax.set_facecolor("#f9f9f9")
        valid5 = ~(np.isnan(scores) | np.isnan(tte) | np.isnan(ev))
        if valid5.sum() > 20:
            sc5  = scores[valid5]; tte5 = tte[valid5]; ev5 = ev[valid5]
            t33  = np.percentile(sc5, 33); t67 = np.percentile(sc5, 67)
            hi   = sc5 >= t67;  lo = sc5 <= t33
            def _km_curve(t_arr, e_arr):
                order = np.argsort(t_arr)
                t_s = t_arr[order]; e_s = e_arr[order]
                n = len(t_s); surv = 1.0; surv_list = [1.0]; t_list = [0.0]
                for i in range(n):
                    if e_s[i] == 1:
                        surv *= (1 - 1.0 / (n - i))
                    surv_list.append(surv); t_list.append(t_s[i])
                return np.array(t_list), np.array(surv_list)
            if hi.sum() >= 5:
                t_hi, s_hi = _km_curve(tte5[hi], ev5[hi])
                ax.step(t_hi, s_hi, where="post", color="#E53935", lw=2.0,
                        label=f"High risk (≥P67, n={hi.sum()})")
            if lo.sum() >= 5:
                t_lo, s_lo = _km_curve(tte5[lo], ev5[lo])
                ax.step(t_lo, s_lo, where="post", color="#1E88E5", lw=2.0,
                        label=f"Low risk (≤P33, n={lo.sum()})")
            ax.set_ylim(0, 1.05); ax.set_xlim(left=0)
            ax.set_xlabel("Days from transplant", fontsize=8)
            ax.set_ylabel("Survival probability", fontsize=8)
            ax.set_title("KM: top vs bottom tertile", fontsize=8, fontweight="bold")
            ax.legend(fontsize=7, framealpha=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7)
        else:
            ax.text(0.5, 0.5, "insufficient data\nfor KM", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#888")
            ax.axis("off")

        # 6: split annotation — color by outer CV split (0–4)
        # Useful to check that the UMAP structure is not driven by data split artefacts.
        ax = axs[6]
        ax.set_facecolor("#f9f9f9")
        split_ids = np.unique(splits_here[splits_here >= 0])
        split_cmap = plt.cm.get_cmap("tab10")
        for si in split_ids:
            smask = splits_here == si
            ax.scatter(xy[smask, 0], xy[smask, 1], s=MS, alpha=0.6,
                       color=split_cmap(int(si) / 10), edgecolors="none",
                       label=f"split{si} (n={smask.sum()})", zorder=3)
        if splits_here[splits_here < 0].any():
            unk = splits_here < 0
            ax.scatter(xy[unk, 0], xy[unk, 1], s=MS * 0.5, alpha=0.3,
                       color="#aaa", edgecolors="none", label="unknown", zorder=2)
        ax.set_title("CV split", fontsize=8, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(fontsize=6, framealpha=0.8, loc="best", markerscale=1.5,
                  handlelength=1, title="split", title_fontsize=6)

        _uniform_lim(axs[:4], xy)
        # Sync split panel limits to match UMAP panels
        xl = axs[0].get_xlim(); yl = axs[0].get_ylim()
        axs[6].set_xlim(xl); axs[6].set_ylim(yl)
        fig.savefig(out_dir / f"G_final_rep_hexbin_{task}.pdf", dpi=150, bbox_inches="tight")
        fig.savefig(out_dir / f"G_final_rep_hexbin_{task}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    print("  G done")


# ── Panel H: information pathway cluster→seed→ABMIL→prediction ───────────────

def panel_H(results, tasks, out_dir, split, fold):
    present_mods_main = [m for m in MOD_ORDER
                          if sum(m in r["present_mods"] for r in results) >= 5]
    K = 16

    aff_pool   = {mod: [] for mod in present_mods_main}
    cnames_ref = {}

    for res in results:
        for mod in present_mods_main:
            h    = res["inst_reps"].get(mod)
            bcos = res.get("pma_bcos", {}).get(mod)
            if bcos is None:
                bcos = res.get("pma_attn", {}).get(mod)
            if h is None or bcos is None or bcos.ndim != 2 or bcos.shape[1] != len(h):
                continue
            # Use pre-computed cluster IDs
            c_ids = res.get("cluster_ids", {}).get(mod)
            if c_ids is not None and len(c_ids) == len(h):
                cl = c_ids
                nms = res.get("cluster_names", {}).get(mod, [])
                if nms and mod not in cnames_ref:
                    cnames_ref[mod] = nms
                k_clus = int(cl.max()) + 1 if len(cl) > 0 else K_PATCH
            else:
                from sklearn.cluster import MiniBatchKMeans
                km = MiniBatchKMeans(n_clusters=K_PATCH, n_init=3, random_state=42, batch_size=4096)
                cl = km.fit_predict(h.astype(np.float32))
                k_clus = K_PATCH
            aff_pool[mod].append(_seed_cluster_mass(bcos, cl, k_clus))

    ntasks = len(tasks)
    fig, axes_grid = plt.subplots(3, max(ntasks, 1), figsize=(5.5 * max(ntasks, 1), 12))
    if ntasks == 1:
        axes_grid = axes_grid[:, np.newaxis]
    fig.suptitle(f"H — Information pathway: input clusters → seeds → ABMIL → prediction\n"
                 f"split{split}_fold{fold}", fontsize=11, fontweight="bold")

    for ti, task in enumerate(tasks):
        ax_aff   = axes_grid[0, ti]
        ax_abmil = axes_grid[1, ti]
        ax_path  = axes_grid[2, ti]

        # row 0: mean seed-cluster affinity stacked across modalities
        aff_rows = []
        mod_tick_pos = []
        mod_tick_lbl = []
        row_offset = 0
        for mod in present_mods_main:
            aff_list = aff_pool.get(mod, [])
            if not aff_list:
                continue
            # Pad to common cluster count before stacking (k_clus can vary per patient)
            max_k_loc = max(a.shape[1] for a in aff_list)
            aff_padded = [np.pad(a, ((0, 0), (0, max_k_loc - a.shape[1]))) for a in aff_list]
            mean_aff = np.stack(aff_padded).mean(0)
            aff_rows.append(mean_aff)
            mod_tick_pos.append(row_offset + K // 2)
            mod_tick_lbl.append(mod)
            row_offset += K

        if aff_rows:
            max_k = max(a.shape[1] for a in aff_rows)
            padded_rows = [np.pad(a, ((0,0),(0, max_k - a.shape[1]))) for a in aff_rows]
            full_aff = np.vstack(padded_rows)
            im0 = ax_aff.imshow(full_aff, cmap="YlOrRd", aspect="auto",
                                vmin=0, vmax=np.percentile(full_aff, 95))
            # X-axis: use cluster names from first modality that has them (or generic)
            all_nms = []
            for mod in present_mods_main:
                nms = cnames_ref.get(mod)
                if nms:
                    all_nms = nms[:max_k]; break
            if not all_nms:
                all_nms = [str(c) for c in range(max_k)]
            step = max(1, max_k // 15)
            shown_x = list(range(0, max_k, step))
            ax_aff.set_xticks(shown_x)
            ax_aff.set_xticklabels([all_nms[c] if c < len(all_nms) else str(c) for c in shown_x],
                                   fontsize=6, rotation=90)
            ax_aff.set_xlabel("Input cluster (pre-computed)", fontsize=8)
            ax_aff.set_yticks(mod_tick_pos)
            ax_aff.set_yticklabels(mod_tick_lbl, fontsize=8)
            for b in range(1, len(aff_rows)):
                ax_aff.axhline(b * K - 0.5, color="white", lw=1.5)
            fig.colorbar(im0, ax=ax_aff, shrink=0.8, label="B-cos attn mass")
        ax_aff.set_title(f"{task}\nSeed ← cluster (B-cos attn)", fontsize=8,
                         color=TASK_COLORS.get(task, "#333"))

        # row 1: mean ABMIL attn per seed — canonical per-patient offsets
        # Build canonical (n_patients, n_present_mods*K) with zeros for absent mods
        n_can = len(present_mods_main) * K
        canonical_rows = []
        for r in results:
            alpha_r = r.get("abmil_attn", {}).get(task)
            if alpha_r is None:
                continue
            pres_ordered = [m for m in MOD_ORDER if m in r.get("present_mods", set())]
            row = np.zeros(n_can)
            offset = 0
            for mo in pres_ordered:
                if mo in present_mods_main:
                    ci = present_mods_main.index(mo) * K
                    row[ci:ci + K] = alpha_r[offset:offset + K]
                offset += K
            canonical_rows.append(row)
        mean_abmil = None
        if canonical_rows:
            mean_abmil = np.stack(canonical_rows).mean(0)
            seed_colors_h = []
            for mod in present_mods_main:
                seed_colors_h.extend([MOD_COLORS[mod]] * K)
            x = np.arange(n_can)
            ax_abmil.bar(x, mean_abmil, color=seed_colors_h, alpha=0.85, width=0.9)
            for b in range(K, n_can, K):
                ax_abmil.axvline(b - 0.5, color="#555", lw=0.8, ls="--")
            for i, mod in enumerate(present_mods_main):
                ax_abmil.text(i * K + K / 2 - 0.5, mean_abmil.max() * 0.93, mod,
                              ha="center", fontsize=7, color=MOD_COLORS[mod], fontweight="bold")
        ax_abmil.set_title("Mean ABMIL seed attn → prediction", fontsize=8)
        ax_abmil.set_xlabel("Seed (MOD_k)", fontsize=7)
        ax_abmil.set_ylabel("Mean attn weight", fontsize=7)
        ax_abmil.set_xticks([])

        # row 2: effective cluster→prediction weight per modality
        if aff_rows and mean_abmil is not None:
            cluster_pred_weights = []
            for mi, mod in enumerate(present_mods_main):
                if not aff_pool.get(mod):
                    continue
                aff_mod_list = aff_pool[mod]
                max_k_m = max(a.shape[1] for a in aff_mod_list)
                aff_mod_pad = [np.pad(a, ((0, 0), (0, max_k_m - a.shape[1]))) for a in aff_mod_list]
                mean_aff_mod = np.stack(aff_mod_pad).mean(0)
                abmil_mod    = mean_abmil[mi * K:(mi + 1) * K]
                cpw = abmil_mod @ mean_aff_mod
                cluster_pred_weights.append((mod, cpw))

            if cluster_pred_weights:
                from matplotlib.colors import LogNorm
                max_k2 = max(cpw.shape[0] for _, cpw in cluster_pred_weights)
                w_mat = np.stack([np.pad(cpw, (0, max_k2 - cpw.shape[0]))
                                  for _, cpw in cluster_pred_weights])
                # Log-norm: most weights are tiny; log scale reveals the full range
                w_pos = np.clip(w_mat, 1e-15, None)
                vmin2 = max(w_pos[w_pos > 0].min(), np.percentile(w_pos, 5))
                vmax2 = np.percentile(w_pos, 99)
                if vmax2 <= vmin2:
                    vmax2 = vmin2 * 10
                norm2 = LogNorm(vmin=vmin2, vmax=vmax2)
                im2 = ax_path.imshow(w_pos, cmap="RdYlGn", aspect="auto", norm=norm2)
                # Cluster names on x-axis
                ref_nms = []
                for mod_nm, _ in cluster_pred_weights:
                    nms = cnames_ref.get(mod_nm)
                    if nms:
                        ref_nms = nms[:max_k2]; break
                if not ref_nms:
                    ref_nms = [str(c) for c in range(max_k2)]
                step2 = max(1, max_k2 // 15)
                shown2 = list(range(0, max_k2, step2))
                ax_path.set_xticks(shown2)
                ax_path.set_xticklabels([ref_nms[c] if c < len(ref_nms) else str(c) for c in shown2],
                                        fontsize=6, rotation=90)
                ax_path.set_yticks(range(len(cluster_pred_weights)))
                ax_path.set_yticklabels([m for m, _ in cluster_pred_weights], fontsize=8)
                # Annotate top 20 cells (by rank) with bold labels; skip tiny values
                flat_idx = np.argsort(w_mat.ravel())[::-1][:20]
                for fi in flat_idx:
                    m_i, c_i = divmod(fi, max_k2)
                    val = w_mat[m_i, c_i]
                    if val < vmin2:
                        continue
                    nm = ref_nms[c_i] if c_i < len(ref_nms) else str(c_i)
                    ax_path.text(c_i, m_i, nm[:8], ha="center", va="center",
                                 fontsize=4.5, color="black", fontweight="bold",
                                 bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.55, lw=0))
                fig.colorbar(im2, ax=ax_path, shrink=0.8,
                             label="Cluster→prediction weight\n(ABMIL·B-cos, log scale)")
        ax_path.set_title("Input cluster → prediction weight\n"
                          "(hypothesis: which clusters drive prediction)", fontsize=7)
        ax_path.set_xlabel("Input patch cluster (pre-computed, named)", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / "H_information_pathway.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(str(out_dir / "H_information_pathway.pdf").replace(".pdf", ".png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── Per-modality seed→cluster affinity figure ──────────────────────────────
    # Each modality has fundamentally different instance types (HE=morphological patches,
    # BAL=cell-type clusters, CT=radiological patterns, Clinical=named features), so
    # they must be shown separately with their own x-axis cluster labels.
    # Rows = seeds (0–15), Cols = clusters for that modality.
    # Color = mean B-cos attention mass: how much seed k attends to cluster c.
    n_mods_present = len([m for m in present_mods_main if aff_pool.get(m)])
    if n_mods_present > 0:
        fig2, axes2 = plt.subplots(1, n_mods_present,
                                   figsize=(max(7, 5 * n_mods_present), 6),
                                   squeeze=False)
        fig2.suptitle(f"H — Seed → cluster affinity (per modality)  |  split{split}_fold{fold}",
                      fontsize=11, fontweight="bold")
        col_idx = 0
        for mod in present_mods_main:
            aff_list = aff_pool.get(mod, [])
            if not aff_list:
                continue
            ax = axes2[0, col_idx]
            max_k_m = max(a.shape[1] for a in aff_list)
            aff_padded = [np.pad(a, ((0, 0), (0, max_k_m - a.shape[1]))) for a in aff_list]
            mean_aff = np.stack(aff_padded).mean(0)   # (K_seeds, K_clusters)
            vmax_m = np.percentile(mean_aff, 95) or 1e-8
            im_m = ax.imshow(mean_aff, cmap="YlOrRd", aspect="auto",
                             vmin=0, vmax=vmax_m)
            # Cluster labels on x-axis
            nms_m = cnames_ref.get(mod, [str(c) for c in range(max_k_m)])
            step_m = max(1, max_k_m // 20)
            shown_m = list(range(0, max_k_m, step_m))
            ax.set_xticks(shown_m)
            ax.set_xticklabels([nms_m[c] if c < len(nms_m) else str(c) for c in shown_m],
                               fontsize=6, rotation=90)
            ax.set_yticks(range(K))
            ax.set_yticklabels([f"s{k:02d}" for k in range(K)], fontsize=7)
            ax.set_xlabel(f"{mod} cluster", fontsize=8)
            ax.set_ylabel("Seed index", fontsize=8)
            ax.set_title(mod, fontsize=10, color=MOD_COLORS.get(mod, "#333"), fontweight="bold")
            fig2.colorbar(im_m, ax=ax, shrink=0.7, label="B-cos mass")
            col_idx += 1
        fig2.tight_layout()
        fig2.savefig(out_dir / "H_seed_cluster_permod.pdf", dpi=150, bbox_inches="tight")
        fig2.savefig(out_dir / "H_seed_cluster_permod.png", dpi=120, bbox_inches="tight")
        plt.close(fig2)

    print("  H done")


# ── Panel I: Per-seed risk stratification (ACR+ vs ACR−) ─────────────────────

def panel_I(results, tasks, out_dir, split, fold):
    """
    Violin plot: for each seed, distribution of mean ABMIL α for ACR+ vs ACR− patients.
    Reveals which seed concepts are systematically higher-attended in rejection cases.
    Only for acr_cls task (binary labels available).
    """
    present_mods_main = [m for m in MOD_ORDER
                         if sum(m in r["present_mods"] for r in results) >= 5]
    K = 16
    seed_labels = []
    seed_colors = []
    mod_boundaries = []
    for mod in present_mods_main:
        mod_boundaries.append(len(seed_labels))
        seed_labels.extend([f"{mod}·s{k:02d}" for k in range(K)])
        seed_colors.extend([MOD_COLORS[mod]] * K)
    n_seeds = len(seed_labels)

    task = "acr_cls"
    if task not in tasks:
        return

    pos_alphas, neg_alphas = [], []
    for r in results:
        lbl = r.get("label")
        a   = r.get("abmil_attn", {}).get(task)
        if a is None or lbl is None:
            continue
        # Canonical per-patient offset: place each modality's seeds at correct position
        pres_ordered = [m for m in MOD_ORDER if m in r.get("present_mods", set())]
        canonical = np.zeros(n_seeds)
        offset = 0
        for mo in pres_ordered:
            if mo in present_mods_main:
                ci = present_mods_main.index(mo) * K
                canonical[ci:ci + K] = a[offset:offset + K]
            offset += K
        if lbl == 1:
            pos_alphas.append(canonical)
        elif lbl == 0:
            neg_alphas.append(canonical)

    if not pos_alphas or not neg_alphas:
        return

    pos_arr = np.stack(pos_alphas)   # (n_pos, n_seeds)
    neg_arr = np.stack(neg_alphas)   # (n_neg, n_seeds)

    # Subtract uniform baseline so Δα is centred on 0: seeds above 0 = overattended,
    # below 0 = underattended relative to a random aggregator (1/N).
    uniform_base = 1.0 / n_seeds if n_seeds > 0 else 0.0
    pos_delta = pos_arr - uniform_base   # (n_pos, n_seeds), can be negative
    neg_delta = neg_arr - uniform_base   # (n_neg, n_seeds)

    # Compute mean difference and t-stat per seed
    from scipy.stats import ttest_ind
    diffs = pos_delta.mean(0) - neg_delta.mean(0)
    pvals = np.array([ttest_ind(pos_delta[:, k], neg_delta[:, k]).pvalue
                      for k in range(n_seeds)])

    # Sort seeds by |diff| descending — show top-16 most discriminative
    top_idx = np.argsort(np.abs(diffs))[::-1][:min(16, n_seeds)]
    top_idx_sorted = sorted(top_idx.tolist(), key=lambda k: -diffs[k])

    fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                             gridspec_kw={"width_ratios": [3, 1], "wspace": 0.35})
    fig.suptitle(
        f"I — Per-Seed ACR Risk Stratification | split{split}_fold{fold}\n"
        f"ACR+ (n={len(pos_alphas)}) vs ACR− (n={len(neg_alphas)}) "
        "· Δα = α − uniform_baseline · Red = higher in ACR+",
        fontsize=10, fontweight="bold")

    # Symmetric y-cap: 97th percentile of |Δα|
    all_delta = np.concatenate([pos_delta[:, top_idx_sorted], neg_delta[:, top_idx_sorted]])
    ymax_cap = np.percentile(np.abs(all_delta), 97) * 1.25

    ax_box = axes[0]
    xs = np.arange(len(top_idx_sorted))
    width = 0.35
    rng = np.random.default_rng(0)
    for xi, k in enumerate(top_idx_sorted):
        col_pos = "#E53935" if diffs[k] > 0 else "#90CAF9"
        col_neg = "#EF9A9A" if diffs[k] > 0 else "#1565C0"
        ax_box.boxplot(pos_delta[:, k], positions=[xi - width / 2], widths=width * 0.8,
                       patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=col_pos, alpha=0.75),
                       medianprops=dict(color="white", linewidth=2),
                       whiskerprops=dict(color=col_pos),
                       capprops=dict(color=col_pos))
        ax_box.boxplot(neg_delta[:, k], positions=[xi + width / 2], widths=width * 0.8,
                       patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor=col_neg, alpha=0.75),
                       medianprops=dict(color="white", linewidth=2),
                       whiskerprops=dict(color=col_neg),
                       capprops=dict(color=col_neg))
        # Strip overlay for ACR+ individual points
        jitter = rng.uniform(-width * 0.3, width * 0.3, size=len(pos_delta))
        ax_box.scatter(xi - width / 2 + jitter, pos_delta[:, k],
                       alpha=0.35, s=8, c=col_pos, zorder=3, linewidths=0)
        # Significance marker
        if pvals[k] < 0.05:
            y_sig = ymax_cap * 0.88
            sig = "***" if pvals[k] < 0.001 else ("**" if pvals[k] < 0.01 else "*")
            ax_box.text(xi, y_sig, sig, ha="center", fontsize=9, color="#333", fontweight="bold")

    ax_box.axhline(0, color="#555", lw=1.0, ls="--")
    ax_box.set_ylim(-ymax_cap, ymax_cap)
    ax_box.set_xticks(xs)
    ax_box.set_xticklabels([seed_labels[k] for k in top_idx_sorted],
                           rotation=55, ha="right", fontsize=8)
    for tick, k in zip(ax_box.get_xticklabels(), top_idx_sorted):
        mod_name = seed_labels[k].split("·")[0]
        tick.set_color(MOD_COLORS.get(mod_name, "#333"))
    ax_box.set_ylabel("Δα  (α − uniform 1/N)", fontsize=10)
    ax_box.set_title("Top-16 discriminative seeds  (dots = ACR+ patients)", fontsize=9)
    handles = [mpatches.Patch(color="#E53935", label="ACR+"),
               mpatches.Patch(color="#1565C0", label="ACR−")]
    ax_box.legend(handles=handles, fontsize=9, framealpha=0.85)
    ax_box.axhline(0, color="grey", lw=0.6, ls=":")

    # Right panel: Δα bar chart (all seeds)
    ax_diff = axes[1]
    all_sorted = np.argsort(diffs)[::-1]
    bar_colors = ["#E53935" if d > 0 else "#1565C0" for d in diffs[all_sorted]]
    ax_diff.barh(range(n_seeds), diffs[all_sorted], color=bar_colors, alpha=0.8, height=0.7)
    ax_diff.set_yticks(range(n_seeds))
    lfs = max(5, min(8, int(110 / n_seeds)))
    ax_diff.set_yticklabels([seed_labels[k] for k in all_sorted], fontsize=lfs)
    for tick, k in zip(ax_diff.get_yticklabels(), all_sorted):
        mod_name = seed_labels[k].split("·")[0]
        tick.set_color(MOD_COLORS.get(mod_name, "#333"))
    ax_diff.axvline(0, color="grey", lw=0.8)
    ax_diff.set_xlabel("Δα  (ACR+ minus ACR−, relative to uniform)", fontsize=9)
    ax_diff.set_title("All seeds sorted by Δα", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_dir / "I_seed_risk_stratification.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(str(out_dir / "I_seed_risk_stratification.pdf").replace(".pdf", ".png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  I done")


# ── Panel J: Seed co-activation correlation matrix ────────────────────────────

def panel_J(results, tasks, out_dir, split, fold):
    """
    Heatmap: Pearson correlation between ABMIL α vectors across all patients.
    Seeds that co-activate together (high positive correlation) likely represent
    the same biological concept; anti-correlated seeds are task-competitive.
    One panel per task.
    """
    present_mods_main = [m for m in MOD_ORDER
                         if sum(m in r["present_mods"] for r in results) >= 5]
    K = 16
    seed_labels = []
    seed_colors = []
    mod_boundaries = []
    for mod in present_mods_main:
        mod_boundaries.append(len(seed_labels))
        seed_labels.extend([f"{mod}·s{k:02d}" for k in range(K)])
        seed_colors.extend([MOD_COLORS[mod]] * K)
    n_seeds = len(seed_labels)

    ntasks = len(tasks)
    cell_sz = max(0.38, min(0.55, 10.0 / n_seeds))
    fig_side = max(5.5, n_seeds * cell_sz + 2.0)
    fig, axes = plt.subplots(1, ntasks, figsize=(fig_side * ntasks, fig_side))
    if ntasks == 1:
        axes = [axes]
    fig.suptitle(
        f"J — Seed Co-Activation Correlation | split{split}_fold{fold}\n"
        "Pearson r of ABMIL α across patients · seeds reordered by hierarchical clustering"
        "  ·  dark red = co-activate · dark blue = competitive",
        fontsize=10, fontweight="bold")

    tick_fs = max(5, min(9, int(110 / n_seeds)))

    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        _have_scipy_hier = True
    except ImportError:
        _have_scipy_hier = False

    for ax, task in zip(axes, tasks):
        alpha_mat = []
        for r in results:
            a = r.get("abmil_attn", {}).get(task)
            if a is None:
                continue
            # Canonical per-patient offset alignment
            pres_ordered = [m for m in MOD_ORDER if m in r.get("present_mods", set())]
            canonical = np.zeros(n_seeds)
            offset = 0
            for mo in pres_ordered:
                if mo in present_mods_main:
                    ci = present_mods_main.index(mo) * K
                    canonical[ci:ci + K] = a[offset:offset + K]
                offset += K
            alpha_mat.append(canonical)
        if len(alpha_mat) < 5:
            ax.set_visible(False)
            continue
        A = np.stack(alpha_mat)        # (N, n_seeds)
        corr = np.corrcoef(A.T)        # (n_seeds, n_seeds)

        # Hierarchical clustering to group co-activated seeds together
        if _have_scipy_hier and n_seeds > 2:
            dist = np.clip(1.0 - corr, 0, 2.0)
            dist = (dist + dist.T) / 2   # enforce exact symmetry (float rounding)
            np.fill_diagonal(dist, 0)
            link = linkage(squareform(dist), method="average")
            order = leaves_list(link)
        else:
            order = np.arange(n_seeds)

        corr_ord = corr[np.ix_(order, order)]
        labels_ord = [seed_labels[i] for i in order]
        colors_ord = [seed_colors[i] for i in order]

        im = ax.imshow(corr_ord, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal",
                       interpolation="nearest")

        ax.set_xticks(range(n_seeds))
        ax.set_xticklabels(labels_ord, fontsize=max(7, tick_fs), rotation=90)
        ax.set_yticks(range(n_seeds))
        ax.set_yticklabels(labels_ord, fontsize=max(7, tick_fs))
        for tick, col in zip(ax.get_xticklabels(), colors_ord):
            tick.set_color(col)
        for tick, col in zip(ax.get_yticklabels(), colors_ord):
            tick.set_color(col)

        # Draw dividers where modality changes in the reordered space and annotate block
        mods_ord = [seed_labels[i].split("·")[0] for i in order]
        prev_mod = None
        block_starts = {}
        for pos, mod in enumerate(mods_ord):
            if mod != prev_mod:
                if prev_mod is not None:
                    ax.axhline(pos - 0.5, color="white", lw=2.0, zorder=5)
                    ax.axvline(pos - 0.5, color="white", lw=2.0, zorder=5)
                block_starts[mod] = pos
                prev_mod = mod
        # Modality labels on diagonal blocks
        block_list = list(block_starts.items())
        for bi, (mod, start) in enumerate(block_list):
            end = block_list[bi + 1][1] if bi + 1 < len(block_list) else n_seeds
            mid = (start + end) / 2 - 0.5
            ax.text(mid, mid, mod, ha="center", va="center", fontsize=8,
                    color="white", fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc=MOD_COLORS[mod],
                              alpha=0.75, lw=0))

        task_color = TASK_COLORS.get(task, "#333")
        TASK_FULL_J = {
            "acr_cls": "ACR Classification", "acr_surv": "ACR Survival",
            "clad": "CLAD Survival", "death": "Death Survival",
            "clad_surv": "CLAD Survival", "death_surv": "Death Survival",
        }
        ax.set_title(f"{TASK_FULL_J.get(task, task)}\n(N={len(alpha_mat)} patients)",
                     fontsize=9, color=task_color, fontweight="bold")

        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.85)
        cb.set_label("Pearson r", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    plt.tight_layout()
    fig.savefig(out_dir / "J_seed_coactivation_corr.pdf", dpi=150, bbox_inches="tight")
    fig.savefig(str(out_dir / "J_seed_coactivation_corr.pdf").replace(".pdf", ".png"),
                dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  J done")


# ── Panel K: Seed attribution chain — instances → seeds → prediction ──────────

def panel_K(results, tasks, out_dir, split, fold):
    """
    For each task, answers:
      1. Which seeds are up-weighted by ABMIL in high-risk vs low-risk patients?
      2. What biological clusters do those seeds attend to?

    Layout per task (2-row figure):
      Row 0: Seed ABMIL attention — left=low-risk, right=high-risk, bar=difference
      Row 1: Per-modality cluster affinity for top differentially active seeds
    """
    present_mods_main = [m for m in MOD_ORDER
                         if sum(m in r["present_mods"] for r in results) >= 5]
    K = 16
    total_seeds = len(present_mods_main) * K

    # Canonical ABMIL attention vector per patient per task
    def _canonical_alpha(r, task):
        a = r.get("abmil_attn", {}).get(task)
        if a is None:
            return None
        pres_ordered = [mo for mo in MOD_ORDER if mo in r.get("present_mods", set())]
        can = np.zeros(total_seeds, dtype=np.float32)
        off = 0
        for mo in pres_ordered:
            if mo in present_mods_main:
                ci = present_mods_main.index(mo) * K
                can[ci:ci + K] = a[off:off + K]
            off += K
        return can

    # Pre-compute seed→cluster affinity per patient per modality (K, k_clus)
    def _get_seed_cluster(r, mod):
        h    = r["inst_reps"].get(mod)
        bcos = r.get("pma_bcos", {}).get(mod)
        if bcos is None:
            bcos = r.get("pma_attn", {}).get(mod)
        c_ids = r.get("cluster_ids", {}).get(mod)
        if h is None or bcos is None or bcos.ndim != 2 or bcos.shape[1] != len(h):
            return None, None
        if c_ids is not None and len(c_ids) == len(h):
            cl = c_ids
            k_clus = int(cl.max()) + 1 if len(cl) > 0 else K_PATCH
        else:
            from sklearn.cluster import MiniBatchKMeans
            km = MiniBatchKMeans(n_clusters=K_PATCH, n_init=3, random_state=42, batch_size=4096)
            cl = km.fit_predict(h.astype(np.float32))
            k_clus = K_PATCH
        aff = _seed_cluster_mass(bcos, cl, k_clus)   # (K, k_clus) raw totals
        # row-normalize so each seed's profile sums to 1 (relative cluster preference)
        row_sums = aff.sum(axis=1, keepdims=True).clip(1e-8, None)
        return aff / row_sums, r.get("cluster_names", {}).get(mod, [])

    ep_keys = {
        "acr_cls":    ("event_acr",   "tte_acr",   "label"),
        "acr_surv":   ("event_acr",   "tte_acr",   "logits"),
        "clad_surv":  ("event_clad",  "tte_clad",  "logits"),
        "death_surv": ("event_death", "tte_death", "logits"),
    }

    for task in tasks:
        # ── Collect canonical alpha + outcome ────────────────────────────────
        alphas, outcomes, valid_results = [], [], []
        for r in results:
            a = _canonical_alpha(r, task)
            if a is None:
                continue
            _, _, outcome_src = ep_keys.get(task, ("", "", "logits"))
            if outcome_src == "label":
                out_val = float(r["label"]) if r["label"] is not None else float("nan")
            else:
                out_val = r["logits"].get(task, float("nan"))
            if np.isnan(out_val):
                continue
            alphas.append(a)
            outcomes.append(out_val)
            valid_results.append(r)

        if len(alphas) < 10:
            continue
        alphas   = np.stack(alphas)          # (N, total_seeds)
        outcomes = np.array(outcomes)

        # Split into high-risk vs low-risk
        if task == "acr_cls":
            hi_mask = outcomes == 1
            lo_mask = outcomes == 0
            hi_label, lo_label = "ACR+", "ACR−"
        else:
            med = np.median(outcomes)
            hi_mask = outcomes >= med
            lo_mask = outcomes < med
            hi_label, lo_label = "High risk", "Low risk"

        if hi_mask.sum() < 3 or lo_mask.sum() < 3:
            continue

        alpha_hi   = alphas[hi_mask].mean(0)     # (total_seeds,)
        alpha_lo   = alphas[lo_mask].mean(0)
        alpha_diff = alpha_hi - alpha_lo          # positive → enriched in high-risk

        # ── Non-collapsed seed mask (high PMA entropy = attends broadly) ────
        nc_mask = _noncollapsed_seed_mask(valid_results, present_mods_main, K, keep_pct=50)
        n_nc = int(nc_mask.sum())

        # ── Build seed labels and modality spans ────────────────────────────
        seed_labels, seed_colors = [], []
        mod_spans = {}   # mod → (start_idx, end_idx)
        for mod in present_mods_main:
            s = len(seed_labels)
            for k in range(K):
                seed_labels.append(f"{mod[:3]}{k}")
                seed_colors.append(MOD_COLORS[mod])
            mod_spans[mod] = (s, s + K)

        # ── Figure layout: 1 row seed bar + 1 row per-mod cluster affinity ──
        n_mods = len(present_mods_main)
        fig = plt.figure(figsize=(max(14, total_seeds * 0.22), 4 + 3.5 * n_mods))
        gs_outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45,
                                     height_ratios=[2, n_mods * 3.5])

        # ── Row 0: seed ABMIL attention by outcome group ─────────────────────
        gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[0],
                                                  width_ratios=[2, 2, 1], wspace=0.08)
        ax_lo   = fig.add_subplot(gs_top[0])
        ax_hi   = fig.add_subplot(gs_top[1])
        ax_diff = fig.add_subplot(gs_top[2])

        x = np.arange(total_seeds)
        for ax, vals, title in [(ax_lo, alpha_lo, f"{lo_label} (n={lo_mask.sum()})"),
                                 (ax_hi, alpha_hi, f"{hi_label} (n={hi_mask.sum()})")]:
            _bar_with_collapse_mask(ax, x, vals, seed_colors, nc_mask)
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.set_xticks([]); ax.set_ylabel("Mean ABMIL α", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            # modality dividers
            for mod in present_mods_main[1:]:
                ax.axvline(mod_spans[mod][0] - 0.5, color="#aaa", lw=0.7, ls="--")
            for mod in present_mods_main:
                mid = (mod_spans[mod][0] + mod_spans[mod][1]) / 2
                ax.text(mid, ax.get_ylim()[1] * 0.9, mod, ha="center", fontsize=7,
                        color=MOD_COLORS[mod], fontweight="bold")

        # Difference bar (signed) — collapsed seeds greyed
        diff_colors = ["#E53935" if v > 0 else "#1E88E5" for v in alpha_diff]
        nc_idx  = np.where(nc_mask)[0]
        col_idx = np.where(~nc_mask)[0]
        if len(nc_idx):
            ax_diff.bar(nc_idx, alpha_diff[nc_idx],
                        color=[diff_colors[i] for i in nc_idx], width=0.85, alpha=0.85)
        if len(col_idx):
            ax_diff.bar(col_idx, alpha_diff[col_idx], color="#d0d0d0", width=0.85, alpha=0.25)
        ax_diff.axhline(0, color="#333", lw=0.8)
        ax_diff.set_title(f"Δα ({hi_label}−{lo_label})  [{n_nc}/{total_seeds} non-collapsed]",
                          fontsize=9, fontweight="bold")
        ax_diff.set_xticks([]); ax_diff.set_ylabel("Δ Mean α", fontsize=8)
        ax_diff.spines[["top", "right"]].set_visible(False)

        split_lbl = "all_splits" if split < 0 else f"split{split}_fold{fold}"
        fig.suptitle(f"K — Seed attribution chain: {task} | {split_lbl}  (N={len(alphas)})",
                     fontsize=11, fontweight="bold")

        # ── Row 1: per-modality cluster affinity for top differential seeds ──
        gs_bot = gridspec.GridSpecFromSubplotSpec(n_mods, 2, subplot_spec=gs_outer[1],
                                                  hspace=0.5, wspace=0.35)

        for mi, mod in enumerate(present_mods_main):
            s0, s1 = mod_spans[mod]
            mod_diff = alpha_diff[s0:s1]    # (K,) difference for this modality's seeds
            top_k_idx = np.argsort(np.abs(mod_diff))[::-1][:5]  # top-5 by |diff|

            # Collect seed→cluster affinities across patients, weighted by outcome group
            aff_hi_list, aff_lo_list, cnames = [], [], []
            for r, hi in zip(valid_results, hi_mask):
                aff, nm = _get_seed_cluster(r, mod)
                if aff is None:
                    continue
                if not cnames and nm:
                    cnames = nm
                # Weight each seed's affinity row by its ABMIL alpha for this patient
                ca = _canonical_alpha(r, task)
                if ca is None:
                    continue
                seed_weights = ca[s0:s1]   # (K,) ABMIL attn for this mod's seeds
                # weighted affinity: alpha_k * aff_k,c for each seed k
                weighted = seed_weights[:, None] * aff    # (K, C)
                if hi:
                    aff_hi_list.append(weighted)
                else:
                    aff_lo_list.append(weighted)

            if not aff_hi_list and not aff_lo_list:
                continue

            # Per-modality: show mean weighted affinity for top-5 differential seeds
            for col, (aff_list, group_lbl) in enumerate([
                    (aff_lo_list, lo_label), (aff_hi_list, hi_label)]):
                ax = fig.add_subplot(gs_bot[mi, col])
                if not aff_list:
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                            transform=ax.transAxes, fontsize=8)
                    continue
                mean_aff = np.stack(aff_list).mean(0)    # (K, C)
                top_aff  = mean_aff[top_k_idx]            # (5, C) top differential seeds
                n_clus   = top_aff.shape[1]
                clus_nms = (cnames[:n_clus] if cnames else
                            [str(c) for c in range(n_clus)])
                # Truncate long cluster names
                clus_nms = [cn[:18] for cn in clus_nms]

                im = ax.imshow(top_aff, aspect="auto", cmap="YlOrRd",
                               vmin=0, vmax=top_aff.max().clip(1e-8))
                ax.set_xticks(range(n_clus))
                ax.set_xticklabels(clus_nms, rotation=45, ha="right", fontsize=5.5)
                ax.set_yticks(range(len(top_k_idx)))
                ax.set_yticklabels([f"seed{top_k_idx[j]} (Δ={mod_diff[top_k_idx[j]]:+.3f})"
                                    for j in range(len(top_k_idx))], fontsize=6)
                ax.set_title(f"{mod} — {group_lbl} | top-5 Δseeds",
                             fontsize=8, color=MOD_COLORS[mod], fontweight="bold")
                plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02).ax.tick_params(labelsize=6)

        fig.savefig(out_dir / f"K_seed_attribution_{task}.pdf", dpi=150, bbox_inches="tight")
        fig.savefig(out_dir / f"K_seed_attribution_{task}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    print("  K done")


def panel_K_multisplit(results, tasks, out_dir):
    """
    One figure per task with 5 rows (one per split).
    Each row: [low-risk mean α | high-risk mean α | Δα diff] bars colored by modality.
    results must have r["_split"] tag (added by extract_all_splits).
    """
    splits_grouped = {}
    for r in results:
        sp = r.get("_split", -1)
        splits_grouped.setdefault(sp, []).append(r)
    split_ids = sorted(splits_grouped.keys())

    present_mods_main = [m for m in MOD_ORDER
                         if sum(m in r["present_mods"] for r in results) >= 5]
    K = 16
    total_seeds = len(present_mods_main) * K

    def _canonical_alpha(r, task):
        a = r.get("abmil_attn", {}).get(task)
        if a is None:
            return None
        pres_ordered = [mo for mo in MOD_ORDER if mo in r.get("present_mods", set())]
        can = np.zeros(total_seeds, dtype=np.float32)
        off = 0
        for mo in pres_ordered:
            if mo in present_mods_main:
                ci = present_mods_main.index(mo) * K
                can[ci:ci + K] = a[off:off + K]
            off += K
        return can

    seed_colors = []
    mod_spans = {}
    for mod in present_mods_main:
        s = len(seed_colors)
        seed_colors.extend([MOD_COLORS[mod]] * K)
        mod_spans[mod] = (s, s + K)

    ep_src = {
        "acr_cls":    "label",
        "acr_surv":   "logits",
        "clad_surv":  "logits",
        "death_surv": "logits",
    }

    for task in tasks:
        n_splits = len(split_ids)
        fig, axes = plt.subplots(n_splits, 3,
                                 figsize=(max(14, total_seeds * 0.25), 2.8 * n_splits),
                                 squeeze=False)
        fig.suptitle(f"K (per-split) — Seed ABMIL attribution: {task}",
                     fontsize=12, fontweight="bold")

        for row_i, sp in enumerate(split_ids):
            sp_results = splits_grouped[sp]
            alphas, outcomes = [], []
            for r in sp_results:
                a = _canonical_alpha(r, task)
                if a is None:
                    continue
                src = ep_src.get(task, "logits")
                if src == "label":
                    out_val = float(r["label"]) if r["label"] is not None else float("nan")
                else:
                    out_val = r["logits"].get(task, float("nan"))
                if np.isnan(out_val):
                    continue
                alphas.append(a)
                outcomes.append(out_val)

            ax_lo, ax_hi, ax_diff = axes[row_i]

            if len(alphas) < 6:
                for ax in (ax_lo, ax_hi, ax_diff):
                    ax.text(0.5, 0.5, f"split{sp}: n={len(alphas)} (too few)",
                            ha="center", va="center", transform=ax.transAxes, fontsize=8)
                    ax.axis("off")
                continue

            alphas_arr   = np.stack(alphas)
            outcomes_arr = np.array(outcomes)

            if task == "acr_cls":
                hi_mask = outcomes_arr == 1
                lo_mask = outcomes_arr == 0
                hi_label, lo_label = "ACR+", "ACR−"
            else:
                med = np.median(outcomes_arr)
                hi_mask = outcomes_arr >= med
                lo_mask = outcomes_arr < med
                hi_label, lo_label = "High risk", "Low risk"

            if hi_mask.sum() < 3 or lo_mask.sum() < 3:
                for ax in (ax_lo, ax_hi, ax_diff):
                    ax.text(0.5, 0.5, f"split{sp}: too few per group",
                            ha="center", va="center", transform=ax.transAxes, fontsize=8)
                    ax.axis("off")
                continue

            alpha_hi   = alphas_arr[hi_mask].mean(0)
            alpha_lo   = alphas_arr[lo_mask].mean(0)
            alpha_diff = alpha_hi - alpha_lo

            # Non-collapsed mask computed per split
            nc_mask = _noncollapsed_seed_mask(sp_results, present_mods_main, K, keep_pct=50)
            n_nc    = int(nc_mask.sum())
            x = np.arange(total_seeds)

            for ax, vals, lbl, n in [
                    (ax_lo,  alpha_lo, lo_label, lo_mask.sum()),
                    (ax_hi,  alpha_hi, hi_label, hi_mask.sum())]:
                _bar_with_collapse_mask(ax, x, vals, seed_colors, nc_mask)
                ax.set_title(f"split{sp} — {lbl} (n={n})", fontsize=8, fontweight="bold")
                ax.set_xticks([])
                ax.set_ylabel("Mean α", fontsize=7)
                ax.spines[["top", "right"]].set_visible(False)
                for mod in present_mods_main[1:]:
                    ax.axvline(mod_spans[mod][0] - 0.5, color="#aaa", lw=0.6, ls="--")
                y_top = vals[nc_mask].max() * 1.15 if nc_mask.any() and vals[nc_mask].max() > 0 else 0.01
                for mod in present_mods_main:
                    mid = (mod_spans[mod][0] + mod_spans[mod][1]) / 2
                    ax.text(mid, y_top * 0.88, mod[:3], ha="center", fontsize=6,
                            color=MOD_COLORS[mod], fontweight="bold")
                ax.set_ylim(0, y_top)

            diff_colors = ["#E53935" if v > 0 else "#1E88E5" for v in alpha_diff]
            nc_idx  = np.where(nc_mask)[0]
            col_idx = np.where(~nc_mask)[0]
            if len(nc_idx):
                ax_diff.bar(nc_idx, alpha_diff[nc_idx],
                            color=[diff_colors[i] for i in nc_idx], width=0.85, alpha=0.85)
            if len(col_idx):
                ax_diff.bar(col_idx, alpha_diff[col_idx], color="#d0d0d0", width=0.85, alpha=0.25)
            ax_diff.axhline(0, color="#333", lw=0.8)
            ax_diff.set_title(f"split{sp} — Δα [{n_nc}/{total_seeds} non-collapsed]",
                              fontsize=8, fontweight="bold")
            ax_diff.set_xticks([])
            ax_diff.set_ylabel("Δ Mean α", fontsize=7)
            ax_diff.spines[["top", "right"]].set_visible(False)
            for mod in present_mods_main[1:]:
                ax_diff.axvline(mod_spans[mod][0] - 0.5, color="#aaa", lw=0.6, ls="--")
            dy_abs = np.abs(alpha_diff[nc_mask]).max() if nc_mask.any() else 0.01
            ax_diff.set_ylim(-dy_abs * 1.3, dy_abs * 1.3)

        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(out_dir / f"K_multisplit_seed_attribution_{task}.pdf",
                    dpi=150, bbox_inches="tight")
        fig.savefig(out_dir / f"K_multisplit_seed_attribution_{task}.png",
                    dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  K_multisplit: {task} done")
    print("  K_multisplit done")


# ── Unified patient summary (SetMIL-MT + Longitudinal) ───────────────────────

def compute_cohort_percentiles(results, tasks):
    """
    Pool logits across all results and return percentile rank [0,1] per stem per task.
    Works for both SetMIL-MT (one record per biopsy) and longitudinal (one per patient).
    """
    from scipy.stats import rankdata
    pct = {}  # stem -> {task -> float in [0,1]}
    for task in tasks:
        logits_all = [(r["stem"], r["logits"].get(task)) for r in results
                      if r["logits"].get(task) is not None]
        if not logits_all:
            continue
        stems_t = [s for s, _ in logits_all]
        vals_t  = np.array([v for _, v in logits_all], dtype=float)
        ranks   = (rankdata(vals_t) - 1) / max(len(vals_t) - 1, 1)  # [0,1]
        for stem, rank in zip(stems_t, ranks):
            pct.setdefault(stem, {})[task] = float(rank)
    return pct


def _flag_discordance(r, tasks, pct_ranks, top_pct=0.10):
    """
    Return list of (task, flag_msg) for strong model-label discordances.
    Classification: sigmoid(logit) vs binary label.
    Survival: percentile rank vs event/TTE.
    """
    EP_KEYS = {
        "acr_cls":    ("event_acr",  "tte_acr"),
        "acr_surv":   ("event_acr",  "tte_acr"),
        "clad_surv":  ("event_clad", "tte_clad"),
        "death_surv": ("event_death","tte_death"),
    }
    flags = []
    stem  = r["stem"]
    for task in tasks:
        logit = r["logits"].get(task)
        if logit is None:
            continue
        rank = pct_ranks.get(stem, {}).get(task)
        ev_key, tte_key = EP_KEYS.get(task, ("event_acr","tte_acr"))
        if task == "acr_cls":
            prob  = 1.0 / (1.0 + np.exp(-logit))
            label = r.get("label")
            if label == 0 and prob > (1 - top_pct):
                flags.append((task, f"P(ACR+)={prob:.2f} but label=0"))
            elif label == 1 and prob < top_pct:
                flags.append((task, f"P(ACR+)={prob:.2f} but label=1"))
        else:
            if rank is None:
                continue
            ev  = r.get(ev_key)
            tte = r.get(tte_key, float("nan"))
            # High predicted risk but no event (very early censoring)
            if rank > (1 - top_pct) and ev == 0 and not np.isnan(tte) and tte < 180:
                flags.append((task, f"rank={rank:.2f} (top {top_pct*100:.0f}%) "
                                    f"but censored at {tte:.0f}d"))
            # Low predicted risk but early event
            elif rank < top_pct and ev == 1 and not np.isnan(tte) and tte < 180:
                flags.append((task, f"rank={rank:.2f} (bot {top_pct*100:.0f}%) "
                                    f"but event at {tte:.0f}d"))
    return flags


def plot_patient_summary_unified(r, tasks, out_dir, pct_ranks,
                                 model_type="set_mil_mt",
                                 all_results=None):
    """
    One-page patient timeline / summary — saved as L0_summary_{stem}.png.

    Covers both SetMIL-MT (biopsy-level records) and LongitudinalMIL (time-series).
    Called by generate_patient_summaries() for every test-set record.

    Purpose
    -------
    These plots let a clinician or biologist audit the model's output for a
    specific patient.  They are especially useful for:
    - Catching mis-annotations: a high-risk score with label=0 (or vice versa).
    - Tracking risk trajectory: does the model detect rising risk before an event?
    - Understanding modality coverage: which data types were available per visit?

    Layout (2×2 grid)
    -----------------
    Top-left — Modality availability over time
        Each row = one modality (HE, BAL, CT, Clinical).  Each column = one visit
        (x-axis = days from transplant).  Filled square = modality present; × = absent.
        Vertical dashed lines: red = ACR+ visit, green = ACR− visit.
        Useful for spotting whether missing modalities explain anomalous scores.

    Top-right — Normalized risk scores (this visit)
        Horizontal bars, one per task, all mapped to [0, 1]:
          acr_cls:  sigmoid P(ACR+) — probability of acute rejection.
          survival: percentile rank of Cox logit within the full test cohort.
                    Rank=1 = highest predicted hazard; rank=0 = lowest.
        ⚠ icon marks tasks flagged as discordant (see mis-annotation logic below).

    Bottom-left — Score trajectory over visits
        For SetMIL-MT: if the patient has multiple biopsy records (multiple
        anchor_dt), the normalized score for each visit is plotted as a line
        coloured by task.  Single-visit patients show "no trajectory" text.
        For LongitudinalMIL: uses hazard_traj stored in the record.

    Bottom-right — TTE / Event context + mis-annotation flags
        Text block showing time-to-event and event status for each endpoint.
        Flags are raised when the model's output is statistically discordant:
          acr_cls:  P(ACR+) > 0.75 AND true label = 0 (false positive), or
                    P(ACR+) < 0.25 AND true label = 1 (false negative).
          survival: percentile rank > 0.90 AND censored within 365 days
                    (high predicted risk but patient was censored early — could
                    indicate a real but unobserved event, or a model error).
        Flagged patients are also written to flagged_patients.csv.

    Aggregation
    -----------
    For SetMIL-MT, all_results is the full pooled list.  The function groups
    records by patient_id to reconstruct the within-patient visit sequence.
    anchor_dt (biopsy date) is used to order visits chronologically.

    Percentile ranks
    ----------------
    pct_ranks is computed once per cohort by compute_cohort_percentiles() before
    any per-patient call.  Ranks are stable across patients (no data leakage).
    """
    EP_KEYS = {
        "acr_cls":   ("event_acr",  "tte_acr"),
        "acr_surv":  ("event_acr",  "tte_acr"),
        "clad_surv": ("event_clad", "tte_clad"),
        "death_surv":("event_death","tte_death"),
    }
    TASK_FULL = {
        "acr_cls":    "ACR classif.", "acr_surv":  "ACR survival",
        "clad_surv":  "CLAD survival","death_surv":"Death survival",
    }
    TASK_COLORS_L = {
        "acr_cls":    "#E53935", "acr_surv":  "#FB8C00",
        "clad_surv":  "#8E24AA", "death_surv":"#1E88E5",
    }

    stem = r["stem"]
    pid  = r.get("patient_id", stem)

    # ── Build trajectory: group same patient's visits across all_results ──────
    if model_type == "set_mil_mt" and all_results is not None:
        visits = sorted(
            [x for x in all_results if x.get("patient_id", x["stem"]) == pid],
            key=lambda x: x.get("anchor_dt", x["stem"])
        )
    elif model_type == "longitudinal":
        # r already has biopsy_days / hazard_traj
        visits = None
    else:
        visits = [r]

    # Days from first visit (for SetMIL-MT)
    if visits and len(visits) > 1:
        t0 = visits[0].get("anchor_dt")
        if t0 is not None:
            days_arr = np.array([
                float((v.get("anchor_dt", t0) - t0).days) if v.get("anchor_dt") else float(i)
                for i, v in enumerate(visits)
            ])
        else:
            days_arr = np.arange(len(visits), dtype=float)
    elif model_type == "longitudinal":
        days_arr = np.array(r.get("biopsy_days", [0.0]), dtype=float)
    else:
        days_arr = np.array([0.0])

    flags = _flag_discordance(r, tasks, pct_ranks)

    fig = plt.figure(figsize=(14, 7))
    ax00 = fig.add_subplot(2, 2, 1)   # modality timeline
    ax01 = fig.add_subplot(2, 2, 2)   # normalized scores
    ax10 = fig.add_subplot(2, 2, 3)   # score trajectory
    ax11 = fig.add_subplot(2, 2, 4)   # TTE/event context + flags

    model_lbl = "LongMIL-MT" if model_type == "longitudinal" else "SetMIL-MT"
    title_str = f"{model_lbl} | Patient {pid} ({stem})  —  {len(days_arr)} visit(s)"
    if flags:
        title_str += "  ⚠ DISCORDANCE"
    fig.suptitle(title_str, fontsize=12, fontweight="bold",
                 color="#B71C1C" if flags else "black")

    # ── TL: Modality availability ─────────────────────────────────────────────
    visit_list = visits if visits else [r]
    for vi, v in enumerate(visit_list):
        pres = v.get("present_mods", set())
        day  = days_arr[vi] if vi < len(days_arr) else float(vi)
        for mi, mo in enumerate(MOD_ORDER):
            has = mo in pres
            ax00.scatter(day, mi,
                         color=MOD_COLORS[mo] if has else "#ccc",
                         s=80 if has else 35,
                         marker="s" if has else "x",
                         edgecolors="white" if has else "none",
                         linewidths=0.5, zorder=3 if has else 2)
    ax00.set_yticks(range(len(MOD_ORDER)))
    ax00.set_yticklabels(MOD_ORDER, fontsize=10)
    for mi, mo in enumerate(MOD_ORDER):
        ax00.get_yticklabels()[mi].set_color(MOD_COLORS[mo])
    ax00.set_xlabel("Days from transplant", fontsize=9)
    ax00.set_title("Modality availability  (■=present, ×=absent)", fontsize=10)
    # ACR label vlines
    for vi, v in enumerate(visit_list):
        lbl = v.get("label")
        day = days_arr[vi] if vi < len(days_arr) else float(vi)
        if lbl is not None:
            ax00.axvline(day, color="#E53935" if lbl else "#43A047",
                         lw=1.2, ls="--", alpha=0.5, zorder=0)

    # CLAD and death event vlines (solid, drawn once at absolute event day)
    _event_annots = []  # list of (abs_day, color, label)
    for _ev_key, _tte_key, _color, _label in [
        ("event_clad",  "tte_clad",  "#8E24AA", "CLAD event"),
        ("event_death", "tte_death", "#00897B", "Death event"),
    ]:
        for vi, v in enumerate(visit_list):
            ev  = v.get(_ev_key)
            tte = v.get(_tte_key, float("nan"))
            if ev == 1 and not (isinstance(tte, float) and np.isnan(tte)):
                vday = days_arr[vi] if vi < len(days_arr) else float(vi)
                abs_day = vday + float(tte)
                _event_annots.append((abs_day, _color, _label))
                break  # first occurrence per event type
    _ytop_00 = len(MOD_ORDER) - 1 + 0.4  # top of modality scatter axis
    for abs_day, _color, _label in _event_annots:
        ax00.axvline(abs_day, color=_color, lw=2.0, ls="-", alpha=0.85, zorder=1)
        ax00.text(abs_day - 2, _ytop_00, _label, color=_color, fontsize=6.5,
                  rotation=90, va="top", ha="right", clip_on=True)

    # ── TR: Normalized scores (this record) ──────────────────────────────────
    score_vals, score_lbls, score_cols = [], [], []
    for task in tasks:
        logit = r["logits"].get(task)
        if logit is None:
            continue
        if task == "acr_cls":
            score = 1.0 / (1.0 + np.exp(-logit))
            lbl   = f"{TASK_FULL.get(task, task)}\nP(ACR+)={score:.2f}"
        else:
            rank  = pct_ranks.get(stem, {}).get(task, float("nan"))
            score = rank
            lbl   = f"{TASK_FULL.get(task, task)}\nPctile={score:.2f}"
        score_vals.append(score)
        score_lbls.append(lbl)
        score_cols.append(TASK_COLORS_L.get(task, "#777"))

    if score_vals:
        ys   = np.arange(len(score_vals))
        bars = ax01.barh(ys, score_vals, color=score_cols,
                         edgecolor="white", linewidth=0.8, height=0.55)
        ax01.axvline(0.5, color="#888", lw=1.0, ls=":")
        ax01.set_xlim(0, 1)
        for bar, val in zip(bars, score_vals):
            ax01.text(min(val + 0.03, 0.97), bar.get_y() + bar.get_height() / 2,
                      f"{val:.2f}", va="center", ha="left", fontsize=9, fontweight="bold")
        ax01.set_yticks(ys)
        ax01.set_yticklabels(score_lbls, fontsize=8)
        for tick, col in zip(ax01.get_yticklabels(), score_cols):
            tick.set_color(col)
        ax01.set_xlabel("Score (0=low risk, 1=high risk)", fontsize=9)
        ax01.set_title("Normalized risk scores (this visit)", fontsize=10)
        ax01.invert_yaxis()
        # Flag discordant tasks with ⚠
        flag_tasks = {ft for ft, _ in flags}
        for yi, task in enumerate([t for t in tasks if r["logits"].get(t) is not None]):
            if task in flag_tasks:
                ax01.text(0.02, yi, "⚠", fontsize=12, color="#B71C1C",
                          va="center", transform=ax01.get_yaxis_transform())

    # ── BL: Score trajectory across visits ───────────────────────────────────
    if model_type == "longitudinal":
        for task in tasks:
            hvals = r.get("hazard_traj", {}).get(task, [])
            if not hvals:
                continue
            n = min(len(hvals), len(days_arr))
            ax10.plot(days_arr[:n], hvals[:n], "o-",
                      color=TASK_COLORS_L.get(task, "#777"),
                      lw=1.8, ms=5, label=TASK_FULL.get(task, task))
        ax10.axhline(0, color="#888", lw=0.6, ls=":")
        ax10.set_ylabel("Cox log-hazard (causal)", fontsize=9)
    elif visits and len(visits) > 1:
        for task in tasks:
            traj = []
            for vi, v in enumerate(visits):
                logit = v["logits"].get(task)
                if logit is None:
                    traj.append(float("nan"))
                    continue
                if task == "acr_cls":
                    traj.append(1.0 / (1.0 + np.exp(-logit)))
                else:
                    rank = pct_ranks.get(v["stem"], {}).get(task, float("nan"))
                    traj.append(rank)
            traj = np.array(traj)
            if not np.all(np.isnan(traj)):
                ax10.plot(days_arr[:len(traj)], traj, "o-",
                          color=TASK_COLORS_L.get(task, "#777"),
                          lw=1.8, ms=5, label=TASK_FULL.get(task, task))
        ax10.set_ylim(0, 1)
        ax10.axhline(0.5, color="#888", lw=0.6, ls=":")
        ax10.set_ylabel("Normalized score (0-1)", fontsize=9)
    else:
        ax10.text(0.5, 0.5, "Single visit — no trajectory",
                  ha="center", va="center", transform=ax10.transAxes, fontsize=10, color="#888")

    # CLAD / death event vlines on trajectory panel
    for abs_day, _color, _label in _event_annots:
        ax10.axvline(abs_day, color=_color, lw=2.0, ls="-", alpha=0.85, zorder=1,
                     label=_label)

    ax10.set_xlabel("Days from transplant", fontsize=9)
    ax10.set_title("Score trajectory over visits", fontsize=10)
    ax10.legend(fontsize=7.5, framealpha=0.9)
    ax10.spines[["top", "right"]].set_visible(False)

    # ── BR: TTE/event context + mis-annotation flags ─────────────────────────
    ax11.axis("off")
    lines = ["TTE / Event context", "─" * 32]
    for task in tasks:
        ev_key, tte_key = EP_KEYS.get(task, ("event_acr", "tte_acr"))
        ev  = r.get(ev_key)
        tte = r.get(tte_key, float("nan"))
        rank = pct_ranks.get(stem, {}).get(task, float("nan"))
        ev_str  = "event" if ev == 1 else ("censored" if ev == 0 else "?")
        tte_str = f"{tte:.0f}d" if not np.isnan(tte) else "?"
        rank_str = f"{rank:.2f}" if not np.isnan(rank) else "?"
        lines.append(f"{TASK_FULL.get(task, task)}: {ev_str} @ {tte_str}  "
                     f"[rank={rank_str}]")

    if flags:
        lines += ["", "⚠ POTENTIAL MIS-ANNOTATION / OUTLIER", "─" * 32]
        for ft, fmsg in flags:
            lines.append(f"• {TASK_FULL.get(ft, ft)}: {fmsg}")

    ax11.text(0.05, 0.95, "\n".join(lines),
              transform=ax11.transAxes, fontsize=8.5,
              va="top", ha="left", family="monospace",
              color="#B71C1C" if flags else "#222",
              bbox=dict(boxstyle="round,pad=0.4", fc="#fff8f8" if flags else "#f9f9f9",
                        ec="#E53935" if flags else "#ccc", lw=1.2))

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem_safe = stem.replace("/", "_")
    png = out_dir / f"L0_summary_{stem_safe}.png"
    fig.savefig(png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return png


def _inject_anchor_dt(results):
    """
    Backfill anchor_dt and patient_id from the splits CSV into npy records that
    were saved before anchor_dt was added to build_splits_multitask (older npys
    will have anchor_dt=None for every record, breaking trajectory plots).
    """
    import pandas as pd
    try:
        df = pd.read_csv(SPLITS_CSV)
        df["anchor_dt"] = pd.to_datetime(df["anchor_dt"])
        df["stem"] = df["file"].apply(lambda x: Path(str(x)).stem)
        stem_to_adt = dict(zip(df["stem"], df["anchor_dt"]))
        stem_to_pid = dict(zip(df["stem"], df["patient_id"].astype(str)))
        n_fixed = 0
        for r in results:
            if r.get("anchor_dt") is None and r["stem"] in stem_to_adt:
                r["anchor_dt"] = stem_to_adt[r["stem"]]
                n_fixed += 1
            if r.get("patient_id") in (None, r["stem"]) and r["stem"] in stem_to_pid:
                r["patient_id"] = stem_to_pid[r["stem"]]
        print(f"  [inject_anchor_dt] fixed {n_fixed}/{len(results)} records")
    except Exception as e:
        print(f"  [inject_anchor_dt] skipped: {e}")


def generate_patient_summaries(results, tasks, out_dir, model_type="set_mil_mt"):
    """
    Compute cohort percentiles from pooled results, then generate one L0_summary
    per record. Uploads a wandb Table of flagged patients.
    """
    print(f"  [patient_summaries] computing cohort percentiles (N={len(results)}) ...")

    # Backfill anchor_dt from CSV if npy was saved before the fix
    _inject_anchor_dt(results)

    pct_ranks = compute_cohort_percentiles(results, tasks)

    # Group by patient_id so trajectory is available inside each call
    # (pass all_results so plot_patient_summary_unified can group)
    flagged = []
    pngs    = []
    for i, r in enumerate(results):
        try:
            png = plot_patient_summary_unified(
                r, tasks, out_dir / "patient_summaries",
                pct_ranks, model_type=model_type, all_results=results)
            pngs.append(png)
            flags = _flag_discordance(r, tasks, pct_ranks)
            if flags:
                flagged.append({"stem": r["stem"],
                                "patient_id": r.get("patient_id", r["stem"]),
                                "flags": "; ".join(f"{t}: {m}" for t, m in flags)})
        except Exception as e:
            print(f"  [warn] patient summary failed for {r['stem']}: {e}")
        if (i + 1) % 50 == 0:
            print(f"  [patient_summaries] {i+1}/{len(results)} done  "
                  f"({len(flagged)} flagged)")

    # Save flagged patients CSV
    import csv
    csv_path = out_dir / "flagged_patients.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem","patient_id","flags"])
        w.writeheader(); w.writerows(flagged)
    print(f"  [patient_summaries] {len(flagged)} flagged → {csv_path}")

    print(f"  [patient_summaries] done: {len(pngs)} plots saved to {out_dir/'patient_summaries'}")

    # wandb upload is deferred to _setmil_log_wandb() where the run is active.

    return pngs, flagged


# ── Paper JSON: structured cluster-affinity data for publication figures ───────

def save_paper_json(results, tasks, out_dir):
    """
    Compute and save structured interpretability data for paper figures.

    For each task: seed Δα per modality, cluster affinity scores (hi/lo/delta),
    and gate weights.  Saved to out_dir/paper_interp_data.json.
    """
    import json as _json

    present_mods = [m for m in MOD_ORDER
                    if sum(m in r["present_mods"] for r in results) >= 5]
    K = 16
    total_seeds = len(present_mods) * K

    def _get_aff(r, mod):
        """Return (aff_norm (K, C), cluster_names) for modality mod in result r."""
        h    = r["inst_reps"].get(mod)
        bcos = r.get("pma_bcos", {}).get(mod)
        if bcos is None:
            bcos = r.get("pma_attn", {}).get(mod)
        c_ids = r.get("cluster_ids", {}).get(mod)
        if h is None or bcos is None or bcos.ndim != 2 or bcos.shape[1] != len(h):
            return None, None
        if c_ids is not None and len(c_ids) == len(h):
            cl = c_ids
            k_clus = int(cl.max()) + 1 if len(cl) > 0 else K_PATCH
        else:
            from sklearn.cluster import MiniBatchKMeans
            km = MiniBatchKMeans(n_clusters=K_PATCH, n_init=3, random_state=42, batch_size=4096)
            cl = km.fit_predict(h.astype(np.float32))
            k_clus = K_PATCH
        aff = _seed_cluster_mass(bcos, cl, k_clus)
        row_sums = aff.sum(axis=1, keepdims=True).clip(1e-8, None)
        return aff / row_sums, r.get("cluster_names", {}).get(mod, [])

    ep_src = {"acr_cls": "label", "acr_surv": "logits",
               "clad_surv": "logits", "death_surv": "logits"}
    # Short key aliases used in some saved npy files
    _task_key = {"clad_surv": ["clad_surv", "clad"],
                 "death_surv": ["death_surv", "death"]}

    def _get_val(r, task):
        """Read logit/label for task, trying aliases."""
        src = ep_src.get(task, "logits")
        if src == "label":
            return float(r["label"]) if r["label"] is not None else float("nan")
        for k in _task_key.get(task, [task]):
            v = r.get("logits", {}).get(k)
            if v is not None:
                return float(v)
        return float("nan")

    def _can_alpha(r, task):
        for k in _task_key.get(task, [task]):
            a = r.get("abmil_attn", {}).get(k)
            if a is not None:
                break
        else:
            return None
        pres = [mo for mo in MOD_ORDER if mo in r.get("present_mods", set())]
        can = np.zeros(total_seeds, dtype=np.float32)
        off = 0
        for mo in pres:
            if mo in present_mods:
                ci = present_mods.index(mo) * K
                can[ci:ci + K] = a[off:off + K]
            off += K
        return can

    out = {"modalities": present_mods, "n_patients": len(results), "tasks": {}}

    for task in tasks:
        alphas, outcomes, vres = [], [], []
        for r in results:
            a = _can_alpha(r, task)
            if a is None:
                continue
            val = _get_val(r, task)
            if np.isnan(val):
                continue
            alphas.append(a); outcomes.append(val); vres.append(r)

        if len(alphas) < 10:
            continue

        alphas_arr   = np.stack(alphas)
        outcomes_arr = np.array(outcomes)

        if task == "acr_cls":
            hi_mask = outcomes_arr == 1
            lo_mask = outcomes_arr == 0
        else:
            med = np.median(outcomes_arr)
            hi_mask = outcomes_arr >= med
            lo_mask = outcomes_arr < med

        if hi_mask.sum() < 3 or lo_mask.sum() < 3:
            continue

        alpha_hi   = alphas_arr[hi_mask].mean(0)
        alpha_lo   = alphas_arr[lo_mask].mean(0)
        alpha_diff = alpha_hi - alpha_lo   # (total_seeds,)

        tdata = {
            "n_patients": len(alphas),
            "n_hi": int(hi_mask.sum()), "n_lo": int(lo_mask.sum()),
            "seed_delta_alpha": {}, "cluster_affinity": {}, "gate_weights": {},
        }

        # Seed Δα per modality
        for mod in present_mods:
            s0 = present_mods.index(mod) * K
            tdata["seed_delta_alpha"][mod] = alpha_diff[s0:s0 + K].tolist()

        # Cluster affinity per modality, weighted by Δα
        for mod in present_mods:
            s0 = present_mods.index(mod) * K
            mod_diff = alpha_diff[s0:s0 + K]   # (K,)

            aff_hi_list, aff_lo_list, cnames = [], [], []
            for r, hi in zip(vres, hi_mask):
                aff, nm = _get_aff(r, mod)
                if aff is None:
                    continue
                if not cnames and nm:
                    cnames = nm
                ca = _can_alpha(r, task)
                if ca is None:
                    continue
                sw = ca[s0:s0 + K]                    # (K,) ABMIL weights
                weighted = sw[:, None] * aff           # (K, C)
                if hi:
                    aff_hi_list.append(weighted)
                else:
                    aff_lo_list.append(weighted)

            if not (aff_hi_list or aff_lo_list):
                continue

            # Pad to uniform n_clus (patients may have fewer clusters if some cell types absent)
            n_clus = max(a.shape[1] for a in aff_hi_list + aff_lo_list)
            def _pad(lst):
                out = []
                for a in lst:
                    if a.shape[1] < n_clus:
                        a = np.pad(a, ((0,0),(0, n_clus - a.shape[1])))
                    out.append(a)
                return out
            aff_hi_list = _pad(aff_hi_list)
            aff_lo_list = _pad(aff_lo_list)
            mean_hi = np.stack(aff_hi_list).mean(0) if aff_hi_list else np.zeros((K, n_clus))
            mean_lo = np.stack(aff_lo_list).mean(0) if aff_lo_list else np.zeros((K, n_clus))

            # Aggregate: Σ_k pos(Δα[k]) * aff_hi[k,c] and Σ_k pos(-Δα[k]) * aff_lo[k,c]
            hi_score = (np.maximum(mod_diff, 0)[:, None] * mean_hi).sum(0)   # (C,)
            lo_score = (np.maximum(-mod_diff, 0)[:, None] * mean_lo).sum(0)  # (C,)
            delta    = hi_score - lo_score

            tdata["cluster_affinity"][mod] = {
                "cluster_names": (cnames[:n_clus] if cnames else [str(i) for i in range(n_clus)]),
                "hi_score": hi_score.tolist(),
                "lo_score": lo_score.tolist(),
                "delta":    delta.tolist(),
            }

        # Gate weights per modality (try aliases for clad/death)
        gate_by_mod = {mo: [] for mo in present_mods}
        for r in vres:
            gv = None
            for k in _task_key.get(task, [task]):
                gv = r.get("gate_vals", {}).get(k)
                if gv is not None:
                    break
            if gv is None:
                continue
            for mi, mo in enumerate(MOD_ORDER):
                if mo in gate_by_mod and mi < len(gv):
                    gate_by_mod[mo].append(float(gv[mi]))
        tdata["gate_weights"] = {
            "mean": {mo: float(np.mean(v)) if v else 0.0 for mo, v in gate_by_mod.items()},
            "std":  {mo: float(np.std(v))  if v else 0.0 for mo, v in gate_by_mod.items()},
        }

        out["tasks"][task] = tdata

    out_path = out_dir / "paper_interp_data.json"
    with open(out_path, "w") as f:
        _json.dump(out, f, indent=2)
    print(f"  [paper_json] {len(out['tasks'])} tasks → {out_path}")


# ── Top-level ─────────────────────────────────────────────────────────────────

def plot_all(results, seeds_init, seeds_init_q, tasks, split, fold, out_dir,
             variant="mega", panels=None):
    """
    panels: set of panel letters to run, e.g. {"K"} or {"A","B","K"}.
            None (default) runs all panels.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    run = panels if panels is not None else set("ABCDEFGIJK")
    all_splits_mode = (split < 0)
    print(f"[plot] {len(results)} patients → {out_dir}  panels={sorted(run)}")
    if "A" in run:
        panel_A(results, out_dir, split, fold, metric="euclidean")
        panel_A(results, out_dir, split, fold, metric="cosine")
    if "B" in run:
        panel_B(results, seeds_init, seeds_init_q, out_dir, split, fold)
    if "C" in run:
        panel_C(results, tasks, out_dir, split, fold)
    if "D" in run:
        panel_D(results, tasks, out_dir, split, fold)
    if "E" in run:
        panel_E(results, tasks, out_dir, split, fold)
    if "F" in run:
        panel_F(results, tasks, out_dir, split, fold, variant=variant)
    if "G" in run:
        panel_G(results, tasks, out_dir, split, fold)
    if "H" in run:
        panel_H(results, tasks, out_dir, split, fold)
    if "I" in run:
        panel_I(results, tasks, out_dir, split, fold)
    if "J" in run:
        panel_J(results, tasks, out_dir, split, fold)
    if "K" in run:
        if all_splits_mode:
            panel_K_multisplit(results, tasks, out_dir)
        else:
            panel_K(results, tasks, out_dir, split, fold)

    # Always save structured JSON for paper figures
    save_paper_json(results, tasks, out_dir)


def _setmil_log_wandb(results, tasks, split, fold, variant, out_dir, project):
    """
    Comprehensive W&B logging for set_mil_mt interpretability.

    Logs:
      • Per-patient table: label, TTE, hazard per task, present modalities,
        ABMIL α entropy (uniformity), gate values per task
      • Mean gate matrix as scalars and image
      • ABMIL seed-importance distribution per task (box plot as table)
      • All PNGs organised by panel (A–H), not a single flat list
      • Model-level summary scalars
    """
    try:
        import wandb
    except ImportError:
        print("  [wandb] wandb not installed — skipping"); return

    fold_tag = f"split{split}_fold{fold}"

    config = {
        "variant": variant, "split": split, "fold": fold, "tasks": tasks,
        "n_samples": len(results),
    }

    try:
        run = wandb.init(
            project=project,
            name=f"set_mil_mt_{variant}_{fold_tag}",
            group="set_mil_mt",
            config=config,
            reinit=True,
        )
    except Exception as e:
        print(f"  [wandb] init failed: {e}"); return

    scalar_log = {"n_samples": len(results)}

    # ── Per-patient table ─────────────────────────────────────────────────────
    table_cols = (["patient_idx", "label", "present_modalities"] +
                  [f"logit_{t}" for t in tasks] +
                  [f"abmil_entropy_{t}" for t in tasks] +
                  [f"gate_{t}_{m}" for t in tasks for m in MOD_ORDER])
    patient_table = wandb.Table(columns=table_cols)

    # ABMIL distribution tables per task
    abmil_dist_tables = {t: wandb.Table(columns=["patient_idx", "seed_idx", "alpha"]) for t in tasks}
    gate_box_tables   = {t: wandb.Table(columns=["modality", "gate_weight"]) for t in tasks}

    for pi, r in enumerate(results):
        logits_row    = [r.get("logits", {}).get(t, float("nan")) for t in tasks]
        present_mods  = "+".join(r.get("present_mods", []))

        # ABMIL α entropy (low = focused on few seeds, high = uniform)
        entropies = []
        for t in tasks:
            alpha = r.get("abmil_attn", {}).get(t)
            if alpha is not None and len(alpha) > 0:
                a = np.clip(alpha, 1e-9, 1.0)
                h = float(-np.sum(a * np.log(a)))
                entropies.append(h)
                for si, av in enumerate(alpha):
                    abmil_dist_tables[t].add_data(pi, si, float(av))
            else:
                entropies.append(float("nan"))

        # Gate values per task per modality
        gate_flat = []
        for t in tasks:
            gv = r.get("gate_vals", {}).get(t, np.zeros(len(MOD_ORDER)))
            if gv is None:
                gv = np.zeros(len(MOD_ORDER))
            for mi, mo in enumerate(MOD_ORDER):
                w = float(gv[mi]) if mi < len(gv) else float("nan")
                gate_flat.append(w)
                gate_box_tables[t].add_data(mo, w)

        label_val = r.get("label")
        patient_table.add_data(
            pi, label_val, present_mods,
            *logits_row, *entropies, *gate_flat)

    scalar_log["patients/summary"] = patient_table
    for t in tasks:
        scalar_log[f"abmil_distribution/{t}"] = abmil_dist_tables[t]
        scalar_log[f"gate_distribution/{t}"]  = gate_box_tables[t]

    # ── Mean gate scalars ──────────────────────────────────────────────────────
    for t in tasks:
        gate_rows = [r.get("gate_vals", {}).get(t) for r in results
                     if r.get("gate_vals", {}).get(t) is not None]
        if gate_rows:
            mean_gate = np.stack(gate_rows).mean(0)
            for mi, mo in enumerate(MOD_ORDER):
                if mi < len(mean_gate):
                    scalar_log[f"mean_gate/{t}/{mo}"] = float(mean_gate[mi])

    # ── Modal contribution mean ────────────────────────────────────────────────
    for t in tasks:
        mc_key = f"modal_contrib_{t}"
        contribs = [r.get(mc_key, {}) for r in results if r.get(mc_key)]
        if contribs:
            all_mods_mc = set(k for d in contribs for k in d)
            for mod in all_mods_mc:
                vals = [d[mod] for d in contribs if mod in d]
                scalar_log[f"modal_contrib/{t}/{mod}"] = float(np.mean(vals))

    wandb.log(scalar_log)

    # ── Panel PNGs: organised by panel letter ─────────────────────────────────
    panel_groups = {
        "A_instance_reps": "A: Instance Embeddings (UMAP)",
        "B_seeds":         "B: PMA Seeds (init vs post)",
        "C_sab":           "C: SAB Cross-Modal Attention",
        "D_abmil":         "D: ABMIL Seed Importance",
        "E_task":          "E: Task–Modal Gate",
        "F_modality":      "F: Modality Combo Ablation",
        "G_final":         "G: Final Rep Hexbin",
        "H_information":   "H: Cluster→Prediction Pathway",
        "I_seed_risk":     "I: Seed Risk Stratification",
        "J_seed_coact":    "J: Seed Co-activation Corr",
    }

    img_log = {}
    for prefix, label in panel_groups.items():
        imgs = [
            wandb.Image(str(p), caption=f"{label} | {p.stem}")
            for p in sorted(out_dir.glob(f"{prefix}*.png"))
        ]
        if imgs:
            img_log[f"panels/{prefix}"] = imgs

    # Remaining PNGs not matched above
    matched = set()
    for prefix in panel_groups:
        matched.update(out_dir.glob(f"{prefix}*.png"))
    other = [wandb.Image(str(p), caption=p.stem)
             for p in sorted(out_dir.glob("*.png")) if p not in matched]
    if other:
        img_log["panels/other"] = other

    if img_log:
        wandb.log(img_log)
        total = sum(len(v) for v in img_log.values())
        print(f"  [wandb] uploaded {total} PNGs across {len(img_log)} panel groups")

    # ── Patient summary upload (if generate_patient_summaries was run earlier) ──
    # generate_patient_summaries() runs before wandb is initialised, so the upload
    # is deferred here where the run is active.
    ps_dir = out_dir / "patient_summaries"
    csv_path_ps = out_dir / "flagged_patients.csv"
    if ps_dir.exists() and any(ps_dir.glob("L0_summary_*.png")):
        all_ps_pngs = sorted(ps_dir.glob("L0_summary_*.png"))
        # Upload all PNGs as a versioned artifact
        art = wandb.Artifact("patient_summaries", type="interpretability",
                             description="Per-patient L0_summary plots (SetMIL-MT, all test splits)")
        art.add_dir(str(ps_dir), name="summaries")
        if csv_path_ps.exists():
            art.add_file(str(csv_path_ps), name="flagged_patients.csv")
        wandb.log_artifact(art)
        print(f"  [wandb] artifact 'patient_summaries' logged ({len(all_ps_pngs)} PNGs)")

        # Upload flagged patient images as a browseable media panel (capped at 200)
        if csv_path_ps.exists():
            import csv as _csv
            with open(csv_path_ps) as _f:
                flagged_stems = {row["stem"] for row in _csv.DictReader(_f)}
            flagged_pngs = [p for p in all_ps_pngs
                            if p.stem.replace("L0_summary_", "") in flagged_stems]
            if flagged_pngs:
                wandb.log({"patient_summaries/flagged_images": [
                    wandb.Image(str(p), caption=p.stem) for p in flagged_pngs[:200]
                ]})
                print(f"  [wandb] logged {min(len(flagged_pngs), 200)} flagged summary images")
            # Also upload as a table
            tbl = wandb.Table(columns=["stem", "patient_id", "flags"])
            with open(csv_path_ps) as _f:
                for row in _csv.DictReader(_f):
                    tbl.add_data(row["stem"], row.get("patient_id", row["stem"]), row["flags"])
            wandb.log({"patient_summaries/flagged_table": tbl})
            print(f"  [wandb] logged flagged_table ({len(flagged_stems)} rows)")

    run.finish()
    print(f"  [wandb] run: {run.url}")


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--split",       type=int, default=0)
    pa.add_argument("--fold",        type=int, default=1)
    pa.add_argument("--variant",     default="mega",
                    choices=["mega","cls","acr_surv","clad_surv","death_surv"])
    pa.add_argument("--max-samples", type=int, default=None)
    pa.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    pa.add_argument("--out-dir",       default=None)
    pa.add_argument("--wandb-project", default="chicago-mil-interpretability",
                    help="W&B project name (set to 'none' to skip W&B logging)")
    pa.add_argument("--panels", default=None,
                    help="Comma-separated panel letters to run, e.g. 'K' or 'A,B,K'. "
                         "Default: all panels.")
    pa.add_argument("--all-splits", action="store_true",
                    help="Pool test patients from all 5 splits (each from their fold-0 model). "
                         "Panels B and F are skipped (model-specific). "
                         "Output dir uses 'all_splits' tag.")
    pa.add_argument("--json-only", action="store_true",
                    help="Load existing results_raw.npy and save paper_interp_data.json. "
                         "No model loading / GPU needed. Use after a full run has finished.")
    pa.add_argument("--patient-summaries", action="store_true",
                    help="Generate per-patient L0_summary plots from cached results_raw.npy. "
                         "Computes cohort percentile ranks across all pooled test patients, "
                         "flags statistical discordances, and saves flagged_patients.csv. "
                         "Requires --all-splits or --json-only to load pooled results.")
    pa.add_argument("--merge-task-dirs", nargs="+", default=None,
                    help="Merge per-task results_raw.npy files from multiple single-task "
                         "all_splits dirs into one combined dataset (keyed by stem). "
                         "E.g.: --merge-task-dirs .../all_splits_cls .../all_splits_acr_surv "
                         ".../all_splits_clad_surv .../all_splits_death_surv "
                         "Use with --out-dir to specify where to save merged results and panels.")
    args = pa.parse_args()

    device  = torch.device(args.device)

    # ── Merge-task-dirs mode: combine single-task npys by stem ───────────────
    if args.merge_task_dirs:
        _alias = {"clad": "clad_surv", "death": "death_surv"}
        # Fields to merge per-task (task-keyed dicts)
        MERGE_FIELDS = ["logits", "abmil_attn", "abmil_raw_logits",
                        "gate_vals", "final_reps", "sab_attn"]
        merged = {}   # stem -> record
        tasks_set = set()
        for d in args.merge_task_dirs:
            npy = Path(d) / "results_raw.npy"
            if not npy.exists():
                print(f"  [merge] skipping {d} — no results_raw.npy"); continue
            print(f"  [merge] loading {npy} ...")
            recs = list(np.load(npy, allow_pickle=True))
            print(f"  [merge] {len(recs)} records from {d}")
            for r in recs:
                stem = r["stem"]
                if stem not in merged:
                    merged[stem] = dict(r)
                    for f in MERGE_FIELDS:
                        if f in merged[stem] and not isinstance(merged[stem][f], dict):
                            merged[stem][f] = {}
                # Merge task-keyed fields
                for f in MERGE_FIELDS:
                    src = r.get(f, {})
                    if isinstance(src, dict):
                        for k, v in src.items():
                            k_norm = _alias.get(k, k)
                            merged[stem].setdefault(f, {})[k_norm] = v
                            tasks_set.add(k_norm)
        results = list(merged.values())
        tasks = [t for t in ["acr_cls","acr_surv","clad_surv","death_surv"] if t in tasks_set]
        out_dir = Path(args.out_dir) if args.out_dir \
                  else OUT_ROOT / "all_splits_merged"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [merge] {len(results)} unique stems, tasks={tasks} → {out_dir}")
        np.save(out_dir / "results_raw.npy", np.array(results, dtype=object), allow_pickle=True)
        save_paper_json(results, tasks, out_dir)
        if args.panels:
            panels_set = set(p.strip().upper() for p in args.panels.split(",")) - {"B", "F"}
            if panels_set:
                plot_all(results, None, None, tasks, -1, 0, out_dir,
                         variant="merged", panels=panels_set)
        if args.patient_summaries:
            generate_patient_summaries(results, tasks, out_dir, model_type="set_mil_mt")
        if args.wandb_project.lower() != "none":
            _setmil_log_wandb(results, tasks, -1, 0, "merged", out_dir, args.wandb_project)
        return

    # ── JSON-only mode: load cached results and write paper JSON ──────────────
    if args.json_only:
        out_dir = Path(args.out_dir) if args.out_dir \
                  else OUT_ROOT / f"all_splits_{args.variant}"
        npy_path = out_dir / "results_raw.npy"
        if not npy_path.exists():
            raise FileNotFoundError(f"No results_raw.npy at {npy_path}. Run without --json-only first.")
        print(f"[json-only] loading {npy_path} ...")
        results = list(np.load(npy_path, allow_pickle=True))
        # Infer tasks; handle short aliases ('clad'→'clad_surv', 'death'→'death_surv')
        _alias = {"clad": "clad_surv", "death": "death_surv"}
        tasks_set = set()
        for r in results:
            for k in list(r.get("logits", {}).keys()) + list(r.get("abmil_attn", {}).keys()):
                tasks_set.add(_alias.get(k, k))
        tasks = [t for t in ["acr_cls","acr_surv","clad_surv","death_surv"] if t in tasks_set]
        print(f"[json-only] {len(results)} patients, tasks={tasks}")
        out_dir.mkdir(parents=True, exist_ok=True)
        save_paper_json(results, tasks, out_dir)
        # Also run any requested panels (B skipped — needs seeds_init from model)
        if args.panels:
            panels_set = set(p.strip().upper() for p in args.panels.split(",")) - {"B", "F"}
            if panels_set:
                plot_all(results, None, None, tasks, -1, 0, out_dir,
                         variant=args.variant, panels=panels_set)
        if args.patient_summaries:
            generate_patient_summaries(results, tasks, out_dir, model_type="set_mil_mt")
        if args.wandb_project.lower() != "none":
            _setmil_log_wandb(results, tasks, -1, 0, args.variant, out_dir, args.wandb_project)
        return

    if args.all_splits:
        out_dir = Path(args.out_dir) if args.out_dir \
                  else OUT_ROOT / f"all_splits_{args.variant}"
        results, seeds_init, seeds_init_q, tasks = extract_all_splits(
            args.variant, device, args.max_samples)
        # B and F are model-specific — exclude from all-splits run
        default_panels = set("ACDEGIJK") - {"B", "F"}
        panels_set = (set(p.strip().upper() for p in args.panels.split(","))
                      if args.panels else default_panels)
        split_tag, fold_tag = -1, 0
    else:
        out_dir = Path(args.out_dir) if args.out_dir \
                  else OUT_ROOT / f"split{args.split}_fold{args.fold}_{args.variant}"
        panels_set = (set(p.strip().upper() for p in args.panels.split(","))
                      if args.panels else None)
        results, seeds_init, seeds_init_q, tasks = extract_all(
            args.split, args.fold, args.variant, device, args.max_samples)
        split_tag, fold_tag = args.split, args.fold

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "results_raw.npy", np.array(results, dtype=object), allow_pickle=True)
    plot_all(results, seeds_init, seeds_init_q, tasks, split_tag, fold_tag, out_dir,
             variant=args.variant, panels=panels_set)

    if args.patient_summaries:
        generate_patient_summaries(results, tasks, out_dir, model_type="set_mil_mt")

    # ── W&B logging ───────────────────────────────────────────────────────────
    if args.wandb_project.lower() != "none":
        _setmil_log_wandb(results, tasks, args.split, args.fold,
                          args.variant, out_dir, args.wandb_project)


if __name__ == "__main__":
    main()
