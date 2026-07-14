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
  A_instance_reps.pdf          UMAP of patch embeddings per modality (coloured by ACR label)
  B_seeds.pdf                  Init seeds (star) vs post-PMA seeds per modality
  C_sab_crossmodal_attn.pdf    SAB cross-modal attention heatmap (mean over patients, per task)
  D_abmil_seed_importance.pdf  ABMIL alpha per seed per task
  E_task_modal_gate.pdf        TaskModalGate weight matrix (task x modality)
  F_modality_combo_ablation.pdf BACC/CI under each modality subset
  G_final_rep_hexbin_{task}.pdf UMAP + hexbin: label/score/TTE/event density

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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mil.models.builders import build_model_v8
from mil.data.registry import MODALITIES, _pres_col
from mil.data.loader import preload_bags
from mil.data.splits import build_splits_multitask

SPLITS_CSV   = "/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
SAMPLES_DIR  = "/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples/"
RESULTS_ROOT = ROOT / "results" / "mm_abmil_v8" / "phase2"
OUT_ROOT     = ROOT / "interpretability" / "set_mil_mt_interp"

CMAP_HAZARD  = "RdBu_r"
CMAP_TTE     = "RdBu"
_EMPTY_COLOR = "#FFE57F"
MOD_COLORS   = {"HE": "#58a6ff", "BAL": "#3fb950", "CT": "#d4a017", "Clinical": "#d2a8ff"}
TASK_COLORS  = {"acr_cls": "#e53935", "acr_surv": "#ff7043",
                "clad_surv": "#7e57c2", "death_surv": "#26a69a"}
MOD_ORDER    = ["HE", "BAL", "CT", "Clinical"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _umap_embed(X, n_neighbors=30, min_dist=0.2, seed=42):
    from umap import UMAP
    return UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                random_state=seed, n_jobs=4).fit_transform(X.astype(np.float32))


def _uniform_lim(axes, xy):
    xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    px, py = (xmax - xmin) * 0.05, (ymax - ymin) * 0.05
    for ax in axes:
        ax.set_xlim(xmin - px, xmax + px)
        ax.set_ylim(ymin - py, ymax + py)


def _hex_panel(ax, xy, values, cmap, vmin, vmax, gridsize=35, title=""):
    im = ax.hexbin(xy[:, 0], xy[:, 1], C=values,
                   gridsize=gridsize, cmap=cmap, vmin=vmin, vmax=vmax,
                   reduce_C_function=np.nanmean, mincnt=1,
                   linewidths=0, edgecolors="none")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    return im


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

    # ABMIL alpha from hooked raw logits
    abmil_attn = {t: torch.softmax(raw, dim=0).squeeze(-1).numpy()
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
    seeds_post = {}
    pma_attn   = {}
    present_mods = []
    for mod, enc in m.encoders.items():
        t = bags.get(mod)
        if t is None:
            continue
        t = t.to(device)
        if t.shape[0] > m.max_he_patches:
            idx = torch.randperm(t.shape[0], device=device)[:m.max_he_patches]
            t   = t[idx]
        crds = bags.get("HE_coords") if mod == "HE" else None
        h    = enc.encode_patches(t, coords=crds)
        inst_reps[mod] = h.cpu().numpy()

        s, aw = m.pma[mod](h, return_attn=True)
        mod_idx = torch.tensor(m._mod_idx[mod], device=device)
        seeds_post[mod] = (s + m.modal_embed(mod_idx)).cpu().numpy()
        pma_attn[mod]   = aw.cpu().numpy()
        present_mods.append(mod)

    return {
        "inst_reps":    inst_reps,
        "seeds_post":   seeds_post,
        "pma_attn":     pma_attn,
        "sab_attn":     sab_attn,
        "abmil_attn":   abmil_attn,
        "gate_vals":    gate_vals,
        "final_reps":   final_reps,
        "logits":       logits_out,
        "present_mods": present_mods,
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

        out["stem"]        = rec["stem"]
        out["label"]       = rec.get("label")
        out["event_acr"]   = rec.get("event_next_acr", float("nan"))
        out["tte_acr"]     = rec.get("tte_next_acr", float("nan"))
        out["event_clad"]  = rec.get("clad_event", float("nan"))
        out["tte_clad"]    = rec.get("clad_time", float("nan"))
        out["event_death"] = rec.get("death_event", float("nan"))
        out["tte_death"]   = rec.get("death_time", float("nan"))
        results.append(out)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(recs)} done")

    seeds_init = {mod: model.pma[mod].seeds.detach().cpu().numpy()
                  for mod in model.encoders}
    return results, seeds_init, tasks


# ── Panel A: instance rep UMAPs ───────────────────────────────────────────────

def panel_A(results, out_dir, split, fold):
    rng = np.random.default_rng(42)
    inst_pool = {mod: [] for mod in MOD_ORDER}
    inst_labs = {mod: [] for mod in MOD_ORDER}
    for res in results:
        lab = res["label"]
        for mod, h in res["inst_reps"].items():
            n   = min(len(h), 150)
            idx = rng.choice(len(h), n, replace=False)
            inst_pool[mod].append(h[idx])
            inst_labs[mod].extend([lab] * n)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(f"A — Instance reps (post-ModalFFNEncoder) | split{split}_fold{fold}",
                 fontsize=11, fontweight="bold")
    for ax, mod in zip(axes, MOD_ORDER):
        pool = np.concatenate(inst_pool[mod]) if inst_pool[mod] else None
        if pool is None or len(pool) < 20:
            ax.text(0.5, 0.5, f"{mod}\n(no data)", ha="center", va="center",
                    transform=ax.transAxes); continue
        xy   = _umap_embed(pool)
        labs = np.array([(1 if l == 1 else (0 if l == 0 else -1))
                          for l in inst_labs[mod]])
        ax.scatter(xy[labs == 0, 0], xy[labs == 0, 1], s=1.5, c="#1E88E5", alpha=0.4, label="ACR-")
        ax.scatter(xy[labs == 1, 0], xy[labs == 1, 1], s=1.5, c="#E53935", alpha=0.5, label="ACR+")
        if (labs == -1).any():
            ax.scatter(xy[labs == -1, 0], xy[labs == -1, 1], s=1.5, c="#bbb", alpha=0.3)
        ax.set_title(mod, fontsize=10, color=MOD_COLORS[mod], fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(markerscale=4, fontsize=8, framealpha=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "A_instance_reps.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  A done")


# ── Panel B: seeds init vs post-PMA ──────────────────────────────────────────

def panel_B(results, seeds_init, out_dir, split, fold):
    seeds_post_pool = {mod: [] for mod in MOD_ORDER}
    for res in results:
        for mod, s in res["seeds_post"].items():
            seeds_post_pool[mod].append(s)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(f"B — Seed vectors: learned (init ★) vs patient post-PMA | split{split}_fold{fold}",
                 fontsize=11, fontweight="bold")
    for ax, mod in zip(axes, MOD_ORDER):
        post_pool = seeds_post_pool.get(mod, [])
        init      = seeds_init.get(mod)
        if not post_pool or init is None:
            ax.text(0.5, 0.5, f"{mod}\n(no data)", ha="center", va="center",
                    transform=ax.transAxes); continue
        K    = init.shape[0]
        post = np.concatenate(post_pool)
        xy   = _umap_embed(np.vstack([post, init]), n_neighbors=15, min_dist=0.1)
        xy_post, xy_init = xy[:-K], xy[-K:]
        seed_ids = np.tile(np.arange(K), len(post_pool))
        ax.scatter(xy_post[:, 0], xy_post[:, 1], c=seed_ids, s=2, alpha=0.25,
                   cmap="tab20", vmin=0, vmax=K - 1)
        ax.scatter(xy_init[:, 0], xy_init[:, 1], c=np.arange(K), s=80,
                   edgecolors="black", linewidths=0.9, zorder=5,
                   cmap="tab20", vmin=0, vmax=K - 1, marker="*", label="Init seed")
        ax.set_title(mod, fontsize=10, color=MOD_COLORS[mod], fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(markerscale=1.5, fontsize=8, framealpha=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "B_seeds.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  B done")


# ── Panel C: SAB cross-modal attention ───────────────────────────────────────

def panel_C(results, tasks, out_dir, split, fold):
    present_mods_main = [m for m in MOD_ORDER
                          if sum(m in r["present_mods"] for r in results) > 0.5 * len(results)]
    K = 16
    expected_T = len(present_mods_main) * K
    mod_boundaries = [i * K for i in range(len(present_mods_main))]

    ntasks = len(tasks)
    fig, axes = plt.subplots(1, ntasks, figsize=(5.5 * ntasks, 5))
    if ntasks == 1:
        axes = [axes]
    fig.suptitle(f"C — SAB cross-modal attention (mean over patients) | split{split}_fold{fold}",
                 fontsize=11, fontweight="bold")

    for ax, task in zip(axes, tasks):
        mats = [r["sab_attn"][task] for r in results
                if task in r["sab_attn"] and r["sab_attn"][task].shape[0] >= expected_T]
        if not mats:
            ax.set_visible(False); continue
        tm = np.stack([m[:expected_T, :expected_T] for m in mats]).mean(0)
        im = ax.imshow(tm, cmap="viridis", aspect="auto", vmin=0)
        for b in mod_boundaries[1:]:
            ax.axvline(b - 0.5, color="white", lw=1.5)
            ax.axhline(b - 0.5, color="white", lw=1.5)
        ticks = [b + K // 2 for b in mod_boundaries]
        ax.set_xticks(ticks); ax.set_xticklabels(present_mods_main, fontsize=9)
        ax.set_yticks(ticks); ax.set_yticklabels(present_mods_main, fontsize=9)
        ax.set_title(task, fontsize=9, color=TASK_COLORS.get(task, "#333"))
        fig.colorbar(im, ax=ax, shrink=0.8, label="Attn")

    fig.tight_layout()
    fig.savefig(out_dir / "C_sab_crossmodal_attn.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  C done")


# ── Panel D: ABMIL seed importances ──────────────────────────────────────────

def panel_D(results, tasks, out_dir, split, fold):
    present_mods_main = [m for m in MOD_ORDER
                          if sum(m in r["present_mods"] for r in results) > 0.5 * len(results)]
    K = 16
    seed_colors  = []
    mod_boundaries = []
    for mod in present_mods_main:
        mod_boundaries.append(len(seed_colors))
        seed_colors.extend([MOD_COLORS[mod]] * K)

    ntasks = len(tasks)
    fig, axes = plt.subplots(1, ntasks, figsize=(5 * ntasks, 4.5))
    if ntasks == 1:
        axes = [axes]
    fig.suptitle(f"D — ABMIL seed importances per task | split{split}_fold{fold}",
                 fontsize=11, fontweight="bold")
    for ax, task in zip(axes, tasks):
        vecs = [r["abmil_attn"][task] for r in results if task in r["abmil_attn"]]
        if len(vecs) == 0:
            arr = np.empty((0,))
        else:
            max_len = max(v.shape[0] for v in vecs)
            padded = [np.pad(v, (0, max_len - v.shape[0])) for v in vecs]
            arr = np.stack(padded)
        if len(arr) == 0:
            ax.set_visible(False); continue
        mu = arr.mean(0)[:len(seed_colors)]
        se = arr.std(0)[:len(seed_colors)]
        x  = np.arange(len(mu))
        ax.bar(x, mu, color=seed_colors[:len(mu)], alpha=0.85, width=0.9)
        ax.errorbar(x, mu, yerr=se, fmt="none", color="#333", lw=0.8, capsize=1)
        for b in mod_boundaries[1:]:
            ax.axvline(b - 0.5, color="#555", lw=0.8, ls="--")
        ylim = ax.get_ylim()
        for b, mod in zip(mod_boundaries, present_mods_main):
            ax.text(b + K / 2 - 0.5, ylim[1] * 0.93, mod, ha="center",
                    fontsize=8, color=MOD_COLORS[mod], fontweight="bold")
        ax.set_title(task, fontsize=9, color=TASK_COLORS.get(task, "#333"))
        ax.set_xlabel("Seed index", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("Mean attention weight", fontsize=8)
        ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(out_dir / "D_abmil_seed_importance.pdf", dpi=150, bbox_inches="tight")
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

    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(gate_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(n_mod)); ax.set_xticklabels(MOD_ORDER, fontsize=10)
    ax.set_yticks(range(len(tasks))); ax.set_yticklabels(tasks, fontsize=9)
    for ti in range(len(tasks)):
        for mi in range(n_mod):
            v = gate_matrix[ti, mi]
            ax.text(mi, ti, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.7 else "black")
    ax.set_title(f"E — TaskModalGate weights (mean, {len(gate_rows)} patients) | split{split}_fold{fold}",
                 fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Gate weight")
    fig.tight_layout()
    fig.savefig(out_dir / "E_task_modal_gate.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  E done")


# ── Panel F: modality combo scores ───────────────────────────────────────────

def panel_F(results, tasks, out_dir, split, fold):
    try:
        from sklearn.metrics import balanced_accuracy_score
        from sksurv.metrics import concordance_index_censored
    except ImportError:
        print("  F skipped (sksurv/sklearn not available)"); return

    present_mods_main = [m for m in MOD_ORDER
                          if sum(m in r["present_mods"] for r in results) > 0.5 * len(results)]
    combos = []
    for r in range(1, len(present_mods_main) + 1):
        for combo in combinations(present_mods_main, r):
            combos.append("+".join(combo))

    labels     = np.array([float(r["label"]) if r["label"] is not None else float("nan")
                            for r in results])
    ep_map = {
        "acr_cls":    (np.array([r.get("event_acr",   float("nan")) for r in results]),
                       np.array([r.get("tte_acr",     float("nan")) for r in results])),
        "acr_surv":   (np.array([r.get("event_acr",   float("nan")) for r in results]),
                       np.array([r.get("tte_acr",     float("nan")) for r in results])),
        "clad_surv":  (np.array([r.get("event_clad",  float("nan")) for r in results]),
                       np.array([r.get("tte_clad",    float("nan")) for r in results])),
        "death_surv": (np.array([r.get("event_death", float("nan")) for r in results]),
                       np.array([r.get("tte_death",   float("nan")) for r in results])),
    }

    # All-modality logits (original model run)
    combo_scores = {}
    all_key = "+".join(present_mods_main)
    for key in combos:
        logits = np.array([r["logits"].get(task, float("nan"))
                           for r in results for task in [tasks[0]]])
        combo_scores[key] = {}
        for task in tasks:
            raw = np.array([r["logits"].get(task, float("nan")) for r in results])
            try:
                if task == "acr_cls":
                    probs = 1 / (1 + np.exp(-raw))
                    valid = ~(np.isnan(labels) | np.isnan(probs))
                    if valid.sum() < 10: continue
                    combo_scores[key][task] = balanced_accuracy_score(
                        labels[valid].astype(int), (probs[valid] > 0.5).astype(int))
                elif task in ep_map:
                    ev, tte = ep_map[task]
                    valid = ~(np.isnan(ev) | np.isnan(tte) | np.isnan(raw))
                    if valid.sum() < 10: continue
                    combo_scores[key][task] = concordance_index_censored(
                        ev[valid].astype(bool), tte[valid], raw[valid])[0]
            except Exception:
                pass

    ntasks = len(tasks)
    fig, axes = plt.subplots(1, ntasks, figsize=(5 * ntasks, max(4, len(combos) * 0.35 + 1.5)))
    if ntasks == 1:
        axes = [axes]
    fig.suptitle(f"F — Modality combo scores (all-modality model) | split{split}_fold{fold}",
                 fontsize=11, fontweight="bold")
    for ax, task in zip(axes, tasks):
        sorted_keys = sorted(combo_scores, key=lambda k: combo_scores[k].get(task, 0))
        scores  = [combo_scores[k].get(task, float("nan")) for k in sorted_keys]
        colors  = []
        for c in sorted_keys:
            mods = c.split("+")
            colors.append(MOD_COLORS[mods[0]] if len(mods) == 1 else
                          "#333" if c == all_key else "#7cb9e8")
        y = np.arange(len(sorted_keys))
        ax.barh(y, scores, color=colors, alpha=0.85)
        ax.set_yticks(y); ax.set_yticklabels(sorted_keys, fontsize=7)
        ax.set_xlabel("BACC" if task == "acr_cls" else "C-index", fontsize=9)
        ax.set_title(task, fontsize=9, color=TASK_COLORS.get(task, "#333"))
        ax.axvline(0.5, color="#aaa", lw=0.8, ls="--")
        if all_key in combo_scores and task in combo_scores[all_key]:
            ax.axvline(combo_scores[all_key][task], color="#E53935", lw=1.2, ls=":")
    fig.tight_layout()
    fig.savefig(out_dir / "F_modality_combo_ablation.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  F done")


# ── Panel G: final rep hexbin ─────────────────────────────────────────────────

def panel_G(results, tasks, out_dir, split, fold):
    ep_keys = {
        "acr_cls":    ("event_acr",   "tte_acr"),
        "acr_surv":   ("event_acr",   "tte_acr"),
        "clad_surv":  ("event_clad",  "tte_clad"),
        "death_surv": ("event_death", "tte_death"),
    }
    for task in tasks:
        idx_valid = [i for i, r in enumerate(results) if task in r["final_reps"]]
        if len(idx_valid) < 20:
            continue
        reps = np.stack([results[i]["final_reps"][task] for i in idx_valid])
        xy   = _umap_embed(reps)

        ev_key, tte_key = ep_keys.get(task, ("event_acr", "tte_acr"))
        logits = np.array([results[i]["logits"].get(task, float("nan")) for i in idx_valid])
        ev     = np.array([results[i].get(ev_key,  float("nan")) for i in idx_valid])
        tte    = np.array([results[i].get(tte_key, float("nan")) for i in idx_valid])
        labs   = np.array([float(results[i]["label"]) if results[i]["label"] is not None
                            else float("nan") for i in idx_valid])

        scores    = 1 / (1 + np.exp(-logits)) if task == "acr_cls" else logits
        score_lbl = "P(ACR+)" if task == "acr_cls" else "Cox risk"
        gs        = 30

        fig = plt.figure(figsize=(18, 5))
        grd = gridspec.GridSpec(1, 4, figure=fig, wspace=0.06)
        axs = [fig.add_subplot(grd[0, i]) for i in range(4)]
        fig.suptitle(f"G — Final rep space: {task} | split{split}_fold{fold}",
                     fontsize=11, fontweight="bold")

        # label scatter
        ax = axs[0]
        ax.scatter(xy[labs == 0, 0], xy[labs == 0, 1], s=5, c="#1E88E5", alpha=0.5, label="Neg")
        ax.scatter(xy[labs == 1, 0], xy[labs == 1, 1], s=5, c="#E53935", alpha=0.7, label="Pos")
        if np.isnan(labs).any():
            ax.scatter(xy[np.isnan(labs), 0], xy[np.isnan(labs), 1],
                       s=3, c="#bbb", alpha=0.4)
        ax.set_title("Label", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        ax.legend(markerscale=2.5, fontsize=7, framealpha=0.6)

        # score hexbin (RED = HIGH RISK, BLUE = LOW RISK)
        ax = axs[1]
        valid = ~np.isnan(scores)
        if valid.sum() > 5:
            vl, vh = np.nanpercentile(scores[valid], 2), np.nanpercentile(scores[valid], 98)
            im = _hex_panel(ax, xy[valid], scores[valid], CMAP_HAZARD, vl, vh, gs, score_lbl)
            fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)

        # TTE hexbin (event patients only; censored shown as background)
        ax = axs[2]
        ev_m  = (~np.isnan(tte)) & (ev == 1)
        ce_m  = (~np.isnan(tte)) & (ev == 0)
        if ev_m.sum() > 5:
            vl, vh = np.nanpercentile(tte[ev_m], 2), np.nanpercentile(tte[ev_m], 98)
            if ce_m.any():
                ax.hexbin(xy[ce_m, 0], xy[ce_m, 1], gridsize=gs,
                          facecolors=_EMPTY_COLOR, linewidths=0.5, edgecolors="#aaa")
            im = _hex_panel(ax, xy[ev_m], tte[ev_m], CMAP_TTE, vl, vh, gs, "TTE (events)")
            fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)

        # Event density hexbin
        ax = axs[3]
        valid = ~np.isnan(ev)
        if valid.sum() > 5:
            im = _hex_panel(ax, xy[valid], ev[valid], CMAP_HAZARD, 0, 1, gs, "Event density")
            fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)

        _uniform_lim(axs, xy)
        fig.savefig(out_dir / f"G_final_rep_hexbin_{task}.pdf", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print("  G done")


# ── Top-level ─────────────────────────────────────────────────────────────────

def plot_all(results, seeds_init, tasks, split, fold, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[plot] {len(results)} patients → {out_dir}")
    panel_A(results, out_dir, split, fold)
    panel_B(results, seeds_init, out_dir, split, fold)
    panel_C(results, tasks, out_dir, split, fold)
    panel_D(results, tasks, out_dir, split, fold)
    panel_E(results, tasks, out_dir, split, fold)
    panel_F(results, tasks, out_dir, split, fold)
    panel_G(results, tasks, out_dir, split, fold)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--split",       type=int, default=0)
    pa.add_argument("--fold",        type=int, default=1)
    pa.add_argument("--variant",     default="mega",
                    choices=["mega","cls","acr_surv","clad_surv","death_surv"])
    pa.add_argument("--max-samples", type=int, default=None)
    pa.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    pa.add_argument("--out-dir",     default=None)
    args = pa.parse_args()

    device  = torch.device(args.device)
    out_dir = Path(args.out_dir) if args.out_dir \
              else OUT_ROOT / f"split{args.split}_fold{args.fold}_{args.variant}"

    results, seeds_init, tasks = extract_all(
        args.split, args.fold, args.variant, device, args.max_samples)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "results_raw.npy", np.array(results, dtype=object), allow_pickle=True)
    plot_all(results, seeds_init, tasks, args.split, args.fold, out_dir)

    # ── W&B logging ───────────────────────────────────────────────────────────
    try:
        import wandb
        fold_tag = f"split{args.split}_fold{args.fold}"
        run = wandb.init(
            project="chicago-mil-interpretability",
            name=f"set_mil_mt_{args.variant}_{fold_tag}",
            group="set_mil_mt",
            config={
                "variant": args.variant, "split": args.split,
                "fold": args.fold, "tasks": tasks,
            },
            reinit=True,
        )
        log_dict = {"n_samples": len(results)}
        # Modal contribution summary
        for task in tasks:
            mc_key = f"modal_contrib_{task}"
            contribs = [r.get(mc_key, {}) for r in results if r.get(mc_key)]
            if contribs:
                all_mods = set(k for d in contribs for k in d)
                for mod in all_mods:
                    vals = [d[mod] for d in contribs if mod in d]
                    log_dict[f"modal_contrib_{task}/{mod}"] = float(np.mean(vals))
        wandb.log(log_dict)

        # All panel PDFs/PNGs — convert PDFs to PNG via ImageMagick first
        import subprocess as _sp
        def _pdf_to_png(pdf):
            png = pdf.with_suffix(".png")
            if not png.exists():
                try:
                    _sp.run(["/usr/bin/convert", "-density", "150",
                             f"{pdf}[0]", str(png)],
                            check=True, capture_output=True)
                except Exception:
                    return None
            return png if png.exists() else None

        wandb_imgs = []
        for p in sorted(out_dir.glob("*.pdf")) + sorted(out_dir.glob("*.png")):
            if p.suffix == ".pdf":
                png = _pdf_to_png(p)
                if png:
                    wandb_imgs.append(wandb.Image(str(png), caption=p.stem))
            else:
                wandb_imgs.append(wandb.Image(str(p), caption=p.name))
        if wandb_imgs:
            wandb.log({"panels": wandb_imgs})
        run.finish()
        print(f"  W&B run: {run.url}")
    except Exception as e:
        print(f"  [wandb] logging failed: {e}")


if __name__ == "__main__":
    main()
