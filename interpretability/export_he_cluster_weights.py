"""
Export per-cluster HE attention weights as CSV — both ABMIL (Phase1) and LongMIL (Phase2).

For each task × model × cluster:
  cluster_id, bio_category, mean_rel_attn, sem_rel_attn, n_instances (ABMIL only)

Reads from:
  - Phase1: runs SingleModalMIL inference on test patients (all 5 splits)
  - Phase2: cluster_aff_data JSONs (precomputed, no GPU needed)

Output: interpretability/he_cluster_weights/
  he_cluster_weights_abmil_{task}.csv
  he_cluster_weights_longmil_{task}.csv
  he_cluster_weights_combined_{task}.csv   ← both side-by-side

Usage: sbatch interpretability/submit_export_he_cluster_weights.sh
"""

import json, sys, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mil.models.phase1 import SingleModalMIL
from mil.data.splits import build_splits_longitudinal
from mil.data.loader import preload_bags
from interpretability.shared import SPLITS_CSV, SAMPLES_DIR, HE_BIO_MAP

PHASE1_ROOT = ROOT / "results" / "mm_abmil_v8" / "phase1"
INTERP_ROOT = ROOT / "interpretability" / "longitudinal_mk_interp"
OUT_ROOT    = ROOT / "interpretability" / "he_cluster_weights"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

VARIANT = "longitudinal_mk_no_alibi_v2"

TASK_CFG = {
    "acr_cls":   {"p1_task": "acr",      "dir_suffix": "acr_cls",   "json_key": "acr_cls"},
    "acr_surv":  {"p1_task": "acr_surv", "dir_suffix": "acr_surv",  "json_key": "acr_surv"},
    "clad_surv": {"p1_task": "clad",     "dir_suffix": "clad_surv", "json_key": "clad"},
    "death_surv":{"p1_task": "death",    "dir_suffix": "death_surv","json_key": "death"},
}


def rel_attn_from_dict(cluster_vals):
    """Normalize to relative attention (1.0 = uniform). Returns {name: (mean, sem, n)}."""
    all_vals = [v for vals in cluster_vals.values() for v in vals]
    overall  = np.mean(all_vals) if all_vals else 1.0
    out = {}
    for name, vals in cluster_vals.items():
        arr = np.array(vals)
        out[name] = (
            float(np.mean(arr) / overall),
            float(np.std(arr) / overall / np.sqrt(max(len(arr), 1))),
            int(len(arr)),
        )
    return out


# ── Phase1 ABMIL ─────────────────────────────────────────────────────────────

def load_phase1(split, p1_task, device):
    ckpt = PHASE1_ROOT / f"split{split}_fold0" / p1_task / "HE" / "final_combined" / "best_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(str(ckpt))
    raw   = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    state = raw.get("model_state_dict", raw.get("state_dict", raw))
    model = SingleModalMIL(feat_dim=1024)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device)


def extract_abmil(p1_task, device, bag_cache, cluster_cache):
    """Per-instance alpha grouped by cluster, across all 5 splits × test set."""
    cluster_alphas = defaultdict(list)
    for split in range(5):
        print(f"  [ABMIL HE] split {split}", flush=True)
        try:
            model = load_phase1(split, p1_task, device)
        except FileNotFoundError as e:
            print(f"    [skip] {e}", flush=True); continue
        splits = build_splits_longitudinal(SAMPLES_DIR, SPLITS_CSV, fold=0, split=split)
        with torch.no_grad():
            for pat in splits["test"]:
                for stem in pat["stems"]:
                    x        = bag_cache.get(stem, {}).get("HE")
                    cl_names = cluster_cache.get(stem)
                    if x is None or cl_names is None or len(cl_names) != x.shape[0]: continue
                    try:
                        _, extras = model(x.to(device), return_extras=True)
                        alpha = extras["alpha"].squeeze().detach().cpu().numpy()
                    except Exception as e:
                        print(f"    [err] {stem}: {e}", flush=True); continue
                    for a, cn in zip(alpha, cl_names):
                        cluster_alphas[cn].append(float(a))
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    return cluster_alphas


# ── Phase2 LongMIL ────────────────────────────────────────────────────────────

def extract_longmil(dir_suffix, json_key):
    """(hi+lo).sum over K seeds, per cluster, all 5 splits."""
    cluster_vals = defaultdict(list)
    for split in range(5):
        p = (INTERP_ROOT / f"{VARIANT}_split{split}_fold0_{dir_suffix}"
             / f"cluster_aff_data_{json_key}.json")
        if not p.exists():
            print(f"  [missing] {p}", flush=True); continue
        d    = json.loads(p.read_text())
        ca   = d.get("cluster_aff", {}).get("HE", {})
        names = ca.get("names", [])
        hi    = np.array(ca.get("hi", []))
        lo    = np.array(ca.get("lo", []))
        if hi.size == 0 or not names: continue
        total = (hi + lo).sum(axis=0)          # (C,)
        for ci, name in enumerate(names):
            cluster_vals[name].append(float(total[ci]))
    return cluster_vals


# ── Build cluster cache ───────────────────────────────────────────────────────

def build_cluster_cache(stems):
    cache = {}
    for stem in stems:
        pt = Path(SAMPLES_DIR) / f"{stem}.pt"
        if not pt.exists(): continue
        try:
            d  = torch.load(str(pt), map_location="cpu", weights_only=False)
            cl = d.get("cluster_labels", {}).get("HE_cells")
            if cl is not None and len(cl) > 0:
                cache[stem] = list(cl)
        except Exception: pass
    return cache


# ── Main ─────────────────────────────────────────────────────────────────────

def to_df(rel, model_label, task_key):
    rows = []
    for cluster_id, (mean_rel, sem_rel, n) in rel.items():
        rows.append({
            "cluster_id":   cluster_id,
            "bio_category": HE_BIO_MAP.get(cluster_id, "Unknown"),
            "model":        model_label,
            "task":         task_key,
            "mean_rel_attn": round(mean_rel, 5),
            "sem_rel_attn":  round(sem_rel, 5),
            "n_instances":   n,
        })
    df = pd.DataFrame(rows)
    df = df.sort_values(["bio_category", "mean_rel_attn"], ascending=[True, False])
    return df


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Preload bags + cluster cache
    print("Loading stems...", flush=True)
    all_stems = set()
    for split in range(5):
        sp = build_splits_longitudinal(SAMPLES_DIR, SPLITS_CSV, fold=0, split=split)
        for pats in sp.values():
            for pat in pats: all_stems.update(pat["stems"])

    bag_cache     = preload_bags(list(all_stems), SAMPLES_DIR, n_workers=4)
    cluster_cache = build_cluster_cache(list(all_stems))
    print(f"  {len(bag_cache)} bags, {len(cluster_cache)} with HE cluster labels", flush=True)

    all_dfs = []

    for task_key, cfg in TASK_CFG.items():
        print(f"\n=== {task_key} ===", flush=True)

        # ABMIL
        abmil_raw = extract_abmil(cfg["p1_task"], device, bag_cache, cluster_cache)
        abmil_rel = rel_attn_from_dict(abmil_raw)
        df_abmil  = to_df(abmil_rel, "ABMIL_HE_Phase1", task_key)
        df_abmil.to_csv(OUT_ROOT / f"he_cluster_weights_abmil_{task_key}.csv", index=False)
        print(f"  ABMIL: {len(df_abmil)} clusters", flush=True)

        # LongMIL
        longmil_raw = extract_longmil(cfg["dir_suffix"], cfg["json_key"])
        longmil_rel = rel_attn_from_dict(longmil_raw)
        df_longmil  = to_df(longmil_rel, "LongMIL_Phase2", task_key)
        df_longmil.to_csv(OUT_ROOT / f"he_cluster_weights_longmil_{task_key}.csv", index=False)
        print(f"  LongMIL: {len(df_longmil)} clusters", flush=True)

        # Combined (merge on cluster_id)
        merged = df_abmil[["cluster_id","bio_category","mean_rel_attn","sem_rel_attn","n_instances"]].rename(
            columns={"mean_rel_attn":"abmil_mean_rel","sem_rel_attn":"abmil_sem","n_instances":"n_instances"}
        ).merge(
            df_longmil[["cluster_id","mean_rel_attn","sem_rel_attn"]].rename(
                columns={"mean_rel_attn":"longmil_mean_rel","sem_rel_attn":"longmil_sem"}
            ), on="cluster_id", how="outer"
        ).sort_values(["bio_category","abmil_mean_rel"], ascending=[True, False])
        merged["task"] = task_key
        merged.to_csv(OUT_ROOT / f"he_cluster_weights_combined_{task_key}.csv", index=False)
        all_dfs.append(merged)

    # All tasks in one file
    pd.concat(all_dfs).to_csv(OUT_ROOT / "he_cluster_weights_all_tasks.csv", index=False)
    print(f"\nDone. → {OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
