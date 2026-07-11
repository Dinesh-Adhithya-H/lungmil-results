"""
run_interpretability.py — Feature importance for cluster_prop (all tasks)
                          + ABMIL per-patch attention for late_cls (ACR cls).

Outputs (saved to OUT_DIR/interpretability/):
  cluster_prop_importances_fold{N}.csv  — logistic coef / Cox params per task
  abmil_attention_late_cls_fold{N}.csv  — mean per-cluster attention on test set
"""

import argparse
import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from mil.data.splits import build_splits_multitask
from mil.data.loader import _load_one_bag
from mil.training.classical_baselines import (
    ClusterPropExtractor, _clean_X, _cls_labels, _surv_labels,
    _fit_logistic, _CoxWrapper, PATCH_MODS,
)
from mil.models.builders import build_model_v8

SAMPLES_DIR = "/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
SPLITS_CSV  = "/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
RESULTS_DIR = Path("/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8")
PROPS_CSV   = RESULTS_DIR / "cluster_proportions.csv"
CKPT_LATE   = RESULTS_DIR / "phase2/split1_fold{fold}/late_cls/model_late_final.pt"
VOCAB_DIR   = Path("/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2")
HE_MAP_PATH = Path("/home/aih/dinesh.haridoss/chicago_mil/results/cluster_name_maps/HE_cluster_map.json")


def _load_name_maps() -> dict:
    """
    Returns {mod: [name_for_cluster_0, name_for_cluster_1, ...]} using vocab JSONs.
    HE:  subcluster_name (e.g. "0_1") → tissue type via HE_cluster_map.json
    BAL: cell type name directly from vocab (e.g. "CCR7+ DC1")
    CT:  numeric string from vocab (e.g. "CT_cluster_0")
    """
    import json
    maps = {}
    he_map = json.load(open(HE_MAP_PATH)) if HE_MAP_PATH.exists() else {}
    for mod, key in [("HE", "HE_cells"), ("BAL", "BAL_cells"), ("CT", "CT_cells")]:
        vocab_path = VOCAB_DIR / f"{key}_cluster_count_vocab.json"
        if not vocab_path.exists():
            maps[mod] = []
            continue
        v = json.load(open(vocab_path))
        cluster_names = v.get("cluster_names", [])
        if mod == "HE":
            # Map subcluster_name → tissue type
            maps[mod] = [he_map.get(cn, cn) for cn in cluster_names]
        elif mod == "BAL":
            maps[mod] = list(cluster_names)
        else:
            maps[mod] = [f"CT_cluster_{cn}" for cn in cluster_names]
    return maps


def _load_clinical_names() -> list:
    import json
    cv_path = VOCAB_DIR / "clinical_vocab.json"
    if cv_path.exists():
        cv = json.load(open(cv_path))
        return cv.get("feature_cols", [])
    return [f"Clinical_{k}" for k in range(102)]


def feat_names(ext: ClusterPropExtractor, clin_names: list, cluster_name_maps: dict) -> list:
    """Feature names matching the order ClusterPropExtractor.transform() builds."""
    names = [f"Clinical: {n}" for n in clin_names[:len(ext._csv_clin_cols)]]
    for mod in PATCH_MODS:
        nc = ext.n_clusters.get(mod, 0)
        mod_names = cluster_name_maps.get(mod, [])
        for k in range(nc):
            label = mod_names[k] if k < len(mod_names) else f"{mod}_clr_{k}"
            names.append(f"{mod}: {label}")
    return names


def coef_to_df(names_post_vt, coefs, task, model_name):
    rows = []
    for name, coef in zip(names_post_vt, coefs):
        if name.startswith("Clinical"):
            mod = "Clinical"
        elif name.startswith("HE"):
            mod = "HE"
        elif name.startswith("BAL"):
            mod = "BAL"
        elif name.startswith("CT"):
            mod = "CT"
        else:
            mod = "unknown"
        rows.append({
            "task": task, "model": model_name,
            "feature": name, "modality": mod,
            "coefficient": float(coef), "abs_coef": abs(float(coef)),
        })
    df = pd.DataFrame(rows).sort_values("abs_coef", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    return df


# ── Part 1: cluster_prop feature importances ──────────────────────────────────

def run_cluster_prop_interp(fold: int, out_dir: Path, clin_names: list, cluster_maps: dict):
    print(f"\n{'='*60}")
    print(f"  cluster_prop interpretability  fold={fold}")
    print(f"{'='*60}")

    splits    = build_splits_multitask(SAMPLES_DIR, SPLITS_CSV, fold, split=1)
    fit_recs  = splits["train"] + splits["val"]
    test_recs = splits["test"]

    ext = ClusterPropExtractor(SAMPLES_DIR, props_csv=str(PROPS_CSV))
    ext.fit(fit_recs)

    all_names = feat_names(ext, clin_names, cluster_maps)
    X_tr = _clean_X(ext.transform(fit_recs))
    X_te = _clean_X(ext.transform(test_recs))

    all_dfs = []

    # ACR classification — logistic regression
    tr_m, tr_y = _cls_labels(fit_recs)
    if tr_m.sum() >= 10:
        from sklearn.feature_selection import VarianceThreshold
        vt = VarianceThreshold(threshold=1e-8).fit(X_tr[tr_m])
        sel_idx   = vt.get_support(indices=True)
        sel_names = [all_names[i] for i in sel_idx]
        pipe  = _fit_logistic(X_tr[tr_m], tr_y[tr_m])
        coefs = pipe.named_steps["lr"].coef_[0]
        all_dfs.append(coef_to_df(sel_names, coefs, "acr_cls", "cluster_prop_logistic"))
        print(f"  ACR cls: {len(sel_names)} features")

    # Survival endpoints — Cox
    for ep in ("acr", "clad", "death"):
        tr_sm, tr_t, tr_e = _surv_labels(fit_recs, ep)
        if tr_sm.sum() < 10 or tr_e[tr_sm].sum() < 3:
            continue
        try:
            cox_w     = _CoxWrapper().fit(X_tr[tr_sm], tr_t[tr_sm], tr_e[tr_sm])
            sel_idx   = cox_w.vt.get_support(indices=True)
            sel_names = [all_names[i] for i in sel_idx]
            params    = cox_w.cox.params_.values
            all_dfs.append(coef_to_df(sel_names, params, f"{ep}_surv", "cluster_prop_cox"))
            print(f"  {ep} surv: {len(sel_names)} features")
        except Exception as e:
            print(f"  {ep} Cox failed: {e}")

    if not all_dfs:
        print("  No results.")
        return

    df = pd.concat(all_dfs, ignore_index=True)
    out_path = out_dir / f"cluster_prop_importances_fold{fold}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")

    for task, grp in df.groupby("task"):
        print(f"\n  Top-15 [{task}]:")
        print(grp.head(15)[["rank", "feature", "modality", "coefficient"]].to_string(index=False))


# ── Part 2: ABMIL late_cls per-patch attention ────────────────────────────────

def run_abmil_attention(fold: int, out_dir: Path, cluster_maps: dict):
    ckpt_path = Path(str(CKPT_LATE).format(fold=fold))
    if not ckpt_path.exists():
        print(f"\n  [warn] Checkpoint not found: {ckpt_path}")
        return

    print(f"\n{'='*60}")
    print(f"  ABMIL late_cls attention  fold={fold}")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    model = build_model_v8("late", task="cls")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval().to(device)

    splits    = build_splits_multitask(SAMPLES_DIR, SPLITS_CSV, fold, split=1)
    test_recs = splits["test"]

    # {mod: {cluster_id: [attn_values...]}}
    cluster_attn = {mod: {} for mod in PATCH_MODS}
    n_patients   = {mod: 0 for mod in PATCH_MODS}

    n_done = 0
    for r in test_recs:
        stem = r["stem"]
        pt_path = Path(SAMPLES_DIR) / f"{stem}.pt"
        if not pt_path.exists():
            continue
        try:
            _, entry = _load_one_bag((stem, pt_path))
        except Exception:
            continue

        # Also load raw instance_cluster_ids
        try:
            raw = torch.load(pt_path, map_location="cpu", weights_only=False)
            iid = raw.get("instance_cluster_ids", {})
        except Exception:
            continue

        with torch.no_grad():
            for mod in PATCH_MODS:
                t = entry.get(mod)
                if t is None or t.numel() == 0:
                    continue

                cids_t = iid.get(f"{mod}_cells")
                if cids_t is None or not isinstance(cids_t, torch.Tensor):
                    continue
                cids_np = cids_t.long().numpy()
                if len(cids_np) != t.shape[0]:
                    continue

                enc = model.encoders[mod]
                try:
                    _, alpha, _ = enc(t.to(device))
                    alpha_np = alpha.cpu().numpy()  # (N,)
                except Exception as e:
                    continue

                for cid, a in zip(cids_np, alpha_np):
                    cid = int(cid)
                    if cid not in cluster_attn[mod]:
                        cluster_attn[mod][cid] = []
                    cluster_attn[mod][cid].append(float(a))
                n_patients[mod] += 1

        n_done += 1
        if n_done % 100 == 0:
            print(f"  {n_done}/{len(test_recs)} patients...")

    print(f"  Done: {n_done} patients, per-mod patients: { {m: n_patients[m] for m in PATCH_MODS} }")

    rows = []
    for mod in PATCH_MODS:
        mod_names = cluster_maps.get(mod, [])
        for cid, vals in cluster_attn[mod].items():
            label = mod_names[cid] if cid < len(mod_names) else f"{mod}_cluster_{cid}"
            rows.append({
                "task": "acr_cls", "model": "late_cls_abmil",
                "modality": mod, "cluster_id": cid,
                "cluster_name": label,
                "mean_attn": float(np.mean(vals)),
                "std_attn":  float(np.std(vals)),
                "n_patches": len(vals),
                "n_patients": n_patients[mod],
            })

    if not rows:
        print("  No attention data collected.")
        return

    df = pd.DataFrame(rows)
    dfs = []
    for mod, grp in df.groupby("modality"):
        grp = grp.sort_values("mean_attn", ascending=False).copy()
        grp["rank"] = range(1, len(grp) + 1)
        dfs.append(grp)
    df = pd.concat(dfs, ignore_index=True)

    out_path = out_dir / f"abmil_attention_late_cls_fold{fold}.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")

    for mod, grp in df.groupby("modality"):
        print(f"\n  Top-10 clusters [{mod}]:")
        print(grp.head(10)[["rank", "cluster_id", "cluster_name", "mean_attn", "n_patches"]].to_string(index=False))


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fold",    type=int, default=0)
    p.add_argument("--out-dir", default=str(RESULTS_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) / "interpretability"
    out_dir.mkdir(parents=True, exist_ok=True)

    clin_names      = _load_clinical_names()
    cluster_maps    = _load_name_maps()
    print(f"  Clinical features ({len(clin_names)}): {clin_names[:5]}...")
    for mod, names in cluster_maps.items():
        print(f"  {mod} cluster names ({len(names)}): {names[:3]}...")

    run_cluster_prop_interp(args.fold, out_dir, clin_names, cluster_maps)
    run_abmil_attention(args.fold, out_dir, cluster_maps)

    print(f"\n  All outputs → {out_dir}")


if __name__ == "__main__":
    main()
