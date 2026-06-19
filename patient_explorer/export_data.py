"""
Export pre-computed data for the Patient Explorer web app.

Run via SLURM (needs GPU + large RAM for bag loading):
    sbatch submit_export.sh

Writes to patient_explorer/data/:
    splits.csv                  — patient metadata / labels
    predictions_all.csv         — merged model predictions (4 tasks)
    episodes.csv                — ACR episode summary
    umap_embeddings.csv         — UMAP x/y per patient per task
    he_cluster_freq.csv         — per-patient HE cluster proportions
    bal_cluster_freq.csv        — per-patient BAL cluster proportions
    ct_cluster_freq.csv         — per-patient CT cluster proportions
    clinical_features.csv       — clinical feature values per patient/date
    clinical_feature_names.csv  — idx → human-readable name mapping
    clinical_attn.csv           — per-patient attention weights (Clinical modality)
"""

from __future__ import annotations
import sys, os, ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── paths ──────────────────────────────────────────────────────────────────
HERE        = Path(__file__).parent
DATA_DIR    = HERE / "data"
DATA_DIR.mkdir(exist_ok=True)

REPO        = HERE.parent
SPLITS_CSV  = Path(os.environ.get("SPLITS_CSV",
    str(REPO / "chicago/plots/multimodal_splits_nested_cv.csv")))
SAMPLES_DIR = Path(os.environ.get("SAMPLES_DIR",
    "/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR",
    str(REPO / "results/full_data_middle")))
EMBD_DIR    = Path(os.environ.get("EMBD_DIR",
    str(REPO / "chicago/plots/phase2_embeddings/fold_0/fusion")))
EPISODE_CSV = REPO / "results/acr_episode_analysis/episode_table.csv"
PROP_DIR    = REPO / "results/cluster_proportions"

sys.path.insert(0, str(REPO / "src"))


# ══════════════════════════════════════════════════════════════════════════
# 1. splits / patient metadata
# ══════════════════════════════════════════════════════════════════════════

def export_splits():
    df = pd.read_csv(SPLITS_CSV, parse_dates=["anchor_dt"])
    # keep useful columns only
    keep = [
        "file", "patient_id", "anchor_dt", "label", "acr_grade", "acr_encoded",
        "clad_status", "clad_days", "death_status", "death_days",
        "has_CT", "has_CT_Radiomics", "has_HE", "has_BAL", "has_Clinical",
        "biopsy_grade_A", "acr_status", "acr_days",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    df["stem"] = df["file"].str.replace(".pt", "", regex=False)
    df.sort_values(["patient_id", "anchor_dt"], inplace=True)
    out = DATA_DIR / "splits.csv"
    df.to_csv(out, index=False)
    print(f"  splits.csv → {len(df)} rows")
    return df


# ══════════════════════════════════════════════════════════════════════════
# 2. model predictions (merge 4 tasks)
# ══════════════════════════════════════════════════════════════════════════

def export_predictions():
    tasks = ["acr_cls", "acr_surv", "clad_surv", "death_surv"]
    dfs = []
    for task in tasks:
        p = RESULTS_DIR / task / "predictions.csv"
        if not p.exists():
            print(f"  [skip] {p} not found")
            continue
        df = pd.read_csv(p, parse_dates=["anchor_dt"])
        df["task"] = task
        dfs.append(df)
    if not dfs:
        print("  No prediction files found.")
        return pd.DataFrame()

    # merge on patient_id + anchor_dt
    base = None
    for df in dfs:
        score_cols = [c for c in df.columns
                      if c in ("pred_prob_acr", "hazard_acr", "hazard_clad", "hazard_death")]
        meta_cols  = ["patient_id", "anchor_dt", "stem",
                      "true_acr_cls", "acr_event", "acr_days",
                      "clad_event", "clad_time", "death_event", "death_time"]
        meta_cols  = [c for c in meta_cols if c in df.columns]
        sub = df[meta_cols + score_cols].copy()
        if base is None:
            base = sub
        else:
            new_scores = [c for c in score_cols if c not in base.columns]
            if new_scores:
                merge_on = [c for c in ["patient_id", "anchor_dt"] if c in base.columns]
                base = base.merge(sub[merge_on + new_scores], on=merge_on, how="outer")

    out = DATA_DIR / "predictions_all.csv"
    base.sort_values(["patient_id", "anchor_dt"]).to_csv(out, index=False)
    print(f"  predictions_all.csv → {len(base)} rows")
    return base


# ══════════════════════════════════════════════════════════════════════════
# 3. ACR episodes
# ══════════════════════════════════════════════════════════════════════════

def export_episodes():
    if not EPISODE_CSV.exists():
        print(f"  [skip] {EPISODE_CSV} not found")
        return
    df = pd.read_csv(EPISODE_CSV)
    out = DATA_DIR / "episodes.csv"
    df.to_csv(out, index=False)
    print(f"  episodes.csv → {len(df)} rows")


# ══════════════════════════════════════════════════════════════════════════
# 4. UMAP embeddings
# ══════════════════════════════════════════════════════════════════════════

def export_umap(splits_df: pd.DataFrame):
    """Load raw embeddings from npz, compute UMAP to 2D, save as CSV."""
    try:
        import umap as umap_lib
    except ImportError:
        print("  [umap] umap-learn not installed — skipping")
        return

    rows = []
    # collect all splits for a combined UMAP (shared embedding space)
    all_embs, all_stems, all_pids = [], [], []
    for split in ["train", "val", "test"]:
        f = EMBD_DIR / f"embeddings_{split}.npz"
        if not f.exists():
            continue
        data = np.load(f, allow_pickle=True)
        if "embeddings" not in data:
            print(f"  [umap] {f.name}: no embeddings key")
            continue
        embs  = data["embeddings"].astype(np.float32)
        stems = [str(s) for s in data["stem"]] if "stem" in data else \
                [str(s) for s in data.get("patient_id", range(len(embs)))]
        pids  = [str(p) for p in data["patient_id"]] if "patient_id" in data else stems
        all_embs.append(embs)
        all_stems.extend(stems)
        all_pids.extend(pids)
        print(f"  [umap] loaded {split}: {len(embs)} samples × {embs.shape[1]}d")

    if not all_embs:
        print("  [umap] No embedding files found.")
        return

    X = np.concatenate(all_embs, axis=0)
    print(f"  [umap] computing UMAP on {len(X)} samples …")
    reducer = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    xy = reducer.fit_transform(X)

    stem_to_dt = splits_df.set_index("stem")["anchor_dt"].to_dict() \
                 if "stem" in splits_df.columns else {}
    for i, (stem, pid) in enumerate(zip(all_stems, all_pids)):
        rows.append({
            "stem":       stem,
            "patient_id": pid,
            "anchor_dt":  stem_to_dt.get(stem),
            "umap_x":     float(xy[i, 0]),
            "umap_y":     float(xy[i, 1]),
        })

    umap_df = pd.DataFrame(rows)
    out = DATA_DIR / "umap_embeddings.csv"
    umap_df.to_csv(out, index=False)
    print(f"  umap_embeddings.csv → {len(umap_df)} rows")


# ══════════════════════════════════════════════════════════════════════════
# 5. Cluster frequencies from bag .pt files
# ══════════════════════════════════════════════════════════════════════════

def _load_bag(stem: str, samples_dir: Path) -> dict | None:
    p = samples_dir / f"{stem}.pt"
    if not p.exists():
        p = samples_dir / f"{int(stem):05d}.pt"
    if not p.exists():
        return None
    try:
        return torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        return None


def _cluster_freq_from_bag(bag_dict: dict, mod: str) -> tuple[np.ndarray | None, list[str]]:
    """Return (freq_vector, cluster_names) or (None, [])."""
    bag = bag_dict.get(mod)
    if bag is None:
        return None, []
    N = len(bag)
    counts_t = bag_dict.get(f"{mod}_count_onehot")
    names    = bag_dict.get(f"_{mod}_cluster_names")
    if counts_t is None:
        return None, []
    t = counts_t.cpu().float()
    if t.dim() == 2:
        K   = t.shape[0]
        ids = t.argmax(dim=0).numpy().astype(int)
    else:
        counts = t.squeeze().numpy().astype(int).flatten()
        K      = len(counts)
        ids    = np.repeat(np.arange(K), counts)
    if len(ids) != N:
        return None, []
    cnt = np.bincount(ids, minlength=K).astype(float)
    total = cnt.sum()
    if total == 0:
        return None, []
    freq = cnt / total
    if names is None or len(names) != K:
        names = [f"cluster_{k}" for k in range(K)]
    else:
        names = [f"cluster_{k}" for k in range(K)]  # use generic keys; names go in separate file
    return freq, names


def export_cluster_freqs(splits_df: pd.DataFrame):
    """Copy pre-computed per-patient cluster proportion CSVs into the data dir."""
    import shutil
    for mod in ("HE", "BAL", "CT"):
        src = PROP_DIR / f"{mod.lower()}_cluster_prop.csv"
        if not src.exists():
            print(f"  [skip] {mod}: {src} not found — run extract_cluster_proportions.py first")
            continue
        dst = DATA_DIR / f"{mod.lower()}_cluster_freq.csv"
        shutil.copy(src, dst)
        df = pd.read_csv(dst, dtype={"stem": str})
        cluster_cols = [c for c in df.columns if c.startswith("cluster_")]
        print(f"  {dst.name} → {len(df)} rows × {len(cluster_cols)} clusters")

        # copy cluster names mapping too
        names_src = PROP_DIR / f"{mod.lower()}_cluster_names.csv"
        if names_src.exists():
            shutil.copy(names_src, DATA_DIR / f"{mod.lower()}_cluster_names.csv")


# ══════════════════════════════════════════════════════════════════════════
# 6. Clinical features (from Clinical bag)
# ══════════════════════════════════════════════════════════════════════════

def export_clinical(splits_df: pd.DataFrame):
    rows = []
    feat_names_dict = {}

    for _, row in splits_df.iterrows():
        if not row.get("has_Clinical", False):
            continue
        stem = row.get("stem", "")
        bd   = _load_bag(stem, SAMPLES_DIR)
        if bd is None:
            continue
        # clinical bag is under inputs["Clinical"] as a (102,) tensor
        bag = None
        inputs = bd.get("inputs")
        if isinstance(inputs, dict):
            bag = inputs.get("Clinical")
        if bag is None:
            bag = bd.get("Clinical")
        if bag is None:
            continue

        bag_np = bag.cpu().float().numpy()
        if bag_np.ndim == 2:
            vals = bag_np[:, 0]
        elif bag_np.ndim == 1:
            vals = bag_np
        else:
            continue

        r = {
            "patient_id": row["patient_id"],
            "anchor_dt":  row["anchor_dt"],
            "stem":       stem,
        }
        for i, v in enumerate(vals):
            r[f"feat_{i}"] = float(v)
        rows.append(r)

        # feature names: stored as clinical_feature_names
        if not feat_names_dict:
            names = bd.get("clinical_feature_names") or bd.get("_Clinical_feature_names")
            if names:
                feat_names_dict = {i: str(n) for i, n in enumerate(names)}

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(DATA_DIR / "clinical_features.csv", index=False)
        print(f"  clinical_features.csv → {len(df)} rows")
    else:
        print("  [skip] clinical_features: no data")

    if feat_names_dict:
        fn_df = pd.DataFrame(
            [{"idx": k, "name": v} for k, v in feat_names_dict.items()]
        )
        fn_df.to_csv(DATA_DIR / "clinical_feature_names.csv", index=False)
        print(f"  clinical_feature_names.csv → {len(fn_df)} features")


# ══════════════════════════════════════════════════════════════════════════
# 7. Per-patient clinical attention weights
# ══════════════════════════════════════════════════════════════════════════

def export_clinical_attention(splits_df: pd.DataFrame):
    """Load the full-data MiddleFusionMIL and extract clinical attention."""
    try:
        from mil.models.middle_fusion import MiddleFusionMIL
        from mil.data.loader import preload_bags
    except ImportError:
        print("  [skip] clinical_attn: mil package not importable")
        return

    model_path = REPO / "results/full_data_middle/acr_cls/model.pt"
    if not model_path.exists():
        # try to find any model checkpoint
        cands = list((REPO / "results/full_data_middle").rglob("*.pt"))
        if not cands:
            print("  [skip] clinical_attn: no model checkpoint found")
            return
        model_path = cands[0]
        print(f"  [attn] using checkpoint {model_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        model = ckpt if isinstance(ckpt, torch.nn.Module) else None
        if model is None and isinstance(ckpt, dict):
            # try to reconstruct — skip for now
            print("  [skip] clinical_attn: checkpoint is dict, model reconstruction needed")
            return
    except Exception as e:
        print(f"  [skip] clinical_attn: {e}")
        return

    model.eval()
    rows = []
    feat_names_list = None

    for _, row in splits_df.iterrows():
        if not row.get("has_Clinical", False):
            continue
        stem = row.get("stem", "")
        bd   = _load_bag(stem, SAMPLES_DIR)
        if bd is None:
            continue
        bag = bd.get("Clinical")
        if bag is None:
            continue

        if feat_names_list is None:
            feat_names_list = bd.get("_Clinical_feature_names")

        bag_t = bag.float().to(device)
        attn  = None
        raw   = {}

        try:
            # hook onto clinical encoder's att_w
            clinical_enc = model.encoders.get("Clinical") or \
                           getattr(model, "clinical_encoder", None)
            if clinical_enc is not None and hasattr(clinical_enc, "att_w"):
                def _hook(m, inp, out):
                    raw["v"] = out.detach().reshape(-1).cpu().float().numpy()
                h = clinical_enc.att_w.register_forward_hook(_hook)
                with torch.no_grad():
                    model({"Clinical": bag_t.unsqueeze(0)})
                h.remove()
                attn = raw.get("v")
        except Exception:
            pass

        if attn is None or len(attn) != 102:
            continue

        r = {
            "patient_id": row["patient_id"],
            "anchor_dt":  row["anchor_dt"],
            "stem":       stem,
        }
        for i, v in enumerate(attn):
            r[f"feat_{i}"] = float(v)
        rows.append(r)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(DATA_DIR / "clinical_attn.csv", index=False)
        print(f"  clinical_attn.csv → {len(df)} rows")


# ══════════════════════════════════════════════════════════════════════════
# 8. Benchmark summary CSV  (parse all metrics_*.json from results dir)
# ══════════════════════════════════════════════════════════════════════════

def export_benchmark():
    """Aggregate all per-fold metrics JSON files into benchmark_summary.csv."""
    import json, re
    results_dir = REPO / "results" / "mm_abmil_v8"
    if not results_dir.exists():
        print(f"  [skip] benchmark: {results_dir} not found")
        return

    rows = []
    for jf in sorted(results_dir.glob("metrics_split*_fold*_*.json")):
        parts = jf.stem.split("_")
        try:
            # stem: metrics_split1_fold0_late_cls
            split_val = int(parts[1].replace("split",""))
            fold_val  = int(parts[2].replace("fold",""))
            variant   = parts[3]
            task      = "_".join(parts[4:])
        except (IndexError, ValueError):
            continue
        with open(jf) as f:
            d = json.load(f)
        td = d.get("test", d)
        row = {
            "split": split_val, "fold": fold_val,
            "variant": variant, "task": task,
            "model": f"{variant}_{task}",
            "auc":     td.get("auc",     float("nan")),
            "auprc":   td.get("auprc",   float("nan")),
            "bacc":    td.get("bacc",    float("nan")),
            "c_index": td.get("c_index", float("nan")),
            "mcc":     td.get("mcc",     float("nan")),
            "sens":    td.get("sens",    float("nan")),
            "spec":    td.get("spec",    float("nan")),
        }
        # unimodal ablation
        for mod, vals in td.get("unimodal_ablation", {}).items():
            rows.append({**row,
                "model": f"{mod}_only (ablation)",
                "variant": "unimodal", "task": task,
                "auc": vals.get("auc", float("nan")),
                "bacc": vals.get("bacc", float("nan")),
                "auprc": float("nan"), "c_index": float("nan"),
                "mcc": float("nan"), "sens": float("nan"), "spec": float("nan"),
            })
        rows.append(row)

    # Also add classical baselines
    bl_path = results_dir / "baselines_summary.json"
    if bl_path.exists():
        with open(bl_path) as f:
            bl = json.load(f)
        for name, vals in bl.items():
            if not isinstance(vals, dict): continue
            rows.append({
                "split": 1, "fold": 0, "variant": "classical", "task": "multi",
                "model": name,
                "auc":     vals.get("test_auc",    float("nan")),
                "auprc":   float("nan"),
                "bacc":    vals.get("test_bacc",   float("nan")),
                "c_index": vals.get("test_ci_acr", float("nan")),
                "mcc": float("nan"), "sens": float("nan"), "spec": float("nan"),
            })
        # unimodal baselines
        for key, vals in bl.get("unimodal_baselines", {}).items():
            mod, task_suffix = key.rsplit("_", 1) if "_" in key else (key, "acr")
            rows.append({
                "split": 1, "fold": 0,
                "variant": "unimodal_baseline", "task": task_suffix,
                "model": f"{mod} baseline ({task_suffix})",
                "auc":     vals.get("auc",     float("nan")),
                "auprc":   vals.get("auprc",   float("nan")),
                "bacc":    vals.get("bacc",    float("nan")),
                "c_index": vals.get("c_index", float("nan")),
                "mcc":     vals.get("mcc",     float("nan")),
                "sens":    vals.get("sens",    float("nan")),
                "spec":    vals.get("spec",    float("nan")),
            })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(DATA_DIR / "benchmark_summary.csv", index=False)
        print(f"  benchmark_summary.csv → {len(df)} rows, {df['model'].nunique()} models")
    else:
        print("  [skip] benchmark: no metrics files found")


# ══════════════════════════════════════════════════════════════════════════
# 9. Copy nature_analysis outputs into data/
# ══════════════════════════════════════════════════════════════════════════

def export_nature_analysis():
    """Copy outputs from nature_analysis.py into data/ for the web app."""
    nature_dir = REPO / "analysis" / "nature_paper"
    if not nature_dir.exists():
        print(f"  [skip] nature analysis outputs not found at {nature_dir}")
        return
    files_to_copy = [
        "differential_abundance.csv",
        "pca_scores.csv",
        "cross_modal_corr.csv",
        "cohort_summary.json",
        "sample_table.csv",
    ]
    import shutil
    for fn in files_to_copy:
        src = nature_dir / fn
        dst = DATA_DIR / fn
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  copied {fn} → data/")
        else:
            print(f"  [skip] {fn} not found in nature_paper/")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("── Export step 1: splits")
    splits_df = export_splits()

    print("── Export step 2: predictions")
    export_predictions()

    print("── Export step 3: episodes")
    export_episodes()

    print("── Export step 4: UMAP embeddings")
    export_umap(splits_df)

    print("── Export step 5: cluster frequencies (reads bag .pt files — slow)")
    export_cluster_freqs(splits_df)

    print("── Export step 6: clinical features")
    export_clinical(splits_df)

    print("── Export step 7: clinical attention (loads model — needs GPU)")
    export_clinical_attention(splits_df)

    print("── Export step 8: benchmark summary")
    export_benchmark()

    print("── Export step 9: nature analysis outputs")
    export_nature_analysis()

    print("\nDone → data/")
    for f in sorted(DATA_DIR.glob("*")):
        kb = f.stat().st_size // 1024
        print(f"  {f.name}  ({kb} KB)")
