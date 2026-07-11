"""
TCGA WSI-only survival benchmark dataset.

Loads patch features + spatial coordinates from H5 files.
Returns records for full-batch Cox trainer (same interface as lung pipeline).

Key: spatial coords are in the H5 files under 'coords_patching' (N, 2).
     These are REQUIRED for GeoMAE-SlotMIL (spatial graph attention).
     Standard methods (ABMIL, TransMIL) ignore coords and use features only.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
import torch

# ── Cancer configs ─────────────────────────────────────────────────────────────
CANCER_CONFIGS = {
    "KIRC": {
        "cache":  "/lustre/groups/aih/dinesh.haridoss/mil/tcga_cache_kirc",
        "h5_dir": "/lustre/groups/aih/dinesh.haridoss/mil/TCGA-KIRC",
    },
    "BRCA": {
        "cache":  "/lustre/groups/aih/dinesh.haridoss/mil/tcga_cache_brca",
        "h5_dir": "/lustre/groups/aih/dinesh.haridoss/mil/TCGA-BRCA",
    },
    "BLCA": {
        "cache":  "/lustre/groups/aih/dinesh.haridoss/mil/tcga_cache_blca",
        "h5_dir": "/lustre/groups/aih/dinesh.haridoss/mil/TCGA-BLCA",
    },
    "LGG":  {
        "cache":  "/lustre/groups/aih/dinesh.haridoss/mil/tcga_cache_lgg",
        "h5_dir": "/lustre/groups/aih/dinesh.haridoss/mil/TCGA-LGG",
    },
    "GBM":  {
        "cache":  "/lustre/groups/aih/dinesh.haridoss/mil/tcga_cache_gbm",
        "h5_dir": "/lustre/groups/aih/dinesh.haridoss/mil/TCGA-GBM",
    },
}
ALL_CANCERS = list(CANCER_CONFIGS.keys())
WSI_DIM     = 1536   # UNI feature dim from H5 files


# ── Splits ────────────────────────────────────────────────────────────────────

def _load_manifest(cache_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(cache_dir / "manifest.csv")
    df["os_status"] = pd.to_numeric(df["os_status"], errors="coerce").fillna(0)
    df["os_time"]   = pd.to_numeric(df["os_time"],   errors="coerce")
    return df.dropna(subset=["os_time"]).reset_index(drop=True)


def make_splits(cancer: str, n_folds: int = 5, seed: int = 42) -> list:
    """
    Stratified K-fold (by OS event). Each fold:
      train (64%) / val (16%) / test (20%).
    Returns list[dict] with keys 'train', 'val', 'test' → row indices.
    """
    from sklearn.model_selection import StratifiedKFold
    cache_dir = Path(CANCER_CONFIGS[cancer]["cache"])
    df  = _load_manifest(cache_dir)
    y   = df["os_status"].values.astype(int)
    idx = np.arange(len(df))

    splits = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for trainval_idx, test_idx in skf.split(idx, y):
        # further split trainval 80/20
        skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        tr, vl = next(skf2.split(trainval_idx, y[trainval_idx]))
        splits.append({
            "train": trainval_idx[tr].tolist(),
            "val":   trainval_idx[vl].tolist(),
            "test":  test_idx.tolist(),
        })
    return splits


def load_records(cancer: str, indices: list) -> list:
    """Load survival metadata for given patient indices."""
    cache_dir = Path(CANCER_CONFIGS[cancer]["cache"])
    df = _load_manifest(cache_dir)
    records = []
    for i in indices:
        row = df.iloc[i]
        records.append({
            "stem":     f"{int(row['idx']):05d}",
            "patient":  row["identifier"],
            "cancer":   cancer,
            "os_time":  float(row["os_time"]),
            "os_event": float(row["os_status"]),
        })
    return records


# ── H5 loading ────────────────────────────────────────────────────────────────

def _find_h5(h5_dir: Path, patient_id: str) -> Optional[Path]:
    """Find H5 file for a patient (filename starts with patient barcode)."""
    prefix = patient_id.upper()
    for h5 in h5_dir.glob(f"{prefix}*.h5"):
        return h5
    # Also try case-insensitive
    for h5 in h5_dir.glob("*.h5"):
        if h5.stem.upper().startswith(prefix):
            return h5
    return None


def _load_h5(h5_path: Path) -> Optional[Dict[str, torch.Tensor]]:
    """Load features (N, 1536) and coords (N, 2) from H5 file."""
    try:
        with h5py.File(h5_path, "r") as f:
            feats = torch.from_numpy(f["features"][0]).float()   # (N, 1536)
            coords = torch.from_numpy(
                f["coords_patching"][:]).float()                  # (N, 2)
        # Drop NaN
        ok = ~torch.isnan(feats).any(1)
        feats = feats[ok]; coords = coords[ok]
        return {"WSI": feats, "WSI_coords": coords}
    except Exception:
        return None


def preload_bags(cancer: str, records: list,
                 n_workers: int = 8) -> Dict[str, dict]:
    """
    Load all H5 files for given records into RAM.
    Returns {stem: {WSI: Tensor(N,1536), WSI_coords: Tensor(N,2)}}.
    """
    cfg    = CANCER_CONFIGS[cancer]
    h5_dir = Path(cfg["h5_dir"])

    def _load(rec):
        h5 = _find_h5(h5_dir, rec["patient"])
        if h5 is None:
            return rec["stem"], None
        entry = _load_h5(h5)
        return rec["stem"], entry

    bag_cache: Dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for stem, entry in ex.map(_load, records):
            if entry is not None:
                bag_cache[stem] = entry

    n_ok     = len(bag_cache)
    n_coords = sum(1 for v in bag_cache.values() if "WSI_coords" in v)
    avg_patches = int(np.mean([v["WSI"].shape[0]
                               for v in bag_cache.values()])) if bag_cache else 0
    print(f"  [{cancer}] loaded {n_ok}/{len(records)} bags  "
          f"coords={n_coords}  avg_patches={avg_patches:,}")
    return bag_cache
