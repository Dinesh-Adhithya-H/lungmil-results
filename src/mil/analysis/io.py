"""Metrics loading, cache read/write, fold statistics."""
import json
import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import VARIANT_TAGS


def load_metrics_dir(results_dir: Path, folds: List[int]) -> Dict:
    """Load metrics_*.json from results_dir/split0_fold{f}/.
    Returns data[variant_tag][fold] = test_dict."""
    data: Dict[str, Dict] = {}
    for f in folds:
        # Try split1 first (v8), then split0 (v7 legacy)
        fold_dir = results_dir / f"split1_fold{f}"
        if not fold_dir.exists():
            fold_dir = results_dir / f"split0_fold{f}"
        if not fold_dir.exists():
            continue
        for mfile in sorted(list(fold_dir.glob("metrics_*.json")) +
                            list(fold_dir.glob("*/metrics_*.json"))):
            tag = mfile.stem[len("metrics_"):]
            try:
                raw = json.loads(mfile.read_text())
                test = raw.get("test", raw)
                # first-found wins — prevents later subdirs (surv/, cls/) overwriting
                # the primary task subdir (both/, clad_surv/, death_surv/)
                if f not in data.setdefault(tag, {}):
                    data[tag][f] = test
            except Exception:
                pass
    return data


def fold_stats(data: Dict, tag: str, metric: str) -> Tuple[float, float]:
    """Mean and std of metric across folds for one variant tag."""
    if tag not in data:
        return float("nan"), float("nan")
    vals = [float(fd[metric]) for fd in data[tag].values()
            if metric in fd and fd[metric] is not None and not math.isnan(float(fd[metric]))]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def ordered_tags(data: Dict) -> List[str]:
    order = [t for t in VARIANT_TAGS if t in data]
    order += [t for t in sorted(data) if t not in order]
    return order


# ── Inference cache ───────────────────────────────────────────────────────────

def save_cache(variant_data: Dict, cache_path: Path) -> None:
    arr_f = str(cache_path) + ".npz"
    pkl_f = str(cache_path) + "_meta.pkl"
    arrays, meta_dfs = {}, {}
    for tag, vd in variant_data.items():
        arrays[f"{tag}__reps_cls"]  = vd["reps_cls"]
        arrays[f"{tag}__reps_tte"]  = vd["reps_tte"]
        meta_dfs[tag] = vd["df"]
    np.savez_compressed(arr_f, **arrays)
    with open(pkl_f, "wb") as f:
        pickle.dump(meta_dfs, f, protocol=4)


def load_cache(cache_path: Path) -> Optional[Dict]:
    arr_f = str(cache_path) + ".npz"
    pkl_f = str(cache_path) + "_meta.pkl"
    if not (Path(arr_f).exists() and Path(pkl_f).exists()):
        return None
    arrays = np.load(arr_f, allow_pickle=True)
    with open(pkl_f, "rb") as f:
        meta_dfs = pickle.load(f)
    result = {}
    for tag, df in meta_dfs.items():
        cls_key = f"{tag}__reps_cls"
        tte_key = f"{tag}__reps_tte"
        # backward compat: old caches used __reps_surv
        if cls_key not in arrays:
            cls_key = f"{tag}__reps_surv" if f"{tag}__reps_surv" in arrays else None
        if tte_key not in arrays:
            tte_key = f"{tag}__reps_surv" if f"{tag}__reps_surv" in arrays else None
        if cls_key and tte_key:
            result[tag] = {"df": df,
                           "reps_cls": arrays[cls_key],
                           "reps_tte": arrays[tte_key]}
    print(f"[cache] Loaded {len(result)} variants from {cache_path}")
    return result
