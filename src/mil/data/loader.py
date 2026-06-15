"""Bag (.pt file) loading with thread-parallel I/O."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional

import torch

from .registry import MODALITIES, _feat_key


def _load_one_bag(args):
    """Load a single stem's .pt file. Runs in a thread-pool worker."""
    stem, path = args
    entry = {m: None for m in MODALITIES}
    if not path.exists():
        return stem, entry
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [warn] load failed {path.name}: {e}", flush=True)
        return stem, entry
    inp = data.get("inputs", {})

    for mod in MODALITIES:
        if mod == "Clinical":
            coh = data.get("clinical_onehot")
            if coh is not None and isinstance(coh, torch.Tensor) and coh.numel() > 0:
                entry["Clinical"] = coh.float(); continue
            t = inp.get("Clinical")
            if t is not None and isinstance(t, torch.Tensor) and t.numel() > 0:
                if t.dtype == torch.float16: t = t.float()
                if t.dim() == 1: t = t.unsqueeze(0)
                entry["Clinical"] = t
        else:
            t = inp.get(_feat_key(mod))
            if t is None or not isinstance(t, torch.Tensor) or t.numel() == 0:
                continue
            if t.dtype == torch.float16: t = t.float()
            if t.dim() == 1: t = t.unsqueeze(0)
            entry[mod] = t

    for coords_key in ("HE_coords", "CT_coords"):
        coords_t = inp.get(coords_key)
        if coords_t is not None and isinstance(coords_t, torch.Tensor) and coords_t.numel() > 0:
            if coords_t.dtype == torch.float16: coords_t = coords_t.float()
            entry[coords_key] = coords_t

    raw_coh = data.get("cluster_count_onehot") or {}
    for mod, agg_key in [("HE","HE_cells"),("BAL","BAL_cells"),("CT","CT_cells")]:
        coh = raw_coh.get(agg_key)
        if coh is not None and isinstance(coh, torch.Tensor) and coh.numel() > 0:
            entry[f"{mod}_count_onehot"] = coh.float()

    cfn = data.get("clinical_feature_names")
    if cfn is not None:
        entry["_clinical_feature_names"] = cfn

    raw_cn = data.get("cluster_names") or {}
    for mod, agg_key in [("HE","HE_cells"),("BAL","BAL_cells"),("CT","CT_cells")]:
        cn = raw_cn.get(agg_key)
        if cn is not None:
            entry[f"_{mod}_cluster_names"] = cn

    inp.clear()
    if hasattr(data, "clear"): data.clear()
    return stem, entry


def preload_bags(stems, samples_dir, n_workers: int = 8) -> Dict:
    """Load all bag .pt files in parallel using a thread pool."""
    sd           = Path(samples_dir)
    stems_sorted = sorted(stems)
    args         = [(s, sd / f"{s}.pt") for s in stems_sorted]

    cache: Dict         = {}
    n_loaded            = {m: 0 for m in MODALITIES}
    total_patches       = {m: 0 for m in MODALITIES}

    print(f"  Preloading {len(stems_sorted)} bags with {n_workers} threads ...")
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = {pool.submit(_load_one_bag, a): a[0] for a in args}
        done = 0
        for fut in as_completed(futs):
            stem, entry = fut.result()
            cache[stem] = entry
            for mod in MODALITIES:
                t = entry.get(mod)
                if t is not None:
                    n_loaded[mod] += 1
                    total_patches[mod] += t.shape[0]
            done += 1
            if done % 200 == 0:
                mb = sum(t.numel() * 4 / 1e6 for e in cache.values()
                         for t in e.values() if isinstance(t, torch.Tensor))
                print(f"    preload {done}/{len(stems_sorted)}  "
                      f"{'  '.join(f'{m}={n_loaded[m]}' for m in MODALITIES)}  "
                      f"RAM={mb:.0f}MB", flush=True)

    mb = sum(t.numel() * 4 / 1e6 for e in cache.values()
             for t in e.values() if isinstance(t, torch.Tensor))
    for mod in MODALITIES:
        avg = total_patches[mod] / max(n_loaded[mod], 1)
        print(f"  {mod:10s}: files={n_loaded[mod]}  "
              f"patches={total_patches[mod]}  avg={avg:.0f}")
    print(f"  Total RAM: {mb:.0f} MB")
    return cache
