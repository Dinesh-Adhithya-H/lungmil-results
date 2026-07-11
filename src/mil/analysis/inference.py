"""
Model inference: build per-stem metadata and run forward passes.

v8 TWO-PHASE TRAINING DESIGN:
  Phase 1 — Each modality trained independently: backbone + ABMIL compresses
             raw patch features into compact summary tokens that are already
             predictive per modality (hinge + Cox + optional CLR).
  Phase 2 — Multimodal fusion: takes Phase 1 summary tokens from all present
             modalities and combines them with a fusion variant (early / middle /
             late / crossmodal / iterative) for joint label prediction.
  Checkpoints live under: {endpoint}/{task}/ckpts_{variant}/best_model.pt
  Phase 1 checkpoints: {endpoint}/phase1/{modality}/best_model.pt
"""
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ENDPOINT, VARIANT_TAGS
from .io import save_cache, load_cache
from .enrich import enrich_all


def _load_v7(chicago_mil_dir: Path):
    src = chicago_mil_dir / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    if str(chicago_mil_dir) not in sys.path:
        sys.path.insert(0, str(chicago_mil_dir))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_mm_abmil_v7", chicago_mil_dir / "train_mm_abmil_v7.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tag_to_base_variant(tag: str):
    """Map ckpt dir tag back to build_model_v7 variant name.

    Ckpt dirs follow _variant_tag() in train_mm_abmil_v7.py:
      "early"  / "late" / "middle"  → same string
      "crossattn_k8" / "crossmodal_k8" → strip _k{n}
      "iterative_r2_k8"              → strip _r{r}_k{n}
    """
    import re
    # strip _cls suffix
    cls_suffix = tag.endswith("_cls")
    base = tag[:-4] if cls_suffix else tag
    # iterative_rN_kN
    m = re.fullmatch(r"iterative_r\d+_k\d+", base)
    if m:
        return "iterative" + ("_cls" if cls_suffix else "")
    # crossattn_kN / crossmodal_kN
    m = re.fullmatch(r"(crossattn|crossmodal)_k\d+", base)
    if m:
        return m.group(1) + ("_cls" if cls_suffix else "")
    # early / late / middle / self_attn (or _cls variants)
    if base in ("early", "late", "middle", "self_attn"):
        return base + ("_cls" if cls_suffix else "")
    return None


# Preferred v8 subdir per endpoint (most informative multitask checkpoint)
_V8_PREFERRED_SUBDIR = {
    "acr":   ["both", "both_alt", "cls"],
    "clad":  ["clad_surv", "surv"],
    "death": ["death_surv", "surv"],
}


def _build_model(tv7, tag: str, ckpt_file: Path, device, task: str = "both"):
    import torch
    variant = _tag_to_base_variant(tag)
    if variant is None:
        raise ValueError(f"Unknown tag: {tag}")
    model = tv7.build_model_v7(variant, task=task)
    ckpt  = torch.load(str(ckpt_file), map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def _build_meta(splits_csv: Path, endpoint: str) -> Tuple[Dict, pd.DataFrame]:
    """Per-stem metadata: ACR gap-time TTE + endpoint-specific TTE/event."""
    cfg = ENDPOINT[endpoint]
    df  = pd.read_csv(str(splits_csv))
    df["anchor_dt"] = pd.to_datetime(df["anchor_dt"])

    # ACR gap-time TTE (for all endpoints — useful as context)
    acr_mask = df["acr_grade"].apply(
        lambda g: isinstance(g, str) and (g.startswith("A1") or g.startswith("A2"))
    )
    acr_dates: Dict = {}
    for _, row in df[acr_mask].iterrows():
        acr_dates.setdefault(row["patient_id"], []).append(row["anchor_dt"])
    last_date = df.groupby("patient_id")["anchor_dt"].max().to_dict()

    meta: Dict = {}
    for _, row in df.iterrows():
        stem = Path(str(row["file"])).stem
        pid  = row["patient_id"]
        t    = row["anchor_dt"]

        # ACR gap-time
        future = sorted([d for d in acr_dates.get(pid, []) if d > t])
        if future:
            tte_acr, ev_acr = float((future[0] - t).days), 1
        else:
            last = last_date.get(pid, t)
            tte_acr, ev_acr = float(max((last - t).days, 0)), 0
        acr_lbl_str = str(row.get("acr_grade", ""))
        acr_lbl = (0.0 if acr_lbl_str.startswith("A0") else
                   1.0 if (acr_lbl_str.startswith("A1") or acr_lbl_str.startswith("A2"))
                   else float("nan"))

        # Endpoint-specific TTE/event
        try:
            ep_event = float(row[cfg["event_col"]])
            ep_time  = float(row[cfg["time_col"]])
            if math.isnan(ep_event):
                ep_time = float("nan"); ep_event = float("nan")
            elif ep_event == 0:
                ep_time = float("nan")   # censoring time not stored; keep event=0
            elif math.isnan(ep_time) or ep_time < 0:
                ep_time = float("nan"); ep_event = float("nan")
        except (KeyError, TypeError, ValueError):
            ep_time = float("nan"); ep_event = float("nan")

        meta[stem] = {
            "patient_id":     pid,
            "anchor_dt":      t,
            "acr_label":      acr_lbl,
            "tte_next_acr":   tte_acr,
            "ev_next_acr":    float(ev_acr),
            cfg["tte_key"]:   ep_time,
            cfg["ev_key"]:    ep_event,
        }
    return meta, df


def _run_fold(tv7, results_dir: Path, split: int, fold: int,
              splits_csv: Path, samples_dir: Path, device,
              endpoint: str) -> List[Dict]:
    """Run inference for one fold, return list of per-sample dicts."""
    import torch
    cfg       = ENDPOINT[endpoint]
    fold_dir  = results_dir / f"split{split}_fold{fold}"
    if not fold_dir.exists():
        return []

    split_col = f"split{split}_fold{fold}"
    df_csv    = pd.read_csv(str(splits_csv))
    df_csv["anchor_dt"] = pd.to_datetime(df_csv["anchor_dt"])
    stem_to_ds = {
        Path(str(r["file"])).stem: str(r.get(split_col, ""))
        for _, r in df_csv.iterrows()
        if str(r.get(split_col, "")) in ("train", "val", "test")
    }
    stems = list(stem_to_ds.keys())
    if not stems:
        return []

    meta, _ = _build_meta(splits_csv, endpoint)
    print(f"[infer] split={split} fold={fold}: loading {len(stems)} bags …")
    bag_cache = tv7.preload_bags(stems, str(samples_dir))

    # Locate ckpts: v7 places them directly in fold_dir; v8 places them one level deeper.
    # For v8, prefer the subdir that trains the endpoint's multitask model.
    ckpt_dirs = sorted(fold_dir.glob("ckpts_*"))
    if not ckpt_dirs:
        preferred = _V8_PREFERRED_SUBDIR.get(endpoint, [])
        for sub in preferred:
            cands = sorted((fold_dir / sub).glob("ckpts_*"))
            if cands:
                ckpt_dirs = cands
                break
        if not ckpt_dirs:
            ckpt_dirs = sorted(fold_dir.glob("*/ckpts_*"))

    rows: List[Dict] = []
    for ckpt_dir in ckpt_dirs:
        tag = ckpt_dir.name[len("ckpts_"):]
        if _tag_to_base_variant(tag) is None:
            continue
        ckpt_file = ckpt_dir / "best_model.pt"
        if not ckpt_file.exists():
            print(f"[infer]   {tag}: no ckpt — skip"); continue
        # Infer task from parent subdir name (v8) or default to "both" (v7)
        parent_name = ckpt_dir.parent.name
        task = parent_name if not parent_name.startswith("split") else "both"
        try:
            model = _build_model(tv7, tag, ckpt_file, device, task=task)
        except Exception as e:
            print(f"[infer]   {tag}: build failed: {e}"); continue

        with torch.no_grad():
            for stem in stems:
                bags = {m: bag_cache.get(stem, {}).get(m) for m in tv7.MODALITIES}
                bags["HE_coords"] = bag_cache.get(stem, {}).get("HE_coords")
                if all(v is None for k, v in bags.items() if k != "HE_coords"):
                    continue
                try:
                    out = model(bags, device)
                    # DualGatedPool returns (logit, hazard, r_cls, r_tte) tuple
                    # MultiTaskHead returns dict {task_name: (scalar, rep)}
                    if isinstance(out, dict):
                        vals = list(out.values())
                        logit  = vals[0][0]
                        hazard = vals[-1][0]
                        r_cls  = vals[0][1]
                        r_tte  = vals[-1][1]
                    elif isinstance(out, tuple) and len(out) >= 4:
                        logit, hazard, r_cls, r_tte = out[0], out[1], out[2], out[3]
                    else:
                        continue
                    import torch as _t
                    prob = float(_t.sigmoid(logit.float()).item())
                    haz  = float(hazard.float().item())
                    m    = meta.get(stem, {})
                    rows.append({
                        "stem":       stem,
                        "variant":    tag,
                        "patient_id": m.get("patient_id"),
                        "anchor_dt":  m.get("anchor_dt"),
                        "split":      split,
                        "fold":       fold,
                        "data_split": stem_to_ds[stem],
                        "cls_prob":   prob,
                        "hazard":     haz,
                        "acr_label":  m.get("acr_label"),
                        "tte_next_acr": m.get("tte_next_acr"),
                        "ev_next_acr":  m.get("ev_next_acr"),
                        cfg["tte_key"]: m.get(cfg["tte_key"]),
                        cfg["ev_key"]:  m.get(cfg["ev_key"]),
                        "rep_cls":    r_cls.detach().float().cpu().numpy(),
                        "rep_tte":    r_tte.detach().float().cpu().numpy(),
                    })
                except Exception as exc:
                    pass
        del model
        if hasattr(device, "type") and device.type == "cuda":
            torch.cuda.empty_cache()
        tag_rows = [r for r in rows if r["variant"] == tag and r["split"] == split and r["fold"] == fold]
        print(f"[infer]   {tag}: {len(tag_rows)} samples")
    return rows


def collect_variant_data(
    results_dir: Path,
    splits_csv:  Path,
    samples_dir: Path,
    splits:      List[int],
    folds:       List[int],
    endpoint:    str,
    device_str:  str = "cpu",
    chicago_mil_dir: Optional[Path] = None,
) -> Optional[Dict]:
    """Run inference across all splits/folds; return variant_data dict."""
    import torch
    chicago_mil = chicago_mil_dir or results_dir.parent.parent
    print(f"[infer] Endpoint:    {endpoint}")
    print(f"[infer] Results dir: {results_dir}")
    try:
        tv7 = _load_v7(chicago_mil)
    except Exception as e:
        print(f"[infer] Failed to load v7: {e}"); return None

    device = torch.device(device_str if (device_str == "cpu" or torch.cuda.is_available()) else "cpu")
    print(f"[infer] Device: {device}")

    all_rows: List[Dict] = []
    for s in splits:
        for f in folds:
            all_rows.extend(_run_fold(tv7, results_dir, s, f,
                                      splits_csv, samples_dir, device, endpoint))

    if not all_rows:
        print("[infer] No data collected."); return None

    result: Dict = {}
    by_tag: Dict[str, List] = {}
    for r in all_rows:
        by_tag.setdefault(r["variant"], []).append(r)

    for tag, rows in by_tag.items():
        reps_cls  = np.stack([r.pop("rep_cls") for r in rows])
        reps_tte  = np.stack([r.pop("rep_tte")  for r in rows])
        df        = pd.DataFrame(rows)
        result[tag] = {"df": df, "reps_cls": reps_cls, "reps_tte": reps_tte}
        print(f"[infer] {tag}: {len(df)} samples, dim={reps_cls.shape[1]}")

    return result


def get_or_run(
    results_dir: Path,
    splits_csv:  Path,
    samples_dir: Path,
    splits:      List[int],
    folds:       List[int],
    endpoint:    str,
    output_dir:  Path,
    device_str:  str = "cpu",
    chicago_mil_dir: Optional[Path] = None,
) -> Optional[Dict]:
    """Load cached inference or run fresh, then enrich."""
    cache_path = output_dir / f"inference_cache_{endpoint}"
    vd = load_cache(cache_path)
    if vd is None:
        vd = collect_variant_data(results_dir, splits_csv, samples_dir,
                                   splits, folds, endpoint, device_str, chicago_mil_dir)
        if vd:
            save_cache(vd, cache_path)
    if vd:
        enrich_all(vd, splits_csv, endpoint)
    return vd
