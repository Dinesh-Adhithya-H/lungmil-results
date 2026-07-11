"""
Extract per-sample predictions on the test set of a given outer split.

Design
------
Each patient appears in exactly one split's test set (5-split exhaustive design),
so running this for splits 0-4 gives full cohort coverage with no cross-split
averaging. Each sample gets exactly one test prediction per model.

P1 (unimodal ABMIL):  4 modalities × 4 tasks = 16 models per split
P2 (multimodal):      loaded for every (variant, task) that has a saved
                      model_*_final.pt — skips silently if not yet done

P2 task→architecture mapping (from TASK_GROUPS in builders.py):
  cls        → ["acr_cls"]              → MultiTaskHead  → dict
  acr_surv   → ["acr_cls","acr_surv"]  → DualGatedPool  → tuple (logit,hazard,r,r)
  clad_surv  → ["clad"]                → MultiTaskHead  → dict
  death_surv → ["death"]               → MultiTaskHead  → dict
  mega       → ["acr_cls","acr_surv","clad","death"] → MultiTaskHead → dict

LongitudinalMIL (longitudinal_mk) takes patient_data (bags_list, days, records)
not a flat bags dict — handled as a special case.

Outputs
-------
results/predictions/raw/split{S}_predictions.csv
  One row per biopsy (test set of that split).

  Columns:
    stem, patient_id, anchor_dt, acr_grade, label, split
    has_HE, has_BAL, has_CT, has_Clinical  (modality presence flags)
    tte_next_acr, event_next_acr, clad_time, clad_event, death_time, death_event

    P1 per-modality (for each modality m ∈ {HE,BAL,CT,Clinical}):
      p1_acr_{m}       ACR classification sigmoid prob
      h1_acr_{m}       ACR survival hazard
      h1_clad_{m}      CLAD survival hazard
      h1_death_{m}     Death survival hazard

    P2 classification prob (populated when model exists):
      p2_early_cls, p2_late_cls, p2_middle_cls
      p2_set_mil_mega, p2_longitudinal_mk_mega

    P2 survival hazards (populated when model exists):
      h2_acr_{variant}_{task}, h2_clad_{variant}_{task}, h2_death_{variant}_{task}
      for each (variant, task) that has a final model
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from mil.data.registry import MODALITIES, _feat_dim
from mil.data.splits import build_splits_multitask
from mil.data.loader import preload_bags
from mil.models.phase1 import SingleModalMIL
from mil.models.builders import build_model_v8

HIDDEN_DIM = 256
DROPOUT    = 0.4

# All P2 (variant, task) combinations — models loaded if file exists
P2_ALL = [
    ("early",          "cls"),
    ("early",          "acr_surv"),
    ("early",          "clad_surv"),
    ("early",          "death_surv"),
    ("late",           "cls"),
    ("late",           "acr_surv"),
    ("late",           "clad_surv"),
    ("late",           "death_surv"),
    ("middle",         "cls"),
    ("middle",         "acr_surv"),
    ("middle",         "clad_surv"),
    ("middle",         "death_surv"),
    ("set_mil",   "mega"),
    ("longitudinal_mk","mega"),
]

# P1 task dirs → what to extract
P1_TASKS = {
    "acr":      "classification",   # logit → sigmoid = ACR pred prob + hazard
    "acr_surv": "acr_survival",     # hazard_head only
    "clad":     "clad_survival",
    "death":    "death_survival",
}


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_p1(split, p1_task, modality, results_dir, device):
    """Load P1 SingleModalMIL. Prefer final_combined, fall back to final."""
    for sub in ("final_combined", "final"):
        ckpt = (results_dir
                / f"phase1/split{split}_fold0/{p1_task}/{modality}/{sub}/best_model.pt")
        if ckpt.exists():
            model = SingleModalMIL(feat_dim=_feat_dim(modality),
                                   hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state)
            model.to(device).eval()
            return model
    return None


def _load_p2(split, variant, task, results_dir, device):
    """Load P2 final model (combined-train on best HP). Returns None if not done."""
    model_path = (results_dir
                  / f"phase2/split{split}_fold0/{variant}_{task}"
                  / f"model_{variant}_final.pt")
    if not model_path.exists():
        return None
    model = build_model_v8(variant, slot_k=16, n_cross_layers=1,
                           modal_dropout=0.0, task=task)
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"  [P2] Loaded {variant}/{task}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _run_p1(model, bag, device):
    """
    Single-sample P1 inference.
    Returns (acr_prob, acr_hazard) — acr_prob from classification head,
    acr_hazard from hazard_head. Both from the same SingleModalMIL forward.
    """
    if bag is None or bag.shape[0] == 0:
        return None, None
    try:
        logit, extras = model(bag.to(device), return_extras=True)
        return (torch.sigmoid(logit.float()).item(),
                extras["hazard"].float().item())
    except Exception as e:
        print(f"  [P1] error: {e}")
        return None, None


@torch.no_grad()
def _run_p2(model, bags, device, variant, task):
    """
    Single-sample P2 inference. Returns dict:
      acr_prob, h_acr_surv, h_clad, h_death
    Handles all P2 output formats:
      - dict  (MultiTaskHead): cls, clad_surv, death_surv, mega
      - tuple (DualGatedPool): acr_surv (task_list=["acr_cls","acr_surv"])
    LongitudinalMIL is handled separately (different input signature).
    """
    out = {"acr_prob": None, "h_acr": None, "h_clad": None, "h_death": None}
    if variant == "longitudinal_mk":
        # LongitudinalMIL takes patient_data, not bags — skip in single-biopsy mode
        # (run via _run_longitudinal_patient for full patient batches)
        return out
    try:
        result = model(bags, device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"  [P2] OOM {variant}/{task}")
        return out
    except Exception as e:
        print(f"  [P2] error {variant}/{task}: {e}")
        return out

    if isinstance(result, torch.Tensor):
        return out

    if isinstance(result, tuple):
        # DualGatedPool: (logit, hazard, r_cls, r_surv)
        logit, hazard = result[0], result[1]
        if isinstance(logit, torch.Tensor):
            out["acr_prob"] = torch.sigmoid(logit.float()).item()
        if isinstance(hazard, torch.Tensor):
            out["h_acr"]    = hazard.float().item()
        return out

    if isinstance(result, dict):
        # MultiTaskHead: {task_name: (scalar, rep)}
        def _extract(key):
            v = result.get(key)
            if v is None:
                return None
            h = v[0] if isinstance(v, (list, tuple)) else v
            return h.float().item() if isinstance(h, torch.Tensor) else None

        acr_logit = _extract("acr_cls")
        out["acr_prob"] = (torch.sigmoid(torch.tensor(acr_logit)).item()
                           if acr_logit is not None else None)
        out["h_acr"]    = _extract("acr_surv")
        out["h_clad"]   = _extract("clad")
        out["h_death"]  = _extract("death")

    return out


@torch.no_grad()
def _run_longitudinal_patient(model, patient_bags_list, patient_days,
                               patient_records, device):
    """
    Run LongitudinalMIL on a single patient's full biopsy history.
    Returns list of per-biopsy dicts with acr_prob, h_acr, h_clad, h_death.
    Each dict is aligned with patient_records (same index = same biopsy).
    """
    default = {"acr_prob": None, "h_acr": None, "h_clad": None, "h_death": None}
    n = len(patient_bags_list)
    defaults = [default.copy() for _ in range(n)]

    try:
        days_t = torch.tensor(patient_days, dtype=torch.float32, device=device)
        result = model({"bags_list": patient_bags_list, "days": days_t,
                        "records": patient_records}, device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return defaults
    except Exception as e:
        print(f"  [LMK] error: {e}")
        return defaults

    if isinstance(result, torch.Tensor) or not isinstance(result, dict):
        return defaults

    # acr_surv: (hazard_scalar, rep) — patient-level
    acr_surv_h = None
    acr_surv = result.get("acr_surv")
    if isinstance(acr_surv, (list, tuple)) and len(acr_surv) >= 1:
        h = acr_surv[0]
        if isinstance(h, torch.Tensor):
            acr_surv_h = h.float().item()

    # acr_cls: [(logit, label), ...] — per labeled biopsy; map back by biopsy index
    acr_cls_map = {}  # stem -> prob
    acr_cls_out = result.get("acr_cls", [])
    if isinstance(acr_cls_out, list):
        for item in acr_cls_out:
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                logit = item[0]
                if isinstance(logit, torch.Tensor) and hasattr(logit, 'item'):
                    # We can't easily map back to biopsy without stem, so store prob
                    acr_cls_map[len(acr_cls_map)] = torch.sigmoid(logit.float()).item()

    # clad / death: [(hazard, t, e), ...] — per biopsy
    def _per_biopsy_hazards(key):
        items = result.get(key, [])
        if not isinstance(items, list):
            return []
        return [item[0].float().item() if isinstance(item, (list, tuple))
                and len(item) >= 1 and isinstance(item[0], torch.Tensor)
                else None
                for item in items]

    clad_hazards  = _per_biopsy_hazards("clad")
    death_hazards = _per_biopsy_hazards("death")

    out_list = []
    for i in range(n):
        out_list.append({
            "acr_prob": acr_cls_map.get(i),
            "h_acr":    acr_surv_h,               # patient-level, same for all biopsies
            "h_clad":   clad_hazards[i] if i < len(clad_hazards)  else None,
            "h_death":  death_hazards[i] if i < len(death_hazards) else None,
        })
    return out_list


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split",       type=int, required=True)
    ap.add_argument("--samples-dir", default="/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples")
    ap.add_argument("--splits-csv",  default="/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")
    ap.add_argument("--results-dir", default=str(REPO / "results/mm_abmil_v8"))
    ap.add_argument("--out-dir",     default=str(REPO / "results/predictions/raw"))
    ap.add_argument("--workers",     type=int, default=8)
    ap.add_argument("--overwrite",   action="store_true")
    args = ap.parse_args()

    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir);  out_dir.mkdir(parents=True, exist_ok=True)
    out_path    = out_dir / f"split{args.split}_predictions.csv"
    print(f"Split {args.split}  device={device}")

    if out_path.exists() and not args.overwrite:
        print(f"Already exists: {out_path}  (--overwrite to redo)")
        return

    # ── Metadata ──────────────────────────────────────────────────────────────
    df_meta = pd.read_csv(args.splits_csv, parse_dates=["anchor_dt"])
    df_meta["stem"] = df_meta["file"].apply(lambda x: Path(str(x)).stem)

    # ── Records ───────────────────────────────────────────────────────────────
    splits    = build_splits_multitask(args.samples_dir, args.splits_csv,
                                       fold=0, split=args.split)
    test_recs = splits.get("test", [])
    print(f"Test records: {len(test_recs)}")
    stem_to_rec = {r["stem"]: r for r in test_recs}

    # ── Bags ──────────────────────────────────────────────────────────────────
    bag_cache = preload_bags([r["stem"] for r in test_recs], args.samples_dir,
                             n_workers=args.workers)

    def _bags_for(stem):
        b = {m: bag_cache.get(stem, {}).get(m) for m in MODALITIES}
        b["HE_coords"] = bag_cache.get(stem, {}).get("HE_coords")
        return b

    # ── Load P1 models: 4 tasks × 4 modalities ────────────────────────────────
    print("\nLoading P1 models...")
    p1_models = {}  # (p1_task, modality) -> model | None
    for p1_task in P1_TASKS:
        for mod in MODALITIES:
            m = _load_p1(args.split, p1_task, mod, results_dir, device)
            p1_models[(p1_task, mod)] = m
            status = "OK" if m else "MISSING"
            print(f"  P1 {p1_task}/{mod}: {status}")

    # ── Load P2 models ─────────────────────────────────────────────────────────
    print("\nLoading P2 models...")
    p2_models = {}  # (variant, task) -> model | None
    for variant, task in P2_ALL:
        m = _load_p2(args.split, variant, task, results_dir, device)
        p2_models[(variant, task)] = m

    # ── Longitudinal: group test records by patient ────────────────────────────
    lmk_model = p2_models.get(("longitudinal_mk", "mega"))
    lmk_preds = {}  # stem -> {acr_prob, h_acr, h_clad, h_death}
    if lmk_model is not None:
        print("\nRunning LongitudinalMIL inference (patient-grouped)...")
        # group by patient_id, sort by anchor_dt
        from collections import defaultdict
        patient_stems = defaultdict(list)
        for rec in test_recs:
            patient_stems[rec["patient_id"]].append(rec["stem"])

        pid_to_anchors = (df_meta[df_meta["stem"].isin(stem_to_rec.keys())]
                          .set_index("stem")["anchor_dt"].to_dict())

        for pid, stems in patient_stems.items():
            stems_sorted = sorted(stems,
                                  key=lambda s: pid_to_anchors.get(s, pd.Timestamp(0)))
            if not stems_sorted:
                continue
            first_day = pid_to_anchors.get(stems_sorted[0], pd.Timestamp(0))
            days = [(pid_to_anchors.get(s, first_day) - first_day).days
                    for s in stems_sorted]
            bags_list = [_bags_for(s) for s in stems_sorted]
            records   = [stem_to_rec[s] for s in stems_sorted]

            per_biopsy = _run_longitudinal_patient(
                lmk_model, bags_list, days, records, device)
            for stem, pred in zip(stems_sorted, per_biopsy):
                lmk_preds[stem] = pred

    # ── Per-sample inference ───────────────────────────────────────────────────
    print("\nRunning per-sample inference...")
    rows = []
    for i, rec in enumerate(test_recs):
        stem = rec["stem"]
        bags = _bags_for(stem)

        row = {
            "stem":           stem,
            "patient_id":     rec.get("patient_id", stem),
            "label":          rec.get("label"),
            "split":          args.split,
            "tte_next_acr":   rec.get("tte_next_acr",   float("nan")),
            "event_next_acr": rec.get("event_next_acr", float("nan")),
            "clad_time":      rec.get("clad_time",  float("nan")),
            "clad_event":     rec.get("clad_event", float("nan")),
            "death_time":     rec.get("death_time",  float("nan")),
            "death_event":    rec.get("death_event", float("nan")),
        }
        for mod in MODALITIES:
            row[f"has_{mod}"] = bags.get(mod) is not None

        # ── P1: all 4 tasks × all 4 modalities ──────────────────────────────
        for p1_task in P1_TASKS:
            for mod in MODALITIES:
                m = p1_models.get((p1_task, mod))
                bag = bags.get(mod)
                if m is None or bag is None:
                    if p1_task == "acr":
                        row[f"p1_acr_{mod}"] = float("nan")
                    row[f"h1_{p1_task}_{mod}"] = float("nan")
                    continue
                prob, hazard = _run_p1(m, bag, device)
                if p1_task == "acr":
                    row[f"p1_acr_{mod}"] = prob   if prob   is not None else float("nan")
                # All P1 models have hazard_head; for acr task this is ACR hazard
                row[f"h1_{p1_task}_{mod}"] = hazard if hazard is not None else float("nan")

        # ── P2: all (variant, task) combinations ─────────────────────────────
        for variant, task in P2_ALL:
            if variant == "longitudinal_mk":
                pred = lmk_preds.get(stem, {})
                col_tag = f"{variant}_{task}"
                row[f"p2_{col_tag}"]      = pred.get("acr_prob",  float("nan")) or float("nan")
                row[f"h2_acr_{col_tag}"]  = pred.get("h_acr",     float("nan")) or float("nan")
                row[f"h2_clad_{col_tag}"] = pred.get("h_clad",    float("nan")) or float("nan")
                row[f"h2_death_{col_tag}"]= pred.get("h_death",   float("nan")) or float("nan")
                continue

            m = p2_models.get((variant, task))
            col_tag = f"{variant}_{task}"
            if m is None:
                row[f"p2_{col_tag}"]      = float("nan")
                row[f"h2_acr_{col_tag}"]  = float("nan")
                row[f"h2_clad_{col_tag}"] = float("nan")
                row[f"h2_death_{col_tag}"]= float("nan")
                continue

            p2out = _run_p2(m, bags, device, variant, task)
            row[f"p2_{col_tag}"]      = p2out["acr_prob"] if p2out["acr_prob"] is not None else float("nan")
            row[f"h2_acr_{col_tag}"]  = p2out["h_acr"]   if p2out["h_acr"]   is not None else float("nan")
            row[f"h2_clad_{col_tag}"] = p2out["h_clad"]  if p2out["h_clad"]  is not None else float("nan")
            row[f"h2_death_{col_tag}"]= p2out["h_death"] if p2out["h_death"] is not None else float("nan")

        rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(test_recs)}] {stem}", flush=True)

    df = pd.DataFrame(rows)
    meta_cols = df_meta[["stem", "anchor_dt", "acr_grade"]].drop_duplicates("stem")
    df = df.merge(meta_cols, on="stem", how="left")
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows  ({df['label'].notna().sum()} labeled)  →  {out_path}")


if __name__ == "__main__":
    main()
