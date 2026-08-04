"""
Unimodal ablation for longitudinal_mk_no_alibi and longitudinal_mk_mt_no_alibi.

For each variant × split × task:
  1. If top-level metrics JSON already has unimodal_ablation → skip.
  2. Else if phase2 JSON has it (LongMK acr_surv/death_surv from prior run) → propagate.
  3. Else run inference with one modality at a time and compute the metric.

Writes unimodal_ablation into:
  results/mm_abmil_v8/metrics_split{s}_fold0_{variant}_{suffix}.json
"""
import argparse, json, math, sys
from pathlib import Path

import torch
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from mil.models.builders import build_model_v8
from mil.data.loader import preload_bags
from mil.data.splits import build_splits_longitudinal
from mil.training.metrics import compute_metrics
from mil.training.losses import c_index

MODALITIES  = ["HE", "BAL", "CT", "Clinical"]
SAMPLES_DIR = Path("/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples")
SPLITS_CSV  = Path("/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")
RESULTS     = REPO / "results/mm_abmil_v8"
PHASE2      = RESULTS / "phase2"

SLOT_K         = 16
N_CROSS_LAYERS = 1
MODAL_DROPOUT  = 0.3
MAX_HE_PATCHES = 4096

# dir_suffix = checkpoint subdir name; task_key passed to build_model_v8; longi_key = model output key
TASK_CFG = {
    "acr_cls":   {"dir_suffix": "cls",       "task_key": "cls",       "longi_key": "acr_cls",  "metric": "bacc"},
    "acr_surv":  {"dir_suffix": "acr_surv",  "task_key": "acr_surv",  "longi_key": "acr_surv", "metric": "c_index"},
    "clad_surv": {"dir_suffix": "clad_surv", "task_key": "clad_surv", "longi_key": "clad",     "metric": "c_index"},
    "death_surv":{"dir_suffix": "death_surv","task_key": "death_surv","longi_key": "death",     "metric": "c_index"},
}


@torch.no_grad()
def run_ablation(model, patient_records, device, bag_cache, task_cfg):
    """Run model with one modality at a time. Returns {mod: {n, metric_val}}."""
    longi_key = task_cfg["longi_key"]
    metric    = task_cfg["metric"]
    use_amp   = device.type == "cuda"
    results   = {}

    for active_mod in MODALITIES:
        if longi_key == "acr_cls":
            probs, labels = [], []
        else:
            hazards, times, events = [], [], []

        for pat in patient_records:
            bags_list = []
            has_mod = False
            for s in pat["stems"]:
                entry = bag_cache.get(s, {})
                b = {m: None for m in MODALITIES}
                b["HE_coords"] = None
                if entry.get(active_mod) is not None:
                    b[active_mod] = entry[active_mod]
                    if active_mod == "HE":
                        b["HE_coords"] = entry.get("HE_coords")
                    has_mod = True
                bags_list.append(b)

            if not has_mod:
                continue

            try:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model({"bags_list": bags_list, "days": pat["days"],
                                 "records": pat["records"]}, device)
            except (torch.cuda.OutOfMemoryError, Exception):
                torch.cuda.empty_cache()
                continue

            if not isinstance(out, dict):
                continue

            if longi_key == "acr_cls":
                for logit, label in (out.get("acr_cls") or []):
                    if isinstance(logit, torch.Tensor):
                        probs.append(torch.sigmoid(logit.float()).item())
                        labels.append(label)
            elif longi_key == "acr_surv":
                res = out.get("acr_surv")
                if isinstance(res, tuple) and len(res) == 4:
                    h, _, t, e = res
                    if isinstance(h, torch.Tensor) and not math.isnan(t):
                        hazards.append(h.float().item()); times.append(t); events.append(e)
            else:
                for h, t, e in (out.get(longi_key) or []):
                    if isinstance(h, torch.Tensor):
                        hazards.append(h.float().item()); times.append(t); events.append(e)

        if longi_key == "acr_cls":
            n = len(probs)
            entry = {"n": n}
            if probs and labels:
                try:
                    m = compute_metrics(np.array(labels), np.array(probs))
                    entry["bacc"] = m.get("bacc"); entry["auc"] = m.get("auc")
                except Exception:
                    pass
        else:
            n = len(hazards)
            entry = {"n": n}
            if n >= 2 and sum(events) > 0:
                entry["c_index"] = c_index(hazards, times, events)

        results[active_mod] = entry
        val = entry.get(metric, None)
        print(f"    {active_mod}: n={n}  {metric}={val:.4f if isinstance(val,float) else '—'}")

    return results


def try_propagate_phase2(variant, split, task_name, dir_suffix):
    """Return existing unimodal_ablation from phase2 JSON, or None."""
    p = PHASE2 / f"split{split}_fold0/{variant}_{dir_suffix}/metrics_{variant}_final.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    abl = d.get("unimodal_ablation")
    if not abl:
        return None
    # Verify it has actual data
    if any(v.get("n", 0) > 0 for v in abl.values()):
        print(f"  [propagate] {p.name} → top-level JSON")
        return abl
    return None


def process(variant, split, task_name, device, bag_cache, patient_test):
    cfg        = TASK_CFG[task_name]
    dir_suffix = cfg["dir_suffix"]
    top_json   = RESULTS / f"metrics_split{split}_fold0_{variant}_{dir_suffix}.json"

    if not top_json.exists():
        print(f"  [skip] top-level JSON missing: {top_json.name}"); return

    existing = json.loads(top_json.read_text())
    if existing.get("unimodal_ablation"):
        print(f"  [skip] already has unimodal_ablation: {top_json.name}"); return

    # Try propagate from phase2
    abl = try_propagate_phase2(variant, split, task_name, dir_suffix)

    if abl is None:
        # Run inference
        ckpt_dir = PHASE2 / f"split{split}_fold0/{variant}_{dir_suffix}"
        ckpt     = ckpt_dir / f"model_{variant}_final.pt"
        if not ckpt.exists():
            print(f"  [skip] checkpoint missing: {ckpt}"); return

        print(f"  [infer] {variant} split{split} {task_name}")
        model = build_model_v8(
            variant=variant,
            slot_k=SLOT_K,
            n_cross_layers=N_CROSS_LAYERS,
            task=cfg["task_key"],
            modal_dropout=MODAL_DROPOUT,
            max_he_patches=MAX_HE_PATCHES,
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()

        abl = run_ablation(model, patient_test, device, bag_cache, cfg)
        del model; torch.cuda.empty_cache()

    existing["unimodal_ablation"] = abl
    top_json.write_text(json.dumps(existing, indent=2))
    print(f"  [saved] {top_json.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits",   type=int, nargs="+", default=list(range(5)))
    ap.add_argument("--variants", nargs="+",
                    default=["longitudinal_mk_no_alibi", "longitudinal_mk_mt_no_alibi"])
    ap.add_argument("--tasks",    nargs="+",
                    default=["acr_cls", "acr_surv", "clad_surv", "death_surv"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for split in args.splits:
        print(f"\n=== split {split} ===")
        long_splits = build_splits_longitudinal(
            samples_dir=SAMPLES_DIR, splits_csv=SPLITS_CSV, fold=0, split=split)
        patient_test = long_splits["test"]
        stems        = list({r["stem"] for pat in patient_test for r in pat["records"]})
        print(f"  test patients={len(patient_test)}  stems={len(stems)}")
        bag_cache = preload_bags(stems, SAMPLES_DIR, n_workers=4)

        for variant in args.variants:
            for task in args.tasks:
                print(f"\n  [{variant}] {task}")
                process(variant, split, task, device, bag_cache, patient_test)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
