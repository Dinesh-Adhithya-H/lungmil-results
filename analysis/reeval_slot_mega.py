#!/usr/bin/env python3
"""
Re-evaluate completed slot_mega_tss folds to add clad_c_index and death_c_index.

Loads saved best model weights, runs p2_evaluate for all three survival endpoints,
and overwrites metrics_slot.json with the complete metrics (preserving existing keys).

Usage (via SLURM):
  python analysis/reeval_slot_mega.py --fold 1
  python analysis/reeval_slot_mega.py --fold 2
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from mil.data.loader import preload_bags
from mil.data.splits import load_splits
from mil.models.builders import build_model_v8
from mil.training.phase2_trainer import p2_evaluate, evaluate_unimodal_ablation

SAMPLES_DIR = "/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples"
SPLITS_CSV  = "/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
RESULTS_DIR = Path(__file__).parent.parent / "results/mm_abmil_v8"
SLOT_K      = 8
TASK        = "mega"
VTAG        = "slot"
SPLIT       = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--slot-k", type=int, default=SLOT_K)
    args = ap.parse_args()

    fold     = args.fold
    save_dir = RESULTS_DIR / f"phase2/split{SPLIT}_fold{fold}/slot_mega_tss"
    mf       = save_dir / f"metrics_{VTAG}.json"
    best_pt  = save_dir / f"model_{VTAG}.pt"

    if not best_pt.exists():
        print(f"No model found at {best_pt}. Job not finished?"); return
    if not mf.exists():
        print(f"No metrics file at {mf}. Job not finished?"); return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Re-evaluating fold={fold} on {device}")

    # Load splits
    splits = load_splits(SPLITS_CSV, split=SPLIT)
    train_recs = splits[fold]["train"]
    val_recs   = splits[fold]["val"]
    test_recs  = splits[fold]["test"]
    all_stems  = list({r["stem"] for r in train_recs + val_recs + test_recs})

    print(f"  Loading bags ({len(all_stems)} stems)...")
    bag_cache = preload_bags(all_stems, SAMPLES_DIR, n_workers=4)

    # Build model and load best weights
    model = build_model_v8(variant="slot", slot_k=args.slot_k, task=TASK).to(device)
    state = torch.load(best_pt, map_location="cpu", weights_only=False)
    state = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state, strict=False)
    model.eval()
    del state
    print("  Model loaded.")

    # Load existing metrics (preserve probs/labels/unimodal_ablation)
    with open(mf) as f:
        all_metrics = json.load(f)

    # Re-evaluate all splits for clad and death C-indices
    for sn, recs in [("train", train_recs), ("val", val_recs), ("test", test_recs)]:
        for ep_name in ("clad", "death"):
            _, _, _, ci_ep, *_ = p2_evaluate(model, recs, device, bag_cache,
                                              surv_endpoint=ep_name, task=TASK)
            key = f"{ep_name}_c_index"
            if ci_ep is not None:
                all_metrics.setdefault(sn, {})[key] = float(ci_ep)
                print(f"  [{sn}] {key} = {ci_ep:.4f}")
            else:
                print(f"  [{sn}] {key} = N/A")

    # Save updated metrics
    with open(mf, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Saved updated metrics → {mf}")


if __name__ == "__main__":
    main()
