"""
Patch unimodal_ablation in set_mil fold1 JSONs with full clad+death c_index.

Existing fold1 ablation only has bacc + acr c_index (surv_ep was "acr" only).
This script loads the fold1 checkpoint and runs evaluate_unimodal_ablation for
clad and death endpoints, then merges into the existing ablation dict.

Run via: sbatch scripts/submit_slotattn_ablation.sh
"""
import argparse, json, sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from mil.models.builders import build_model_v8
from mil.data.loader import preload_bags
from mil.data.splits import build_splits_multitask
from mil.training.phase2_trainer import evaluate_unimodal_ablation

SAMPLES_DIR = Path("/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples")
SPLITS_CSV  = Path("/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")
RESULTS_DIR = REPO / "results/mm_abmil_v8/phase2"

SLOT_K         = 16
N_CROSS_LAYERS = 1
MODAL_DROPOUT  = 0.3


def process_split(split, device):
    # Use fold1 since fold0 doesn't exist yet
    fold = 1
    save_dir = RESULTS_DIR / f"split{split}_fold{fold}/set_mil_mega"
    ckpt     = save_dir / "model_set_mil_final.pt"
    json_out = save_dir / "metrics_set_mil_final.json"

    if not ckpt.exists():
        print(f"[split{split}] checkpoint missing, skip"); return
    if not json_out.exists():
        print(f"[split{split}] metrics JSON missing, skip"); return

    existing = json.loads(json_out.read_text())
    existing_abl = existing.get("unimodal_ablation", {})

    # Check if clad and death are already patched
    already_done = all(
        existing_abl.get(m, {}).get("clad_c_index") is not None
        for m in ["HE", "BAL", "CT", "Clinical"]
        if existing_abl.get(m, {}).get("n", 0) > 0
    )
    if already_done and any(existing_abl.get(m, {}).get("clad_c_index") is not None for m in ["HE","BAL","CT","Clinical"]):
        print(f"[split{split}] clad/death ablation already present, skip"); return

    print(f"\n[split{split}] Loading splits (fold{fold})...")
    splits = build_splits_multitask(SAMPLES_DIR, SPLITS_CSV, fold=fold, split=split)
    test_recs = splits["test"]
    print(f"  test records: {len(test_recs)}")

    print(f"[split{split}] Preloading bags...")
    stems = list({r["stem"] for r in test_recs})
    bag_cache = preload_bags(stems, SAMPLES_DIR, n_workers=4)

    print(f"[split{split}] Building model and loading checkpoint...")
    model = build_model_v8(
        variant="set_mil",
        slot_k=SLOT_K,
        n_cross_layers=N_CROSS_LAYERS,
        task="mega",
        modal_dropout=MODAL_DROPOUT,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    # Run ablation for clad and death endpoints (acr already exists in fold1 JSON)
    merged_abl = {m: dict(existing_abl.get(m, {})) for m in ["HE","BAL","CT","Clinical"]}

    for ep in ("clad", "death"):
        print(f"  Running ablation: surv_endpoint={ep}")
        ep_abl = evaluate_unimodal_ablation(
            model, test_recs, device, bag_cache,
            surv_endpoint=ep, task="mega"
        )
        for mod, vals in ep_abl.items():
            if mod not in merged_abl:
                merged_abl[mod] = {"n": vals.get("n", 0)}
            ci = vals.get("c_index")
            if ci is not None:
                merged_abl[mod][f"{ep}_c_index"] = ci
                print(f"    {mod}: {ep}_c_index={ci:.3f}  (n={vals.get('n',0)})")

    # Rename existing "c_index" → "acr_c_index" for consistency with new format
    for mod in merged_abl:
        if "c_index" in merged_abl[mod] and "acr_c_index" not in merged_abl[mod]:
            merged_abl[mod]["acr_c_index"] = merged_abl[mod].pop("c_index")

    existing["unimodal_ablation"] = merged_abl
    json_out.write_text(json.dumps(existing, indent=2))
    print(f"[split{split}] Patched {json_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, nargs="+", default=list(range(5)))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for s in args.splits:
        process_split(s, device)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
