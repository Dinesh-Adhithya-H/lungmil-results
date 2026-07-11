"""
Compute unimodal ablation for longitudinal_mk fold-0 models.

Loads the saved model_longitudinal_mk_final.pt for each split, runs the
multimodal model with one modality at a time on the test set, and patches
the unimodal_ablation key into metrics_longitudinal_mk_final.json.

Run via:  sbatch scripts/submit_longitudinal_ablation.sh
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

MODALITIES   = ["HE", "BAL", "CT", "Clinical"]
SAMPLES_DIR  = Path("/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples")
SPLITS_CSV   = Path("/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")
RESULTS_DIR  = REPO / "results/mm_abmil_v8/phase2"

# Model hyperparameters (from job script)
SLOT_K         = 16
N_CROSS_LAYERS = 1
MODAL_DROPOUT  = 0.3
MAX_HE_PATCHES = 4096


@torch.no_grad()
def ablate_longitudinal(model, patient_records, device, bag_cache):
    """Run longitudinal model with one modality at a time.

    Mirrors p2_evaluate_longitudinal but zeros out all modalities except
    active_mod in bags_list before each forward pass.

    Returns dict: {mod: {n, bacc, auc, acr_c_index, clad_c_index, death_c_index}}
    """
    use_amp = (device.type == "cuda")
    results = {}

    for active_mod in MODALITIES:
        cls_probs, cls_labels = [], []
        surv = {tk: {"h": [], "t": [], "e": []} for tk in ("acr_surv", "clad", "death")}

        for pat in patient_records:
            stems   = pat["stems"]
            days    = pat["days"]
            records = pat["records"]

            # Build bags_list with only active_mod; all others None
            bags_list = []
            has_mod = False
            for s in stems:
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
                continue  # patient has no data for this modality

            try:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    result = model({"bags_list": bags_list, "days": days, "records": records}, device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); continue
            except Exception as ex:
                continue

            if not isinstance(result, dict):
                continue

            # ACR classification
            cls_out = result.get("acr_cls", [])
            if not isinstance(cls_out, list): cls_out = []
            for logit, label in cls_out:
                if isinstance(logit, torch.Tensor):
                    cls_probs.append(torch.sigmoid(logit.float()).item())
                    cls_labels.append(label)

            # ACR survival
            acr_out = result.get("acr_surv")
            if acr_out is not None and isinstance(acr_out, tuple) and len(acr_out) == 4:
                hazard, _, acr_t, acr_e = acr_out
                if isinstance(hazard, torch.Tensor) and not math.isnan(acr_t):
                    surv["acr_surv"]["h"].append(hazard.float().item())
                    surv["acr_surv"]["t"].append(acr_t)
                    surv["acr_surv"]["e"].append(acr_e)

            # CLAD + Death
            for tk in ("clad", "death"):
                biopsy_hazards = result.get(tk, [])
                if not isinstance(biopsy_hazards, list): continue
                for hazard, t_val, e_val in biopsy_hazards:
                    if isinstance(hazard, torch.Tensor):
                        surv[tk]["h"].append(hazard.float().item())
                        surv[tk]["t"].append(t_val)
                        surv[tk]["e"].append(e_val)

        n_cls = len(cls_probs)
        n_surv = max(len(surv[tk]["h"]) for tk in surv)
        entry = {"n": max(n_cls, n_surv)}

        if cls_probs and cls_labels:
            try:
                m = compute_metrics(np.array(cls_labels), np.array(cls_probs))
                entry["bacc"] = m.get("bacc")
                entry["auc"]  = m.get("auc")
            except Exception:
                pass

        _tk_key = {"acr_surv": "acr_c_index", "clad": "clad_c_index", "death": "death_c_index"}
        for tk in ("acr_surv", "clad", "death"):
            sd = surv[tk]
            if len(sd["h"]) >= 2 and sum(sd["e"]) > 0:
                entry[_tk_key[tk]] = c_index(sd["h"], sd["t"], sd["e"])

        results[active_mod] = entry
        def _f(v): return f"{v:.3f}" if isinstance(v, float) else "—"
        print(f"    {active_mod}: n={entry['n']}  bacc={_f(entry.get('bacc'))}  "
              f"acr_ci={_f(entry.get('acr_c_index'))}  "
              f"clad_ci={_f(entry.get('clad_c_index'))}  "
              f"death_ci={_f(entry.get('death_c_index'))}")

    return results


def process_split(split, device):
    save_dir = RESULTS_DIR / f"split{split}_fold0/longitudinal_mk_mega"
    ckpt     = save_dir / "model_longitudinal_mk_final.pt"
    json_out = save_dir / "metrics_longitudinal_mk_final.json"

    if not ckpt.exists():
        print(f"[split{split}] checkpoint missing, skip"); return
    if not json_out.exists():
        print(f"[split{split}] metrics JSON missing, skip"); return

    existing = json.loads(json_out.read_text())
    # Overwrite if ablation is missing or was written empty (n=0 for all mods)
    existing_abl = existing.get("unimodal_ablation", {})
    if existing_abl and any(existing_abl.get(m, {}).get("n", 0) > 0 for m in MODALITIES):
        print(f"[split{split}] unimodal_ablation already present with data, skip"); return

    print(f"\n[split{split}] Loading test splits...")
    long_splits = build_splits_longitudinal(
        samples_dir=SAMPLES_DIR,
        splits_csv=SPLITS_CSV,
        fold=0,
        split=split,
    )
    patient_test = long_splits["test"]
    flat_test    = [r for pat in patient_test for r in pat["records"]]
    print(f"  test patients={len(patient_test)}  records={len(flat_test)}")

    print(f"[split{split}] Preloading bags...")
    stems = list({r["stem"] for r in flat_test})
    bag_cache = preload_bags(stems, SAMPLES_DIR, n_workers=4)

    print(f"[split{split}] Building model and loading checkpoint...")
    model = build_model_v8(
        variant="longitudinal_mk",
        slot_k=SLOT_K,
        n_cross_layers=N_CROSS_LAYERS,
        task="mega",
        modal_dropout=MODAL_DROPOUT,
        max_he_patches=MAX_HE_PATCHES,
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    print(f"[split{split}] Running unimodal ablation on test set...")
    abl = ablate_longitudinal(model, patient_test, device, bag_cache)

    existing["unimodal_ablation"] = abl
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
