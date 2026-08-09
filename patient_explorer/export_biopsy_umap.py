"""
Export per-biopsy UMAP embeddings from LongMK biopsy_reps .pt files.
Produces: patient_explorer/data/biopsy_umap_{task}.csv
  columns: patient_id, stem, biopsy_days, split, risk, tte, event, label, umap_x, umap_y

Run via: sbatch patient_explorer/submit_export_biopsy_umap.sh
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPS_DIR = ROOT / "results" / "mm_abmil_v8" / "biopsy_reps"
OUT_DIR  = ROOT / "patient_explorer" / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ["acr_cls", "acr_surv", "clad_surv", "death_surv"]


def load_task(task):
    rows = []
    for split in range(5):
        p = REPS_DIR / f"biopsy_reps_{task}_split{split}.pt"
        if not p.exists():
            print(f"  missing: {p.name}")
            continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        n = len(d["patient_ids"])
        for i in range(n):
            rows.append({
                "patient_id":  d["patient_ids"][i],
                "stem":        d["stems"][i],
                "biopsy_days": float(d["biopsy_days"][i]),
                "split":       split,
                "risk":        float(d["risk"][i]),
                "tte":         float(d["tte"][i]),
                "event":       float(d["event"][i]),
                "label":       float(d["label"][i]),
                "_rep":        d["reps"][i].numpy(),
            })
    return rows


def compute_umap(rows, n_neighbors=15, seed=42):
    import umap as umap_lib
    X = np.stack([r["_rep"] for r in rows]).astype(np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.clip(norms, 1e-8, None)
    reducer = umap_lib.UMAP(n_neighbors=n_neighbors, n_components=2,
                             metric="euclidean", random_state=seed)
    return reducer.fit_transform(X)


def main():
    for task in TASKS:
        print(f"\n=== {task} ===")
        rows = load_task(task)
        if not rows:
            print("  no data, skipping")
            continue
        print(f"  {len(rows)} biopsies — computing UMAP...")
        emb = compute_umap(rows)

        df = pd.DataFrame([{k: v for k, v in r.items() if k != "_rep"} for r in rows])
        df["umap_x"] = emb[:, 0]
        df["umap_y"] = emb[:, 1]

        out = OUT_DIR / f"biopsy_umap_{task}.csv"
        df.to_csv(out, index=False)
        print(f"  saved → {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
