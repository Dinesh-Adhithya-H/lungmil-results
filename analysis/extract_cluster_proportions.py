"""
Extract per-patient cluster proportions from instance_cluster_ids in bag .pt files.
Writes:
    results/cluster_proportions/he_cluster_prop.csv
    results/cluster_proportions/bal_cluster_prop.csv
    results/cluster_proportions/ct_cluster_prop.csv

Run via SLURM (submit_cluster_prop.sh) — needs ~60GB RAM for full bag set.
"""

from pathlib import Path
import sys, numpy as np, pandas as pd, torch
from concurrent.futures import ThreadPoolExecutor, as_completed

SPLITS_CSV  = "/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
SAMPLES_DIR = Path("/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples")
OUT_DIR     = Path("/home/aih/dinesh.haridoss/chicago_mil/results/cluster_proportions")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# modality key in instance_cluster_ids dict → output name
MOD_KEYS = {"HE": "HE_cells", "BAL": "BAL_cells", "CT": "CT_cells"}


def _process_stem(row):
    stem = str(row["stem"])
    pt   = SAMPLES_DIR / f"{stem}.pt"
    if not pt.exists():
        return stem, {}
    try:
        data = torch.load(pt, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [warn] {stem}: {e}")
        return stem, {}

    ici  = data.get("instance_cluster_ids") or {}
    cnames = data.get("cluster_names") or {}
    result = {}
    for mod, bag_key in MOD_KEYS.items():
        ids_t = ici.get(bag_key)
        if ids_t is None or not isinstance(ids_t, torch.Tensor):
            continue
        ids = ids_t.numpy().astype(int)
        if len(ids) == 0:
            continue
        K = int(ids.max()) + 1
        counts = np.bincount(ids, minlength=K).astype(float)
        total  = counts.sum()
        prop   = counts / total
        names  = cnames.get(bag_key)
        result[mod] = {"prop": prop, "K": K, "names": names, "n_instances": len(ids)}
    return stem, result


def main():
    splits = pd.read_csv(SPLITS_CSV, parse_dates=["anchor_dt"])
    splits["stem"] = splits["file"].str.replace(".pt", "", regex=False)
    # deduplicate stems
    splits["stem"] = splits["stem"].str.zfill(5)   # ensure zero-padded
    stems_df = splits[["stem", "patient_id", "anchor_dt", "acr_encoded",
                        "clad_status", "clad_days", "death_status", "death_days"]].drop_duplicates("stem")
    rows_by_mod = {mod: [] for mod in MOD_KEYS}
    K_max = {mod: 0 for mod in MOD_KEYS}
    names_by_mod = {mod: None for mod in MOD_KEYS}

    print(f"Processing {len(stems_df)} stems with 8 threads...")
    records = stems_df.to_dict("records")

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_process_stem, r): r for r in records}
        done = 0
        for fut in as_completed(futs):
            rec  = futs[fut]
            stem, result = fut.result()
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(records)}")
            for mod, info in result.items():
                K_max[mod] = max(K_max[mod], info["K"])
                if names_by_mod[mod] is None and info["names"] is not None:
                    names_by_mod[mod] = list(info["names"])
                rows_by_mod[mod].append({
                    "stem": stem,
                    "patient_id": rec["patient_id"],
                    "anchor_dt":  rec["anchor_dt"],
                    "acr_encoded": rec.get("acr_encoded"),
                    "clad_status": rec.get("clad_status"),
                    "death_status": rec.get("death_status"),
                    "n_instances": info["n_instances"],
                    "_prop": info["prop"],
                    "_K":    info["K"],
                })

    for mod, rows in rows_by_mod.items():
        if not rows:
            print(f"  [skip] {mod}: no data")
            continue
        K = K_max[mod]
        names = names_by_mod[mod] or [f"cluster_{k}" for k in range(K)]
        # pad names to K
        if len(names) < K:
            names = list(names) + [f"cluster_{i}" for i in range(len(names), K)]
        col_names = [f"cluster_{k}" for k in range(K)]

        out_rows = []
        for r in rows:
            prop = np.zeros(K, dtype=np.float32)
            p = r.pop("_prop")
            k = r.pop("_K")
            prop[:k] = p[:k]
            d = dict(r)
            for i, v in enumerate(prop):
                d[col_names[i]] = float(v)
            out_rows.append(d)

        df = pd.DataFrame(out_rows).sort_values(["patient_id", "anchor_dt"])
        out_path = OUT_DIR / f"{mod.lower()}_cluster_prop.csv"
        df.to_csv(out_path, index=False)
        print(f"  {out_path.name}: {len(df)} rows × {K} clusters")

        # also save cluster names mapping
        nm_df = pd.DataFrame({"idx": range(len(names)), "name": names})
        nm_df.to_csv(OUT_DIR / f"{mod.lower()}_cluster_names.csv", index=False)
        print(f"  {mod.lower()}_cluster_names.csv saved")


if __name__ == "__main__":
    main()
