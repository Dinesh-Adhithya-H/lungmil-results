"""Export SetMIL-MT per-visit predictions from results_raw.npy → setmilmt_preds.csv."""

from pathlib import Path
import numpy as np
import pandas as pd

NPY = Path("interpretability/set_mil_mt_interp/all_splits_merged/results_raw.npy")
SPLITS_CSV = Path("/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")
OUT_CSV = Path("patient_explorer/data/setmilmt_preds.csv")
FLAGGED_CSV = Path("interpretability/set_mil_mt_interp/all_splits_merged/flagged_patients.csv")

print(f"Loading {NPY} ...")
data = list(np.load(NPY, allow_pickle=True))
print(f"  {len(data)} records")

# Backfill anchor_dt / patient_id from splits CSV
df_splits = pd.read_csv(SPLITS_CSV)
df_splits["anchor_dt"] = pd.to_datetime(df_splits["anchor_dt"])
df_splits["stem"] = df_splits["file"].apply(lambda x: Path(str(x)).stem)
stem_to_adt = dict(zip(df_splits["stem"], df_splits["anchor_dt"]))
stem_to_pid = dict(zip(df_splits["stem"], df_splits["patient_id"].astype(str)))

for r in data:
    if r.get("anchor_dt") is None and r["stem"] in stem_to_adt:
        r["anchor_dt"] = stem_to_adt[r["stem"]]
    if r.get("patient_id") in (None, r["stem"]) and r["stem"] in stem_to_pid:
        r["patient_id"] = stem_to_pid[r["stem"]]

# Compute percentile ranks for survival tasks across all records
SURV_TASKS = ["acr_surv", "clad_surv", "death_surv"]
logit_arrays = {t: [] for t in SURV_TASKS}
stems_order = [r["stem"] for r in data]
for r in data:
    logits = r.get("logits", {})
    for t in SURV_TASKS:
        v = logits.get(t)
        logit_arrays[t].append(float(v) if v is not None else float("nan"))

pct_ranks = {}
for t in SURV_TASKS:
    arr = np.array(logit_arrays[t])
    valid = ~np.isnan(arr)
    ranks = np.full(len(arr), float("nan"))
    if valid.sum() > 0:
        order = np.argsort(arr[valid])
        pct = np.empty(valid.sum())
        pct[order] = np.linspace(0, 1, valid.sum())
        ranks[valid] = pct
    pct_ranks[t] = {stems_order[i]: ranks[i] for i in range(len(stems_order))}

# Load flagged patients
flagged_stems = set()
if FLAGGED_CSV.exists():
    df_flag = pd.read_csv(FLAGGED_CSV)
    flagged_stems = set(df_flag["stem"].astype(str))

# Build rows
rows = []
for r in data:
    stem = r["stem"]
    pid  = r.get("patient_id", stem)
    adt  = r.get("anchor_dt")
    logits = r.get("logits", {})

    acr_logit = logits.get("acr_cls")
    score_acr_cls = float(1.0 / (1.0 + np.exp(-acr_logit))) if acr_logit is not None else float("nan")

    rows.append({
        "stem":           stem,
        "patient_id":     pid,
        "anchor_dt":      adt,
        "score_acr_cls":  score_acr_cls,
        "pct_acr_surv":   pct_ranks["acr_surv"].get(stem, float("nan")),
        "pct_clad_surv":  pct_ranks["clad_surv"].get(stem, float("nan")),
        "pct_death_surv": pct_ranks["death_surv"].get(stem, float("nan")),
        "event_acr":      r.get("event_acr"),
        "tte_acr":        r.get("tte_acr"),
        "event_clad":     r.get("event_clad"),
        "tte_clad":       r.get("tte_clad"),
        "event_death":    r.get("event_death"),
        "tte_death":      r.get("tte_death"),
        "present_mods":   ",".join(sorted(r.get("present_mods", set()))),
        "flagged":        stem in flagged_stems,
    })

df = pd.DataFrame(rows)
df["anchor_dt"] = pd.to_datetime(df["anchor_dt"])
df = df.sort_values(["patient_id", "anchor_dt"]).reset_index(drop=True)

# Compute days_from_tx per patient
df["days_from_tx"] = df.groupby("patient_id")["anchor_dt"].transform(
    lambda x: (x - x.min()).dt.days
)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"Saved {len(df)} rows → {OUT_CSV}")
print(df.head(3).to_string())
