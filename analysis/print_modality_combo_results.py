"""Print modality combo metric table to stdout."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT     = Path(__file__).resolve().parent.parent
REPS_DIR = ROOT / "results" / "mm_abmil_v8" / "biopsy_reps"
MASTER   = Path("/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")

MODS      = ["has_HE", "has_BAL", "has_CT", "has_Clinical"]
MOD_NAMES = ["HE", "BAL", "CT", "Clinical"]
MIN_N     = 20

TASKS = {
    "death_surv": ("cindex", "C-index"),
    "acr_surv":   ("cindex", "C-index"),
    "clad_surv":  ("cindex", "C-index"),
    "acr_cls":    ("bacc",   "BACC"),
}

def load_master():
    df = pd.read_csv(MASTER)
    df["stem"]  = df["file"].str.replace(".pt", "", regex=False)
    df["combo"] = df[MODS].apply(
        lambda r: "+".join(n for n, m in zip(MOD_NAMES, MODS) if r[m]) or "None", axis=1)
    return df.set_index("stem")["combo"].to_dict()

def load_reps(task, stem_to_combo):
    rows = []
    for s in range(5):
        p = REPS_DIR / f"biopsy_reps_{task}_split{s}.pt"
        if not p.exists(): continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        stems  = list(d["stems"])
        risk   = d["risk"].numpy()
        tte    = d["tte"].numpy()   if "tte"   in d else np.full(len(stems), np.nan)
        event  = d["event"].numpy() if "event" in d else np.full(len(stems), np.nan)
        label  = d["label"].numpy() if "label" in d else np.full(len(stems), np.nan)
        for i, stem in enumerate(stems):
            rows.append({"stem": stem, "risk": float(risk[i]),
                         "tte": float(tte[i]), "event": float(event[i]),
                         "label": float(label[i]),
                         "combo": stem_to_combo.get(stem, "Unknown")})
    return pd.DataFrame(rows)

def cindex(risk, tte, event):
    from sksurv.metrics import concordance_index_censored
    ev = event.astype(bool)
    if ev.sum() < 2: return np.nan
    c, *_ = concordance_index_censored(ev, tte, risk)
    return c

def bacc(risk, label, threshold):
    pred = (risk >= threshold).astype(int)
    lbl  = label.astype(int)
    classes = np.unique(lbl)
    recalls = [(pred[lbl == c] == c).mean() for c in classes if (lbl == c).sum() > 0]
    return float(np.mean(recalls))

stem_to_combo = load_master()

for task, (metric, mlabel) in TASKS.items():
    df = load_reps(task, stem_to_combo)
    if df.empty: continue

    counts = df["combo"].value_counts()
    shown  = counts[counts >= MIN_N].index.tolist()
    groups = [("All biopsies", df)] + [(c, df[df["combo"] == c]) for c in sorted(shown, key=lambda x: len(x.split("+")))]

    threshold = np.nanmedian(df["risk"].values) if metric == "bacc" else None

    print(f"\n{'='*55}")
    print(f"  {task}  ({mlabel})")
    print(f"{'='*55}")
    print(f"  {'Combo':<30}  {mlabel:>7}   n")
    print(f"  {'-'*48}")
    for combo, grp in groups:
        if metric == "cindex":
            mask = ~(np.isnan(grp["risk"]) | np.isnan(grp["tte"]) | np.isnan(grp["event"]))
            r, t, e = grp["risk"].values[mask], grp["tte"].values[mask], grp["event"].values[mask]
            if len(r) < 3 or e.astype(bool).sum() < 2: continue
            val = cindex(r, t, e)
        else:
            mask = ~(np.isnan(grp["risk"]) | np.isnan(grp["label"]))
            r, l = grp["risk"].values[mask], grp["label"].values[mask]
            if len(r) < 5 or len(np.unique(l)) < 2: continue
            val = bacc(r, l, threshold)
        print(f"  {combo:<30}  {val:>7.3f}   {int(mask.sum())}")
