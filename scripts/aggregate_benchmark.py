"""
aggregate_benchmark.py — Parse all metrics JSON files from v8 results
and produce benchmark_summary.csv for the patient explorer website.
Reads only from existing result JSON files (no GPU/model loading).
"""
import json, os, re
from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8")
OUT_CSV     = Path("/home/aih/dinesh.haridoss/chicago_mil/patient_explorer/data/benchmark_summary.csv")

rows = []

# ── 1. Per-fold DL metrics (split1_foldN_variant_task) ──────────────────────
pattern = re.compile(
    r"metrics_split(\d+)_fold(\d+)_(.+?)_(cls|acr_surv|clad_surv|death_surv|slot_mega)\.json"
)

for f in sorted(RESULTS_DIR.glob("metrics_*.json")):
    m = pattern.match(f.name)
    if not m:
        continue
    split, fold, variant, task = m.group(1), int(m.group(2)), m.group(3), m.group(4)

    try:
        d = json.loads(f.read_text())
    except Exception:
        continue

    test = d.get("test", {})
    row = {
        "model":     f"[dl] {variant}_{task}",
        "variant":   variant,
        "task":      task,
        "split":     int(split),
        "fold":      fold,
        "auc":       test.get("auc",    None),
        "auprc":     test.get("auprc",  None),
        "bacc":      test.get("bacc",   None),
        "c_index":   test.get("c_index",None),
        "mcc":       test.get("mcc",    None),
        "sens":      test.get("sens",   None),
        "spec":      test.get("spec",   None),
        "source":    "DL",
    }

    # unimodal ablation block inside the same JSON
    unimod = d.get("unimodal_ablation", {})
    for mod, stats in unimod.items():
        rows.append({
            "model":   f"[unimodal_abla] {mod}",
            "variant": mod,
            "task":    task,
            "split":   int(split),
            "fold":    fold,
            "auc":     stats.get("auc"),
            "auprc":   None,
            "bacc":    stats.get("bacc"),
            "c_index": None,
            "mcc":     None,
            "sens":    None,
            "spec":    None,
            "source":  "unimodal_ablation",
        })

    # unimodal_baselines block (per-modality DL baselines, not k-fold)
    unib = d.get("unimodal_baselines", {})
    for key, stats in unib.items():
        # key = "HE_acr", "HE_acr_surv", etc.
        parts = key.split("_", 1)
        if len(parts) == 2:
            mod_b, task_b = parts[0], parts[1]
        else:
            mod_b, task_b = key, task
        rows.append({
            "model":   f"[unimodal_dl] {mod_b}_{task_b}",
            "variant": mod_b,
            "task":    task_b,
            "split":   int(split),
            "fold":    fold,
            "auc":     stats.get("auc"),
            "auprc":   stats.get("auprc"),
            "bacc":    stats.get("bacc"),
            "c_index": stats.get("c_index"),
            "mcc":     stats.get("mcc"),
            "sens":    stats.get("sens"),
            "spec":    stats.get("spec"),
            "source":  "unimodal_dl",
        })

    rows.append(row)

# ── 2. Classical baselines summary ──────────────────────────────────────────
bsl_path = RESULTS_DIR / "baselines_summary.json"
if bsl_path.exists():
    bsl = json.loads(bsl_path.read_text())
    for model_name, stats in bsl.items():
        rows.append({
            "model":   model_name,
            "variant": "classical",
            "task":    "multi",
            "split":   1,
            "fold":    -1,     # aggregated
            "auc":     stats.get("test_auc"),
            "auprc":   None,
            "bacc":    stats.get("test_bacc"),
            "c_index": stats.get("test_ci_acr"),
            "mcc":     None,
            "sens":    None,
            "spec":    None,
            "source":  "classical",
        })

# ── 3. Phase-1 metrics (if available) ───────────────────────────────────────
p1_dir = RESULTS_DIR / "phase1"
if p1_dir.exists():
    for f in sorted(p1_dir.glob("**/*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        test = d.get("test", d)
        if not isinstance(test, dict):
            continue
        # extract fold from path
        parts = f.parts
        fold_str = next((p for p in parts if "fold" in p), "fold0")
        fold_n = int(re.search(r"fold(\d+)", fold_str).group(1)) if re.search(r"fold(\d+)", fold_str) else 0
        mod_task = f.stem.replace("metrics_", "")
        rows.append({
            "model":   f"[p1] {mod_task}",
            "variant": "phase1",
            "task":    mod_task,
            "split":   1,
            "fold":    fold_n,
            "auc":     test.get("auc"),
            "auprc":   test.get("auprc"),
            "bacc":    test.get("bacc"),
            "c_index": test.get("c_index"),
            "mcc":     test.get("mcc"),
            "sens":    test.get("sens"),
            "spec":    test.get("spec"),
            "source":  "phase1",
        })

df = pd.DataFrame(rows)
df = df[df["model"].notna()]
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"Saved {len(df)} rows → {OUT_CSV}")
print(df.groupby("source").size().to_string())
