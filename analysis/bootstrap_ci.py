"""
Bootstrap confidence intervals for test-fold metrics.

Two modes:
1. Within-split bootstrap CI for ACR classification (probs+labels stored in metrics JSON)
2. Across-split CI for all tasks (from the 5 split scalar values, t-distribution)

Run via sbatch:
  sbatch analysis/bootstrap_ci.sh
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from sklearn.metrics import balanced_accuracy_score

RESULTS_DIR = Path("/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/phase2")
N_BOOT = 2000
SEED = 42
rng = np.random.default_rng(SEED)


# ── per-split values from Supplementary Table 1 ──────────────────────────────
# Source: metrics_*_final.json across splits 0-4, best model per task
SPLIT_VALUES = {
    "ACR_cls": {
        "SetMIL-MT (no SAB)":      [0.578, 0.610, 0.680, 0.635, 0.615],
        "Linear baseline":          [0.664, 0.568, 0.620, 0.561, 0.526],
        "Early fusion":             [0.612, 0.599, 0.632, 0.472, 0.600],
        "Late fusion":              [0.594, 0.596, 0.640, 0.550, 0.580],
        "Middle fusion":            [0.522, 0.520, 0.616, 0.540, 0.599],
        "SetMIL-MT (SAB)":         [0.597, 0.546, 0.597, 0.605, 0.630],
        "SetMIL (single-task)":    [0.644, 0.564, 0.626, 0.601, 0.619],
        "Longitudinal-MK":         [0.546, 0.565, 0.510, 0.512, 0.615],
        "Longitudinal-MK-MT":      [0.460, 0.570, 0.504, 0.493, 0.602],
    },
    "ACR_surv": {
        "Longitudinal-MK":         [0.573, 0.673, 0.748, 0.660, 0.741],
        "Linear baseline":          [0.550, 0.614, 0.538, 0.527, 0.706],
        "Early fusion":             [0.550, 0.661, 0.598, 0.527, 0.540],
        "Late fusion":              [0.559, 0.665, 0.635, 0.525, 0.541],
        "Middle fusion":            [0.526, 0.613, 0.708, 0.466, 0.557],
        "SetMIL-MT (no SAB)":      [0.541, 0.668, 0.610, 0.509, 0.634],
        "SetMIL (single-task)":    [0.584, 0.596, 0.585, 0.523, 0.614],
        "Longitudinal-MK-MT":      [0.557, 0.539, 0.539, 0.690, 0.823],
    },
    "CLAD_surv": {
        "SetMIL-MT (SAB)":         [0.429, 0.616, 0.663, 0.577, 0.531],
        "Linear baseline":          [0.432, 0.622, 0.495, 0.460, 0.516],
        "Early fusion":             [0.432, 0.622, 0.495, 0.460, 0.516],
        "Late fusion":              [0.372, 0.583, 0.603, 0.553, 0.561],
        "Middle fusion":            [0.429, 0.610, 0.537, 0.470, 0.532],
        "SetMIL-MT (no SAB)":      [0.476, 0.619, 0.528, 0.469, 0.589],
        "Longitudinal-MK":         [0.461, 0.495, 0.516, 0.453, 0.520],
        "Longitudinal-MK-MT":      [0.721, 0.456, 0.523, 0.439, 0.533],
    },
    "Death_surv": {
        "Longitudinal-MK":         [0.779, 0.670, 0.843, 0.772, 0.793],
        "Linear baseline":          [0.541, 0.638, 0.560, 0.535, 0.625],
        "Early fusion":             [0.550, 0.640, 0.679, 0.635, 0.721],
        "Late fusion":              [0.551, 0.624, 0.693, 0.638, 0.685],
        "Middle fusion":            [0.555, 0.643, 0.707, 0.666, 0.711],
        "SetMIL-MT (SAB)":         [0.599, 0.646, 0.670, 0.725, 0.681],
        "SetMIL-MT (no SAB)":      [0.593, 0.650, 0.689, 0.662, 0.688],
        "SetMIL (single-task)":    [0.625, 0.671, 0.684, 0.695, 0.691],
        "Longitudinal-MK-MT":      [0.706, 0.628, 0.855, 0.815, 0.848],
    },
}


def across_split_ci(values, alpha=0.05):
    """95% CI on mean across N=5 splits using t-distribution."""
    arr = np.array(values)
    n = len(arr)
    mean = arr.mean()
    se = arr.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return mean, mean - t * se, mean + t * se


def bootstrap_bacc(probs, labels, n_boot=N_BOOT):
    """Bootstrap CI for BACC from stored probability array."""
    probs = np.array(probs)
    labels = np.array(labels)
    preds = (probs >= 0.5).astype(int)
    observed = balanced_accuracy_score(labels, preds)
    n = len(labels)
    boot_scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(labels[idx])) < 2:
            continue
        boot_scores.append(balanced_accuracy_score(labels[idx], preds[idx]))
    lo, hi = np.percentile(boot_scores, [2.5, 97.5])
    return observed, lo, hi


def load_cls_preds(variant_dir_name, split):
    """Load probs and labels from cls metrics JSON for within-split bootstrap."""
    path = RESULTS_DIR / f"split{split}_fold0" / variant_dir_name / f"metrics_{variant_dir_name.split('_cls')[0].split('_')[-1] if '_cls' not in variant_dir_name else variant_dir_name.replace('_cls','')}_final.json"
    # Try common naming patterns
    for candidate in RESULTS_DIR.glob(f"split{split}_fold0/*_cls/metrics_*_final.json"):
        d = json.loads(candidate.read_text())
        test = d.get("test", {})
        if "probs" in test and "labels" in test:
            return test["probs"], test["labels"], candidate.parent.name
    return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# 1. Across-split CI for all tasks and models
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("ACROSS-SPLIT 95% CI (t-distribution, N=5 splits)")
print("=" * 70)

for task, models in SPLIT_VALUES.items():
    print(f"\n── {task} ──")
    print(f"{'Model':<30} {'Mean':>6}  {'95% CI':>18}")
    print("-" * 58)
    for model, vals in models.items():
        mean, lo, hi = across_split_ci(vals)
        print(f"{model:<30} {mean:.3f}  ({lo:.3f} – {hi:.3f})")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Within-split bootstrap BACC for classification (best model per split)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("WITHIN-SPLIT BOOTSTRAP 95% CI — ACR CLASSIFICATION (N=2000 resamples)")
print("=" * 70)

CLS_DIRS = {
    "SetMIL-MT (no SAB)": "set_mil_mt_no_sab_cls",
    "SetMIL-MT (SAB)":    "set_mil_mt_cls",
    "Early fusion":        "early_cls",
    "Late fusion":         "late_cls",
    "Middle fusion":       "middle_cls",
    "Longitudinal-MK":    "longitudinal_mk_no_alibi_cls",
    "Longitudinal-MK-MT": "longitudinal_mk_mt_no_alibi_cls",
}

# Determine the variant tag from the directory name
def metrics_file(split, dir_name):
    folder = RESULTS_DIR / f"split{split}_fold0" / dir_name
    candidates = list(folder.glob("metrics_*_final.json")) if folder.exists() else []
    return candidates[0] if candidates else None

print(f"\n{'Model':<26} {'Split':>5}  {'BACC':>6}  {'95% CI':>18}")
print("-" * 62)

results_by_model = {}
for model_name, dir_name in CLS_DIRS.items():
    split_results = []
    for split in range(5):
        mf = metrics_file(split, dir_name)
        if mf is None:
            continue
        d = json.loads(mf.read_text())
        test = d.get("test", {})
        probs = test.get("probs")
        labels = test.get("labels")
        if probs is None or labels is None:
            continue
        obs, lo, hi = bootstrap_bacc(probs, labels)
        split_results.append((split, obs, lo, hi))
        print(f"{model_name:<26} s{split}     {obs:.3f}  ({lo:.3f} – {hi:.3f})")
    results_by_model[model_name] = split_results

# ─────────────────────────────────────────────────────────────────────────────
# 3. Save summary CSV
# ─────────────────────────────────────────────────────────────────────────────
out_dir = Path("/home/aih/dinesh.haridoss/chicago_mil/results/bootstrap_ci")
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
# Across-split CIs
for task, models in SPLIT_VALUES.items():
    for model, vals in models.items():
        mean, lo, hi = across_split_ci(vals)
        rows.append({"type": "across_split", "task": task, "model": model,
                     "split": "all", "mean": mean, "ci_lo": lo, "ci_hi": hi})

# Within-split cls bootstrap CIs
for model_name, split_results in results_by_model.items():
    for split, obs, lo, hi in split_results:
        rows.append({"type": "within_split_bootstrap", "task": "ACR_cls",
                     "model": model_name, "split": f"s{split}",
                     "mean": obs, "ci_lo": lo, "ci_hi": hi})

import csv
out_csv = out_dir / "bootstrap_ci_summary.csv"
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["type","task","model","split","mean","ci_lo","ci_hi"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\n\nSummary saved to: {out_csv}")
print("\nNOTE: Within-split bootstrap for SURVIVAL tasks requires re-running")
print("inference to recover raw hazard scores. Submit:")
print("  sbatch analysis/bootstrap_survival_ci.sh")
