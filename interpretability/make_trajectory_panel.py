"""
Select 4 representative patients from split-2 test fold and assemble
a publication-quality trajectory panel figure (Fig. 7).

Patients selected:
  A — non-survivor (death_event=1): rising hazard predicts death
  B — long-term survivor (death_event=0, long censored): flat/low hazard
  C — ACR episode patient: detects acute rejection spike
  D — CLAD patient (clad_event=1): progressive hazard rise

Run via sbatch interpretability/submit_trajectory_panel.sh
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
INTERP_DIR = ROOT / "interpretability" / "longitudinal_mk_interp"
DEATH_DIR  = INTERP_DIR / "longitudinal_mk_no_alibi_split2_fold0_death_surv"
SURV_DIR   = INTERP_DIR / "longitudinal_mk_no_alibi_split2_fold0_acr_surv"
PREDS_CSV  = ROOT / "results" / "predictions" / "raw" / "split2_predictions.csv"
OUT_DIR    = ROOT / "figures" / "trajectories"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS_CSV = Path("/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")

# ── Load split-2 test patients with outcomes ──────────────────────────────────
# Outcomes come from predictions CSV (has death_time, clad_event, etc.)
preds = pd.read_csv(PREDS_CSV)
preds_test = preds[preds["split"] == 2].copy()

# Also load splits CSV for patient_id and acr_grade (not in predictions CSV)
splits = pd.read_csv(SPLITS_CSV)
test2 = splits[splits["split2_fold0"] == "test"].copy()
test2["file_str"] = test2["file"].astype(str).str.zfill(5)

# ACR grade info from splits CSV
acr_info = (
    test2.groupby("patient_id")
    .agg(
        n_acr_pos=("acr_grade", lambda x: x.isin(["A1","A2","A1B1","A2B2","A1B0","A2B0"]).sum()),
        n_visits =("file", "count"),
    )
    .reset_index()
)

# Outcomes from predictions CSV — get per-patient max (survival time is same for all visits of a patient)
outcome_cols = [c for c in ["death_time","death_event","clad_time","clad_event"] if c in preds_test.columns]
pid_cols = [c for c in ["patient_id","stem"] if c in preds_test.columns]
if not pid_cols:
    # fall back: use file/stem column
    preds_test["patient_id_tmp"] = preds_test.index  # placeholder

if "patient_id" in preds_test.columns and outcome_cols:
    pid_outcomes_raw = (
        preds_test.groupby("patient_id")[outcome_cols]
        .max()
        .reset_index()
    )
else:
    # Build from splits CSV — outcomes stored in .pt files, read from what we have
    pid_outcomes_raw = test2[["patient_id"]].drop_duplicates()
    for col in ["death_time","death_event","clad_time","clad_event"]:
        pid_outcomes_raw[col] = np.nan

pid_outcomes = pid_outcomes_raw.merge(acr_info, on="patient_id", how="left")
pid_outcomes["n_visits"] = pid_outcomes["n_visits"].fillna(1).astype(int)
pid_outcomes["n_acr_pos"] = pid_outcomes["n_acr_pos"].fillna(0).astype(int)

# Available patients with L4 death plots
avail_death = {p.stem.replace("L4_hazard_trajectory_pid", ""): p
               for p in sorted(DEATH_DIR.glob("L4_hazard_trajectory_pid*.png"))}
avail_surv  = {p.stem.replace("L4_hazard_trajectory_pid", ""): p
               for p in sorted(SURV_DIR.glob("L4_hazard_trajectory_pid*.png"))}

pid_outcomes["has_death_plot"] = pid_outcomes["patient_id"].isin(avail_death)

print(pid_outcomes[["patient_id","death_event","death_time","clad_event","n_acr_pos","n_visits","has_death_plot"]]
      .sort_values("death_time").to_string())

# ── Patient selection logic ───────────────────────────────────────────────────
avail = pid_outcomes[pid_outcomes["has_death_plot"]]

# A: non-survivor — died, 4+ visits (enough trajectory), earliest death
non_survivors = avail[avail["death_event"] == 1].sort_values("n_visits", ascending=False)
patient_A = non_survivors.iloc[0]["patient_id"] if len(non_survivors) else None

# B: long survivor — censored, longest follow-up, many visits
survivors = avail[(avail["death_event"] == 0) & (avail["n_visits"] >= 4)].sort_values("death_time", ascending=False)
patient_B = survivors.iloc[0]["patient_id"] if len(survivors) else None

# C: ACR episode — most ACR+ biopsies
acr_pts = avail[avail["n_acr_pos"] >= 2].sort_values("n_acr_pos", ascending=False)
# Prefer one not already selected
acr_pts = acr_pts[~acr_pts["patient_id"].isin([patient_A, patient_B])]
patient_C = acr_pts.iloc[0]["patient_id"] if len(acr_pts) else None

# D: CLAD patient — clad_event=1, not already selected
clad_pts = avail[(avail["clad_event"] == 1)].sort_values("clad_time")
clad_pts = clad_pts[~clad_pts["patient_id"].isin([patient_A, patient_B, patient_C])]
patient_D = clad_pts.iloc[0]["patient_id"] if len(clad_pts) else None

selected = [
    ("A", patient_A, "Non-survivor (death event)",         avail_death),
    ("B", patient_B, "Long-term survivor (censored)",       avail_death),
    ("C", patient_C, "Recurrent ACR episodes",              avail_surv),
    ("D", patient_D, "CLAD onset",                         avail_death),
]

print("\nSelected patients:")
for label, pid, desc, _ in selected:
    info = pid_outcomes[pid_outcomes["patient_id"] == pid]
    if pid is None:
        print(f"  {label}: None ({desc})")
    else:
        row = info.iloc[0]
        print(f"  {label}: {pid}  — {desc}  "
              f"(death={row['death_event']:.0f}, TTE={row['death_time']:.0f}d, "
              f"clad={row['clad_event']:.0f}, ACR+={row['n_acr_pos']:.0f}, visits={row['n_visits']:.0f})")

# ── Assemble 2×2 panel ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    "Per-patient longitudinal risk trajectories\n"
    "Longitudinal-MK model, split 2 test fold (C-index = 0.843 for death, 0.748 for ACR survival)",
    fontsize=13, fontweight="bold", y=1.01
)

label_colors = {"A": "#c0392b", "B": "#2980b9", "C": "#e67e22", "D": "#8e44ad"}

for ax, (panel_label, pid, desc, plot_dir) in zip(axes.flat, selected):
    if pid is None or pid not in plot_dir:
        ax.text(0.5, 0.5, f"Panel {panel_label}\n{desc}\n(No plot available)",
                ha="center", va="center", fontsize=11, transform=ax.transAxes)
        ax.axis("off")
        continue

    img_path = plot_dir[pid]
    img = mpimg.imread(str(img_path))
    ax.imshow(img)
    ax.axis("off")

    # Overlay panel label
    ax.text(0.02, 0.97, panel_label, transform=ax.transAxes,
            fontsize=18, fontweight="bold", va="top",
            color=label_colors.get(panel_label, "black"),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=label_colors.get(panel_label, "gray"), lw=1.5))

    # Subtitle
    row = pid_outcomes[pid_outcomes["patient_id"] == pid]
    if len(row):
        r = row.iloc[0]
        subtitle = (f"{pid} — {desc}\n"
                    f"Death: {'yes' if r['death_event'] else 'no (censored)'}, "
                    f"Follow-up: {r['death_time']:.0f} d, "
                    f"CLAD: {'yes' if r['clad_event'] else 'no'}, "
                    f"ACR+ visits: {r['n_acr_pos']:.0f}")
        ax.set_title(subtitle, fontsize=9, pad=4)

plt.tight_layout()

out_png = OUT_DIR / "Fig7_patient_trajectories.png"
out_pdf = OUT_DIR / "Fig7_patient_trajectories.pdf"
fig.savefig(out_png, dpi=200, bbox_inches="tight")
fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")
plt.close()

# ── Also copy the individual PNGs to figures/trajectories ────────────────────
import shutil
for panel_label, pid, desc, plot_dir in selected:
    if pid and pid in plot_dir:
        src = plot_dir[pid]
        dst = OUT_DIR / f"panel_{panel_label}_{pid}.png"
        shutil.copy(src, dst)
        print(f"Copied: {dst.name}")
