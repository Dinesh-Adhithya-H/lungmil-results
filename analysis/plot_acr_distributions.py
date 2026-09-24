"""
ACR event distribution figures for PI review.
Panels: none/single/multiple, events per patient, time to first ACR.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT    = Path(__file__).resolve().parent.parent
CSV     = Path("/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")
OUT_DIR = ROOT / "figures" / "cohort"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BG = "#FAF6F2"
C_NEG  = "#90CAF9"   # ACR−  blue
C_POS  = "#EF5350"   # ACR+  red
C_CLAD = "#AB47BC"   # CLAD  purple
C_DEATH= "#546E7A"   # Death slate

FS_TITLE = 20
FS_LABEL = 16
FS_TICK  = 14
FS_ANNOT = 13

df = pd.read_csv(CSV)
df["anchor_dt"] = pd.to_datetime(df["anchor_dt"])

lab = df[df["label"].notna()].copy()

# ── Per-patient aggregation ───────────────────────────────────────────────────
per_pat = lab.groupby("patient_id").agg(
    n_biopsies    = ("label", "count"),
    n_acr_pos     = ("label", "sum"),
    first_biopsy  = ("anchor_dt", "min"),
    last_biopsy   = ("anchor_dt", "max"),
    clad_status   = ("clad_status", "first"),
    death_status  = ("death_status", "first"),
    clad_days     = ("clad_days", "first"),
    death_days    = ("death_days", "first"),
).reset_index()

per_pat["followup_days"] = (per_pat["last_biopsy"] - per_pat["first_biopsy"]).dt.days
per_pat["has_acr"]       = per_pat["n_acr_pos"] > 0

# Time to first ACR event
acr_first = lab[lab["label"] == 1].groupby("patient_id")["anchor_dt"].min().reset_index()
acr_first.columns = ["patient_id", "first_acr_dt"]
per_pat = per_pat.merge(acr_first, on="patient_id", how="left")
per_pat["days_to_first_acr"] = (per_pat["first_acr_dt"] - per_pat["first_biopsy"]).dt.days

acr_only = per_pat[per_pat["has_acr"]].copy()
days_to_acr = acr_only["days_to_first_acr"].dropna()

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor=BG)
fig.patch.set_facecolor(BG)
fig.suptitle(
    f"ACR Event Distributions  ·  {len(per_pat)} labeled patients  ·  "
    f"{int(lab['label'].sum())} ACR+ biopsies  ·  {int((lab['label']==0).sum())} ACR− biopsies",
    fontsize=18, fontweight="bold", y=1.01
)

# ── Panel A: None / Single / Multiple ─────────────────────────────────────────
ax = axes[0]
ax.set_facecolor(BG)
n_none  = (per_pat["n_acr_pos"] == 0).sum()
n_single= (per_pat["n_acr_pos"] == 1).sum()
n_multi = (per_pat["n_acr_pos"] >  1).sum()
cats  = ["None\n(0 ACR+)", "Single\n(1 ACR+)", "Multiple\n(≥2 ACR+)"]
vals  = [n_none, n_single, n_multi]
pcts  = [100 * v / len(per_pat) for v in vals]
cols  = [C_NEG, "#FFA726", C_POS]
bars  = ax.bar(cats, vals, color=cols, alpha=0.88, width=0.55,
               edgecolor="#333", linewidth=0.8, zorder=2)
for bar, v, p in zip(bars, vals, pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f"n={v}\n({p:.0f}%)", ha="center", va="bottom",
            fontsize=FS_ANNOT, fontweight="bold")
ax.set_ylabel("Number of patients", fontsize=FS_LABEL)
ax.set_title("ACR events per patient\n(among 199 labeled patients)", fontsize=FS_TITLE, fontweight="bold")
ax.tick_params(axis="both", labelsize=FS_TICK)
ax.set_ylim(0, n_none + 25)
ax.spines[["top", "right"]].set_visible(False)
ax.axhline(0, color="#ccc", linewidth=0.8)

# ── Panel B: Histogram — ACR+ biopsies per ACR+ patient ───────────────────────
ax = axes[1]
ax.set_facecolor(BG)
vc = acr_only["n_acr_pos"].value_counts().sort_index()
ax.bar(vc.index.astype(int), vc.values, color=C_POS, alpha=0.85,
       edgecolor="#333", linewidth=0.8, zorder=2, width=0.7)
for x, y in zip(vc.index.astype(int), vc.values):
    ax.text(x, y + 0.3, str(y), ha="center", va="bottom", fontsize=FS_ANNOT, fontweight="bold")
ax.set_xlabel("ACR+ biopsies per patient", fontsize=FS_LABEL)
ax.set_ylabel("Number of patients", fontsize=FS_LABEL)
ax.set_title(
    f"ACR+ biopsies per ACR+ patient\n(n=56 patients  ·  mean={acr_only['n_acr_pos'].mean():.1f}  ·  "
    f"median={int(acr_only['n_acr_pos'].median())}  ·  max={int(acr_only['n_acr_pos'].max())})",
    fontsize=FS_TITLE, fontweight="bold"
)
ax.tick_params(axis="both", labelsize=FS_TICK)
ax.spines[["top", "right"]].set_visible(False)
ax.set_xticks(vc.index.astype(int))

# ── Panel C: Histogram — days from first biopsy to first ACR event ─────────────
ax = axes[2]
ax.set_facecolor(BG)
bins = [0, 1, 30, 90, 180, 365, 730, 1400]
labels_bin = ["Day 0", "1–30", "31–90", "91–180", "181–365", "366–730", ">730"]
counts_bin = pd.cut(days_to_acr, bins=bins, right=True, include_lowest=True).value_counts().sort_index()
x_pos = np.arange(len(labels_bin))
bars2 = ax.bar(x_pos, counts_bin.values, color=C_POS, alpha=0.85,
               edgecolor="#333", linewidth=0.8, zorder=2, width=0.65)
for bar, v in zip(bars2, counts_bin.values):
    if v > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(v), ha="center", va="bottom", fontsize=FS_ANNOT, fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels(labels_bin, fontsize=FS_TICK, rotation=30, ha="right")
ax.set_xlabel("Days from first biopsy to first ACR+ event", fontsize=FS_LABEL)
ax.set_ylabel("Number of ACR+ patients", fontsize=FS_LABEL)
ax.set_title(
    f"Time to first ACR event\n(n=56 patients  ·  median={int(days_to_acr.median())} days  ·  "
    f"mean={int(days_to_acr.mean())} days)",
    fontsize=FS_TITLE, fontweight="bold"
)
ax.tick_params(axis="y", labelsize=FS_TICK)
ax.spines[["top", "right"]].set_visible(False)

# ── Note: 29 patients had ACR at their very first biopsy (day 0) ──────────────
n_day0 = int((days_to_acr == 0).sum())
ax.text(0.02, 0.96, f"* {n_day0} patients: ACR+ already at\ntheir first biopsy (day 0)",
        transform=ax.transAxes, fontsize=FS_ANNOT - 1, va="top",
        color="#555", style="italic")

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"acr_event_distributions.{ext}", dpi=200,
                bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved acr_event_distributions")

# ── Summary printout for reference ───────────────────────────────────────────
print("\n=== SUMMARY FOR PI ===")
print(f"Total patients (labeled): {len(per_pat)}")
print(f"  ACR+ patients (≥1 ACR+ biopsy): {per_pat['has_acr'].sum()} ({100*per_pat['has_acr'].mean():.0f}%)")
print(f"  ACR- patients (0 ACR+ biopsies): {(~per_pat['has_acr']).sum()} ({100*(~per_pat['has_acr']).mean():.0f}%)")
print(f"\nTotal labeled biopsies: {len(lab)}")
print(f"  ACR+ biopsies: {int(lab['label'].sum())} ({100*lab['label'].mean():.1f}%)")
print(f"  ACR- biopsies: {int((lab['label']==0).sum())} ({100*(lab['label']==0).mean():.1f}%)")
print(f"\nNone/Single/Multiple: {n_none}/{n_single}/{n_multi} patients ({n_none/len(per_pat):.0%}/{n_single/len(per_pat):.0%}/{n_multi/len(per_pat):.0%})")
print(f"\nACR+ biopsies per ACR+ patient: mean={acr_only['n_acr_pos'].mean():.1f}, median={acr_only['n_acr_pos'].median():.0f}, max={acr_only['n_acr_pos'].max():.0f}")
print(f"\nDays to first ACR: median={days_to_acr.median():.0f}, mean={days_to_acr.mean():.0f}, max={days_to_acr.max():.0f}")
print(f"  {n_day0} patients had ACR+ at their very first biopsy (day 0)")
print(f"\nCLAD events: {int(per_pat['clad_status'].sum())} patients, median days: {per_pat[per_pat['clad_status']==1]['clad_days'].median():.0f}")
print(f"Death events: {int(per_pat['death_status'].sum())} patients, median days: {per_pat[per_pat['death_status']==1]['death_days'].median():.0f}")
print("Done.")
