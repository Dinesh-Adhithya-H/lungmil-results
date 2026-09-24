"""
Event distributions for all 4 endpoints:
  ACR classification, ACR survival, CLAD survival, Death survival.
Panels: event counts, time-to-event histograms.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

ROOT    = Path(__file__).resolve().parent.parent
CSV     = Path("/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")
OUT_DIR = ROOT / "figures" / "cohort"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BG  = "#FAF6F2"
C_ACR   = "#EF5350"   # red
C_CLAD  = "#AB47BC"   # purple
C_DEATH = "#546E7A"   # slate
C_NEG   = "#90CAF9"   # blue (ACR-)
C_CENS  = "#B0BEC5"   # grey (censored)

FS_TITLE = 18
FS_LABEL = 14
FS_TICK  = 12
FS_ANNOT = 12

df = pd.read_csv(CSV)
df["anchor_dt"] = pd.to_datetime(df["anchor_dt"])
lab = df[df["label"].notna()].copy()

# ── All 263 patients as base ──────────────────────────────────────────────────
all_pat = df.drop_duplicates("patient_id").copy()
first_biopsy = df.groupby("patient_id")["anchor_dt"].min()
all_pat["first_biopsy_dt"] = all_pat["patient_id"].map(first_biopsy)

# ACR: count positive biopsies per patient (0 if no labeled biopsies)
acr_pos_per_pat = lab[lab["label"] == 1].groupby("patient_id").size().reindex(
    all_pat["patient_id"], fill_value=0).values
all_pat["n_acr_pos"] = acr_pos_per_pat
all_pat["has_acr"]   = all_pat["n_acr_pos"] > 0

# time to first ACR+ biopsy from patient's first biopsy
acr_first = lab[lab["label"] == 1].groupby("patient_id")["anchor_dt"].min().reset_index()
acr_first.columns = ["patient_id", "first_acr_dt"]
all_pat = all_pat.merge(acr_first, on="patient_id", how="left")
all_pat["days_to_first_acr"] = (
    pd.to_datetime(all_pat["first_acr_dt"]) - pd.to_datetime(all_pat["first_biopsy_dt"])
).dt.days

# keep per_pat_lab alias for ACR+ subgroup plots
per_pat_lab = all_pat

# Pre-compute follow-up and biopsies per patient for new panels
bpp = df.groupby("patient_id").size().rename("n_biopsies")
last_biopsy  = df.groupby("patient_id")["anchor_dt"].max()
followup_days = (last_biopsy - first_biopsy).dt.days.rename("followup_days")

# ACR survival (patient-level, from acr_status / acr_days)
# For censored patients missing acr_days, use death_days as censoring time
acr_surv_ev   = all_pat[all_pat["acr_status"] == 1]["acr_days"].dropna()
_cens_pat     = all_pat[all_pat["acr_status"] == 0].copy()
_cens_pat["cens_days"] = _cens_pat["acr_days"].fillna(_cens_pat["death_days"])
acr_surv_cens = _cens_pat["cens_days"].dropna()

n_all     = len(all_pat)
n_acr_pat = int(all_pat["has_acr"].sum())
n_clad    = int(all_pat["clad_status"].sum())
n_death   = int(all_pat["death_status"].sum())
n_bpos    = int(lab["label"].sum())
n_bneg    = int((lab["label"] == 0).sum())
n_none    = int((all_pat["n_acr_pos"] == 0).sum())
n_single  = int((all_pat["n_acr_pos"] == 1).sum())
n_multi   = int((all_pat["n_acr_pos"] >  1).sum())

# Pre-compute biopsy TTE to first ACR event
acr_pos_pids = set(all_pat[all_pat["has_acr"]]["patient_id"])
lab_acr_pos  = lab[lab["patient_id"].isin(acr_pos_pids)].copy()
first_acr_date = (lab[lab["label"] == 1]
                  .groupby("patient_id")["anchor_dt"].min()
                  .rename("first_acr_dt"))
lab_acr_pos = lab_acr_pos.merge(first_acr_date, on="patient_id", how="left")
lab_acr_pos["days_to_acr"] = (
    pd.to_datetime(lab_acr_pos["first_acr_dt"]) -
    pd.to_datetime(lab_acr_pos["anchor_dt"])
).dt.days
biopsy_tte  = lab_acr_pos[lab_acr_pos["days_to_acr"] >= 0]["days_to_acr"]

clad_ev     = all_pat[all_pat["clad_status"] == 1]["clad_days"].dropna()
clad_cens   = all_pat[all_pat["clad_status"] == 0]["clad_days"].dropna()
death_ev    = all_pat[all_pat["death_status"] == 1]["death_days"].dropna()
death_cens  = all_pat[all_pat["death_status"] == 0]["death_days"].dropna()
acr_only    = per_pat_lab[per_pat_lab["has_acr"]]
med_bpp     = int(bpp.median())
med_fu      = int(followup_days.median())
C_ACR_SURV  = "#FF7043"

# Modality availability (correct patient- and biopsy-level counts)
MOD_COLS   = {"H&E": "has_HE", "BAL": "has_BAL", "CT": "has_CT", "Clinical": "has_Clinical"}
MOD_PAT    = {m: df[df[c] == 1]["patient_id"].nunique() for m, c in MOD_COLS.items() if c in df.columns}
MOD_BIO    = {m: int(df[c].sum())                       for m, c in MOD_COLS.items() if c in df.columns}
MOD_COLORS_D = {"H&E": "#E64A19", "BAL": "#1565C0", "CT": "#2E7D32", "Clinical": "#7B1FA2"}


def modality_panel(ax_pat, ax_bio):
    """Two bar charts: patients with modality, biopsies with modality."""
    mods = list(MOD_PAT.keys())
    cols = [MOD_COLORS_D[m] for m in mods]

    # Patient-level
    pat_vals = [MOD_PAT[m] for m in mods]
    bars = ax_pat.bar(mods, pat_vals, color=cols, alpha=0.88,
                      edgecolor="#333", linewidth=0.8, width=0.55, zorder=2)
    for bar, v in zip(bars, pat_vals):
        ax_pat.text(bar.get_x() + bar.get_width()/2, v + 1,
                    f"n={v}\n({100*v/n_all:.0f}%)", ha="center", va="bottom",
                    fontsize=FS_ANNOT, fontweight="bold")
    ax_pat.set_ylim(0, n_all + 30)
    ax_pat.axhline(n_all, color="#555", linewidth=1, linestyle="--", alpha=0.5)
    ax_pat.set_ylabel("Patients", fontsize=FS_LABEL)
    ax_pat.set_title(f"Patients with modality  (total={n_all})",
                     fontsize=FS_TITLE, fontweight="bold")
    ax_pat.tick_params(labelsize=FS_TICK)
    ax_pat.spines[["top", "right"]].set_visible(False)
    ax_pat.set_facecolor(BG)

    # Biopsy-level
    n_total_bio = len(df)
    bio_vals = [MOD_BIO[m] for m in mods]
    bars2 = ax_bio.bar(mods, bio_vals, color=cols, alpha=0.88,
                       edgecolor="#333", linewidth=0.8, width=0.55, zorder=2)
    for bar, v in zip(bars2, bio_vals):
        ax_bio.text(bar.get_x() + bar.get_width()/2, v + 20,
                    f"n={v}\n({100*v/n_total_bio:.0f}%)", ha="center", va="bottom",
                    fontsize=FS_ANNOT, fontweight="bold")
    ax_bio.set_ylim(0, n_total_bio + 200)
    ax_bio.axhline(n_total_bio, color="#555", linewidth=1, linestyle="--", alpha=0.5)
    ax_bio.set_ylabel("Biopsies", fontsize=FS_LABEL)
    ax_bio.set_title(f"Biopsies with modality  (total={n_total_bio})",
                     fontsize=FS_TITLE, fontweight="bold")
    ax_bio.tick_params(labelsize=FS_TICK)
    ax_bio.spines[["top", "right"]].set_visible(False)
    ax_bio.set_facecolor(BG)


def cohort_panel(ax):
    """Shared cohort overview bar (biopsies per patient)."""
    bins_bpp = np.arange(0.5, bpp.max() + 1.5, 2)
    ax.hist(bpp.values, bins=bins_bpp, color="#5C6BC0", alpha=0.85,
            edgecolor="#333", linewidth=0.6, zorder=2)
    ax.axvline(med_bpp, color="#1A237E", linewidth=2, linestyle="--",
               label=f"Median: {med_bpp}")
    ax.set_xlabel("Biopsies per patient", fontsize=FS_LABEL)
    ax.set_ylabel("Patients", fontsize=FS_LABEL)
    ax.set_title(
        f"Biopsies per patient  (n={n_all})\n"
        f"median={med_bpp}  IQR=[{int(bpp.quantile(0.25))},{int(bpp.quantile(0.75))}]  max={int(bpp.max())}",
        fontsize=FS_TITLE, fontweight="bold")
    ax.legend(fontsize=FS_TICK, framealpha=0.9)
    ax.tick_params(labelsize=FS_TICK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor(BG)


def followup_panel(ax):
    """Shared follow-up duration histogram."""
    bins_fu = np.arange(0, followup_days.max() + 180, 180)
    ax.hist(followup_days.values, bins=bins_fu, color="#26A69A", alpha=0.85,
            edgecolor="#333", linewidth=0.6, zorder=2)
    ax.axvline(med_fu, color="#00695C", linewidth=2, linestyle="--",
               label=f"Median: {med_fu} d")
    ax.set_xlabel("Days (first to last biopsy)", fontsize=FS_LABEL)
    ax.set_ylabel("Patients", fontsize=FS_LABEL)
    ax.set_title(
        f"Follow-up  (n={n_all})\n"
        f"median={med_fu} d  IQR=[{int(followup_days.quantile(0.25))},{int(followup_days.quantile(0.75))}] d",
        fontsize=FS_TITLE, fontweight="bold")
    ax.legend(fontsize=FS_TICK, framealpha=0.9)
    ax.tick_params(labelsize=FS_TICK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_facecolor(BG)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 — ACR CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(22, 14), facecolor=BG)
fig.patch.set_facecolor(BG)
fig.suptitle("ACR Classification — Northwestern Lung Transplant Cohort  (n=263 patients, 4,210 visits)",
             fontsize=20, fontweight="bold", y=1.01)

# (0,0) None / Single / Multiple
ax = axes[0, 0]; ax.set_facecolor(BG)
cats = ["None\n(0 ACR+)", "Single\n(1 ACR+)", "Multiple\n(≥2 ACR+)"]
vals = [n_none, n_single, n_multi]
bars = ax.bar(cats, vals, color=[C_NEG, "#FFA726", C_ACR], alpha=0.88,
              width=0.55, edgecolor="#333", linewidth=0.8, zorder=2)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f"n={v}\n({100*v/n_all:.0f}%)", ha="center", va="bottom",
            fontsize=FS_ANNOT, fontweight="bold")
ax.set_ylabel("Patients", fontsize=FS_LABEL); ax.set_ylim(0, n_none + 30)
ax.set_title(f"ACR events per patient  (all {n_all})", fontsize=FS_TITLE, fontweight="bold")
ax.tick_params(labelsize=FS_TICK); ax.spines[["top","right"]].set_visible(False)

# (0,1) ACR+ vs ACR- biopsies
ax = axes[0, 1]; ax.set_facecolor(BG)
ax.bar(["ACR+", "ACR−"], [n_bpos, n_bneg], color=[C_ACR, C_NEG],
       alpha=0.88, width=0.5, edgecolor="#333", linewidth=0.8, zorder=2)
ax.text(0, n_bpos + 10, f"n={n_bpos}\n({100*n_bpos/(n_bpos+n_bneg):.1f}%)",
        ha="center", va="bottom", fontsize=FS_ANNOT, fontweight="bold")
ax.text(1, n_bneg + 10, f"n={n_bneg}\n({100*n_bneg/(n_bpos+n_bneg):.1f}%)",
        ha="center", va="bottom", fontsize=FS_ANNOT, fontweight="bold")
ax.set_ylabel("Biopsies", fontsize=FS_LABEL)
ax.set_title(f"ACR+ vs ACR− biopsies\n(total={n_bpos+n_bneg} labeled)", fontsize=FS_TITLE, fontweight="bold")
ax.set_ylim(0, n_bneg + 150); ax.tick_params(labelsize=FS_TICK)
ax.spines[["top","right"]].set_visible(False)

# (0,2) ACR+ biopsies per ACR+ patient
ax = axes[0, 2]; ax.set_facecolor(BG)
vc = acr_only["n_acr_pos"].value_counts().sort_index()
ax.bar(vc.index.astype(int), vc.values, color=C_ACR, alpha=0.85,
       edgecolor="#333", linewidth=0.8, width=0.7, zorder=2)
for x, y in zip(vc.index.astype(int), vc.values):
    ax.text(x, y + 0.2, str(y), ha="center", va="bottom", fontsize=FS_ANNOT, fontweight="bold")
ax.set_xlabel("ACR+ biopsies per patient", fontsize=FS_LABEL)
ax.set_ylabel("Patients", fontsize=FS_LABEL)
ax.set_title(
    f"ACR+ biopsies / ACR+ patient  (n={int(all_pat['has_acr'].sum())})\n"
    f"mean={acr_only['n_acr_pos'].mean():.1f}  median={int(acr_only['n_acr_pos'].median())}  max={int(acr_only['n_acr_pos'].max())}",
    fontsize=FS_TITLE, fontweight="bold")
ax.set_xticks(vc.index.astype(int)); ax.tick_params(labelsize=FS_TICK)
ax.spines[["top","right"]].set_visible(False)

# (1,0) Biopsy TTE to first ACR event
ax = axes[1, 0]; ax.set_facecolor(BG)
bins_d = np.arange(0, biopsy_tte.max() + 60, 60)
ax.hist(biopsy_tte, bins=bins_d, color=C_ACR, alpha=0.85,
        edgecolor="#333", linewidth=0.6, zorder=2)
med = int(biopsy_tte.median())
ax.axvline(med, color="#7B0000", linewidth=2, linestyle="--", label=f"Median {med} d")
ax.set_xlabel("Days from biopsy to first ACR+ event", fontsize=FS_LABEL)
ax.set_ylabel("Biopsies", fontsize=FS_LABEL)
ax.set_title(
    f"Biopsy TTE to first ACR event  ({lab_acr_pos['patient_id'].nunique()} patients)\n"
    f"median={med} d  mean={int(biopsy_tte.mean())} d  max={int(biopsy_tte.max())} d",
    fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_TICK, framealpha=0.9); ax.tick_params(labelsize=FS_TICK)
ax.spines[["top","right"]].set_visible(False)

# (1,1) Biopsies per patient
cohort_panel(axes[1, 1])

# (1,2) Follow-up
followup_panel(axes[1, 2])

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"event_dist_acr_cls.{ext}", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved event_dist_acr_cls")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 — ACR SURVIVAL
# 56 events (acr_days); 207 censored (acr_days if available, else death_days)
# ═══════════════════════════════════════════════════════════════════════════════
n_acr_surv = len(acr_surv_ev) + len(acr_surv_cens)   # should be 263

fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor=BG)
fig.patch.set_facecolor(BG)
fig.suptitle(
    f"ACR Survival — Northwestern Lung Transplant Cohort  (n={n_acr_surv} patients)",
    fontsize=20, fontweight="bold", y=1.01)

# (0,0) Event vs censored bar
ax = axes[0]; ax.set_facecolor(BG)
ax.bar(["ACR episode", "Censored"], [len(acr_surv_ev), len(acr_surv_cens)],
       color=[C_ACR_SURV, C_CENS], alpha=0.88, width=0.5, edgecolor="#333", linewidth=0.8, zorder=2)
ax.text(0, len(acr_surv_ev) + 1, f"n={len(acr_surv_ev)}\n({100*len(acr_surv_ev)/n_acr_surv:.0f}%)",
        ha="center", va="bottom", fontsize=FS_ANNOT, fontweight="bold")
ax.text(1, len(acr_surv_cens) + 1, f"n={len(acr_surv_cens)}\n({100*len(acr_surv_cens)/n_acr_surv:.0f}%)",
        ha="center", va="bottom", fontsize=FS_ANNOT, fontweight="bold")
ax.set_ylabel("Patients", fontsize=FS_LABEL)
ax.set_title(f"ACR survival events\n(all {n_acr_surv} patients)", fontsize=FS_TITLE, fontweight="bold")
ax.set_ylim(0, len(acr_surv_cens) + 30); ax.tick_params(labelsize=FS_TICK)
ax.spines[["top","right"]].set_visible(False)

# (0,1) TTE histogram
ax = axes[1]; ax.set_facecolor(BG)
bins_as = np.arange(0, 3600, 180)
ax.hist(acr_surv_ev,   bins=bins_as, color=C_ACR_SURV, alpha=0.85,
        label=f"ACR episode (n={len(acr_surv_ev)})", edgecolor="#333", linewidth=0.6, zorder=3)
ax.hist(acr_surv_cens, bins=bins_as, color=C_CENS, alpha=0.50,
        label=f"Censored (n={len(acr_surv_cens)})", edgecolor="#888", linewidth=0.4, zorder=2)
ax.axvline(acr_surv_ev.median(), color="#BF360C", linewidth=2, linestyle="--",
           label=f"Median event: {int(acr_surv_ev.median())} d")
ax.set_xlabel("Days from transplant to ACR episode / censoring", fontsize=FS_LABEL)
ax.set_ylabel("Patients", fontsize=FS_LABEL)
ax.set_title(
    f"ACR survival TTE  ·  {len(acr_surv_ev)} events / {n_acr_surv} patients  ({100*len(acr_surv_ev)/n_acr_surv:.0f}%)\n"
    f"Median event: {int(acr_surv_ev.median())} d  ·  Median follow-up (censored): {int(acr_surv_cens.median())} d",
    fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_TICK, framealpha=0.9); ax.tick_params(labelsize=FS_TICK)
ax.spines[["top","right"]].set_visible(False)

# (0,2) Biopsies per patient
cohort_panel(axes[2])

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"event_dist_acr_surv.{ext}", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved event_dist_acr_surv")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 — CLAD SURVIVAL
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor=BG)
fig.patch.set_facecolor(BG)
fig.suptitle("CLAD Survival — Northwestern Lung Transplant Cohort  (n=263 patients)",
             fontsize=20, fontweight="bold", y=1.01)

# (0,0) Event vs censored bar
ax = axes[0]; ax.set_facecolor(BG)
ax.bar(["CLAD event", "Censored"], [len(clad_ev), len(clad_cens)],
       color=[C_CLAD, C_CENS], alpha=0.88, width=0.5, edgecolor="#333", linewidth=0.8, zorder=2)
ax.text(0, len(clad_ev) + 1, f"n={len(clad_ev)}\n({100*len(clad_ev)/n_all:.0f}%)",
        ha="center", va="bottom", fontsize=FS_ANNOT, fontweight="bold")
ax.text(1, len(clad_cens) + 1, f"n={len(clad_cens)}\n({100*len(clad_cens)/n_all:.0f}%)",
        ha="center", va="bottom", fontsize=FS_ANNOT, fontweight="bold")
ax.set_ylabel("Patients", fontsize=FS_LABEL)
ax.set_title(f"CLAD survival events\n(all {n_all} patients)", fontsize=FS_TITLE, fontweight="bold")
ax.set_ylim(0, len(clad_cens) + 30); ax.tick_params(labelsize=FS_TICK)
ax.spines[["top","right"]].set_visible(False)

# (0,1) TTE histogram
ax = axes[1]; ax.set_facecolor(BG)
bins_c = np.arange(0, 2800, 180)
ax.hist(clad_ev,   bins=bins_c, color=C_CLAD, alpha=0.85,
        label=f"CLAD event (n={len(clad_ev)})", edgecolor="#333", linewidth=0.6, zorder=3)
ax.hist(clad_cens, bins=bins_c, color=C_CENS,  alpha=0.50,
        label=f"Censored (n={len(clad_cens)})", edgecolor="#888", linewidth=0.4, zorder=2)
ax.axvline(clad_ev.median(), color=C_CLAD, linewidth=2, linestyle="--",
           label=f"Median event: {int(clad_ev.median())} d")
ax.set_xlabel("Days from transplant to CLAD / censoring", fontsize=FS_LABEL)
ax.set_ylabel("Patients", fontsize=FS_LABEL)
ax.set_title(
    f"CLAD survival TTE  ·  {len(clad_ev)} events / {n_all} patients  ({100*len(clad_ev)/n_all:.0f}%)\n"
    f"Median event: {int(clad_ev.median())} d  ·  Median follow-up: {int(all_pat['clad_days'].median())} d",
    fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_TICK, framealpha=0.9); ax.tick_params(labelsize=FS_TICK)
ax.spines[["top","right"]].set_visible(False)

# (0,2) Follow-up
followup_panel(axes[2])

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"event_dist_clad_surv.{ext}", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved event_dist_clad_surv")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 — DEATH SURVIVAL
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor=BG)
fig.patch.set_facecolor(BG)
fig.suptitle("Post-Transplant Mortality — Northwestern Lung Transplant Cohort  (n=263 patients)",
             fontsize=20, fontweight="bold", y=1.01)

# (0,0) Event vs censored bar
ax = axes[0]; ax.set_facecolor(BG)
ax.bar(["Death", "Censored"], [len(death_ev), len(death_cens)],
       color=[C_DEATH, C_CENS], alpha=0.88, width=0.5, edgecolor="#333", linewidth=0.8, zorder=2)
ax.text(0, len(death_ev) + 1, f"n={len(death_ev)}\n({100*len(death_ev)/n_all:.0f}%)",
        ha="center", va="bottom", fontsize=FS_ANNOT, fontweight="bold")
ax.text(1, len(death_cens) + 1, f"n={len(death_cens)}\n({100*len(death_cens)/n_all:.0f}%)",
        ha="center", va="bottom", fontsize=FS_ANNOT, fontweight="bold")
ax.set_ylabel("Patients", fontsize=FS_LABEL)
ax.set_title(f"Death events\n(all {n_all} patients)", fontsize=FS_TITLE, fontweight="bold")
ax.set_ylim(0, len(death_cens) + 30); ax.tick_params(labelsize=FS_TICK)
ax.spines[["top","right"]].set_visible(False)

# (0,1) TTE histogram
ax = axes[1]; ax.set_facecolor(BG)
bins_d = np.arange(0, 3500, 250)
ax.hist(death_ev,   bins=bins_d, color=C_DEATH, alpha=0.85,
        label=f"Death (n={len(death_ev)})", edgecolor="#333", linewidth=0.6, zorder=3)
ax.hist(death_cens, bins=bins_d, color=C_CENS,  alpha=0.50,
        label=f"Censored (n={len(death_cens)})", edgecolor="#888", linewidth=0.4, zorder=2)
ax.axvline(death_ev.median(), color=C_DEATH, linewidth=2, linestyle="--",
           label=f"Median: {int(death_ev.median())} d")
ax.set_xlabel("Days from transplant to death / censoring", fontsize=FS_LABEL)
ax.set_ylabel("Patients", fontsize=FS_LABEL)
ax.set_title(
    f"Death survival TTE  ·  {len(death_ev)} events / {n_all} patients  ({100*len(death_ev)/n_all:.0f}%)\n"
    f"Median event: {int(death_ev.median())} d  ·  Median follow-up: {int(all_pat['death_days'].median())} d",
    fontsize=FS_TITLE, fontweight="bold")
ax.legend(fontsize=FS_TICK, framealpha=0.9); ax.tick_params(labelsize=FS_TICK)
ax.spines[["top","right"]].set_visible(False)

# (0,2) Follow-up
followup_panel(axes[2])

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"event_dist_death_surv.{ext}", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved event_dist_death_surv")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 — MODALITY AVAILABILITY
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
fig.patch.set_facecolor(BG)
fig.suptitle("Modality Availability — Northwestern Lung Transplant Cohort  (n=263 patients, 4,210 visits)",
             fontsize=18, fontweight="bold", y=1.02)
modality_panel(axes[0], axes[1])
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT_DIR / f"modality_availability.{ext}", dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("Saved modality_availability")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"""
=== EVENT SUMMARY ===
Cohort: {len(all_pat)} patients total, 4210 biopsy visits
Labeled biopsies: {len(lab)} ({n_bpos} ACR+, {n_bneg} ACR-)

ACR classification  : {n_acr_pat}/{n_all} patients ({100*n_acr_pat/n_all:.0f}%) have ≥1 ACR+ biopsy
  None/Single/Multi : {n_none}/{n_single}/{n_multi}
  Biopsies ACR+ / - : {n_bpos} / {n_bneg} ({100*n_bpos/(n_bpos+n_bneg):.1f}% / {100*n_bneg/(n_bpos+n_bneg):.1f}%)

ACR survival        : {len(acr_surv_ev)}/{n_acr_surv} events ({100*len(acr_surv_ev)/n_acr_surv:.0f}%), {len(acr_surv_cens)} censored (death_days used for 171 missing acr_days)
CLAD survival       : {len(clad_ev)}/{len(all_pat)} events ({100*len(clad_ev)/len(all_pat):.0f}%), median event {int(clad_ev.median())} days
Death survival      : {len(death_ev)}/{len(all_pat)} events ({100*len(death_ev)/len(all_pat):.0f}%), median event {int(death_ev.median())} days
""")
print("Done.")
