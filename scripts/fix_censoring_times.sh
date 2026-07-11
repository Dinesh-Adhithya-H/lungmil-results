#!/usr/bin/env bash
#SBATCH --job-name=fix_censor
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/fix_censor_%j.out
#SBATCH --error=/lustre/groups/aih/dinesh.haridoss/home_backup/chicago_mil/fix_censor_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 -u << 'PYEOF'
"""
Rebuild clad_days and death_days in the splits CSV from scratch.

clad_days[biopsy]  = (clad_event_date  - anchor_dt).days  for CLAD events
                   = (last_fu_date     - anchor_dt).days  for censored
death_days[biopsy] = (death_event_date - anchor_dt).days  for Death events
                   = (last_fu_date     - anchor_dt).days  for censored

Event info comes from Mortality_updated.csv (per-patient).
Last follow-up date = max(anchor_dt) across all biopsies in the splits CSV.
"""
import pandas as pd
import numpy as np

CSV      = "/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"
MORT_CSV = "/lustre/groups/aih/dinesh.haridoss/datasets/Mortality_updated.csv"
OUT_CSV  = "/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv"

df   = pd.read_csv(CSV)
mort = pd.read_csv(MORT_CSV)

df['anchor_dt'] = pd.to_datetime(df['anchor_dt'])
mort['clad_date']       = pd.to_datetime(mort['clad_date'],       errors='coerce')
mort['date_of_death']   = pd.to_datetime(mort['date_of_death'],   errors='coerce')
mort = mort.set_index('record_id')

print(f"Splits CSV:  {len(df)} rows, {df['patient_id'].nunique()} patients")
print(f"Mortality:   {len(mort)} patients, CLAD={mort['clad'].sum():.0f}, Deaths={mort['death_status'].sum():.0f}")

# ── Last follow-up date per patient ───────────────────────────────────────
last_fu = df.groupby('patient_id')['anchor_dt'].max().rename('last_fu')

# ── Reset to clean state — recompute everything ───────────────────────────
df['clad_status']  = np.nan
df['clad_days']    = np.nan
df['death_status'] = np.nan
df['death_days']   = np.nan

for idx, row in df.iterrows():
    pid = row['patient_id']
    biopsy_dt = row['anchor_dt']
    fu_dt = last_fu[pid]

    if pid not in mort.index:
        continue

    m = mort.loc[pid]

    # CLAD
    clad_st = float(m['clad']) if pd.notna(m['clad']) else 0.0
    df.at[idx, 'clad_status'] = clad_st
    if clad_st == 1 and pd.notna(m['clad_date']):
        df.at[idx, 'clad_days'] = (m['clad_date'] - biopsy_dt).days
    else:
        df.at[idx, 'clad_days'] = (fu_dt - biopsy_dt).days   # censored

    # Death
    death_st = float(m['death_status']) if pd.notna(m['death_status']) else 0.0
    df.at[idx, 'death_status'] = death_st
    if death_st == 1 and pd.notna(m['date_of_death']):
        df.at[idx, 'death_days'] = (m['date_of_death'] - biopsy_dt).days
    else:
        df.at[idx, 'death_days'] = (fu_dt - biopsy_dt).days  # censored

print(f"\nclad_status  — event: {(df['clad_status']==1).sum()},  censored: {(df['clad_status']==0).sum()},  NaN: {df['clad_status'].isna().sum()}")
print(f"death_status — event: {(df['death_status']==1).sum()},  censored: {(df['death_status']==0).sum()},  NaN: {df['death_status'].isna().sum()}")
print(f"clad_days  NaN: {df['clad_days'].isna().sum()}")
print(f"death_days NaN: {df['death_days'].isna().sum()}")

print(f"\nclad_days stats:\n{df.groupby('clad_status')['clad_days'].describe().round(1).to_string()}")
print(f"\ndeath_days stats:\n{df.groupby('death_status')['death_days'].describe().round(1).to_string()}")

# Sanity: pre-event biopsies (clad_days > 0) for event patients
pre_event = (df['clad_status']==1) & (df['clad_days'] > 0)
post_event = (df['clad_status']==1) & (df['clad_days'] <= 0)
print(f"\nCLAD event biopsies — pre-CLAD: {pre_event.sum()},  post-CLAD: {post_event.sum()}")
pre_death = (df['death_status']==1) & (df['death_days'] > 0)
print(f"Death event biopsies — pre-death: {pre_death.sum()},  post-death: {(~pre_death & (df['death_status']==1)).sum()}")

df.to_csv(OUT_CSV, index=False)
print(f"\nSaved updated CSV to {OUT_CSV}")
PYEOF
