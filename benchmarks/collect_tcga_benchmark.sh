#!/usr/bin/env bash
#SBATCH --job-name=collect_benchmark
#SBATCH --partition=cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/home/aih/dinesh.haridoss/chicago_mil/results_mm_abmil_v8/slurm_logs/%j_collect_benchmark.out

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 - << 'PYEOF'
import os, json, glob, pickle, numpy as np, pandas as pd

LUSTRE   = "/lustre/groups/aih/dinesh.haridoss/mil"
REPO     = "/home/aih/dinesh.haridoss/chicago_mil"
CANCERS  = ["blca", "brca", "gbmlgg", "kirc", "luad"]
N_FOLDS  = 5

def mean_std(vals):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if not vals: return None, None
    return float(np.mean(vals)), float(np.std(vals))

# ──────────────────────────────────────────────
# 1. Our method (train_tcga_multitask.py)
# ──────────────────────────────────────────────
ours = {}
for cancer in CANCERS:
    f = f"{REPO}/results_tcga_multitask/{cancer}/summary.json"
    if not os.path.exists(f): continue
    d = json.load(open(f))
    folds = d.get("folds", {})
    os_cis  = [folds[str(i)].get("os_ci")  for i in range(N_FOLDS) if str(i) in folds]
    dss_cis = [folds[str(i)].get("dss_ci") for i in range(N_FOLDS) if str(i) in folds]
    pfi_cis = [folds[str(i)].get("pfi_ci") for i in range(N_FOLDS) if str(i) in folds]
    baccs   = [folds[str(i)].get("cls_bacc") for i in range(N_FOLDS) if str(i) in folds and "cls_bacc" in folds[str(i)]]
    ours[cancer] = {
        "os":  mean_std(os_cis),
        "dss": mean_std(dss_cis),
        "pfi": mean_std(pfi_cis),
        "bacc": mean_std(baccs) if baccs else (None, None),
    }

# ──────────────────────────────────────────────
# 2. ABMIL-WSI (WSI only, unimodal)
# ──────────────────────────────────────────────
def load_abmil(prefix):
    out = {}
    for cancer in CANCERS:
        cis = []
        for fold in range(N_FOLDS):
            f = f"{LUSTRE}/results_abmil/{prefix}_{cancer}_fold{fold}/summary.json"
            if os.path.exists(f):
                d = json.load(open(f))
                ci = d.get("test_ci")
                if ci is not None and not np.isnan(ci): cis.append(ci)
        out[cancer] = mean_std(cis)
    return out

abmil_wsi = load_abmil("abmil_wsi")
abmil_mm  = load_abmil("abmil_mm")

# ──────────────────────────────────────────────
# 3. MCAT coattn — PKL files
# ──────────────────────────────────────────────
def load_mcat_like(method):
    out = {}
    BASE = f"{LUSTRE}/baseline_results"
    for cancer in CANCERS:
        cis = []
        for fold in range(N_FOLDS):
            d = f"{BASE}/{method}_{cancer}_fold{fold}"
            if not os.path.exists(d): continue
            pkls = glob.glob(f"{d}/**/split_latest_val_{fold}_results.pkl", recursive=True)
            for p in pkls:
                try:
                    data = pickle.load(open(p, 'rb'))
                    ci = data.get('c_index', data.get('cindex', data.get('cindexF', None)))
                    if ci is not None and not np.isnan(ci):
                        cis.append(float(ci))
                        break
                except: pass
        out[cancer] = mean_std(cis)
    return out

mcat   = load_mcat_like("mcat")
amil_b = load_mcat_like("amil")   # ABMIL from MCAT repo (unimodal WSI baseline)

# ──────────────────────────────────────────────
# 4. SurvPath — CSV summary files
# ──────────────────────────────────────────────
def load_survpath(method):
    out = {}
    BASE = f"{LUSTRE}/baseline_results"
    for cancer in CANCERS:
        cis = []
        for fold in range(N_FOLDS):
            d = f"{BASE}/{method}_{cancer}_fold{fold}"
            if not os.path.exists(d): continue
            csvs = glob.glob(f"{d}/**/summary_latest.csv", recursive=True)
            for csv in csvs:
                try:
                    df = pd.read_csv(csv)
                    if 'val_cindex' in df.columns:
                        ci = float(df['val_cindex'].iloc[0])
                        if not np.isnan(ci): cis.append(ci); break
                except: pass
            # also try summary.csv
            csvs2 = glob.glob(f"{d}/**/summary.csv", recursive=True)
            if not cis:
                for csv in csvs2:
                    try:
                        df = pd.read_csv(csv)
                        if 'val_cindex' in df.columns:
                            ci = float(df['val_cindex'].iloc[0])
                            if not np.isnan(ci): cis.append(ci); break
                    except: pass
        out[cancer] = mean_std(cis)
    return out

survpath = load_survpath("survpath")
coattn   = load_survpath("coattn")   # MCAT coattn via SurvPath repo

# ──────────────────────────────────────────────
# 5. New competitor results (from our new benchmark scripts)
# ──────────────────────────────────────────────
NEW_BASE = f"{REPO}/results_tcga_competitors"

def load_new_competitors(method):
    out = {}
    for cancer in CANCERS:
        cis = []
        for fold in range(N_FOLDS):
            d = f"{NEW_BASE}/{method}_{cancer}_fold{fold}"
            if not os.path.exists(d): continue
            # Try all PKL naming variants (MCAT/MOTCAT vs SurvPath/coattn)
            pkl_patterns = [
                f"{d}/**/split_latest_val_{fold}_results.pkl",
                f"{d}/**/split_{fold}_results.pkl",
            ]
            found_pkl = False
            for pat in pkl_patterns:
                pkls = glob.glob(pat, recursive=True)
                for p in pkls:
                    try:
                        data = pickle.load(open(p, 'rb'))
                        ci = data.get('c_index', data.get('cindex', data.get('cindexF', None)))
                        # Patient-level dict fallback (MOTCAT/MCAT format)
                        if ci is None and isinstance(data, dict):
                            pts = [v for v in data.values() if isinstance(v, dict) and 'risk' in v]
                            if pts:
                                from sksurv.metrics import concordance_index_censored
                                risks = np.array([float(np.atleast_1d(pt['risk'])[0]) for pt in pts])
                                times = np.array([float(pt['survival']) for pt in pts])
                                cens  = np.array([float(pt['censorship']) for pt in pts])
                                mask  = np.isfinite(risks) & np.isfinite(times) & np.isfinite(cens)
                                if mask.sum() > 1 and (1 - cens[mask]).astype(bool).any():
                                    ci = concordance_index_censored((1-cens[mask]).astype(bool), times[mask], risks[mask])[0]
                        if ci is not None and not np.isnan(ci):
                            cis.append(float(ci)); found_pkl = True; break
                    except: pass
                if found_pkl: break
            # SurvPath-style CSV fallback
            if not found_pkl:
                csvs = glob.glob(f"{d}/**/summary_latest.csv", recursive=True) + \
                       glob.glob(f"{d}/**/summary.csv", recursive=True)
                for csv in csvs:
                    try:
                        df = pd.read_csv(csv)
                        if 'val_cindex' in df.columns:
                            ci = float(df['val_cindex'].iloc[0])
                            if not np.isnan(ci): cis.append(ci); break
                    except: pass
        out[cancer] = mean_std(cis)
    return out

mcat_new     = load_new_competitors("mcat")
amil_new     = load_new_competitors("amil")
survpath_new = load_new_competitors("survpath")
coattn_new   = load_new_competitors("coattn")
motcat_new   = load_new_competitors("motcat")

# ──────────────────────────────────────────────
# Print benchmark table
# ──────────────────────────────────────────────
def fmt(pair, endpoint="os"):
    if pair is None or pair[0] is None: return "  —   "
    return f"{pair[0]:.3f}±{pair[1]:.3f}"

def fmt_ours(cancer, ep):
    d = ours.get(cancer)
    if not d: return "  —   "
    return fmt(d.get(ep))

methods = [
    ("MCAT coattn (new)",     mcat_new,    "os"),
    ("MCAT coattn-SurvPath",  coattn_new,  "os"),
    ("ABMIL (new)",           amil_new,    "os"),
    ("SurvPath (new)",        survpath_new,"os"),
    ("MOTCAT (new)",          motcat_new,  "os"),
    ("Ours (multitask)",      None,        "os"),
]

print("\n" + "="*90)
print("TCGA BENCHMARK  —  OS C-index (mean ± std, 5-fold CV)")
print("="*90)
header = f"{'Method':<28}" + "".join(f"  {c.upper():<12}" for c in CANCERS)
print(header)
print("-"*90)

for name, d, ep in methods:
    if name == "Ours (multitask)":
        row = f"{'Ours (multitask MIL)':<28}" + "".join(
            f"  {fmt_ours(c, 'os'):<12}" for c in CANCERS)
    else:
        row = f"{name:<28}" + "".join(
            f"  {fmt(d.get(c), ep):<12}" if d.get(c) and d[c][0] else f"  {'—':<12}"
            for c in CANCERS)
    print(row)

print("="*90)

# Also save full JSON
all_results = {
    "ours": ours,
    "mcat_new": {c: {"os": mcat_new[c]} for c in CANCERS},
    "coattn_new": {c: {"os": coattn_new[c]} for c in CANCERS},
    "amil_new": {c: {"os": amil_new[c]} for c in CANCERS},
    "survpath_new": {c: {"os": survpath_new[c]} for c in CANCERS},
    "motcat_new": {c: {"os": motcat_new[c]} for c in CANCERS},
}
out_path = f"{REPO}/results_tcga_multitask/benchmark_table.json"
json.dump(all_results, open(out_path, "w"), indent=2)
print(f"\nSaved to {out_path}")
PYEOF
