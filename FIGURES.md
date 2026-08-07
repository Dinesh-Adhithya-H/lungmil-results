# Figures Reference

All figures are in `figures/`. PDF and PNG are generated for every plot. The scripts that produce each group are listed; all must be run via `sbatch` (never directly on the login node).

---

## 1. Benchmark

**Script:** `analysis/plot_benchmark.py` → `analysis/submit_benchmark.sh`

| File | Description |
|------|-------------|
| `benchmark/benchmark_{task}.pdf/png` | Horizontal bar chart: all models × one task, mean±std BACC or C-index across 5 splits. Best model marked ★. x-axis 0–1. Tasks: `acr_cls`, `acr_surv`, `clad_surv`, `death_surv`, `all` (panel). |
| `benchmark/benchmark_table_v2.pdf/png` | Tabular summary: rows = models, columns = tasks, cells = mean±std. |

**Script:** `analysis/plot_benchmark_multimodal.py` → `analysis/submit_benchmark.sh`

| File | Description |
|------|-------------|
| `benchmark/benchmark_multimodal_{task}.pdf/png` | Multimodal model family comparison. P1 (unimodal ABMIL) vs P2 (LongMK variants). Uses `SHARED_MODEL_COLORS`. `mario_kempes` removed; LongMK highlighted ★. |

**Script:** `analysis/plot_benchmark_pvalues.py` → `analysis/submit_benchmark.sh`

| File | Description |
|------|-------------|
| `benchmark/benchmark_pvalues_longmk.pdf/png` | Heatmap: one-sided paired t-test p(LongMK > model) for each (model, task) pair. Rows = 17 other models, cols = 4 tasks. LogNorm colormap (vmin=0.001, vmax=1.0); green=significant, red=not. Cell text: p-value + stars. |

**Script:** `analysis/plot_modality_combo_ablation.py` → `analysis/submit_benchmark.sh`

| File | Description |
|------|-------------|
| `benchmark/modality_combo_{task}.pdf/png` | Modality combination ablation: all 2^4=16 subsets of {HE, BAL, CT, Clinical}. Bar chart per task showing which combinations achieve best performance. |

---

## 2. Kaplan-Meier Curves

**Script:** `analysis/plot_km_from_model.py` → `analysis/submit_km.sh`

| File | Description |
|------|-------------|
| `km_curves/km_{task}.pdf/png` | KM curves from LongMK patient-level risk stratification (high/low risk). Greenwood 95% CI shading. Tasks: `acr_cls`, `acr_surv`, `clad_surv`, `death_surv`. |
| `km_curves/km_all_tasks.pdf/png` | 4-panel combined KM overview. |
| `km_curves/{task}_km_longi_biopsy.pdf/png` | *(Legacy)* Biopsy-level KM from older extraction pipeline. |

---

## 3. Interpretability — Per-Task

All per-task plots are in `figures/interpretability/{task}/` where `task` ∈ {`acr_cls`, `acr_surv`, `clad_surv`, `death_surv`}.

### 3a. Patient-Rep UMAP (unified)

**Script:** `analysis/plot_unified_rep_umap.py` → `analysis/submit_unified_rep_umap.sh`

8-panel 2×4 figure showing patient-level representation space for each task. LongMK reps for `acr_surv`/`death_surv`; SetMIL reps for `acr_cls`/`clad_surv`.

| Panel | Content |
|-------|---------|
| ① | Risk score scatter (rank-percentile or sigmoid) |
| ② | Event indicator / true label scatter |
| ③ | TTE scatter + ×/○ event/censored markers |
| ④ | Event density hexbin (median-centred) |
| ⑤ | Modality combination scatter |
| ⑥ | KM top vs bottom risk tertile |
| ⑦ | CV split annotation |
| ⑧ | Avg TTE hexbin (median-centred) |

| File | Description |
|------|-------------|
| `{task}/unified_rep_umap_{task}.pdf/png` | Per-task 8-panel figure |
| `agg/unified_rep_umap_all_tasks.pdf/png` | 2×2 overview (risk score coloring, all 4 tasks) |

### 3b. Biopsy-Rep UMAP

**Script:** `analysis/plot_biopsy_rep_umap_km.py` → `analysis/submit_biopsy_rep_umap_km.sh`

8-panel 2×4 figure at the **biopsy level** — each point is a biopsy, predictions use all past context up to that biopsy. `tte = event_date − biopsy_date`; prospective biopsies (tte>0) colored, post-event (tte<0) shown in orange, no-outcome in grey.

| Panel | Content |
|-------|---------|
| ① | Biopsy-level risk score |
| ② | Biopsy timing vs event (red=pre-event, orange=post-event, blue=censored) |
| ③ | TTE scatter: prospective (colormap) / post-event (orange) / no-data (grey) |
| ④ | Event density hexbin (all biopsies with outcome, adaptive gridsize) |
| ⑤ | Days post-transplant (biopsy timeline) |
| ⑥ | KM top vs bottom risk tertile (biopsy-level, prospective only) |
| ⑦ | CV split annotation |
| ⑧ | Avg TTE hexbin (prospective biopsies, falls back to all if <10) |

| File | Description |
|------|-------------|
| `{task}/biopsy_rep_umap.pdf/png` | Per-task 8-panel biopsy-level UMAP |

### 3c. Seed Attribution (LongMK multi-modal)

**Script:** `interpretability/interpret_longitudinal_mk.py` → `interpretability/submit_*.sh`

| File | Description |
|------|-------------|
| `{task}/multimod_seed_attribution_{task}.pdf/png` | Heatmap: K=16 seeds × modalities, showing which seeds attend to which modality. Sorted by seed diversity. |
| `agg/multimod_seed_attribution_all_tasks.pdf/png` | 4-panel combined across all tasks |
| `agg/Lpop_K_agg_{task}.pdf/png` | Aggregated seed attribution across population |
| `agg/longitudinal_mk_{variant}_Lpop_K_agg_{task}.pdf/png` | Variant-specific population aggregates |

### 3d. Global Temporal Weight Heatmap

**Script:** `interpretability/interpret_longitudinal_mk.py`

| File | Description |
|------|-------------|
| `{task}/L_global_weight_heatmap.pdf/png` | Per-split temporal attention heatmaps |
| `{task}/L_global_weight_heatmap_avg.pdf/png` | Averaged across splits |
| `{task}/L_global_learned_weights.pdf/png` | Learned recency decay weights |
| `agg/L_global_weight_heatmap_avg_all.pdf/png` | All tasks combined |

### 3e. Cluster Affinity (HE/BAL/CT)

**Script:** `interpretability/interpret_longitudinal_mk.py`

| File | Description |
|------|-------------|
| `cluster_agg/{task}_cluster_aff_agg.pdf/png` | Cluster-level affinity aggregated across patients. HE/BAL/CT clusters sorted biologically; shows which tissue/cell/radiomics clusters drive risk. |

### 3f. Clinical Feature Importance

**Script:** `analysis/plot_clinical_feature_imp.py` → `analysis/submit_benchmark.sh`

Derived from PMA attention weights for the Clinical modality. Each of 106 named feature tokens (e.g. `fev1`, `creatinine`, `tacrolimus`) gets a weighted affinity from the ABMIL seeds. Delta = mean_hi − mean_lo between high/low risk patients. Top 20 features by |delta| plotted.

| File | Description |
|------|-------------|
| `{task}/clinical_feature_imp_{task}.pdf/png` | Top-20 clinical features by risk-group delta, mean±std across 5 splits. Red=high-risk enriched, blue=low-risk enriched. All 4 tasks. |

---

## 4. Unimodal Ablation

**Script:** `analysis/plot_unimodal_ablation_v2.py` → `analysis/submit_benchmark.sh`
**LongMK data:** `interpretability/run_longmk_unimodal_ablation.py` → `interpretability/submit_longmk_unimodal_ablation.sh`

For each model variant and task: performance when only one modality's tokens are presented (others set to None). LongMK handles missing modalities natively. Results aggregated from 5 splits.

| File | Description |
|------|-------------|
| `interpretability/unimodal_ablation/unimodal_ablation_v2_{task}.pdf/png` | Grouped bar chart: modalities on x-axis, model variants as bars. Mean±std across splits, per-split dots overlaid. |
| `interpretability/unimodal_ablation/unimodal_ablation_v2_all.pdf/png` | Combined panel: all 4 tasks |
| `interpretability/unimodal_ablation/unimodal_ablation_v2_heatmap_{task}.pdf/png` | Heatmap variant: models × modalities |

---

## 5. Patient Trajectories

**Script:** `interpretability/plot_patient_trajectories.py`

| File | Description |
|------|-------------|
| `trajectories/Fig7_patient_trajectories.pdf/png` | 4-panel case study: longitudinal risk trajectories for selected patients (LT100, LT119, LT062, LT227). Shows biopsy-level risk evolution over time post-transplant. |
| `trajectories/panel_{A-D}_LT{id}.png` | Individual patient panels |

---

## Naming Conventions

- Tasks: `acr_cls` (ACR classification), `acr_surv` (ACR survival), `clad_surv` (CLAD survival), `death_surv` (Death survival)
- Models: `early` (early-fusion ABMIL), `longitudinal_mk_no_alibi` (LongMK), `longitudinal_mk_mt_no_alibi` (LongMK-MT)
- Splits: 5 outer CV splits (s0–s4), fold 0 for inner CV
- All figures have PDF (publication-quality) and PNG (preview) versions

---

## Clinical Feature Importance — Methodology Detail

Clinical feature importance is derived from the PMA (pooling-by-multihead-attention) attention weights within the LongMK model and aggregated across patients, biopsies, and cross-validation splits.

**Key distinction from HE/BAL/CT modalities:** HE, BAL, and CT affinities are computed over k-means cluster assignments (patch clusters → cluster-level affinity). Clinical has no patch clustering — each of the 106 feature tokens is a named variable (e.g. `fev1`, `fvc`, `CREATININE`, `tacrolimus`, `ALBUMIN`), so the PMA attention matrix is used directly without any cluster binning.

### Input representation

Each biopsy contributes 106 clinical feature tokens, one per variable. Each token is a 491-dimensional one-hot vector encoding the binned value of that variable. The PMA module maps these 106 tokens into K=16 learned seed vectors; the attention matrix for the Clinical modality at biopsy t is:

```
pa[k, f]  shape (K=16, 106)   — K seeds × 106 named feature tokens
```

### Per-patient weighted affinity

For each patient the ABMIL aggregator assigns each seed a scalar weight `alpha[k]` (sum to 1). The weighted affinity per feature f is:

```
weighted_aff[f] = sum_k( alpha[k] * pa[k, f] )
```

averaged over all biopsies in the patient's longitudinal timeline. This yields a scalar per feature representing how much the model, weighted by seed importance, attends to that clinical variable for this patient.

### Group delta

Patients are split into high-risk (`hi`) and low-risk (`lo`) groups by outcome (median TTE for survival tasks; ACR+ vs. ACR− for classification). The importance delta per feature is:

```
delta[f] = mean_hi( weighted_aff[f] ) − mean_lo( weighted_aff[f] )
```

Positive delta → feature receives more attention in high-risk patients; negative → low-risk enriched.

### Cross-split aggregation and plot

The delta vector is computed independently for each of the 5 outer CV splits and aggregated as:

```
delta_mean[f] ± delta_std[f]   (mean ± s.d. across 5 splits)
```

The top-20 features by |delta_mean| are plotted as a horizontal bar chart with error bars. Red bars = enriched in high-risk; blue = enriched in low-risk.
