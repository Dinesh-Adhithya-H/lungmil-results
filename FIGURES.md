# Figures Reference

All figures are in `figures/`. PDF and PNG are generated for every plot. The scripts that produce each group are listed; all must be run via `sbatch` (never directly on the login node).

**Cohort:** n=263 patients, 4210 biopsies. 5 outer CV splits (s0–s4), fold 0 for inner CV.

---

## 1. Benchmark

**Script:** `analysis/plot_benchmark.py` → `analysis/submit_benchmark_v2.sh`

Horizontal bar charts. All 18 models in fixed order: linear baselines → unimodal ABMIL → fusion → SetMIL → LongMK. Colour-coded model groups (grey/blue/teal/purple/red bands), per-split dots overlaid, chance line at 0.5 in legend. Best multimodal model per task marked ★ (star, fontsize 14). x-axis fixed 0–1 for all tasks.

| File | Description |
|------|-------------|
| `benchmark/benchmark_{task}.pdf/png` | Single-task bar chart. Tasks: `acr_cls` (BACC), `acr_surv` (C-index), `clad_surv` (C-index), `death_surv` (C-index). |
| `benchmark/benchmark_all.pdf/png` | 4-panel combined; legend in reserved bottom margin (10% of figure height) to avoid overlap with x-axis labels. |
| `benchmark/benchmark_table_v2.pdf/png` | Tabular summary: rows = models, columns = tasks, cells = mean±std. |

**Best results:** ACR cls 0.623±0.034 (SetMIL-MT no SAB) · ACR surv 0.679±0.064 (LongMK) · CLAD 0.563±0.080 (SetMIL-MT) · Death 0.771±0.056 (LongMK).

**Script:** `analysis/plot_benchmark_multimodal.py` → `analysis/submit_benchmark.sh`

| File | Description |
|------|-------------|
| `benchmark/benchmark_multimodal_{task}.pdf/png` | Multimodal model family comparison only (no linear/ABMIL unimodal). P1 ABMIL ensemble vs P2 fusion vs SetMIL vs LongMK. |

**Script:** `analysis/plot_benchmark_pvalues.py` → `analysis/submit_benchmark.sh`

| File | Description |
|------|-------------|
| `benchmark/benchmark_pvalues_longmk.pdf/png` | Heatmap: one-sided paired t-test p(LongMK > model) for each (model, task) pair. Rows = 17 other models, cols = 4 tasks. LogNorm colormap (vmin=0.001, vmax=1.0); green=significant (p<0.05), red=not significant. Cell text: p-value + significance stars. |

**Script:** `analysis/plot_modality_combo_ablation.py` → `analysis/submit_benchmark.sh`

| File | Description |
|------|-------------|
| `benchmark/modality_combo_{task}.pdf/png` | Modality combination ablation: all 2^4=16 subsets of {HE, BAL, CT, Clinical}. Bar chart per task showing performance by modality combination. Identifies which modality combinations are synergistic vs redundant. |

---

## 2. Kaplan-Meier Curves

**Script:** `analysis/plot_km_from_model.py` → `analysis/submit_km.sh`

| File | Description |
|------|-------------|
| `km_curves/km_{task}.pdf/png` | KM curves stratified by LongMK patient-level risk (top vs bottom tertile). Greenwood 95% CI shading. Log-rank p-value annotated. Tasks: `acr_cls`, `acr_surv`, `clad_surv`, `death_surv`. |
| `km_curves/km_all_tasks.pdf/png` | 4-panel combined KM overview. |

---

## 3. Interpretability — Per-Task

All per-task plots are in `figures/interpretability/{task}/` where `task` ∈ {`acr_cls`, `acr_surv`, `clad_surv`, `death_surv`}.

### 3a. Patient-Rep UMAP (unified)

**Script:** `analysis/plot_unified_rep_umap.py` → `analysis/submit_unified_rep_umap.sh`

8-panel 2×4 figure showing patient-level representation space for each task. LongMK reps for `acr_surv`/`death_surv`; SetMIL-MT reps for `acr_cls`/`clad_surv`.

| Panel | Content |
|-------|---------|
| ① | Risk score scatter (rank-percentile 0–1) |
| ② | Event indicator / true label scatter |
| ③ | TTE scatter + ×/○ event/censored markers |
| ④ | Event density hexbin (0–1 normalised; colorbar "Event rate (0=none, 1=all)") |
| ⑤ | Modality combination scatter |
| ⑥ | KM top vs bottom risk tertile with log-rank p-value |
| ⑦ | CV split annotation |
| ⑧ | Avg TTE hexbin |

| File | Description |
|------|-------------|
| `{task}/unified_rep_umap_{task}.pdf/png` | Per-task 8-panel figure |
| `agg/unified_rep_umap_all_tasks.pdf/png` | 2×2 overview (risk score coloring, all 4 tasks) |

### 3b. Biopsy-Rep UMAP

**Script:** `analysis/plot_biopsy_rep_umap_km.py` → `analysis/submit_biopsy_rep_umap_km.sh`

8-panel 2×4 figure at the **biopsy level** — each point is one biopsy from the LongMK 256-dim representation space (extracted by `interpretability/extract_biopsy_reps.py`). Predictions use all past visits up to that biopsy. `tte = event_date − biopsy_date`; prospective biopsies (tte>0) coloured, post-event (tte<0) shown in orange, no-outcome in grey.

Scale: ACR surv 4210 biopsies / 263 patients; death surv 3564 biopsies.

| Panel | Content |
|-------|---------|
| ① | Biopsy-level risk score (percentile rank 0–1) |
| ② | Biopsy timing vs event (red=pre-event prospective, orange=post-event, blue=censored) |
| ③ | TTE scatter: prospective (colormap) / post-event (orange) / no-data (grey) |
| ④ | Event density hexbin — **0–1 normalised scale** (colorbar "Event rate (0=none, 1=all)"); no median-centring |
| ⑤ | Days post-transplant (biopsy timeline) |
| ⑥ | KM top vs bottom risk tertile (biopsy-level, prospective only); log-rank p-value formatted as p<0.001 (=X.XXe-N), p=X.XXX, or p=X.XXX (n.s.) |
| ⑦ | CV split annotation |
| ⑧ | Avg TTE hexbin (prospective biopsies, falls back to all if <10) |

| File | Description |
|------|-------------|
| `{task}/biopsy_rep_umap.pdf/png` | Per-task 8-panel biopsy-level UMAP |

### 3c. Seed Attribution (LongMK multi-modal)

**Script:** `interpretability/interpret_longitudinal_mk.py` → `interpretability/submit_*.sh`

| File | Description |
|------|-------------|
| `{task}/multimod_seed_attribution_{task}.pdf/png` | Heatmap: K=16 seeds × modalities, showing which seeds attend to which modality. Sorted by seed diversity score. All 5 splits overlaid as separate rows. |
| `agg/multimod_seed_attribution_all_tasks.pdf/png` | 4-panel combined across all tasks |
| `agg/Lpop_K_agg_{task}.pdf/png` | Aggregated seed attribution across population |

### 3d. Global Temporal Weight Heatmap

**Script:** `interpretability/interpret_longitudinal_mk.py`

Learned biopsy-weighting surface: 2D heatmap of weight(current_prediction_day, prior_biopsy_day). Reveals which temporal windows drive each task without any temporal supervision.

Key findings: ACR surv concentrates in early window (<350d); ACR cls concentrates in late window (>350d); death suppresses peri-operative window (<50d); CLAD near-uniform.

| File | Description |
|------|-------------|
| `{task}/L_global_weight_heatmap.pdf/png` | Per-split temporal weight heatmaps |
| `{task}/L_global_weight_heatmap_avg.pdf/png` | Averaged across 5 splits |
| `{task}/L_global_learned_weights.pdf/png` | Learned scalar weights per biopsy position |
| `agg/L_global_weight_heatmap_avg_all.pdf/png` | All 4 tasks combined |

### 3e. Cluster Affinity — HE / BAL / CT / Clinical

**Script:** `interpretability/gen_cluster_aff_agg.py` → `interpretability/submit_cluster_aff.sh`

Cross-split aggregated cluster-level attribution for the best model per task. **Now includes Clinical modality panel.** All 4 modalities share the same x-axis scale for direct comparison.

- **HE**: biological cluster labels from `results/cluster_name_maps/HE_cluster_map.json`; clusters sorted by biological category. Top 14 clusters by |delta| shown.
- **BAL**: cell-type clusters (TRAM, neutrophil-enriched, lymphocyte-enriched, etc.)
- **CT**: structural/radiomics clusters (mosaic attenuation, bronchiectasis, preserved parenchyma)
- **Clinical**: expanded feature names (e.g. "FVC% (% predicted)", "GFR (glomerular filtration rate)", "PGD score at 72h"). Top 14 clinical features by |delta| shown.

Shared x-axis: `xlim = ±max(|delta|+std) × 1.15` across all modalities. Red bars = enriched in high-risk group; blue = enriched in low-risk group. n=5 splits annotated per panel.

| File | Description |
|------|-------------|
| `cluster_agg/{task}_cluster_aff_agg.pdf/png` | 4-panel (HE/BAL/CT/Clinical) cluster attribution for the best model per task. |

**Best model per task:** ACR cls → SetMIL-MT (no SAB) · ACR surv → LongMK · CLAD → SetMIL-MT · Death → LongMK.

**Key findings by task:**
- **Death survival**: TRAM (tissue-resident alveolar macrophages, BAL) strongly enriched in low-risk (survivors); alveolar with haemorrhage and inflammation (HE) high-risk; CT structural deterioration high-risk; Clinical: FVC%, albumin, GFR low-risk; donor risk score, PGD at 72h, RDW high-risk.
- **ACR classification**: lymphocytoplasmic inflammation (HE) high-risk; preserved alveolar histology low-risk; clinical markers of immune activation high-risk.
- **CLAD survival**: neutrophil-enriched BAL clusters high-risk; mosaic CT patterns high-risk; FVC% low-risk.
- **ACR survival**: early alveolar inflammation high-risk; TRAM low-risk; donor risk score high-risk.

### 3f. Clinical Feature Importance

**Script:** `analysis/plot_clinical_feature_imp.py`

Derived from PMA attention weights for the Clinical modality. See **Clinical Feature Importance — Methodology Detail** below.

| File | Description |
|------|-------------|
| `{task}/clinical_feature_imp_{task}.pdf/png` | Top-20 clinical features by risk-group delta (mean±std across 5 splits). Red=high-risk enriched, blue=low-risk enriched. All 4 tasks. |

### 3g. LongMK Per-Patient Summaries

**Script:** `interpretability/interpret_longitudinal_mk.py`

Generated for every patient × every split × every task. Stored in `interpretability/longitudinal_mk_interp/{variant}_split{s}_fold0_{task}/`.

| File | Description |
|------|-------------|
| `L0_summary_pid{patient_id}.png` | Per-patient L0 summary: risk trajectory over all biopsies, learned temporal weight profile, seed-modality attribution heatmap per biopsy. |
| `L1_seed_timeline_pid{patient_id}.png` | Per-patient L1 seed timeline: K=16 seed contributions over time, coloured by modality. Shows which seeds dominate at each biopsy visit. |

---

## 4. Unimodal Ablation

**Script:** `analysis/plot_unimodal_ablation_v2.py` → `analysis/submit_unimodal_ablation_v2.sh`
**LongMK data:** `interpretability/run_longmk_unimodal_ablation.py`

For each model variant and task: performance when only one modality's tokens are presented (others set to None). LongMK handles missing modalities natively via modal dropout. Results aggregated from 5 splits. **y-axis fixed 0–1 for all panels** (BACC and C-index both have chance at 0.5).

| File | Description |
|------|-------------|
| `interpretability/unimodal_ablation/unimodal_ablation_v2_{task}.pdf/png` | Grouped bar chart: modalities on x-axis, model variants as grouped bars. Mean±std across splits, per-split dots overlaid. y-axis 0–1. |
| `interpretability/unimodal_ablation/unimodal_ablation_v2_all.pdf/png` | Combined panel: all 4 tasks, same y-axis scale. |
| `interpretability/unimodal_ablation/unimodal_ablation_v2_heatmap_{task}.pdf/png` | Heatmap variant: models × modalities, 0–1 colorscale. |

---

## 5. Patient Trajectories

**Script:** `interpretability/plot_patient_trajectories.py`

| File | Description |
|------|-------------|
| `trajectories/Fig7_patient_trajectories.pdf/png` | Illustrative case studies: longitudinal risk trajectories from LongMK. Shows biopsy-level risk evolution over post-transplant time, with event markers (CLAD, death, ACR+). |

Interactive versions of illustrative cases (LT070, LT073, LT038) are available in the patient explorer website (page 9 — Illustrative Cases).

---

## Naming Conventions

- **Tasks:** `acr_cls` (ACR binary classification), `acr_surv` (time-to-next-ACR), `clad_surv` (time-to-CLAD), `death_surv` (overall survival)
- **Best models:** `longitudinal_mk_no_alibi` (LongMK) → ACR surv + Death surv; `set_mil_mt_no_sab` → ACR cls; `set_mil_mt` → CLAD surv
- **Splits:** 5 outer CV splits (s0–s4), fold 0 inner CV; all test sets non-overlapping
- **All figures:** PDF (publication-quality vector) + PNG (preview/website); 150–300 dpi

---

## Clinical Feature Importance — Methodology Detail

Clinical feature importance is derived from the PMA (pooling-by-multihead-attention) attention weights within the LongMK model, aggregated across patients, biopsies, and cross-validation splits.

**Key distinction from HE/BAL/CT modalities:** HE, BAL, and CT affinities are computed over k-means cluster assignments (patch clusters → cluster-level affinity). Clinical has no patch clustering — each of the 106 feature tokens is a named variable (e.g. `fvc`, `gfr`, `pgd_t72`, `albumin`, `rdw`), so the PMA attention matrix is used directly without cluster binning.

### Input representation

Each biopsy contributes 106 clinical feature tokens. Each token is a 491-dimensional one-hot vector encoding the binned value of that variable. The PMA module maps these 106 tokens into K=16 learned seed vectors; the attention matrix at biopsy t is:

```
pa[k, f]   shape (K=16, 106) — K seeds × 106 named feature tokens
```

### Per-patient weighted affinity

For each patient the ABMIL aggregator assigns each seed a scalar weight `alpha[k]` (sum to 1). The weighted affinity per feature f is:

```
weighted_aff[f] = sum_k( alpha[k] * pa[k, f] )
```

averaged over all biopsies in the patient's longitudinal timeline.

### Group delta

Patients are split into high-risk (`hi`) and low-risk (`lo`) groups by outcome (median TTE for survival tasks; ACR+ vs. ACR− for classification):

```
delta[f] = mean_hi( weighted_aff[f] ) − mean_lo( weighted_aff[f] )
```

Positive delta → feature attends more in high-risk patients. Negative → low-risk enriched.

### Cross-split aggregation

```
delta_mean[f] ± delta_std[f]   (mean ± s.d. across 5 splits)
```

Top features by |delta_mean| plotted. The same delta is used in the cluster_aff_agg Clinical panel (top 14 features) and clinical_feature_imp plot (top 20 features).

### Key findings

| Task | High-risk features (positive delta) | Low-risk features (negative delta) |
|------|--------------------------------------|-------------------------------------|
| Death survival | Donor risk score, PGD at 72h, RDW | FVC% (% predicted), Albumin, GFR |
| ACR classification | Lymphocytic/immune activation markers | Stable spirometry, low inflammation |
| CLAD survival | PGD at 72h, donor risk | FVC (L), FVC% |
| ACR survival | Donor risk score | FVC%, GFR |
