# Figures — Longitudinal Multimodal MIL, Lung Transplant

Evaluation: 5-split × 4-fold nested cross-validation. Best model per task = fold 0 test set.  
All PNGs have matching PDFs (300 dpi, vector).

---

## Sample Size Note

**N=4210** — biopsy-level models (SetMIL: `set_mil_mt`, `set_mil_mt_no_sab`, early/late/middle fusions). One prediction per biopsy record.  
**N=226** — patient-level models (`longitudinal_mk_no_alibi`). One prediction per unique patient (full biopsy sequence → single risk score). ACR surv and Death survival use this model.

---

## Best Model per Task

| Task | Metric | Best model | Score (mean ± std) | Linear baseline | N |
|------|--------|------------|-------------------|-----------------|---|
| **Death survival** | C-index ↑ | `longitudinal_mk_no_alibi` | **0.771 ± 0.056** | 0.580 | 226 patients |
| **ACR survival** | C-index ↑ | `longitudinal_mk_no_alibi` | **0.679 ± 0.064** | 0.587 | 226 patients |
| **CLAD survival** | C-index ↑ | `set_mil_mt` | **0.563 ± 0.080** | 0.501 | 4210 biopsies |
| **ACR classification** | BACC ↑ | `set_mil_mt_no_sab` | **0.623 ± 0.034** | 0.588 | 4210 biopsies |

---

## Figure Checklist

### 1 — Patient representation UMAPs (test set, best model per task)

| # | File | Task | Status |
|---|------|------|--------|
| 1a | `interpretability/acr_cls/unified_rep_umap_acr_cls.png` | ACR cls | ✅ |
| 1b | `interpretability/acr_surv/unified_rep_umap_acr_surv.png` | ACR surv | ✅ |
| 1c | `interpretability/clad_surv/unified_rep_umap_clad_surv.png` | CLAD | ✅ |
| 1d | `interpretability/death_surv/unified_rep_umap_death_surv.png` | Death | ✅ |
| 1e | `interpretability/agg/unified_rep_umap_all_tasks.png` | All 4 (2×2 grid) | ✅ |

---

#### How made — unified patient representation UMAPs

Script: `analysis/plot_unified_rep_umap.py` via `analysis/submit_unified_rep_umap.sh` (cpu_p, 350G, 8 CPUs).  
350G is required because the SetMIL `results_raw.npy` file (covering all 4210 biopsies across all splits) is ~40GB and must be loaded into memory.

**Step 1 — What the representation is**

The 256-dimensional vector fed into UMAP is the ABMIL output representation — the weighted sum of PMA seed vectors after attention pooling.  
Specifically, each patient/biopsy's representation = Σ_k (α_k × seed_k), where α_k is the learned ABMIL attention weight for seed k and seed_k is the 256-dim PMA prototype vector. This is the same vector the final classification/survival head reads from, so it is the most task-informative summary of the patient's multimodal data.

**Step 2 — Loading representations**

Source files per task (keyed by best model):

| Task | Model | Source `results_raw.npy` |
|------|-------|-------------------------|
| ACR cls | `set_mil_mt_no_sab` | `set_mil_mt_interp/all_splits_cls/results_raw.npy` (single file, all 5 splits pooled) |
| CLAD surv | `set_mil_mt` | `set_mil_mt_interp/all_splits_clad_surv/results_raw.npy` |
| ACR surv | `longitudinal_mk_no_alibi` | `longitudinal_mk_interp/longitudinal_mk_no_alibi_split{0-4}_fold0_acr_surv/results_raw.npy` (5 separate files, merged) |
| Death surv | `longitudinal_mk_no_alibi` | `longitudinal_mk_interp/longitudinal_mk_no_alibi_split{0-4}_fold0_death_surv/results_raw.npy` |

**Critical design: representations are always out-of-sample**

Every patient representation in the UMAP was produced by a model that **never saw that patient during training**.

- **SetMIL tasks:** `extract_all_splits()` in `interpret_set_mil_mt.py` loops over splits 0–4. For each split *s* it (1) loads the checkpoint trained on `split{s}_fold0` (trained on train+val of split *s*), (2) runs inference on `splits["test"]` — the held-out patients for that split only, (3) tags each record with `_split = s`. All 5 test sets are pooled → `all_splits_{task}/results_raw.npy`. Each patient (or biopsy) appears **exactly once**, always as a test sample. Total ≈ 4210 biopsy records across 5 splits.

- **Longitudinal tasks:** 5 separate `results_raw.npy` files, one per `longitudinal_mk_no_alibi_split{s}_fold0_{task}/`. Each is produced by `interpret_longitudinal_mk.py --split {s} --fold 0`, which loads that split's model and evaluates on that split's test patients only. The UMAP script merges all 5 files. Total = 226 unique patients, each appearing once.

- **Panel ⑦ CV split annotation** lets you visually verify this: each colour = one outer split. If representations were not out-of-sample, we would expect within-split clustering driven by model-specific bias — absence of such structure confirms proper evaluation.

SetMIL npy structure: list of dicts with keys `final_reps` (dict by internal task name), `logits`, `label`, `present_mods`, `_split`.  
Longitudinal npy structure: slim dicts with `rep_full` (dict by internal task name), `logits`, `patient_id`, `_split` — but **no patient-level outcome fields** (event, TTE are `None` because biopsy-level records don't carry them).

**Step 3 — Patient outcome enrichment (longitudinal models)**

Because `longitudinal_mk_no_alibi` npy records have `event_*` and `tte_*` as `None`, outcomes are joined from the splits CSV (`multimodal_splits_nested_cv.csv`) by `patient_id`:  
```
lookup[patient_id] = {
    event_acr, tte_acr, event_clad, tte_clad, event_death, tte_death,
    present_mods (derived from has_HE / has_BAL / has_CT / has_Clinical columns)
}
```
For each record, `_get(key)` first checks the npy dict, then falls back to the lookup. This ensures all panels have valid outcome data even for longitudinal patients.

**Internal representation key mapping** (critical — must match `TASK_GROUPS` in `builders.py`):  
- `clad_surv` → `rep_key = "clad"` (SetMIL head registered as "clad" not "clad_surv")  
- `death_surv` → `rep_key = "death"` (longitudinal head registered as "death")  
- `acr_surv` → `rep_key = "acr_surv"` (default TASK_GROUP key)  
- `acr_cls` → `rep_key = "acr_cls"`

**Step 4 — UMAP embedding**

Representations are StandardScaler-normalised (zero mean, unit variance per dimension) before embedding.  
UMAP parameters: `n_neighbors=min(15, N-1)`, `min_dist=0.1`, `metric="cosine"`, `random_state=42`.  
Cosine metric is used because the representations are attention-weighted sums of L2-normalised seed vectors, so direction matters more than magnitude.

**Step 5 — Risk score computation**

- Classification (ACR cls): score = sigmoid(logit) → P(ACR+) in [0, 1].  
- Survival (ACR surv, CLAD, Death): score = percentile rank of logit across all test patients → [0, 1]. Percentile rank is used so the colour scale is uniform regardless of logit magnitude.

**Step 6 — Eight panels (2×4 layout, all cells filled)**

| Panel | What is shown | Colourmap | Notes |
|-------|--------------|-----------|-------|
| ① Risk score | Sigmoid(logit) for cls; percentile rank of logit for surv | RdBu_r (blue=low risk, red=high risk) | Percentile rank used for survival so colour scale is uniform regardless of logit magnitude |
| ② Event / label | **Survival:** event indicator (1=event, 0=censored). **Classification:** true class label (1=ACR+, 0=ACR-) | RdBu_r | Task-aware: surv uses event status, cls uses true label |
| ③ Time to event | Continuous TTE in years (clipped at 95th pct). **Survival:** `×` = events, `○` = censored overlaid. **Classification:** event/censored colour only | plasma_r (bright=short TTE=urgent) | |
| ④ Event density | Hexbin of event rate per hexagon: mean(event indicator) per hex cell, median-centred | RdBu_r (red=above-median event rate) | `reduce_C_function=np.mean` on event indicator; `vmin/vmax` set symmetrically around median of occupied hexagons; colorbar label shows median value |
| ⑤ Modality combination | Unique modality combinations present per patient (HE+BAL+CT, HE+Clinical, etc.); combos with <5 patients → "Other" (grey) | tab20 | |
| ⑥ KM: top vs bottom risk tertile | Kaplan-Meier curves: top tertile (score ≥ 67th pct) red vs bottom tertile (score ≤ 33rd pct) blue; middle excluded to maximise separation | — | Log-rank p computed inline |
| ⑦ CV split annotation | Which of the 5 outer CV splits each patient came from; absence of within-split clustering confirms out-of-sample integrity | 5 fixed colours | |
| ⑧ Avg TTE per hexagon | Mean TTE (years) per hex cell, median-centred | RdBu (blue=above-median TTE=safer, red=below-median=urgent) | Same `reduce_C_function=np.mean` approach; median-centred diverging scale so above/below average urgency is immediately visible |

The combined 4-panel overview (`agg/unified_rep_umap_all_tasks.png`) shows only panel ① (risk score) for all 4 tasks in a 2×2 grid.

---

### 2 — Instance → seed → disease (attention chain)

| # | File | Task | What | Status |
|---|------|------|------|--------|
| 2a | `interpretability/acr_cls/A_instance_reps_cosine.png` | ACR cls | UMAP of all patch instances, coloured by modality | ✅ |
| 2b | `interpretability/acr_cls/B_seeds.png` | ACR cls | Seed prototypes (★) in instance space | ✅ |
| 2c | `interpretability/acr_cls/D_abmil_seed_importance.png` | ACR cls | Mean ABMIL α per seed | ✅ |
| 2d | `interpretability/acr_cls/K_seed_attribution_acr_cls.png` | ACR cls | Δα high vs low risk, per seed | ✅ |
| 2e | `interpretability/acr_cls/I_seed_risk_stratification.png` | ACR cls | Seed UMAP coloured by risk group | ✅ |
| 2f | `interpretability/acr_cls/E_task_modal_gate.png` | ACR cls | Learned modality gate weights | ✅ |
| 2g | `interpretability/acr_cls/F_modality_combo_ablation.png` | ACR cls | BACC for all 2⁴ modality subsets | ✅ |
| 2h | `interpretability/agg/Lpop_K_agg_acr_cls.png` | ACR cls | Δα 5-split aggregate | ✅ |
| 2i | `interpretability/agg/Lpop_K_agg_acr_surv.png` | ACR surv | Δα 5-split aggregate | ✅ |
| 2j | `interpretability/agg/Lpop_K_agg_clad_surv.png` | CLAD | Δα 5-split aggregate | ✅ |
| 2k | `interpretability/agg/Lpop_K_agg_death_surv.png` | Death | Δα 5-split aggregate | ✅ |
| 2l | `interpretability/acr_surv/multimod_seed_attribution_acr_surv.png` | ACR surv | Per-modality Δα bars (HE/BAL/CT) | ✅ |
| 2m | `interpretability/clad_surv/multimod_seed_attribution_clad_surv.png` | CLAD | Per-modality Δα bars | ✅ |
| 2n | `interpretability/death_surv/multimod_seed_attribution_death_surv.png` | Death | Per-modality Δα bars | ✅ |
| 2o | `interpretability/agg/multimod_seed_attribution_all_tasks.png` | All 3 surv | Combined 3-task × 3-modality grid | ✅ |
| 2p | `interpretability/cluster_agg/death_cluster_aff_agg.png` | Death | Named biological cluster affinity | ✅ |
| 2q | `interpretability/cluster_agg/acr_surv_cluster_aff_agg.png` | ACR surv | Named biological cluster affinity | ✅ |
| 2r | `interpretability/cluster_agg/acr_cls_cluster_aff_agg.png` | ACR cls | Named biological cluster affinity | ✅ |
| 2s | `interpretability/cluster_agg/clad_cluster_aff_agg.png` | CLAD | Named biological cluster affinity | ✅ |
| **2t** | **A+B+D panels for acr_surv + death** | ACR surv, Death | Instance+seed UMAPs for longitudinal model | ❌ MISSING |

---

#### How made — 2a–2g (instance + seed panels, SetMIL tasks)

Script: `interpretability/interpret_set_mil_mt.py` — GPU job (~30 min per task/split).  
- Loads `set_mil_mt_no_sab` (ACR cls) or `set_mil_mt` (CLAD) checkpoint from the best split.  
- Runs a forward pass on the full test set to collect: PMA seed vectors (K=16 per modality × 4 modalities = 64 seeds), ABMIL attention weights α per seed, final patient representations (256-dim ABMIL output).  
- UMAP of all instance feature vectors (patches / cells / CT voxels) with cosine metric → instance space for panels A and B.  
- Seeds overlaid in instance space at their B-cos attention-weighted centroid (panel B); seed size ∝ mean α across test patients.  
- ABMIL α per seed averaged over patients → bar chart (panel D).  
- Δα = mean α(predicted high-risk) − mean α(predicted low-risk) per seed → bar chart (panel K).

---

#### How made — 2h–2k (Lpop_K_agg — aggregated seed attribution, all tasks)

Script: `interpretability/gen_cluster_aff_agg.py` (also handles seed agg from `seed_attribution_data_*.json`).  
Source files: `{variant}_split{0-4}_fold0_{task}/seed_attribution_data_{task}.json` — one JSON per split containing:  
- `seed_labels`: list of 48 strings with modality prefix (`HE·s00 … HE·s15`, `BAL·s00 … BAL·s15`, `CT·s00 … CT·s15`).  
- `alpha_diff`: list of 48 floats, Δα per seed (precomputed inside the interpretability pipeline as mean_hi − mean_lo).  

Aggregation: stack Δα arrays across 5 splits → mean and std per seed. Plot as horizontal bar chart sorted by |Δα|, colour-coded by sign.

---

#### How made — 2l–2o (multimod_seed_attribution — per-modality Δα, all surv tasks)

Script: `analysis/plot_multimod_seed_attribution.py` — CPU job (sbatch cpu_p, 8G, ~5 min).  
Same source JSONs as Lpop_K_agg (`seed_attribution_data_{task}.json`, 5 splits).  
Additional step: seeds are **grouped by modality prefix** (split on `·`), so HE·s00–HE·s15 form one panel, BAL·s00–BAL·s15 another, CT·s00–CT·s15 a third. Each modality panel shows its 16 seeds as horizontal bars sorted by |Δα|. Combined 3-task × 3-modality summary figure in `agg/multimod_seed_attribution_all_tasks.png`.

---

#### How made — 2p–2s (cluster_aff_agg — named biological cluster attribution)

Script: `interpretability/gen_cluster_aff_agg.py` — CPU job.  
Output: `figures/interpretability/cluster_agg/{task}_cluster_aff_agg.png` — one figure per task, one panel per modality (HE, BAL, CT), top 14 clusters by |Δ affinity|.

**What is cluster affinity?**  
Each instance (patch, cell, CT voxel) is assigned to a named biological cluster (unsupervised k-means or published annotation). For a given patient/biopsy, the **cluster affinity** for cluster *c* in modality *m* = sum of PMA B-cos attention weights over all instances in that cluster. It measures how strongly the model focuses on tissue type *c* when making its prediction.  
Δ affinity = mean cluster affinity in the high-risk group − mean cluster affinity in the low-risk group.  
Positive Δ → this cluster/tissue type is more attended to in high-risk patients → model uses it as a risk signal.

**Data sources by model family:**

*SetMIL (ACR cls, CLAD):*  
- Source: `set_mil_mt_interp/{variant}_split{s}_fold0_{task}/paper_interp_data.json`  
- This JSON has a `tasks.{task}.cluster_affinity.{modality}` dict containing `delta` (already computed as hi − lo mean affinity per cluster) and `cluster_names`.  
- Loaded for splits 0–4, stacked into arrays, mean and std computed across splits.

*Longitudinal (ACR surv, Death):*  
- Source: `longitudinal_mk_interp/{variant}_split{s}_fold0_{task}/cluster_aff_data_{task}.json`  
- This JSON has `cluster_aff.{modality}.hi` (list of per-patient affinity vectors, high-risk group), `.lo` (low-risk group), and `.names` (cluster names).  
- Δ computed per split: `hi.mean(axis=0) − lo.mean(axis=0)`, then stacked across 5 splits → mean ± std.

**HE cluster name mapping:**  
HE cluster IDs are mapped to biological tissue names via `results/cluster_name_maps/HE_cluster_map.json`:  
- "Alveolar with hemorrhage and inflammation" (red)  
- "Alveolar with empty spaces" (orange)  
- "Alveolar" (green)  
- "Bronchial" (blue)  
- "Lymphocytoplasmic inflammation" (purple)  
- "Cartilage" (brown)  

BAL cluster names = published cell-type annotations (e.g. macrophages, neutrophils, lymphocytes, TRAM).  
CT cluster names = CT morphology-based labels.  

**Plot layout:** one panel per modality. X-axis = Δ cluster affinity. Top 14 clusters by |Δ| shown. Red bars = enriched in high-risk; blue = enriched in low-risk. Error bars = std across 5 CV splits.

---

### 3 — Benchmark (all models, all tasks)

#### 3a — Main benchmark bar plots (v2, current)

| # | File | What | Status |
|---|------|------|--------|
| 3a | `benchmark/benchmark_v2_acr_cls.png` | 18 models vs ACR cls | ✅ |
| 3b | `benchmark/benchmark_v2_acr_surv.png` | 18 models vs ACR surv | ✅ |
| 3c | `benchmark/benchmark_v2_clad_surv.png` | 18 models vs CLAD | ✅ |
| 3d | `benchmark/benchmark_v2_death_surv.png` | 18 models vs Death | ✅ |
| 3e | `benchmark/benchmark_v2_all.png` | 4-panel combined | ✅ |

Script: `analysis/plot_benchmark_v2.py` via `analysis/submit_benchmark_ablation_v2.sh`.

**Fixed model order (18 models):**

| Group | Models |
|-------|--------|
| Linear baselines | Linear HE, Linear BAL, Linear CT, Linear Clinical, wt avg Linear |
| ABMIL unimodal | ABMIL HE, ABMIL BAL, ABMIL CT, ABMIL Clinical, wt avg ABMIL |
| Fusion | Early fusion, Middle fusion, Late fusion |
| SetMIL | SetMIL, SetMIL-MT, SetMIL-MT (no SAB) |
| LongitudinalMK | LongMK-MT, LongMK |

**Design choices:**
- Each model has a unique colour within its group family (greys → blues → teals → purples → reds); colour is identical across all 3 benchmark figures (defined in `SHARED_MODEL_COLORS`).
- No best-model highlighting — readers rank visually from bar length.
- Per-split dots (white fill, coloured edge) show cross-split variability on each bar.
- Error bars = ± std across 5 splits. Dashed vertical at 0.5 = chance level.
- Y-axis inverted: best bar at top.

**Data loading:**
- Linear baselines: read from `results/linear_models/metrics_summary.csv` (`load_linear()` function). Modality map: H&E → "Linear HE", All → "wt avg Linear".
- All other models: per-split JSON files `results/mm_abmil_v8/metrics_split{s}_fold0_{variant}_{suffix}.json`.
  - Flat JSON for non-longi models: `d["test"]["c_index"]`.
  - Nested JSON for longi models: `d["test"]["acr_surv"]["c_index"]`, `d["test"]["clad"]["c_index"]`, `d["test"]["death"]["c_index"]`.
- CSV_TO_DISPLAY maps legacy CSV model names (e.g. "P1 wtd ensemble") to display labels ("wt avg ABMIL").

---

#### 3b — Unimodal ablation (v2)

| # | File | What | Status |
|---|------|------|--------|
| 6a | `interpretability/unimodal_ablation/unimodal_ablation_v2_{task}.png` | Grouped bars per modality, all models | ✅ |
| 6b | `interpretability/unimodal_ablation/unimodal_ablation_v2_heatmap_{task}.png` | Model×modality heatmap | ✅ |
| 6c | `interpretability/unimodal_ablation/unimodal_ablation_v2_all.png` | 4-task combined | ✅ |

Script: `analysis/plot_unimodal_ablation_v2.py`.

**What unimodal ablation shows:** For each model, what metric is achieved when only ONE modality is active (others zeroed at the feature level). This is different from training unimodal-only models — the same jointly-trained weights are used but with zeroed input for the suppressed modalities.

**Data source:** `unimodal_ablation` block inside each metrics JSON:
```json
"unimodal_ablation": { "HE": {"bacc": 0.58}, "BAL": {"c_index": 0.61}, ... }
```
Longitudinal models (`longitudinal_mk_*`) do not have this block — they appear only in the "All modalities" column of the combo plot.

**Layout:** Grouped bar chart. X-axis = modalities (HE, BAL, CT, Clinical). Within each group, one bar per model in fixed MODEL_ORDER. Same SHARED_MODEL_COLORS. Heatmap version: rows = models, columns = modalities, colour = mean metric value.

---

#### 3c — Modality combination ablation

| # | File | What | Status |
|---|------|------|--------|
| 7a | `benchmark/modality_combo_{task}.png` | Single mods + "All 4" for all models | ✅ |
| 7b | `benchmark/modality_combo_all.png` | 4-task 2×2 combined | ✅ |

Script: `analysis/plot_modality_combo_ablation.py`.

**What it shows:** For each model, performance with each single modality alone (HE, BAL, CT, Clinical) vs with all 4 combined. Reveals which modalities matter most per model and whether multi-modal fusion actually helps vs any single modality.

**Data loading:** Same `unimodal_ablation` block for per-modality, `d["test"]` for the All-4 value. Longitudinal models: per-modality = all NaN (no ablation data), All-4 extracted from nested JSON using `longi_key` mapping (`clad_surv` → `"clad"`, `death_surv` → `"death"`). This is why longitudinal models only appear in the "All 4" column.

**X-axis:** HE | BAL | CT | Clinical | All 4 (separator line before last column). Error bars and per-split dots shown. Same SHARED_MODEL_COLORS as main benchmark.

---

#### How made — benchmark figures (legacy v1)

Script: `analysis/plot_benchmark_multimodal.py` (older version, single-colour per group, amber best-model border, different label names). Kept for reference only; v2 is the current version.

P2 variant display names (v1):  
`early` → "Early fusion", `middle` → "Middle fusion", `late` → "Late fusion",  
`set_mil_no_sab` → "SetMIL", `set_mil_mt` → "SetMIL-MT", `set_mil_mt_no_sab` → "SetMIL-MT (no SAB)",  
`longitudinal_mk_mt_no_alibi` → "LongMK-MT (no ALiBi)", `longitudinal_mk_no_alibi` → "LongMK (no ALiBi) ★"

---

### 4 — KM curves (model-stratified survival)

| # | File | What | Status |
|---|------|------|--------|
| 4a | `km_curves/km_acr_cls.png` | ACR cls KM: top vs bottom risk tertile | ✅ |
| 4b | `km_curves/km_acr_surv.png` | ACR surv KM | ✅ |
| 4c | `km_curves/km_clad_surv.png` | CLAD surv KM | ✅ |
| 4d | `km_curves/km_death_surv.png` | Death surv KM | ✅ |
| 4e | `km_curves/km_all_tasks.png` | 4-panel combined | ✅ |

---

#### How made — KM curves

Script: `analysis/plot_km_from_model.py` via `analysis/submit_km_from_model.sh` (cpu_p, 350G, 1h).  
350G required because SetMIL `results_raw.npy` is ~40GB.

**Step 1 — Load predicted logits**  
Same `results_raw.npy` source as the UMAP script (best model per task). For each record, `logits[rep_key]` is extracted as the model's raw output before any sigmoid or softmax. Higher logit = higher predicted risk.

**Step 2 — Join outcomes**  
Patient-level event indicator and TTE joined from the splits CSV by `patient_id` (same enrichment logic as the UMAP script). For longitudinal models, the npy doesn't store outcomes — they come from the CSV lookup.

**Step 3 — Risk stratification**  
Patients ranked by predicted logit → split into tertiles (33rd and 67th percentile cutoffs).  
- Top tertile (logit ≥ 67th pct) = high-risk group  
- Bottom tertile (logit ≤ 33rd pct) = low-risk group  
Middle tertile excluded to maximise separation between the curves.

**Step 4 — KM curves**  
Kaplan-Meier step function computed manually (no external library needed for basic KM).  
Log-rank test p-value computed via `lifelines.statistics.logrank_test`; falls back to scipy chi-squared if lifelines is unavailable.  
Plot: high-risk in red, low-risk in blue, p-value annotated on the figure.  
TTE converted from days to years for display.

---

### 5 — Time weighting heatmaps (longitudinal model)

| # | File | Task | Note | Status |
|---|------|------|------|--------|
| 5a | `interpretability/acr_surv/L_global_weight_heatmap.png` | ACR surv | Single split (split 0) | ✅ |
| 5b | `interpretability/clad_surv/L_global_weight_heatmap.png` | CLAD | Single split (split 0) | ✅ |
| 5c | `interpretability/death_surv/L_global_weight_heatmap.png` | Death | Single split (split 0) | ✅ |
| 5d | `interpretability/acr_surv/L_global_weight_heatmap_avg.png` | ACR surv | **5-split mean ± std** | ✅ |
| 5e | `interpretability/clad_surv/L_global_weight_heatmap_avg.png` | CLAD | **5-split mean ± std** | ✅ |
| 5f | `interpretability/death_surv/L_global_weight_heatmap_avg.png` | Death | **5-split mean ± std** | ✅ |
| 5g | `interpretability/agg/L_global_weight_heatmap_avg_all.png` | All 3 tasks | **5-split combined** | ✅ |

---

#### How made — time weighting heatmaps

**What `biopsy_weight_net` is**

`longitudinal_mk_no_alibi` processes each patient's ordered biopsy sequence. Instead of the ALiBi fixed-distance bias, it uses a **fully learned 2-layer MLP** (`biopsy_weight_net`) per task: `Linear(2,16) → ReLU → Linear(16,1) → Sigmoid`. The input is `[current_biopsy_day, previous_biopsy_day]` (both in days post-transplant); the output is a scalar gate `w ∈ (0,1)` that multiplies the PMA seed vectors from the previous biopsy before ABMIL aggregation. This lets the model discover task-specific temporal weighting without imposing recency bias.

The MLP is a **model-level parameter**: all patients share the same `biopsy_weight_net` within one trained model. Its 2D output surface `w(curr, prev)` is the "time weighting heatmap".

**Heatmap layout**
- X-axis = Previous biopsy date (days post-transplant, 0–2000)
- Y-axis = Current biopsy date (days post-transplant, 0–2000)
- Colour = `w(curr, prev) ∈ (0,1)`; red = high weight (this historical biopsy strongly informs current prediction), blue = low weight (suppressed)
- Region above the diagonal (prev > curr = impossible) is masked to NaN (white)
- Dashed diagonal = self-attention at zero lag

**Biological read-out**
- ACR surv: near-diagonal structure → recent biopsies more informative for rejection prediction
- Death surv: broader off-diagonal weight → full history integrated for long-term mortality risk
- CLAD surv: intermediate pattern — early-to-late transitions weighted

**Single-split version** (5a–5c)

Script: `interpretability/interpret_longitudinal_mk.py`, GPU job, split 0 fold 0 only. Evaluates `biopsy_weight_net[task]` on 100×100 grid, renders directly.

**5-split averaged version** (5d–5g)

Script: `analysis/plot_L_global_avg.py` via `analysis/submit_L_global_avg.sh` (CPU-only, ~2 min).  
No GPU needed: extracts only the `biopsy_weight_net` weights from each checkpoint (4 parameters tensors totalling <1KB), reconstructs the MLP on CPU, evaluates on the same 100×100 grid for each of the 5 splits, then computes:
- `mean_W[i,j] = mean over splits 0–4 of w(curr_i, prev_j)`
- `std_W[i,j] = std over splits 0–4`

Output per task: left panel = mean heatmap, right panel = std heatmap (orange; shows where weight varies across training runs).

Low std = the temporal pattern is robust across different train/test splits; high std = sensitivity to which patients were in training.

Checkpoint path: `results/mm_abmil_v8/phase2/split{s}_fold0/longitudinal_mk_no_alibi_{task}/model_longitudinal_mk_no_alibi_final.pt`  
State-dict key prefix: `biopsy_weight_net.{net_key}.` where `net_key` = `"acr_surv"`, `"clad"`, or `"death"` for the three tasks.

---

### 6 — Unimodal ablation

| # | File | What | Status |
|---|------|------|--------|
| 6a | `interpretability/unimodal_ablation/unimodal_ablation_barplot.png` | Modality-in-isolation BACC/C-index, all models | ✅ |
| 6b | `interpretability/unimodal_ablation/unimodal_ablation_heatmap.png` | Heatmap: model×task rows, modality cols | ✅ |
| 6c | `interpretability/unimodal_ablation/unimodal_ablation_summary.csv` | Mean ± std across 5 splits | ✅ |

---

#### How made — unimodal ablation figures

Script: `interpretability/unimodal_ablation_summary.py`

**What "unimodal ablation" means**  
During training and evaluation, each model is also evaluated with only one modality active at a time. The other modalities are **zeroed at the feature level** (embedding vectors set to zero before PMA/fusion), not simply dropped. This means the model architecture is identical — the same trained weights — but it can only use signal from one modality. The resulting metric shows how much each modality alone contributes when the model has been trained on all four.

**Data source**  
Every training run writes `results/mm_abmil_v8/phase2/split{s}_fold0/{variant}_{task}/metrics_{variant}_{task}_final.json`.  
This JSON has a `unimodal_ablation` block structured as:  
```json
"unimodal_ablation": {
  "HE": {"bacc": 0.58, "c_index": 0.61},
  "BAL": {"bacc": 0.55, "c_index": 0.59},
  ...
}
```
The script reads this block for all variants × tasks × splits.

**Aggregation**  
Rows collected: `(split, variant, task, modality, metric_value)`.  
Grouped by `variant × task × modality`, aggregated across 5 splits → mean ± std.  
Saved to `unimodal_ablation_summary.csv` (one row per group) and `unimodal_ablation_raw.csv` (per-split).

**Bar plot (6a)**  
Grouped bar chart per task. X-axis = metric value. Each group = one model variant. Bars within a group = modalities (HE red, BAL blue, CT green, Clinical purple). Allows reading: "for SetMIL-MT on Death survival, which single modality performs best?"

**Heatmap (6b)**  
Rows = `(model variant × task)`, columns = modality. Colour = mean metric. Allows scanning which model × task combinations are most dependent on a single modality.

Coverage: early, late, middle, set_mil_mt, set_mil_mt_no_sab, set_mil_no_sab. Longitudinal models not included (would require re-running inference with per-modality zeroing for each split).

---

### 7 — Patient trajectories (archetypal patients)

| # | File | Patient | Archetype |
|---|------|---------|-----------|
| 7a | `trajectories/panel_A_LT100.png` | LT100 | Stable survivor — low risk throughout |
| 7b | `trajectories/panel_B_LT119.png` | LT119 | Early-onset non-survivor — high risk from biopsy 1 |
| 7c | `trajectories/panel_C_LT062.png` | LT062 | Late-escalating — risk rises after year 1 with CLAD onset |
| 7d | `trajectories/panel_D_LT227.png` | LT227 | Treatment-responsive — logit drops after IS adjustment |
| 7e | `trajectories/Fig7_patient_trajectories.png` | Combined | 4-panel composite |

---

#### How made — patient trajectories

Script: `interpretability/interpret_longitudinal_mk.py` — GPU, split 0 fold 0, `--task death_surv`.  
- 4 patients hand-selected from the Death survival test set to represent archetypal clinical courses (stable survivor, early-onset non-survivor, late-escalating, treatment-responsive).  
- For each patient the script runs a full forward pass, extracting the log-hazard score (model logit) at every biopsy visit.  
- X-axis = days post-transplant. Y-axis = log-hazard (raw logit, not transformed). Clinical event annotations overlaid: ACR grade (triangle markers), CLAD onset (vertical line), death or censoring (end marker).  
- Red dashed line = logit > 1.5, the proposed threshold for surveillance intensification (chosen empirically from the training cohort risk distribution).  
- These are **in-sample** for split 0 fold 0 — they are illustrative case studies, not held-out validation. The selection criterion was clinical diversity, not model performance.

---

## Directory Map

```
figures/
├── FIGURES.md
├── benchmark/                    ← Section 3: all models vs all tasks
├── km_curves/                    ← Section 4: KM curves by model risk
├── interpretability/
│   ├── acr_cls/                  ← ACR cls · set_mil_mt_no_sab · split 2 fold 0
│   ├── acr_surv/                 ← ACR surv · longitudinal_mk_no_alibi · split 0 fold 0 · N=226
│   ├── clad_surv/                ← CLAD · set_mil_mt · split 0 fold 0
│   ├── death_surv/               ← Death · longitudinal_mk_no_alibi · split 0 fold 0 · N=226
│   ├── agg/                      ← Cross-task aggregates (Lpop_K_agg, unified UMAPs, multimod)
│   ├── cluster_agg/              ← Named biological cluster attribution, all 4 tasks
│   └── unimodal_ablation/        ← Section 6: single modality contribution
└── trajectories/                 ← Section 7: per-patient risk over time
```

---

## Missing / Potential improvements

| # | What | Priority | Notes |
|---|------|----------|-------|
| 2t | A/B/D instance+seed panels for ACR surv + Death (longi model) | Medium | Requires GPU interpret job for longitudinal model |
| 3h | Numeric benchmark table (mean±std grid, all models × tasks) | Low | `plot_benchmark_table.py` exists but needs label sync with v2 |
| 6d | Unimodal ablation for LongMK models | Medium | Needs re-inference with per-modality zeroing per split |
| — | KM curves: add confidence bands (Greenwood formula) | Low | Currently plain step functions |
| — | UMAP: add convex hull or 95%-ellipse per risk tertile on panel ① | Low | Visually clarifies group separation |
| — | Cluster affinity: add permutation p-values per cluster | Low | Bootstrap n=1000, FDR-correct across clusters |
| — | Benchmark: add Wilcoxon signed-rank vs linear baseline (annotate bars) | Low | Paired test across 5 splits, FDR-correct |

---

## Reproducibility

| Script | Produces | Compute |
|--------|---------|---------|
| `interpretability/interpret_set_mil_mt.py` | Panels A–K (acr_cls, clad) | GPU ~30 min |
| `interpretability/interpret_longitudinal_mk.py` | Panels L, Lpop, trajectories, results_raw.npy | GPU ~60 min/split |
| `interpretability/gen_cluster_aff_agg.py` | cluster_agg/ panels | CPU |
| `interpretability/unimodal_ablation_summary.py` | unimodal_ablation/ plots + CSVs | CPU |
| `analysis/rebuild_benchmark_csvs.py` | results/predictions/comparison_*.csv | CPU |
| `analysis/plot_benchmark_multimodal.py` | benchmark/ bar plots | CPU 5 min |
| `analysis/plot_unified_rep_umap.py` | unified_rep_umap per task + 4-panel agg | CPU 30 min |
| `analysis/plot_km_from_model.py` | km_curves/ | CPU 5 min (needs 350G for SetMIL npy) |
| `analysis/plot_multimod_seed_attribution.py` | multimod_seed_attribution per task + combined | CPU 5 min |
