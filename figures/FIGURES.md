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

**How made:** `analysis/plot_unified_rep_umap.py` via `analysis/submit_unified_rep_umap.sh` (cpu_p, 350G).  
- Loads `results_raw.npy` from the best model per task (SetMIL: splits CSV join for labels; Longi: patient lookup from splits CSV because biopsy records don't carry patient-level outcomes).  
- UMAP with cosine metric, n_neighbors=30, min_dist=0.3, fit on 256-dim ABMIL output representations.  
- 2×4 layout per task: ① Risk score (plasma), ② Event indicator (tab10), ③ TTE continuous (plasma_r), ④ Modality combo (tab20), ⑤ Hexbin density, ⑥ Hexbin risk score, ⑦ Hexbin TTE, ⑧ CV split.

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

**How made (2a–2g, 2e–2g):** `interpretability/interpret_set_mil_mt.py` — GPU job; loads SetMIL checkpoint, runs forward pass on test set, extracts PMA seed vectors and ABMIL attention weights, runs UMAP, plots all panels.  

**How made (2h–2k — Lpop_K_agg):** `interpretability/gen_cluster_aff_agg.py` reads `seed_attribution_data_*.json` from all 5 splits, averages Δα per seed, plots aggregated bar chart.  

**How made (2l–2o — multimod_seed_attribution):** `analysis/plot_multimod_seed_attribution.py` — CPU job (sbatch cpu_p, 8G).  
- Reads `seed_attribution_data_{task}.json` from `longitudinal_mk_no_alibi_split{0-4}_fold0_{task}/` for all 5 splits.  
- `seed_labels` field has modality prefix: `HE·s00 ... HE·s15, BAL·s00 ... BAL·s15, CT·s00 ... CT·s15` (48 total).  
- `alpha_diff` = Δα per seed (mean_hi_risk − mean_lo_risk), precomputed by the interpretability pipeline.  
- Groups seeds by modality prefix, averages Δα and std across splits, plots horizontal bars sorted by |Δα|. Red = high-risk associated, blue = low-risk associated.  

**How made (2p–2s — cluster affinity):** `interpretability/gen_cluster_aff_agg.py` — maps PMA seed attention to named biological clusters (published BAL cell-type annotations + HE/CT unsupervised labels), aggregates over 5 splits.

---

### 3 — Benchmark (all models, all tasks)

| # | File | What | Status |
|---|------|------|--------|
| 3a | `benchmark/benchmark_multimodal_acr_cls.png` | All models vs ACR cls | ✅ |
| 3b | `benchmark/benchmark_multimodal_acr_surv.png` | All models vs ACR surv | ✅ |
| 3c | `benchmark/benchmark_multimodal_clad_surv.png` | All models vs CLAD | ✅ |
| 3d | `benchmark/benchmark_multimodal_death_surv.png` | All models vs Death | ✅ |
| 3e | `benchmark/benchmark_multimodal_all.png` | 4-panel combined | ✅ |

**How made:** `analysis/plot_benchmark_multimodal.py` (CPU, <5 min).  
- Reads per-split CSVs from `results/predictions/comparison_{task}.csv` (rebuilt by `analysis/rebuild_benchmark_csvs.py` which parses raw JSON metric files `metrics_split{s}_fold0_{variant}_{task}.json`).  
- Longi model metrics needed nested JSON lookup: `d["test"]["acr_surv"]["c_index"]` not flat `d["test"]["c_index"]`.  
- Horizontal bar chart: mean ± std, hollow per-split dots, amber border on best model.  
- Models grouped: P1 unimodal, P1 ensemble, P2 non-temporal fusions (early/middle/late), P2 SetMIL variants, P2 longitudinal_mk variants.

---

### 4 — KM curves (model-stratified survival)

| # | File | What | Status |
|---|------|------|--------|
| 4a | `km_curves/km_acr_cls.png` | ACR cls KM: top vs bottom risk tertile | ✅ |
| 4b | `km_curves/km_acr_surv.png` | ACR surv KM | ✅ |
| 4c | `km_curves/km_clad_surv.png` | CLAD surv KM | ✅ |
| 4d | `km_curves/km_death_surv.png` | Death surv KM | ✅ |
| 4e | `km_curves/km_all_tasks.png` | 4-panel combined | ✅ |

**How made:** `analysis/plot_km_from_model.py` via `analysis/submit_km_from_model.sh` (cpu_p, 350G, 1h).  
- Loads model logit scores from `results_raw.npy` (best model per task; SetMIL npy is 40GB, hence 350G mem).  
- Joins patient-level outcomes (event indicator, TTE) from splits CSV `/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv` by `patient_id`.  
- Stratifies into top-vs-bottom risk tertile by predicted logit.  
- KM curves with 95% CI; log-rank p-value (lifelines; scipy chi2 fallback).  
- Output: per-task PNG/PDF + 4-panel combined figure.

---

### 5 — Time weighting heatmaps (longitudinal model)

| # | File | Task | Status |
|---|------|------|--------|
| 5a | `interpretability/acr_surv/L_global_weight_heatmap.png` | ACR surv | ✅ |
| 5b | `interpretability/clad_surv/L_global_weight_heatmap.png` | CLAD | ✅ |
| 5c | `interpretability/death_surv/L_global_weight_heatmap.png` | Death | ✅ |

**How made:** `interpretability/interpret_longitudinal_mk.py` — GPU job.  
- Reads the learned `L_global` temporal weight matrix from `longitudinal_mk_no_alibi` checkpoint (split 0, fold 0).  
- `L_global` is a T×T matrix where `L[i,j]` = weight applied to biopsy j when predicting at time i.  
- Plotted as heatmap: rows = current biopsy date (post-Tx), cols = previous biopsy; colour = weight magnitude.  
- Diagonal = self-weight (recency=0); off-diagonal = how much history is integrated.

---

### 6 — Unimodal ablation

| # | File | What | Status |
|---|------|------|--------|
| 6a | `interpretability/unimodal_ablation/unimodal_ablation_barplot.png` | Modality-in-isolation BACC/C-index, all models | ✅ |
| 6b | `interpretability/unimodal_ablation/unimodal_ablation_heatmap.png` | Heatmap: model×task rows, modality cols | ✅ |
| 6c | `interpretability/unimodal_ablation/unimodal_ablation_summary.csv` | Mean ± std across 5 splits | ✅ |

**How made:** `interpretability/unimodal_ablation_summary.py` — CPU.  
- Reads ablation results from `results_raw.npy` files where only one modality was active (others zeroed at the feature level).  
- Groups by `variant × task × modality`, aggregates across 5 splits.  
- Coverage: early, late, middle, set_mil_mt, set_mil_mt_no_sab, set_mil_no_sab. Longitudinal models not yet included.

---

### 7 — Patient trajectories (archetypal patients)

| # | File | Patient | Archetype |
|---|------|---------|-----------|
| 7a | `trajectories/panel_A_LT100.png` | LT100 | Stable survivor — low risk throughout |
| 7b | `trajectories/panel_B_LT119.png` | LT119 | Early-onset non-survivor — high risk from biopsy 1 |
| 7c | `trajectories/panel_C_LT062.png` | LT062 | Late-escalating — risk rises after year 1 with CLAD onset |
| 7d | `trajectories/panel_D_LT227.png` | LT227 | Treatment-responsive — logit drops after IS adjustment |
| 7e | `trajectories/Fig7_patient_trajectories.png` | Combined | 4-panel composite |

**How made:** `interpretability/interpret_longitudinal_mk.py` — GPU.  
- Runs `longitudinal_mk_no_alibi` on 4 hand-selected patients from the Death survival test set.  
- Plots log-hazard at each biopsy visit (x = days post-transplant) with clinical event annotations (ACR grade, CLAD, death/censoring). Red dashed line = logit > 1.5 (proposed surveillance threshold).

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

## Missing

| # | What | Priority |
|---|------|----------|
| 2t | A/B/D instance+seed panels for ACR surv + Death (longi model) | Medium |
| 3h | Full benchmark table with per-split rows (s0–s4 + mean±std) | Low |
| 6d | Unimodal ablation updated with longitudinal_mk models | Medium |

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
