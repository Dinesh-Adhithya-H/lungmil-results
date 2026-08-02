# Figures — Longitudinal Multimodal MIL, Lung Transplant

Evaluation: 5-split × 4-fold nested cross-validation. Best model per task = fold 0 test set.  
All PNGs have matching PDFs (300 dpi, vector).

---

## Best Model per Task

| Task | Metric | Best model | Score (mean ± std) | Linear baseline |
|------|--------|------------|-------------------|-----------------|
| **Death survival** | C-index ↑ | `longitudinal_mk_no_alibi` | **0.771 ± 0.056** | 0.580 |
| **ACR survival** | C-index ↑ | `longitudinal_mk_no_alibi` | **0.679 ± 0.064** | 0.587 |
| **CLAD survival** | C-index ↑ | `set_mil_mt` | **0.563 ± 0.080** | 0.501 |
| **ACR classification** | BACC ↑ | `set_mil_mt_no_sab` | **0.623 ± 0.034** | 0.588 |

---

## Required Figure Checklist

### 1 — Patient representation UMAPs (test set, best model per task)

| # | Figure | Task | Status |
|---|--------|------|--------|
| 1a | `interpretability/acr_cls/G_final_rep_hexbin_acr_cls.png` | ACR cls | ✅ exists (7 panels, hexbin style) |
| 1b | `interpretability/acr_surv/Lpop_rep_umap_acr_surv.png` | ACR surv | ✅ exists (4 panels, scatter) |
| 1c | `interpretability/clad/G_final_rep_hexbin_clad_surv.png` | CLAD | ✅ exists (7 panels, hexbin style) |
| 1d | `interpretability/death/Lpop_rep_umap_death.png` | Death | ✅ exists (4 panels, scatter) |
| 1e | `interpretability/{task}/unified_rep_umap_{task}.png` | All 4 | ⏳ generating (job 38976209, GPU+CPU chain) |
| 1f | `interpretability/agg/unified_rep_umap_all_tasks.png` | All 4 (2×2) | ⏳ generating |

**Unified format (1e/1f):** 6 identical panels per task — ① Risk score, ② True label/event, ③ TTE (size∝urgency), ④ Modality combo, ⑤ KM top-vs-bottom tertile, ⑥ CV split annotation.

---

### 2 — Instance → seed → disease (attention chain)

| # | Figure | Task | Panel type | Status |
|---|--------|------|-----------|--------|
| 2a | `interpretability/acr_cls/A_instance_reps_cosine.png` | ACR cls | Instance space (UMAP of all patches, 4 modalities) | ✅ |
| 2b | `interpretability/acr_cls/B_seeds.png` | ACR cls | Seed prototypes in instance space (★ per seed) | ✅ |
| 2c | `interpretability/acr_cls/D_abmil_seed_importance.png` | ACR cls | ABMIL α-weights per seed → which seed predicts disease | ✅ |
| 2d | `interpretability/acr_cls/K_seed_attribution_acr_cls.png` | ACR cls | Δattention (high vs low risk) per seed | ✅ |
| 2e | `interpretability/clad/A_instance_reps_cosine.png` | CLAD | Instance space | ✅ |
| 2f | `interpretability/clad/B_seeds.png` | CLAD | Seed prototypes | ✅ |
| 2g | `interpretability/clad/D_abmil_seed_importance.png` | CLAD | ABMIL α-weights per seed | ✅ |
| 2h | `interpretability/clad/K_seed_attribution_clad.png` | CLAD | Δattention per seed | ✅ |
| 2i | `interpretability/acr_surv/Lpop_K_seed_attribution_acr_surv.png` | ACR surv | Δ seed attribution aggregated over test patients | ✅ |
| 2j | `interpretability/death/Lpop_K_seed_attribution_death.png` | Death | Δ seed attribution | ✅ |
| 2k | `interpretability/agg/Lpop_K_agg_acr_cls.png` | ACR cls | Δ seed attribution, 5-split aggregate | ✅ |
| 2l | `interpretability/agg/Lpop_K_agg_acr_surv.png` | ACR surv | 5-split aggregate | ✅ |
| 2m | `interpretability/agg/longitudinal_mk_no_alibi_Lpop_K_agg_death_surv.png` | Death | 5-split aggregate | ✅ |
| 2n | `interpretability/agg/longitudinal_mk_no_alibi_Lpop_K_agg_clad_surv.png` | CLAD (longi) | 5-split aggregate | ✅ |
| 2o | `interpretability/cluster_agg/death_cluster_aff_agg.png` | Death | Biological cluster affinity (named clusters) | ✅ |
| 2p | `interpretability/cluster_agg/acr_surv_cluster_aff_agg.png` | ACR surv | Biological cluster affinity | ✅ |
| 2q | `interpretability/cluster_agg/acr_cls_cluster_aff_agg.png` | ACR cls | Biological cluster affinity | ✅ |
| 2r | `interpretability/cluster_agg/clad_cluster_aff_agg.png` | CLAD | Biological cluster affinity | ✅ |
| **2s** | **A+B+D panels for acr_surv + death** | ACR surv, Death | Instance space + seeds (longitudinal model) | ❌ **MISSING** |

> **Note 2s:** The longitudinal model also uses PMA seeds (K=16 per modality). Instance-space and seed-position UMAPs (panels A, B, D) for `longitudinal_mk_no_alibi` on ACR surv and Death do not yet exist. These would require adding an A/B/D extraction pass to `interpret_longitudinal_mk.py` and submitting a GPU job.

---

### 3 — Benchmark tables (full multimodal, all models, all tasks)

| # | Figure | What it shows | Status |
|---|--------|--------------|--------|
| 3a | `benchmark/benchmark_multimodal_acr_cls.png` | All models vs ACR cls — bar+strip, per-split dots | ✅ |
| 3b | `benchmark/benchmark_multimodal_acr_surv.png` | All models vs ACR surv | ✅ |
| 3c | `benchmark/benchmark_multimodal_clad_surv.png` | All models vs CLAD surv | ✅ |
| 3d | `benchmark/benchmark_multimodal_death_surv.png` | All models vs Death surv | ✅ |
| 3e | `benchmark/benchmark_multimodal_all.png` | 4-panel combined (all tasks side by side) | ✅ |
| 3f | `results/analysis_v8_full/benchmark_v8_full.png` | Extended benchmark including all variants | ✅ (not in figures/) |
| 3g | `analysis/nature_paper/benchmark_table_full.png` | Paper-quality benchmark table | ✅ (not in figures/) |
| **3h** | **benchmark/benchmark_full_table.png** | Full table with all splits per row (s0–s4 + mean±std) | ❌ **MISSING** |

> **Raw data:** `results/predictions/comparison_{acr_cls,acr_surv,clad,death}.csv` — per-split values for all models, all tasks.

---

### 4 — Unimodal ablation (single modality contribution, all models)

| # | Figure | What it shows | Status |
|---|--------|--------------|--------|
| 4a | `interpretability/unimodal_ablation/unimodal_ablation_barplot.png` | Per-task bar: each modality in isolation across model families | ✅ |
| 4b | `interpretability/unimodal_ablation/unimodal_ablation_heatmap.png` | Matrix: rows=model×task, cols=modality | ✅ |
| 4c | `interpretability/unimodal_ablation/unimodal_ablation_summary.csv` | Mean ± std across 5 splits per model × task × modality | ✅ |
| 4d | `interpretability/unimodal_ablation/unimodal_ablation_raw.csv` | Per-split raw values | ✅ |
| **4e** | **Updated barplot/heatmap including longitudinal_mk models** | Longitudinal models currently absent from ablation | ⏳ generating (jobs 38976119 → 38976125) |

> **Coverage:** Current ablation (4a/4b) includes: `early`, `late`, `middle`, `set_mil_mt`, `set_mil_mt_no_sab`, `set_mil_no_sab`. Longitudinal models (`longitudinal_mk_no_alibi`, etc.) will be added once GPU job 38976119 finishes (~2 h).

---

## Directory Map

```
figures/
├── FIGURES.md                              ← this file (authoritative index + checklist)
├── benchmark/                              ← Benchmark comparison bar plots
│   ├── benchmark_multimodal_acr_cls.png/pdf
│   ├── benchmark_multimodal_acr_surv.png/pdf
│   ├── benchmark_multimodal_clad_surv.png/pdf
│   ├── benchmark_multimodal_death_surv.png/pdf
│   └── benchmark_multimodal_all.png/pdf    ← 4-panel combined
├── interpretability/
│   ├── acr_cls/                            ← ACR cls · set_mil_mt_no_sab · split 2 fold 0
│   ├── acr_surv/                           ← ACR surv · longitudinal_mk_no_alibi · split 0 fold 0
│   ├── clad/                               ← CLAD · set_mil_mt · split 2 fold 0
│   ├── death/                              ← Death · longitudinal_mk_no_alibi · split 0 fold 0
│   ├── agg/                                ← Seed attribution, 5-split aggregate
│   ├── cluster_agg/                        ← Named biological cluster attribution
│   └── unimodal_ablation/                  ← Single-modality contribution
└── trajectories/                           ← Per-patient risk over time (4 archetypes)
```

---

## Panel-by-Panel Descriptions

### A — Instance representation space (`A_instance_reps_cosine.png`)
**Tasks:** ACR cls, CLAD  
**What:** UMAP of all patch/cell instances across HE, BAL, CT, Clinical modalities.  
Each point = one instance (patch or cell), coloured by modality. Seed prototypes (★) overlaid at their attention-weighted centroid in instance space.  
**Why:** Shows the biological diversity captured — do seeds separate tissue phenotypes? Are HE inflammatory patches co-located with BAL macrophages?  
**Generated by:** `interpret_set_mil_mt.py`, panel A

### B — Seed prototypes in instance space (`B_seeds.png`)
**Tasks:** ACR cls, CLAD  
**What:** Same UMAP as A, but now each of the K=16 seed vectors per modality is shown as a star (★), sized by its mean ABMIL attention weight α across test patients. Seeds with large α are shown larger.  
**Why:** Reveals which regions of biological space the model focused its K prototype slots on. Seeds co-located with inflammatory patches in HE confirm the model has found clinically relevant tissue patterns.  
**Generated by:** `interpret_set_mil_mt.py`, panel B

### D — ABMIL seed importance (`D_abmil_seed_importance.png`)
**Tasks:** ACR cls, CLAD  
**What:** Horizontal bar chart: mean ABMIL α attention weight per seed (K=16 per modality = 64 bars), ranked. Error bars = std across test patients. Top seeds highlighted.  
**Why:** Direct read-out of which seed the task prediction depends on. If seed s₃ (BAL, TRAM cluster) has the highest α for Death survival, TRAM tissue is mechanistically important.  
**Generated by:** `interpret_set_mil_mt.py`, panel D

### E — Task modality gate (`E_task_modal_gate.png`)
**Tasks:** ACR cls  
**What:** Learned scalar gate per modality: how much each data stream (HE, BAL, CT, Clinical) contributed to the final prediction, averaged over test patients.  
**Why:** Model-level evidence of multimodal integration — a gate near 1 for all modalities means complementary signal rather than redundancy.

### F — Modality combination ablation (`F_modality_combo_ablation.png`)
**Tasks:** ACR cls  
**What:** BACC for all 2⁴=16 subsets of {HE, BAL, CT, Clinical}. Each bar = one combination.  
**Why:** Identifies which pair (or trio) of modalities is sufficient, and which combinations are synergistic beyond individual contributions.

### G — Patient representation space (`G_final_rep_hexbin_{task}.png`)
**Tasks:** ACR cls, CLAD  
**What:** UMAP of patient-level final representations (256-dim ABMIL output). 7-panel layout:  
① ACR label (red=+), ② Normalised risk score (red=high), ③ TTE+event markers, ④ Modality combo, ⑤ KM top vs bottom tertile, ⑥ Anchor date, ⑦ CV split.  
**Why:** Validates that the learned patient embedding is biologically meaningful — do high-risk patients cluster separately? Is the representation confounded by missing modalities?  
**Note:** Will be replaced/supplemented by unified_rep_umap (see §1e) when job 38976209 completes.

### I — Seed risk stratification (`I_seed_risk_stratification.png`)
**Tasks:** ACR cls  
**What:** Seed UMAP (from B) with test patients overlaid: red = high-risk biopsies (above median prediction), blue = low-risk. Shows where in seed space risk concentrates.

### K — Seed attribution per task (`K_seed_attribution_{task}.png`)
**Tasks:** ACR cls, CLAD  
**What:** Δα = mean α(high-risk) − mean α(low-risk) per seed. Positive = seed drives risk; negative = seed is protective. Seeds sorted by |Δα|.  
**Why:** The core interpretability read-out: connects model attention to biology. A seed with Δα > 0.02 and known cluster identity (TRAM, CD8-T-cell) is a testable biological prediction.  
**Generated by:** `interpret_set_mil_mt.py`, panel K

### L_global — Biopsy weighting heatmap (`L_global_weight_heatmap.png`)
**Tasks:** ACR surv, Death  
**What:** Heatmap: rows = current biopsy date (post-Tx), cols = previous biopsy date. Colour = learned temporal weight w ∈ (0,1) — how much the model weights this transition. Diagonal = current biopsy (recency = 0).  
**Why:** Reveals which temporal windows matter: early dominance (ACR surv) vs uniform full-history (Death).  
**Generated by:** `interpret_longitudinal_mk.py`

### Lpop_alpha_recency — Recency bias (`Lpop_alpha_recency.png`)
**Tasks:** Death  
**What:** Mean biopsy weight vs days-before-anchor-date across all test patients. Flat = no recency bias.  
**Why:** Confirms the longitudinal model's full-history integration for Death rather than overweighting recent visits.

### Lpop_K — Seed attribution aggregate (`Lpop_K_seed_attribution_{task}.png`)
**Tasks:** ACR surv, Death  
**What:** Same as K, but computed from all test patients of the longitudinal model (not split by risk group — instead: seeds sorted by Δα across full test cohort).  
**Generated by:** `interpret_longitudinal_mk.py`

### Lpop_rep_umap — Patient UMAP longitudinal (`Lpop_rep_umap_{task}.png`)
**Tasks:** ACR surv, Death  
**What:** UMAP of patient-level longitudinal representations (rep_full from ABMIL on seeds). 2×2 panels: logit score, #biopsies, anchor day, risk group (median split).  
**Note:** Will be superseded by unified_rep_umap (job 38976209).  
**Generated by:** `interpret_longitudinal_mk.py`

### unified_rep_umap — Unified patient UMAP (`unified_rep_umap_{task}.png`) ⏳
**Tasks:** ALL 4  
**What:** 6 identical panels for every task — ① Risk score, ② True label/event, ③ TTE (size∝urgency, red=event), ④ Modality combination, ⑤ KM: top vs bottom risk tertile, ⑥ CV split annotation.  
**Status:** ⏳ Generating — GPU job 38976208 produces `results_raw.npy` for longitudinal models (all 5 splits × 2 tasks); CPU job 38976209 reads all npy caches and renders unified figures.  
**Generated by:** `analysis/plot_unified_rep_umap.py`

### Cluster affinity aggregated (`cluster_agg/{task}_cluster_aff_agg.png`)
**Tasks:** ALL 4  
**What:** Δ affinity per named biological cluster = mean cluster affinity in high-risk biopsies − mean in low-risk. Clusters = published BAL cell-type annotations + HE/CT unsupervised labels. Aggregated across 5 CV splits.  
**Why:** Translates model attention into interpretable biology: which cell type / tissue pattern is the model actually attending to?  
**Generated by:** `interpretability/gen_cluster_aff_agg.py`

---

## Benchmark Figures

### `benchmark/benchmark_multimodal_{task}.png`
**What:** Horizontal bar chart (one per task). X-axis = metric (BACC or C-index). Models on Y-axis:  
— P1 unimodal (HE, BAL, CT, Clinical separately) — blue  
— P1 weighted ensemble — dark plum  
— P2 non-temporal fusion (early/late/middle) — teal  
— P2 longitudinal_mk — crimson ★  
Each bar has error bars (±std), per-split dots (hollow white markers).  
Amber border highlights the best model.  
**Data source:** `results/predictions/comparison_{task}.csv`  
**Generated by:** `analysis/plot_benchmark_multimodal.py`

### `benchmark/benchmark_multimodal_all.png`
**What:** 4-panel combined figure (all tasks side by side). Same format. Shared legend.

---

## Unimodal Ablation Figures

### `interpretability/unimodal_ablation/unimodal_ablation_barplot.png`
**What:** Per-task grouped bar chart: each modality (HE/BAL/CT/Clinical) run in isolation within the full model architecture, across all model families. Shows how much each modality contributes when other modalities are zeroed.  
**Coverage:** early, late, middle, set_mil_mt, set_mil_mt_no_sab, set_mil_no_sab. **Longitudinal models: ⏳ pending (job 38976119 → 38976125).**  
**Generated by:** `interpretability/unimodal_ablation_summary.py`

### `interpretability/unimodal_ablation/unimodal_ablation_heatmap.png`
**What:** Heatmap — rows = (model × task), columns = modality subset. Colour = metric value. Allows quick scan of which model×task combinations benefit most from each modality.

### `interpretability/unimodal_ablation/unimodal_ablation_summary.csv`
Columns: `variant, task, modality, metric, mean, std, count, mean_std`  
5 splits aggregated. Source of truth for all ablation plots.

---

## Patient Trajectory Figures

### `trajectories/Fig7_patient_trajectories.png`
**Model:** `longitudinal_mk_no_alibi` — Death survival task.  
**What:** 4-panel composite. Each panel shows: predicted log-hazard at every biopsy visit (x = days post-transplant), clinical events marked (ACR grade, CLAD diagnosis, death/censoring). Red dashed line = clinical flagging threshold (logit > 1.5).

| Panel | Patient | Archetype |
|-------|---------|-----------|
| A (`panel_A_LT100.png`) | LT100 | Stable survivor — consistently low logit, no events |
| B (`panel_B_LT119.png`) | LT119 | Early-onset non-survivor — high risk from biopsy 1, died year 2 |
| C (`panel_C_LT062.png`) | LT062 | Late-escalating — risk rises after year 1, coincides with CLAD onset |
| D (`panel_D_LT227.png`) | LT227 | Treatment-responsive — logit drops after IS adjustment |

**Clinical use:** Model issues an updated risk score at each routine biopsy. Threshold logit > 1.5 proposed for surveillance intensification.

---

## Missing / In-Progress Summary

| # | What | Priority | Status |
|---|------|----------|--------|
| 1e | Unified 6-panel UMAPs, all 4 tasks | High | ⏳ job 38976208 (GPU) → 38976209 (CPU) |
| 2s | A/B/D (instance+seed) panels for ACR surv + Death (longitudinal model) | Medium | ❌ needs new GPU job |
| 3h | Full benchmark table (s0–s4 + mean±std per row) | Medium | ❌ needs plot script |
| 4e | Unimodal ablation updated with longitudinal_mk models | High | ⏳ job 38976119 (GPU) → 38976125 (CPU) |

---

## Reproducibility

| Script | Produces | Compute |
|--------|---------|---------|
| `interpretability/interpret_set_mil_mt.py` | A, B, D, E, F, G, I, J, K panels (acr_cls, clad) | GPU ~30 min |
| `interpretability/interpret_longitudinal_mk.py` | L, Lpop panels + results_raw.npy (acr_surv, death) | GPU ~60 min per split |
| `interpretability/gen_cluster_aff_agg.py` | cluster_agg/ panels | CPU |
| `interpretability/unimodal_ablation_summary.py` | unimodal_ablation/ plots + CSVs | CPU (after ablation npy available) |
| `interpretability/regen_G_panel.py` | G hexbin panels from cached npy | CPU |
| `analysis/plot_benchmark_multimodal.py` | benchmark/ bar plots | CPU 5 min |
| `analysis/plot_unified_rep_umap.py` | unified_rep_umap per task + 4-panel agg | CPU 30 min (after GPU npy cache) |
| `scripts/compute_longitudinal_ablation.py` | Unimodal ablation for longitudinal models | GPU ~2 h |
