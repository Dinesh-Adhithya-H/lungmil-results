# Figures — Longitudinal Multimodal MIL, Lung Transplant

All plots organized by **task → best model → figure type**.  
Evaluation: 5-split × 4-fold nested cross-validation.  
Best model per task selected on inner-fold validation C-index / BACC.

---

## Best Model per Task

| Task | Metric | Best model | Score (mean ± std) | Linear baseline |
|------|--------|------------|-------------------|-----------------|
| **Death survival** | C-index ↑ | `longitudinal_mk_no_alibi` | **0.771 ± 0.056** | 0.580 |
| **ACR survival** | C-index ↑ | `longitudinal_mk_no_alibi` | **0.679 ± 0.064** | 0.587 |
| **CLAD survival** | C-index ↑ | `set_mil_mt` | **0.563 ± 0.080** | 0.501 |
| **ACR classification** | BACC ↑ | `set_mil_mt_no_sab` | **0.623 ± 0.034** | 0.588 |

---

## Directory Map

```
figures/
├── FIGURES.md                          ← this file
├── interpretability/
│   ├── acr_cls/                        ← ACR classification · set_mil_mt_no_sab
│   ├── acr_surv/                       ← ACR survival · longitudinal_mk_no_alibi
│   ├── death/                          ← Death survival · longitudinal_mk_no_alibi
│   ├── clad/                           ← CLAD survival · set_mil_mt
│   ├── agg/                            ← Seed attribution aggregated across 5 splits
│   ├── cluster_agg/                    ← Biological cluster → task attribution
│   └── unimodal_ablation/              ← Unimodal ablation (single modality vs all)
└── trajectories/                       ← Per-patient risk trajectories (death, 4 cases)
```

---

## Figure Legend

### Panel prefixes (SetMIL tasks: acr_cls, clad)

| Prefix | Description |
|--------|-------------|
| `A_instance_reps_cosine` | Cosine similarity matrix of all patch instance embeddings per modality (UMAP neighbourhood) |
| `B_seeds` | UMAP of K=16 PMA seed vectors per modality — shows what distinct patterns each seed captures |
| `D_abmil_seed_importance` | ABMIL attention weights on seeds: which seed the task model attends to most |
| `E_task_modal_gate` | Learned gate weight per modality (0–1 scalar) — contribution of each data stream |
| `F_modality_combo_ablation` | BACC/C-index for every subset of modalities (all 2^4 = 16 combos) |
| `G_final_rep_hexbin` | Biopsy-level UMAP (N=916) coloured by: ACR label, risk score, TTE, modality combo, anchor day, Risk×TTE, KM curve, #biopsies |
| `I_seed_risk_stratification` | High-risk vs low-risk biopsies coloured in seed UMAP space |
| `K_seed_attribution` | Δattention per seed (high-risk vs low-risk) — which seeds drive the prediction |

### Panel prefixes (Longitudinal tasks: acr_surv, death)

| Prefix | Description |
|--------|-------------|
| `L_global_weight_heatmap` | Per-task biopsy-weighting heatmap: rows = current biopsy day, cols = previous biopsy day, colour = learned weight ∈ (0,1) |
| `Lpop_alpha_recency` | Recency bias analysis: weight vs time-to-anchor-date |
| `Lpop_K_seed_attribution` | Seed-level attribution Δα aggregated over test patients |
| `Lpop_rep_umap` | Patient-level UMAP of longitudinal representations, coloured by outcome / risk score |

---

## `interpretability/acr_cls/` — ACR Classification
**Model:** `set_mil_mt_no_sab` · split 2 fold 0 · BACC 0.623

| File | What it shows |
|------|--------------|
| `A_instance_reps_cosine.png` | Cosine similarity of HE/BAL/CT/Clinical patch embeddings |
| `B_seeds.png` | K=16 seed UMAPs per modality — each star = one prototype |
| `D_abmil_seed_importance.png` | ABMIL attention on seeds: Clinical seed 0 is top discriminator |
| `E_task_modal_gate.png` | All four modalities gated ≥0.88 — equally valued |
| `F_modality_combo_ablation.png` | Ablation across 16 modality subsets: removing any single modality reduces BACC |
| `G_final_rep_hexbin_acr_cls.png` | Biopsy UMAP (N=916), 2×4 panel layout |
| `I_seed_risk_stratification.png` | ACR+ vs ACR− biopsies in seed space |
| `K_seed_attribution_acr_cls.png` | Top seeds: Clinical-s0, CD8-T-2 (BAL), CT-s04/s07 (protective) |

**Key finding:** Clinical labs (creatinine, lymphocytes) are the strongest single discriminator; all modalities contribute uniquely.

---

## `interpretability/acr_surv/` — ACR Survival
**Model:** `longitudinal_mk_no_alibi` · split 2 fold 0 · C-index 0.679

| File | What it shows |
|------|--------------|
| `L_global_weight_heatmap.png` | Biopsy weights: concentrated in lower-left (early post-Tx visits) — year-1 immune set-point dominates |
| `Lpop_K_seed_attribution_acr_surv.png` | TRAM-3/6 (BAL) → long TTE; HE inflammation seeds → short TTE |
| `Lpop_rep_umap_acr_surv.png` | Patient UMAP coloured by TTE quartile |

**Key finding:** The model discovers, without temporal supervision, that early post-transplant biopsies determine long-term rejection trajectory.

---

## `interpretability/death/` — Death Survival
**Model:** `longitudinal_mk_no_alibi` · split 2 fold 0 · C-index 0.771

| File | What it shows |
|------|--------------|
| `L_global_weight_heatmap.png` | Biopsy weights: uniformly high across all dates — full history matters equally |
| `Lpop_alpha_recency.png` | Weight vs recency: no recency bias (flat), confirming full-history integration |
| `Lpop_K_seed_attribution_death.png` | Top seeds: HE-s11/15/02 (protective, Δα ≈ −0.016); CT-s13/11 (risky, Δα ≈ +0.009) |
| `Lpop_rep_umap_death.png` | Patient UMAP: clear survivor (blue, left) vs non-survivor (red, right) clusters; stable across anchor dates |

**Key finding:** All HE seeds are protective (alveolar tissue = survival signal); CT structural clusters mark non-survivors. Full biopsy history integrates cumulative allograft burden.

---

## `interpretability/clad/` — CLAD Survival
**Model:** `set_mil_mt` · split 2 fold 0 · C-index 0.563

| File | What it shows |
|------|--------------|
| `A_instance_reps_cosine.png` | Patch similarity structure |
| `B_seeds.png` | K=16 seed UMAPs per modality |
| `D_abmil_seed_importance.png` | CT seeds dominate ABMIL attention |
| `G_final_rep_hexbin_clad_surv.png` | Biopsy UMAP for CLAD, 2×4 panel layout |
| `K_seed_attribution_clad.png` | CT seeds 5, 13, 21 are top positive predictors; MoAM (BAL) is top BAL predictor |

**Key finding:** CLAD is primarily a structural disease — CT and clinical data dominate; BAL MoAM marks high CLAD risk.

---

## `interpretability/agg/` — Seed Attribution Aggregated (5 splits)

Aggregated Δα = mean attention (high-risk) − mean attention (low-risk), computed over all test patients across 5 splits.

| File | Task | Model |
|------|------|-------|
| `Lpop_K_agg_acr_cls.png` | ACR cls | set_mil_mt_no_sab |
| `Lpop_K_agg_acr_surv.png` | ACR surv | longitudinal_mk_no_alibi |
| `longitudinal_mk_no_alibi_Lpop_K_agg_death_surv.png` | Death | longitudinal_mk_no_alibi |
| `longitudinal_mk_no_alibi_Lpop_K_agg_clad_surv.png` | CLAD | longitudinal_mk_no_alibi |
| `longitudinal_mk_mt_no_alibi_Lpop_K_agg_death_surv.png` | Death | longitudinal_mk_mt_no_alibi |
| `longitudinal_mk_mt_no_alibi_Lpop_K_agg_clad_surv.png` | CLAD | longitudinal_mk_mt_no_alibi |

PDF versions also available (same names, `.pdf` extension).

---

## `interpretability/cluster_agg/` — Biological Cluster Attribution (300 DPI, paper quality)

Δ affinity per named biological cluster = mean affinity in high-risk biopsies − mean affinity in low-risk biopsies. Aggregated across 5 splits. Clusters derived from published BAL cell-type annotations and HE/CT unsupervised clustering.

| File | Task |
|------|------|
| `death_cluster_aff_agg.png` | Death — TRAM-4 most protective; CT-struct clusters most risky |
| `acr_surv_cluster_aff_agg.png` | ACR survival — TRAM-3/6 protective; HE inflammation risky |
| `acr_cls_cluster_aff_agg.png` | ACR classification — CD8-T-cell-2 risky; some CT clusters protective |
| `clad_cluster_aff_agg.png` | CLAD — MoAM/Monocytes risky; CT structural clusters dominant |

PDF versions also available.

**Key finding (cross-task):**  
- **Death & ACR surv:** TRAM ↔ survival, MoAM ↔ non-survival (macrophage homeostasis axis)  
- **ACR cls:** CD8 T cells + clinical labs are the primary signal  
- **CLAD:** CT structural deterioration + MoAM are dominant

---

## `interpretability/unimodal_ablation/` — Unimodal Ablation

Each modality trained in isolation (and all pairs) to isolate per-modality contribution.

| File | Description |
|------|-------------|
| `unimodal_ablation_barplot.png` | Bar chart: each modality subset vs full multimodal model, per task |
| `unimodal_ablation_heatmap.png` | Heatmap: metric (rows=tasks, cols=modality subsets) |
| `unimodal_ablation_summary.csv` | Mean ± std across splits, per modality subset × task |
| `unimodal_ablation_raw.csv` | Per-split raw values |

**Key finding:** No single modality matches the full multimodal model. Clinical + H&E is the strongest pair. CT alone is surprisingly competitive for CLAD.

---

## `trajectories/` — Patient Risk Trajectories (Death Survival)

**Model:** `longitudinal_mk_no_alibi` · Predicted log-hazard at each biopsy visit.

| File | Description |
|------|-------------|
| `Fig7_patient_trajectories.png` | 4-panel composite: one patient per archetype |
| `panel_A_LT100.png` | Stable low-risk survivor (consistent low logit through follow-up) |
| `panel_B_LT119.png` | Early-onset non-survivor (high risk from first post-Tx biopsy) |
| `panel_C_LT062.png` | Late-escalating non-survivor (risk rises after year 1, coincides with CLAD onset) |
| `panel_D_LT227.png` | Treatment-responsive: risk decreases after immunosuppression adjustment |

PDF also available (`Fig7_patient_trajectories.pdf`).

**Clinical use:** At each biopsy visit the model produces an updated log-hazard score. A threshold of logit > 1.5 is proposed for clinical flagging and surveillance intensification.

---

## Reproducibility

All figures generated by:
- `interpretability/interpret_set_mil_mt.py` → acr_cls/, clad/ panels
- `interpretability/interpret_longitudinal_mk.py` → acr_surv/, death/ panels  
- `interpretability/gen_cluster_aff_agg.py` → cluster_agg/ panels
- `interpretability/regen_G_panel.py` → G hexbin panels (CPU, from cached results_raw.npy)
- Unimodal ablation: `train_mm_abmil_v8.py` with single-modality config

Cached inference results: `results/results_raw.npy` (916 records, no GPU re-inference needed for interpretability).
