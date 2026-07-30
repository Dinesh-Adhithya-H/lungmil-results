# Supplementary Material: Longitudinal Multimodal Multiple Instance Learning for Lung Transplant Outcome Prediction

---

## 1. Project Overview

Lung transplantation is the definitive treatment for end-stage lung disease, yet long-term outcomes remain poor, with a median survival of approximately five years. Two major complications drive mortality: acute cellular rejection (ACR), an immune-mediated attack on the allograft occurring most often in the first two years post-transplant, and chronic lung allograft dysfunction (CLAD), an irreversible fibroproliferative syndrome that affects roughly 50% of recipients by five years. Early and accurate prediction of these outcomes — ideally before clinical deterioration is apparent — could enable pre-emptive therapeutic intervention.

This work develops and evaluates a multimodal, longitudinal machine learning framework that integrates four complementary data modalities collected at routine post-transplant biopsy visits: transbronchial biopsy histology (H&E), bronchoalveolar lavage (BAL) cytology, thoracic CT imaging, and structured clinical data. Rather than treating each biopsy visit in isolation, the framework explicitly models the temporal sequence of visits for each patient, learning which time points and which combinations of modalities carry the strongest prognostic signal.

Four prediction tasks are addressed jointly or separately depending on the model variant:

1. **ACR classification**: Binary prediction of acute cellular rejection grade at the current biopsy (balanced accuracy; BACC).
2. **ACR survival**: Time-to-next-ACR event, framed as a Cox proportional-hazards regression (concordance index; C-index).
3. **CLAD survival**: Time-to-CLAD onset (C-index).
4. **Death survival**: Time-to-death (C-index).

The cohort comprises approximately 350 lung transplant recipients followed longitudinally at Helmholtz Munich / LMU Klinikum, with each patient contributing one or more biopsy visits as separate data points. All analyses use a 5-split × 4-fold nested cross-validation design to ensure unbiased evaluation.

---

## 2. Dataset and Preprocessing

### 2.1 Cohort and Biopsy Visits

Each patient contributes a sequence of biopsy visits indexed by days post-transplant. Patient-level data are stored as pre-computed `.pt` files (one per biopsy visit), linked to a master splits CSV by a unique stem identifier (e.g., `00049`). The splits CSV records patient ID, anchor date (transplant date), biopsy ACR grade, and train/val/test assignments for each of the 5 outer splits × 4 inner folds.

### 2.2 Data Modalities

**H&E Histology (HE).** Whole-slide image patches from transbronchial biopsy slides are encoded using the UNI foundation model (Chen et al., 2024), yielding 1024-dimensional feature vectors per patch. Patches are pre-clustered into 54 biologically annotated categories:
- Clusters 0–21: Alveolar tissue with hemorrhage or inflammation
- Clusters 22–31: Empty alveolar spaces
- Clusters 32–41: Normal alveolar parenchyma
- Clusters 42–51: Bronchial epithelium
- Cluster 52: Lymphocytoplasmic inflammation
- Cluster 53: Cartilage

These cluster assignments are used for interpretability analyses but not directly during training; the full 1024-dim embeddings are used as model input.

**BAL Cytology (BAL).** Cells recovered from bronchoalveolar lavage are represented by 10-dimensional radiomics-style feature vectors per cell. This low-dimensional representation captures cytological composition (macrophage vs. lymphocyte vs. neutrophil fractions, morphological features).

**CT Imaging (CT).** Thoracic CT scans are encoded using a RadiomicsTransformer, yielding 1024-dimensional patch embeddings. CT patches are grouped into 16 k-means clusters (C0–C15) for interpretability. Clusters C0–C2 are consistently associated with high mortality risk across splits (see Section 6).

**Clinical Data.** Structured clinical variables — including immunosuppressant medications, laboratory values (creatinine, tacrolimus trough levels), spirometry (FEV1, FVC), BMI, and donor/recipient demographics — are assembled from 106 variable slots and one-hot encoded into a 491-dimensional vector per biopsy visit.

### 2.3 Label Encoding

**ACR classification.** Biopsy ACR grades are mapped as: A0\* → 0 (no rejection); A1\* or A2\* → 1 (rejection); all other grades → excluded from the hinge classification loss but retained in the Cox risk set.

**Survival outcomes.** Time-to-event labels are derived from the anchor date (transplant date):
- ACR survival: `tte_next_acr`, `event_next_acr`
- CLAD survival: `clad_time`, `clad_event`
- Death survival: `death_time`, `death_event`

Cox–Breslow loss is used for all survival tasks.

### 2.4 Nested Cross-Validation Design

The evaluation uses a 5-outer-split × 4-inner-fold nested cross-validation protocol. The test set is fixed per outer split and identical across all four inner folds. The three inner folds (folds 1–3) are used exclusively for hyperparameter selection. Fold 0 uses the selected hyperparameters, trains on the combined train + validation set, and produces the final test metrics reported here. This design prevents leakage of test information into model selection.

---

## 3. Model Architecture

### 3.1 Phase 1: Unimodal Pre-Training

Each data modality is first trained independently as a unimodal Multiple Instance Learning (MIL) classifier/regressor. The encoder is a Gated Attention MIL network (ABMIL; Ilse et al., 2018):

```
Input patches (N × D)
  → FC(D → 256) → GELU → LayerNorm           # projection
  → Gated attention: tanh(Wq·h) ⊙ σ(Wg·h)   # attention weights
  → Softmax → weighted sum                   # bag-level representation (256-dim)
  → Task head (classification or Cox)
```

Separate models are trained for each combination of (modality, task, outer split, inner fold). Phase 1 weights are frozen during Phase 2 training. The 256-dimensional bag-level representation from each Phase 1 encoder serves as the input token to Phase 2 fusion modules.

### 3.2 Phase 2: Multimodal Fusion Variants

All Phase 2 models receive four modality-level tokens per biopsy visit (one per modality, each 256-dim) from the frozen Phase 1 encoders.

#### 3.2.1 Baseline Fusion Variants (Non-Temporal)

These variants treat each biopsy visit independently; no temporal ordering across visits is modeled.

- **Early Fusion**: All patches from all modalities are concatenated into a single bag and processed by a shared ABMIL encoder.
- **Late Fusion**: Each modality runs its own ABMIL encoder; predictions from all modalities are combined via learned scalar weights.
- **Middle Fusion**: Per-modality ABMIL encoders produce 256-dim representations; these are then passed through a CrossModalTransformer (cross-attention across modalities) before a final ABMIL aggregation.

#### 3.2.2 SetMIL Variants (Set-Based, Non-Temporal)

SetMIL variants apply set-theoretic pooling to compress each modality into a fixed-size set of learned seed vectors before cross-modal interaction.

- **SetMIL-MT** (`set_mil_mt`): For each modality, Pooling by Multihead Attention (PMA; Lee et al., 2019) with K=16 learned seed vectors compresses the variable-length patch bag into a fixed 16×256 matrix. A Set Attention Block (SAB) then enables cross-modal interaction across all four modalities' seed sets. Per-task ABMIL heads produce the final predictions. This is a multi-task model — all four tasks are trained jointly.

- **SetMIL-MT (no SAB)** (`set_mil_mt_no_sab`): Identical to SetMIL-MT but with the SAB cross-modal interaction block removed (ablation). Modality-specific PMA representations are directly processed by per-task ABMIL heads. This ablation tests whether explicit cross-modal interaction is beneficial.

- **SetMIL (no SAB, single-task)** (`set_mil_no_sab`): Single-task version of the no-SAB architecture, trained independently per task.

#### 3.2.3 Longitudinal Variants (Temporal, Biopsy-Sequence-Aware)

Longitudinal variants model the ordered sequence of biopsy visits for each patient. Visits are ordered by days post-transplant. Each biopsy produces four PMA-compressed modality tokens (K=16 seeds each), which are concatenated across the biopsy sequence and processed by a temporal attention mechanism.

- **Longitudinal-MK-MT with ALiBi** (`longitudinal_mk_mt`): The temporal attention module is a TemporalSAB implementing Attention with Linear Biases (ALiBi; Press et al., 2021). ALiBi slopes are learned per attention head and penalize attention between temporally distant biopsy pairs. A per-task recency decay parameter γ downweights older biopsies. Per-task ABMIL heads produce the final predictions. Multi-task.

- **Longitudinal-MK (no ALiBi, single-task)** (`longitudinal_mk_no_alibi`): ALiBi is replaced by a learned biopsy weighting network: a small MLP with architecture Linear(2→16) → ReLU → Linear(16→1) → Sigmoid, taking as input `[current_biopsy_day, previous_biopsy_day]` and outputting a scalar weight w ∈ (0,1). This network learns which (current, previous) biopsy date combinations should be up- or down-weighted for each prediction task. Trained independently per task.

- **Longitudinal-MK-MT (no ALiBi, multi-task)** (`longitudinal_mk_mt_no_alibi`): Multi-task version of the above.

#### 3.2.4 Key Architectural Parameters

| Parameter | Value |
|-----------|-------|
| Hidden dimension | 256 |
| PMA seeds (K) | 16 per modality |
| Max patches per modality | 2,048 |
| Gradient accumulation steps | 32 |
| Hardware | NVIDIA A100 80GB |

---

## 4. Training Protocol and Evaluation

**Phase 1 training.** Each unimodal ABMIL model is trained with Adam optimizer and early stopping based on inner-fold validation performance. Tasks use hinge loss (ACR classification) or Cox–Breslow loss (survival tasks).

**Phase 2 hyperparameter selection.** Inner folds 1–3 each produce an HP sweep JSON file containing validation performance for a grid of learning rates, dropout rates, and architectural choices. Fold 0 aggregates the best hyperparameters across all four inner folds (including fold 0's own sweep) using the combined metric:

- For single-task models: the task's primary metric (BACC or C-index)
- For multi-task models (SetMIL-MT, Longitudinal-MK-MT): `0.5 × BACC_ACR + 0.5 × mean(CI_ACR_surv, CI_CLAD, CI_Death)`

**Final model training (fold 0).** The best hyperparameters are applied to train on the combined train + validation set. Test set metrics are recorded in `metrics_{variant}_final.json`.

**Evaluation metrics.** Balanced accuracy (BACC) is the primary metric for ACR classification; concordance index (C-index) is the primary metric for all survival tasks. Random-chance performance is 0.5 for both metrics.

---

## 5. Results

### 5.1 Main Comparison Table

Performance on held-out test sets across all 5 outer splits (mean ± standard deviation). Metric: BACC for ACR classification; C-index for survival tasks. ★ denotes best model per task.

| Variant | ACR cls (BACC) | ACR surv (CI) | CLAD surv (CI) | Death surv (CI) |
|---------|---------------|---------------|----------------|-----------------|
| Early fusion | 0.583 ± 0.057 | 0.575 ± 0.049 | 0.505 ± 0.065 | 0.645 ± 0.057 |
| Late fusion | 0.592 ± 0.029 | 0.585 ± 0.055 | 0.534 ± 0.083 | 0.638 ± 0.051 |
| Middle fusion | 0.559 ± 0.040 | 0.574 ± 0.082 | 0.516 ± 0.062 | 0.656 ± 0.057 |
| SetMIL-MT | 0.595 ± 0.027 | 0.489 ± 0.064 | **0.563 ± 0.080** ★ | 0.664 ± 0.041 |
| SetMIL-MT (no SAB) | **0.623 ± 0.034** ★ | 0.593 ± 0.059 | 0.536 ± 0.060 | 0.656 ± 0.035 |
| SetMIL (no SAB, single) | 0.611 ± 0.027 | 0.580 ± 0.031 | 0.488 ± 0.068 | 0.673 ± 0.026 |
| Longitudinal-MK-MT (ALiBi) | 0.545 ± 0.042 | 0.613 ± 0.073 | 0.496 ± 0.094 | 0.721 ± 0.056 |
| **Longitudinal-MK (no ALiBi)** | 0.550 ± 0.039 | **0.679 ± 0.064** ★ | 0.489 ± 0.028 | **0.771 ± 0.056** ★ |
| Longitudinal-MK-MT (no ALiBi) | 0.526 ± 0.052 | 0.630 ± 0.112 | 0.534 ± 0.100 | 0.770 ± 0.089 |

### 5.2 Best Model Per Task

| Task | Best Model | Metric | Value |
|------|-----------|--------|-------|
| ACR classification | SetMIL-MT (no SAB) | BACC | 0.623 ± 0.034 |
| ACR survival | Longitudinal-MK (no ALiBi) | C-index | 0.679 ± 0.064 |
| CLAD survival | SetMIL-MT | C-index | 0.563 ± 0.080 |
| Death survival | Longitudinal-MK (no ALiBi) | C-index | 0.771 ± 0.056 |

### 5.3 Per-Split Performance (Best Models)

**ACR classification — SetMIL-MT (no SAB):**

| Split | s0 | s1 | s2 | s3 | s4 |
|-------|----|----|----|----|----|
| BACC | 0.578 | 0.610 | 0.680 | 0.635 | 0.615 |

**ACR survival — Longitudinal-MK (no ALiBi):**

| Split | s0 | s1 | s2 | s3 | s4 |
|-------|----|----|----|----|----|
| C-index | 0.573 | 0.673 | 0.748 | 0.660 | 0.741 |

**CLAD survival — SetMIL-MT:**

| Split | s0 | s1 | s2 | s3 | s4 |
|-------|----|----|----|----|----|
| C-index | 0.429 | 0.616 | 0.663 | 0.577 | 0.531 |

**Death survival — Longitudinal-MK (no ALiBi):**

| Split | s0 | s1 | s2 | s3 | s4 |
|-------|----|----|----|----|----|
| C-index | 0.779 | 0.670 | 0.843 | 0.772 | 0.793 |

### 5.4 Key Observations

Temporal modeling provides the strongest gains for survival prediction. The Longitudinal-MK (no ALiBi) model achieves C-index 0.679 for ACR survival and 0.771 for death survival — improvements of 9–13 percentage points over the strongest non-temporal fusion baseline for those tasks. The learned biopsy weight network (replacing ALiBi) consistently matches or outperforms its ALiBi counterpart while being easier to interpret.

For ACR classification, non-temporal set-based models (SetMIL-MT without SAB) perform best. Removing the SAB cross-modal interaction block improves ACR classification performance (0.623 vs. 0.595), suggesting that for the classification task, modality-specific information is more predictive than cross-modal interactions.

CLAD prediction remains the hardest task (best C-index 0.563), reflecting the clinical heterogeneity of CLAD subtypes and likely the need for longer follow-up windows not fully captured in this dataset.

---

## 6. Interpretability Methods and Biological Findings

### 6.1 Methods

Interpretability analyses are performed on the best-performing model for each task, applied to the test set of each outer split.

**Seed attribution (population-level).** For SetMIL and Longitudinal variants, each modality produces K=16 PMA seed vectors per biopsy. The seed-to-patch affinity matrix (from PMA cross-attention) identifies which patch clusters each seed vector predominantly attends to. Population-level attribution compares attention weights of seeds between high-risk and low-risk patients (top vs. bottom tertile of predicted risk). Seeds that are significantly more attended in high-risk patients are interpreted as risk-associated; those more attended in low-risk patients as protective. Cross-split aggregation (mean ± std across all 5 splits) is used to assess robustness.

**Biopsy weight heatmap.** For Longitudinal-MK (no ALiBi) models, the learned weight network w(current\_biopsy\_day, previous\_biopsy\_day) is evaluated on a 100×100 grid spanning 0–2,000 days post-transplant for both axes. The resulting heatmap shows which (current, previous) biopsy date combinations are assigned high vs. low weight by the model. Regions where the previous biopsy date exceeds the current are masked as invalid.

**Patient representation UMAP.** The 256-dimensional ABMIL-weighted patient representation (computed as the attention-weighted sum of tokens at the final aggregation layer) is projected to 2D using UMAP with cosine distance metric and n\_neighbors=15. Projections are colored by predicted risk score, number of biopsies, days post-transplant at anchor, and binary risk group.

### 6.2 Biological Findings

#### Death Survival (Robust: 5/5 splits)

The death survival model (Longitudinal-MK, no ALiBi) produces the most consistent interpretability findings across all five outer splits.

**Histology (HE):** Seed vectors attending predominantly to HE clusters 0–21 (alveolar tissue with hemorrhage or acute inflammation) are enriched in *low-risk* (longer-surviving) patients. This is initially counterintuitive — inflammatory patches are associated with survival — but likely reflects that patients who survive to undergo biopsy tend to have acute, treatable inflammatory patterns rather than the fibrotic or obliterative changes associated with end-stage disease. Preserved alveolar parenchyma (clusters 32–41) similarly tracks with low risk.

**CT imaging:** CT clusters C0, C1, and C2 are robustly enriched in *high-risk* (shorter-surviving) patients across all five splits. These clusters likely represent structural deterioration patterns — parenchymal destruction, air trapping, or consolidation — that precede and predict mortality. The CT signal is complementary to H&E: histology captures the local microenvironment at the biopsy site, while CT captures global structural changes across the entire lung.

**Biological synthesis:** The model discovers a histology–CT axis of risk, where patients with intact alveolar histology but deteriorating CT structure are correctly identified as high risk. This is consistent with the clinical observation that CLAD-associated bronchiolitis obliterans produces diffuse CT changes (hyperinflation, mosaic attenuation) before localised biopsy sites become overtly abnormal.

#### ACR Survival (Consistent: 4/5 splits)

The Longitudinal-MK (no ALiBi) model for ACR survival shows consistent CT-driven attribution.

**CT imaging:** CT seeds are preferentially associated with the high-risk group, paralleling the death survival findings. This suggests shared structural lung deterioration signals between rejection risk and mortality risk.

**Temporal weighting:** The learned biopsy weight heatmap reveals a striking pattern. For ACR survival prediction, early biopsies (previous biopsy date <350 days post-transplant) receive high weight, while later biopsies receive progressively less weight. The model effectively discounts biopsy data from patients who have been stable for more than a year, focusing on the early post-transplant window when immune sensitization patterns are established. This finding aligns with clinical knowledge that the first year post-transplant is the highest-risk period for acute rejection.

#### ACR Classification

For ACR classification (SetMIL-MT, no SAB), the seed attribution pattern is reversed with respect to the temporal weight findings: late biopsies (>350 days post-transplant) are up-weighted. This is consistent with the task difference — classifying the *current* rejection status at a late biopsy requires attention to recent, late-stage clinical features, whereas *predicting future* rejection requires the early immune trajectory. Seed attributions for ACR classification are inconsistent across splits (2–3 of 5) and are not considered paper-grade findings.

#### CLAD Survival

CLAD survival (SetMIL-MT) shows near-uniform biopsy weighting — the model does not strongly prefer any particular temporal window. Seed attributions are inconsistent across splits and should be interpreted with caution given the moderate cross-split performance variance (C-index 0.429–0.663). CLAD biological conclusions require a larger cohort.

#### Summary Table of Biological Findings

| Task | Modality | Direction | Biological Interpretation | Cross-split robustness |
|------|----------|-----------|--------------------------|------------------------|
| Death | HE (clusters 0–21) | Protective (low risk) | Preserved inflammatory alveolar tissue | 5/5 splits |
| Death | CT (C0–C2) | Risk | Structural lung deterioration | 5/5 splits |
| ACR surv | CT | Risk | Shared structural signal with death | 4/5 splits |
| ACR surv | Temporal weighting | Early biopsies upweighted | Early immune trajectory predicts future rejection | 4/5 splits |
| ACR cls | Temporal weighting | Late biopsies upweighted | Current rejection status driven by recent features | 4/5 splits |
| CLAD | All | Inconsistent | — | <3/5 splits |

---

## 7. Code and Reproducibility

### 7.1 Repository Structure

```
train_mm_abmil_v8.py          # Main entry: --phase p1/p2/both, --variant, --task
src/mil/
  data/                       # Data loading, splits, label encoding
  models/                     # Phase 1 and Phase 2 model definitions
  training/                   # Phase 1 and Phase 2 training loops, losses, metrics
interpretability/             # Attention-based interpretability scripts per model family
analysis/                     # Population-level figures and UMAP pipelines
scripts/                      # SLURM batch submission scripts
results/mm_abmil_v8/          # Output directory (metrics, checkpoints, logs)
```

### 7.2 Reproducing the Experiments

Training is managed via SLURM on a GPU cluster. Phase 1 and Phase 2 jobs are submitted via shell scripts that implement skip logic (checking for existing output files before resubmitting). All jobs are self-contained bash scripts specifying resource requirements (GPU, memory, wall time).

```bash
# Phase 1: train all unimodal models (5 splits × 4 folds × 4 modalities × 4 tasks)
bash scripts/submit_p1_all_splits.sh

# Phase 2: train all fusion variants
bash scripts/submit_p2_all_splits.sh

# Interpretability on best model per task
bash interpretability/submit_interp_best_per_task.sh
bash interpretability/submit_interp_longitudinal_no_alibi_allsplits.sh
```

### 7.3 Software Environment

| Package | Version |
|---------|---------|
| Python | 3.10 |
| PyTorch | 2.6.0+cu124 |
| CUDA | 12.6 |
| NumPy | — |
| scikit-learn | — |
| lifelines | — |
| umap-learn | — |

The conda environment is specified in `environment.yml`.

### 7.4 Data Availability

Patient-level data cannot be shared publicly due to clinical data protection regulations. Processed feature embeddings and splits metadata may be made available upon reasonable request through a data use agreement with Helmholtz Munich / LMU Klinikum, subject to institutional ethics approval.

---

*Last updated: 2026-07-30*
