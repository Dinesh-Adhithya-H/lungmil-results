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

The cohort comprises 263 lung transplant recipients (4,210 biopsy visits) followed longitudinally at Helmholtz Munich / LMU Klinikum, with each patient contributing one or more biopsy visits as separate data points. All analyses use a 5-split × 4-fold nested cross-validation design to ensure unbiased evaluation.

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

Performance on held-out test sets across all 5 outer splits (mean ± standard deviation). Metric: BACC for ACR classification; C-index for survival tasks. ★ denotes best model per task. The complete 18-model benchmark — including per-modality Linear and ABMIL unimodal baselines and their weighted-average ensembles — is shown in Figure 3a (benchmark_table_v2).

| Variant | ACR cls (BACC) | ACR surv (CI) | CLAD surv (CI) | Death surv (CI) |
|---------|---------------|---------------|----------------|-----------------|
| Early fusion | 0.583 ± 0.063 | 0.575 ± 0.055 | 0.505 ± 0.073 | 0.645 ± 0.064 |
| Late fusion | 0.592 ± 0.032 | 0.585 ± 0.061 | 0.534 ± 0.093 | 0.638 ± 0.057 |
| Middle fusion | 0.559 ± 0.045 | 0.574 ± 0.092 | 0.516 ± 0.069 | 0.656 ± 0.063 |
| SetMIL-MT | 0.595 ± 0.031 | 0.489 ± 0.072 | **0.563 ± 0.080** ★ | 0.664 ± 0.046 |
| SetMIL-MT (no SAB) | **0.623 ± 0.034** ★ | 0.593 ± 0.066 | 0.536 ± 0.067 | 0.656 ± 0.039 |
| SetMIL | 0.611 ± 0.030 | 0.580 ± 0.034 | 0.488 ± 0.076 | 0.673 ± 0.029 |
| LongMK-MT | 0.526 ± 0.058 | 0.630 ± 0.125 | 0.534 ± 0.112 | 0.770 ± 0.099 |
| **LongMK** | 0.550 ± 0.043 | **0.679 ± 0.071** ★ | 0.489 ± 0.031 | **0.771 ± 0.063** ★ |

### 5.2 Best Model Per Task

| Task | Best Model | Metric | Value |
|------|-----------|--------|-------|
| ACR classification | SetMIL-MT (no SAB) | BACC | 0.623 ± 0.034 |
| ACR survival | LongMK | C-index | 0.679 ± 0.064 |
| CLAD survival | SetMIL-MT | C-index | 0.563 ± 0.080 |
| Death survival | LongMK | C-index | 0.771 ± 0.056 |

### 5.3 Per-Split Performance (All DL Variants)

Per-split values s0–s4 for all deep-learning model variants. Linear and ABMIL unimodal baselines are tabulated in Figure 3a. All splits use fold 0 (train+val combined).

**ACR classification (BACC):**

| Variant | s0 | s1 | s2 | s3 | s4 | Mean ± Std |
|---------|----|----|----|----|----|----|
| Early fusion | 0.612 | 0.599 | 0.632 | 0.472 | 0.600 | 0.583 ± 0.063 |
| Middle fusion | 0.522 | 0.520 | 0.616 | 0.540 | 0.599 | 0.559 ± 0.045 |
| Late fusion | 0.594 | 0.596 | 0.640 | 0.550 | 0.580 | 0.592 ± 0.032 |
| SetMIL-MT | 0.597 | 0.546 | 0.597 | 0.605 | 0.630 | 0.595 ± 0.031 |
| **SetMIL-MT (no SAB)** ★ | **0.588** | **0.623** | **0.681** | **0.617** | **0.606** | **0.623 ± 0.034** |
| SetMIL | 0.644 | 0.564 | 0.626 | 0.601 | 0.619 | 0.611 ± 0.030 |
| LongMK-MT | 0.460 | 0.570 | 0.504 | 0.493 | 0.602 | 0.526 ± 0.058 |
| LongMK | 0.546 | 0.565 | 0.510 | 0.512 | 0.615 | 0.550 ± 0.043 |

**ACR survival (C-index):**

| Variant | s0 | s1 | s2 | s3 | s4 | Mean ± Std |
|---------|----|----|----|----|----|----|
| Early fusion | 0.550 | 0.661 | 0.598 | 0.527 | 0.540 | 0.575 ± 0.055 |
| Middle fusion | 0.526 | 0.613 | 0.708 | 0.466 | 0.557 | 0.574 ± 0.092 |
| Late fusion | 0.559 | 0.665 | 0.635 | 0.525 | 0.541 | 0.585 ± 0.061 |
| SetMIL-MT | 0.585 | 0.539 | 0.454 | 0.467 | 0.403 | 0.489 ± 0.072 |
| SetMIL-MT (no SAB) | 0.541 | 0.668 | 0.610 | 0.509 | 0.634 | 0.593 ± 0.066 |
| SetMIL | 0.584 | 0.596 | 0.585 | 0.523 | 0.614 | 0.580 ± 0.034 |
| LongMK-MT | 0.557 | 0.539 | 0.539 | 0.690 | 0.823 | 0.630 ± 0.125 |
| **LongMK** ★ | **0.573** | **0.673** | **0.748** | **0.660** | **0.741** | **0.679 ± 0.071** |

**CLAD survival (C-index):**

| Variant | s0 | s1 | s2 | s3 | s4 | Mean ± Std |
|---------|----|----|----|----|----|----|
| Early fusion | 0.432 | 0.622 | 0.495 | 0.460 | 0.516 | 0.505 ± 0.073 |
| Middle fusion | 0.429 | 0.610 | 0.537 | 0.470 | 0.532 | 0.516 ± 0.069 |
| Late fusion | 0.372 | 0.583 | 0.603 | 0.553 | 0.561 | 0.534 ± 0.093 |
| **SetMIL-MT** ★ | **0.528** | **0.626** | **0.669** | **0.547** | **0.446** | **0.563 ± 0.080** |
| SetMIL-MT (no SAB) | 0.476 | 0.619 | 0.528 | 0.469 | 0.589 | 0.536 ± 0.067 |
| SetMIL | 0.478 | 0.605 | 0.451 | 0.401 | 0.503 | 0.488 ± 0.076 |
| LongMK-MT | 0.721 | 0.456 | 0.523 | 0.439 | 0.533 | 0.534 ± 0.112 |
| LongMK | 0.461 | 0.495 | 0.516 | 0.453 | 0.520 | 0.489 ± 0.031 |

**Death survival (C-index):**

| Variant | s0 | s1 | s2 | s3 | s4 | Mean ± Std |
|---------|----|----|----|----|----|----|
| Early fusion | 0.550 | 0.640 | 0.679 | 0.635 | 0.721 | 0.645 ± 0.064 |
| Middle fusion | 0.555 | 0.643 | 0.707 | 0.666 | 0.711 | 0.656 ± 0.063 |
| Late fusion | 0.551 | 0.624 | 0.693 | 0.638 | 0.685 | 0.638 ± 0.057 |
| SetMIL-MT | 0.599 | 0.646 | 0.670 | 0.725 | 0.681 | 0.664 ± 0.046 |
| SetMIL-MT (no SAB) | 0.593 | 0.650 | 0.689 | 0.662 | 0.688 | 0.656 ± 0.039 |
| SetMIL | 0.625 | 0.671 | 0.684 | 0.695 | 0.691 | 0.673 ± 0.029 |
| LongMK-MT | 0.706 | 0.628 | 0.855 | 0.815 | 0.848 | 0.770 ± 0.099 |
| **LongMK** ★ | **0.779** | **0.670** | **0.843** | **0.772** | **0.793** | **0.771 ± 0.063** |

### 5.4 Key Observations

Temporal modeling provides the strongest gains for survival prediction. LongMK achieves C-index 0.679 for ACR survival and 0.771 for death survival — improvements of 9–13 percentage points over the strongest non-temporal fusion baseline for those tasks. The learned biopsy weight network (replacing ALiBi temporal bias) consistently matches or outperforms its ALiBi counterpart while being easier to interpret. LongMK-MT (the multi-task variant) achieves comparable death survival performance (0.770 ± 0.099) at the cost of higher variance, suggesting that multi-task training introduces useful inductive bias but also noisier optimization.

For ACR classification, non-temporal set-based models (SetMIL-MT without SAB) perform best. Removing the SAB cross-modal interaction block improves ACR classification performance (0.623 vs. 0.595), suggesting that for the classification task, modality-specific information is more predictive than explicit cross-modal interactions. This is consistent with the ACR grading system being based primarily on histological features at the biopsy site.

CLAD prediction remains the hardest task (best C-index 0.563), reflecting the clinical heterogeneity of CLAD subtypes and the need for longer follow-up windows. The best model for CLAD (SetMIL-MT) shows high inter-split variance (0.429–0.663), indicating sensitivity to patient composition in the test folds. A classical linear baseline using H&E cluster proportions achieves C-index 0.534, showing that non-DL features already capture a meaningful portion of the CLAD signal.

The LongMK models operate at patient level (N=226 unique patients) and produce representations from all biopsy visits per patient simultaneously. All other DL variants and linear baselines operate at biopsy level (N=4,210 biopsies). The out-of-sample guarantee holds for all variants: each patient's representation is produced by a model that was never trained on that patient's data.

---

## 6. Interpretability Methods and Biological Findings

### 6.1 Methods

Interpretability analyses are performed on the best-performing model for each task, applied to the test set of each outer split.

**Seed attribution (population-level).** For SetMIL and Longitudinal variants, each modality produces K=16 PMA seed vectors per biopsy. The seed-to-patch affinity matrix (from PMA cross-attention) identifies which patch clusters each seed vector predominantly attends to. Population-level attribution compares attention weights of seeds between high-risk and low-risk patients (top vs. bottom tertile of predicted risk). Seeds that are significantly more attended in high-risk patients are interpreted as risk-associated; those more attended in low-risk patients as protective. Cross-split aggregation (mean ± std across all 5 splits) is used to assess robustness.

**Biopsy weight heatmap.** For LongMK models, the learned weight network w(current\_biopsy\_day, previous\_biopsy\_day) is evaluated on a 100×100 grid with task-specific day ranges derived from the observed data in the splits CSV: ACR tasks use 0–2,000 days, CLAD survival uses 0–3,000 days, and Death survival uses 0–3,500 days (each rounded up to the nearest 500 from the observed maximum follow-up). The resulting heatmap shows which (current, previous) biopsy date combinations are assigned high vs. low weight by the model. Regions where the previous biopsy date exceeds the current biopsy date are masked as invalid (structural constraint). 5-split averaged heatmaps (mean and std across splits 0–4) are computed by extracting only the 4 weight tensors of the `biopsy_weight_net` MLP (~1 kB total) from each checkpoint on CPU, without loading the full model. Output heatmaps are saved per task and as a combined 4-panel aggregate.

**Patient representation UMAP.** The 256-dimensional ABMIL-weighted patient representation (computed as the attention-weighted sum of tokens at the final aggregation layer) is projected to 2D using UMAP with cosine distance metric and n\_neighbors=15. Projections are colored by predicted risk score, number of biopsies, days post-transplant at anchor, and binary risk group. A unified UMAP is computed across all four tasks simultaneously by pooling representations from the best model for each task, enabling direct comparison of how the latent space organizes by task-specific risk.

### 6.2 Biological Findings

#### Death Survival (Robust: 5/5 splits)

The death survival model (Longitudinal-MK, no ALiBi) produces the most consistent interpretability findings across all five outer splits.

**Histology (HE):** Seed vectors attending predominantly to HE clusters 0–21 (alveolar tissue with hemorrhage or acute inflammation) are enriched in *low-risk* (longer-surviving) patients. This is initially counterintuitive — inflammatory patches are associated with survival — but likely reflects that patients who survive to undergo biopsy tend to have acute, treatable inflammatory patterns rather than the fibrotic or obliterative changes associated with end-stage disease. Preserved alveolar parenchyma (clusters 32–41) similarly tracks with low risk.

**CT imaging:** CT clusters C0, C1, and C2 are robustly enriched in *high-risk* (shorter-surviving) patients across all five splits. These clusters likely represent structural deterioration patterns — parenchymal destruction, air trapping, or consolidation — that precede and predict mortality. The CT signal is complementary to H&E: histology captures the local microenvironment at the biopsy site, while CT captures global structural changes across the entire lung.

**Biological synthesis:** The model discovers a histology–CT axis of risk, where patients with intact alveolar histology but deteriorating CT structure are correctly identified as high risk. This is consistent with the clinical observation that CLAD-associated bronchiolitis obliterans produces diffuse CT changes (hyperinflation, mosaic attenuation) before localised biopsy sites become overtly abnormal.

**Clinical features:** Clinical attribution reveals a prognostically coherent set of structured variables. Low-risk (longer-surviving) patients show high attention to percent-predicted forced vital capacity (FVC%), glomerular filtration rate (GFR), and serum albumin — markers of preserved graft function, renal health, and nutritional status. High-risk (shorter-surviving) patients show high attention to donor risk score, primary graft dysfunction (PGD) at 72 hours post-transplant, and red cell distribution width (RDW). PGD-72h is the strongest established early predictor of long-term graft failure; elevated RDW reflects systemic inflammatory burden. The model thus recovers both early post-transplant graft insult signals (donor risk, PGD) and progressive systemic markers (RDW, albumin, GFR) without any prior labelling of their prognostic relevance. This Clinical attribution is consistent across 4/5 outer splits and complements the CT and BAL signals described above.

#### ACR Survival (Consistent: 4/5 splits)

The Longitudinal-MK (no ALiBi) model for ACR survival shows consistent CT-driven attribution.

**CT imaging:** CT seeds are preferentially associated with the high-risk group, paralleling the death survival findings. This suggests shared structural lung deterioration signals between rejection risk and mortality risk.

**Temporal weighting:** The learned biopsy weight heatmap reveals a striking pattern. For ACR survival prediction, early biopsies (previous biopsy date <350 days post-transplant) receive high weight, while later biopsies receive progressively less weight. The model effectively discounts biopsy data from patients who have been stable for more than a year, focusing on the early post-transplant window when immune sensitization patterns are established. This finding aligns with clinical knowledge that the first year post-transplant is the highest-risk period for acute rejection.

#### ACR Classification

For ACR classification (SetMIL-MT, no SAB), the seed attribution pattern is reversed with respect to the temporal weight findings: late biopsies (>350 days post-transplant) are up-weighted. This is consistent with the task difference — classifying the *current* rejection status at a late biopsy requires attention to recent, late-stage clinical features, whereas *predicting future* rejection requires the early immune trajectory. Seed attributions for ACR classification are inconsistent across splits (2–3 of 5) and are not considered paper-grade findings.

#### CLAD Survival

CLAD survival (SetMIL-MT) shows near-uniform biopsy weighting — the model does not strongly prefer any particular temporal window. Seed attributions are inconsistent across splits and should be interpreted with caution given the moderate cross-split performance variance (C-index 0.429–0.663). CLAD biological conclusions require a larger cohort.

#### BAL Cytology — Macrophage Subtype Stratification

BAL seed attribution analysis reveals a robust macrophage subtype stratification signal for death survival, consistent across splits.

**TRAM (Tissue-Resident Alveolar Macrophages):** Seeds attending to TRAM-associated cytological features (large, vacuolated macrophages with high forward scatter, characteristic of long-term tissue residency) are enriched in the *low-risk (surviving)* group. TRAM populations are maintained in a healthy alveolar niche and their depletion or replacement is associated with poor graft outcomes.

**MoAM/Monocyte-derived Macrophages:** Seeds attending to monocyte-derived macrophage (MoAM) features (smaller, less vacuolated, with higher nuclear-to-cytoplasm ratio) are enriched in the *high-risk (non-surviving)* group. MoAM influx into the airspace indicates ongoing recruitment from the bloodstream — a hallmark of unresolved alveolar injury and a known driver of fibroproliferative disease. This TRAM → MoAM shift in BAL composition serves as a sensitive early indicator of graft deterioration, preceding overt clinical decline.

This finding aligns with recent single-cell analyses showing that TRAM-to-MoAM ratio in BAL inversely correlates with CLAD-free survival in lung transplant recipients.

#### Unified Representation UMAP

A joint UMAP projection across all four prediction tasks reveals that the learned patient representations are task-specific rather than shared: representations from different task-trained models do not cluster together in the joint embedding, indicating that each task head has shaped the upstream representation toward distinct biological axes of variation. Within each task, the UMAP separates high-risk from low-risk patients along coherent trajectories. Patients with many biopsies (longitudinal depth) cluster in distinct regions, suggesting the temporal modeling enriches representations in a geometrically separable way relative to cross-sectional models.

#### Summary Table of Biological Findings

| Task | Modality | Direction | Biological Interpretation | Cross-split robustness |
|------|----------|-----------|--------------------------|------------------------|
| Death | HE (clusters 0–21) | Protective (low risk) | Preserved inflammatory alveolar tissue | 5/5 splits |
| Death | CT (C0–C2) | Risk | Structural lung deterioration | 5/5 splits |
| Death | BAL TRAM | Protective (low risk) | Tissue-resident alveolar macrophage maintenance | 4/5 splits |
| Death | BAL MoAM/Monocytes | Risk | Monocyte-derived macrophage influx = unresolved injury | 4/5 splits |
| Death | Clinical FVC%/GFR/Albumin | Protective (low risk) | Preserved pulmonary reserve and systemic health | 4/5 splits |
| Death | Clinical Donor risk/PGD/RDW | Risk | Early graft insult + systemic inflammatory burden | 4/5 splits |
| ACR surv | CT | Risk | Shared structural signal with death | 4/5 splits |
| ACR surv | Temporal weighting | Early biopsies upweighted | Early immune trajectory predicts future rejection | 4/5 splits |
| ACR cls | Temporal weighting | Late biopsies upweighted | Current rejection status driven by recent features | 4/5 splits |
| CLAD | Clinical (FEV1/spirometry) | Risk | Spirometric decline precedes CLAD diagnosis | 3/5 splits |
| CLAD | All other | Inconsistent | — | <3/5 splits |

---

## 7. Figure Legends

### Figure 1: Benchmark Bar Chart (benchmark_v2)

`figures/benchmark/benchmark_v2_{task}.pdf`

Bar charts showing held-out test performance (mean ± std across 5 outer splits) for all model variants, grouped by task. Models are arranged in fixed order: Linear unimodal variants (HE, BAL, CT, Clinical, wt avg Linear), ABMIL unimodal variants (HE, BAL, CT, Clinical, wt avg ABMIL), multimodal fusion variants (Early, Middle, Late), SetMIL variants (SetMIL, SetMIL-MT, SetMIL-MT no SAB), and longitudinal variants (LongMK-MT, LongMK). Each bar is colored by a consistent per-model palette (SHARED_MODEL_COLORS) used across all three benchmark figure types. Error bars show standard deviation across 5 splits. Metric: BACC for ACR classification; C-index for survival tasks.

### Figure 2: Benchmark Numeric Table (benchmark_table_v2)

`figures/benchmark/benchmark_table_v2.pdf`

18-model × 4-task table with RdYlGn background coloring per column (green = better, red = worse, normalized per task). Each cell shows mean±std on the top row and per-split values s0–s4 in monospace below. Group separator rows divide the table into five model families (Linear, ABMIL, Fusion, SetMIL, Longitudinal). The left margin stripe uses the SHARED_MODEL_COLORS palette. Missing cells (models not applicable to a task in certain configurations) are shown as em-dashes.

### Figure 3: Unimodal Ablation — Modality Contribution (unimodal_ablation_v2)

`figures/benchmark/unimodal_ablation_v2_{task}.pdf`

For each multimodal DL variant, bar groups show performance when each modality is ablated (removed) individually, evaluated via the model's built-in unimodal pathway. Models in the same fixed order as Figure 1. Each modality (HE, BAL, CT, Clinical) is shown as a bar within each model group, colored by modality. This quantifies the marginal contribution of each modality per model per task.

### Figure 4: Modality Combination Ablation (modality_combo)

`figures/benchmark/modality_combo_{task}.pdf`

For each model variant, bars show performance across all 15 non-empty subsets of the four modalities (HE, BAL, CT, Clinical), displayed in inclusion order. Longitudinal variants use the task-specific nested JSON key for correct metric extraction. The best-performing modality combination per variant is annotated. This figure identifies which modality subsets are sufficient for near-peak performance.

### Figure 5: Biopsy Weight Heatmaps — L_global (L_global heatmaps)

`figures/interpretability/{task}/L_global_weight_heatmap{_avg}.pdf`

Heatmaps of the learned biopsy weight function w(current\_day, previous\_day) ∈ (0,1) for the LongMK model. Each cell shows the scalar weight assigned to a (current biopsy day, previous biopsy day) pair. The lower-left triangle (valid region) is coloured on a blue–red diverging scale (blue = low weight, red = high weight). The upper-right triangle (previous > current) is masked grey. Day axes are data-driven: ACR tasks 0–2000 d, CLAD 0–3000 d, Death 0–3500 d. Single-split heatmaps show fold-0 weights; 5-split averaged heatmaps (suffix `_avg`) show the pixel-wise mean across all 5 outer splits with a separate panel for the standard deviation. An aggregate panel (`agg/L_global_weight_heatmap_avg_all.pdf`) shows all four tasks side by side.

### Figure 6: Cluster Affinity Aggregate (cluster_aff_agg)

`figures/interpretability/{task}/cluster_aff_agg_{risk}.pdf`

Population-level seed attribution heatmaps. For each PMA seed (rows: HE·s00–s15, BAL·s00–s15, CT·s00–s15, Clinical·s00–s15, 48 total) the mean attention to each patch cluster (columns: HE clusters 0–53, BAL clusters 0–7, CT clusters 0–15, Clinical feature groups) is shown, separately for the high-risk and low-risk tertile groups. Seeds are sorted by diversity (entropy of cluster affinity distribution). Rows within each modality block are labeled by modality prefix and seed index. Tab20 color blocks denote modality identity. Cross-split mean±std overlay is shown where available.

### Figure 7: Multimodal Seed Attribution — Instance Panels (multimod_seed_attribution)

`figures/interpretability/{task}/multimod_seed_attribution_{split}.pdf`

Per-patient panel combining (A) per-biopsy risk trajectory, (B) seed attention bar chart per modality, (C) patient UMAP position colored by risk, and (D) biopsy weight heatmap for that patient's visit sequence. Intended for paper main figures showing representative high-risk vs. low-risk patients.

### Figure 8: Kaplan–Meier Curves (km_curves)

`figures/km_curves/{task}_km.pdf`

Kaplan–Meier survival curves stratifying patients by predicted risk tertile (low / medium / high) from the best model for each task. Log-rank p-values are annotated. Shaded bands show 95% confidence intervals. Separate panels per outer split and an aggregate panel combining all splits.

### Figure 9: Unified Representation UMAP (unified_rep_umap)

`figures/interpretability/{task}/unified_rep_umap_{task}.pdf`  
`figures/interpretability/agg/unified_rep_umap_all_tasks.pdf`

UMAP projections (cosine distance, n\_neighbors=15) of the 256-dim ABMIL-pooled patient representations for the best model per task. Individual task UMAPs are colored by predicted risk score (continuous) and binary risk group. The aggregate panel (`agg/`) shows all four tasks in a 2×2 grid with a shared colorbar. Points are sized by number of biopsies per patient to highlight the effect of longitudinal depth on representation geometry.

---

## 8. Code and Reproducibility

### 8.1 Repository Structure

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

### 8.2 Reproducing the Experiments

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

### 8.3 Software Environment

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

### 8.4 Data Availability

Patient-level data cannot be shared publicly due to clinical data protection regulations. Processed feature embeddings and splits metadata may be made available upon reasonable request through a data use agreement with Helmholtz Munich / LMU Klinikum, subject to institutional ethics approval.

---

*Last updated: 2026-08-10*
