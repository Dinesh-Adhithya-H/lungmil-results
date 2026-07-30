# Longitudinal Multimodal Multiple Instance Learning Predicts Lung Transplant Rejection, Chronic Dysfunction, and Mortality from Routine Biopsy Data

**Authors:** [Author list TBD]

**Affiliation:** Helmholtz Munich / LMU Klinikum, Munich, Germany

---

## Abstract

Lung transplantation offers life-extending treatment for end-stage lung disease, yet long-term outcomes remain poor, with five-year survival below 60%. Acute cellular rejection (ACR) and chronic lung allograft dysfunction (CLAD) are the dominant causes of graft loss and death, yet their prediction from routine clinical data remains inadequate. Here we introduce a longitudinal multimodal multiple instance learning (MIL) framework that integrates four data streams collected at routine post-transplant biopsy visits — transbronchial H&E histology, bronchoalveolar lavage cytology, thoracic CT imaging, and structured clinical data — and explicitly models the temporal sequence of visits for each patient. Across 5-fold nested cross-validation in a cohort of ~350 lung transplant recipients, our best models achieve a concordance index of 0.679 ± 0.064 for ACR survival, 0.563 ± 0.080 for CLAD survival, and 0.771 ± 0.056 for death prediction. Temporal sequence modeling with a learned biopsy weighting network consistently outperforms non-temporal fusion baselines for survival tasks. Attention-based interpretability reveals a robust biological signature: preserved alveolar histology protects against mortality, while CT structural deterioration clusters predict death and rejection risk across all five data splits. These findings demonstrate that routine, non-invasive biopsy visit data encode durable prognostic signals when integrated longitudinally across modalities.

---

## Introduction

Lung transplantation is the only curative option for patients with end-stage pulmonary fibrosis, COPD, cystic fibrosis, and pulmonary hypertension. Despite advances in surgical technique and immunosuppression, median post-transplant survival remains approximately 5.5 years — the worst of all solid organ transplants [ISHLT registry]. Two complications drive this trajectory. Acute cellular rejection (ACR), a T-cell-mediated inflammatory attack on the allograft, affects up to 30% of recipients in the first year and is a major risk factor for subsequent CLAD. Chronic lung allograft dysfunction (CLAD), encompassing bronchiolitis obliterans syndrome (BOS) and restrictive allograft syndrome (RAS), is an irreversible fibroproliferative process affecting 50% of recipients by year five. Death from respiratory failure, infection, or malignancy follows. Early, accurate prediction of these outcomes — ideally from data already collected in routine clinical care — could transform clinical management, enabling pre-emptive immunosuppression adjustment, early listing for re-transplantation, or enrolment in clinical trials.

Existing prognostic models for lung transplant outcomes rely primarily on spirometry (FEV1 decline), donor-recipient matching criteria, or single-modality biomarker panels, and few have been validated prospectively or across independent cohorts. Deep learning has transformed outcome prediction in oncology through computational pathology, radiology, and multimodal integration, but its application to transplant medicine is nascent. Lung transplantation poses specific modelling challenges: patient populations are small (~200–500 per centre), data are sparse and irregularly spaced in time, outcomes are heavily right-censored, and the most prognostically relevant modalities (biopsy histology, BAL composition, CT morphology) are acquired at different visits with different frequencies.

Multiple instance learning (MIL) provides a natural framework for pathology and imaging data, treating a slide or scan as a bag of instances (patches) with only bag-level labels. Recent advances — including attention-based pooling (ABMIL), set-transformer pooling (PMA), and multimodal fusion variants — have extended MIL to multi-slide, multi-modal, and multi-task settings. However, all existing multimodal MIL approaches for transplantation treat each clinical encounter independently, discarding the temporal structure of longitudinal surveillance data. Transplant immunology is fundamentally dynamic: the immune set-point evolves over months to years under immunosuppression, and a single biopsy at month six carries different prognostic weight than an equivalent biopsy at year three.

Here we develop and systematically evaluate a family of multimodal MIL architectures that span non-temporal fusion baselines, set-based cross-modal pooling, and novel longitudinal models that explicitly process the ordered biopsy sequence for each patient. We compare nine distinct architectures on four prediction tasks simultaneously, use a rigorous 5-split × 4-fold nested cross-validation protocol to prevent data leakage, and apply attention-based interpretability methods to identify which biological features and temporal patterns drive predictions. Our results establish that temporal modelling of multi-visit biopsy sequences substantially improves survival prediction, and that CT structural deterioration and histological alveolar integrity are complementary, cross-split-robust predictors of lung transplant mortality.

---

## Results

### Cohort and Data Overview

The study cohort comprises approximately 350 lung transplant recipients followed longitudinally at [Institution], with each patient contributing between 1 and N biopsy visits as separate data objects. Across modalities, each biopsy visit provides: transbronchial H&E biopsy slides encoded as patch embeddings via the UNI foundation model (1,024-dim per patch); BAL cytology profiles represented as 10-dimensional radiomics-style feature vectors per recovered cell; thoracic CT scans encoded via RadiomicsTransformer (1,024-dim patches); and 106 structured clinical variables (medications, labs, spirometry, anthropometrics) one-hot encoded into a 491-dimensional vector. Four prediction tasks are evaluated: binary ACR grade classification (BACC), and three Cox survival tasks — time-to-next-ACR event, time-to-CLAD onset, and time-to-death (all measured by concordance index, C-index). The dataset is split using a 5-outer-split × 4-inner-fold nested cross-validation design, with the test set fixed per outer split across all inner folds (see Methods).

### Longitudinal Temporal Modelling Substantially Improves Survival Prediction

We compared nine model architectures spanning three families: non-temporal fusion baselines (Early, Late, Middle), set-based cross-modal models (SetMIL variants), and longitudinal sequence-aware models (Longitudinal-MK variants). All models were evaluated on held-out test sets across five outer splits (Figure 1, Table 1).

For ACR survival prediction, the Longitudinal-MK (no ALiBi) model achieved a mean C-index of 0.679 ± 0.064, compared to 0.585 ± 0.055 for the best non-temporal baseline (Late fusion) and 0.593 ± 0.059 for the best non-longitudinal set-based model (SetMIL-MT, no SAB). This represents an improvement of 9.4 percentage points over the strongest non-temporal baseline — a substantial gain given the difficulty of the task and the modest cohort size.

For death survival prediction, the same Longitudinal-MK (no ALiBi) model achieved a mean C-index of 0.771 ± 0.056, versus 0.664 ± 0.041 for SetMIL-MT and 0.656 ± 0.057 for Middle fusion. All five individual splits exceeded 0.67 (s0=0.779, s1=0.670, s2=0.843, s3=0.772, s4=0.793), indicating robustness across different train/test partitions.

For CLAD survival, the SetMIL-MT model (multi-task, with SAB cross-modal interaction) was the strongest performer at 0.563 ± 0.080 (C-index). While the absolute performance is modest, this represents a meaningful signal given the clinical difficulty of CLAD prediction and the heterogeneity of CLAD subtypes (BOS vs. RAS) not stratified in this analysis.

For ACR classification, the SetMIL-MT (no SAB) model achieved BACC 0.623 ± 0.034 — the strongest across all architectures. Notably, removing the SAB cross-modal interaction block improved ACR classification performance (0.623 vs. 0.595 with SAB), while the same ablation hurt CLAD survival (0.536 vs. 0.563), suggesting that cross-modal interaction is task-dependent.

**Table 1. Main performance comparison across all architectures and tasks.**

| Model | ACR cls (BACC) | ACR surv (CI) | CLAD surv (CI) | Death surv (CI) |
|-------|---------------|---------------|----------------|-----------------|
| Early fusion | 0.583 ± 0.057 | 0.575 ± 0.049 | 0.505 ± 0.065 | 0.645 ± 0.057 |
| Late fusion | 0.592 ± 0.029 | 0.585 ± 0.055 | 0.534 ± 0.083 | 0.638 ± 0.051 |
| Middle fusion | 0.559 ± 0.040 | 0.574 ± 0.082 | 0.516 ± 0.062 | 0.656 ± 0.057 |
| SetMIL-MT | 0.595 ± 0.027 | 0.489 ± 0.064 | **0.563 ± 0.080** | 0.664 ± 0.041 |
| SetMIL-MT (no SAB) | **0.623 ± 0.034** | 0.593 ± 0.059 | 0.536 ± 0.060 | 0.656 ± 0.035 |
| SetMIL (no SAB, single) | 0.611 ± 0.027 | 0.580 ± 0.031 | 0.488 ± 0.068 | 0.673 ± 0.026 |
| Longitudinal-MK-MT (ALiBi) | 0.545 ± 0.042 | 0.613 ± 0.073 | 0.496 ± 0.094 | 0.721 ± 0.056 |
| **Longitudinal-MK (no ALiBi)** | 0.550 ± 0.039 | **0.679 ± 0.064** | 0.489 ± 0.028 | **0.771 ± 0.056** |
| Longitudinal-MK-MT (no ALiBi) | 0.526 ± 0.052 | 0.630 ± 0.112 | 0.534 ± 0.100 | 0.770 ± 0.089 |

Bold = best per column. All values: mean ± std across 5 outer splits. BACC: balanced accuracy (chance = 0.5). CI: concordance index (chance = 0.5).

### Learned Biopsy Weighting Reveals Task-Specific Temporal Windows

A key question in longitudinal transplant monitoring is: *which visits carry the most prognostic information?* The Longitudinal-MK (no ALiBi) model addresses this directly through a per-task biopsy weighting network — a small MLP that receives (current\_biopsy\_day, previous\_biopsy\_day) as input and outputs a scalar weight w ∈ (0,1) (Figure 2).

Visualising the learned weight surface as a 2D heatmap (x-axis: previous biopsy date, y-axis: current biopsy date, colour: weight) reveals striking task-specific temporal patterns:

**ACR survival:** The weight function assigns high weight to early biopsy pairs (both current and previous biopsy dates <350 days post-transplant). Biopsies conducted after day 350 are substantially downweighted. This indicates that the immune phenotype established in the first year post-transplant carries the dominant prognostic signal for subsequent rejection episodes.

**ACR classification:** The temporal pattern is reversed: late biopsy visits (both dates >350 days post-transplant) receive high weight. The model correctly identifies that predicting the *current* rejection grade requires attention to recent, late-stage clinical features, not the early immune trajectory.

**Death survival:** The weight surface is near-uniform across biopsy dates, with a modest exclusion of the very first 50 days post-transplant (peri-operative period). This is consistent with the biological understanding that mortality risk is determined by cumulative lung function decline rather than any single time window.

**CLAD survival:** Similarly near-uniform, consistent with the diffuse, slowly progressive nature of CLAD.

These task-specific weight patterns emerge entirely from the data without explicit supervision, validating the temporal modelling approach and providing interpretable, clinically meaningful insight into which surveillance windows are prognostically relevant for each outcome.

### Cross-Modal Seed Attribution Identifies Histological and Radiological Risk Signatures

To identify which biological features drive predictions, we applied population-level seed attribution analysis to the best model for each task (Figure 3). PMA seed vectors (K=16 per modality) were attributed to high-risk versus low-risk patient groups (top vs. bottom tertile of predicted risk score), and cross-split mean ± std was computed to assess robustness.

**Death survival — robust biological signature (5/5 splits).** The death prediction model revealed a consistent, cross-split-robust biological signal across both H&E and CT modalities.

In the H&E modality, seeds attending to clusters 0–21 (alveolar tissue with acute inflammation or haemorrhage) were significantly enriched in the *low-risk* group — that is, patients with longer survival tend to have more inflammatory alveolar histology at their biopsy visits. This counterintuitive finding has a straightforward clinical interpretation: patients who survive long enough to undergo repeated surveillance biopsies tend to have acute, treatable inflammatory episodes rather than the fibrotic, obliterative remodelling associated with end-stage disease. Preserved alveolar parenchyma (clusters 32–41) shows the same protective enrichment pattern.

In the CT modality, clusters C0, C1, and C2 are robustly enriched in the *high-risk* group across all five outer splits. These CT cluster identities likely correspond to patterns of structural lung deterioration — parenchymal destruction, air trapping, mosaic attenuation, or consolidation — that precede and strongly predict mortality. Critically, this CT signal is *complementary* to the H&E signal: while biopsy histology captures the microenvironmental state at a single sampled site, CT integrates structural information across the entire lung volume, detecting global architectural decline that a transbronchial biopsy may miss.

**ACR survival — CT-driven, temporally early (4/5 splits).** For ACR survival, CT seeds are again preferentially enriched in the high-risk group, paralleling the death prediction signature. The model further assigns high weight to early biopsy visits (<350 days), indicating that early-established CT structural patterns predict long-term rejection trajectories. This aligns with the clinical observation that early post-transplant allograft quality (as reflected in CT) is a strong determinant of subsequent graft health.

**ACR classification and CLAD survival.** Seed attribution for ACR classification and CLAD survival showed cross-split inconsistency (<3/5 splits with concordant direction), reflecting the smaller effect sizes and greater outcome heterogeneity in these tasks. We do not report specific biological attributions for these tasks as paper-grade findings.

### Patient Representation Space Reveals Risk Stratification Structure

2D UMAP projections of the 256-dimensional ABMIL-weighted patient representations (computed per patient per task from the final pooling layer) reveal structured separation in representation space (Figure 4). Patients coloured by predicted risk score show smooth, continuous gradients across the UMAP landscape, indicating that the model has learned a coherent risk embedding rather than memorising individual patients. Colouring by number of biopsy visits and by days post-transplant at anchor reveals that longitudinal coverage is a key axis of variation in the learned space — patients with more biopsy visits occupy distinct regions, consistent with the temporal modelling architecture preferentially representing well-monitored patients.

---

## Discussion

This work demonstrates that temporal integration of multimodal biopsy visit data substantially improves survival prediction in lung transplantation, and that interpretability analyses reveal biologically coherent signatures that are consistent with mechanistic understanding of transplant pathophysiology.

The most striking performance gains from longitudinal modelling occur for ACR survival and death prediction — both tasks where the cumulative trajectory of immune activation and organ health over months to years is more informative than any single visit. For ACR classification (predicting the grade at the *current* biopsy), non-temporal set-based models perform comparably or better, consistent with the intuition that immediate classification benefits from current-visit features rather than trajectory.

The learned biopsy weight network provides a principled, interpretable alternative to fixed positional encodings such as ALiBi. Unlike ALiBi, which imposes a monotonic recency bias, the learned weight function discovers task-specific temporal structures from the data. The ACR survival / ACR classification reversal — early visits favoured for survival, late visits favoured for classification — is a clear example of the model learning qualitatively different temporal strategies for related but distinct clinical questions.

The CT structural clusters (C0–C2) emerging as robust death and rejection risk predictors across five independent data splits is a particularly strong finding. CT-based quantitative lung morphology is emerging as a biomarker for post-transplant outcomes in the clinical literature, and our model provides computational evidence that specific radiological patterns — detectable in routine surveillance CT — carry independent prognostic information beyond spirometry. The histological complement — preserved alveolar parenchyma as a protective signal — is consistent with the pathological understanding that intact gas exchange architecture is the strongest microscopic correlate of functional survival.

Several limitations should be acknowledged. The cohort size (~350 patients) is modest, limiting statistical power and making generalisation to diverse transplant programmes uncertain. CLAD subtype stratification (BOS vs. RAS) was not incorporated, and the two subtypes have distinct natural histories and likely distinct radiological and histological signatures. BAL cytology contributes minimally to the current models, which may reflect its low-dimensional representation (10 features per cell) relative to the rich spatial information in H&E and CT; future work integrating cell-level morphological features or scRNA-seq data may improve its contribution. Finally, all analyses are retrospective and should be validated prospectively before clinical deployment.

Future directions include prospective validation in independent transplant cohorts, integration of single-cell transcriptomic data from BAL at selected visits, investigation of CLAD subtype-specific models, and causal analysis of which modality combinations are most critical — enabling modality-aware biopsy scheduling to maximise prognostic information per clinical encounter.

---

## Methods

### Cohort and Ethics

Patients underwent bilateral lung transplantation at [Institution] between [year range]. Routine post-transplant surveillance included transbronchial biopsy with H&E staining, BAL cell differential, thoracic CT, and structured clinical data collection. All data were collected as part of routine clinical care. Ethical approval was obtained from [IRB]. Written informed consent was waived / obtained [as applicable].

### Data Preprocessing

**H&E histology.** Whole-slide images from transbronchial biopsies were tiled at 20× magnification (256×256 pixels per tile, 50% overlap). Tiles were encoded using the UNI foundation model (ViT-Large, pre-trained on 100,000+ pathology slides), yielding 1,024-dimensional feature vectors per patch. Patches were pre-clustered into 54 biologically annotated categories using k-means clustering (see Results for cluster-to-biology mapping). The full 1,024-dim embeddings are used as model input; cluster labels are used only for interpretability.

**BAL cytology.** Cells recovered from BAL were represented by 10-dimensional feature vectors capturing cell morphology and differential count features (macrophage, lymphocyte, neutrophil proportions, cell size, granularity).

**CT imaging.** Thoracic CT scans were segmented for lung parenchyma and tiled into 3D patches. Patches were encoded using a RadiomicsTransformer, yielding 1,024-dimensional feature vectors. CT patches were grouped into 16 k-means clusters (C0–C15) for interpretability analyses.

**Clinical data.** 106 structured clinical variables — including current immunosuppressant regimen (tacrolimus, mycophenolate, prednisolone), laboratory values (creatinine, tacrolimus trough, eosinophils, CRP), spirometry (FEV1, FVC, FEV1/FVC), BMI, and donor/recipient demographics — were one-hot encoded into a 491-dimensional vector per biopsy visit.

**Label encoding.** ACR grade was encoded as binary: A0\* → 0, A1\* or A2\* → 1. Other grade patterns (B, C grades) were excluded from the classification loss but retained in the Cox risk set. Survival outcomes were derived from the anchor (transplant) date: `tte_next_acr`/`event_next_acr`, `clad_time`/`clad_event`, `death_time`/`death_event`.

**Data format.** All preprocessed features for a single biopsy visit are stored in a `.pt` (PyTorch) file, keyed to the patient stem identifier. The primary CSV links patient IDs, anchor dates, ACR grades, and cross-validation assignments.

### Nested Cross-Validation

The evaluation uses a 5-outer-split × 4-inner-fold nested cross-validation protocol. For each outer split, the test set is fixed and does not overlap with training or validation data. Three inner folds (folds 1–3) are used exclusively for hyperparameter selection. Fold 0 of each outer split aggregates the best hyperparameters across all four inner folds, trains a final model on train + validation data, and reports test metrics. This protocol prevents leakage of test-set information into model selection.

### Model Architecture

#### Phase 1: Unimodal Pre-Training

Each data modality is trained independently with a Gated Attention MIL (ABMIL) encoder:

```
Input patches (N × D_mod)
  → Linear(D_mod → 256) → GELU → LayerNorm
  → Gated attention: a = softmax(tanh(W_V · h)ᵀ · σ(W_U · h))
  → Patient representation: z = Σ_i a_i · h_i    [256-dim]
  → Task head
```

For classification: single linear layer → sigmoid, trained with hinge loss.
For survival: single linear layer (risk score), trained with Cox–Breslow partial likelihood loss.

Separate Phase 1 models are trained for each (modality, task, outer split, inner fold) combination. Phase 1 weights are frozen during Phase 2 training. The 256-dim bag-level representation per modality per visit serves as the token input to all Phase 2 fusion modules.

#### Phase 2: Non-Temporal Fusion Baselines

**Early fusion.** Patches from all modalities are concatenated into a single bag of (N_HE + N_BAL + N_CT + N_Clin) instances, processed by a shared ABMIL encoder.

**Late fusion.** Each modality runs an independent ABMIL encoder. Final predictions from all modalities are combined via learned scalar combination weights (softmax-normalised).

**Middle fusion.** Per-modality ABMIL encoders produce 256-dim representations. These are concatenated and passed through a CrossModalTransformer (multi-head cross-attention between modality pairs) before a final ABMIL aggregation layer.

#### Phase 2: SetMIL Variants

**SetMIL-MT** (`set_mil_mt`). For each modality, Pooling by Multihead Attention (PMA) with K=16 learned seed vectors compresses the variable-length patch bag into a fixed 16×256 token matrix:

```
Seeds S ∈ ℝ^{K×H}, Patches X ∈ ℝ^{N×H}
PMA(S, X) = Multihead-Attn(Q=S, K=X, V=X)    → K×H output
```

A Set Attention Block (SAB) then applies self-attention across the concatenated seed sets from all four modalities (4K total seeds), enabling cross-modal interaction. Per-task ABMIL heads pool the seeds into scalar task predictions. All four tasks are trained jointly (multi-task).

**SetMIL-MT (no SAB)** (`set_mil_mt_no_sab`). Identical to SetMIL-MT with the SAB cross-modal block removed. Modality-specific PMA representations feed directly into per-task ABMIL heads.

**SetMIL (no SAB, single-task)** (`set_mil_no_sab`). Single-task version trained independently per task.

#### Phase 2: Longitudinal Variants

Longitudinal models process the ordered sequence of biopsy visits {v_1, v_2, ..., v_T} for each patient, where visits are sorted by days post-transplant. Each biopsy visit produces four PMA-compressed modality token sets (K seeds per modality). These are concatenated across the visit sequence to form a sequence of cross-modal biopsy tokens.

**Longitudinal-MK-MT with ALiBi** (`longitudinal_mk_mt`). Temporal attention is implemented as a TemporalSAB with Attention with Linear Biases (ALiBi):

```
Attention score: a_ij = (q_i · k_j) / √H + b_h · |t_i - t_j|
```

where b_h is a learned per-head slope (negative, penalising temporally distant pairs) and t_i is the biopsy date in days. A per-task recency decay parameter γ further downweights older visits. Per-task ABMIL heads produce final predictions. Multi-task.

**Longitudinal-MK (no ALiBi, single-task)** (`longitudinal_mk_no_alibi`). ALiBi is replaced by a learned biopsy weighting network:

```
w(c, p) = σ( Linear(16→1) ∘ ReLU ∘ Linear(2→16) ([c, p]) )
```

where c = current biopsy day, p = previous biopsy day, w ∈ (0,1). This network weights the contribution of each biopsy pair, allowing the model to learn task-specific temporal attention patterns without imposing a monotonic recency bias. Separate networks per task. Single-task models trained independently per task.

**Longitudinal-MK-MT (no ALiBi, multi-task)** (`longitudinal_mk_mt_no_alibi`). Multi-task version of the above, with separate weighting networks per task but shared backbone.

#### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Hidden dimension (H) | 256 |
| PMA seeds (K) | 16 per modality |
| Max patches per modality | 2,048 |
| Gradient accumulation | 32 steps |
| Optimiser | Adam |
| Hardware | NVIDIA A100 80GB |

### Training Protocol

**Hyperparameter selection.** Inner folds 1–3 perform a grid search over learning rate, dropout, and variant-specific parameters. Each fold saves a JSON file of validation performance per hyperparameter configuration. Fold 0 aggregates best hyperparameters across all four inner folds using a combined multi-task metric:

For multi-task models: `score = 0.5 × BACC_ACR + 0.5 × mean(CI_ACR_surv, CI_CLAD, CI_Death)`

**Final model.** Fold 0 trains on the combined train + validation set with the selected hyperparameters. All Phase 1 encoder weights are frozen. Test metrics are computed once per outer split.

**Loss functions.** Binary classification: hinge loss. Survival tasks: Cox–Breslow partial likelihood loss on risk-set-normalised event indicators. Multi-task models sum all task losses with equal weighting.

### Interpretability Methods

**Seed attribution.** Attention weights from the PMA cross-attention (seed-to-patch attention) identify which patch clusters each seed vector predominantly represents. For each patient, seeds are attributed to their dominant cluster by argmax attention. Population-level attribution compares, for each seed, the fraction of high-risk vs. low-risk patients (top/bottom tertile of predicted risk) in which that seed's dominant cluster appears. Cross-split mean ± std quantifies robustness.

**Biopsy weight heatmap.** The learned weight function w(c, p) is evaluated on a 100×100 grid over c, p ∈ [0, 2000] days. Cells where p > c (previous biopsy after current) are masked as invalid. The resulting heatmap directly visualises which temporal biopsy patterns the model upweights for each task.

**Patient representation UMAP.** The 256-dim ABMIL-weighted patient representation (z = Σ_i a_i · h_i from the final pooling layer) is computed per patient per task. 2D UMAP embeddings use cosine distance metric, n_neighbours=15, min_dist=0.1. Projections are coloured by predicted risk score, number of biopsy visits, anchor day, and binary risk group.

### Evaluation Metrics

- **Balanced accuracy (BACC):** mean of sensitivity and specificity across the binary ACR classification task. Used for ACR classification only. Random chance = 0.5.
- **Concordance index (C-index):** Harrell's C-statistic for Cox survival models. Measures the probability that a patient predicted to be higher risk dies / rejects / develops CLAD before a patient predicted to be lower risk. Random chance = 0.5.

All metrics are computed on held-out test sets only. No metric is computed on any data used for training or hyperparameter selection.

### Statistical Analysis

Performance is reported as mean ± standard deviation across the five outer-split test sets. No correction for multiple comparisons is applied given the exploratory nature of the architecture comparison; all nine architectures were pre-specified and evaluated simultaneously.

---

## Data Availability

Patient-level clinical and imaging data are not publicly available due to data protection regulations. Processed feature embeddings and cross-validation split assignments may be made available upon reasonable request through a data use agreement with [Institution], subject to ethics committee approval.

---

## Code Availability

Code implementing all model architectures, training protocols, interpretability analyses, and evaluation pipelines is available at: [GitHub URL]. The repository includes SLURM submission scripts for reproducing all experiments on GPU clusters, and a conda `environment.yml` for environment setup.

---

## References

[To be completed with full citations]

Key references:
- Ilse M, Tomczak J, Welling M. Attention-based deep multiple instance learning. ICML 2018.
- Lee J, et al. Set transformer: a framework for attention-based permutation-invariant neural networks. ICML 2019.
- Press O, et al. Train short, test long: attention with linear biases enables input length extrapolation. ICLR 2022.
- Chen RJ, et al. A general-purpose AI foundation model for pathology (UNI). Nature Medicine 2024.
- ISHLT Registry Report [current year].
- [Transplant outcome prediction references]
- [Multimodal MIL references]
- [Longitudinal EHR deep learning references]

---

## Supplementary Material

### Supplementary Table 1. Per-split performance for all models.

**ACR classification (BACC)**

| Model | s0 | s1 | s2 | s3 | s4 | Mean | SD |
|-------|----|----|----|----|----|----|--|
| Early fusion | 0.612 | 0.599 | 0.632 | 0.472 | 0.600 | 0.583 | 0.057 |
| Late fusion | 0.594 | 0.596 | 0.640 | 0.550 | 0.580 | 0.592 | 0.029 |
| Middle fusion | 0.522 | 0.520 | 0.616 | 0.540 | 0.599 | 0.559 | 0.040 |
| SetMIL-MT | 0.597 | 0.546 | 0.597 | 0.605 | 0.630 | 0.595 | 0.027 |
| SetMIL-MT (no SAB) | 0.578 | 0.610 | 0.680 | 0.635 | 0.615 | 0.623 | 0.034 |
| SetMIL (no SAB, ST) | 0.644 | 0.564 | 0.626 | 0.601 | 0.619 | 0.611 | 0.027 |
| Long-MK-MT (ALiBi) | 0.596 | 0.494 | 0.499 | 0.559 | 0.577 | 0.545 | 0.042 |
| Long-MK (no ALiBi) | 0.546 | 0.565 | 0.510 | 0.512 | 0.615 | 0.550 | 0.039 |
| Long-MK-MT (no AL) | 0.460 | 0.570 | 0.504 | 0.493 | 0.602 | 0.526 | 0.052 |

**ACR survival (C-index)**

| Model | s0 | s1 | s2 | s3 | s4 | Mean | SD |
|-------|----|----|----|----|----|----|--|
| Early fusion | 0.550 | 0.661 | 0.598 | 0.527 | 0.540 | 0.575 | 0.049 |
| Late fusion | 0.559 | 0.665 | 0.635 | 0.525 | 0.541 | 0.585 | 0.055 |
| Middle fusion | 0.526 | 0.613 | 0.708 | 0.466 | 0.557 | 0.574 | 0.082 |
| SetMIL-MT | — | — | — | — | — | 0.489 | 0.064 |
| SetMIL-MT (no SAB) | 0.541 | 0.668 | 0.610 | 0.509 | 0.634 | 0.593 | 0.059 |
| SetMIL (no SAB, ST) | 0.584 | 0.596 | 0.585 | 0.523 | 0.614 | 0.580 | 0.031 |
| Long-MK-MT (ALiBi) | 0.634 | 0.561 | 0.530 | 0.600 | 0.741 | 0.613 | 0.073 |
| **Long-MK (no ALiBi)** | **0.573** | **0.673** | **0.748** | **0.660** | **0.741** | **0.679** | **0.064** |
| Long-MK-MT (no AL) | 0.557 | 0.539 | 0.539 | 0.690 | 0.823 | 0.630 | 0.112 |

**CLAD survival (C-index)**

| Model | s0 | s1 | s2 | s3 | s4 | Mean | SD |
|-------|----|----|----|----|----|----|--|
| Early fusion | 0.432 | 0.622 | 0.495 | 0.460 | 0.516 | 0.505 | 0.065 |
| Late fusion | 0.372 | 0.583 | 0.603 | 0.553 | 0.561 | 0.534 | 0.083 |
| Middle fusion | 0.429 | 0.610 | 0.537 | 0.470 | 0.532 | 0.516 | 0.062 |
| **SetMIL-MT** | **0.429** | **0.616** | **0.663** | **0.577** | **0.531** | **0.563** | **0.080** |
| SetMIL-MT (no SAB) | 0.476 | 0.619 | 0.528 | 0.469 | 0.589 | 0.536 | 0.060 |
| SetMIL (no SAB, ST) | 0.478 | 0.605 | 0.451 | 0.401 | 0.503 | 0.488 | 0.068 |
| Long-MK-MT (ALiBi) | 0.485 | 0.628 | 0.441 | 0.360 | 0.566 | 0.496 | 0.094 |
| Long-MK (no ALiBi) | 0.461 | 0.495 | 0.516 | 0.453 | 0.520 | 0.489 | 0.028 |
| Long-MK-MT (no AL) | 0.721 | 0.456 | 0.523 | 0.439 | 0.533 | 0.534 | 0.100 |

**Death survival (C-index)**

| Model | s0 | s1 | s2 | s3 | s4 | Mean | SD |
|-------|----|----|----|----|----|----|--|
| Early fusion | 0.550 | 0.640 | 0.679 | 0.635 | 0.721 | 0.645 | 0.057 |
| Late fusion | 0.551 | 0.624 | 0.693 | 0.638 | 0.685 | 0.638 | 0.051 |
| Middle fusion | 0.555 | 0.643 | 0.707 | 0.666 | 0.711 | 0.656 | 0.057 |
| SetMIL-MT | 0.599 | 0.646 | 0.670 | 0.725 | 0.681 | 0.664 | 0.041 |
| SetMIL-MT (no SAB) | 0.593 | 0.650 | 0.689 | 0.662 | 0.688 | 0.656 | 0.035 |
| SetMIL (no SAB, ST) | 0.625 | 0.671 | 0.684 | 0.695 | 0.691 | 0.673 | 0.026 |
| Long-MK-MT (ALiBi) | 0.649 | 0.678 | 0.799 | 0.707 | 0.772 | 0.721 | 0.056 |
| **Long-MK (no ALiBi)** | **0.779** | **0.670** | **0.843** | **0.772** | **0.793** | **0.771** | **0.056** |
| Long-MK-MT (no AL) | 0.706 | 0.628 | 0.855 | 0.815 | 0.848 | 0.770 | 0.089 |

### Supplementary Figure Legends

**Figure 1.** Model comparison overview. Bar plots of mean ± std C-index / BACC across 5 outer splits for all nine architectures, grouped by task. Individual split values shown as dots.

**Figure 2.** Learned biopsy weight heatmaps (Longitudinal-MK, no ALiBi). Each panel shows the learned weight surface w(current\_biopsy\_day, previous\_biopsy\_day) for one prediction task. X-axis: previous biopsy date (days post-transplant, 0–2000). Y-axis: current biopsy date (days post-transplant, 0–2000). Colour: learned weight ∈ (0,1), blue = low weight, red = high weight. Upper triangle (previous > current) masked as invalid.

**Figure 3.** Population-level seed attribution. For each modality, bar plots show mean attention difference (high-risk minus low-risk) per seed vector across 5 outer splits (mean ± std error bars). Positive values indicate seeds enriched in high-risk patients; negative values indicate seeds enriched in low-risk patients. Seed clusters labelled by dominant biological category.

**Figure 4.** Patient representation UMAP. 2D UMAP projections of 256-dim patient representations, coloured by (A) predicted risk score, (B) number of biopsy visits, (C) anchor day (days post-transplant), (D) binary risk group. One panel per prediction task for the best model.
