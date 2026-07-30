# Longitudinal multimodal multiple-instance learning of routine biopsy surveillance predicts rejection, chronic dysfunction and mortality after lung transplantation

**Authors:** [Author list TBD]

**Affiliations:** Helmholtz Munich, Munich, Germany; LMU Klinikum, Munich, Germany

---

## Abstract

Lung transplantation is the only definitive therapy for end-stage lung disease, yet it has the poorest long-term survival of any solid-organ transplant, driven by acute cellular rejection (ACR) and chronic lung allograft dysfunction (CLAD). Prognostic tools that exploit the data already generated during routine post-transplant surveillance remain lacking. We present a longitudinal multimodal multiple-instance learning (MIL) framework that integrates four surveillance data streams collected at each biopsy visit — transbronchial H&E histology, bronchoalveolar lavage cytology, thoracic CT and structured clinical variables — and explicitly models the ordered sequence of visits per patient. Benchmarking nine architectures across four clinical endpoints under strict 5-split × 4-fold nested cross-validation in ~350 recipients, longitudinal models achieved a concordance index of 0.771 ± 0.056 for death and 0.679 ± 0.064 for time-to-next-ACR, exceeding the best non-temporal fusion baselines by roughly 10 concordance points. A learned per-task biopsy-weighting network discovered opposite temporal windows for related endpoints — early visits for rejection risk, recent visits for current-grade classification. Attention attribution revealed a reproducible biology: preserved inflammatory alveolar histology marks survivors, whereas CT structural-deterioration patterns mark high mortality risk across all five splits.

---

## Introduction

Lung transplantation extends life for patients with end-stage pulmonary fibrosis, chronic obstructive pulmonary disease, cystic fibrosis and pulmonary arterial hypertension, but its long-term results trail every other solid-organ transplant. Median survival is approximately 6.5 years and five-year survival remains below 60%, a gap that has narrowed only marginally over three decades despite improvements in surgical technique and immunosuppression. Two complications dominate this trajectory. Acute cellular rejection (ACR), a T-cell-mediated inflammatory assault on the allograft, affects up to a third of recipients within the first year and is the strongest modifiable risk factor for later chronic injury. Chronic lung allograft dysfunction (CLAD) — encompassing bronchiolitis obliterans syndrome and restrictive allograft syndrome — is an irreversible fibro-obliterative process that afflicts roughly half of recipients by five years and is the leading cause of late death.

Because these complications evolve silently, transplant programmes perform intensive scheduled surveillance: transbronchial biopsy with histological grading, bronchoalveolar lavage (BAL) with cell differential, thoracic computed tomography (CT) and dense clinical and laboratory monitoring. This generates a rich, multimodal, longitudinal record for every recipient. Yet the prognostic models used in practice reduce this record to a handful of scalars — most often the trajectory of forced expiratory volume (FEV1), on which the clinical definition of CLAD itself rests. FEV1 decline is, by construction, a lagging indicator: it registers injury only once it is functionally manifest and frequently irreversible. Biomarker panels, donor–recipient matching scores and single-modality signatures have been proposed, but few have been validated across cohorts, and none integrates the full breadth of routinely acquired surveillance data.

Deep learning has reshaped prognostication in oncology through computational pathology, quantitative radiology and multimodal integration, but its translation to transplantation is nascent. The setting is unforgiving for data-hungry models: single-centre cohorts number in the low hundreds, outcomes are heavily right-censored, and the most informative modalities are acquired at different visits and cadences. Multiple-instance learning (MIL) is a natural fit for the imaging and pathology components, representing a slide or scan as a bag of patch instances supervised only at the bag level; attention-based pooling (ABMIL), set-transformer pooling and multimodal fusion have extended MIL to multi-slide, multi-modal and multi-task problems. However, essentially all multimodal MIL approaches applied to transplant surveillance treat each clinical encounter in isolation, discarding the temporal structure that is central to transplant immunology. The allograft's immunological set-point is not static: it is established in the first post-operative year and drifts thereafter under immunosuppression, so a biopsy at month six and an equivalent biopsy at year three carry different prognostic weight.

Here we develop and systematically benchmark a family of multimodal MIL architectures that spans three design philosophies: non-temporal fusion baselines (early, late and middle fusion), set-based cross-modal pooling that compresses each modality into a small number of learned seed vectors, and novel longitudinal models that process the ordered biopsy sequence of each patient with temporal attention and learned per-visit weighting. Every architecture is trained to solve four clinical tasks — binary ACR-grade classification, and Cox survival models for time-to-next-ACR, time-to-CLAD and time-to-death — and is evaluated under a rigorous 5-outer-split × 4-inner-fold nested cross-validation protocol that isolates model selection from test estimation.

We report three principal findings. First, explicit longitudinal modelling substantially improves survival prediction, lifting the death concordance index to 0.771 and the ACR-survival index to 0.679, each roughly ten concordance points above the strongest non-temporal baseline. Second, a learned biopsy-weighting network discovers, without supervision, opposite temporal windows for related endpoints — it upweights early visits for long-term rejection risk but recent visits for current-grade classification — a reversal that recapitulates transplant physiology. Third, attention attribution uncovers a reproducible and mechanistically coherent biology in which histology and CT contribute complementary information: preserved, acutely inflamed alveolar tissue marks survivors, whereas specific CT structural-deterioration patterns mark high mortality risk across every data split.

---

## Results

### Cohort, modalities and study design

The cohort comprised approximately 350 bilateral lung transplant recipients under longitudinal surveillance, each contributing one or more biopsy visits treated as time-stamped data objects and grouped by patient for the longitudinal analyses. Each visit contributed up to four modalities (Fig. 1a). Transbronchial H&E slides were tiled and encoded with the UNI pathology foundation model into 1,024-dimensional patch embeddings. BAL cytology was represented as 10-dimensional per-cell feature vectors. Thoracic CT was encoded into 1,024-dimensional patch embeddings. Structured clinical data — immunosuppressant regimen, laboratory values, spirometry, anthropometrics and demographics — were encoded as 106 tokens of a 491-dimensional one-hot vocabulary per visit. Modalities were frequently missing at the visit level, motivating architectures robust to partial observation.

Four endpoints were modelled jointly wherever the architecture permitted: binary ACR grade (A0 versus A1/A2, evaluated by balanced accuracy, BACC), and three right-censored Cox endpoints — time to the next ACR episode, time to CLAD onset and time to death — evaluated by Harrell's concordance index (C-index). All performance is reported on held-out test folds under a 5-outer-split × 4-inner-fold nested cross-validation scheme in which the test set is fixed per outer split and never contaminates hyperparameter selection (Fig. 1b; Methods).

### A benchmark of nine multimodal architectures

We evaluated nine architectures spanning three families (Fig. 1c). The non-temporal fusion baselines — early fusion (all patches pooled by a shared attention MIL), late fusion (per-modality decisions combined by learned weights) and middle fusion (per-modality summaries passed through a cross-modal transformer) — treat each visit independently. The set-based family compresses each modality's patch bag into K = 16 learned seed vectors by pooling-by-multihead-attention (PMA), optionally exchanges information across modalities with a set-attention block (SAB), and reads out per-task predictions with attention pooling; we tested a multi-task variant with SAB (SetMIL-MT), the same without SAB (SetMIL-MT, no SAB) and a single-task variant (SetMIL, no SAB). The longitudinal family extends the set-based backbone across the ordered visit sequence, adding either an ALiBi temporal-attention block (Longitudinal-MK-MT) or a learned biopsy-weighting network in place of ALiBi (Longitudinal-MK, single-task, and its multi-task counterpart).

Headline test performance is summarised in Table 1 (per-split values in Supplementary Table 1). No single architecture won every task, but the pattern was clear and biologically interpretable. Survival endpoints that depend on cumulative allograft trajectory — death and time-to-next-ACR — were won decisively by longitudinal models. Endpoints that are either intrinsically noisier or more local — CLAD onset and current-grade classification — were best served by set-based models. This division of labour, rather than a uniform winner, is itself the central result: performance tracks whether an endpoint is governed by a temporal trajectory or by the state of the allograft at a single visit.

**Table 1 | Test performance across nine architectures and four endpoints.**

| Architecture | ACR cls (BACC) | ACR surv (C-index) | CLAD surv (C-index) | Death surv (C-index) |
|---|---|---|---|---|
| Early fusion | 0.583 ± 0.057 | 0.575 ± 0.049 | 0.505 ± 0.065 | 0.645 ± 0.057 |
| Late fusion | 0.592 ± 0.029 | 0.585 ± 0.055 | 0.534 ± 0.083 | 0.638 ± 0.051 |
| Middle fusion | 0.559 ± 0.040 | 0.574 ± 0.082 | 0.516 ± 0.062 | 0.656 ± 0.057 |
| SetMIL-MT (with SAB) | 0.595 ± 0.027 | 0.489 ± 0.064 | **0.563 ± 0.080** | 0.664 ± 0.041 |
| SetMIL-MT (no SAB) | **0.623 ± 0.034** | 0.593 ± 0.059 | 0.536 ± 0.060 | 0.656 ± 0.035 |
| SetMIL (no SAB, single-task) | 0.611 ± 0.027 | 0.580 ± 0.031 | 0.488 ± 0.068 | 0.673 ± 0.026 |
| Longitudinal-MK-MT (ALiBi) | 0.545 ± 0.042 | 0.613 ± 0.073 | 0.496 ± 0.094 | 0.721 ± 0.056 |
| **Longitudinal-MK (learned weights)** | 0.550 ± 0.039 | **0.679 ± 0.064** | 0.489 ± 0.028 | **0.771 ± 0.056** |
| Longitudinal-MK-MT (learned weights) | 0.526 ± 0.052 | 0.630 ± 0.112 | 0.534 ± 0.100 | 0.770 ± 0.089 |

Values are mean ± s.d. across five outer-split test folds. Bold marks the best entry per column. BACC and C-index both have a chance level of 0.5.

### Longitudinal temporal modelling substantially improves survival prediction

For time-to-death — the endpoint of ultimate clinical importance — the Longitudinal-MK model with learned biopsy weighting reached a mean C-index of 0.771 ± 0.056, against 0.673 ± 0.026 for the strongest non-temporal architecture (single-task SetMIL) and 0.656 for middle fusion. The near-ten-point gain was consistent, not driven by an outlier split: individual splits scored 0.779, 0.670, 0.843, 0.772 and 0.793, with every split exceeding 0.67 and three exceeding 0.77. A death C-index approaching 0.77 is a strong result for an endpoint confounded by non-graft causes of mortality (infection, malignancy, cardiovascular disease) and is competitive with function-based prognostic scores while requiring no prospective functional follow-up.

The same longitudinal architecture won time-to-next-ACR at 0.679 ± 0.064, versus 0.593 ± 0.059 for the best non-temporal model (SetMIL-MT, no SAB) and 0.585 for late fusion — an improvement of roughly 8.6 to 9.4 concordance points. Here too the advantage was distributed across splits (0.573, 0.673, 0.748, 0.660, 0.741), the two weakest of which still matched or exceeded the non-temporal ceiling.

Two ablations localise the source of the gain. First, the advantage is specific to trajectory-governed endpoints: on CLAD and on current-grade classification the longitudinal models did not improve over, and often trailed, the set-based baselines (Table 1), ruling out a generic capacity effect. Second, and more pointedly, the temporal mechanism matters. Replacing the fixed monotonic recency prior of ALiBi with a learned biopsy-weighting network raised the ACR-survival C-index from 0.613 to 0.679 and left death essentially unchanged at the top of the range (0.771 versus 0.721), while ALiBi's rigid recency bias actively hurt tasks whose informative window is not the most recent visit. Learning the temporal weighting, rather than assuming it, is what converts longitudinal structure into predictive power.

### Learned biopsy-weighting networks discover opposite temporal windows for related endpoints

A recurring clinical question is which surveillance visits carry the most prognostic information. The Longitudinal-MK model answers it directly and interpretably. In place of a fixed recency prior, each task owns a small multilayer perceptron that maps a biopsy pair — the day of the current visit and of the preceding visit — to a scalar weight in (0,1) that scales that visit's contribution before attention pooling. Because the weight is a smooth learned function of visit timing, we can render the entire learned policy as a two-dimensional surface over (previous-visit day, current-visit day) and read off, per task, which temporal windows the model chose to trust (Fig. 2).

The surfaces revealed a striking, unsupervised reversal between two related ACR endpoints. For **time-to-next-ACR**, weight concentrated in the lower-left quadrant — high for visit pairs in which both current and previous biopsies fell within the first ~350 days after transplant, and sharply attenuated thereafter. The model discovered on its own that the immunological trajectory established in the first post-transplant year dominates long-term rejection risk, mirroring the clinical observation that early immune sensitisation sets the allograft's rejection tempo. For **current-grade ACR classification**, the surface inverted: weight concentrated in the upper-right quadrant, favouring recent, late (>350 day) visits. This too is correct behaviour — classifying the allograft's present rejection state requires its present features, not its early history.

The two survival endpoints that lack a privileged window behaved accordingly. **Death** produced a near-uniform surface with a modest suppression of the first ~50 days, precisely the peri-operative period whose events are dominated by surgical rather than immunological causes; the model effectively learned to discount the immediate post-operative window as prognostically confounded. **CLAD** likewise yielded a near-uniform surface, consistent with its diffuse, slowly accumulating pathology. That a single architecture, trained only on outcomes, recovers an early-window prior for rejection risk, a recent-window prior for rejection classification, a peri-operative exclusion for death, and a flat prior for CLAD — each matching independent clinical reasoning — is strong evidence that the learned temporal weighting reflects genuine biology rather than fitting artefact.

### Histology and CT encode complementary and reproducible mortality signatures

To ask what the models attend to, we performed population-level attribution on the best model for each task, tracing PMA seed vectors back through their patch attention to pre-computed morphological clusters, then contrasting the seed content of the highest- and lowest-risk patient tertiles and testing reproducibility across all five outer splits (Fig. 3). The death model, our strongest and most stable, yielded a signature reproducible in all five splits and consistent across two independent modalities.

In **H&E**, seeds attending to alveolar clusters with acute inflammation and haemorrhage (clusters 0–21), and to preserved alveolar parenchyma, were robustly enriched in the *low-risk* (survivor) group. The direction is initially counterintuitive but clinically transparent: patients who live to undergo repeated surveillance biopsies tend to experience acute, treatable inflammatory episodes on a substrate of intact gas-exchange architecture, rather than the fibrotic, obliterative remodelling of end-stage disease. Preserved alveolar structure is the microscopic correlate of functional reserve, and acute inflammation on that substrate is a treatable, survivable state.

In **CT**, a small set of clusters (C0–C2) was reproducibly enriched in the *high-risk* group across every split. These correspond to patterns of global structural deterioration — parenchymal destruction, air trapping and mosaic attenuation — that a single transbronchial biopsy, sampling one airway, cannot capture. The two modalities are therefore complementary by construction: histology reports the local microenvironment at one sampled site, while CT integrates architectural decline across the whole lung. Their concordant contribution — protective local inflammation, harmful global structural loss — is what underwrites the death model's accuracy, and it renders a coherent mechanistic narrative: early, treatable alveolar inflammation (the substrate of manageable ACR) is compatible with long survival, whereas diffuse structural deterioration (the radiological substrate of pre-clinical CLAD) is not.

The ACR-survival model echoed this picture. CT seeds were again enriched in the high-risk group (reproducible in four of five splits), and the learned biopsy weighting concentrated on early visits (above) — together indicating that early-established structural allograft quality predicts the long-term rejection trajectory. In contrast, attribution for CLAD and for ACR classification was inconsistent across splits, reflecting smaller effect sizes and greater outcome heterogeneity; we therefore do not advance modality-level biological claims for those two endpoints.

### Cross-modal interaction helps some tasks and hurts others

The set-based family isolates a clean architectural contrast: SetMIL-MT with and without the SAB cross-modal block are identical except that SAB lets seeds from different modalities exchange information before read-out. Its effect was task-dependent and, informatively, bidirectional. Removing SAB *improved* ACR classification (BACC 0.623 versus 0.595) but *degraded* CLAD survival (0.563 versus 0.536). Cross-modal mixing thus helps when an endpoint genuinely integrates signals across modalities (CLAD, a whole-organ process) and hurts when a single modality carries most of the signal and cross-talk merely injects noise (ACR grade, read primarily from histology). This is consistent with the set-based models' per-task modality gate, which learns to admit or suppress each modality independently per endpoint, and argues against one-size-fits-all fusion.

### Patient-representation geometry reflects risk and monitoring intensity

Two-dimensional UMAP projections of the 256-dimensional attention-pooled patient representations, computed per task from the final pooling layer, showed smooth, continuous organisation by predicted risk rather than clustering by individual patient, indicating that the models learned a coherent risk manifold rather than memorising recipients (Fig. 4). Colouring the same embeddings by number of biopsy visits and by days-since-transplant revealed that longitudinal coverage is a principal axis of variation, consistent with the temporal architectures preferentially structuring their representation space around how densely each patient was monitored.

---

## Discussion

We show that explicitly modelling the longitudinal, multimodal surveillance record improves outcome prediction after lung transplantation, and that the resulting models are interpretable in ways that recapitulate transplant physiology. The performance gains concentrate exactly where transplant biology predicts they should: on death and on time-to-next-ACR, endpoints governed by the cumulative trajectory of allograft health and immune activation, longitudinal models add roughly ten concordance points over non-temporal fusion; on current-grade classification, where the relevant information is the present visit, non-temporal set-based models are as good or better. Rather than a single dominant architecture, the benchmark reveals a principled correspondence between an endpoint's temporal structure and the modelling inductive bias that serves it.

The learned biopsy-weighting network is, in our view, the conceptual advance. Fixed temporal priors such as ALiBi assume that recency equals relevance — a reasonable default for language but a poor one for transplant surveillance, where the prognostically decisive window differs by endpoint. By letting each task learn its own smooth weighting over visit timing, the model both performs better and becomes legible: it recovered an early-window prior for long-term rejection risk, a recent-window prior for rejection classification, a peri-operative exclusion for death, and a flat prior for CLAD, each without temporal supervision and each matching independent clinical reasoning. The early/recent reversal between two ACR endpoints, obtained from a single architecture, is particularly compelling evidence that the temporal weights carry biological meaning.

The interpretability analyses converge on a mechanistic account of post-transplant mortality. That preserved, acutely inflamed alveolar histology marks survivors while specific CT deterioration patterns mark high risk — reproducibly across five independent splits — supports a model in which treatable acute inflammation on an intact parenchymal substrate is survivable, whereas diffuse structural loss is the harbinger of death. The complementarity of the two modalities has direct clinical resonance: a transbronchial biopsy samples one site, whereas surveillance CT integrates the whole organ, and our models exploit both. This provides computational support for the emerging view that quantitative CT morphology carries prognostic information beyond spirometry and deserves a larger role in surveillance.

Several limitations temper these conclusions. The cohort of ~350 recipients from a single programme is modest; the wide inter-split variance on the harder endpoints (for example ACR classification and CLAD) reflects this, and external validation across centres is essential before any clinical claim. CLAD in particular was modelled as a single endpoint despite its two divergent phenotypes — obliterative and restrictive — which have distinct natural histories and almost certainly distinct radiological and histological signatures; conflating them likely caps the achievable CLAD C-index at the modest values we observed, and its attribution did not reproduce across splits. BAL contributed little, plausibly because a 10-dimensional per-cell representation cannot compete with the high-dimensional spatial content of H&E and CT; richer cytological or single-cell features may rehabilitate it. Finally, all analyses are retrospective, and the seed-attribution biology, while reproducible, is associational rather than causal.

Future work follows directly. External, ideally prospective, validation is the priority. CLAD should be modelled by phenotype. The learned modality gates and biopsy weights invite a prospective study of modality-aware, timing-aware surveillance — using the model to identify which modality at which visit yields the most prognostic information per encounter, and thereby to rationalise an intensive and invasive monitoring schedule. Integrating single-cell transcriptomics from BAL at selected visits, and formal causal analysis of modality contributions, are natural extensions.

In sum, routine surveillance data already contain durable, multimodal prognostic signal; realising it requires only that models respect the temporal structure of the record. The framework presented here does so, and in doing so becomes not merely more accurate but more interpretable — surfacing when to look, where to look, and what to look for.

---

## Methods

### Cohort and data

Recipients of bilateral lung transplantation underwent routine post-transplant surveillance comprising transbronchial biopsy with H&E staining, BAL with cell differential, thoracic CT and structured clinical data capture. All data were generated in the course of clinical care under the governing ethics approval; consent provisions are as specified by the responsible ethics committee. Each biopsy visit is stored as a single PyTorch (`.pt`) object keyed by a stem identifier, and a master table links stems to patient identifiers, anchor (visit) dates, ACR grades and cross-validation assignments.

### Feature extraction

**H&E.** Whole-slide transbronchial biopsy images were tiled and each tile encoded with the UNI pathology foundation model into a 1,024-dimensional embedding; a slide is represented as a variable-length bag of patch embeddings. For interpretability only, patches were assigned to pre-computed morphological clusters; cluster labels never enter the models as input.

**BAL.** Each recovered cell was represented by a 10-dimensional feature vector; a visit is a bag of per-cell vectors.

**CT.** Lung parenchyma was segmented and tiled, and each patch encoded into a 1,024-dimensional embedding; a scan is a bag of patch embeddings. CT patches were grouped into clusters for interpretability.

**Clinical.** 106 structured variables per visit — immunosuppressant regimen, laboratory values, spirometry, anthropometrics and demographics — were encoded as 106 tokens over a 491-dimensional one-hot vocabulary.

Modality feature dimensions are fixed in a single registry (HE 1,024; BAL 10; CT 1,024; Clinical 491) that is the sole source of truth for feature keys, dimensions and per-modality presence flags.

### Labels and endpoints

ACR grade was binarised as A0 → 0 and A1/A2 → 1; grades outside this scheme were excluded from the classification loss but retained in the Cox risk set. Survival targets were derived relative to visit anchor dates: time-to-next-ACR with its event indicator, time-to-CLAD (`clad_time`/`clad_event`) and time-to-death (`death_time`/`death_event`). For CLAD, censored visits were assigned a time proxy from the death day or study end where necessary; for death, censored visits were carried to study end. ACR classification is scored by balanced accuracy; all survival endpoints by Harrell's C-index. Chance is 0.5 for every metric.

### Nested cross-validation

Evaluation used a 5-outer-split × 4-inner-fold nested design. Within each outer split the test set is fixed across all four inner folds and never participates in training or hyperparameter selection. Inner folds 1–3 perform hyperparameter sweeps only; fold 0 aggregates the best hyperparameters across all four folds ("global HP"), retrains on the combined train+validation set, and produces the single test estimate for that split. Skip logic keyed to the presence of the fold's sweep JSON (folds 1–3) or final-metrics JSON (fold 0) prevents redundant computation. All reported numbers are test-fold estimates; means ± s.d. are taken over the five outer splits, and per-split values are given in Supplementary Table 1.

### Phase 1: per-modality attention MIL encoders

Each modality is first trained independently with a gated-attention MIL encoder (`SingleModalMIL`). Given a bag of patch features X ∈ ℝ^{N×D}, the encoder projects each patch through a backbone of Linear(D→256) → Tanh → Dropout to h_i ∈ ℝ^{256}, then pools by gated attention:

```
a_i = softmax_i( w · ( tanh(V h_i) ⊙ σ(U h_i) ) ),    z = Σ_i a_i h_i
```

where V, U ∈ ℝ^{256×256}, w ∈ ℝ^{256} (no bias), ⊙ is elementwise product and z ∈ ℝ^{256} is the bag representation. For H&E an optional fixed 2-D sinusoidal positional encoding of tile coordinates may be added to h. A classification head (Dropout → Linear(256→1)) is trained with weighted hinge loss; a survival head (Linear(256→1)) is trained with the Cox–Breslow partial-likelihood loss. One encoder is trained per (modality, task, outer split, inner fold). Phase 1 weights are frozen and provide the per-patch backbone consumed by all Phase 2 fusion modules.

### Phase 2: non-temporal fusion baselines

**Early fusion** concatenates encoded patches from all present modalities into one bag (budget-balanced across modalities) and applies two independent gated-attention pools, one per task family.

**Late fusion** runs an independent per-modality attention MIL, produces per-modality task decisions and combines them by softmax-normalised learned scalar weights.

**Middle fusion** forms one 256-dimensional summary per modality, passes the set of summaries through a cross-modal transformer (multi-head self-attention with pre-norm FFN residual blocks), and pools the contextualised summaries with per-task attention heads.

Modal dropout randomly withholds modalities during training (always retaining at least one) to enforce robustness to missing data.

### Phase 2: set-based cross-modal pooling

Set-based models replace the gated backbone with a per-modality feed-forward projector (Linear(D→512) → Tanh → Dropout → Linear(512→256), followed by L2 normalisation, which places patch tokens on the unit sphere so that dot products are cosine similarities). Each modality's bag is then compressed by **pooling-by-multihead-attention (PMA)**: K = 16 learned seed vectors cross-attend to the patch tokens. We use a b-cos attention that sharpens seed specialisation,

```
attn(q, k) = ReLU(q · k)^b / Σ_n ReLU(q · k_n)^b,    b = 4,
```

with weight-normalised query and key projections (unit-norm weight rows) so that scores remain cosine-based, collapsing to standard softmax attention at b = 0. Each seed set receives an additive learned modality-identity embedding so downstream attention knows each token's source modality. Seeds from all present modalities are concatenated into M·K tokens.

In the SAB variants a **set-attention block** (multi-head self-attention with pre-norm FFN) mixes information across the concatenated seeds before read-out; the "no SAB" variants omit it. In the multi-task variants a **per-task modality gate** — a small MLP producing an independent sigmoid weight per modality per task, initialised near 1 — scales each modality's K seeds before the block, allowing a task to suppress an uninformative modality without softmax competition. Each task reads out with its own gated-attention pool over the seed tokens followed by a linear head (classification or hazard).

### Phase 2: longitudinal models

Longitudinal models (`LongitudinalMIL`) operate on the ordered sequence of a patient's T visits, sorted by days since first biopsy. For each visit and modality, patches are projected and PMA-compressed to K seeds exactly as above, a modality-identity embedding is added, and all seeds across all visits are concatenated into one temporally ordered token sequence tagged with per-token visit day.

**Temporal attention.** In the ALiBi variant, a **temporal SAB** applies multi-head self-attention with (i) a causal mask that forbids a token from attending to visits in its future and (ii) an ALiBi bias that penalises temporally distant pairs,

```
logit_{q,k} += − |m_h| · |t_q − t_k| / (Δt + 1),
```

where t are visit days, Δt the day range and m_h a per-head slope learned from an initial 0.1. In the learned-weight variant the temporal SAB is replaced by a plain SAB stack, and temporal structure is instead carried by the biopsy-weighting network below.

**Read-out anchoring.** Read-out is anchored to the clinically appropriate visit: for patient-level time-to-next-ACR the anchor is the last visit day; for the per-visit endpoints (death, CLAD, ACR classification) the anchor is each visit's own day, and one prediction is emitted per eligible visit (contributing multiple gap-time Cox terms per patient for death and CLAD).

**Learned biopsy weighting.** In place of a fixed recency prior, each task owns an MLP

```
w = σ( Linear(16→1) ∘ ReLU ∘ Linear(2→16) ( [ d_anchor , d_i ] ) )  ∈ (0,1),
```

that maps the anchor day and a visit's day to a scalar weight; this weight multiplies all K seeds from that visit before attention pooling, so the model freely learns which visits to trust for each task (a weight near 0 suppresses a visit entirely). When learned weighting is disabled, a per-task recency parameter γ instead adds an exponential distance penalty to the pooling logits. Multi-task variants share the backbone but keep separate weighting networks and heads per task; single-task variants train one network per endpoint.

### Losses and training

Binary classification uses weighted hinge loss (class weights balanced and capped at 20×) or, equivalently, a pos-weighted BCE. Survival uses the Cox–Breslow partial-likelihood loss: for a risk set ordered by time, the negative log partial likelihood is accumulated with a numerically stabilised suffix-sum of exponentiated hazards and normalised by the event count. Multi-task models sum the per-task losses with equal weight. Training used the Adam optimiser with gradient accumulation of 32 steps on NVIDIA A100 80 GB GPUs; a per-modality patch budget of up to 2,048 tokens bounds memory, and `expandable_segments` allocation is enabled for the largest multi-task longitudinal models. Hidden dimension is 256 and PMA uses K = 16 seeds throughout. All Phase 1 encoder weights are frozen during Phase 2. Hyperparameters (learning rate, dropout and variant-specific settings) were selected on inner folds 1–3 using, for multi-task models, the combined objective 0.5·BACC + 0.5·mean(C-index over ACR-survival, CLAD, death); fold 0 uses the aggregated best configuration.

### Interpretability

**Seed attribution.** PMA seed-to-patch attention identifies the dominant morphological cluster each seed represents. Population-level attribution contrasts, per seed, its prevalence in the top versus bottom risk tertile of predicted score, and reports cross-split mean ± s.d. as a reproducibility check; a modality-level claim is advanced only where direction is concordant in at least four of five splits.

**Biopsy-weight surface.** The learned weight function w(d_current, d_previous) is evaluated on a dense grid over [0, 2000] days with the invalid region (previous > current) masked, yielding the per-task temporal-weight heatmaps.

**Representation UMAP.** The 256-dimensional attention-pooled patient representation is embedded in 2-D by UMAP (cosine metric, 15 neighbours, min-dist 0.1) and coloured by predicted risk, visit count, anchor day and binary risk group.

### Evaluation and statistics

All metrics are computed exclusively on held-out test folds. Performance is summarised as mean ± s.d. over the five outer splits, with per-split values reported in full (Supplementary Table 1). All nine architectures were pre-specified and evaluated simultaneously; given the exploratory, architecture-comparison design, no multiple-comparison correction is applied.

---

## Figure legends

**Figure 1 | Study design.** (a) The four surveillance modalities per biopsy visit and their patch/token representations. (b) The 5-outer-split × 4-inner-fold nested cross-validation protocol; the fixed per-split test fold and the fold-0 global-hyperparameter retraining step. (c) Schematics of the three architecture families — non-temporal fusion, set-based cross-modal pooling and longitudinal temporal models.

**Figure 2 | Learned biopsy-weight surfaces (Longitudinal-MK, learned weights).** One panel per endpoint. Axes: previous-visit day (x) and current-visit day (y), 0–2000 days post-transplant; colour: learned weight ∈ (0,1) (red high, blue low); upper triangle masked. ACR-survival concentrates in the early lower-left quadrant; ACR-classification in the late upper-right quadrant; death is near-uniform with peri-operative suppression; CLAD is near-uniform.

**Figure 3 | Population-level seed attribution.** Per modality, mean high-minus-low-risk attention difference per seed across five splits (error bars = s.d.), with seeds annotated by dominant morphological cluster. Positive = high-risk-enriched. Death and ACR-survival panels highlight CT high-risk clusters (C0–C2) and H&E low-risk alveolar/inflammatory clusters.

**Figure 4 | Patient-representation UMAP.** 2-D UMAP of 256-dimensional attention-pooled representations for the best model per task, coloured by (a) predicted risk, (b) visit count, (c) anchor day and (d) binary risk group.

---

## Data availability

Patient-level clinical and imaging data cannot be shared publicly owing to data-protection regulation. Processed feature embeddings and cross-validation assignments may be made available on reasonable request under a data-use agreement subject to ethics-committee approval.

## Code availability

Code for all architectures, training, interpretability and evaluation, including cluster submission scripts and an environment specification, is available at [GitHub URL].

---

## References

[Full citations to be completed.] Key references:
- Ilse M, Tomczak J, Welling M. Attention-based deep multiple instance learning. *ICML* 2018.
- Lee J et al. Set Transformer: a framework for attention-based permutation-invariant neural networks. *ICML* 2019.
- Böhle M et al. B-cos networks: alignment is all we need for interpretability. *CVPR* 2022.
- Press O, Smith NA, Lewis M. Train short, test long: attention with linear biases enables input length extrapolation. *ICLR* 2022.
- Chen RJ et al. Towards a general-purpose foundation model for computational pathology (UNI). *Nat. Med.* 2024.
- The International Society for Heart and Lung Transplantation Registry — annual report.
- [Additional transplant-outcome, multimodal-MIL and longitudinal-EHR references to be added.]

---

## Supplementary information

### Supplementary Table 1 | Per-split test performance for all architectures.

**ACR classification (BACC)**

| Architecture | s0 | s1 | s2 | s3 | s4 | Mean | s.d. |
|---|---|---|---|---|---|---|---|
| Early fusion | 0.612 | 0.599 | 0.632 | 0.472 | 0.600 | 0.583 | 0.057 |
| Late fusion | 0.594 | 0.596 | 0.640 | 0.550 | 0.580 | 0.592 | 0.029 |
| Middle fusion | 0.522 | 0.520 | 0.616 | 0.540 | 0.599 | 0.559 | 0.040 |
| SetMIL-MT (SAB) | 0.597 | 0.546 | 0.597 | 0.605 | 0.630 | 0.595 | 0.027 |
| SetMIL-MT (no SAB) | 0.578 | 0.610 | 0.680 | 0.635 | 0.615 | **0.623** | 0.034 |
| SetMIL (no SAB, ST) | 0.644 | 0.564 | 0.626 | 0.601 | 0.619 | 0.611 | 0.027 |
| Long-MK-MT (ALiBi) | 0.596 | 0.494 | 0.499 | 0.559 | 0.577 | 0.545 | 0.042 |
| Long-MK (learned) | 0.546 | 0.565 | 0.510 | 0.512 | 0.615 | 0.550 | 0.039 |
| Long-MK-MT (learned) | 0.460 | 0.570 | 0.504 | 0.493 | 0.602 | 0.526 | 0.052 |

**Time-to-next-ACR (C-index)**

| Architecture | s0 | s1 | s2 | s3 | s4 | Mean | s.d. |
|---|---|---|---|---|---|---|---|
| Early fusion | 0.550 | 0.661 | 0.598 | 0.527 | 0.540 | 0.575 | 0.049 |
| Late fusion | 0.559 | 0.665 | 0.635 | 0.525 | 0.541 | 0.585 | 0.055 |
| Middle fusion | 0.526 | 0.613 | 0.708 | 0.466 | 0.557 | 0.574 | 0.082 |
| SetMIL-MT (SAB) | 0.585 | 0.539 | 0.454 | 0.467 | 0.403 | 0.489 | 0.064 |
| SetMIL-MT (no SAB) | 0.541 | 0.668 | 0.610 | 0.509 | 0.634 | 0.593 | 0.059 |
| SetMIL (no SAB, ST) | 0.584 | 0.596 | 0.585 | 0.523 | 0.614 | 0.580 | 0.031 |
| Long-MK-MT (ALiBi) | 0.634 | 0.561 | 0.530 | 0.600 | 0.741 | 0.613 | 0.073 |
| Long-MK (learned) | 0.573 | 0.673 | 0.748 | 0.660 | 0.741 | **0.679** | 0.064 |
| Long-MK-MT (learned) | 0.557 | 0.539 | 0.539 | 0.690 | 0.823 | 0.630 | 0.112 |

**Time-to-CLAD (C-index)**

| Architecture | s0 | s1 | s2 | s3 | s4 | Mean | s.d. |
|---|---|---|---|---|---|---|---|
| Early fusion | 0.432 | 0.622 | 0.495 | 0.460 | 0.516 | 0.505 | 0.065 |
| Late fusion | 0.372 | 0.583 | 0.603 | 0.553 | 0.561 | 0.534 | 0.083 |
| Middle fusion | 0.429 | 0.610 | 0.537 | 0.470 | 0.532 | 0.516 | 0.062 |
| SetMIL-MT (SAB) | 0.429 | 0.616 | 0.663 | 0.577 | 0.531 | **0.563** | 0.080 |
| SetMIL-MT (no SAB) | 0.476 | 0.619 | 0.528 | 0.469 | 0.589 | 0.536 | 0.060 |
| SetMIL (no SAB, ST) | 0.478 | 0.605 | 0.451 | 0.401 | 0.503 | 0.488 | 0.068 |
| Long-MK-MT (ALiBi) | 0.485 | 0.628 | 0.441 | 0.360 | 0.566 | 0.496 | 0.094 |
| Long-MK (learned) | 0.461 | 0.495 | 0.516 | 0.453 | 0.520 | 0.489 | 0.028 |
| Long-MK-MT (learned) | 0.721 | 0.456 | 0.523 | 0.439 | 0.533 | 0.534 | 0.100 |

**Time-to-death (C-index)**

| Architecture | s0 | s1 | s2 | s3 | s4 | Mean | s.d. |
|---|---|---|---|---|---|---|---|
| Early fusion | 0.550 | 0.640 | 0.679 | 0.635 | 0.721 | 0.645 | 0.057 |
| Late fusion | 0.551 | 0.624 | 0.693 | 0.638 | 0.685 | 0.638 | 0.051 |
| Middle fusion | 0.555 | 0.643 | 0.707 | 0.666 | 0.711 | 0.656 | 0.057 |
| SetMIL-MT (SAB) | 0.599 | 0.646 | 0.670 | 0.725 | 0.681 | 0.664 | 0.041 |
| SetMIL-MT (no SAB) | 0.593 | 0.650 | 0.689 | 0.662 | 0.688 | 0.656 | 0.035 |
| SetMIL (no SAB, ST) | 0.625 | 0.671 | 0.684 | 0.695 | 0.691 | 0.673 | 0.026 |
| Long-MK-MT (ALiBi) | 0.649 | 0.678 | 0.799 | 0.707 | 0.772 | 0.721 | 0.056 |
| Long-MK (learned) | 0.779 | 0.670 | 0.843 | 0.772 | 0.793 | **0.771** | 0.056 |
| Long-MK-MT (learned) | 0.706 | 0.628 | 0.855 | 0.815 | 0.848 | 0.770 | 0.089 |

### Supplementary Note 1 | Architecture hyperparameters

| Parameter | Value |
|---|---|
| Hidden dimension H | 256 |
| PMA seeds K | 16 per modality |
| PMA b-cos exponent b | 4 |
| Max patches per modality | 2,048 |
| Gradient accumulation | 32 steps |
| Optimiser | Adam |
| Phase 1 backbone | Linear(D→256) → Tanh → Dropout |
| Set-based projector | Linear(D→512) → Tanh → Dropout → Linear(512→256) → L2-norm |
| Biopsy-weight network | Linear(2→16) → ReLU → Linear(16→1) → Sigmoid |
| ALiBi slope init (per head) | 0.1 |
| Hardware | NVIDIA A100 80 GB |
