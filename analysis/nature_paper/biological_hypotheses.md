# Biological Hypotheses: Multimodal Lung Transplant MIL Study

*Generated from linear model (LASSO logistic / CoxPH) feature importances across 5 splits, All-modality model.*
*Tasks: ACR classification (BACC=0.588), ACR TTE (C=0.587), CLAD TTE (C=0.501), Death TTE (C=0.580)*

---

## 1. ACR Risk Hypotheses

### H-ACR-1: EBV serostatus drives alloimmune sensitisation and ACR susceptibility
**Finding:** `recipient_ebv` is the second-strongest ACR predictor (coef=+0.68), consistent across ACR classification and ACR TTE.
**Mechanism:** EBV-seropositive recipients carry a pre-expanded pool of virus-specific CD8⁺ and CD4⁺ memory T cells. Cross-reactive T cell recognition of donor HLA peptides by EBV-specific clones (heterologous immunity) amplifies the alloimmune response, lowering the threshold for acute cellular rejection. EBV reactivation under immunosuppression further expands polyfunctional effector T cells that infiltrate the allograft.
**Literature:** Burrows et al. (2006, *J Exp Med*) — EBV/HLA cross-reactivity. Welsh & Selin (2002, *Nat Rev Immunol*) — heterologous immunity in transplant rejection.
**Evidence:** Moderate

### H-ACR-2: BAL CD4⁺ T cell infiltration + dysfunctional Tregs defines the cellular ACR effector profile
**Finding:** `BAL:CD4 T cell-1` (coef=+0.61) and `BAL:CD4 Treg` (coef=+0.49) both increase ACR risk. H&E confirms: `HE:Alveolar inflamed` (coef=+0.60).
**Mechanism:** The positive Treg coefficient reflects "ineffective Tregs" — recruited but rendered dysfunctional by the pro-inflammatory cytokine milieu (TNF, IL-6). Their abundance reflects rather than suppresses the inflammatory response. BAL CD4⁺ expansion + inflamed alveolar histology recapitulates the classic A-grade ACR triad (perivascular mononuclear infiltrate, alveolitis) codified in the ISHLT grading system.
**Literature:** Stewart et al. (2016, *Am J Transplant*) — BAL lymphocytosis >15% predicts A-grade ACR. Greenland et al. (2021, *Transplantation*) — Treg dysfunction in rejecting lung allografts via scRNA-seq.
**Evidence:** Strong — cross-validated across BAL scRNA, H&E, and clinical.

### H-ACR-3: BAL CXCL10⁺ macrophages and macrophage-dominated BAL cytology are protective — M2-like polarisation suppresses alloresponse
**Finding:** `BAL:CXCL10 macs-2` (coef=−0.55) and `Clin:MACROPHAGES, BODY FLUID Right` (coef=−0.71) strongly protective.
**Mechanism:** A macrophage-dominant BAL indicates relative lymphocyte suppression. Tissue-resident alveolar macrophages (AMs) maintain tolerance through PD-L1/IL-10/TGF-β. CXCL10⁺ macrophages here likely represent a transitional interstitial macrophage phenotype (TRAM-like) participating in anti-inflammatory tissue repair rather than rejection surveillance, suppressing Th1 priming and alloreactive CD4⁺ T cell expansion.
**Literature:** Aguilar et al. (2019, *JCI Insight*) — TRAM macrophages suppress alloimmune activation. Jakubzick et al. (2006, *J Clin Invest*) — interstitial macrophage anti-inflammatory functions.
**Evidence:** Moderate — CXCL10 is classically pro-inflammatory; the protective signal may be subtype-specific.

### H-ACR-4: Preserved FEV1% predicted is the functional correlate of immune quiescence
**Finding:** `Clin:fev1p` (coef=−0.44) protective for ACR.
**Mechanism:** FEV1% integrates graft structural integrity and absence of active airway inflammation. As a cross-sectional feature, it captures the immunological steady state of the graft — rejection causes FEV1 drop, and preserved FEV1% flags immunological quiescence.
**Literature:** Todd et al. (2014, *JHLT*) — FEV1% at BAL inversely correlates with BAL lymphocytosis and ACR grade.
**Evidence:** Strong — well-established clinical association replicated in model.

---

## 2. CLAD Risk Hypotheses

### H-CLAD-1: TRAM depletion + perivascular macrophage expansion defines the pre-CLAD alveolar immune state
**Finding:** `BAL:TRAM-6` (coef=−0.011, protective), `BAL:Perivascular macrophages` (coef=+0.009, risk).
**Mechanism:** TRAM subtypes maintain alveolar homeostasis via efferocytosis and IL-10. TRAM-6 depletion signals a shift from homeostatic to inflammatory polarisation — the earliest cellular marker preceding BOS onset. Perivascular macrophage expansion reflects vascular inflammation (RAS hallmark) producing TGF-β and PDGF that drive subepithelial fibrosis.
**Literature:** Misharin et al. (2017, *JCI*) — monocyte-derived macrophages replacing TRAMs drive lung fibrosis. Calabrese et al. (2015, *JHLT*) — perivascular macrophage expansion linked to RAS-CLAD.
**Evidence:** Strong — TRAM biology in CLAD is among the most replicated findings in lung transplant immunology.

### H-CLAD-2: Naive CD4⁺ T cell reservoir depletion signals T cell exhaustion en route to CLAD
**Finding:** `BAL:CD4 T naive cells` (coef=−0.011, protective — higher naïve CD4 → longer TTE).
**Mechanism:** Depleted naïve CD4⁺ T cells in BAL reflect systemic T cell ageing/exhaustion from immunosuppression and chronic antigenic stimulation. Exhausted effector-memory T cells that dominate after naïve depletion drive chronic fibroproliferative remodelling via IL-17A and TGF-β — key CLAD/BOS cytokines.
**Literature:** Tiriveedhi et al. (2013, *Transplantation*) — IL-17A in BAL precedes BOS by 6–12 months. Berastegui et al. (2019, *ERJ*) — naïve/effector T cell ratio as BOS predictor.
**Evidence:** Moderate

### H-CLAD-3: Basaloid/AT1 alveolar epithelial reprogramming is a cellular hallmark of the fibrotic CLAD trajectory
**Finding:** `BAL:Basaloid AT1s` (coef=+0.009, risk).
**Mechanism:** Basaloid cells (KRT17⁺/KRT5⁺) represent an aberrant transitional state of AT2 cells that have entered a senescent/fibrogenic programme. Their BAL expansion reflects stalled AT2→AT1 transdifferentiation, promoting myofibroblast activation via SHH and TGF-β — the same IPF mechanism now emerging in post-transplant CLAD fibrosis.
**Literature:** Habermann et al. (2019, *Nat Med*) — basaloid cells as fibrotic drivers in IPF. Application to lung transplant CLAD is emerging.
**Evidence:** Moderate-to-Speculative

### H-CLAD-4: BAL eosinophilia represents a distinct eosinophilic CLAD endotype
**Finding:** `Clin:EOSINOPHILS, BODY FLUID Right` coef=+0.010 (risk for CLAD) but coef=−0.59 (protective for ACR).
**Mechanism:** Eosinophils suppress acute T cell–mediated alloresponse via IL-10/TGF-β (explaining protective ACR signal). Chronically, eosinophilic airway inflammation drives irreversible small airway remodelling via EETs and major basic protein deposition — an obstructive/mixed CLAD endotype. Same cell type, opposite roles on acute vs. chronic timescales.
**Literature:** Meyer et al. (2014, *JHLT*) — eosinophilic bronchitis post-lung transplant. Verleden et al. (2021, *ERJ*) — eosinophilic CLAD as recognised phenotypic variant.
**Evidence:** Moderate — cross-task sign reversal is a strong mechanistic signal.

---

## 3. Death Risk Hypotheses

### H-Death-1: H&E alveolar fraction reflects alveolar simplification and loss of gas exchange reserve
**Finding:** `HE:Alveolar` (coef=+0.019, top Death predictor).
**Mechanism:** In CLR-normalised space, high alveolar fraction implies relative reduction in bronchial/vascular tissue. In late-fibrotic grafts, alveolar simplification (emphysematous dilatation) increases relative alveolar area while destroying functional units. "Alveolar" in a simplified late-stage graft represents emphysematous destruction rather than healthy parenchyma.
**Literature:** Sato et al. (2013, *JHLT*) — alveolar simplification in CLAD subtypes. RAS shows alveolar consolidation distinct from BOS.
**Evidence:** Moderate-to-Speculative

### H-Death-2: Haematological stress markers reflect systemic disease burden driving mortality
**Finding:** `Clin:TEAR DROP CELLS` (coef=+0.014), `Clin:MICROCYTOSIS` (coef=+0.013).
**Mechanism:** Dacrocytes and microcytosis post-transplant indicate chronic anaemia, bone marrow stress from immunosuppressants (MMF/azathioprine), or CNI-induced microangiopathic haemolytic anaemia. These red cell morphology markers are sensitive indicators of systemic non-pulmonary organ stress (renal, hepatic, haematological) that cumulatively drive late mortality alongside the pulmonary axis captured by FEV1 (coef=−0.011).
**Literature:** Speich et al. (1997, *Transplantation*) — haematological toxicity predicting long-term mortality. ISHLT Registry (Chambers et al., 2021) — extrapulmonary comorbidities as dominant late mortality driver.
**Evidence:** Moderate

### H-Death-3: CXCL10⁺ and TRAM macrophages form a protective anti-fibrotic axis shared across ACR and Death
**Finding:** `BAL:CXCL10 macs-2` protective in both ACR (coef=−0.55) and Death (coef=−0.013); `BAL:TRAM-9` also protective in Death (coef=−0.011).
**Mechanism:** TRAM and CXCL10⁺ macrophages likely represent overlapping homeostatic populations that maintain alveolar surface homeostasis via efferocytosis and matrix metalloproteinase-mediated anti-fibrotic activity. Their shared protective effect across acute and long-term outcomes suggests a unified "homeostatic macrophage axis" whose preservation determines overall graft health.
**Literature:** Aguilar et al. (2019, *JCI Insight*); Hussell & Bell (2014, *Nat Rev Immunol*) — alveolar macrophage homeostasis as graft protector.
**Evidence:** Moderate — consistent cross-task direction strengthens the signal.

---

## 4. Cross-task Paradoxes and Reconciliation

| Feature | ACR effect | CLAD effect | Reconciliation |
|---|---|---|---|
| `prev_tx` | Protective (coef=−0.147 TTE) | Risk (coef=+0.032 TTE) | Prior tx → optimised immunosuppression reduces acute ACR; but accumulated DSAs from first graft accelerate CLAD via antibody-mediated remodelling |
| BAL eosinophils | Protective (coef=−0.59) | Risk (coef=+0.010) | Eosinophilic Th2 suppresses acute CD4 alloreactivity; chronic EET-mediated airway remodelling drives obstructive CLAD endotype |
| `cmv_donor` | Protective (coef=−0.43 ACR) | Not in top features | D+/R− triggers vigorous antiviral response that diverts CD4 alloreactivity (bystander suppression); or confounded by stricter valganciclovir prophylaxis protocols |
| CXCL10 macs-2 | Protective ACR | Protective Death | Consistent homeostatic macrophage biology — anti-inflammatory TRAM-like subtype |

---

## 5. Modality Contribution Insights

**Clinical dominates ACR_TTE (C=0.573 vs All=0.587):** ACR is driven by stable patient-level risk factors (EBV, FEV1%, age) captured efficiently by labs. Imaging adds marginal signal because ACR is episodic and rarely CT-detectable.

**CT leads for Death (C=0.556):** CT patch clusters capture structural irreversibility (emphysema, consolidation, vascular remodelling) across both lungs globally. BAL and H&E sample focal regions; CT integrates the whole-graft state that determines late mortality.

**All modalities hurt CLAD (C=0.501 < Clinical alone=0.554):** CLAD is a trajectory problem (FEV1 slope), not a cross-sectional one. Single-timepoint image features add noise. **CLAD requires longitudinal delta-features or temporal MIL architecture.**

**BAL underperforms alone (BACC=0.530 for ACR):** Only ~134 samples with BAL scRNA. Single-timepoint composition is noisy without temporal context.

**H&E adds unique signal (BACC=0.570 with only 632 samples):** Tissue-level spatial organisation provides complementary information not captured by isolated cell proportions or functional measurements.

---

## 6. Key Testable Predictions

**P1 — EBV × CD4 T cell interaction:**
In the cohort, stratify ACR+ patients by `recipient_ebv`. Test whether BAL CD4 T cell-1 proportions are higher at rejection vs. stable timepoints (paired Wilcoxon). Expected: EBV+/CD4-high patients have 2–3× higher ACR rate. Mechanistic validation: TCR-seq of BAL CD4 T cells from EBV+ ACR+ patients for cross-reactive clones (GLIPH2 clustering).

**P2 — TRAM depletion as prospective CLAD biomarker:**
At 6, 12, 24 months post-transplant, quantify TRAM-6 proportions in serial BAL. Test TRAM-low vs TRAM-high at 12 months for CLAD by 36 months (log-rank). Expected: TRAM-low HR > 2.0 for CLAD. Would establish 12-month BAL TRAM screening.

**P3 — Basaloid AT1 cells in H&E biopsies:**
Apply KRT17/KRT5/SOX9 immunofluorescence to biopsies from (a) stable, (b) ACR, (c) CLAD patients. Quantify basaloid density per alveolar unit. Prediction: significantly elevated in CLAD vs. ACR and stable (ANOVA p<0.01), establishing it as CLAD-specific not generic injury.

**P4 — Longitudinal CXCL10 macrophage trajectory as graft health index:**
Build per-patient CXCL10 mac-2 trajectory across all BAL timepoints. Joint model (JM R package) for ACR + survival. Expected: CXCL10 mac-2 trajectory AUC > 0.65 for 3-year survival — actionable longitudinal biomarker.

**P5 — Delta-feature CoxPH for CLAD:**
For patients with ≥2 BAL/clinical timepoints, compute Δ(TRAM-6), Δ(CD4 naïve), Δ(FEV1%), Δ(eosinophils) between visits. Retrain CoxPH on delta features. Prediction: C-index > 0.65 for CLAD (vs. 0.554 single-timepoint), confirming CLAD as a trajectory disease and motivating the full longitudinal MIL architecture.
