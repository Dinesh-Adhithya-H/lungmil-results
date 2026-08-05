# Figures Reference

## Clinical Feature Importance — Methodology

Clinical feature importance is derived from the PMA (pooling-by-multihead-attention) attention weights within the LongMK model and aggregated across patients, biopsies, and cross-validation splits.

**Key distinction from HE/BAL/CT modalities:** HE, BAL, and CT affinities are computed over k-means cluster assignments (patch clusters → cluster-level affinity). Clinical has no patch clustering — each of the 106 feature tokens is a named variable (e.g. `fev1`, `fvc`, `CREATININE`, `tacrolimus`, `ALBUMIN`), so the PMA attention matrix is used directly without any cluster binning.

### Input representation

Each biopsy contributes 106 clinical feature tokens, one per variable. Each token is a 491-dimensional one-hot vector encoding the binned value of that variable. The PMA module maps these 106 tokens into K=16 learned seed vectors; the attention matrix for the Clinical modality at biopsy t is:

```
pa[k, f]  shape (K=16, 106)   — K seeds × 106 named feature tokens
```

### Per-patient weighted affinity

For each patient the ABMIL aggregator assigns each seed a scalar weight `alpha[k]` (sum to 1). The weighted affinity per feature f is:

```
weighted_aff[f] = sum_k( alpha[k] * pa[k, f] )
```

averaged over all biopsies in the patient's longitudinal timeline. This yields a scalar per feature representing how much the model, weighted by seed importance, attends to that clinical variable for this patient.

### Group delta

Patients are split into high-risk (`hi`) and low-risk (`lo`) groups by outcome (median TTE for survival tasks; ACR+ vs. ACR− for classification). The importance delta per feature is:

```
delta[f] = mean_hi( weighted_aff[f] ) − mean_lo( weighted_aff[f] )
```

Positive delta → feature receives more attention in high-risk patients; negative → low-risk enriched.

### Cross-split aggregation and plot

The delta vector is computed independently for each of the 5 outer CV splits and aggregated as:

```
delta_mean[f] ± delta_std[f]   (mean ± s.d. across 5 splits)
```

The top-20 features by |delta_mean| are plotted as a horizontal bar chart with error bars. Red bars = enriched in high-risk; blue = enriched in low-risk.

### Output

`figures/interpretability/{task}/clinical_feature_imp_{task}.png` — one plot per task (acr_cls, acr_surv, clad_surv, death_surv), top-20 features by |delta_mean|.
