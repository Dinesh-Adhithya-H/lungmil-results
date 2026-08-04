# Figures Reference

## Clinical Feature Importance — Methodology

Clinical feature importance is derived from the PMA (pooling-by-multihead-attention) attention weights within the LongMK/SetMIL models and aggregated across patients, biopsies and cross-validation splits.

### Input representation

Each biopsy contributes 106 clinical feature tokens, one per variable (e.g. `fvc`, `fev1`, `ALBUMIN`, `CREATININE`, etc.). Each token is a 491-dimensional one-hot vector encoding the binned value of that variable. The PMA module maps these 106 tokens into K=16 learned seed vectors; the attention matrix for the Clinical modality at biopsy t is:

```
pa[k, f]  shape (K=16, 106)
```

where `pa[k, f]` is the attention weight from seed k to feature token f.

### Per-patient weighted affinity

For each patient, the ABMIL aggregator assigns each seed a scalar importance weight `alpha[k]` (summing to 1 across K seeds). The seed-weighted clinical feature affinity is:

```
affinity[k, f] = alpha[k] * mean_over_biopsies( pa[k, f] )
```

This is averaged over all biopsies in the patient's timeline. The result is a (K, 106) matrix expressing how much each seed, weighted by its ABMIL importance, attends to each clinical feature.

### Group delta

Patients are stratified into high-risk (`hi`) and low-risk (`lo`) groups based on outcome (short vs. long TTE for survival tasks; ACR+ vs. ACR− for classification). For each group, the per-patient affinity matrices are averaged:

```
delta[f] = mean_hi( affinity[:, f] ) - mean_lo( affinity[:, f] )
```

Features with large |delta[f]| are differentially attended to between risk groups — positive delta indicates the feature receives more attention in high-risk patients, negative in low-risk patients.

### Cross-split aggregation

The delta vector is computed independently for each of the 5 outer CV splits. The final importance score per feature is:

```
delta_mean[f] ± delta_std[f]   (mean ± s.d. across 5 splits)
```

The top-N features by |delta_mean| are shown as a horizontal bar chart with error bars. The colour indicates direction: red = enriched in high-risk, blue = enriched in low-risk.

### Output

`figures/interpretability/cluster_agg/{task}_cluster_aff_agg.png` — one panel per modality (HE, BAL, CT, Clinical), top-14 features by |delta_mean|.
