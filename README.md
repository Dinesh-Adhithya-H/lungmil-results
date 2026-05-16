# Chicago MIL — Multimodal ABMIL for ACR Prediction

Multimodal attention-based multiple-instance learning (ABMIL) pipeline for predicting acute cellular rejection (ACR) from transplant biopsies. Combines histology (HE), BAL fluid cytology, CT imaging, and clinical data.

---

## Repository layout

```
chicago_mil/
├── train_mm_abmil_v7.py      # Main training script — single-phase, all fusion variants
├── benchmarks/
│   └── train_mm_abmil_v7.py  # TripleStreamFusionMIL benchmark (not used in main pipeline)
├── analysis/
│   └── analyze_v7.py         # Analysis + UMAP visualisation script
├── results_mm_abmil_v7/      # Training outputs
│   ├── split{s}_fold{f}/     # Per-fold results
│   │   ├── metrics_{variant}.json
│   │   ├── ckpts_{variant}/best_model.pt
│   │   └── status_{variant}.json
│   ├── analysis/             # Analysis outputs
│   │   ├── variant_table/
│   │   ├── variant_bars/
│   │   └── umap/
│   └── job_scripts/          # SLURM submission scripts
└── README.md                 # This file
```

---

## Data

### Source
`/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv`

- **4,210 samples** from **263 patients**
- **Nested cross-validation**: 5 outer splits × 4 inner folds = 20 folds total

### Key columns

| Column | Description |
|--------|-------------|
| `file` | Sample stem (e.g. `00049.pt`) |
| `patient_id` | Patient identifier |
| `anchor_dt` | Biopsy date |
| `acr_grade` | Biopsy ACR grade (`A0B0`, `A1B0`, `A2B0`, etc.) |
| `split{s}_fold{f}` | Split assignment: `train` / `val` / `test` |

### Label derivation (`acr_label`)

Labels are derived directly from `acr_grade`:

```python
def acr_label(grade_str) -> Optional[int]:
    g = str(grade_str)
    if g.startswith("A0"): return 0   # No rejection
    if g.startswith("A1") or g.startswith("A2"): return 1  # Rejection
    return None  # Unknown — excluded from hinge loss, included for Cox risk set
```

Label distribution:
- A0 (no rejection): **1,415** samples
- A1/A2 (rejection): **180** samples
- Unknown / no grade: **2,615** samples

### Modalities

| Key | Data source | Feature dim |
|-----|------------|-------------|
| `HE` | Histology patch embeddings | 1024 |
| `BAL` | BAL cytology patch embeddings | 10 |
| `CT` | CT scan patch embeddings | 1024 |
| `Clinical` | One-hot clinical features | 408 |

Sample `.pt` files: `/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples/`

---

## Survival: Time-to-next-ACR (gap-time approach)

Each biopsy is treated as a fresh start. For sample at date T for patient P:

```python
def compute_tte_next_acr(df) -> dict:
    # ACR events = biopsies with acr_grade starting A1 or A2
    # future_acr = [dates of A1/A2 biopsies for P strictly after T]
    # if future_acr:  tte = (earliest - T).days,   event = 1
    # else:           tte = (last_biopsy_P - T).days, event = 0  (censored)
```

| Sample situation | tte | event |
|-----------------|-----|-------|
| Pre-first-ACR | days until first A1/A2 | 1 |
| Between two ACR episodes | days until next A1/A2 | 1 |
| After last ACR, no more events | days to last biopsy | 0 (censored) |
| Patient never had ACR | days to last biopsy | 0 (censored) |

Result: **510** event=1 samples, **3,700** censored.

---

## Training

### Single-phase end-to-end

All multimodal fusion variants are trained directly from randomly initialised encoders. There is no per-modality pre-training phase.

### Loss

```
L = L_hinge + λ_cox · L_cox
```

| Term | Applied to | Notes |
|------|-----------|-------|
| `L_hinge` | Samples with known ACR label (A0/A1/A2) | `w · max(0, 1 − y · logit)`, class-weighted |
| `L_cox` | All samples with valid TTE (censored + events) | Cox-Breslow; censored contribute to risk set denominator only |

Defaults: `λ_cox = 1.0`.

### Optimiser / schedule

| Parameter | Value |
|-----------|-------|
| Optimiser | AdamW |
| LR | 5e-5 |
| Weight decay | 1e-3 |
| Scheduler | CosineAnnealing (T_max=200, η_min=1e-6) |
| Grad accum | 4 steps |
| Grad clip | max_norm=1.0 |
| AMP | `torch.amp.autocast("cuda")` + GradScaler |
| Epochs | 200 |
| Eval every | 20 epochs |

---

## Model architecture

### Per-modality encoder: `GatedAttentionEncoder`

```
patches (N, D_m) → Linear → backbone (N, H)
                → att_V (Tanh) × att_U (Sigmoid) → att_w → softmax → weighted sum
→ r_m (H,)       H = 256
```

### Dual-task pooling

Each model returns a 4-tuple `(logit, hazard, r_cls, r_surv)`.

Classification and survival are **competing objectives** — each task gets its own independent pooling pathway from the shared features.

#### Early / Late / Middle fusion — `DualGatedPool`

Two independent gated ABMIL pools on the same token set:

```
tokens (N, H)
  ├── cls pathway:  A_cls  = softmax(w_cls  · (V_cls(x)  ⊙ U_cls(x)))
  │                r_cls  = A_cls^T · x  →  cls_head  →  logit
  └── surv pathway: A_surv = softmax(w_surv · (V_surv(x) ⊙ U_surv(x)))
                   r_surv = A_surv^T · x → hazard_head → hazard
```

#### Slot-attention models — `DualTaskHead`

Two learnable CLS tokens independently attend to the K summary slots:

```
slots (K, H)
  ├── CLS_cls  → MultiheadAttention(Q=CLS_cls,  KV=slots) → r_cls  → logit
  └── CLS_surv → MultiheadAttention(Q=CLS_surv, KV=slots) → r_surv → hazard
```

### Fusion variants

| Variant | How tokens are produced | Pooling |
|---------|------------------------|---------|
| `early` | All patches concatenated → backbone (N_total, H) | `DualGatedPool` |
| `early_cls` | Same as early | `DualGatedPool` |
| `late` | Per-modality ABMIL → M modality summaries (M, H) | `DualGatedPool` |
| `middle` | Per-modality ABMIL → cross-modal transformer (L=2) → (M, H) | `DualGatedPool` |
| `middle_cls` | Same as middle | `DualGatedPool` |
| `crossattn` | All-pairs bidir patch cross-attn → slot attn K slots/mod → cross-modal xfmr → (K·M, H) | `DualTaskHead` |
| `crossattn_cls` | Same as crossattn | `DualTaskHead` |
| `crossmodal` | Shared-Q slot attention → cross-modal transformer → (K·M, H) | `DualTaskHead` |
| `crossmodal_cls` | Same as crossmodal | `DualTaskHead` |
| `iterative` | R rounds of (self + cross patch attention) → slot attn → cross-modal xfmr | `DualTaskHead` |
| `iterative_cls` | Same as iterative | `DualTaskHead` |

`_cls` suffix variants: for `early/middle` the `_cls` suffix is retained in the tag but both use `DualGatedPool` (slot-attn models always use `DualTaskHead`).

Default hyperparameters: `K=8` slots, `R=2` iterative rounds, `L=2` cross-modal transformer layers.

---

## Training loop

```python
for rec in records:
    out = model(bags, device)
    logit, hazard, r_cls, r_surv = out

    # Hinge loss — labeled samples only
    if rec["label"] is not None:
        cls_losses.append(hinge_loss(logit, label, class_weights))

    # Cox buffer — all samples with valid TTE (including censored)
    tte, ev = rec["tte_next_acr"], rec["event_next_acr"]
    if not isnan(tte) and tte >= 0:
        cox_buffer.append((hazard, tte, ev))

# Flush every grad_accum=4 steps:
L = mean(cls_losses) + λ_cox · cox_breslow_loss(cox_buffer)
L.backward()
clip_grad_norm_(max_norm=1.0)
optimizer.step()
```

---

## Running training (SLURM)

All compute jobs must be submitted via sbatch — never run Python directly on login nodes.

```bash
# Submit training (split 0, all 4 folds, all variants) + analysis as dependency
bash results_mm_abmil_v7/job_scripts/submit_v7.sh

# Submit training only
sbatch results_mm_abmil_v7/job_scripts/v7_train.sh

# Submit analysis only (after training completes)
sbatch results_mm_abmil_v7/job_scripts/v7_analyze_cpu.sh
sbatch results_mm_abmil_v7/job_scripts/v7_analyze_umap.sh
```

### Key CLI arguments

```
--folds          inner fold indices (default: 0 1 2 3)
--split          outer split index
--p2_variants    fusion variants to train (default: all 11)
--p2_iter_r      iterative rounds R (default: 2)
--p2_slot_k      slot count K (default: 8)
--p2_slot_iters  slot attention rounds (default: 3)
--lambda_cox     Cox loss weight (default: 1.0)
--v7_patience    early-stop patience in eval periods (default: 10)
--save_dir       output directory
--samples_dir    path to .pt sample files
--splits_csv     path to nested CV splits CSV
```

---

## Analysis: `analysis/analyze_v7.py`

### Tasks

| Task | Output | Partition |
|------|--------|-----------|
| `variant_table` | `analysis/variant_table/variant_heatmap.png` + `variant_table.csv` | CPU |
| `variant_bars` | `analysis/variant_bars/variant_bars.png` + `auc_ranked.png` | CPU |
| `umap` | `analysis/umap/{variant}/umap_grid.png` + `umap_variant_comparison.png` | GPU |

### UMAP pipeline

For each variant found in `split 0 / ckpts_{variant}/best_model.pt`:

1. Load `train_mm_abmil_v7.py` as a module (`build_model_v7`)
2. Rebuild model with fresh random init, same architecture
3. Load checkpoint state dict
4. Forward: `model(bags, device)` → `(logit, hazard, r_cls, r_surv)`
5. Fit UMAP on `r_cls` (classification representation) for test samples
6. Plot 2×3 grid per variant:
   - ACR label (red=ACR, blue=no-ACR, grey=unknown)
   - Classification probability
   - Days to next ACR (gap-time)
   - Hazard score
   - ACR status
   - Event flag

Also produces `umap_variant_comparison.png` — all variants × all colorings in one figure.

### Meta-data derivation

`stem_to_meta` in `analyze_v7.py` uses the same gap-time logic as the training script:
- `label`: from `acr_grade` (A0*→0, A1*/A2*→1)
- `days_to_acr`: `tte_next_acr` for event=1 samples only
- `tte_next_acr` / `event_next_acr`: gap-time to next A1/A2 biopsy

---

## Key design decisions

**Single-phase training**
All fusion variants are trained end-to-end from random init. This avoids phase-1 pre-training overhead and lets each variant find its own optimal encoder representations.

**Dual-task pooling (DualGatedPool / DualTaskHead)**
Classification (ACR label) and survival (time-to-next-ACR) have competing gradient directions. Each task gets its own independent pooling pathway over the same shared features, so neither objective forces the other's attention weights into a suboptimal configuration.

**Gap-time survival**
ACR is a recurrent event. Gap-time treats each biopsy as a fresh start and asks: when is the *next* ACR from this biopsy? This gives **510 observed events** and correctly handles patients with multiple ACR episodes.

**Include censored samples in Cox loss**
Standard Cox-Breslow uses all samples for the risk set denominator. Dropping censored samples biases hazard estimates toward shorter event times.

**Labels from `acr_grade`, not a pre-computed column**
`acr_grade` is the ground truth. A0* → no rejection, A1*/A2* → rejection. All other grades are treated as unknown and excluded from the hinge loss (but still contribute to the Cox risk set via their TTE).
