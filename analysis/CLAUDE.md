# Analysis & Visualization Rules

## UMAP + Hexbin Design Rules (NEVER violate these)

### Hexbin construction
1. Compute UMAP embeddings first, then define the hexbin grid from `xy_s` (all test samples, `mincnt=1`)
2. Use the **SAME grid** (same `gridsize`, same reference population `xy_s`) for ALL hexbin subplots in a panel — do NOT recompute per subplot
3. Only draw bins where samples actually live (`mincnt=1` on `xy_s`) — no bins in empty corners of the plane
4. If a bin has no valid values for a specific metric (e.g., TTE is NaN for censored samples), draw a **solid yellow hexagon** (`_EMPTY_COLOR = "#FFE57F"`) with thin grey edge — NEVER overlap: background layer gets `linewidths=0.5`, foreground layer gets `linewidths=0, edgecolors="none"` so no background color bleeds around data bins
5. `_uniform_lim(axes, xy_s)` must be called after all plots in a panel so scatter and hexbin axes share identical xlim/ylim

### Color scheme — RED = HIGH RISK, BLUE = LOW RISK (always)
- **Hazard score**: high hazard → red → use `CMAP_HAZARD = "RdBu_r"`
- **Time to event (TTE)**: short TTE (imminent event) → red → use `CMAP_TTE = "RdBu"`
- **Event density**: many events in region → red → use `CMAP_DENSITY = "RdBu_r"`
- **ACR label / event status**: ACR+ / event → red (`#E53935`); ACR- / censored → blue (`#1E88E5`)
- **cls_prob (P(ACR+))**: high probability → red → use `"RdBu_r"`
- **Error |pred-label|**: high error → red → use `"RdBu_r"` (consistent with all other risk metrics)
- **NEVER** use colormaps that go red→white (e.g. `"Reds"`) — always red↔blue for all metrics

### Task-specific panels
- **cls tasks** (acr_cls, acr_alt_cls): use `r_cls` embedding, show ACR label + cls_prob panels
- **TTE tasks** (acr_tte, acr_alt_tte, clad, death): use `r_tte` embedding, show event/censored + hazard + TTE panels
- **Multitask** (acr_alt): generates BOTH cls panel (from r_cls) AND tte panel (from r_tte)
- Each fold contributes test samples → pool all folds for ~4210 unique test samples (each sample appears once)

### Build & submit
- ALWAYS submit via `bash analysis/submit_analysis.sh` — NEVER run Python directly on the login node
- Caches are saved per endpoint; benchmarks use JSON metrics only (fast, no GPU)
- Fix bugs then `scancel <JID>` before resubmitting

## Model Variants
Tags in checkpoint dirs → `build_model_v7()` base variant:
- `early`, `late`, `middle` → same name
- `crossattn_k8` → `"crossattn"`
- `crossmodal_k8` → `"crossmodal"`
- `iterative_r2_k8` → `"iterative"`

## Combo Performance
- cls tasks: compute real **BAcc** (`balanced_accuracy_score`) per modality combo, not raw prob mean
- TTE tasks: compute real **C-index** (Harrell's concordance) per modality combo, not raw hazard mean
