# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical rule: never run Python on the login node

All compute (training, analysis, data prep) must be submitted via `sbatch`. Running Python directly on the login node is forbidden. To run a quick script yourself, use `! sbatch <script>` syntax from the terminal.

## Environment

```bash
conda activate chicago   # torch 2.6.0+cu124, CUDA 12.6
```

Data lives on Lustre: `/lustre/groups/aih/dinesh.haridoss/datasets/mil_v2/samples/`  
Results are written under: `results/mm_abmil_v8/` (inside this repo, not on Lustre)

## Repository structure

```
train_mm_abmil_v8.py        # Main entry point — phases p1, p2, or both
src/mil/
  data/
    registry.py             # MODALITIES = [HE, BAL, CT, Clinical], feat dims
    loader.py               # preload_bags() — thread-parallel .pt loading + BagCache
    splits.py               # build_splits_multitask() — reads nested CV CSV
    labels.py               # acr_label(), survival label derivation
  models/
    phase1.py               # SingleModalMIL — per-modality ABMIL encoder
    phase2.py               # EarlyFusionMIL, LateFusionMIL, MiddleFusionMIL, SetTransformerMIL
    encoders.py             # GatedAttentionEncoder, ModalFFNEncoder, PMA, SAB, CrossModalTransformer
    builders.py             # build_model_v8() — dispatches variant string → model class
  training/
    phase1_trainer.py       # run_phase1_modality(), run_phase1_hp_sweep()
    phase2_trainer.py       # run_phase2_hp_sweep(), run_phase2_final(), evaluate_unimodal_ablation()
    losses.py               # cox_breslow_loss(), hinge_loss()
    metrics.py              # compute_metrics() — BACC, AUC, C-index
results_mm_abmil_v8/
  job_scripts/              # Per-split SLURM scripts (p1_split{s}.sh, p2_s{s}_{variant}.sh)
  slurm_logs/               # %j_<name>.out / .err
analysis/
  CLAUDE.md                 # Visualization rules (hexbin, color scheme — read before touching plots)
  nature_figs_all.py        # All Nature-quality figures
```

## Two-phase nested CV design

**Phase 1 (P1):** Per-modality ABMIL trained independently. One model per `(split, fold, modality, task)`. Tasks: `acr_cls`, `acr_surv`, `clad`, `death`.

**Phase 2 (P2):** Multimodal fusion. Takes P1 encoder weights as frozen backbone, learns fusion on top.

**Nested CV structure:** 5 outer splits × 4 inner folds. Test set is fixed per split (same samples across all 4 folds). Fold 0 = final test model; folds 1–3 = HP sweep only.

**Correct HP selection protocol:**
- Folds 1–3: `--p2-hp-sweep` only (no `--combined-train`) — contributes HP sweep JSON
- Fold 0: `--p2-hp-sweep --global-hp --combined-train` — aggregates best HP from all 4 folds, trains on train+val, evaluates on test

**Skip logic:**
- Folds 1–3 skip: check `phase2/split{s}_fold{f}/{variant}_{task}/hp_sweep/hp_sweep_p2.json`
- Fold 0 skip: check `phase2/split{s}_fold0/{variant}_{task}/metrics_{variant}_final.json`

## P2 variants

| Variant | Architecture | Task(s) |
|---------|-------------|---------|
| `early` | All patches concatenated → ABMIL | `cls`, `acr_surv`, `clad_surv`, `death_surv` |
| `late` | Per-modality ABMIL → weighted combination | same |
| `middle` | Per-modality ABMIL → CrossModalTransformer → ABMIL | same |
| `mario_kempes` | PMA per modality → SAB cross-modal → per-task ABMIL | `mega` (all tasks jointly) |

`mario_kempes` uses `--task mega` — all four tasks in one model. All others use separate per-task runs.

## Key constants (phase2_trainer.py)

```python
P2_GRAD_ACCUM = 32        # gradient accumulation steps (A100 80GB)
P2_MAX_PATCHES = 2048     # max patches per modality per patient
HIDDEN_DIM = 256
```

## Data: .pt file format

Each sample file `{stem}.pt` contains:
- `inputs.HE_cells` → `(N, 1024)` patch embeddings
- `inputs.BAL_cells` → `(N, 10)` patch embeddings  
- `inputs.CT_cells` → `(N, 1024)` patch embeddings
- `clinical_onehot` → `(106, 491)` one-hot clinical features
- Survival labels: `tte_next_acr`, `event_next_acr`, `clad_time`, `clad_event`, `death_time`, `death_event`

The stem (e.g. `00049`) is the primary key linking `.pt` files to the splits CSV.

## Splits CSV

`/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv`

Key columns: `file` (stem), `patient_id`, `anchor_dt`, `acr_grade`, `split{s}_fold{f}` (`train`/`val`/`test`).

Label: A0* → 0 (no rejection), A1*/A2* → 1 (rejection), other → None (excluded from hinge loss, included in Cox risk set).

## Metrics

- Classification tasks: primary = **BACC** (balanced accuracy), secondary = AUC
- Survival tasks: primary = **C-index**
- Combined HP selection metric for `mario_kempes`: `0.5 × BACC + 0.5 × mean(CI_acr, CI_clad, CI_death)`

## Submitting jobs

```bash
# Submit all 5 P1 splits
bash scripts/submit_p1_all_splits.sh

# Submit all 20 P2 jobs (5 splits × 4 variants)
bash scripts/submit_p2_all_splits.sh

# Individual scripts are in results_mm_abmil_v8/job_scripts/
sbatch results_mm_abmil_v8/job_scripts/p1_split0.sh
sbatch results_mm_abmil_v8/job_scripts/p2_s0_mario_kempes.sh
```

Jobs self-resubmit on wall-time (SIGUSR1 trap, 2-min warning). Skip logic prevents re-running completed work.

## OOM prevention (mario_kempes)

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This is set in all mario_kempes job scripts. `P2_GRAD_ACCUM=32` (not 64) is required to avoid OOM on A100 80GB.

## Results location

```
results/mm_abmil_v8/
  phase1/split{s}_fold{f}/{task}/{modality}/final/metrics.json   # P1 test metrics
  phase2/split{s}_fold0/{variant}_{task}/metrics_{variant}_final.json  # P2 test metrics
  phase2/split{s}_fold{f}/{variant}_{task}/hp_sweep/hp_sweep_p2.json   # HP sweep results
```

`metrics_*_final.json` contains keys: `train`, `val`, `test`, and optionally `unimodal_ablation` (multimodal model evaluated with one modality at a time).

## Analysis / visualization

See `analysis/CLAUDE.md` for detailed rules on hexbin plots, color schemes, and UMAP pipelines. Key rule: RED = HIGH RISK, BLUE = LOW RISK in all plots. Submit analysis jobs via `bash analysis/submit_analysis.sh`.
