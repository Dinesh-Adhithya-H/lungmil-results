# Reproducible Pipeline

Full sequence from raw model outputs to paper figures.
All steps run via `sbatch` from the repo root (`/home/aih/dinesh.haridoss/chicago_mil`).
Stages within a group can run in parallel; groups must be run in order.

---

## Stage 0 — Linear baselines

```bash
sbatch analysis/submit_linear_models.sh        # trains CoxPH/LogReg on each modality×task
```

Output: `results/linear_models/metrics_summary.csv`

---

## Stage 1 — Deep model interpretability (GPU, ~2–4 h each)

These produce per-split interp JSONs and cached `.npy` files consumed by all downstream plots.
Run all three in parallel.

```bash
sbatch interpretability/submit_interp_longitudinal_no_alibi_allsplits.sh   # LongMK (best model)
sbatch interpretability/submit_interp_set_mil_mt_allsplits.sh              # SetMIL-MT
sbatch interpretability/submit_longitudinal_regen_npy.sh                   # regen .npy caches for UMAP
```

Output: `interpretability/longitudinal_mk_interp/`, `interpretability/set_mil_mt_interp/`

---

## Stage 2 — Biopsy representation extraction (GPU)

Depends on Stage 1 interp JSONs.

```bash
sbatch interpretability/submit_extract_biopsy_reps.sh    # patient-level embedding cache
```

Output: `interpretability/biopsy_reps/`

---

## Stage 3 — Unimodal ablation summary (CPU)

Depends on Stage 1.

```bash
sbatch interpretability/submit_unimodal_ablation.sh    # builds unimodal_ablation_summary.csv
```

Output: `interpretability/unimodal_ablation/unimodal_ablation_summary.csv`

---

## Stage 4 — Aggregated interpretability panels (CPU)

Depends on Stage 1. Run in parallel.

```bash
sbatch interpretability/submit_cluster_aff_agg.sh    # cluster affinity heatmaps (HE/BAL/CT)
sbatch interpretability/submit_agg_death_clad.sh     # death+CLAD seed attribution aggregation
sbatch interpretability/submit_regen_G.sh            # patient-rep UMAP from npy caches
sbatch interpretability/submit_trajectory_panel.sh   # longitudinal trajectory panels
```

Output: `figures/interpretability/cluster_agg/`, `figures/interpretability/agg/`, `figures/trajectories/`

---

## Stage 5 — Benchmark figures (CPU)

Depends on Stages 0, 1, 3. Run in parallel.

```bash
sbatch analysis/submit_benchmark_ablation_v2.sh      # benchmark bars + unimodal ablation + modality combo
sbatch analysis/submit_benchmark_multimodal.sh       # multimodal comparison matrix
sbatch analysis/submit_benchmark_table_v2.py         # summary table (PDF)
sbatch analysis/submit_linear_benchmarks.sh          # linear baseline figure
```

Output: `figures/benchmark/`

---

## Stage 6 — Interpretability figures (CPU)

Depends on Stages 1–4. Run in parallel.

```bash
sbatch analysis/submit_biopsy_rep_umap_km.sh         # biopsy-level UMAP + KM curves (all tasks)
sbatch analysis/submit_clinical_feature_imp.sh       # clinical feature importance bars (all tasks)
sbatch analysis/submit_unified_rep_umap.sh           # unified cross-modal UMAP
sbatch analysis/submit_multimod_seed_attribution.sh  # seed attribution panels
sbatch analysis/submit_L_global_avg.sh               # biopsy-weighting heatmaps
sbatch analysis/submit_km_from_model.sh              # KM from model risk scores
```

Output: `figures/interpretability/`

---

## Stage 7 — Nature manuscript figures

Depends on all prior stages.

```bash
sbatch analysis/submit_nature_figs_all.sh    # main figures (omnibus)
sbatch analysis/submit_hero_fig.sh           # Fig 0 hero panel
```

Output: `analysis/nature_paper/`

---

## Utility scripts (run on demand, not part of the main pipeline)

| Script | Purpose |
|---|---|
| `analysis/rebuild_benchmark_csvs.py` | Rebuild `comparison_*.csv` from raw JSONs if results change |
| `analysis/aggregate_predictions.py` | Per-patient prediction tables |
| `analysis/extract_predictions.py` | Per-split test set extraction |
| `analysis/extract_cluster_names.py` | Cluster name maps |
| `analysis/extract_cluster_proportions.py` | Cluster proportion CSVs |
| `analysis/extract_clinical_features.py` | 106-feature clinical tensor extraction |
| `analysis/compare_modalities.py` | Modality comparison tables |
| `analysis/bootstrap_ci.py` | Bootstrap CI helper (called by benchmark scripts) |
| `interpretability/shared.py` | Shared colors, constants, UMAP helper |
