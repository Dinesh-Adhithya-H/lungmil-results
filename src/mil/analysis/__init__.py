"""
Analysis and visualisation subpackage.

Entry point
-----------
    python analysis/run_analysis.py --results_base ... --output_dir v7_analysis/ --tasks all

Modules
-------
run         Unified entry point (argparse) dispatching all tasks
config      Task registry, endpoint configs, color schemes
io          Metrics loading, cache read/write, fold stats
enrich      Post-hoc cache enrichment (combo, event labels)
inference   Run model inference, cache representations
plots/benchmark  Bar charts, heatmap, single vs multitask comparison
plots/umap       UMAP panels per task (test-set, all variants)
plots/combo      Per-modality-combo performance breakdown
"""
