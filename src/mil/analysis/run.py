"""
Unified analysis entry point.

Usage:
    python -m mil.analysis.run \\
        --results_base /path/to/chicago_mil \\
        --splits_csv   /path/to/splits.csv  \\
        --samples_dir  /path/to/samples     \\
        --output_dir   /path/to/v7_analysis \\
        --tasks all \\
        --folds 0 1 2 3 \\
        --splits 0 \\
        --device cuda \\
        [--skip_umap] [--skip_combo] [--skip_benchmark]

All outputs land under output_dir/:
    benchmark/                   -- task × variant bar chart, heatmap, CSV
    umap_{task_key}/             -- UMAP panels per task
    combo/                       -- modality-combo performance
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .config import TASKS, TASK_ENDPOINT, VARIANT_TAGS
from .io import load_metrics_dir, fold_stats, ordered_tags
from .inference import get_or_run


def _split_sizes(splits_csv: Path, splits: List[int], folds: List[int]) -> None:
    """Print train/val/test split sizes per fold."""
    import pandas as pd
    df = pd.read_csv(str(splits_csv))
    print("\n[split sizes]")
    for s in splits:
        for f in folds:
            col = f"split{s}_fold{f}"
            if col not in df.columns:
                continue
            cnts = df[col].value_counts()
            print(f"  split={s} fold={f}: "
                  f"train={cnts.get('train',0)}  "
                  f"val={cnts.get('val',0)}  "
                  f"test={cnts.get('test',0)}")


def run(
    results_base: Path,
    splits_csv:   Path,
    samples_dir:  Path,
    output_dir:   Path,
    tasks:        List[str],
    splits:       List[int],
    folds:        List[int],
    device:       str = "cpu",
    chicago_mil_dir: Optional[Path] = None,
    skip_umap:    bool = False,
    skip_combo:   bool = False,
    skip_benchmark: bool = False,
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve task list
    if "all" in tasks:
        task_keys = list(TASKS.keys())
    else:
        task_keys = [t for t in tasks if t in TASKS]
        unknown = [t for t in tasks if t not in TASKS]
        if unknown:
            print(f"[run] Unknown tasks ignored: {unknown}")

    print(f"[run] Tasks: {task_keys}")
    _split_sizes(splits_csv, splits, folds)

    # ── Benchmark (metrics JSON only — no inference needed) ────────────────────
    if not skip_benchmark:
        print("\n[run] === Benchmark ===")
        from .plots.benchmark import task_benchmark
        task_benchmark(results_base, output_dir, folds,
                       task_keys=None if "all" in tasks else task_keys)

    # ── Inference cache ────────────────────────────────────────────────────────
    # Group tasks by (results_dir, endpoint) to avoid duplicate inference
    if not skip_umap or not skip_combo:
        inferred: Dict[str, Optional[Dict]] = {}
        seen_dirs: Dict[str, str] = {}  # results_dir_str → first task_key that used it

        for task_key in task_keys:
            dir_suf, prim, label, color, task_type = TASKS[task_key]
            endpoint  = TASK_ENDPOINT[task_key]
            results_d = results_base / dir_suf
            dir_str   = str(results_d)

            # For acr_alt_cls and acr_alt_tte: same results dir and endpoint — reuse cache
            cache_key = f"{dir_str}::{endpoint}"
            if cache_key in seen_dirs:
                reuse_key = seen_dirs[cache_key]
                print(f"[run] {task_key}: reusing inference from {reuse_key}")
                inferred[task_key] = inferred[reuse_key]
                continue

            seen_dirs[cache_key] = task_key
            print(f"\n[run] === Inference: {task_key} ({label}) ===")
            vd = get_or_run(
                results_dir      = results_d,
                splits_csv       = splits_csv,
                samples_dir      = samples_dir,
                splits           = splits,
                folds            = folds,
                endpoint         = endpoint,
                output_dir       = output_dir,
                device_str       = device,
                chicago_mil_dir  = chicago_mil_dir,
            )
            inferred[task_key] = vd

        # ── UMAP plots ─────────────────────────────────────────────────────────
        if not skip_umap:
            from .plots.umap import task_umap
            for task_key in task_keys:
                vd = inferred.get(task_key)
                if not vd:
                    print(f"[run] {task_key}: no inference data — skip UMAP"); continue
                dir_suf, prim, label, color, task_type = TASKS[task_key]
                endpoint = TASK_ENDPOINT[task_key]
                umap_out = output_dir / f"umap_{task_key}"
                print(f"\n[run] === UMAP: {task_key} ===")
                task_umap(
                    task         = task_key,
                    variant_data = vd,
                    output_dir   = umap_out,
                    endpoint     = endpoint,
                )

        # ── Combo performance ──────────────────────────────────────────────────
        if not skip_combo:
            from .plots.combo import task_combo
            print("\n[run] === Combo performance ===")
            task_combo(inferred, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Unified MIL analysis — benchmark, UMAP, combo plots"
    )
    parser.add_argument("--results_base", required=True, type=Path,
                        help="Root dir containing all results_mm_abmil_v7_* subdirs")
    parser.add_argument("--splits_csv",   required=True, type=Path)
    parser.add_argument("--samples_dir",  required=True, type=Path)
    parser.add_argument("--output_dir",   required=True, type=Path)
    parser.add_argument("--tasks", nargs="+", default=["all"],
                        help="Task keys or 'all'. Options: " + " ".join(TASKS.keys()))
    parser.add_argument("--folds",  nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--splits", nargs="+", type=int, default=[0])
    parser.add_argument("--device", default="cpu",
                        help="'cuda' or 'cpu'")
    parser.add_argument("--chicago_mil_dir", type=Path, default=None,
                        help="Root of chicago_mil repo (for loading model code)")
    parser.add_argument("--skip_umap",      action="store_true")
    parser.add_argument("--skip_combo",     action="store_true")
    parser.add_argument("--skip_benchmark", action="store_true")
    args = parser.parse_args()

    run(
        results_base    = args.results_base,
        splits_csv      = args.splits_csv,
        samples_dir     = args.samples_dir,
        output_dir      = args.output_dir,
        tasks           = args.tasks,
        splits          = args.splits,
        folds           = args.folds,
        device          = args.device,
        chicago_mil_dir = args.chicago_mil_dir,
        skip_umap       = args.skip_umap,
        skip_combo      = args.skip_combo,
        skip_benchmark  = args.skip_benchmark,
    )


if __name__ == "__main__":
    main()
