"""
Post-hoc W&B uploader for chicago-mil results.

Uploads:
  - Phase 1 test metrics (per split/fold/task/modality)
  - Phase 2 test metrics (per split/fold/variant/task)
  - TCGA benchmark table
  - Nature paper figures (analysis/nature_paper/*.png)
  - Interpretability figures (interpretability/**/*.png)

Usage:
    python scripts/log_results_to_wandb.py [--project chicago-mil] [--dry-run]
"""
import argparse
import json
import os
import re
import glob
from pathlib import Path

import wandb

RESULTS_ROOT = Path(__file__).parent.parent / "results" / "mm_abmil_v8"
REPO_ROOT = Path(__file__).parent.parent

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def scalar_subset(d: dict) -> dict:
    """Return only scalar (non-list) entries from a metrics dict."""
    return {k: v for k, v in d.items() if isinstance(v, (int, float)) and v is not None}


def primary_metric(task: str, metrics: dict) -> float | None:
    """Return the single headline metric for a task."""
    if "cls" in task or task == "acr":
        return metrics.get("bacc")
    return metrics.get("c_index")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1
# ──────────────────────────────────────────────────────────────────────────────

def upload_phase1(project: str, dry_run: bool):
    """One wandb run per (split, fold, task, modality)."""
    pattern = str(RESULTS_ROOT / "phase1" / "split*_fold*" / "*" / "*" / "*/metrics.json")
    files = glob.glob(pattern, recursive=True)
    print(f"[P1] found {len(files)} metrics.json files")

    for fpath in sorted(files):
        parts = Path(fpath).parts
        # …/phase1/split{s}_fold{f}/{task}/{modality}/final*/metrics.json
        try:
            phaseidx = parts.index("phase1")
            split_fold = parts[phaseidx + 1]   # split0_fold0
            task = parts[phaseidx + 2]
            modality = parts[phaseidx + 3]
        except (ValueError, IndexError):
            continue

        m = re.match(r"split(\d+)_fold(\d+)", split_fold)
        if not m:
            continue
        split, fold = int(m.group(1)), int(m.group(2))

        with open(fpath) as f:
            d = json.load(f)

        test = scalar_subset(d.get("test", {}))
        if not test:
            continue

        run_name = f"p1_s{split}f{fold}_{task}_{modality}"
        config = {"phase": 1, "split": split, "fold": fold, "task": task, "modality": modality}
        summary = {f"test/{k}": v for k, v in test.items()}
        pm = primary_metric(task, test)
        if pm is not None:
            summary["test/primary"] = pm

        if dry_run:
            print(f"  [DRY] {run_name}: {summary}")
            continue

        with wandb.init(project=project, name=run_name, config=config,
                        group=f"phase1_s{split}", tags=["phase1", task, modality],
                        job_type="phase1", reinit=True) as run:
            run.summary.update(summary)

    print("[P1] done")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2
# ──────────────────────────────────────────────────────────────────────────────

def upload_phase2(project: str, dry_run: bool):
    """One wandb run per (split, fold, variant, task)."""
    # Flat metrics in results root: metrics_split{s}_fold{f}_{variant}_{task}.json
    flat_files = glob.glob(str(RESULTS_ROOT / "metrics_split*_fold*_*.json"))
    # Nested phase2 finals: phase2/split{s}_fold{f}/{variant}_{task}/metrics_*_final.json
    nested_files = glob.glob(str(RESULTS_ROOT / "phase2" / "*" / "*" / "metrics_*_final.json"))

    all_files: list[tuple[str, int, int, str, str]] = []

    for fpath in flat_files:
        stem = Path(fpath).stem  # metrics_split0_fold0_early_acr_surv
        m = re.match(r"metrics_split(\d+)_fold(\d+)_(.+)", stem)
        if not m:
            continue
        split, fold = int(m.group(1)), int(m.group(2))
        rest = m.group(3)
        # rest is like "early_acr_surv" or "set_mil_mt_mega"
        # variant is first word before task suffix
        for variant in ("early", "late", "middle", "mario_kempes", "set_mil_mt", "longitudinal_mk_mt", "longitudinal_mk"):
            if rest.startswith(variant):
                task = rest[len(variant):].lstrip("_") or "mega"
                all_files.append((fpath, split, fold, variant, task))
                break

    for fpath in nested_files:
        parts = Path(fpath).parts
        try:
            p2idx = parts.index("phase2")
            split_fold = parts[p2idx + 1]
            variant_task = parts[p2idx + 2]
        except (ValueError, IndexError):
            continue
        m = re.match(r"split(\d+)_fold(\d+)", split_fold)
        if not m:
            continue
        split, fold = int(m.group(1)), int(m.group(2))
        for variant in ("early", "late", "middle", "mario_kempes", "set_mil_mt", "longitudinal_mk_mt", "longitudinal_mk"):
            if variant_task.startswith(variant):
                task = variant_task[len(variant):].lstrip("_") or "mega"
                all_files.append((fpath, split, fold, variant, task))
                break

    print(f"[P2] found {len(all_files)} metrics files")

    rows = []  # for benchmark table

    for fpath, split, fold, variant, task in sorted(all_files):
        with open(fpath) as f:
            d = json.load(f)
        test = scalar_subset(d.get("test", {}))
        if not test:
            continue

        run_name = f"p2_s{split}f{fold}_{variant}_{task}"
        config = {"phase": 2, "split": split, "fold": fold, "variant": variant, "task": task}
        summary = {f"test/{k}": v for k, v in test.items()}
        pm = primary_metric(task, test)
        if pm is not None:
            summary["test/primary"] = pm
            rows.append({"split": split, "fold": fold, "variant": variant, "task": task,
                         "primary_metric": pm, "metric_name": "bacc" if "cls" in task else "c_index"})

        # unimodal ablation
        ablation = d.get("unimodal_ablation", {})
        for mod, abl_metrics in ablation.items():
            abl_scalar = scalar_subset(abl_metrics)
            for k, v in abl_scalar.items():
                summary[f"ablation/{mod}/{k}"] = v

        if dry_run:
            print(f"  [DRY] {run_name}: primary={pm:.4f}" if pm else f"  [DRY] {run_name}")
            continue

        with wandb.init(project=project, name=run_name, config=config,
                        group=f"phase2_{variant}", tags=["phase2", variant, task],
                        job_type="phase2", reinit=True) as run:
            run.summary.update(summary)

    # Upload summary table to a dedicated run
    if rows and not dry_run:
        with wandb.init(project=project, name="p2_benchmark_table",
                        job_type="summary", reinit=True) as run:
            table = wandb.Table(
                columns=["split", "fold", "variant", "task", "primary_metric", "metric_name"],
                data=[[r["split"], r["fold"], r["variant"], r["task"],
                       r["primary_metric"], r["metric_name"]] for r in rows]
            )
            run.log({"p2_results": table})
    print("[P2] done")


# ──────────────────────────────────────────────────────────────────────────────
# TCGA benchmark table
# ──────────────────────────────────────────────────────────────────────────────

def upload_tcga_benchmark(project: str, dry_run: bool):
    bench_path = REPO_ROOT / "results_tcga_multitask" / "benchmark_table.json"
    if not bench_path.exists():
        print("[TCGA] benchmark_table.json not found, skipping")
        return

    with open(bench_path) as f:
        bench = json.load(f)

    rows = []
    for model, cancers in bench.items():
        for cancer, metrics in cancers.items():
            for metric_name, (mean, std) in metrics.items():
                if mean is not None:
                    rows.append([model, cancer, metric_name, round(mean, 4), round(std, 4)])

    if dry_run:
        print(f"[TCGA] {len(rows)} rows in benchmark table")
        return

    with wandb.init(project=project, name="tcga_benchmark",
                    job_type="benchmark", tags=["tcga"], reinit=True) as run:
        table = wandb.Table(
            columns=["model", "cancer", "metric", "mean", "std"],
            data=rows
        )
        run.log({"tcga_benchmark": table})
        # Also log as summary scalars for easy filtering
        for row in rows:
            run.summary[f"{row[0]}/{row[1]}/{row[2]}/mean"] = row[3]

    print(f"[TCGA] uploaded {len(rows)} benchmark rows")


# ──────────────────────────────────────────────────────────────────────────────
# Nature paper figures
# ──────────────────────────────────────────────────────────────────────────────

def upload_nature_figures(project: str, dry_run: bool):
    fig_dir = REPO_ROOT / "analysis" / "nature_paper"
    pngs = sorted(fig_dir.glob("*.png"))
    print(f"[Nature] found {len(pngs)} figures")

    if not pngs:
        return

    if dry_run:
        for p in pngs[:3]:
            print(f"  [DRY] {p.name}")
        return

    with wandb.init(project=project, name="nature_paper_figures",
                    job_type="analysis", tags=["figures", "nature"], reinit=True) as run:
        images = {p.stem: wandb.Image(str(p), caption=p.stem) for p in pngs}
        run.log(images)

    print(f"[Nature] uploaded {len(pngs)} figures")


# ──────────────────────────────────────────────────────────────────────────────
# Interpretability figures
# ──────────────────────────────────────────────────────────────────────────────

def upload_interpretability(project: str, dry_run: bool):
    interp_dir = REPO_ROOT / "interpretability"
    pngs = sorted(interp_dir.rglob("*.png"))
    print(f"[Interp] found {len(pngs)} figures")

    if not pngs:
        return

    if dry_run:
        for p in pngs[:3]:
            print(f"  [DRY] {p.relative_to(interp_dir)}")
        return

    # Group by parent dir (one wandb.Image log per subdir)
    by_dir: dict[str, list[Path]] = {}
    for p in pngs:
        key = p.parent.name
        by_dir.setdefault(key, []).append(p)

    with wandb.init(project=project, name="interpretability_figures",
                    job_type="interpretability", tags=["interpretability"], reinit=True) as run:
        payload: dict = {}
        for subdir, imgs in by_dir.items():
            for p in imgs:
                key = f"{subdir}/{p.stem}"
                payload[key] = wandb.Image(str(p), caption=f"{subdir}/{p.stem}")
        run.log(payload)

    print(f"[Interp] uploaded {len(pngs)} figures across {len(by_dir)} subdirs")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="chicago-mil")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--phase1", action="store_true")
    parser.add_argument("--phase2", action="store_true")
    parser.add_argument("--tcga", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--interp", action="store_true")
    args = parser.parse_args()

    # If no specific flags, do everything
    do_all = not any([args.phase1, args.phase2, args.tcga, args.figures, args.interp])

    os.environ.setdefault("WANDB_SILENT", "true")

    if do_all or args.phase1:
        upload_phase1(args.project, args.dry_run)
    if do_all or args.phase2:
        upload_phase2(args.project, args.dry_run)
    if do_all or args.tcga:
        upload_tcga_benchmark(args.project, args.dry_run)
    if do_all or args.figures:
        upload_nature_figures(args.project, args.dry_run)
    if do_all or args.interp:
        upload_interpretability(args.project, args.dry_run)


if __name__ == "__main__":
    main()
