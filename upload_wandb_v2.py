"""
Upload longitudinal_mk_no_alibi_v2 test results to wandb.
Creates one run per split×task with final test metrics,
then updates the live benchmark table.

Usage (sbatch only):
  sbatch results_mm_abmil_v8/job_scripts/upload_wandb_v2.sh
"""

import json, re, sys
from pathlib import Path

import wandb

VARIANT      = "longitudinal_mk_no_alibi_v2"
PROJECT      = "chicago-mil"
RESULTS_ROOT = Path("/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8")

TASK_MAP = {
    "acr_surv":  {"surv_key": "acr_surv",  "ci_key": "c_index",  "label": "ACR Survival"},
    "clad_surv": {"surv_key": "clad",       "ci_key": "c_index",  "label": "CLAD Survival"},
    "death_surv":{"surv_key": "death",      "ci_key": "c_index",  "label": "Death Survival"},
    "cls":       {"surv_key": None,         "ci_key": None,       "label": "ACR Classification"},
}


def upload_run(split: int, task: str, metrics: dict):
    test = metrics.get("test", {})

    # extract per-task metrics
    acr_cls  = test.get("acr_cls",  {})
    acr_surv = test.get("acr_surv", {})
    clad     = test.get("clad",     {})
    death    = test.get("death",    {})

    summary = {
        "final/test_bacc":      acr_cls.get("bacc",    float("nan")),
        "final/test_auc":       acr_cls.get("auc",     float("nan")),
        "final/test_ci_acr":    acr_surv.get("c_index", float("nan")),
        "final/test_ci_clad":   clad.get("c_index",    float("nan")),
        "final/test_ci_death":  death.get("c_index",   float("nan")),
        "final/val_score":      test.get("val_score",  float("nan")),
    }

    run_name = f"{VARIANT}_s{split}_f0_{task}"
    print(f"  Uploading: {run_name}")
    with wandb.init(
        project=PROJECT,
        name=run_name,
        group=f"split{split}",
        tags=[VARIANT, f"split{split}", task, "test_metrics"],
        config={
            "variant": VARIANT, "split": split, "fold": 0,
            "task": task,
        },
        job_type="test_upload",
        reinit="finish_previous",
    ) as run:
        run.summary.update(summary)
        # also log as a step so it appears in charts
        run.log(summary)
    print(f"    → {summary}")


def update_benchmark_table():
    rows = []
    for fpath in sorted(RESULTS_ROOT.glob("metrics_split*_fold0_*.json")):
        stem = fpath.stem
        m = re.match(r"metrics_split(\d+)_fold0_(.+)", stem)
        if not m:
            continue
        split = int(m.group(1))
        rest  = m.group(2)

        # parse variant + task from filename
        variant, task = rest, "unknown"
        for v in ("longitudinal_mk_no_alibi_v2", "longitudinal_mk_no_alibi",
                  "longitudinal_mk_mt_no_alibi", "longitudinal_mk_mt",
                  "longitudinal_mk", "set_mil_mt_no_sab", "set_mil_mt",
                  "set_mil_no_sab", "set_mil",
                  "early", "late", "middle"):
            if rest.startswith(v):
                variant = v
                task = rest[len(v):].lstrip("_") or "mega"
                break

        with open(fpath) as f:
            d = json.load(f)
        t = d.get("test", {})

        # prefer task-specific metric
        ci_acr   = t.get("acr_surv",  {}).get("c_index", t.get("c_index"))
        ci_clad  = t.get("clad",      {}).get("c_index")
        ci_death = t.get("death",     {}).get("c_index")
        bacc     = t.get("acr_cls",   {}).get("bacc",    t.get("bacc"))

        rows.append([split, variant, task,
                     round(ci_acr   or 0, 4),
                     round(ci_clad  or 0, 4),
                     round(ci_death or 0, 4),
                     round(bacc     or 0, 4)])

    if not rows:
        print("  No rows found for benchmark table.")
        return

    print(f"\nUpdating benchmark table with {len(rows)} rows...")
    with wandb.init(
        project=PROJECT,
        name="live_benchmark_table",
        id="live_benchmark_table",
        resume="allow",
        job_type="benchmark",
        reinit="finish_previous",
    ) as run:
        table = wandb.Table(
            columns=["split", "variant", "task",
                     "ci_acr", "ci_clad", "ci_death", "bacc"],
            data=rows
        )
        run.log({"benchmark": table, "n_completed": len(rows)})
    print("  Benchmark table updated.")


def main():
    tasks   = ["acr_surv", "clad_surv", "cls", "death_surv"]
    splits  = [0, 1, 2, 3, 4]

    print(f"Uploading {VARIANT} results to wandb project '{PROJECT}'")

    for split in splits:
        for task in tasks:
            fname = RESULTS_ROOT / f"metrics_split{split}_fold0_{VARIANT}_{task}.json"
            if not fname.exists():
                print(f"  [MISSING] {fname.name}")
                continue
            with open(fname) as f:
                metrics = json.load(f)
            upload_run(split, task, metrics)

    update_benchmark_table()
    print("\nAll done.")


if __name__ == "__main__":
    main()
