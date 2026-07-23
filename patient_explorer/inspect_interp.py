import json, numpy as np
from pathlib import Path

ROOT = Path("/ictstr01/home/aih/dinesh.haridoss/chicago_mil/interpretability/set_mil_mt_interp")

for task_dir_name, task_key in [
    ("all_splits_death_surv", "death_surv"),
    ("all_splits_clad_surv",  "clad_surv"),
    ("all_splits_cls",        "acr_cls"),
]:
    jf = ROOT / task_dir_name / "paper_interp_data.json"
    if not jf.exists():
        print(f"MISSING: {jf}"); continue
    with open(jf) as f:
        d = json.load(f)
    task = d["tasks"][task_key]
    ca = task["cluster_affinity"]
    print(f"\n{'='*60}")
    print(f"TASK: {task_key}  n_hi={task['n_hi']}  n_lo={task['n_lo']}")
    print(f"  gate_weights mean: { {k: round(v,3) for k,v in task['gate_weights']['mean'].items()} }")
    for mod in ["HE", "BAL", "CT", "Clinical"]:
        info = ca[mod]
        print(f"\n  [{mod}] keys={list(info.keys())}")
        print(f"    n_clusters={len(info['cluster_names'])}")
        print(f"    cluster_names[:10]: {info['cluster_names'][:10]}")
        for key in ["hi_mean", "lo_mean", "delta", "hi_lo_diff", "hi_lo_ratio"]:
            if key in info:
                arr = np.array(info[key])
                top3 = np.argsort(arr)[-3:][::-1]
                top3_names = [info['cluster_names'][i] for i in top3]
                print(f"    {key}: range=[{arr.min():.4f}, {arr.max():.4f}]  top3={top3_names}")
