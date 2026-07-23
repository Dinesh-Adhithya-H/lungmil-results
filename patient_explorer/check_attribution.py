"""Check biological correctness of feature attribution data."""
import json, numpy as np
from pathlib import Path

ROOT = Path("/ictstr01/home/aih/dinesh.haridoss/chicago_mil/interpretability/set_mil_mt_interp")

for task_dir, task_key, label in [
    ("all_splits_death_surv", "death_surv",  "DEATH SURVIVAL"),
    ("all_splits_clad_surv",  "clad_surv",   "CLAD SURVIVAL"),
    ("all_splits_cls",        "acr_cls",     "ACR CLASSIFICATION"),
]:
    with open(ROOT / task_dir / "paper_interp_data.json") as f:
        d = json.load(f)
    task = d["tasks"][task_key]
    ca = task["cluster_affinity"]

    print(f"\n{'='*65}")
    print(f"  {label}  (n_hi={task['n_hi']}, n_lo={task['n_lo']})")
    print(f"{'='*65}")

    for mod in ["Clinical", "BAL", "HE", "CT"]:
        info = ca[mod]
        names = info["cluster_names"]
        hi = np.array(info["hi_score"])
        lo = np.array(info["lo_score"])
        delta = np.array(info["delta"])

        order = np.argsort(delta)[::-1]  # descending
        top10_pos = [(names[i], delta[i], hi[i], lo[i]) for i in order[:10]]
        top5_neg  = [(names[i], delta[i], hi[i], lo[i]) for i in order[-5:]]

        print(f"\n  [{mod}]  range Δ=[{delta.min():.5f}, {delta.max():.5f}]")
        print(f"  High-risk enriched (top 10 by Δ):")
        for n, d_, h, l in top10_pos:
            bar = "█" * max(1, int(abs(d_) / delta.max() * 20)) if delta.max() > 0 else ""
            print(f"    {n:<55}  Δ={d_:+.5f}  hi={h:.4f}  lo={l:.4f}  {bar}")
        print(f"  Low-risk enriched (bottom 5 by Δ):")
        for n, d_, h, l in top5_neg:
            bar = "█" * max(1, int(abs(d_) / abs(delta.min()) * 20)) if delta.min() < 0 else ""
            print(f"    {n:<55}  Δ={d_:+.5f}  hi={h:.4f}  lo={l:.4f}  {bar}")
