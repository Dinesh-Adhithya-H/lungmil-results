#!/bin/bash
# upload_interp_wandb.sh
# Convert set_mil_mt PDFs → PNG and upload all 16 interp runs to W&B.
# Early/late/middle already uploaded; this re-logs them + fixes set_mil_mt.
#SBATCH --job-name=interp_wandb
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4
#SBATCH --mem=16G --time=00:30:00
#SBATCH --partition=cpu_p --qos=cpu_normal
#SBATCH --output=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%j_interp_wandb.out
#SBATCH --error=/ictstr01/home/aih/dinesh.haridoss/chicago_mil/results/mm_abmil_v8/slurm_logs/%j_interp_wandb.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=dinesh.haridoss@helmholtz-munich.de

set -euo pipefail
export PYTHONUNBUFFERED=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate chicago

python3 -u - <<'PYEOF'
import wandb
import numpy as np
from pathlib import Path

INTERP_BASE = Path("/ictstr01/home/aih/dinesh.haridoss/chicago_mil/interpretability/v8_interp_s0f0")
PROJECT = "chicago-mil-interpretability"

def pdf_to_png(pdf_path):
    """Convert first page of PDF to PNG, return PNG path."""
    png_path = pdf_path.with_suffix(".png")
    if png_path.exists():
        return png_path
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
        pix.save(str(png_path))
        doc.close()
        return png_path
    except ImportError:
        pass
    try:
        import subprocess
        subprocess.run(
            ["pdftoppm", "-r", "150", "-png", "-singlefile", str(pdf_path),
             str(pdf_path.with_suffix(""))],
            check=True, capture_output=True
        )
        candidate = pdf_path.parent / (pdf_path.stem + "-1.png")
        if candidate.exists():
            candidate.rename(png_path)
            return png_path
    except Exception:
        pass
    try:
        import subprocess
        subprocess.run(
            ["/usr/bin/convert", "-density", "150", f"{pdf_path}[0]", str(png_path)],
            check=True, capture_output=True
        )
        if png_path.exists():
            return png_path
    except Exception:
        pass
    return None

def collect_images(out_dir):
    """Return list of (path, caption) for all plots, converting PDFs → PNG."""
    imgs = []
    for p in sorted(out_dir.glob("*.pdf")) + sorted(out_dir.glob("*.png")):
        if p.suffix == ".pdf":
            png = pdf_to_png(p)
            if png:
                imgs.append((png, p.name.replace(".pdf", "")))
        else:
            imgs.append((p, p.name))
    for subdir in ("cluster_abmil", "cluster_slot", "cluster_xmodal",
                   "cluster_connection", "cluster_self_attn_coattn",
                   "clinical_cluster_attn", "per_sample"):
        sd = out_dir / subdir
        if sd.is_dir():
            for p in sorted(sd.glob("*.png"))[:20] + sorted(sd.glob("*.pdf"))[:10]:
                if p.suffix == ".pdf":
                    png = pdf_to_png(p)
                    if png:
                        imgs.append((png, f"{subdir}/{p.stem}"))
                else:
                    imgs.append((p, f"{subdir}/{p.name}"))
    return imgs

# ── set_mil_mt: 4 tasks ────────────────────────────────────────────────────
smmt_tasks = ["cls", "acr_surv", "clad_surv", "death_surv"]
for variant in smmt_tasks:
    out_dir = INTERP_BASE / f"set_mil_mt_{variant}"
    if not out_dir.exists():
        print(f"[skip] {out_dir} not found")
        continue

    print(f"\n--- set_mil_mt_{variant} ---")
    run = wandb.init(
        project=PROJECT,
        name=f"set_mil_mt_{variant}_split0_fold0",
        group="set_mil_mt",
        config={"variant": variant, "split": 0, "fold": 0},
        reinit=True,
    )

    # Load raw results for scalar metrics
    raw_npy = out_dir / "results_raw.npy"
    if raw_npy.exists():
        results = np.load(str(raw_npy), allow_pickle=True).tolist()
        wandb.log({"n_samples": len(results)})

    # Convert + log all plots
    imgs = collect_images(out_dir)
    print(f"  Found {len(imgs)} images/plots")
    if imgs:
        wandb.log({"panels": [wandb.Image(str(p), caption=cap) for p, cap in imgs]})

    run.finish()
    print(f"  W&B: {run.url}")

# ── early / late / middle: re-log with subdir plots (they may have been missed) ──
arch_tasks = [
    ("early",  "cls"),       ("early",  "acr_surv"),
    ("early",  "clad_surv"), ("early",  "death_surv"),
    ("late",   "cls"),       ("late",   "acr_surv"),
    ("late",   "clad_surv"), ("late",   "death_surv"),
    ("middle", "cls"),       ("middle", "acr_surv"),
    ("middle", "clad_surv"), ("middle", "death_surv"),
]
for arch, task in arch_tasks:
    out_dir = INTERP_BASE / f"{arch}_{task}"
    if not out_dir.exists():
        print(f"[skip] {out_dir}")
        continue

    print(f"\n--- {arch}_{task} ---")
    run = wandb.init(
        project=PROJECT,
        name=f"{arch}_split0_fold0_{task}",
        group=arch,
        config={"variant": arch, "task": task, "split": 0, "fold": 0},
        reinit=True,
    )

    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        import json
        summary = json.loads(summary_path.read_text())
        log_dict = {k: v for k, v in summary.items() if isinstance(v, (int, float))}
        wandb.log(log_dict)

    imgs = collect_images(out_dir)
    print(f"  Found {len(imgs)} images")
    if imgs:
        wandb.log({"plots": [wandb.Image(str(p), caption=cap) for p, cap in imgs[:60]]})

    run.finish()
    print(f"  W&B: {run.url}")

print("\n=== All W&B uploads complete ===")
PYEOF
