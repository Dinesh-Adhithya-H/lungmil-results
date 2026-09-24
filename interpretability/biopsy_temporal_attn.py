"""
Biopsy temporal attention analysis for longitudinal_mk_no_alibi_v2.

For each task (acr_surv, clad_surv, death_surv), across all 5 splits:
  - Load best model checkpoint
  - Run forward on all test patients, hooking _abmil_rep to capture raw alpha
  - Sum alpha per biopsy
  - Plot: x=biopsy day, y=raw summed alpha, per patient lines + mean curves by outcome

Usage (sbatch only):
  sbatch interpretability/submit_biopsy_temporal_attn.sh
"""

import argparse, json, math, os, sys, traceback, types, warnings
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mil.models.builders import build_model_v8
from mil.data.splits import build_splits_longitudinal
from mil.data.registry import MODALITIES
from mil.data.loader import preload_bags
from shared import SPLITS_CSV, SAMPLES_DIR, RESULTS_ROOT, TASK_COLORS, TASK_LABELS

VARIANT  = "longitudinal_mk_no_alibi_v2"
OUT_ROOT = ROOT / "interpretability" / "longitudinal_mk_interp" / "biopsy_temporal_attn_v2"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

TASK_DIR = {"acr_surv": "acr_surv", "clad_surv": "clad_surv", "death_surv": "death_surv"}
TASK_BUILD = {"acr_surv": "acr_surv", "clad_surv": "clad_surv", "death_surv": "death_surv"}

# internal task key used inside the model
INTERNAL_KEY = {"acr_surv": "acr_surv", "clad_surv": "clad", "death_surv": "death"}

SURV_KEYS = {
    "acr_surv":   ("tte_next_acr",  "event_next_acr"),
    "clad_surv":  ("clad_time",     "clad_event"),
    "death_surv": ("death_time",    "death_event"),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.dpi": 120, "savefig.dpi": 150, "savefig.bbox": "tight",
})


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(split: int, task: str, device: torch.device):
    ckpt = RESULTS_ROOT / f"split{split}_fold0" / f"{VARIANT}_{TASK_DIR[task]}" \
           / f"model_{VARIANT}_final.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    for k in list(state.keys()):
        if k.endswith(".seeds") and state[k].ndim == 3:
            state[k] = state[k].squeeze(0)
    n_seeds = 16
    for k, v in state.items():
        if k.endswith(".seeds") and v.ndim == 2:
            n_seeds = v.shape[0]; break
    model = build_model_v8(variant=VARIANT, task=TASK_BUILD[task], slot_k=n_seeds)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:    print(f"  [load] missing:    {missing[:3]}")
    if unexpected: print(f"  [load] unexpected: {unexpected[:3]}")
    model.eval()
    return model.to(device)


# ── Alpha hook via register_forward_hook on abmil_w ──────────────────────────

def make_alpha_hook(target_task_internal: str, store: dict):
    """
    Hook on abmil_w[target_task_internal] Linear layer.
    abmil_w takes gate (N,H) → raw score (N,1).
    We compute softmax here to get alpha per token.
    """
    def hook(module, inp, out):
        # out shape: (N, 1) — raw attention logits
        raw = out.squeeze(-1)              # (N,)
        alpha = torch.softmax(raw, dim=0)  # (N,)
        store["alpha"] = alpha.detach().cpu().numpy()
    return hook


# ── Per-split extraction ──────────────────────────────────────────────────────

def extract_split(split: int, task: str, device: torch.device,
                  bag_cache: dict) -> List[dict]:
    print(f"  split={split}  task={task}", flush=True)
    model = load_model(split, task, device)

    # register hook on the scoring head for this task
    internal = INTERNAL_KEY[task]
    alpha_store: dict = {}
    hook_handle = model.abmil_w[internal].register_forward_hook(
        make_alpha_hook(internal, alpha_store))

    splits = build_splits_longitudinal(SAMPLES_DIR, SPLITS_CSV, fold=0, split=split)
    test_pats = splits["test"]
    t_key, e_key = SURV_KEYS[task]

    records_out = []

    with torch.no_grad():
        for pat in test_pats:
            pid = pat["patient_id"]
            stems   = pat["stems"]
            days_raw = pat["days"]  # use pre-computed days directly, like the trainer

            # build bags_list from cache
            bags_list = [{m: bag_cache.get(s, {}).get(m) for m in MODALITIES}
                         for s in stems]

            # filter out biopsies with None days
            valid_idx = [i for i, d in enumerate(days_raw)
                         if d is not None and not (isinstance(d, float) and math.isnan(d))]
            if len(valid_idx) < 2:
                continue

            bags_v = [bags_list[i] for i in valid_idx]
            days_v = [float(days_raw[i]) for i in valid_idx]
            recs_v = [pat["records"][i] for i in valid_idx
                      if i < len(pat["records"])]
            if len(recs_v) < len(days_v):
                recs_v += [{}] * (len(days_v) - len(recs_v))

            # skip if no modality data at all
            if all(all(b.get(m) is None for m in MODALITIES) for b in bags_v):
                continue

            # get event label from last record with it
            event = None
            for rec in reversed(pat["records"]):
                v = rec.get(e_key)
                if v is not None:
                    try:
                        event = int(float(v))
                        break
                    except (ValueError, TypeError):
                        pass
            if event is None:
                continue

            alpha_store.clear()
            try:
                model({"bags_list": bags_v, "days": days_v, "records": recs_v}, device)
            except Exception as e:
                print(f"    [skip] pid={pid}: {e}", flush=True)
                continue

            alpha_arr = alpha_store.get("alpha")
            if alpha_arr is None:
                continue

            # We need tok_biopsy to sum alpha per biopsy.
            # Re-run a lightweight pass to get tok_biopsy by patching _concat_meta.
            tok_biopsy_store: dict = {}
            orig_cm_func = type(model)._concat_meta

            def patched_cm(self_m, task_arg, tokens, tok_biopsy, tok_mod,
                           days_arr, anchor_day, n_biopsies, device_arg):
                if task_arg == internal:
                    tok_biopsy_store["tok_biopsy"] = list(tok_biopsy)
                return orig_cm_func(self_m, task_arg, tokens, tok_biopsy, tok_mod,
                                    days_arr, anchor_day, n_biopsies, device_arg)

            model._concat_meta = types.MethodType(patched_cm, model)
            alpha_store.clear()
            try:
                model({"bags_list": bags_v, "days": days_v, "records": recs_v}, device)
            except Exception as e:
                print(f"    [skip2] pid={pid}: {e}", flush=True)
                model._concat_meta = orig_cm_func.__get__(model, type(model))
                continue
            model._concat_meta = orig_cm_func.__get__(model, type(model))

            alpha_arr2   = alpha_store.get("alpha")
            tok_b_list   = tok_biopsy_store.get("tok_biopsy")
            if alpha_arr2 is None or tok_b_list is None:
                continue

            n_biopsies = len(days_v)
            biopsy_attn = np.zeros(n_biopsies)
            for i, b in enumerate(tok_b_list):
                if 0 <= b < n_biopsies and i < len(alpha_arr2):
                    biopsy_attn[b] += alpha_arr2[i]

            records_out.append({
                "pid":         pid,
                "split":       split,
                "days":        days_v,
                "biopsy_attn": biopsy_attn.tolist(),
                "event":       event,
            })

    hook_handle.remove()
    print(f"    → {len(records_out)} patients", flush=True)
    return records_out


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_all_tasks(all_data: Dict[str, List[dict]]):
    tasks = ["acr_surv", "clad_surv", "death_surv"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)

    for ax, task in zip(axes, tasks):
        records = all_data.get(task, [])
        color_event  = TASK_COLORS.get(task, "#e53935")
        color_censor = "#aaaaaa"

        event_days_all, event_attn_all   = [], []
        censor_days_all, censor_attn_all = [], []

        for rec in records:
            days = np.array(rec["days"])
            attn = np.array(rec["biopsy_attn"])
            order = np.argsort(days)
            days  = days[order]
            attn  = attn[order]
            if rec["event"] == 1:
                ax.plot(days, attn, color=color_event,  alpha=0.15, lw=0.8)
                event_days_all.append(days)
                event_attn_all.append(attn)
            else:
                ax.plot(days, attn, color=color_censor, alpha=0.12, lw=0.8)
                censor_days_all.append(days)
                censor_attn_all.append(attn)

        def mean_curve(days_list, attn_list, bins=np.arange(0, 1826, 90)):
            centers, means = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                vals = [a[j] for d, a in zip(days_list, attn_list)
                        for j in range(len(d)) if lo <= d[j] < hi]
                if vals:
                    means.append(np.mean(vals))
                    centers.append((lo + hi) / 2)
            return np.array(centers), np.array(means)

        if event_days_all:
            xc, ym = mean_curve(event_days_all, event_attn_all)
            ax.plot(xc, ym, color=color_event,  lw=2.5,
                    label=f"Event (n={len(event_days_all)})", zorder=5)
        if censor_days_all:
            xc, ym = mean_curve(censor_days_all, censor_attn_all)
            ax.plot(xc, ym, color=color_censor, lw=2.5,
                    label=f"Censored (n={len(censor_days_all)})", zorder=5)

        n_total = len(event_days_all) + len(censor_days_all)
        ax.set_title(f"{TASK_LABELS.get(task, task)}  (n={n_total})")
        ax.set_xlabel("Days since transplant")
        ax.set_ylabel("Raw ABMIL attention (summed per biopsy)")
        ax.legend(fontsize=9)
        ax.set_xlim(left=0)

    fig.suptitle(f"Biopsy temporal attention — {VARIANT} (all splits)", fontsize=14, y=1.02)
    plt.tight_layout()

    out_png = OUT_ROOT / "biopsy_temporal_attn_all_tasks.png"
    out_pdf = OUT_ROOT / "biopsy_temporal_attn_all_tasks.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out_png}", flush=True)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", type=int, default=[0,1,2,3,4])
    parser.add_argument("--tasks",  nargs="+",
                        default=["acr_surv", "clad_surv", "death_surv"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # preload bag cache once for all splits/tasks
    print("Preloading bag cache...", flush=True)
    splits_union = build_splits_longitudinal(SAMPLES_DIR, SPLITS_CSV, fold=0, split=0)
    all_stems: set = set()
    for split in args.splits:
        for s in [0,1,2,3,4]:
            try:
                sp = build_splits_longitudinal(SAMPLES_DIR, SPLITS_CSV, fold=0, split=s)
                for pats in sp.values():
                    for pat in pats:
                        all_stems.update(pat["stems"])
            except Exception:
                pass
    bag_cache = preload_bags(list(all_stems), SAMPLES_DIR, n_workers=4)
    print(f"  Loaded {len(bag_cache)} stems", flush=True)

    all_data: Dict[str, List[dict]] = {t: [] for t in args.tasks}

    for task in args.tasks:
        for split in args.splits:
            try:
                recs = extract_split(split, task, device, bag_cache)
                all_data[task].extend(recs)
            except Exception as e:
                traceback.print_exc()
                print(f"  [error] split={split} task={task}: {e}", flush=True)

    raw_out = OUT_ROOT / "biopsy_attn_raw.json"
    with open(raw_out, "w") as f:
        json.dump(all_data, f)
    print(f"Raw data → {raw_out}", flush=True)

    for task, recs in all_data.items():
        print(f"  {task}: {len(recs)} patients total", flush=True)

    plot_all_tasks(all_data)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
