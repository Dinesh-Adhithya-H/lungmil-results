"""
Biopsy temporal attention — histogram/bar version.

4 tasks: acr_cls (LongMK), acr_surv, clad_surv, death_surv (LongMK-v2).
Per task: binned bar chart (event vs non-event), individual biopsy dots,
error bars = ±1 std, plus a smooth mean curve overlay.

Usage (sbatch only):
  sbatch interpretability/submit_biopsy_temporal_attn_hist.sh
"""

import argparse, json, math, os, sys, traceback, types, warnings
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

OUT_ROOT = ROOT / "interpretability" / "longitudinal_mk_interp" / "biopsy_temporal_attn_v2"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# Per-task config: variant, task dir suffix, build task arg, internal key, event key
TASK_CFG = {
    "acr_cls":    dict(variant="longitudinal_mk_no_alibi_v2", task_dir="cls",       build="cls",
                       internal="acr_cls", event_key="label",        label="ACR Classification"),
    "acr_surv":   dict(variant="longitudinal_mk_no_alibi_v2", task_dir="acr_surv",  build="acr_surv",
                       internal="acr_surv", event_key="event_next_acr", label="ACR Survival"),
    "clad_surv":  dict(variant="longitudinal_mk_no_alibi_v2", task_dir="clad_surv", build="clad_surv",
                       internal="clad",     event_key="clad_event",    label="CLAD Survival"),
    "death_surv": dict(variant="longitudinal_mk_no_alibi_v2", task_dir="death_surv",build="death_surv",
                       internal="death",    event_key="death_event",   label="Death Survival"),
}

TASK_COLOR_EVENT  = {"acr_cls": "#7c83ff", "acr_surv": "#ef5350", "clad_surv": "#f78166", "death_surv": "#3fb950"}
TASK_COLOR_CENSOR = "#aaaaaa"

# Time bins: 0–3m, 3–6m, 6–12m, 1–2y, 2–3y, 3–5y
BIN_EDGES   = [0, 90, 180, 365, 730, 1095, 1826]
BIN_LABELS  = ["0–3 mo", "3–6 mo", "6–12 mo", "1–2 yr", "2–3 yr", "3–5 yr"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.2, "grid.linewidth": 0.5,
    "figure.dpi": 120, "savefig.dpi": 150,
})


# ── Model loading ─────────────────────────────────────────────────────────────

def _detect_abmil_in(state: dict) -> int:
    for k, v in state.items():
        if "abmil_V" in k and ".0.weight" in k:
            return v.shape[1]
    return None


def _patch_old_abmil(model, abmil_in: int, hidden_dim: int = 256):
    """
    For old checkpoints where abmil_in < current model's abmil_in:
    Replace ABMIL heads with correct dims, then register pre-hooks on abmil_V/U
    to truncate tokens to abmil_in dims regardless of what _concat_meta produces.
    This is robust to any internal method-call ordering.
    """
    def _V(d, h):
        return nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(),
                              nn.Linear(h, h), nn.Tanh())
    def _U(d, h):
        return nn.Sequential(nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(),
                              nn.Linear(h, h), nn.Sigmoid())

    tasks = list(model.abmil_V.keys())
    model.abmil_V = nn.ModuleDict({t: _V(abmil_in, hidden_dim) for t in tasks})
    model.abmil_U = nn.ModuleDict({t: _U(abmil_in, hidden_dim) for t in tasks})
    model.abmil_w = nn.ModuleDict({t: nn.Linear(hidden_dim, 1, bias=False) for t in tasks})

    # truncate token dim at the ABMIL head input — robust regardless of _concat_meta patching
    def _trunc_hook(m, inp):
        return (inp[0][:, :abmil_in],)

    for t in tasks:
        model.abmil_V[t].register_forward_pre_hook(_trunc_hook)
        model.abmil_U[t].register_forward_pre_hook(_trunc_hook)


def load_model(split: int, cfg: dict, device: torch.device):
    variant  = cfg["variant"]
    task_dir = cfg["task_dir"]
    ckpt = RESULTS_ROOT / f"split{split}_fold0" / f"{variant}_{task_dir}" \
           / f"model_{variant}_final.pt"
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
    model = build_model_v8(variant=variant, task=cfg["build"], slot_k=n_seeds)

    # detect if checkpoint uses old 256-dim ABMIL (before mod/time concat was added)
    ckpt_abmil_in = _detect_abmil_in(state)
    if ckpt_abmil_in is not None and ckpt_abmil_in != 264:
        print(f"  [compat] patching ABMIL to {ckpt_abmil_in}-dim (checkpoint is old format)", flush=True)
        _patch_old_abmil(model, ckpt_abmil_in)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:    print(f"  [load] missing:    {missing[:3]}")
    if unexpected: print(f"  [load] unexpected: {unexpected[:3]}")
    model.eval()
    return model.to(device)


# ── Attention hook — raw logits, no normalization ─────────────────────────────

def make_alpha_hook(store: dict):
    def hook(module, inp, out):
        raw = out.squeeze(-1)          # (N,) raw attention logits, no softmax
        store["alpha"] = raw.detach().cpu().numpy()
    return hook


# ── Per-split extraction ──────────────────────────────────────────────────────

def extract_split(split: int, task: str, cfg: dict, device: torch.device,
                  bag_cache: dict) -> List[dict]:
    print(f"  split={split}  task={task}", flush=True)
    model    = load_model(split, cfg, device)
    internal = cfg["internal"]
    event_key = cfg["event_key"]

    alpha_store: dict = {}
    hook_handle = model.abmil_w[internal].register_forward_hook(make_alpha_hook(alpha_store))

    splits   = build_splits_longitudinal(SAMPLES_DIR, SPLITS_CSV, fold=0, split=split)
    test_pats = splits["test"]

    records_out = []
    with torch.no_grad():
        for pat in test_pats:
            pid      = pat["patient_id"]
            stems    = pat["stems"]
            days_raw = pat["days"]

            bags_list = [{m: bag_cache.get(s, {}).get(m) for m in MODALITIES} for s in stems]

            valid_idx = [i for i, d in enumerate(days_raw)
                         if d is not None and not (isinstance(d, float) and math.isnan(d))]
            if len(valid_idx) < 2:
                continue

            bags_v = [bags_list[i] for i in valid_idx]
            days_v = [float(days_raw[i]) for i in valid_idx]
            recs_v = [pat["records"][i] for i in valid_idx if i < len(pat["records"])]
            if len(recs_v) < len(days_v):
                recs_v += [{}] * (len(days_v) - len(recs_v))

            if all(all(b.get(m) is None for m in MODALITIES) for b in bags_v):
                continue

            # resolve event label
            # for acr_cls: patient is positive if ANY biopsy is positive (ever-ACR)
            # for survival tasks: take last available value (time-to-event label)
            event = None
            if event_key == "label":
                # acr_cls: ever positive
                vals = []
                for rec in pat["records"]:
                    v = rec.get(event_key)
                    if v is not None:
                        try: vals.append(int(float(v)))
                        except (ValueError, TypeError): pass
                if vals:
                    event = 1 if any(v == 1 for v in vals) else 0
            else:
                for rec in reversed(pat["records"]):
                    v = rec.get(event_key)
                    if v is not None:
                        try: event = int(float(v)); break
                        except (ValueError, TypeError): pass
            if event is None:
                continue

            # first forward: get alpha
            alpha_store.clear()
            try:
                model({"bags_list": bags_v, "days": days_v, "records": recs_v}, device)
            except Exception as e:
                print(f"    [skip] pid={pid}: {e}", flush=True)
                continue

            alpha_arr = alpha_store.get("alpha")
            if alpha_arr is None:
                continue

            # second forward: capture tok_biopsy via _concat_meta patch
            tok_biopsy_store: dict = {}
            orig_cm = type(model)._concat_meta

            def patched_cm(self_m, task_arg, tokens, tok_biopsy, tok_mod,
                           days_arr, anchor_day, n_biopsies, device_arg):
                if task_arg == internal:
                    tok_biopsy_store["tok_biopsy"] = list(tok_biopsy)
                return orig_cm(self_m, task_arg, tokens, tok_biopsy, tok_mod,
                               days_arr, anchor_day, n_biopsies, device_arg)

            model._concat_meta = types.MethodType(patched_cm, model)
            alpha_store.clear()
            try:
                model({"bags_list": bags_v, "days": days_v, "records": recs_v}, device)
            except Exception as e:
                print(f"    [skip2] pid={pid}: {e}", flush=True)
                model._concat_meta = orig_cm.__get__(model, type(model))
                continue
            model._concat_meta = orig_cm.__get__(model, type(model))

            alpha_arr2 = alpha_store.get("alpha")
            tok_b_list = tok_biopsy_store.get("tok_biopsy")
            if alpha_arr2 is None or tok_b_list is None:
                continue

            n_biopsies = len(days_v)
            # one entry per token: (day_of_biopsy, raw_attn_logit)
            token_days  = []
            token_attn  = []
            for i, b in enumerate(tok_b_list):
                if 0 <= b < n_biopsies and i < len(alpha_arr2):
                    token_days.append(days_v[b])
                    token_attn.append(float(alpha_arr2[i]))

            if not token_days:
                continue

            records_out.append({
                "pid":        pid,
                "split":      split,
                "token_days": token_days,   # one per token
                "token_attn": token_attn,   # raw logit per token
                "event":      event,
            })

    hook_handle.remove()
    print(f"    → {len(records_out)} patients", flush=True)
    return records_out


# ── Plotting ──────────────────────────────────────────────────────────────────

def bin_tokens(records, event_val):
    """Return per-bin list of raw token attention scores for patients with event==event_val."""
    bins = [[] for _ in BIN_LABELS]
    for rec in records:
        if rec["event"] != event_val:
            continue
        for d, a in zip(rec["token_days"], rec["token_attn"]):
            for bi, (lo, hi) in enumerate(zip(BIN_EDGES[:-1], BIN_EDGES[1:])):
                if lo <= d < hi:
                    bins[bi].append(a)
                    break
    return bins


def plot_all_tasks(all_data: Dict[str, List[dict]]):
    tasks = ["acr_cls", "acr_surv", "clad_surv", "death_surv"]
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharey=False)
    fig.patch.set_facecolor("white")

    bar_w = 0.35
    x     = np.arange(len(BIN_LABELS))

    for ax, task in zip(axes, tasks):
        records    = all_data.get(task, [])
        col_event  = TASK_COLOR_EVENT[task]
        col_censor = TASK_COLOR_CENSOR
        cfg        = TASK_CFG[task]

        lbl_event, lbl_censor = ("ACR event", "No ACR") if task == "acr_cls" else ("Event", "Censored")

        n_event  = sum(1 for r in records if r["event"] == 1)
        n_censor = sum(1 for r in records if r["event"] == 0)

        # token-level bins (all raw attention scores, one per token)
        ev_bins  = bin_tokens(records, 1)
        ce_bins  = bin_tokens(records, 0)

        ev_means = np.array([np.mean(b) if b else np.nan for b in ev_bins])
        ev_stds  = np.array([np.std(b)  if b else np.nan for b in ev_bins])
        ce_means = np.array([np.mean(b) if b else np.nan for b in ce_bins])
        ce_stds  = np.array([np.std(b)  if b else np.nan for b in ce_bins])

        # bars (mean over all tokens in bin)
        ax.bar(x - bar_w/2, ev_means, bar_w, color=col_event,  alpha=0.70, zorder=3,
               label=f"{lbl_event} (n={n_event} pts)")
        ax.bar(x + bar_w/2, ce_means, bar_w, color=col_censor, alpha=0.70, zorder=3,
               label=f"{lbl_censor} (n={n_censor} pts)")

        # error bars
        for i in range(len(BIN_LABELS)):
            if not np.isnan(ev_means[i]) and ev_stds[i] > 0:
                ax.errorbar(x[i] - bar_w/2, ev_means[i], yerr=ev_stds[i],
                            fmt="none", ecolor="#333", elinewidth=1.0, capsize=3, zorder=5)
            if not np.isnan(ce_means[i]) and ce_stds[i] > 0:
                ax.errorbar(x[i] + bar_w/2, ce_means[i], yerr=ce_stds[i],
                            fmt="none", ecolor="#555", elinewidth=1.0, capsize=3, zorder=5)

        # ALL individual token dots — jittered within each bar
        rng = np.random.default_rng(42)
        for bi, b_vals in enumerate(ev_bins):
            if b_vals:
                jitter = rng.uniform(-bar_w * 0.45, bar_w * 0.45, len(b_vals))
                ax.scatter(x[bi] - bar_w/2 + jitter, b_vals,
                           s=2, color=col_event, alpha=0.15, zorder=4, linewidths=0)
        for bi, b_vals in enumerate(ce_bins):
            if b_vals:
                jitter = rng.uniform(-bar_w * 0.45, bar_w * 0.45, len(b_vals))
                ax.scatter(x[bi] + bar_w/2 + jitter, b_vals,
                           s=2, color=col_censor, alpha=0.12, zorder=4, linewidths=0)

        # token count annotation
        for bi in range(len(BIN_LABELS)):
            n_e = len(ev_bins[bi])
            n_c = len(ce_bins[bi])
            if n_e:
                ax.text(x[bi] - bar_w/2, (ev_means[bi] or 0) + (ev_stds[bi] or 0) * 0.1 + 0.002,
                        f"n={n_e}", ha="center", va="bottom", fontsize=5.5, color=col_event, zorder=6)
            if n_c:
                ax.text(x[bi] + bar_w/2, (ce_means[bi] or 0) + (ce_stds[bi] or 0) * 0.1 + 0.002,
                        f"n={n_c}", ha="center", va="bottom", fontsize=5.5, color="#777", zorder=6)

        ax.set_xticks(x)
        ax.set_xticklabels(BIN_LABELS, rotation=30, ha="right", fontsize=8)
        ax.set_xlabel("Time since transplant", fontsize=9)
        ax.set_ylabel("Raw ABMIL attention logit (per token)", fontsize=8.5)
        ax.set_title(f"{cfg['label']}\n(n={len(records)} patients)", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.8)
        ax.set_facecolor("#FAFAFA")
        ax.yaxis.grid(True, alpha=0.2, linewidth=0.5)
        ax.xaxis.grid(False)

        # variant label
        variant_short = "LongMK" if "v2" not in cfg["variant"] else "LongMK-v2"
        ax.text(0.98, 0.97, variant_short, transform=ax.transAxes,
                ha="right", va="top", fontsize=7.5, color="#666",
                style="italic")

    fig.suptitle(
        "Biopsy temporal attention by outcome — binned by time since transplant\n"
        "Bars = mean ± s.d. · dots = individual biopsies · n above bars = biopsy count",
        fontsize=11, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    out_png = OUT_ROOT / "biopsy_temporal_attn_histogram.png"
    out_pdf = OUT_ROOT / "biopsy_temporal_attn_histogram.pdf"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
    print(f"  Saved → {out_png}", flush=True)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", type=int, default=[0,1,2,3,4])
    parser.add_argument("--tasks",  nargs="+",
                        default=["acr_cls", "acr_surv", "clad_surv", "death_surv"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Preloading bag cache...", flush=True)
    all_stems: set = set()
    for s in range(5):
        try:
            sp = build_splits_longitudinal(SAMPLES_DIR, SPLITS_CSV, fold=0, split=s)
            for pats in sp.values():
                for pat in pats:
                    all_stems.update(pat["stems"])
        except Exception:
            pass
    bag_cache = preload_bags(list(all_stems), SAMPLES_DIR, n_workers=4)
    print(f"  Loaded {len(bag_cache)} stems", flush=True)

    # try to reuse existing raw data for tasks already extracted
    raw_path = OUT_ROOT / "biopsy_attn_raw.json"
    if raw_path.exists():
        print(f"Loading existing raw data from {raw_path}", flush=True)
        with open(raw_path) as f:
            all_data = json.load(f)
        # convert lists back to native types (already fine)
    else:
        all_data = {}

    for task in args.tasks:
        if task not in all_data:
            all_data[task] = []
        cfg = TASK_CFG[task]
        for split in args.splits:
            # skip if we already have data for this split+task
            if any(r["split"] == split for r in all_data[task]):
                print(f"  [skip existing] split={split} task={task}", flush=True)
                continue
            try:
                recs = extract_split(split, task, cfg, device, bag_cache)
                all_data[task].extend(recs)
            except Exception as e:
                traceback.print_exc()
                print(f"  [error] split={split} task={task}: {e}", flush=True)

    with open(raw_path, "w") as f:
        json.dump(all_data, f)
    print(f"Raw data → {raw_path}", flush=True)

    for task, recs in all_data.items():
        print(f"  {task}: {len(recs)} patients total", flush=True)

    plot_all_tasks(all_data)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
