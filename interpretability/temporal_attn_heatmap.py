"""
Temporal attention heatmap for LongMIL (longitudinal_mk_no_alibi, full cohort).

Hooks _abmil_rep to capture ABMIL alpha per anchor call, using the same pattern
as biopsy_temporal_attn.py (register_forward_hook on abmil_w + _concat_meta patch
for tok_biopsy), extended to record every (anchor_day, sample_days, alpha) tuple.

Heatmap axes:
  x = anchor_day  (biopsy being predicted, days post-Tx)
  y = sample_day  (attended biopsy, days post-Tx)
  value = mean summed-alpha over all prediction events in that bin

For ACR cls and ACR surv, train+val and test, all 5 splits.
Output: figures/interpretability/temporal_attn_heatmap/

Usage: sbatch interpretability/submit_temporal_attn_heatmap.sh
"""
import math, sys, types, warnings
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mil.models.builders import build_model_v8
from mil.data.splits import build_splits_longitudinal
from mil.data.registry import MODALITIES
from mil.data.loader import preload_bags
from interpretability.shared import SPLITS_CSV, SAMPLES_DIR, RESULTS_ROOT

VARIANT  = "longitudinal_mk_no_alibi_v2"
OUT_ROOT = ROOT / "figures" / "interpretability" / "temporal_attn_heatmap"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "cls":       {"label": "ACR classification",  "internal": "acr_cls"},
    "acr_surv":  {"label": "Time to first ACR",   "internal": "acr_surv"},
    "clad_surv": {"label": "Time to CLAD",        "internal": "clad"},
    "death_surv":{"label": "Time to death",       "internal": "death"},
}

FS_TITLE  = 18
FS_LABEL  = 14
FS_TICK   = 12

DAY_BINS = np.arange(0, 1826, 90)
N_BINS   = len(DAY_BINS) - 1
BG = "#FAF6F2"


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(split, task_suffix, device):
    ckpt = RESULTS_ROOT / f"split{split}_fold0" / \
           f"{VARIANT}_{task_suffix}" / f"model_{VARIANT}_final.pt"
    if not ckpt.exists():
        raise FileNotFoundError(str(ckpt))
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
    model = build_model_v8(variant=VARIANT, task=task_suffix, slot_k=n_seeds)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device)


# ── Collect (anchor_day, biopsy_days, biopsy_alpha) records ──────────────────

def collect_records(split, task_suffix, internal_key, split_set, device, bag_cache):
    """
    For each patient, for each anchor biopsy:
      - capture raw abmil_w output via register_forward_hook
      - patch _concat_meta to capture tok_biopsy for that anchor call
      - sum alpha over K tokens per biopsy index
      - record (anchor_day, sample_days, alpha_per_biopsy)
    """
    model = load_model(split, task_suffix, device)

    # State shared between hook and _abmil_rep patch
    hook_store: dict = {}   # {"raw": tensor (N,)}
    meta_store: dict = {}   # {"tok_biopsy": list, "days_arr": list, "anchor_day": float}

    # Hook on abmil_w to capture raw logits before softmax
    def alpha_hook(module, inp, out):
        hook_store["raw"] = out.squeeze(-1).detach().cpu()   # (N,)

    hook_handle = model.abmil_w[internal_key].register_forward_hook(alpha_hook)

    # Patch _concat_meta to capture tok_biopsy and anchor_day per call.
    orig_cm = type(model)._concat_meta

    def patched_cm(self, task, tokens, tok_biopsy, tok_mod, days_arr, anchor_day, n_biopsies, dev):
        if task == internal_key:
            meta_store["tok_biopsy"] = list(tok_biopsy)
            meta_store["days_arr"]   = list(days_arr)
            meta_store["anchor_day"] = float(anchor_day)
        return orig_cm(self, task, tokens, tok_biopsy, tok_mod, days_arr, anchor_day, n_biopsies, dev)

    model._concat_meta = types.MethodType(patched_cm, model)

    # Patch _abmil_rep to capture one record per anchor call
    orig_abmil = type(model)._abmil_rep
    records_out = []

    def patched_abmil(self, task, tokens, days, anchor_day):
        out = orig_abmil(self, task, tokens, days, anchor_day)
        if task == internal_key:
            raw = hook_store.get("raw")
            tb  = meta_store.get("tok_biopsy")
            da  = meta_store.get("days_arr")
            ad  = meta_store.get("anchor_day", float(anchor_day))
            if raw is not None and tb is not None and da is not None:
                alpha = torch.softmax(raw, dim=0).numpy()
                n_biopsies = len(da)
                biopsy_alpha = np.zeros(n_biopsies)
                for i, b in enumerate(tb):
                    if 0 <= b < n_biopsies and i < len(alpha):
                        biopsy_alpha[b] += alpha[i]
                records_out.append({
                    "anchor_day":  ad,
                    "sample_days": [float(da[b]) for b in range(n_biopsies)],
                    "alpha":       biopsy_alpha.tolist(),
                })
        return out

    model._abmil_rep = types.MethodType(patched_abmil, model)

    with torch.no_grad():
        for pat in split_set:
            days_raw  = pat["days"]
            bags_list = [{m: bag_cache.get(s, {}).get(m) for m in MODALITIES}
                         for s in pat["stems"]]
            valid_idx = [i for i, d in enumerate(days_raw)
                         if d is not None and not (isinstance(d, float) and math.isnan(d))]
            if len(valid_idx) < 2:
                continue
            bags_v = [bags_list[i] for i in valid_idx]
            days_v = [float(days_raw[i]) for i in valid_idx]
            recs_v = [pat["records"][i] for i in valid_idx
                      if i < len(pat.get("records", []))]
            if len(recs_v) < len(days_v):
                recs_v += [{}] * (len(days_v) - len(recs_v))
            if all(all(b.get(m) is None for m in MODALITIES) for b in bags_v):
                continue
            try:
                model({"bags_list": bags_v, "days": days_v, "records": recs_v}, device)
            except Exception as e:
                print(f"    [skip] {pat.get('patient_id','?')}: {e}", flush=True)

    hook_handle.remove()
    del model._abmil_rep
    del model._concat_meta
    return records_out


# ── Build heatmap ─────────────────────────────────────────────────────────────

def build_heatmap(records):
    """
    mean_mat[i, j] = mean alpha assigned to sample_day bin j
                     when anchor_day is in bin i.
    """
    S = np.zeros((N_BINS, N_BINS), dtype=np.float64)
    C = np.zeros((N_BINS, N_BINS), dtype=np.int64)
    for rec in records:
        ab = np.clip(np.digitize(rec["anchor_day"], DAY_BINS) - 1, 0, N_BINS - 1)
        for sd, al in zip(rec["sample_days"], rec["alpha"]):
            sb = np.clip(np.digitize(sd, DAY_BINS) - 1, 0, N_BINS - 1)
            S[ab, sb] += al
            C[ab, sb] += 1
    with np.errstate(invalid="ignore"):
        return np.where(C > 0, S / C, np.nan)


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_heatmap(ax, mat, title, cmap="magma"):
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked.T, origin="lower", aspect="auto", cmap=cmap,
                   extent=[DAY_BINS[0], DAY_BINS[-1], DAY_BINS[0], DAY_BINS[-1]])
    ax.plot([0, 1825], [0, 1825], "w--", lw=0.8, alpha=0.6)
    ax.set_xticks(np.arange(0, 1826, 365))
    ax.set_yticks(np.arange(0, 1826, 365))
    ax.set_xlabel("Anchor day (prediction point, days post-Tx)", fontsize=FS_LABEL)
    ax.set_ylabel("Sample day (days post-Tx)", fontsize=FS_LABEL)
    ax.set_title(title, fontsize=FS_LABEL, fontweight="bold")
    ax.tick_params(labelsize=FS_TICK)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=FS_TICK)
    cbar.set_label("Mean LongMIL α", fontsize=FS_LABEL)
    return im


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print("Preloading bags...", flush=True)
    all_stems = set()
    for split in range(5):
        try:
            sp = build_splits_longitudinal(SAMPLES_DIR, SPLITS_CSV, fold=0, split=split)
            for pats in sp.values():
                for pat in pats:
                    all_stems.update(pat["stems"])
        except Exception:
            pass
    bag_cache = preload_bags(list(all_stems), SAMPLES_DIR, n_workers=4)
    print(f"  {len(bag_cache)} stems loaded", flush=True)

    for task_suffix, tcfg in TASKS.items():
        internal = tcfg["internal"]
        label    = tcfg["label"]
        print(f"\n=== {label} ===", flush=True)

        all_train_recs, all_test_recs = [], []

        for split in range(5):
            print(f"  Split {split}", flush=True)
            try:
                splits = build_splits_longitudinal(SAMPLES_DIR, SPLITS_CSV, fold=0, split=split)
            except Exception as e:
                print(f"    [error] {e}"); continue

            try:
                recs = collect_records(split, task_suffix, internal,
                                       splits["train"] + splits.get("val", []),
                                       device, bag_cache)
                print(f"    train: {len(recs)} anchor records", flush=True)
                all_train_recs.extend(recs)
            except Exception as e:
                print(f"    [train error] {e}", flush=True)

            try:
                recs = collect_records(split, task_suffix, internal,
                                       splits["test"], device, bag_cache)
                print(f"    test:  {len(recs)} anchor records", flush=True)
                all_test_recs.extend(recs)
            except Exception as e:
                print(f"    [test error] {e}", flush=True)

        train_mat = build_heatmap(all_train_recs)
        test_mat  = build_heatmap(all_test_recs)

        # Aggregate train+test
        fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor=BG)
        fig.patch.set_facecolor(BG)
        plot_heatmap(axes[0], train_mat,
                     f"{label} — Train+Val  (n={len(all_train_recs)} anchor events)")
        plot_heatmap(axes[1], test_mat,
                     f"{label} — Test  (n={len(all_test_recs)} anchor events)")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT_ROOT / f"abmil_attn_heatmap_{task_suffix}.{ext}",
                        dpi=300, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        print(f"  Saved abmil_attn_heatmap_{task_suffix}", flush=True)

        # Per-split test
        fig2, axes2 = plt.subplots(1, 5, figsize=(36, 7), facecolor=BG)
        fig2.patch.set_facecolor(BG)
        for sp, ax in enumerate(axes2):
            try:
                splits = build_splits_longitudinal(SAMPLES_DIR, SPLITS_CSV, fold=0, split=sp)
                recs   = collect_records(sp, task_suffix, internal,
                                          splits["test"], device, bag_cache)
            except Exception as e:
                print(f"    [split {sp} test error] {e}", flush=True)
                recs = []
            plot_heatmap(ax, build_heatmap(recs), f"{label}\nSplit {sp}  (n={len(recs)})")
        fig2.tight_layout()
        for ext in ("png", "pdf"):
            fig2.savefig(OUT_ROOT / f"abmil_attn_heatmap_{task_suffix}_per_split.{ext}",
                         dpi=300, bbox_inches="tight", facecolor=BG)
        plt.close(fig2)
        print(f"  Saved per-split heatmap for {task_suffix}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
