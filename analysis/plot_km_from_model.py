"""
Standalone KM curves stratified by model risk score — per task, from best model predictions.

Reads logit scores from the unified UMAP npy caches (same source), joins outcomes
from splits CSV, plots KM curves for top vs bottom tertile with log-rank p-value.

Output: figures/km_curves/km_{task}.png/.pdf
        figures/km_curves/km_all_tasks.png/.pdf

Run via: sbatch analysis/submit_km_from_model.sh
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import rankdata

ROOT     = Path(__file__).resolve().parent.parent
INTERP   = ROOT / "interpretability"
FIG_ROOT = ROOT / "figures" / "km_curves"
FIG_ROOT.mkdir(parents=True, exist_ok=True)

SPLITS_CSV = Path("/home/aih/dinesh.haridoss/chicago/plots/multimodal_splits_nested_cv.csv")

BG  = "#FAF6F2"
HI  = "#952030"
LO  = "#1A5C8A"
MU  = "#7D6D78"

# ── patient-level outcome lookup ──────────────────────────────────────────────
df_splits = pd.read_csv(SPLITS_CSV)
PT = {}
for pid, grp in df_splits.groupby("patient_id"):
    r = grp.iloc[0]
    PT[pid] = {
        "label":       float(r.get("label",        float("nan"))),
        "event_acr":   float(r.get("acr_status",   float("nan"))),
        "tte_acr":     float(r.get("acr_days",     float("nan"))),
        "event_clad":  float(r.get("clad_status",  float("nan"))),
        "tte_clad":    float(r.get("clad_days",    float("nan"))),
        "event_death": float(r.get("death_status", float("nan"))),
        "tte_death":   float(r.get("death_days",   float("nan"))),
    }
print(f"[lookup] {len(PT)} patients")

# ── Task configuration ────────────────────────────────────────────────────────
TASK_CFG = {
    "acr_cls": {
        "label":      "ACR classification",
        "rep_key":    "acr_cls",
        "ev_key":     "event_acr",
        "tte_key":    "tte_acr",
        "score_type": "cls",
        "npy_paths":  [INTERP / "set_mil_mt_interp/all_splits_cls/results_raw.npy"],
        "ylabel":     "ACR-free probability",
        "event_lbl":  "ACR",
    },
    "clad_surv": {
        "label":      "CLAD survival",
        "rep_key":    "clad",
        "ev_key":     "event_clad",
        "tte_key":    "tte_clad",
        "score_type": "surv",
        "npy_paths":  [INTERP / "set_mil_mt_interp/all_splits_clad_surv/results_raw.npy"],
        "ylabel":     "CLAD-free probability",
        "event_lbl":  "CLAD",
    },
    "acr_surv": {
        "label":      "ACR survival",
        "rep_key":    "acr_surv",
        "ev_key":     "event_acr",
        "tte_key":    "tte_acr",
        "score_type": "surv",
        "npy_paths":  [
            INTERP / f"longitudinal_mk_interp/longitudinal_mk_no_alibi_split{s}_fold0_acr_surv/results_raw.npy"
            for s in range(5)
        ],
        "ylabel":     "ACR-free probability",
        "event_lbl":  "ACR",
    },
    "death_surv": {
        "label":      "Death survival",
        "rep_key":    "death",
        "ev_key":     "event_death",
        "tte_key":    "tte_death",
        "score_type": "surv",
        "npy_paths":  [
            INTERP / f"longitudinal_mk_interp/longitudinal_mk_no_alibi_split{s}_fold0_death_surv/results_raw.npy"
            for s in range(5)
        ],
        "ylabel":     "Survival probability",
        "event_lbl":  "Death",
    },
}


def _try_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_scores(cfg):
    rep_key = cfg["rep_key"]
    ev_key  = cfg["ev_key"]
    tte_key = cfg["tte_key"]
    rows = []
    for p in cfg["npy_paths"]:
        if not p.exists():
            print(f"  [warn] {p}")
            continue
        raw = np.load(p, allow_pickle=True)
        items = list(raw) if raw.ndim > 0 else [raw.item()]
        for item in items:
            if not isinstance(item, dict):
                continue
            # logit
            ld = item.get("logits", {})
            logit = float(ld.get(rep_key, float("nan"))
                          if isinstance(ld, dict) else float("nan"))
            # outcome from npy or lookup
            pid = item.get("patient_id")
            lkp = PT.get(pid, {})
            ev  = _try_float(item.get(ev_key)  or lkp.get(ev_key))
            tte = _try_float(item.get(tte_key) or lkp.get(tte_key))
            rows.append({"logit": logit, "ev": ev, "tte": tte})
    return pd.DataFrame(rows)


def _km(ev, tte):
    order = np.argsort(tte)
    ev_o, tte_o = ev[order], tte[order]
    n = len(ev_o)
    at_risk, surv = n, 1.0
    ts, ss = [0.0], [1.0]
    for i in range(n):
        if ev_o[i] == 1:
            surv *= (at_risk - 1) / at_risk
        at_risk -= 1
        ts.append(float(tte_o[i]))
        ss.append(surv)
    return np.array(ts), np.array(ss)


def logrank_p(ev1, tte1, ev2, tte2):
    """Simple log-rank test p-value."""
    try:
        from lifelines.statistics import logrank_test
        r = logrank_test(tte1, tte2, event_observed_A=ev1, event_observed_B=ev2)
        return r.p_value
    except ImportError:
        pass
    try:
        from scipy.stats import chi2
        all_t = np.unique(np.concatenate([tte1[ev1==1], tte2[ev2==1]]))
        O1, E1, O2, E2 = 0.0, 0.0, 0.0, 0.0
        n1, n2 = len(tte1), len(tte2)
        for t in all_t:
            r1 = (tte1 >= t).sum(); r2 = (tte2 >= t).sum()
            d1 = ((tte1 == t) & (ev1 == 1)).sum()
            d2 = ((tte2 == t) & (ev2 == 1)).sum()
            d  = d1 + d2; r = r1 + r2
            if r < 2: continue
            E1 += d * r1 / r
            E2 += d * r2 / r
            O1 += d1; O2 += d2
        denom = (E1 + E2) * E1 * E2 / max((E1 + E2)**2, 1e-9)
        stat  = (O1 - E1)**2 / max(E1, 1e-9) + (O2 - E2)**2 / max(E2, 1e-9)
        return float(chi2.sf(stat, df=1))
    except Exception:
        return float("nan")


def plot_km(ax, df, cfg, show_legend=False):
    valid = df.dropna(subset=["logit", "ev", "tte"])
    valid = valid[valid["tte"] > 0]
    if len(valid) < 20:
        ax.text(0.5, 0.5, f"N={len(valid)} (insufficient)", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        return

    logits = valid["logit"].values
    ev     = valid["ev"].values.astype(float)
    tte    = valid["tte"].values / 365.25   # days → years

    if cfg["score_type"] == "cls":
        scores = 1.0 / (1.0 + np.exp(-logits))
    else:
        scores = (rankdata(logits) - 1) / max(len(logits) - 1, 1)

    q33, q67 = np.percentile(scores, [33, 67])
    hi = (scores >= q67)
    lo = (scores <= q33)

    t_hi, s_hi = _km(ev[hi], tte[hi])
    t_lo, s_lo = _km(ev[lo], tte[lo])

    ax.step(t_hi, s_hi, where="post", color=HI, lw=2.2, label=f"High risk (n={hi.sum()})")
    ax.step(t_lo, s_lo, where="post", color=LO, lw=2.2, label=f"Low risk  (n={lo.sum()})")

    # Censor ticks
    for mask, col in [(hi, HI), (lo, LO)]:
        cen = valid[mask & (ev == 0)]
        ax.scatter(cen["tte"].values / 365.25,
                   np.interp(cen["tte"].values / 365.25, t_hi if (col == HI) else t_lo,
                             s_hi if (col == HI) else s_lo),
                   marker="|", color=col, s=30, alpha=0.6, linewidths=1.0)

    p = logrank_p(ev[hi], tte[hi] * 365.25, ev[lo], tte[lo] * 365.25)
    p_str = f"p={p:.3f}" if p >= 0.001 else f"p<0.001"
    ax.text(0.97, 0.95, p_str, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, fontweight="bold",
            color="#1A1018" if p < 0.05 else MU)

    ax.set_title(f"{cfg['label']}\n(top vs bottom risk tertile, N={len(valid)})",
                 fontsize=9, fontweight="bold", pad=4)
    ax.set_xlabel("Time (years)", fontsize=8)
    ax.set_ylabel(cfg["ylabel"], fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_facecolor(BG)
    ax.spines[["top","right"]].set_visible(False)
    if show_legend:
        ax.legend(fontsize=7.5, framealpha=0.85, loc="lower left")


# ── Per-task figures ──────────────────────────────────────────────────────────
for task_key, cfg in TASK_CFG.items():
    print(f"[{task_key}] loading...")
    df = load_scores(cfg)
    print(f"  N={len(df)}  ev_nnan={df['ev'].notna().sum()}  tte_nnan={df['tte'].notna().sum()}")
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    plot_km(ax, df, cfg, show_legend=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_ROOT / f"km_{task_key}.{ext}", dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  [done] km_{task_key}")

# ── 4-panel combined ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
fig.patch.set_facecolor(BG)
fig.suptitle("Kaplan-Meier: top vs bottom model risk tertile — all tasks",
             fontsize=11, fontweight="bold")
for ax, (tk, cfg) in zip(axes, TASK_CFG.items()):
    ax.set_facecolor(BG)
    df = load_scores(cfg)
    plot_km(ax, df, cfg, show_legend=(tk == "acr_cls"))

handles = [
    Line2D([0],[0], color=HI, lw=2.5, label="Top tertile (high risk)"),
    Line2D([0],[0], color=LO, lw=2.5, label="Bottom tertile (low risk)"),
]
fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=9,
           bbox_to_anchor=(0.5, 1.0), framealpha=0.88)
fig.tight_layout(rect=[0, 0, 1, 0.94])
for ext in ("png", "pdf"):
    fig.savefig(FIG_ROOT / f"km_all_tasks.{ext}", dpi=180, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print("[done] km_all_tasks")
