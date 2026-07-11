"""
classical_baselines.py — non-DL baselines using precomputed .pt file fields.

Baselines
---------
1. MeanConcat
   Per modality: mean pool inputs → one vector per modality.
   Concatenate → logistic regression (cls) / Cox (surv).

2. ClusterProp
   HE / BAL / CT : read instance_cluster_ids from .pt → proportion vector
                   (length = n_clusters for that modality) → CLR normalisation.
   Clinical       : mean pool inputs (tabular, not compositional).
   Concatenate → logistic regression / Cox.

Both handle missing modalities via zero-imputation (training-set mean).
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from lifelines import CoxPHFitter
import pandas as pd

# Modality keys as stored in .pt files
MOD_INPUT_KEYS = {
    "HE":       "HE_cells",
    "BAL":      "BAL_cells",
    "CT":       "CT_cells",
    "Clinical": "clinical_onehot",
}
CLINICAL_MOD = "Clinical"
PATCH_MODS   = [m for m in MOD_INPUT_KEYS if m != CLINICAL_MOD]
CLR_EPS      = 1e-6
LR_MAX_ITER  = 2000


# ── .pt file loader ────────────────────────────────────────────────────────────

def load_pt(stem: str, samples_dir: str) -> dict:
    path = Path(samples_dir)
    # stems may be zero-padded indices or patient IDs — find the file
    candidates = list(path.glob(f"{stem}.pt")) + list(path.glob(f"*{stem}*.pt"))
    if not candidates:
        return {}
    return torch.load(candidates[0], map_location="cpu", weights_only=False)


def _inputs(pt: dict, mod: str) -> Optional[np.ndarray]:
    key = MOD_INPUT_KEYS.get(mod)
    if key is None:
        return None
    t = pt.get("inputs", {}).get(key)
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return t.float().numpy()
    return np.array(t, dtype=np.float32)


def _cluster_ids(pt: dict, mod: str) -> Optional[np.ndarray]:
    key = MOD_INPUT_KEYS.get(mod)
    if key is None:
        return None
    t = pt.get("instance_cluster_ids", {}).get(key)
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return t.long().numpy()
    return np.array(t, dtype=np.int64)


# ── CLR ───────────────────────────────────────────────────────────────────────

def _clr(proportions: np.ndarray) -> np.ndarray:
    p = proportions + CLR_EPS
    log_p = np.log(p)
    return log_p - log_p.mean()


# ── MeanConcat ─────────────────────────────────────────────────────────────────

class MeanConcatExtractor:
    def __init__(self, samples_dir: str):
        self.samples_dir = samples_dir
        self.fallback: Dict[str, np.ndarray] = {}

    def fit(self, records: List[dict]) -> "MeanConcatExtractor":
        sums: Dict[str, list] = {m: [] for m in MOD_INPUT_KEYS}
        for r in records:
            pt = load_pt(r["stem"], self.samples_dir)
            for mod in MOD_INPUT_KEYS:
                arr = _inputs(pt, mod)
                if arr is not None and arr.ndim == 2:
                    sums[mod].append(arr.mean(0))
        for mod in MOD_INPUT_KEYS:
            if sums[mod]:
                self.fallback[mod] = np.stack(sums[mod]).mean(0)
            else:
                self.fallback[mod] = np.zeros(1, dtype=np.float32)
        return self

    def transform(self, records: List[dict]) -> np.ndarray:
        rows = []
        for r in records:
            pt = load_pt(r["stem"], self.samples_dir)
            parts = []
            for mod in MOD_INPUT_KEYS:
                arr = _inputs(pt, mod)
                if arr is not None and arr.ndim == 2:
                    parts.append(arr.mean(0))
                else:
                    parts.append(self.fallback[mod])
            rows.append(np.concatenate(parts))
        return np.stack(rows)


# ── ClusterProp ────────────────────────────────────────────────────────────────

class ClusterPropExtractor:
    """
    Cluster proportion features for patch modalities (CLR-normalised), clinical mean pool.

    Fast path: if a pre-computed CSV (from save_cluster_props.sh) is provided,
    reads CLR columns directly — no .pt loading needed for patch mods.
    Falls back to reading .pt files if CSV is absent or a stem is missing.
    """
    def __init__(self, samples_dir: str, props_csv: Optional[str] = None):
        self.samples_dir  = samples_dir
        self.n_clusters: Dict[str, int] = {}
        self.fallback_clr: Dict[str, np.ndarray] = {}
        self.clinical_fallback: Optional[np.ndarray] = None
        # Load pre-computed CSV index: stem → {mod_clr_k: value}
        self._csv_index: Dict[str, np.ndarray] = {}
        self._csv_clr_cols: Dict[str, List[str]] = {}  # mod → ordered CLR col names
        self._csv_clin_cols: List[str] = []
        if props_csv and Path(props_csv).exists():
            df = pd.read_csv(props_csv, index_col="stem")
            for mod in PATCH_MODS:
                cols = sorted([c for c in df.columns if c.startswith(f"{mod}_clr_")],
                              key=lambda c: int(c.split("_")[-1]))
                self._csv_clr_cols[mod] = cols
                self.n_clusters[mod] = len(cols)
            self._csv_clin_cols = sorted(
                [c for c in df.columns if c.startswith("Clinical_mean_")],
                key=lambda c: int(c.split("_")[-1]))
            for stem, row in df.iterrows():
                self._csv_index[str(stem)] = row
            print(f"    [ClusterProp] loaded CSV: {len(self._csv_index)} stems  "
                  + "  ".join(f"{m}={self.n_clusters[m]}" for m in PATCH_MODS)
                  + f"  Clinical_mean={len(self._csv_clin_cols)}")

    def fit(self, records: List[dict]) -> "ClusterPropExtractor":
        # If CSV covers everything, no .pt scanning needed
        csv_has_clin = bool(self._csv_clin_cols)
        all_from_csv = (len(self.n_clusters) == len(PATCH_MODS)) and csv_has_clin

        clin_vecs = []
        for r in records:
            if all_from_csv:
                break
            pt = load_pt(r["stem"], self.samples_dir)
            for mod in PATCH_MODS:
                if mod in self.n_clusters:
                    continue
                key = MOD_INPUT_KEYS[mod]
                cco = pt.get("cluster_count_onehot", {}).get(key)
                if cco is not None and isinstance(cco, torch.Tensor) and cco.ndim == 2:
                    self.n_clusters[mod] = cco.shape[0]
            if not csv_has_clin:
                arr = _inputs(pt, CLINICAL_MOD)
                if arr is not None and arr.ndim == 2:
                    clin_vecs.append(arr.mean(0))
            if len(self.n_clusters) == len(PATCH_MODS) and (csv_has_clin or clin_vecs):
                break

        for mod in PATCH_MODS:
            if mod not in self.n_clusters:
                self.n_clusters[mod] = 1
            self.fallback_clr[mod] = np.zeros(self.n_clusters[mod], dtype=np.float32)

        if csv_has_clin:
            self.clinical_fallback = np.zeros(len(self._csv_clin_cols), dtype=np.float32)
        else:
            self.clinical_fallback = (np.stack(clin_vecs).mean(0) if clin_vecs
                                      else np.zeros(1, dtype=np.float32))
        for mod, nc in self.n_clusters.items():
            print(f"    [ClusterProp] {mod}: n_clusters={nc}")
        return self

    def _bag_clr(self, ids: np.ndarray, mod: str) -> np.ndarray:
        nc = self.n_clusters[mod]
        counts = np.bincount(ids, minlength=nc).astype(np.float32)
        prop   = counts / max(counts.sum(), 1)
        return _clr(prop)

    def transform(self, records: List[dict]) -> np.ndarray:
        rows = []
        for r in records:
            stem = r["stem"]
            csv_row = self._csv_index.get(stem)
            # CSV may store stems as plain integers while records use zero-padded strings
            if csv_row is None:
                try:
                    csv_row = self._csv_index.get(str(int(stem)))
                except (ValueError, TypeError):
                    pass
            pt: dict = {}
            parts = []

            # Clinical: use CSV if available, else load .pt
            if csv_row is not None and self._csv_clin_cols:
                parts.append(csv_row[self._csv_clin_cols].values.astype(np.float32))
            else:
                pt = load_pt(stem, self.samples_dir)
                arr = _inputs(pt, CLINICAL_MOD)
                parts.append(arr.mean(0) if arr is not None and arr.ndim == 2
                             else self.clinical_fallback)

            # Patch modalities: use CSV if available, else .pt
            for mod in PATCH_MODS:
                if csv_row is not None and mod in self._csv_clr_cols:
                    parts.append(csv_row[self._csv_clr_cols[mod]].values.astype(np.float32))
                else:
                    if not pt:
                        pt = load_pt(stem, self.samples_dir)
                    ids = _cluster_ids(pt, mod)
                    parts.append(self._bag_clr(ids, mod) if ids is not None and len(ids) > 0
                                 else self.fallback_clr[mod])

            rows.append(np.concatenate(parts))
        return np.stack(rows)


# ── Label helpers ──────────────────────────────────────────────────────────────

def _cls_labels(records: List[dict]) -> Tuple[np.ndarray, np.ndarray]:
    mask   = np.array([r.get("label") is not None for r in records])
    labels = np.array([r.get("label", 0) for r in records], dtype=np.float32)
    return mask, labels


def _surv_labels(records: List[dict], endpoint: str
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    time_key  = {"acr": "tte_next_acr", "clad": "clad_time",  "death": "death_time" }[endpoint]
    event_key = {"acr": "event_next_acr","clad": "clad_event", "death": "death_event"}[endpoint]
    mask, times, events = [], [], []
    for r in records:
        t = r.get(time_key,  float("nan"))
        e = r.get(event_key, float("nan"))
        valid = not (math.isnan(t) or math.isnan(e)) and t > 0
        mask.append(valid)
        times.append(t if valid else 0.0)
        events.append(float(e) if valid else 0.0)
    return np.array(mask), np.array(times), np.array(events)


# ── Classifiers ────────────────────────────────────────────────────────────────

def _clean_X(X: np.ndarray) -> np.ndarray:
    X = X.copy()
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(X, -1e6, 1e6, out=X)
    return X


C_GRID          = [0.01, 0.1, 1.0, 10.0, 100.0]
PENALIZER_GRID  = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]


def _fit_logistic(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> Pipeline:
    pipe = Pipeline([
        ("vt", VarianceThreshold(threshold=1e-8)),
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=LR_MAX_ITER, C=C,
                                   class_weight="balanced",
                                   solver="lbfgs", random_state=42)),
    ])
    pipe.fit(_clean_X(X), y)
    return pipe




def _bacc_raw(pipe, X: np.ndarray, y: np.ndarray) -> float:
    p = pipe.predict(_clean_X(X))
    pos = y == 1; neg = y == 0
    sens = p[pos].mean() if pos.any() else 0.5
    spec = (1 - p[neg]).mean() if neg.any() else 0.5
    return float((sens + spec) / 2)


def _bacc(pipe, X: np.ndarray, y: np.ndarray) -> float:
    return _bacc_raw(pipe, X, y)


def _auc(pipe, X: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, pipe.predict_proba(_clean_X(X))[:, 1]))
    except Exception:
        return float("nan")


class _CoxWrapper:
    """Variance-filter → StandardScaler → CoxPH pipeline."""

    def __init__(self, penalizer: float = 1.0):
        self.penalizer = penalizer

    @staticmethod
    def _clean(Xs: np.ndarray) -> np.ndarray:
        # nan_to_num replaces inf → finfo.max (~1e308) which Cox can't handle.
        # Replace nan/±inf with 0 then clip to ±10 (post-standardisation, >10σ = outlier).
        np.nan_to_num(Xs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(Xs, -10.0, 10.0, out=Xs)
        return Xs

    def fit(self, X: np.ndarray, times: np.ndarray, events: np.ndarray) -> "_CoxWrapper":
        self.vt = VarianceThreshold(threshold=1e-8).fit(X)
        Xv = self.vt.transform(X)
        self.sc = StandardScaler().fit(Xv)
        Xs = self._clean(self.sc.transform(Xv))
        df = pd.DataFrame(Xs, columns=[f"f{i}" for i in range(Xs.shape[1])])
        df["T"] = times; df["E"] = events
        self.cox = CoxPHFitter(penalizer=self.penalizer)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.cox.fit(df, duration_col="T", event_col="E")
        return self

    def score(self, X: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:
        Xv = self.vt.transform(X)
        Xs = self._clean(self.sc.transform(Xv))
        df = pd.DataFrame(Xs, columns=[f"f{i}" for i in range(Xs.shape[1])])
        df["T"] = times; df["E"] = events
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(self.cox.score(df, scoring_method="concordance_index"))


# ── HP selection helpers ───────────────────────────────────────────────────────

def _select_C_on_val(X_tr: np.ndarray, y_tr: np.ndarray, tr_m: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray, val_m: np.ndarray) -> float:
    """Train on X_tr[tr_m], pick C with best val BACC."""
    best_C, best_score = C_GRID[0], -1.0
    if val_m.sum() < 5:
        return 1.0
    for C in C_GRID:
        try:
            pipe = _fit_logistic(X_tr[tr_m], y_tr[tr_m], C=C)
            score = _bacc_raw(pipe, X_val[val_m], y_val[val_m])
            if score > best_score:
                best_score, best_C = score, C
        except Exception:
            pass
    print(f"    [HP-cls] best C={best_C}  val_bacc={best_score:.3f}")
    return best_C


def _select_penalizer_on_val(X_tr: np.ndarray, t_tr: np.ndarray, e_tr: np.ndarray,
                             tr_sm: np.ndarray,
                             X_val: np.ndarray, t_val: np.ndarray, e_val: np.ndarray,
                             val_sm: np.ndarray) -> float:
    """Train Cox on X_tr[tr_sm], pick penalizer with best val C-index."""
    best_p, best_ci = 1.0, -1.0
    if tr_sm.sum() < 10 or e_tr[tr_sm].sum() < 3 or val_sm.sum() < 5:
        return best_p
    for p in PENALIZER_GRID:
        try:
            w = _CoxWrapper(penalizer=p).fit(X_tr[tr_sm], t_tr[tr_sm], e_tr[tr_sm])
            ci = w.score(X_val[val_sm], t_val[val_sm], e_val[val_sm])
            if ci > best_ci:
                best_ci, best_p = ci, p
        except Exception:
            pass
    print(f"    [HP-cox] best penalizer={best_p}  val_ci={best_ci:.3f}")
    return best_p


# ── Eval with HP selection ─────────────────────────────────────────────────────

def _eval_one(name: str,
              X_tr_hp: np.ndarray,   # train-only features (for HP selection)
              X_tr_fit: np.ndarray,  # train+val features  (for final fitting)
              X_te: np.ndarray,
              train_recs: List[dict],
              fit_recs: List[dict],
              test_recs: List[dict],
              X_val: Optional[np.ndarray] = None,
              val_recs: Optional[List[dict]] = None) -> dict:
    """HP selection on val, final fit on train+val, evaluate on test."""
    res: dict = {}
    do_hp = X_val is not None and val_recs is not None

    # ── Classification ────────────────────────────────────────────────────────
    tr_m_hp,  tr_y_hp  = _cls_labels(train_recs)
    tr_m_fit, tr_y_fit = _cls_labels(fit_recs)
    te_m,     te_y     = _cls_labels(test_recs)
    if tr_m_fit.sum() >= 10 and te_m.sum() >= 5:
        try:
            if do_hp:
                val_m, val_y = _cls_labels(val_recs)
                best_C = _select_C_on_val(X_tr_hp, tr_y_hp, tr_m_hp,
                                          X_val,   val_y,   val_m)
            else:
                best_C = 1.0
            pipe = _fit_logistic(X_tr_fit[tr_m_fit], tr_y_fit[tr_m_fit], C=best_C)
            res["test_bacc"] = _bacc(pipe, X_te[te_m], te_y[te_m])
            res["test_auc"]  = _auc(pipe,  X_te[te_m], te_y[te_m])
            print(f"  [{name}] ACR cls  C={best_C}  test_bacc={res['test_bacc']:.3f}")
        except Exception as exc:
            print(f"  [{name}] ACR cls failed: {exc}")

    # ── Survival ──────────────────────────────────────────────────────────────
    for ep in ("acr", "clad", "death"):
        tr_sm_hp,  tr_t_hp,  tr_e_hp  = _surv_labels(train_recs, ep)
        tr_sm_fit, tr_t_fit, tr_e_fit = _surv_labels(fit_recs,   ep)
        te_sm,     te_t,     te_e     = _surv_labels(test_recs,  ep)
        if tr_sm_fit.sum() < 10 or tr_e_fit[tr_sm_fit].sum() < 3:
            continue
        try:
            if do_hp:
                val_sm, val_t, val_e = _surv_labels(val_recs, ep)
                best_p = _select_penalizer_on_val(
                    X_tr_hp, tr_t_hp, tr_e_hp, tr_sm_hp,
                    X_val,   val_t,   val_e,   val_sm)
            else:
                best_p = 1.0
            cox_w = _CoxWrapper(penalizer=best_p).fit(
                X_tr_fit[tr_sm_fit], tr_t_fit[tr_sm_fit], tr_e_fit[tr_sm_fit])
            if te_sm.sum() >= 5:
                ci = cox_w.score(X_te[te_sm], te_t[te_sm], te_e[te_sm])
                res[f"test_ci_{ep}"] = ci
                print(f"  [{name}] {ep} surv  pen={best_p}  test_ci={ci:.3f}")
        except Exception as exc:
            print(f"  [{name}] {ep} Cox failed: {exc}")

    return res


# ── Main entry ─────────────────────────────────────────────────────────────────

def run_classical_baselines(
    samples_dir: str,
    train_recs: List[dict],
    test_recs: List[dict],
    val_recs: Optional[List[dict]] = None,
    props_csv: Optional[str] = None,
) -> Dict[str, dict]:
    """Fit and evaluate cluster_prop (multimodal + per-modality) baselines.

    HP selection: if val_recs provided, sweeps C/penalizer on train→val, then
    retrains on train+val with best HP. Without val_recs, uses C=1/pen=1.
    """
    fit_recs = train_recs + (val_recs if val_recs else [])
    results: Dict[str, dict] = {}

    ext = ClusterPropExtractor(samples_dir, props_csv=props_csv)
    print(f"\n  [cluster_prop] fitting extractor on {len(fit_recs)} bags (train+val)...")
    ext.fit(fit_recs)

    print(f"  [cluster_prop] extracting features...")
    X_fit = _clean_X(ext.transform(fit_recs))
    X_te  = _clean_X(ext.transform(test_recs))
    X_tr  = _clean_X(ext.transform(train_recs))
    X_val = _clean_X(ext.transform(val_recs)) if val_recs else None

    # Feature layout: [Clinical(n_clin), HE(n_he), BAL(n_bal), CT(n_ct)]
    n_clin = len(ext._csv_clin_cols) if ext._csv_clin_cols else ext.clinical_fallback.shape[0]
    n_he   = ext.n_clusters.get("HE",  0)
    n_bal  = ext.n_clusters.get("BAL", 0)
    n_ct   = ext.n_clusters.get("CT",  0)
    slices = {
        "Clinical": (0,            n_clin),
        "HE":       (n_clin,       n_clin + n_he),
        "BAL":      (n_clin + n_he, n_clin + n_he + n_bal),
        "CT":       (n_clin + n_he + n_bal, n_clin + n_he + n_bal + n_ct),
    }
    print(f"  [cluster_prop] total dim={X_fit.shape[1]}  "
          + "  ".join(f"{m}={e-s}" for m, (s, e) in slices.items()))

    def _slice(arr, s, e): return arr[:, s:e] if arr is not None else None

    # Multimodal — HP on train→val, final fit on train+val
    results["cluster_prop"] = _eval_one(
        "cluster_prop",
        X_tr, X_fit, X_te,
        train_recs, fit_recs, test_recs,
        X_val=X_val, val_recs=val_recs)

    # Unimodal per modality
    for mod, (s, e) in slices.items():
        if e <= s:
            continue
        name = f"cluster_prop_{mod}"
        print(f"\n  [{name}] dim={e-s}")
        results[name] = _eval_one(
            name,
            _slice(X_tr,  s, e), _slice(X_fit, s, e), _slice(X_te, s, e),
            train_recs, fit_recs, test_recs,
            X_val=_slice(X_val, s, e), val_recs=val_recs)

    return results
