"""
precompute_dataset.py
=====================
ONE-TIME job: iterates the full MultimodalTimeseriesDataset (all 11 modalities),
applies KEY_REMAP and label_fn, and saves each aligned timepoint as an
individual <idx:05d>.pt file.

DIAGNOSE MODE  (fast — filters data sources before building dataset)
─────────────────────────────────────────────────────────────────────
  python precompute_dataset.py --diagnose --diagnose_patients P001 P002
  python precompute_dataset.py --diagnose --diagnose_n 3
  python precompute_dataset.py --diagnose --diagnose_n 5 --no_raw_times

  Each h5ad / CSV is filtered to only the requested patients (using that
  file's own identifier column) before the dataset is built, so startup
  takes seconds rather than minutes.

  All files share the same patient ID values (LT001, LT002, …) — only
  the column name differs per file (record_id, Patient, …). Pass the
  same IDs for all modalities.

NORMAL PRECOMPUTE
──────────────────
  python precompute_dataset.py
  python precompute_dataset.py --cache_dir /lustre/…/mil/cache
  python precompute_dataset.py --skip_existing   # resume
  python precompute_dataset.py --workers 8
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import anndata as ad
import numpy as np
import pandas as pd
import torch

# ── module path ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = Path("/home/aih/dinesh.haridoss")
for _p in [str(SCRIPT_DIR), str(BASE_DIR / "chicago")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from multimodal_dataset import MultimodalTimeseriesDataset


# ===========================================================================
# SECTION 0 — Clinical Feature Tokenizer
# ===========================================================================

class ClinicalFeatureTokenizer:
    """
    Tokenize clinical tabular features for transformer input.

    Each continuous feature is discretized into n_quantiles bins (default 4).
    Categorical/binary features (≤ categorical_threshold unique values) get one
    token per unique category.  NaN values get a dedicated per-feature NaN token.

    Output stored per .pt sample:
      clinical_token_ids  — int64 tensor of shape (n_features,)
      clinical_vocab      — list of {id, feature, bin, label, [range]} for visualization
    """

    def __init__(self, n_quantiles: int = 4, categorical_threshold: int = 10):
        self.n_quantiles = n_quantiles
        self.categorical_threshold = categorical_threshold
        self.feature_cols: List[str] = []
        self.feature_info: Dict[str, dict] = {}
        self.vocab: Dict[int, dict] = {}
        self._n_tokens: int = 0

    def fit(
        self,
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        exclude_cols: Union[None, List[str]] = None,
    ) -> "ClinicalFeatureTokenizer":
        exclude = {id_col, time_col} | set(exclude_cols or [])
        self.feature_cols = [c for c in df.columns if c not in exclude]
        self._n_tokens = 0
        self.feature_info = {}
        self.vocab = {}

        for col in self.feature_cols:
            num_series = pd.to_numeric(df[col], errors="coerce")
            unique_count = df[col].dropna().nunique()

            if unique_count <= self.categorical_threshold:
                categories = sorted(df[col].dropna().unique(), key=str)
                cat_to_token: Dict[str, int] = {}
                token_ids = []
                for cat in categories:
                    tid = self._n_tokens
                    self.vocab[tid] = {"feature": col, "bin": f"cat_{cat}", "label": f"{col}_cat_{cat}"}
                    cat_to_token[str(cat)] = tid
                    token_ids.append(tid)
                    self._n_tokens += 1
                nan_tid = self._n_tokens
                self.vocab[nan_tid] = {"feature": col, "bin": "nan", "label": f"{col}_nan"}
                self._n_tokens += 1
                self.feature_info[col] = {
                    "type": "categorical",
                    "categories": [str(c) for c in categories],
                    "cat_to_token": cat_to_token,
                    "nan_token": nan_tid,
                    "token_ids": token_ids,
                }
            else:
                valid = num_series.dropna()
                quantiles = np.linspace(0, 1, self.n_quantiles + 1)
                boundaries = np.unique(valid.quantile(quantiles).values)
                actual_bins = max(len(boundaries) - 1, 1)
                token_ids = []
                for q in range(actual_bins):
                    tid = self._n_tokens
                    lo = float(boundaries[q])
                    hi = float(boundaries[q + 1]) if q + 1 < len(boundaries) else float("inf")
                    self.vocab[tid] = {
                        "feature": col, "bin": f"q{q}", "label": f"{col}_q{q}",
                        "range": [lo, hi],
                    }
                    token_ids.append(tid)
                    self._n_tokens += 1
                nan_tid = self._n_tokens
                self.vocab[nan_tid] = {"feature": col, "bin": "nan", "label": f"{col}_nan"}
                self._n_tokens += 1
                self.feature_info[col] = {
                    "type": "continuous",
                    "boundaries": boundaries.tolist(),
                    "token_ids": token_ids,
                    "nan_token": nan_tid,
                    "n_bins": actual_bins,
                }

        print(f"  [ClinicalTokenizer] fit on {len(self.feature_cols)} features → "
              f"{self._n_tokens} total tokens")
        return self

    def transform_row(self, row: pd.Series) -> np.ndarray:
        """Transform one row to int64 token IDs, shape (n_features,)."""
        out = []
        for col in self.feature_cols:
            info = self.feature_info[col]
            val = row.get(col, np.nan)
            if info["type"] == "categorical":
                if pd.isna(val) or str(val) not in info["cat_to_token"]:
                    out.append(info["nan_token"])
                else:
                    out.append(info["cat_to_token"][str(val)])
            else:
                try:
                    fval = float(val)
                    if np.isnan(fval):
                        out.append(info["nan_token"])
                    else:
                        inner = info["boundaries"][1:-1]
                        bin_idx = int(np.searchsorted(inner, fval, side="right"))
                        bin_idx = min(bin_idx, info["n_bins"] - 1)
                        out.append(info["token_ids"][bin_idx])
                except (ValueError, TypeError):
                    out.append(info["nan_token"])
        return np.array(out, dtype=np.int64)

    def vocab_list(self) -> list:
        return [{"id": tid, **v} for tid, v in sorted(self.vocab.items())]

    def to_dict(self) -> dict:
        return {
            "n_quantiles": self.n_quantiles,
            "categorical_threshold": self.categorical_threshold,
            "feature_cols": self.feature_cols,
            "feature_info": self.feature_info,
            "vocab": {str(k): v for k, v in self.vocab.items()},
            "n_tokens": self._n_tokens,
        }

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def transform_df(self, df: pd.DataFrame, id_col: str, time_col: str) -> pd.DataFrame:
        """Apply tokenizer to every row of df; return DataFrame of int64 token IDs."""
        rows = []
        for _, row in df.iterrows():
            rows.append(self.transform_row(row))
        return pd.DataFrame(rows, columns=self.feature_cols)

    def plot_feature_distributions(
        self,
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        save_dir: "Path",
        cols_per_row: int = 6,
        rows_per_page: int = 5,
    ) -> None:
        """
        Save a grid of per-feature plots to save_dir/feature_distributions_page{N}.png.
        Continuous features: histogram + vertical lines at quantile boundaries.
        Categorical features: bar chart of category counts.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        bin_colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"]
        per_page = cols_per_row * rows_per_page
        n_feats = len(self.feature_cols)

        for page_start in range(0, n_feats, per_page):
            page_feats = self.feature_cols[page_start: page_start + per_page]
            n_plots = len(page_feats)
            ncols = min(cols_per_row, n_plots)
            nrows = (n_plots + ncols - 1) // ncols

            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8))
            axes = np.array(axes).flatten()

            for ax_i, col in enumerate(page_feats):
                ax = axes[ax_i]
                info = self.feature_info[col]
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                n_nan = df[col].isna().sum() + (df[col].notna() & pd.to_numeric(df[col], errors="coerce").isna()).sum()
                pct_nan = 100 * n_nan / max(len(df), 1)

                if info["type"] == "continuous":
                    if len(series) > 0:
                        ax.hist(series, bins=30, color="#aec7e8", edgecolor="white", linewidth=0.4)
                        bounds = info["boundaries"]
                        for b_i, b in enumerate(bounds[1:-1], 1):
                            ax.axvline(b, color=bin_colors[b_i % len(bin_colors)],
                                       lw=1.2, ls="--", label=f"Q{b_i}")
                    ax.set_ylabel("count", fontsize=7)
                else:
                    cats = info["categories"]
                    counts = [df[col].astype(str).eq(c).sum() for c in cats]
                    ax.bar(range(len(cats)), counts,
                           color=[bin_colors[i % len(bin_colors)] for i in range(len(cats))])
                    ax.set_xticks(range(len(cats)))
                    ax.set_xticklabels([str(c)[:8] for c in cats], rotation=45,
                                       ha="right", fontsize=6)

                title = f"{col[:28]}\n(NaN {pct_nan:.0f}%)"
                ax.set_title(title, fontsize=7, pad=2)
                ax.tick_params(axis="both", labelsize=6)

            for ax_i in range(n_plots, len(axes)):
                axes[ax_i].set_visible(False)

            page_num = page_start // per_page
            fig.suptitle(
                f"Clinical feature distributions  (page {page_num + 1})",
                fontsize=10, y=1.01,
            )
            fig.tight_layout()
            out = save_dir / f"feature_distributions_page{page_num + 1}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)

        print(f"  [viz] Feature distribution plots → {save_dir}  "
              f"({(n_feats + per_page - 1) // per_page} page(s))")

    def plot_missing_rates(
        self,
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        save_path: "Path",
    ) -> None:
        """
        Horizontal bar chart of NaN rate per feature, sorted descending.
        Saved to save_path.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        exclude = {id_col, time_col}
        feat_cols = [c for c in self.feature_cols if c not in exclude]
        nan_rates = {
            col: 100 * (
                df[col].isna().sum() +
                (df[col].notna() & pd.to_numeric(df[col], errors="coerce").isna()).sum()
            ) / max(len(df), 1)
            for col in feat_cols
        }
        sorted_items = sorted(nan_rates.items(), key=lambda x: x[1], reverse=True)
        cols_sorted = [k for k, _ in sorted_items]
        rates_sorted = [v for _, v in sorted_items]

        fig_h = max(6, len(cols_sorted) * 0.18)
        fig, ax = plt.subplots(figsize=(9, fig_h))
        bars = ax.barh(range(len(cols_sorted)), rates_sorted, color="#4e79a7", edgecolor="white")
        ax.set_yticks(range(len(cols_sorted)))
        ax.set_yticklabels(cols_sorted, fontsize=6)
        ax.set_xlabel("Missing rate (%)", fontsize=9)
        ax.set_title("Clinical feature — missing value rates", fontsize=11)
        ax.set_xlim(0, max(rates_sorted or [1]) * 1.12)
        ax.axvline(20, color="orange", lw=1, ls="--", label="20%")
        ax.axvline(50, color="red",    lw=1, ls="--", label="50%")
        ax.legend(fontsize=7)
        for bar, rate in zip(bars, rates_sorted):
            if rate > 0.5:
                ax.text(rate + 0.3, bar.get_y() + bar.get_height() / 2,
                        f"{rate:.0f}%", va="center", fontsize=5.5)

        fig.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [viz] Missing-rate chart → {save_path}")

    def plot_bin_occupancy_heatmap(
        self,
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        save_dir: "Path",
    ) -> None:
        """
        Heatmap: features × bins — colour = % of non-NaN samples in each bin.
        Saves separate PNGs for continuous and categorical features.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        def _heatmap(feat_list, title_suffix, fname):
            if not feat_list:
                return
            max_bins = max(
                (self.feature_info[c]["n_bins"] if self.feature_info[c]["type"] == "continuous"
                 else len(self.feature_info[c]["categories"]))
                for c in feat_list
            )
            matrix = np.full((len(feat_list), max_bins + 1), np.nan)  # +1 for NaN col
            col_labels = [f"bin_{b}" for b in range(max_bins)] + ["NaN"]

            for r, col in enumerate(feat_list):
                info = self.feature_info[col]
                total = max(len(df), 1)
                n_nan = (
                    df[col].isna().sum() +
                    (df[col].notna() & pd.to_numeric(df[col], errors="coerce").isna()).sum()
                )
                matrix[r, -1] = 100 * n_nan / total

                if info["type"] == "continuous":
                    series = pd.to_numeric(df[col], errors="coerce").dropna()
                    bounds = info["boundaries"]
                    n_bins = info["n_bins"]
                    for b in range(n_bins):
                        lo = bounds[b]
                        hi = bounds[b + 1] if b + 1 < len(bounds) else float("inf")
                        count = ((series >= lo) & (series < hi)).sum() if b < n_bins - 1 \
                            else (series >= lo).sum()
                        matrix[r, b] = 100 * count / total
                else:
                    for b_i, cat in enumerate(info["categories"]):
                        count = df[col].astype(str).eq(cat).sum()
                        matrix[r, b_i] = 100 * count / total

            fig_h = max(5, len(feat_list) * 0.22)
            fig, ax = plt.subplots(figsize=(min(18, max_bins + 4), fig_h))
            im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
            ax.set_xticks(range(max_bins + 1))
            ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(feat_list)))
            ax.set_yticklabels(feat_list, fontsize=6)
            fig.colorbar(im, ax=ax, label="% samples", shrink=0.6)
            ax.set_title(f"Bin occupancy — {title_suffix}  (% of all rows)", fontsize=10)
            fig.tight_layout()
            out = save_dir / fname
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  [viz] Bin-occupancy heatmap → {out}")

        cont_feats = [c for c in self.feature_cols
                      if self.feature_info[c]["type"] == "continuous"]
        cat_feats  = [c for c in self.feature_cols
                      if self.feature_info[c]["type"] == "categorical"]
        _heatmap(cont_feats, "continuous features", "bin_occupancy_continuous.png")
        _heatmap(cat_feats,  "categorical features", "bin_occupancy_categorical.png")

    def plot_all(
        self,
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        save_dir: "Path",
    ) -> None:
        """Generate all clinical feature visualizations into save_dir."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.plot_missing_rates(df, id_col, time_col, save_dir / "missing_rates.png")
        self.plot_feature_distributions(df, id_col, time_col, save_dir)
        self.plot_bin_occupancy_heatmap(df, id_col, time_col, save_dir)
        print(f"  [viz] All clinical plots saved to {save_dir}")


# ===========================================================================
# SECTION 0b — Cluster Count Tokenizer
# ===========================================================================

class ClusterCountTokenizer:
    """
    Tokenises per-bag cluster-count vectors (CLR-normalised floats).

    Each cluster count value is binned into n_quantiles bins fitted over the
    full dataset.  A NaN/missing token is reserved per cluster.

    Output per bag:
      token_ids  — int64 tensor of shape (n_clusters,)
      vocab      — list of {id, cluster_name, bin, label, [range]}

    Cluster names come from the annotation column (e.g. 'subcluster_renamed').
    """

    def __init__(self, n_quantiles: int = 4):
        self.n_quantiles  = n_quantiles
        self.cluster_names: List[str] = []
        self.bins: Dict[str, np.ndarray] = {}  # cluster_name → quantile edges
        self.vocab: Dict[int, dict] = {}
        self._n_tokens: int = 0
        self._token_offset: Dict[str, int] = {}  # cluster_name → start token id

    def fit(self, count_matrix: np.ndarray, cluster_names: List[str]) -> "ClusterCountTokenizer":
        """
        count_matrix : (n_bags, K) float — CLR counts across all bags
        cluster_names: list of K string names corresponding to columns
        """
        self.cluster_names = list(cluster_names)
        K = len(cluster_names)
        self._n_tokens = 0
        self.vocab = {}
        self.bins = {}
        self._token_offset = {}

        for k, name in enumerate(cluster_names):
            vals = count_matrix[:, k]
            self._token_offset[name] = self._n_tokens
            quantiles = np.nanquantile(vals, np.linspace(0, 1, self.n_quantiles + 1))
            quantiles = np.unique(quantiles)
            actual_bins = len(quantiles) - 1
            self.bins[name] = quantiles

            for b in range(actual_bins):
                lo, hi = quantiles[b], quantiles[b + 1]
                label = f"Q{b+1}/{actual_bins}"
                self.vocab[self._n_tokens] = {
                    "id": self._n_tokens, "cluster": name, "bin": b,
                    "label": label, "range": [float(lo), float(hi)],
                }
                self._n_tokens += 1

            # NaN / missing token
            self.vocab[self._n_tokens] = {
                "id": self._n_tokens, "cluster": name, "bin": -1,
                "label": "NaN", "range": None,
            }
            self._n_tokens += 1

        print(f"  [ClusterCountTokenizer] K={K}  {self._n_tokens} total tokens")
        return self

    def transform(self, count_vec: np.ndarray) -> np.ndarray:
        """count_vec: (K,) floats → int64 (K,)"""
        out = []
        for k, name in enumerate(self.cluster_names):
            val = count_vec[k] if k < len(count_vec) else np.nan
            edges = self.bins.get(name)
            nan_tok = self._token_offset[name] + len(edges) - 1
            if edges is None or np.isnan(val):
                out.append(nan_tok)
            else:
                idx = int(np.searchsorted(edges[1:-1], val))
                out.append(self._token_offset[name] + idx)
        return np.array(out, dtype=np.int64)

    def vocab_list(self) -> list:
        return [self.vocab[i] for i in sorted(self.vocab)]

    def to_dict(self) -> dict:
        return {
            "n_quantiles":    self.n_quantiles,
            "cluster_names":  self.cluster_names,
            "vocab":          {str(k): v for k, v in self.vocab.items()},
            "n_tokens":       self._n_tokens,
        }


# ===========================================================================
# SECTION 1 — Constants
# ===========================================================================

PRECOMPUTED_DIR = Path("/lustre/groups/aih/dinesh.haridoss/datasets/precomputed")
LUSTRE          = Path("/lustre/groups/aih/dinesh.haridoss")

BAL_H5AD      = str(LUSTRE / "datasets/08clad-4-annotated-v2-FIXED_date_from_id_dropped_recipient.h5ad")
HE_H5AD       = str(LUSTRE / "datasets/adata_v3.h5ad")
CT_H5AD       = str(LUSTRE / "datasets/combined_ct_embeddings_processed.h5ad")
RADIOMICS_CSV = str(LUSTRE / "datasets/radiomics_selected_metadata.csv")
CLINICAL_CSV  = str(LUSTRE / "month_binned_data_v2_oct_24.csv")

BAL_CENTROIDS_H5AD = str(PRECOMPUTED_DIR / "BAL_centroids.h5ad")
BAL_COUNTS_CSV     = str(PRECOMPUTED_DIR / "BAL_cluster_counts.csv")
HE_CENTROIDS_H5AD  = str(PRECOMPUTED_DIR / "HE_centroids.h5ad")
HE_COUNTS_CSV      = str(PRECOMPUTED_DIR / "HE_cluster_counts.csv")
CT_CENTROIDS_H5AD  = str(PRECOMPUTED_DIR / "CT_centroids.h5ad")
CT_COUNTS_CSV      = str(PRECOMPUTED_DIR / "CT_cluster_counts.csv")

BAL_CLUSTER_COL = "resolution_v2"
HE_CLUSTER_COL  = "subcluster_renamed"   # human-readable patch-type names
CT_CLUSTER_COL  = "leiden"

# Clinical columns that must be excluded — they are labels or directly derived from outcomes
CLINICAL_LABEL_COLS = [
    "biopsy_grade_A",           # ACR biopsy rejection grade — the primary label
    "delta_fvc_from_previous",  # change in FVC since prior visit — outcome-derived
    "pseudoslope_fvc",          # longitudinal FVC slope — outcome-derived
    "delta_fev1_from_previous", # change in FEV1 since prior visit — outcome-derived
    "pseudoslope_fev1",         # longitudinal FEV1 slope — outcome-derived
]

BAL_IDENTIFIER_COL = "record_id";  BAL_TIME_COL = "date_from_id"
HE_IDENTIFIER_COL  = "record_id";  HE_TIME_COL  = "biopsy_date"
CT_IDENTIFIER_COL  = "Patient";    CT_TIME_COL  = "date_of_CT"

SURVIVAL_FILE = str(LUSTRE / "datasets/Mortality_updated.csv")
METADATA_ENC  = str(LUSTRE / "datasets/ACR_biopsies_clean_dates_fixed_encoded.csv")
METADATA_RAW  = str(LUSTRE / "datasets/ACR_biopsies_clean_dates_fixed.csv")
SPLITS_CSV    = str(SCRIPT_DIR / "multimodal_splits.csv")

DEFAULT_CACHE = str(LUSTRE / "datasets/mil")

DATA_PATHS = {
    "sc_rna":             BAL_H5AD,
    "BAL_centroids_h5ad": BAL_CENTROIDS_H5AD,
    "BAL_counts_csv":     BAL_COUNTS_CSV,
    "h&e":                HE_H5AD,
    "HE_centroids_h5ad":  HE_CENTROIDS_H5AD,
    "HE_counts_csv":      HE_COUNTS_CSV,
    "ct_scan":            CT_H5AD,
    "CT_centroids_h5ad":  CT_CENTROIDS_H5AD,
    "CT_counts_csv":      CT_COUNTS_CSV,
    "radiomics":          RADIOMICS_CSV,
    "clinical":           CLINICAL_CSV,
}
KINDS = {
    "sc_rna": "structured",  "BAL_centroids_h5ad": "structured",  "BAL_counts_csv": "simple",
    "h&e":    "structured",  "HE_centroids_h5ad":  "structured",  "HE_counts_csv":  "simple",
    "ct_scan":"structured",  "CT_centroids_h5ad":  "structured",  "CT_counts_csv":  "simple",
    "radiomics": "simple",   "clinical": "simple",
}
IDENTIFIER_COLS = {
    "sc_rna":             BAL_IDENTIFIER_COL,
    "BAL_centroids_h5ad": BAL_IDENTIFIER_COL,
    "BAL_counts_csv":     BAL_IDENTIFIER_COL,
    "h&e":                HE_IDENTIFIER_COL,
    "HE_centroids_h5ad":  HE_IDENTIFIER_COL,
    "HE_counts_csv":      HE_IDENTIFIER_COL,
    "ct_scan":            CT_IDENTIFIER_COL,
    "CT_centroids_h5ad":  CT_IDENTIFIER_COL,
    "CT_counts_csv":      CT_IDENTIFIER_COL,
    "radiomics":          "Patient",
    "clinical":           "record_id",
}
TIME_COLS = {
    "sc_rna":             BAL_TIME_COL,
    "BAL_centroids_h5ad": BAL_TIME_COL,
    "BAL_counts_csv":     BAL_TIME_COL,
    "h&e":                HE_TIME_COL,
    "HE_centroids_h5ad":  HE_TIME_COL,
    "HE_counts_csv":      HE_TIME_COL,
    "ct_scan":            CT_TIME_COL,
    "CT_centroids_h5ad":  CT_TIME_COL,
    "CT_counts_csv":      CT_TIME_COL,
    "radiomics":          "Exam_Date",
    "clinical":           "spiro_date",
}
FEATURE_SPACES = {
    "sc_rna":             "X_scVI",
    "BAL_centroids_h5ad": "X",
    "h&e":                "X",
    "HE_centroids_h5ad":  "X",
    "ct_scan":            "X",
    "CT_centroids_h5ad":  "X",
}
LEIDEN_COLS  = {"sc_rna": BAL_CLUSTER_COL, "h&e": HE_CLUSTER_COL, "ct_scan": CT_CLUSTER_COL}
SAMPLE_RATE  = {k: 0 for k in ["sc_rna","BAL_centroids_h5ad","h&e","HE_centroids_h5ad","ct_scan","CT_centroids_h5ad"]}
ANCHOR_MODALITY = ["sc_rna","BAL_centroids_h5ad","BAL_counts_csv" ,"h&e","HE_centroids_h5ad", "HE_counts_csv",  "ct_scan","CT_counts_csv", "CT_centroids_h5ad", "radiomics", "clinical"]

# Spatial coordinates to extract per instance for each structured modality.
# HE: pixel coords of the top-left tile corner on the WSI slide.
# CT: voxel-space position of each patch (depth, height, width).
SPATIAL_COORDS = {
    "h&e":     ["tile_left", "tile_top"],
    "ct_scan": ["d_voxel", "h_voxel", "w_voxel"],
}

KEY_REMAP: Dict[str, str] = {
    "sc_rna":             "BAL_cells",
    "BAL_centroids_h5ad": "BAL_centroids",
    "BAL_counts_csv":     "BAL_counts",
    "h&e":                "HE_cells",
    "h&e_coords":         "HE_coords",
    "HE_centroids_h5ad":  "HE_centroids",
    "HE_counts_csv":      "HE_counts",
    "ct_scan":            "CT_cells",
    "CT_centroids_h5ad":  "CT_centroids",
    "CT_counts_csv":      "CT_counts",
    "radiomics":          "CT_radiomics",
    "clinical":           "Clinical",
}

# cells modality → (centroids modality, counts modality)
MODALITY_GROUPS = {
    "sc_rna":  ("BAL_centroids_h5ad", "BAL_counts_csv"),
    "h&e":     ("HE_centroids_h5ad",  "HE_counts_csv"),
    "ct_scan": ("CT_centroids_h5ad",  "CT_counts_csv"),
}


# ===========================================================================
# SECTION 2 — Label function
# ===========================================================================

def _binarise_acr(grade: str) -> int:
    if grade is None:
        return 0
    g = str(grade).strip()
    return 1 if ("A1" in g or "A2" in g) else 0


def label_fn(sample: dict) -> int:
    meta = sample.get("metadata", {})
    enc  = meta.get("acr_encoded")
    if enc is not None and not (isinstance(enc, float) and np.isnan(enc)):
        return int(enc)
    raw = meta.get("ACR Status/Grade")
    if raw is not None:
        return _binarise_acr(raw)
    return 0


# ===========================================================================
# SECTION 3 — Build dataset
# ===========================================================================

def _encode_acr_labels(raw_csv: str, out_csv: str):
    df = pd.read_csv(raw_csv)
    col = "ACR Status/Grade"
    if col in df.columns:
        df["acr_encoded"] = df[col].apply(
            lambda v: _binarise_acr(v) if pd.notna(v) else float("nan"))
    df.to_csv(out_csv, index=False)
    print(f"  Encoded ACR labels saved to {out_csv}")


def _build_dataset(
    data_paths_override: Optional[Dict[str, str]] = None,
) -> MultimodalTimeseriesDataset:
    """Build dataset, optionally pointing at filtered/overridden data paths."""
    if not Path(METADATA_ENC).exists():
        print("  [setup] Encoding ACR labels...")
        _encode_acr_labels(METADATA_RAW, METADATA_ENC)

    paths = data_paths_override if data_paths_override is not None else DATA_PATHS

    ds = MultimodalTimeseriesDataset(
        data_paths       = paths,
        identifier_cols  = IDENTIFIER_COLS,
        time_cols        = TIME_COLS,
        anchor_modality  = ANCHOR_MODALITY,
        window           = 45 * 24 * 3600,
        kinds            = KINDS,
        where            = FEATURE_SPACES,
        leiden_cluster   = LEIDEN_COLS,
        sample_rate      = SAMPLE_RATE,
        return_unpaired  = True,
        min_modalities   = 1,
        survival_file     = SURVIVAL_FILE,
        survival_id_col   = "record_id",
        event_time_cols   = {"CLAD": "clad_date", "Death": "date_of_death"},
        event_status_cols = {"CLAD": "clad",      "Death": "death_status"},
        event_mode        = "date",
        metadata_file     = METADATA_ENC,
        metadata_id_col   = "Record id",
        metadata_time_col = "Biopsy Date",
        metadata_cols     = ["ACR Status/Grade", "acr_encoded"],
        transplant_date_source_mod = "sc_rna",
        transplant_date_col        = "tx_date",
        spatial_coords             = SPATIAL_COORDS,
        exclude_feature_cols       = {"clinical": CLINICAL_LABEL_COLS},
    )
    return ds


def _fit_clinical_tokenizer() -> ClinicalFeatureTokenizer:
    """Fit ClinicalFeatureTokenizer on the full clinical CSV."""
    print("  [ClinicalTokenizer] fitting on raw clinical CSV...")
    df = pd.read_csv(CLINICAL_CSV)
    tok = ClinicalFeatureTokenizer(n_quantiles=4, categorical_threshold=10)
    tok.fit(df, id_col="record_id", time_col="spiro_date", exclude_cols=CLINICAL_LABEL_COLS)
    return tok


def build_base_dataset() -> MultimodalTimeseriesDataset:
    print("\n  Building MultimodalTimeseriesDataset (11 inputs)...")
    ds = _build_dataset()
    print(f"  Dataset: {len(ds)} aligned timepoints")
    return ds


# ===========================================================================
# SECTION 4 — Patient-filtered data sources (diagnose fast-path)
# ===========================================================================

def _read_ids_from_source(path: str, id_col: str) -> Set[str]:
    """
    Read only the identifier column from a data source — no feature matrix loaded.
    h5ad: reads backed (obs only).  csv: reads single column.
    """
    p = Path(path)
    if not p.exists():
        return set()
    try:
        if p.suffix.lower() == ".h5ad":
            adata = ad.read_h5ad(path, backed="r")
            ids   = set(adata.obs[id_col].astype(str).str.strip().unique())
            adata.file.close()
        else:
            ids = set(pd.read_csv(path, usecols=[id_col])[id_col].astype(str).str.strip().unique())
        return ids
    except Exception as e:
        print(f"    [warn] could not read ids from {path}: {e}")
        return set()


def _filter_h5ad(src: str, id_col: str, keep_ids: Set[str], dst: str) -> int:
    adata    = ad.read_h5ad(src)
    mask     = adata.obs[id_col].astype(str).str.strip().isin(keep_ids)
    adata[mask].copy().write_h5ad(dst)
    return int(mask.sum())


def _filter_csv(src: str, id_col: str, keep_ids: Set[str], dst: str) -> int:
    df   = pd.read_csv(src)
    mask = df[id_col].astype(str).str.strip().isin(keep_ids)
    df[mask].reset_index(drop=True).to_csv(dst, index=False)
    return int(mask.sum())


def _build_filtered_data_paths(
    patient_ids: List[str],
    tmp_dir:     Path,
    verbose:     bool = True,
) -> Dict[str, str]:
    """
    For every physical file referenced in DATA_PATHS:
      1. Determine the identifier column used by that file (from IDENTIFIER_COLS).
      2. Check which of the requested patient_ids actually exist in that file.
         (The column name differs per file but values are the same, e.g. LT001.)
      3. Write a filtered copy to tmp_dir.
      4. Return new DATA_PATHS pointing to the filtered copies.

    Each physical file is only read + written once even if multiple modalities
    share the same path.
    """
    wanted = set(str(p).strip() for p in patient_ids)

    # ── group modalities by physical file path ───────────────────────────────
    # path → first modality that references it (to get id_col)
    path_to_first_mod: Dict[str, str] = {}
    for mod, path in DATA_PATHS.items():
        if path not in path_to_first_mod:
            path_to_first_mod[path] = mod

    # ── filter each unique physical file once ────────────────────────────────
    path_to_filtered: Dict[str, str] = {}

    for src_path, first_mod in path_to_first_mod.items():
        id_col  = IDENTIFIER_COLS[first_mod]
        p       = Path(src_path)
        dst     = str(tmp_dir / p.name)

        # All modality names that share this file (for display)
        shared_mods = [m for m, pp in DATA_PATHS.items() if pp == src_path]

        available = _read_ids_from_source(src_path, id_col)
        keep      = wanted & available

        if verbose:
            print(f"\n  ▸ {p.name}")
            print(f"    modalities   : {shared_mods}")
            print(f"    id_col       : {id_col!r}")
            print(f"    ids in file  : {len(available)}")
            print(f"    requested    : {sorted(wanted)}")
            print(f"    matched      : {sorted(keep)}")
            if not keep:
                print(f"    {_warn('⚠  0 matches — patient IDs not found in this file.')}")
                print(f"    {_warn('   Possible causes: patient truly absent, typo, or not in this modality.')}")

        try:
            if p.suffix.lower() == ".h5ad":
                n = _filter_h5ad(src_path, id_col, keep, dst)
            else:
                n = _filter_csv(src_path, id_col, keep, dst)
            if verbose:
                print(f"    kept rows    : {n}  →  {dst}")
        except Exception as e:
            print(f"    {_err(f'[ERROR] filtering {p.name}: {e}')}")
            shutil.copy2(src_path, dst)   # fallback: use original (slow but safe)

        path_to_filtered[src_path] = dst

    # ── remap DATA_PATHS to filtered copies ──────────────────────────────────
    return {mod: path_to_filtered[DATA_PATHS[mod]] for mod in DATA_PATHS}


def _pick_default_patients(n: int) -> List[str]:
    """Pick the first n unique patient IDs from the BAL h5ad (record_id column)."""
    try:
        adata = ad.read_h5ad(BAL_H5AD, backed="r")
        pids  = sorted(adata.obs[BAL_IDENTIFIER_COL].astype(str).str.strip().unique())
        adata.file.close()
        return pids[:n]
    except Exception as e:
        print(f"  [warn] could not read default patients from BAL h5ad: {e}")
        return []


# ===========================================================================
# SECTION 5 — ANSI colour helpers
# ===========================================================================

_RED = "\033[91m"; _YLW = "\033[93m"; _GRN = "\033[92m"
_CYN = "\033[96m"; _BOLD = "\033[1m"; _RST = "\033[0m"

def _c(t, code): return f"{code}{t}{_RST}"
def _ok(t):      return _c(t, _GRN)
def _warn(t):    return _c(t, _YLW)
def _err(t):     return _c(t, _RED)
def _hdr(t):     return _c(t, _BOLD + _CYN)


# ===========================================================================
# SECTION 6 — DIAGNOSE
# ===========================================================================

def _check_orphan_issue(
    ds:          MultimodalTimeseriesDataset,
    pid:         str,
    sample:      dict,
    window_days: float,
) -> List[str]:
    issues = []
    mods   = sample["modalities"]
    for cells_key, (centroids_key, counts_key) in MODALITY_GROUPS.items():
        cells_e     = mods.get(cells_key)
        centroids_e = mods.get(centroids_key)
        counts_e    = mods.get(counts_key)
        if cells_e is None:
            continue
        if centroids_e is not None and counts_e is not None:
            continue

        cells_time  = cells_e["time"]
        avail_c = sorted(t for t, _ in ds.time_dict.get(centroids_key, {}).get(pid, []))
        avail_k = sorted(t for t, _ in ds.time_dict.get(counts_key,    {}).get(pid, []))

        def _closest(ts, ref):
            if not ts:
                return None, None
            pairs = sorted((abs((t - ref).total_seconds()), t) for t in ts)
            return pairs[0]

        c_dt, c_t     = _closest(avail_c, cells_time)
        window_sec    = window_days * 86400

        if not avail_c and not avail_k:
            root = "patient has NO entries in precomputed files at all"
        elif c_dt is None or c_dt > window_sec:
            root = (
                f"closest centroid ts={c_t} (Δ={c_dt/86400:.1f}d > window={window_days:.0f}d)"
                if c_dt is not None else "NO centroid entries for this patient"
            )
        else:
            root = f"within-window match exists ({c_t}) but alignment still missed it"

        missing = [k for k, v in [(centroids_key, centroids_e), (counts_key, counts_e)] if v is None]
        issues.append(
            f"  {_err('ORPHAN')} {cells_key} present (t={cells_time.date()}) "
            f"but {', '.join(missing)} missing.\n"
            f"    Root-cause hint : {root}\n"
            f"    Centroid times  : {[str(t.date()) for t in avail_c[:6]]}{'…' if len(avail_c)>6 else ''}\n"
            f"    Counts   times  : {[str(t.date()) for t in avail_k[:6]]}{'…' if len(avail_k)>6 else ''}"
        )
    return issues


def diagnose(
    patient_ids: Optional[List[str]],
    n_patients:  int  = 3,
    verbose_raw: bool = True,
):
    """
    Fast diagnose mode.

    1. Resolves patient IDs (auto-picks from BAL if none given).
    2. Filters every h5ad / CSV to only those patients → tiny temp copies.
    3. Builds the dataset on the filtered copies (fast — seconds not minutes).
    4. Prints per-file filter summary, modality availability table,
       global orphan scan, and per-patient alignment detail.
    5. Cleans up temp files.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="diagnose_"))
    print(_hdr(f"\n{'='*80}"))
    print(_hdr("  DIAGNOSE MODE  (fast — data sources filtered to requested patients)"))
    print(_hdr(f"  Temp dir: {tmp_dir}"))
    print(_hdr(f"{'='*80}"))

    # ── Resolve patient IDs ──────────────────────────────────────────────────
    if not patient_ids:
        print(f"\n  No --diagnose_patients given — auto-picking first {n_patients} "
              f"patients from BAL ({BAL_IDENTIFIER_COL})…")
        patient_ids = _pick_default_patients(n_patients)
        if not patient_ids:
            print(_err("  Could not determine default patients. "
                       "Pass --diagnose_patients explicitly."))
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return

    print(f"\n  Requested patient IDs ({len(patient_ids)}): {patient_ids}")

    # ── Filter data sources ──────────────────────────────────────────────────
    print(_hdr(f"\n── Filtering data sources ──"))
    try:
        filtered_paths = _build_filtered_data_paths(patient_ids, tmp_dir, verbose=True)
    except Exception as e:
        print(_err(f"  Filtering failed: {e}"))
        traceback.print_exc()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    # ── Build filtered dataset ───────────────────────────────────────────────
    print(_hdr(f"\n── Building filtered dataset ──"))
    try:
        ds = _build_dataset(data_paths_override=filtered_paths)
    except Exception as e:
        print(_err(f"  Dataset build failed: {e}"))
        traceback.print_exc()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return

    n_samples   = len(ds)
    all_pids    = sorted({s["identifier"] for s in ds.flat_samples})
    window_days = ds.window / 86400

    print(_hdr(f"\n{'='*80}"))
    print(_hdr(f"  Filtered dataset: {n_samples} sample(s)  |  "
               f"{len(all_pids)} patient(s)  |  window={window_days:.0f}d"))
    print(_hdr(f"  Anchors: {ds.anchor_modality}"))
    print(_hdr(f"{'='*80}"))

    missing_pids = [p for p in patient_ids if p not in all_pids]
    if missing_pids:
        print(_warn(f"\n  ⚠  Requested but not found in any modality: {missing_pids}"))
        print(_warn(f"     These patients have no data across any of the 11 modalities."))
        print(_warn(f"     Check for typos — IDs should be in the form LT001, LT002, …"))

    # ── Modality availability table ──────────────────────────────────────────
    print(_hdr("\n── Modality availability in filtered dataset ──"))
    mw = max(len(m) for m in ds.modalities) + 2
    print(f"  {'Modality':<{mw}}  {'id_col':>20}  {'#patients':>10}  {'#timepoints':>12}  {'kind':>12}")
    print(f"  {'-'*mw}  {'-'*20}  {'-'*10}  {'-'*12}  {'-'*12}")
    for mod in ds.modalities:
        tmap  = ds.time_dict.get(mod, {})
        n_pts = len(tmap)
        n_tps = sum(len(v) for v in tmap.values())
        kind  = ds.kinds.get(mod, "?")
        icol  = IDENTIFIER_COLS.get(mod, "?")
        flag  = "" if n_pts > 0 else _warn(" ← no data")
        print(f"  {mod:<{mw}}  {icol:>20}  {n_pts:>10}  {n_tps:>12}  {kind:>12}{flag}")

    # ── Global orphan scan ───────────────────────────────────────────────────
    print(_hdr("\n── Orphan scan (cells present but centroids/counts missing) ──"))
    orphan_counts:       Dict[str, int] = {k: 0 for k in MODALITY_GROUPS}
    total_cells_present: Dict[str, int] = {k: 0 for k in MODALITY_GROUPS}
    for sample in ds.flat_samples:
        mods = sample["modalities"]
        for ck, (centk, cntk) in MODALITY_GROUPS.items():
            if mods.get(ck) is not None:
                total_cells_present[ck] += 1
                if mods.get(centk) is None or mods.get(cntk) is None:
                    orphan_counts[ck] += 1
    for ck, (centk, cntk) in MODALITY_GROUPS.items():
        n_c = total_cells_present[ck]
        n_o = orphan_counts[ck]
        rate = n_o / max(n_c, 1) * 100
        flag = _err(f"{n_o:>5} orphans ({rate:.1f}%)") if n_o > 0 else _ok("    0 orphans")
        print(f"  {ck:<30}  cells present in {n_c:>4} samples  →  {flag}")

    # ══════════════════════════════════════════════════════════════════════════
    # Per-patient detail
    # ══════════════════════════════════════════════════════════════════════════
    flat_id_map = {id(s): i for i, s in enumerate(ds.flat_samples)}
    inspect_pids = [p for p in patient_ids if p in all_pids] or all_pids

    for pid in inspect_pids:
        patient_samples = sorted(
            [s for s in ds.flat_samples if s["identifier"] == pid],
            key=lambda s: s["anchor_time"],
        )
        print(_hdr(f"\n{'─'*80}"))
        print(_hdr(f"  PATIENT: {pid}   ({len(patient_samples)} aligned sample(s))"))
        print(_hdr(f"{'─'*80}"))

        # Raw time_dict entries
        if verbose_raw:
            print(_c("\n  Raw time_dict entries per modality (before alignment):", _BOLD))
            for mod in ds.modalities:
                entries = ds.time_dict.get(mod, {}).get(pid, [])
                icol    = IDENTIFIER_COLS.get(mod, "?")
                if entries:
                    ts_str = "  ".join(str(t.date()) for t, _ in sorted(entries))
                    print(f"    {mod:<35} [{icol:>12}]  "
                          f"{_ok(str(len(entries))+' tp')}  →  {ts_str}")
                else:
                    print(f"    {mod:<35} [{icol:>12}]  {_warn('no entries')}")

        # Aligned samples
        print(_c(f"\n  Aligned samples:", _BOLD))
        for s_idx, sample in enumerate(patient_samples):
            anchor_t = sample["anchor_time"]
            unpaired = sample.get("unpaired", False)
            print(f"\n    [{s_idx:>3}]  anchor={anchor_t.date()}"
                  + (f"  {_warn('[UNPAIRED]')}" if unpaired else ""))

            raw_item = None
            try:
                flat_idx = flat_id_map[id(sample)]
                raw_item = ds[flat_idx]
            except Exception as e:
                print(f"      {_warn(f'[could not fetch item: {e}]')}")

            for mod in ds.modalities:
                entry = sample["modalities"].get(mod)
                icol  = IDENTIFIER_COLS.get(mod, "?")

                if entry is not None:
                    t_str  = str(entry["time"].date()) if entry.get("time") else "?"
                    dt_sec = entry.get("dt")
                    dt_str = f"Δ={dt_sec/86400:+.1f}d" if dt_sec is not None else "Δ=anchor"
                    shape_str = "shape=?"
                    if raw_item is not None:
                        t = raw_item["inputs"].get(mod)
                        shape_str = str(tuple(t.shape)) if t is not None else "shape=None"
                    print(f"      {_ok('✓'):3}  {mod:<35}  t={t_str}  "
                          f"{dt_str:>12}  {shape_str}  [{icol}]")
                else:
                    any_entries = bool(ds.time_dict.get(mod, {}).get(pid))
                    if any_entries:
                        print(f"      {_warn('~'):3}  {mod:<35}  "
                              f"{_warn('data exists but outside window')}  [{icol}]")
                    else:
                        print(f"      {_err('✗'):3}  {mod:<35}  "
                              f"{_err('absent — no data for this patient')}  [{icol}]")

            # Orphan check
            issues = _check_orphan_issue(ds, pid, sample, window_days)
            if issues:
                print(f"\n      {_err('⚠  ORPHAN ISSUES:')}")
                for iss in issues:
                    print(iss)

            # Survival + label
            if raw_item is not None:
                surv = raw_item.get("survival", {})
                if surv:
                    parts = [f"{e}(status={v.get('status','?')}, days={v.get('days','?')})"
                             for e, v in surv.items()]
                    print(f"\n      Survival : {'  '.join(parts)}")
                meta  = raw_item.get("metadata", {})
                acr   = meta.get("ACR Status/Grade") or meta.get("acr_encoded")
                if acr is not None:
                    print(f"      Label    : ACR={acr!r}  →  binary={label_fn(raw_item)}")

    # ── Final verdict ─────────────────────────────────────────────────────────
    total_orphans = sum(orphan_counts.values())
    if total_orphans > 0:
        print(_err(f"\n{'='*80}"))
        print(_err(f"  ⚠  {total_orphans} ORPHAN INSTANCE(S) — likely causes:"))
        print(_err(f"    1. ID mismatch  : --patient_col in precompute_modality_clusters.py"))
        print(_err(f"                      doesn't match IDENTIFIER_COLS for that modality"))
        print(_err(f"    2. Time mismatch: date column differs between raw h5ad and precomputed"))
        print(_err(f"                      (e.g. biopsy_date vs date_from_id have different formats)"))
        print(_err(f"    3. Window tight : derived bags fall just outside the {window_days:.0f}d window"))
        print(_err(f"    4. Precompute dropped patient: NaN IDs/times during clustering"))
        print(_err(f"{'='*80}\n"))
    else:
        print(_ok(f"\n{'='*80}"))
        print(_ok(f"  ✓  No orphan issues detected."))
        print(_ok(f"{'='*80}\n"))

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"  Temp files removed: {tmp_dir}")


# ===========================================================================
# SECTION 7 — Per-sample extraction and saving
# ===========================================================================

def _remap_sample(
    raw: dict,
    clinical_tok: Optional["ClinicalFeatureTokenizer"] = None,
    cluster_count_toks: Optional[Dict[str, "ClusterCountTokenizer"]] = None,
) -> dict:
    inputs: Dict[str, Optional[torch.Tensor]] = {}
    for old_key, new_key in KEY_REMAP.items():
        t = raw["inputs"].get(old_key)
        inputs[new_key] = t.float().cpu() if t is not None else None

    def _ts_to_str(ts):
        if ts is None:
            return None
        if hasattr(ts, "isoformat"):
            return ts.isoformat()
        return str(ts)

    at = _ts_to_str(raw.get("anchor_time"))

    def _clean(d):
        out = {}
        for k, v in (d or {}).items():
            if isinstance(v, (str, int, float, bool, type(None))):
                out[k] = v
            else:
                try:    out[k] = float(v)
                except: out[k] = str(v)
        return out

    # Per-modality measurement times
    modality_times = {
        new_key: _ts_to_str((raw.get("modality_times") or {}).get(old_key))
        for old_key, new_key in KEY_REMAP.items()
    }

    # Transplant date (ISO string or None)
    tx_date = _ts_to_str(raw.get("transplant_date"))

    # Cluster labels per instance per modality (remapped keys)
    cluster_labels: Dict[str, Optional[list]] = {}
    raw_classifier = raw.get("classifier") or {}
    for old_key, new_key in KEY_REMAP.items():
        cl = raw_classifier.get(old_key)
        cluster_labels[new_key] = [str(c) for c in cl] if cl is not None else None

    # Spatial coords per modality (remapped keys): Tensor(N, n_dims) or None
    coords: Dict[str, Optional[torch.Tensor]] = {}
    raw_coords = raw.get("coords") or {}
    for old_key, new_key in KEY_REMAP.items():
        c = raw_coords.get(old_key)
        coords[new_key] = c.float().cpu() if c is not None else None

    # Clinical tokenization
    clinical_token_ids: Optional[torch.Tensor] = None
    clinical_vocab: Optional[list] = None
    if clinical_tok is not None:
        raw_clin_row = raw.get("_clinical_raw_row")
        if raw_clin_row is not None:
            tids = clinical_tok.transform_row(raw_clin_row)
            clinical_token_ids = torch.tensor(tids, dtype=torch.int64)
            clinical_vocab = clinical_tok.vocab_list()

    # Bag-level aggregates: centroids, cluster-count token IDs, cluster names
    bag_centroids:       Dict[str, Optional[torch.Tensor]] = {}
    bag_count_token_ids: Dict[str, Optional[torch.Tensor]] = {}
    bag_cluster_names:   Dict[str, Optional[list]]         = {}
    bag_count_vocab:     Dict[str, Optional[list]]         = {}

    raw_bag_agg = raw.get("bag_aggregates") or {}
    for old_key, new_key in KEY_REMAP.items():
        agg = raw_bag_agg.get(old_key)
        if agg is None:
            bag_centroids[new_key]       = None
            bag_count_token_ids[new_key] = None
            bag_cluster_names[new_key]   = None
            bag_count_vocab[new_key]     = None
            continue

        centroids = agg["centroids"]
        bag_centroids[new_key] = (
            centroids.float().cpu() if isinstance(centroids, torch.Tensor)
            else torch.tensor(centroids, dtype=torch.float32)
        )
        bag_cluster_names[new_key] = agg.get("cluster_names")

        # Tokenise cluster count vector
        cct = (cluster_count_toks or {}).get(new_key)
        if cct is not None:
            counts_np = (
                agg["counts"].cpu().numpy() if isinstance(agg["counts"], torch.Tensor)
                else np.array(agg["counts"])
            )
            bag_count_token_ids[new_key] = torch.tensor(
                cct.transform(counts_np), dtype=torch.int64
            )
            bag_count_vocab[new_key] = cct.vocab_list()
        else:
            bag_count_token_ids[new_key] = None
            bag_count_vocab[new_key]     = None

    out = {
        "inputs":             inputs,
        "label":              label_fn(raw),
        "identifier":         str(raw.get("identifier", "")),
        "anchor_time":        at,
        "modality_times":     modality_times,
        "transplant_date":    tx_date,
        # Per-instance cluster/cell-type labels
        "cluster_labels":     cluster_labels,
        # Per-instance spatial coordinates
        "coords":             coords,
        # Bag-level aggregates (computed inline from annotation)
        "bag_centroids":         bag_centroids,
        "bag_count_token_ids":   bag_count_token_ids,
        "bag_cluster_names":     bag_cluster_names,
        "metadata":           _clean(raw.get("metadata", {})),
        "survival":           _clean(raw.get("survival", {})),
    }
    if clinical_token_ids is not None:
        out["clinical_token_ids"] = clinical_token_ids
        out["clinical_vocab"]     = clinical_vocab
    for new_key, voc in bag_count_vocab.items():
        if voc is not None:
            out.setdefault("bag_count_vocab", {})[new_key] = voc
    return out


def _save_one(args_tuple):
    idx, raw_sample, out_path, clinical_tok, cluster_count_toks = args_tuple
    try:
        sample = _remap_sample(raw_sample, clinical_tok=clinical_tok,
                               cluster_count_toks=cluster_count_toks)
        torch.save(sample, out_path)
        return idx, sample["label"], sample["identifier"], sample["anchor_time"], None
    except Exception as e:
        return idx, -1, "", "", traceback.format_exc()


# ===========================================================================
# SECTION 8 — Dimension introspection
# ===========================================================================

def _collect_dims(ds: MultimodalTimeseriesDataset) -> Dict[str, int]:
    dims: Dict[str, int] = {}
    for i in range(min(50, len(ds))):
        try:
            raw = ds[i]
        except Exception:
            continue
        for old_key, new_key in KEY_REMAP.items():
            if new_key in dims:
                continue
            t = raw["inputs"].get(old_key)
            if t is not None:
                dims[new_key] = t.shape[-1]
        if len(dims) == len(KEY_REMAP):
            break
    return dims


# ===========================================================================
# SECTION 9 — Main precompute loop
# ===========================================================================

def precompute(
    cache_dir:      Path,
    skip_existing:  bool = False,
    workers:        int  = 1,
    progress_every: int  = 100,
    max_samples:    int  = 0,   # 0 = all; >0 = stop after N samples (preview mode)
):
    samples_dir = cache_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Precompute → {cache_dir}")
    print(f"  skip_existing={skip_existing}  workers={workers}")
    print(f"{'='*70}\n")

    ds = build_base_dataset()
    n  = len(ds)
    print(f"\n  Total samples to cache: {n}")

    ds.plot_alignment(show_aligned=True, max_patients=200, save_path=str(cache_dir))

    import pickle
    pkl_path = cache_dir / "dataset.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(ds, f)
    print(f"  Dataset cached to: {pkl_path}")

    # Fit clinical tokenizer on raw (un-normalized) clinical CSV
    clinical_tok = _fit_clinical_tokenizer()
    vocab_path = cache_dir / "clinical_vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(clinical_tok.to_dict(), f, indent=2)
    print(f"  Clinical vocab saved: {vocab_path}  ({clinical_tok._n_tokens} tokens, "
          f"{len(clinical_tok.feature_cols)} features)")

    # Load raw clinical df for tokenizing individual rows at precompute time.
    # (ds.df_dict["clinical"] is already StandardScaler-normalized; we need originals.)
    clinical_raw_df = pd.read_csv(CLINICAL_CSV)
    clinical_raw_df["record_id"] = clinical_raw_df["record_id"].astype(str).str.strip()
    clinical_raw_df["spiro_date"] = pd.to_datetime(clinical_raw_df["spiro_date"], errors="coerce")
    clinical_raw_df = clinical_raw_df.dropna(subset=["spiro_date"])
    clinical_raw_df = clinical_raw_df.drop_duplicates(
        subset=["record_id", "spiro_date"], keep="first"
    ).reset_index(drop=True)

    # Generate clinical feature visualization plots
    print("  Generating clinical feature plots…")
    clinical_tok.plot_all(clinical_raw_df, "record_id", "spiro_date", cache_dir / "clinical_plots")
    print(f"  Clinical plots saved to: {cache_dir / 'clinical_plots'}")

    print("  Collecting feature dims from first 50 samples…")
    dims = _collect_dims(ds)
    print(f"  Dims: {dims}")

    # --- Pass 1: collect alignment metadata only — no tensors loaded yet --------
    # flat_samples entries are lightweight dicts (identifier + anchor_time + index
    # pointers).  We defer calling ds[idx] — which actually extracts tensors from
    # the in-memory AnnData objects — until the moment we are about to save that
    # sample, so only ONE sample's worth of tensors lives in RAM at a time.
    to_process = []   # list of (idx, out_path) — no tensors
    n_skip = 0
    for idx in range(n):
        out_path = samples_dir / f"{idx:05d}.pt"
        if skip_existing and out_path.exists():
            n_skip += 1
        else:
            to_process.append((idx, out_path))

    # Preview mode: truncate to first max_samples entries
    if max_samples > 0 and len(to_process) > max_samples:
        print(f"  [preview] max_samples={max_samples} — truncating to first {max_samples} samples")
        to_process = to_process[:max_samples]

    total = len(to_process)
    if total == 0:
        print("  All samples already cached — nothing to do.")
    else:
        print(f"  Saving {total} samples (skipping {n_skip} existing)…\n")

    manifest_rows = []
    n_errors = 0
    t0 = time.time()

    def _attach_clinical_row(raw_sample: dict, sample_meta: dict) -> dict:
        """Attach raw (un-normalized) clinical row to raw_sample for tokenization."""
        clin_entry = sample_meta["modalities"].get("clinical")
        if clin_entry is not None and not clinical_raw_df.empty:
            row_idx = clin_entry["idx"]
            if row_idx < len(clinical_raw_df):
                raw_sample["_clinical_raw_row"] = clinical_raw_df.iloc[row_idx]
        return raw_sample

    if workers <= 1 or total == 0:
        # --- Sequential streaming: load → save → free, one sample at a time ----
        for i, (idx, out_path) in enumerate(to_process):
            try:
                raw_sample = ds[idx]   # loads tensors for this one sample only
                _attach_clinical_row(raw_sample, ds.flat_samples[idx])
            except Exception as e:
                print(f"  [ERROR] sample {idx} load: {e}")
                n_errors += 1
                manifest_rows.append({"idx": idx, "identifier": "", "anchor_time": "", "label": -1})
                continue

            _, label, ident, at, err = _save_one((idx, raw_sample, out_path, clinical_tok))
            del raw_sample  # free tensors immediately — do not accumulate in RAM

            if err:
                print(f"  [ERROR] sample {idx}: {err}")
                n_errors += 1
                label, ident, at = -1, "", ""
            manifest_rows.append({"idx": idx, "identifier": ident, "anchor_time": at, "label": label})
            if (i + 1) % progress_every == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                rate    = (i + 1) / max(elapsed, 1e-6)
                eta     = (total - i - 1) / max(rate, 1e-6)
                print(f"  [{i+1:>5}/{total}]  elapsed={elapsed/60:.1f}m  "
                      f"rate={rate:.1f}/s  ETA={eta/60:.1f}m  errors={n_errors}")
    else:
        # --- Parallel streaming: bounded in-flight queue -----------------------
        max_in_flight = workers * 2
        pending: dict = {}   # future → idx
        task_iter = iter(to_process)
        done = 0

        def _submit_next(pool):
            try:
                nidx, npath = next(task_iter)
            except StopIteration:
                return
            raw = ds[nidx]
            _attach_clinical_row(raw, ds.flat_samples[nidx])
            fut = pool.submit(_save_one, (nidx, raw, npath, clinical_tok))
            pending[fut] = nidx
            del raw

        with ProcessPoolExecutor(max_workers=workers) as pool:
            for _ in range(max_in_flight):
                _submit_next(pool)

            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for fut in completed:
                    idx2, label, ident, at, err = fut.result()
                    del pending[fut]
                    done += 1
                    if err:
                        print(f"  [ERROR] sample {idx2}: {err}")
                        n_errors += 1
                        label, ident, at = -1, "", ""
                    manifest_rows.append({"idx": idx2, "identifier": ident, "anchor_time": at, "label": label})
                    if done % progress_every == 0 or done == total:
                        elapsed = time.time() - t0
                        rate    = done / max(elapsed, 1e-6)
                        eta     = (total - done) / max(rate, 1e-6)
                        print(f"  [{done:>5}/{total}]  elapsed={elapsed/60:.1f}m  "
                              f"rate={rate:.1f}/s  ETA={eta/60:.1f}m  errors={n_errors}")
                    _submit_next(pool)

    if skip_existing and n_skip > 0:
        existing_indices = {r["idx"] for r in manifest_rows}
        for idx in range(n):
            if idx in existing_indices:
                continue
            out_path = samples_dir / f"{idx:05d}.pt"
            if out_path.exists():
                try:
                    s = torch.load(out_path, map_location="cpu", weights_only=False)
                    manifest_rows.append({
                        "idx": idx, "identifier": s.get("identifier", ""),
                        "anchor_time": s.get("anchor_time", ""), "label": s.get("label", -1),
                    })
                except Exception as e:
                    print(f"  [WARN] could not read existing {out_path}: {e}")

    manifest_rows.sort(key=lambda r: r["idx"])
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = cache_dir / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\n  Manifest written: {manifest_path}  ({len(manifest_df)} rows)")

    info = {
        "n_samples": n, "n_cached": len(manifest_rows), "n_errors": n_errors,
        "feature_dims": dims, "key_remap": KEY_REMAP,
        "spatial_coords": SPATIAL_COORDS,
        "leiden_cols": LEIDEN_COLS,
        "clinical_vocab_path": str(vocab_path),
        "clinical_n_tokens": clinical_tok._n_tokens,
        "clinical_n_features": len(clinical_tok.feature_cols),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "cache_dir": str(cache_dir),
    }
    info_path = cache_dir / "info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Info  written: {info_path}")

    elapsed_total = time.time() - t0
    print(f"\n  Done.  Total time: {elapsed_total/60:.1f} min  Errors: {n_errors}")
    if n_errors > 0:
        print(f"\n  WARNING: {n_errors} samples failed — rerun with --skip_existing to fill gaps.")
    return manifest_df


# ===========================================================================
# SECTION 10 — CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute MultimodalTimeseriesDataset → per-sample .pt files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cache_dir", default=DEFAULT_CACHE)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress_every", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Preview mode: save only the first N samples then stop (0 = all)")

    parser.add_argument(
        "--diagnose", action="store_true",
        help=(
            "Fast diagnose mode. Filters every h5ad/CSV to the requested patients, "
            "builds a tiny dataset, then prints per-file filter stats, modality "
            "availability, orphan scan, and per-patient alignment detail. "
            "No files are written."
        ),
    )
    parser.add_argument(
        "--diagnose_patients", nargs="+", default=None, metavar="PID",
        help=(
            "Patient IDs to inspect, e.g. LT001 LT002 LT003. "
            "The values are the same across all files — only the column name differs. "
            "If omitted, the first --diagnose_n patients from the BAL h5ad are used."
        ),
    )
    parser.add_argument(
        "--diagnose_n", type=int, default=3,
        help="Number of patients to auto-pick when --diagnose_patients is not given.",
    )
    parser.add_argument(
        "--no_raw_times", action="store_true",
        help="Skip printing raw time_dict entries per modality (less verbose output).",
    )

    args = parser.parse_args()

    if args.diagnose:
        diagnose(
            patient_ids = args.diagnose_patients,
            n_patients  = args.diagnose_n,
            verbose_raw = not args.no_raw_times,
        )
        sys.exit(0)

    manifest = precompute(
        cache_dir      = Path(args.cache_dir),
        skip_existing  = args.skip_existing,
        workers        = args.workers,
        progress_every = args.progress_every,
        max_samples    = args.max_samples,
    )
    n_pos = (manifest["label"] == 1).sum()
    n_neg = (manifest["label"] == 0).sum()
    n_unk = (manifest["label"] == -1).sum()
    print(f"\n  Label distribution: pos={n_pos}  neg={n_neg}  unknown={n_unk}")
    print(f"  Pos fraction: {n_pos / max(n_pos + n_neg, 1):.3f}")
    print(f"\n  To train, run:")
    print(f"    python run_grouped_mil_real.py --cache_dir {args.cache_dir} --fold 0")