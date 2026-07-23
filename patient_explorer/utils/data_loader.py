"""Central data loader — reads pre-processed CSVs / NPZ files."""

from __future__ import annotations
import os
from pathlib import Path
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Paths (resolved relative to this file or via env override)
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent.parent  # patient_explorer/

DATA_DIR = Path(os.environ.get(
    "EXPLORER_DATA",
    str(_HERE / "data")
))

SPLITS_CSV      = DATA_DIR / "splits.csv"
PREDS_CSV       = DATA_DIR / "predictions_all.csv"
EPISODES_CSV    = DATA_DIR / "episodes.csv"
UMAP_CSV        = DATA_DIR / "umap_embeddings.csv"
HE_FREQ_CSV     = DATA_DIR / "he_cluster_freq.csv"
BAL_FREQ_CSV    = DATA_DIR / "bal_cluster_freq.csv"
CT_FREQ_CSV     = DATA_DIR / "ct_cluster_freq.csv"
CLIN_CSV        = DATA_DIR / "clinical_features.csv"
FEAT_NAMES      = DATA_DIR / "clinical_feature_names.csv"
# New data files
BENCHMARK_CSV   = DATA_DIR / "benchmark_summary.csv"
DIFF_CSV        = DATA_DIR / "differential_abundance.csv"
PCA_CSV         = DATA_DIR / "pca_scores.csv"
COHORT_JSON     = DATA_DIR / "cohort_summary.json"
XMODAL_CSV      = DATA_DIR / "cross_modal_corr.csv"
SAMPLE_TABLE    = DATA_DIR / "sample_table.csv"
SETMILMT_CSV        = DATA_DIR / "setmilmt_preds.csv"
BENCHMARK_RESULTS_CSV = DATA_DIR / "benchmark_results.csv"
SUMMARY_PNG_DIR     = Path("/ictstr01/home/aih/dinesh.haridoss/chicago_mil/interpretability/set_mil_mt_interp/all_splits_merged/patient_summaries")
PAPER_INTERP_JSON   = Path("/ictstr01/home/aih/dinesh.haridoss/chicago_mil/interpretability/set_mil_mt_interp/all_splits_merged/paper_interp_data.json")
LONGI_INTERP_DIR    = Path("/ictstr01/home/aih/dinesh.haridoss/chicago_mil/interpretability/longitudinal_mk_interp")
PANEL_FIG_DIR       = Path("/ictstr01/home/aih/dinesh.haridoss/chicago_mil/interpretability/set_mil_mt_interp/all_splits_merged")


# ---------------------------------------------------------------------------
# Loaders (cached so each CSV is read only once per session)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_splits() -> pd.DataFrame:
    df = pd.read_csv(SPLITS_CSV, parse_dates=["anchor_dt"])
    df["days_since_tx"] = (df["anchor_dt"] - df.groupby("patient_id")["anchor_dt"].transform("min")).dt.days
    return df


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    if not PREDS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(PREDS_CSV, parse_dates=["anchor_dt"])
    return df


@st.cache_data(show_spinner=False)
def load_episodes() -> pd.DataFrame:
    if not EPISODES_CSV.exists():
        return pd.DataFrame()
    import ast
    df = pd.read_csv(EPISODES_CSV)
    for col in ["episode_durations", "episode_sizes", "inter_ep_gaps"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
    return df


@st.cache_data(show_spinner=False)
def load_umap() -> pd.DataFrame:
    if not UMAP_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(UMAP_CSV)
    if "anchor_dt" in df.columns:
        df["anchor_dt"] = pd.to_datetime(df["anchor_dt"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_cluster_freq(mod: str) -> pd.DataFrame:
    path = {"HE": HE_FREQ_CSV, "BAL": BAL_FREQ_CSV, "CT": CT_FREQ_CSV}.get(mod)
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["anchor_dt"] if "anchor_dt" in pd.read_csv(path, nrows=0).columns else [])


@st.cache_data(show_spinner=False)
def load_clinical() -> pd.DataFrame:
    if not CLIN_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(CLIN_CSV, parse_dates=["anchor_dt"] if "anchor_dt" in pd.read_csv(CLIN_CSV, nrows=0).columns else [])
    return df


@st.cache_data(show_spinner=False)
def load_feature_names() -> dict[int, str]:
    if not FEAT_NAMES.exists():
        return {}
    df = pd.read_csv(FEAT_NAMES)
    if "idx" in df.columns and "name" in df.columns:
        return dict(zip(df["idx"], df["name"]))
    return {}


# ---------------------------------------------------------------------------
# Derived helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def patient_list() -> list[str]:
    df = load_splits()
    return sorted(df["patient_id"].dropna().unique().tolist())


def patient_splits(pid: str) -> pd.DataFrame:
    return load_splits().query("patient_id == @pid").sort_values("anchor_dt")


def patient_predictions(pid: str) -> pd.DataFrame:
    df = load_predictions()
    if df.empty:
        return df
    return df.query("patient_id == @pid").sort_values("anchor_dt")


@st.cache_data(show_spinner=False)
def load_setmilmt() -> pd.DataFrame:
    if not SETMILMT_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(SETMILMT_CSV, parse_dates=["anchor_dt"])
    return df


def patient_setmilmt(pid: str) -> pd.DataFrame:
    df = load_setmilmt()
    if df.empty:
        return df
    return df.query("patient_id == @pid").sort_values("anchor_dt")


def setmilmt_summary_png(stem: str) -> Optional[Path]:
    p = SUMMARY_PNG_DIR / f"L0_summary_{stem}.png"
    return p if p.exists() else None


def patient_episode(pid: str) -> Optional[pd.Series]:
    df = load_episodes()
    if df.empty:
        return None
    rows = df.query("patient_id == @pid")
    return rows.iloc[0] if len(rows) else None


def patient_umap(pid: str) -> pd.DataFrame:
    df = load_umap()
    if df.empty:
        return df
    return df.query("patient_id == @pid")


def patient_cluster_freq(pid: str, mod: str) -> pd.DataFrame:
    df = load_cluster_freq(mod)
    if df.empty:
        return df
    if "patient_id" in df.columns:
        return df.query("patient_id == @pid")
    return df


def cohort_cluster_freq_mean(mod: str, group_col: str = "acr_encoded") -> pd.DataFrame:
    """Return mean cluster fractions per group for the whole cohort."""
    df = load_cluster_freq(mod)
    if df.empty or group_col not in df.columns:
        return pd.DataFrame()
    cluster_cols = [c for c in df.columns if c.startswith("cluster_")]
    return df.groupby(group_col)[cluster_cols].mean().reset_index()


@st.cache_data(show_spinner=False)
def load_benchmark() -> pd.DataFrame:
    if not BENCHMARK_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(BENCHMARK_CSV)


@st.cache_data(show_spinner=False)
def load_differential() -> pd.DataFrame:
    if not DIFF_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(DIFF_CSV)


@st.cache_data(show_spinner=False)
def load_pca_scores() -> pd.DataFrame:
    if not PCA_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(PCA_CSV)


@st.cache_data(show_spinner=False)
def load_cohort_summary() -> dict:
    if not COHORT_JSON.exists():
        return {}
    import json
    return json.loads(COHORT_JSON.read_text())


@st.cache_data(show_spinner=False)
def load_cross_modal_corr() -> pd.DataFrame:
    if not XMODAL_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(XMODAL_CSV, index_col=0)


@st.cache_data(show_spinner=False)
def load_sample_table() -> pd.DataFrame:
    if not SAMPLE_TABLE.exists():
        return pd.DataFrame()
    return pd.read_csv(SAMPLE_TABLE)


@st.cache_data(show_spinner=False)
def load_benchmark_results() -> pd.DataFrame:
    """Load benchmark_results.csv (new format: phase,model,task,metric,mean,std,s0-s4).
    Falls back to benchmark_summary.csv if new file not present."""
    if BENCHMARK_RESULTS_CSV.exists():
        return pd.read_csv(BENCHMARK_RESULTS_CSV)
    if BENCHMARK_CSV.exists():
        return pd.read_csv(BENCHMARK_CSV)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_paper_interp() -> dict:
    if not PAPER_INTERP_JSON.exists():
        return {}
    import json
    return json.loads(PAPER_INTERP_JSON.read_text())


def longitudinal_summary_png(pid: str) -> Optional[Path]:
    """Find longitudinal model patient summary PNG across all splits."""
    for split_dir in sorted(LONGI_INTERP_DIR.glob("split*_fold0")):
        p = split_dir / f"L0_summary_pid{pid}.png"
        if p.exists():
            return p
    return None


def available_data(pid: str) -> dict[str, bool]:
    """Which data sections are available for this patient."""
    sp = patient_splits(pid)
    pr = patient_predictions(pid)
    um = patient_umap(pid)
    cl = load_clinical()
    return {
        "timeline":  len(sp) > 0,
        "predictions": len(pr) > 0,
        "umap":      len(um) > 0,
        "clinical":  len(cl) > 0 and "patient_id" in cl.columns and pid in cl["patient_id"].values,
        "he_freq":   not load_cluster_freq("HE").empty,
        "bal_freq":  not load_cluster_freq("BAL").empty,
        "ct_freq":   not load_cluster_freq("CT").empty,
    }
