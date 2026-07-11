"""Post-hoc enrichment of cached DataFrames.
Fixes combo/n_mods and event labels without re-running inference."""
from pathlib import Path
from typing import Dict

import pandas as pd

from .config import ENDPOINT


def _combo_from_row(row) -> str:
    parts = []
    for mod, col in [("HE","has_HE"), ("BAL","has_BAL"),
                     ("CT","has_CT"), ("Clin","has_Clinical")]:
        v = row.get(col, False)
        if v is True or str(v).lower() in ("true", "1", "1.0"):
            parts.append(mod)
    return "+".join(parts) if parts else "Unknown"


def _n_mods(combo: str) -> int:
    return len(combo.split("+")) if combo and combo != "Unknown" else 0


def enrich_combo(variant_data: Dict, splits_csv: Path) -> None:
    """Recompute combo/n_mods from has_* columns — fixes stale cached values."""
    df_csv = pd.read_csv(str(splits_csv))
    df_csv["_stem"]  = df_csv["file"].apply(lambda f: Path(str(f)).stem)
    df_csv["_combo"] = df_csv.apply(_combo_from_row, axis=1)
    df_csv["_nmods"] = df_csv["_combo"].apply(_n_mods)
    s2combo = dict(zip(df_csv["_stem"], df_csv["_combo"]))
    s2nmods = dict(zip(df_csv["_stem"], df_csv["_nmods"]))
    for vd in variant_data.values():
        df = vd["df"]
        if "stem" in df.columns:
            df["combo"]  = df["stem"].map(s2combo).fillna("Unknown")
            df["n_mods"] = df["stem"].map(s2nmods).fillna(0).astype(int)


def enrich_events(variant_data: Dict, splits_csv: Path, endpoint: str) -> None:
    """Fix event labels (0=censored, 1=event) from CSV — corrects NaN-for-censored bug."""
    cfg    = ENDPOINT[endpoint]
    ev_key = cfg["ev_key"]
    ev_col = cfg["event_col"]
    df_csv = pd.read_csv(str(splits_csv))
    df_csv["_stem"] = df_csv["file"].apply(lambda f: Path(str(f)).stem)

    def _parse(v):
        try:
            v = float(v)
            return 0.0 if v == 0 else (1.0 if v == 1 else float("nan"))
        except (ValueError, TypeError):
            return float("nan")

    df_csv["_ev"] = df_csv[ev_col].apply(_parse)
    s2ev = dict(zip(df_csv["_stem"], df_csv["_ev"]))
    for vd in variant_data.values():
        df = vd["df"]
        if "stem" in df.columns:
            df[ev_key] = df["stem"].map(s2ev)


def enrich_all(variant_data: Dict, splits_csv: Path, endpoint: str) -> None:
    enrich_combo(variant_data, splits_csv)
    enrich_events(variant_data, splits_csv, endpoint)
