"""ACR label helpers and gap-time TTE computation."""

import math
from pathlib import Path
from typing import Optional

import numpy as np


def acr_label(grade_str) -> Optional[int]:
    """A0* → 0,  A1*/A2* → 1,  anything else → None."""
    if grade_str is None:
        return None
    if isinstance(grade_str, float) and math.isnan(grade_str):
        return None
    g = str(grade_str).strip()
    if not g or g.lower() in ("nan", "none", "n/a", "na", "", "?"):
        return None
    if g.startswith("A0"):
        return 0
    if g.startswith("A1") or g.startswith("A2"):
        return 1
    return None


def compute_tte_next_acr(df) -> dict:
    """
    Gap-time approach for recurrent ACR events.

    For each biopsy (row) returns:
      tte_next_acr  : days from anchor_dt to next A1/A2 biopsy (0 if this IS the event)
      event_next_acr: 1 if an event exists, 0 if censored at last biopsy

    Returns dict: stem → (tte: float, event: int)
    """
    import pandas as _pd

    if not hasattr(df["anchor_dt"], "dt"):
        df = df.copy()
        df["anchor_dt"] = _pd.to_datetime(df["anchor_dt"])

    acr_mask = df["acr_grade"].apply(
        lambda g: isinstance(g, str) and (g.startswith("A1") or g.startswith("A2"))
    )
    acr_dates: dict = {}
    for _, row in df[acr_mask].iterrows():
        acr_dates.setdefault(row["patient_id"], []).append(row["anchor_dt"])

    last_date: dict = df.groupby("patient_id")["anchor_dt"].max().to_dict()

    result: dict = {}
    for _, row in df.iterrows():
        stem = str(Path(str(row["file"])).stem)
        pid  = row["patient_id"]
        t    = row["anchor_dt"]
        is_acr_pos = (
            isinstance(row.get("acr_grade"), str)
            and (row["acr_grade"].startswith("A1") or row["acr_grade"].startswith("A2"))
        )
        future = sorted([d for d in acr_dates.get(pid, []) if d > t])
        if is_acr_pos:
            tte, ev = 0, 1
        elif future:
            tte, ev = (future[0] - t).days, 1
        else:
            last    = last_date.get(pid, t)
            tte, ev = max(int((last - t).days), 0), 0
        result[stem] = (float(tte), int(ev))
    return result
