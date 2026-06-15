"""ACR label helpers and TTE computation."""

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


def compute_tte_first_acr_episode(df) -> dict:
    """
    Landmark TTE to end of first ACR episode.

    Episode definition (A0-break rule):
      - A0 biopsy ends the current episode
      - nan/unknown grade biopsies are ignored
      - First episode = first unbroken run of A1*/A2* biopsies before the first A0

    Returns dict: stem → (tte_days: float, event: int)
      Pre-episode biopsies          : (TTE_to_episode_end, 1)  — positive TTE
      Biopsies WITHIN first episode : (0.0, 1)  — event NOW (TTE=0)
      Post-episode biopsies         : (nan, nan) — excluded from Cox
      No ACR episode (censored)     : (days_to_last_biopsy, 0)

    Including episode biopsies as TTE=0 events means the graded ACR+ biopsies
    (which always have HE imaging) contribute to Cox training, solving the
    missing-imaging problem for pre-episode routine visits.
    """
    import pandas as _pd

    if not hasattr(df["anchor_dt"], "dt"):
        df = df.copy()
        df["anchor_dt"] = _pd.to_datetime(df["anchor_dt"])

    _nan = float("nan")
    result: dict = {}

    def _is_pos(g) -> bool:
        return isinstance(g, str) and (g.startswith("A1") or g.startswith("A2"))

    for pid, pdata in df.groupby("patient_id"):
        pdata  = pdata.sort_values("anchor_dt").reset_index(drop=True)
        stems  = [str(Path(str(r["file"])).stem) for _, r in pdata.iterrows()]
        dates  = list(pdata["anchor_dt"])
        grades = list(pdata["acr_grade"])

        # Find index of first ACR+ biopsy
        first_idx = next((i for i, g in enumerate(grades) if _is_pos(g)), None)

        if first_idx is None:
            # No ACR episode — censored at last biopsy date
            last_d = dates[-1]
            for i, stem in enumerate(stems):
                tte = max(int((last_d - dates[i]).days), 0)
                result[stem] = (float(tte), 0)
            continue

        # Find end of first continuous ACR+ run — ends at first A0 biopsy
        episode_end = first_idx
        for j in range(first_idx + 1, len(grades)):
            if _is_pos(grades[j]):
                episode_end = j
            elif isinstance(grades[j], str) and grades[j].startswith("A0"):
                break   # A0 ends the episode; nan grades are ignored

        episode_end_date = dates[episode_end]

        for i, stem in enumerate(stems):
            if i < first_idx:
                # Pre-episode: positive TTE to episode end, event=1
                tte = max(int((episode_end_date - dates[i]).days), 0)
                result[stem] = (float(tte), 1)
            elif i <= episode_end:
                # Within first episode: TTE=0, event=1 (the event is NOW)
                result[stem] = (0.0, 1)
            else:
                # Post-episode: excluded from Cox
                result[stem] = (_nan, _nan)

    return result


def compute_tte_next_acr(df) -> dict:
    """
    Legacy gap-time approach (kept for backward compat). Use
    compute_tte_first_acr_episode for new experiments.
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
