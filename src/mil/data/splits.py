"""Train/val/test split builders for ACR, survival (CLAD/Death), and multitask."""

import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .registry import MODALITIES, _pres_col
from .labels import acr_label, compute_tte_first_acr_episode, compute_tte_next_acr


def build_splits(samples_dir, splits_csv, fold, split=None):
    """Classification-only splits: keep rows with a valid ACR grade (A0/A1/A2)."""
    import pandas as pd

    df       = pd.read_csv(splits_csv)
    fold_col = f"split{split}_fold{fold}" if split is not None else f"fold_{fold}"
    assert fold_col in df.columns, f"Column {fold_col!r} not in {splits_csv}"

    splits_dict: Dict[str, list] = {"train": [], "val": [], "test": []}
    n_dropped = 0
    for _, row in df.iterrows():
        grade = row.get("acr_grade")
        if grade is None or (isinstance(grade, float) and np.isnan(grade)):
            n_dropped += 1; continue
        grade_str = str(grade).strip()
        if not grade_str or grade_str.lower() in ("nan", "none", "n/a", "na", "", "?"):
            n_dropped += 1; continue
        if not re.search(r"A\d", grade_str, re.IGNORECASE):
            n_dropped += 1; continue
        label = 1 if re.search(r"A[12]", grade_str, re.IGNORECASE) else 0
        sp = str(row[fold_col])
        if sp not in splits_dict:
            n_dropped += 1; continue
        stem = Path(str(row["file"])).stem
        rec  = {"stem": stem, "label": label,
                "patient_id": str(row.get("patient_id", stem))}
        for mod in MODALITIES:
            rec[_pres_col(mod)] = bool(row.get(_pres_col(mod), False))
        for ep, sc, dc in [("clad","clad_status","clad_days"),
                           ("death","death_status","death_days")]:
            try:
                s = float(row.get(sc, float("nan")))
                d = float(row.get(dc, float("nan")))
                if not math.isnan(d) and not math.isnan(s):
                    rec[f"{ep}_time"]  = max(d, 0.0)
                    rec[f"{ep}_event"] = float(s)
                else:
                    rec[f"{ep}_time"]  = float("nan")
                    rec[f"{ep}_event"] = float("nan")
            except (TypeError, ValueError):
                rec[f"{ep}_time"]  = float("nan")
                rec[f"{ep}_event"] = float("nan")
        splits_dict[sp].append(rec)

    tag = f"split{split}_fold{fold}" if split is not None else f"fold_{fold}"
    print(f"  [{tag}] dropped={n_dropped}  "
          f"train={len(splits_dict['train'])}  val={len(splits_dict['val'])}  "
          f"test={len(splits_dict['test'])}")
    return splits_dict["train"], splits_dict["val"], splits_dict["test"]


def build_splits_survival(samples_dir, splits_csv, fold, split=None, endpoint="clad"):
    """
    Survival-endpoint splits for CLAD, Death, or ACR-TTE.

    CLAD: pre-CLAD samples only; censored → death_days or study_end proxy.
    Death: all samples; censored → study_end proxy.
    ACR:  pre-episode A0-confirmed samples + first ACR+ sample.
    """
    import pandas as pd

    df       = pd.read_csv(splits_csv, parse_dates=["anchor_dt"])
    fold_col = f"split{split}_fold{fold}" if split is not None else f"fold_{fold}"
    assert fold_col in df.columns

    time_col  = f"{endpoint}_days"
    event_col = f"{endpoint}_status"
    study_end = df["anchor_dt"].max()

    splits_dict: Dict[str, list] = {"train": [], "val": [], "test": []}
    n_dropped = 0
    for _, row in df.iterrows():
        sp = str(row.get(fold_col, ""))
        if sp not in splits_dict:
            n_dropped += 1; continue
        try:
            t = float(row.get(time_col,  float("nan")))
            e = float(row.get(event_col, float("nan")))
        except (TypeError, ValueError):
            n_dropped += 1; continue

        if endpoint == "acr":
            if math.isnan(e):
                n_dropped += 1; continue
            label_val = float(row.get("label", float("nan"))) if not math.isnan(
                float(row.get("label", float("nan")) if row.get("label") is not None else float("nan"))
            ) else float("nan")
            if e == 0.0:
                if math.isnan(label_val) or label_val != 0.0:
                    n_dropped += 1; continue
                try:
                    t = float((study_end - pd.Timestamp(row["anchor_dt"])).days)
                except Exception:
                    t = float("nan")
                if math.isnan(t) or t <= 0:
                    n_dropped += 1; continue
            else:
                if math.isnan(t) or t < 0:
                    n_dropped += 1; continue
                if t > 1 and (math.isnan(label_val) or label_val != 0.0):
                    n_dropped += 1; continue

        elif endpoint == "clad":
            if math.isnan(e):
                n_dropped += 1; continue
            if e == 0.0:
                try:
                    proxy_t = float(row.get("death_days", float("nan")))
                except (TypeError, ValueError):
                    proxy_t = float("nan")
                if math.isnan(proxy_t) or proxy_t <= 0:
                    try:
                        proxy_t = float((study_end - pd.Timestamp(row["anchor_dt"])).days)
                    except Exception:
                        proxy_t = float("nan")
                if math.isnan(proxy_t) or proxy_t <= 0:
                    n_dropped += 1; continue
                t = proxy_t
            else:
                if math.isnan(t) or t <= 0:
                    n_dropped += 1; continue
        else:
            if math.isnan(e):
                n_dropped += 1; continue
            if e == 0.0 and (math.isnan(t) or t <= 0):
                try:
                    t = float((study_end - pd.Timestamp(row["anchor_dt"])).days)
                except Exception:
                    t = float("nan")
            if math.isnan(t) or t <= 0:
                n_dropped += 1; continue

        stem = Path(str(row["file"])).stem
        rec  = {"stem": stem, "label": int(e),
                "patient_id": str(row.get("patient_id", stem))}
        for mod in MODALITIES:
            rec[_pres_col(mod)] = bool(row.get(_pres_col(mod), False))
        for ep, sc, dc in [("clad","clad_status","clad_days"),
                           ("death","death_status","death_days"),
                           ("acr","acr_status","acr_days")]:
            try:
                s = float(row.get(sc, float("nan")))
                d = float(row.get(dc, float("nan")))
                if not math.isnan(d) and not math.isnan(s) and d > 0:
                    rec[f"{ep}_time"]  = d
                    rec[f"{ep}_event"] = float(s)
                else:
                    rec[f"{ep}_time"]  = float("nan")
                    rec[f"{ep}_event"] = float("nan")
            except (TypeError, ValueError):
                rec[f"{ep}_time"]  = float("nan")
                rec[f"{ep}_event"] = float("nan")
        if e == 0.0:
            rec[f"{endpoint}_time"]  = t
            rec[f"{endpoint}_event"] = 0.0
        splits_dict[sp].append(rec)

    tag = f"split{split}_fold{fold}" if split is not None else f"fold_{fold}"
    print(f"  [{tag}] survival({endpoint}) dropped={n_dropped}  "
          f"train={len(splits_dict['train'])}  val={len(splits_dict['val'])}  "
          f"test={len(splits_dict['test'])}")
    return splits_dict["train"], splits_dict["val"], splits_dict["test"]


def build_splits_multitask(samples_dir, splits_csv, fold, split=None):
    """
    Multitask splits: all samples, ACR gap-time TTE + CLAD/Death TTE per record.
    Label derived from acr_grade; missing label → None (excluded from hinge loss).
    """
    import pandas as pd

    df       = pd.read_csv(splits_csv)
    df["anchor_dt"] = pd.to_datetime(df["anchor_dt"])
    fold_col = f"split{split}_fold{fold}" if split is not None else f"fold_{fold}"
    assert fold_col in df.columns

    tte_map = compute_tte_first_acr_episode(df)

    splits_dict: Dict[str, list] = {"train": [], "val": [], "test": []}
    n_dropped = 0
    for _, row in df.iterrows():
        sp = str(row.get(fold_col, ""))
        if sp not in splits_dict:
            n_dropped += 1; continue

        stem   = str(Path(str(row["file"])).stem)
        tte, ev = tte_map.get(stem, (float("nan"), 0))

        rec = {
            "stem":           stem,
            "patient_id":     str(row.get("patient_id", stem)),
            "anchor_dt":      row["anchor_dt"],   # pd.Timestamp — needed for trajectory plots
            "label":          acr_label(row.get("acr_grade")),
            "tte_next_acr":   tte,
            "event_next_acr": ev,
            "acr_days":       float(row["acr_days"]) if pd.notna(row.get("acr_days")) else float("nan"),
            "acr_status":     float(row["acr_status"]) if pd.notna(row.get("acr_status")) else float("nan"),
        }
        for mod in MODALITIES:
            rec[_pres_col(mod)] = bool(row.get(_pres_col(mod), False))
        rec["disease_times_clr"] = [tte] if ev == 1 and not math.isnan(tte) else []

        for ep, sc, dc in [("clad","clad_status","clad_days"),
                           ("death","death_status","death_days")]:
            try:
                s = float(row.get(sc, float("nan")))
                d = float(row.get(dc, float("nan")))
                rec[f"{ep}_time"]  = d if not math.isnan(d) and d > 0 else float("nan")
                rec[f"{ep}_event"] = float(s) if not math.isnan(s) else float("nan")
            except (TypeError, ValueError):
                rec[f"{ep}_time"]  = float("nan")
                rec[f"{ep}_event"] = float("nan")
        splits_dict[sp].append(rec)

    tag = f"split{split}_fold{fold}" if split is not None else f"fold_{fold}"
    print(f"  [{tag}] n_dropped={n_dropped}")
    for sn, recs in splits_dict.items():
        n_cls  = sum(1 for r in recs if r["label"] is not None)
        n_ev   = sum(1 for r in recs if r["event_next_acr"] == 1)
        n_cens = sum(1 for r in recs if r["event_next_acr"] == 0)
        print(f"  [{tag}] {sn:5s}  total={len(recs)}  cls={n_cls}  "
              f"surv_event={n_ev}  surv_censored={n_cens}")
    return splits_dict


def build_splits_longitudinal(samples_dir, splits_csv, fold, split=None):
    """
    Patient-level splits for LongitudinalMIL.

    Groups all biopsies by patient_id, sorted by anchor_dt.
    Each record = one PATIENT with T ordered biopsies.

    Returns splits_dict {'train'/'val'/'test': [patient_dict, ...]}.
    Each patient_dict:
      patient_id, stems (T), days (T floats from first biopsy), records (T label dicts)
    """
    import math
    import pandas as pd
    from collections import defaultdict

    df = pd.read_csv(splits_csv)
    df["anchor_dt"] = pd.to_datetime(df["anchor_dt"])
    fold_col = f"split{split}_fold{fold}" if split is not None else f"fold_{fold}"
    assert fold_col in df.columns

    tte_map = compute_tte_first_acr_episode(df)

    patients: dict = defaultdict(list)
    split_of_patient: dict = {}

    for _, row in df.iterrows():
        sp = str(row.get(fold_col, ""))
        if sp not in ("train", "val", "test"):
            continue
        pid    = str(row.get("patient_id", ""))
        stem   = str(Path(str(row["file"])).stem)
        anchor = row["anchor_dt"]
        tte, ev = tte_map.get(stem, (float("nan"), 0))

        biopsy_rec = {
            "stem":           stem,
            "anchor_dt":      anchor,
            "label":          acr_label(row.get("acr_grade")),
            "tte_next_acr":   tte,
            "event_next_acr": ev,
            "acr_days":       float(row["acr_days"]) if pd.notna(row.get("acr_days")) else float("nan"),
            "acr_status":     float(row["acr_status"]) if pd.notna(row.get("acr_status")) else float("nan"),
        }
        for mod in MODALITIES:
            biopsy_rec[_pres_col(mod)] = bool(row.get(_pres_col(mod), False))
        for ep, sc, dc in [("clad", "clad_status", "clad_days"),
                           ("death", "death_status", "death_days")]:
            try:
                s = float(row.get(sc, float("nan")))
                d = float(row.get(dc, float("nan")))
                biopsy_rec[f"{ep}_time"]  = d if not math.isnan(d) and d > 0 else float("nan")
                biopsy_rec[f"{ep}_event"] = float(s) if not math.isnan(s) else float("nan")
            except (TypeError, ValueError):
                biopsy_rec[f"{ep}_time"]  = float("nan")
                biopsy_rec[f"{ep}_event"] = float("nan")

        patients[pid].append(biopsy_rec)
        if pid not in split_of_patient:
            split_of_patient[pid] = sp

    splits_dict: Dict[str, list] = {"train": [], "val": [], "test": []}

    for pid, biopsy_list in patients.items():
        sp = split_of_patient.get(pid)
        if sp not in splits_dict:
            continue
        biopsy_list.sort(key=lambda r: r["anchor_dt"])
        t0   = biopsy_list[0]["anchor_dt"]
        days = [float((b["anchor_dt"] - t0).days) for b in biopsy_list]
        stems   = [b["stem"] for b in biopsy_list]
        records = [{k: v for k, v in b.items() if k != "anchor_dt"}
                   for b in biopsy_list]
        splits_dict[sp].append({
            "patient_id": pid,
            "stems":      stems,
            "days":       days,
            "records":    records,
        })

    tag = f"split{split}_fold{fold}" if split is not None else f"fold_{fold}"
    for sn, recs in splits_dict.items():
        n_biopsies = sum(len(p["stems"]) for p in recs)
        print(f"  [{tag}] longitudinal {sn:5s}: patients={len(recs)} biopsies={n_biopsies}")
    return splits_dict


def update_presence_from_cache(records, bag_cache):
    """Refresh has_* flags in records from a loaded bag cache."""
    for rec in records:
        entry = bag_cache.get(rec["stem"], {})
        for mod in MODALITIES:
            rec[_pres_col(mod)] = entry.get(mod) is not None
    return records
