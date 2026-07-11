"""
All task configurations, color schemes, and display constants.
One entry per task — both the results directory and the endpoint columns.
"""
from typing import Dict, Tuple

# ── Task registry ─────────────────────────────────────────────────────────────
# key → (results_dir_suffix, primary_metric, display_label, color, task_type)
#   task_type: "cls" | "tte"
TASKS: Dict[str, Tuple[str, str, str, str, str]] = {
    "acr_cls":     ("results_mm_abmil_v7_cls_p1v3",  "bacc",    "ACR Cls (single)",    "#1565C0", "cls"),
    "acr_tte":     ("results_mm_abmil_v7_surv_p1v3", "c_index", "ACR TTE (single)",    "#0277BD", "tte"),
    "acr_alt_cls": ("results_mm_abmil_v7_alt_p1v3",  "bacc",    "ACR Cls (multitask)", "#1B5E20", "cls"),
    "acr_alt_tte": ("results_mm_abmil_v7_alt_p1v3",  "c_index", "ACR TTE (multitask)", "#2E7D32", "tte"),
    "clad":        ("results_mm_abmil_v7_clad",       "c_index", "CLAD",                "#FB8500", "tte"),
    "death":       ("results_mm_abmil_v7_death",       "c_index", "Death",               "#C62828", "tte"),
    # v8 multimodal ABMIL — phase2 results, split1_fold{f}/
    "acr_v8":      ("results/mm_abmil_v8/phase2",     "bacc",    "ACR v8",              "#1565C0", "cls"),
    "acr_tte_v8":  ("results/mm_abmil_v8/phase2",     "c_index", "ACR TTE v8",          "#0277BD", "tte"),
    "clad_v8":     ("results/mm_abmil_v8/phase2",     "c_index", "CLAD v8",             "#FB8500", "tte"),
    "death_v8":    ("results/mm_abmil_v8/phase2",     "c_index", "Death v8",            "#C62828", "tte"),
}

# ── Endpoint columns (event label + TTE from splits CSV) ──────────────────────
# Used for UMAP coloring and event enrichment. Multiple tasks can share one endpoint.
ENDPOINT: Dict[str, Dict] = {
    "acr":   {"time_col": "acr_days",   "event_col": "acr_status",
              "tte_key":  "acr_time",   "ev_key":    "acr_event",
              "label": "ACR",   "color": "#1565C0"},
    "clad":  {"time_col": "clad_days",  "event_col": "clad_status",
              "tte_key":  "clad_time",  "ev_key":    "clad_event",
              "label": "CLAD",  "color": "#FB8500"},
    "death": {"time_col": "death_days", "event_col": "death_status",
              "tte_key":  "death_time", "ev_key":    "death_event",
              "label": "Death", "color": "#C62828"},
}

# Which endpoint does each task use for UMAP / event labels?
TASK_ENDPOINT = {
    "acr_cls":     "acr",
    "acr_tte":     "acr",
    "acr_alt_cls": "acr",
    "acr_alt_tte": "acr",
    "clad":        "clad",
    "death":       "death",
    "acr_v8":      "acr",
    "acr_tte_v8":  "acr",
    "clad_v8":     "clad",
    "death_v8":    "death",
}

# ── Variant ordering & display ────────────────────────────────────────────────
VARIANT_TAGS = [
    # v8 variants (new)
    "slot_cls", "slot_clad_surv", "slot_death_surv", "slot_acr_surv",
    "slot_cls_v2", "slot_clad_surv_v2", "slot_death_surv_v2",
    "slot_cls_geomae", "slot_clad_surv_geomae", "slot_death_surv_geomae",
    "early_cls", "middle_cls", "late_cls",
    # v7 legacy
    "early", "late", "middle", "self_attn",
]
VARIANT_DISPLAY = {
    "slot_cls":             "Slot (v1)",
    "slot_cls_v2":          "Slot v2 (d=0.3)",
    "slot_cls_geomae":      "Slot+GeoMAE",
    "slot_clad_surv":       "Slot CLAD (v1)",
    "slot_clad_surv_v2":    "Slot CLAD v2",
    "slot_death_surv":      "Slot Death (v1)",
    "slot_death_surv_v2":   "Slot Death v2",
    "slot_acr_surv":        "Slot ACR-TTE",
    "early_cls":            "Early",
    "middle_cls":           "Middle",
    "late_cls":             "Late",
    "early":                "Early",
    "late":                 "Late",
    "middle":               "Middle",
    "self_attn":            "Self-Attn",
}

# ── Color schemes ─────────────────────────────────────────────────────────────
# Modality combos
COMBO_COLORS = {
    "HE":                   "#E53935",
    "BAL":                  "#1E88E5",
    "CT":                   "#43A047",
    "Clin":                 "#8E24AA",
    "HE+BAL":               "#FB8C00",
    "HE+CT":                "#00ACC1",
    "HE+Clin":              "#6D4C41",
    "BAL+CT":               "#3949AB",
    "BAL+Clin":             "#D81B60",
    "CT+Clin":              "#00897B",
    "HE+BAL+CT":            "#F4511E",
    "HE+BAL+Clin":          "#7CB342",
    "HE+CT+Clin":           "#039BE5",
    "BAL+CT+Clin":          "#8D6E63",
    "HE+BAL+CT+Clin":       "#FDD835",
}
DEFAULT_COMBO_COLOR = "#9E9E9E"

FOLD_COLORS = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A", "#00838F"]

# Uniform red=high-risk/imminent, blue=low-risk/far — NEVER use Reds/Blues alone
# Rule: closer to event → more red; higher hazard → more red; ACR+ → red; ACR- → blue
CMAP_HAZARD  = "RdBu_r"   # high hazard  = red  (low hazard = blue)
CMAP_TTE     = "RdBu"     # short TTE    = red  (long TTE   = blue)
CMAP_DENSITY = "RdBu_r"   # high density = red  (low density = blue)

# Nature-style plot params
NATURE_RC = {
    "font.family":      "sans-serif",
    "font.size":        7,
    "axes.linewidth":   0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth":  1.0,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
}
TWO_COL_W = 7.0  # inches
