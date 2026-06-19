"""Shared colour palette and Plotly theme."""

BG      = "#0e1117"
BG2     = "#1a1d27"
CARD    = "#1e2130"
BORDER  = "#2d3147"
TEXT    = "#e8eaf0"
MUTED   = "#6c7199"
ACCENT  = "#7c83ff"

ACR_COLORS = {
    "A0B0": "#3fb950",   # green  – no rejection
    "A1B0": "#f78166",   # orange – mild
    "A1B1": "#ff7b72",
    "A2B0": "#e05c4b",   # red
    "A2B1": "#e05c4b",
    "A3":   "#bf3a2e",   # severe
    "A0B1": "#d29922",   # yellow-ish – B-grade only
    "ACR+": "#e05c4b",
    "ACR-": "#3fb950",
}

TASK_COLORS = {
    "acr_cls":  "#7c83ff",
    "acr_surv": "#58a6ff",
    "clad_surv":"#f78166",
    "death_surv":"#ff7b72",
}

MOD_COLORS = {
    "HE":       "#a5d6ff",
    "BAL":      "#ffa657",
    "CT":       "#7ee787",
    "Clinical": "#f2cc60",
}

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor=BG,
    plot_bgcolor=BG2,
    font=dict(family="Inter, sans-serif", color=TEXT, size=12),
)


def card_css() -> str:
    return f"""
    <style>
    [data-testid="stAppViewContainer"] {{ background: {BG}; }}
    [data-testid="stSidebar"] {{ background: {BG2}; }}
    .metric-card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }}
    .metric-card .label {{ color: {MUTED}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .06em; }}
    .metric-card .value {{ color: {TEXT}; font-size: 1.6rem; font-weight: 700; margin-top: 4px; }}
    .metric-card .sub   {{ color: {MUTED}; font-size: 0.82rem; margin-top: 2px; }}
    .section-title {{ color: {MUTED}; font-size: 0.75rem; text-transform: uppercase;
                      letter-spacing: .08em; margin: 18px 0 6px 0; }}
    </style>
    """


def metric_card(label: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="metric-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'<div class="sub">{sub}</div>'
        f"</div>"
    )
