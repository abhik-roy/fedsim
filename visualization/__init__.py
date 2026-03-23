# Shared color constants for consistent theming across all FEDSIM plots.
# All plot modules should import from here instead of defining their own.

THEME = "plotly_dark"

# Client status colors
COLOR_BENIGN = "#2ecc71"           # green
COLOR_ATTACKED = "#e74c3c"         # red
COLOR_MALICIOUS_IDLE = "#95a5a6"   # gray
COLOR_EMPTY = "#1A1D23"            # dark background

# Strategy colors (ordered)
STRATEGY_COLORS = {
    "fedavg": "#e74c3c",
    "trimmed_mean": "#2ecc71",
    "krum": "#3498db",
    "median": "#9b59b6",
    "reputation": "#f39c12",
    "bulyan": "#1abc9c",
    "rfa": "#e67e22",
}

# Accent
COLOR_ACCENT = "#6C63FF"
