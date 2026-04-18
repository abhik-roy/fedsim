# Shared color constants for consistent theming across all FEDSIM plots.
# All plot modules should import from here instead of defining their own.
# Palette: "Grayish Mint" — a sophisticated neutral palette for research dashboards.

THEME = "plotly_dark"

# Core palette
COLOR_PRIMARY = "#7FB5A0"        # muted sage/mint
COLOR_SECONDARY = "#5A9E87"      # deeper mint (hover/active)
COLOR_BG_DARK = "#1C1F26"        # warm dark gray background
COLOR_BG_SURFACE = "#232730"     # card/surface background
COLOR_BORDER = "#2D3140"         # subtle border
COLOR_TEXT = "#C8CCD4"           # soft white text
COLOR_TEXT_MUTED = "#8B919E"     # muted gray text

# Client status colors
COLOR_BENIGN = "#7FB5A0"           # mint (success/benign)
COLOR_ATTACKED = "#D4726A"         # muted coral (error/attacked)
COLOR_MALICIOUS_IDLE = "#6B7280"   # neutral gray (idle)
COLOR_EMPTY = "#1C1F26"            # dark background
COLOR_EXCLUDED = "#D4A76A"         # warm sand (attacked & caught)
COLOR_FALSE_POS = "#B088C4"        # muted lavender (benign & wrongly excluded)

# Strategy colors (desaturated, harmonious, distinguishable)
STRATEGY_COLORS = {
    "fedavg": "#8B9EC4",              # slate gray-blue (neutral baseline)
    "trimmed_mean": "#7FB5A0",        # sage
    "krum": "#6A9FD4",                # slate blue
    "median": "#B088C4",              # muted lavender
    "reputation": "#D4A76A",          # warm sand
    "custom:Reputation": "#D4A76A",   # warm sand (plugin alias)
    "bulyan": "#6AC4B8",              # soft teal
    "rfa": "#C4886A",                 # terracotta
}

# Accent
COLOR_ACCENT = "#7FB5A0"


def fedsim_layout_defaults() -> dict:
    """Return a dict of common Plotly layout overrides for the FEDSIM palette.

    Usage:
        fig.update_layout(**fedsim_layout_defaults(), title="My Chart", ...)
    """
    return dict(
        paper_bgcolor=COLOR_BG_DARK,
        plot_bgcolor=COLOR_BG_SURFACE,
        font=dict(color=COLOR_TEXT),
        xaxis=dict(
            gridcolor="rgba(45,49,64,0.6)",
            gridwidth=1,
            title_font=dict(color=COLOR_TEXT_MUTED),
            tickfont=dict(color=COLOR_TEXT_MUTED),
        ),
        yaxis=dict(
            gridcolor="rgba(45,49,64,0.6)",
            gridwidth=1,
            title_font=dict(color=COLOR_TEXT_MUTED),
            tickfont=dict(color=COLOR_TEXT_MUTED),
        ),
    )
