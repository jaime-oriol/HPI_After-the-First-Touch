"""common — Shared LIGHT OPTA style for the set-piece visualisations.

White background, black pitch lines, Chakra Petch type, saturated ATT blue /
DEF red nodes, mplsoccer football marker. One house identity, reused by every plot.

Exports
  Palette : BG, TEXT, LEGEND, GRID, PITCH, ATT, DEF, GK, ROLE_COLORS, REACH_CMAP
  Type    : FONT_STACK (Chakra Petch registered into rcParams), PE_S
  Pitch   : VPKW (mplsoccer VerticalPitch kwargs, half pitch, opta coords),
            new_half_pitch()
  Icons   : ICON_PATHS (corner/shot PNG silhouettes), _draw_icon, _draw_stat_icon
  Header  : draw_header (crest LEFT + title/subtitle + JO logo RIGHT)
  Legend  : draw_side_legend (right column: stats on top, grey divider,
            node key + ball on the bottom)
  Stats   : corner_stats (the match-level counts draw_side_legend's `stats` row takes)

Images degrade rather than fail: a missing or unreadable font, crest, logo or icon is
skipped and the figure is drawn without it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle
from mplsoccer import VerticalPitch
from mplsoccer.soccer.markers import scatter_football
from PIL import Image


# ── Repo paths, fonts and icons ──────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FONT_DIR  = _REPO_ROOT / "figures" / "fonts"
_LOGO_PATH = _REPO_ROOT / "figures" / "logos" / "jo_logo.png"

for _f in sorted(_FONT_DIR.glob("ChakraPetch*.ttf")):
    try:
        _fm.fontManager.addfont(str(_f))
    except Exception:
        pass                       # the font stack below falls back

# Goals reuse the mplsoccer football marker, the same as the "Ball" legend row.
_ICON_DIR  = _REPO_ROOT / "figures" / "icons"
ICON_PATHS = {"corner": _ICON_DIR / "corner.png", "shot": _ICON_DIR / "player.png"}


def _draw_icon(fig: plt.Figure, ax: plt.Axes, x: float, y: float, kind: str,
               h_frac: float) -> None:
    """Place a stat PNG icon centred at (x, y) in axes-fraction coords.
    h_frac = icon height as a fraction of the figure height."""
    p = ICON_PATHS.get(kind)
    img = _load_rgba(p) if p else None
    if img is None:
        return
    img = _normalize_img(img)
    zoom = (h_frac * fig.get_size_inches()[1] * fig.dpi) / img.shape[0]
    ab = AnnotationBbox(OffsetImage(img, zoom=zoom), (x, y), frameon=False,
                        xycoords=ax.transAxes, box_alignment=(0.5, 0.5))
    ab.set_clip_on(False)
    ax.add_artist(ab)


def _draw_stat_icon(fig: plt.Figure, ax: plt.Axes, x: float, y: float, kind: str,
                    h_frac: float, node_ms: float) -> None:
    """Stat-row icon, dispatched on `kind`:
      hex colour -> filled role chip | 'edge' -> red node + black ring (an edge protector) |
      'goal' -> football | else -> PNG silhouette (corner/shot)."""
    if isinstance(kind, str) and kind.startswith("#"):
        ax.plot(x, y, "o", ms=node_ms, color=kind, markeredgecolor=TEXT,
                markeredgewidth=1.2, transform=ax.transAxes, clip_on=False, zorder=5)
    elif kind == "edge":
        ax.plot(x, y, "o", ms=node_ms, color=DEF, markeredgecolor=TEXT,
                markeredgewidth=1.2, transform=ax.transAxes, clip_on=False, zorder=5)
        ax.plot(x, y, "o", ms=node_ms * 1.75, markerfacecolor="none", markeredgecolor=TEXT,
                markeredgewidth=2.2, transform=ax.transAxes, clip_on=False, zorder=6)
    elif kind == "goal":
        scatter_football([x], [y], ax=ax, s=node_ms ** 2, transform=ax.transAxes,
                         zorder=5, clip_on=False)
    else:
        _draw_icon(fig, ax, x, y, kind, h_frac)


# ── Palette (LIGHT OPTA) ────────────────────────────────────────────────
BG      = "#ffffff"
TEXT    = "#000000"
LEGEND  = "#666666"
GRID    = "#dddddd"
PITCH   = "#000000"

ATT     = "#3b82f6"
DEF     = "#ef4444"
GK      = "#000000"

# Commitment-signature roles: meet the delivery, stake the drop, hold back for rest defence.
ROLE_COLORS = {"attack_cross": "#2563eb",
               "stake_drop":   "#f59e0b",
               "hold_back":    "#94a3b8"}

# The reach field s04 draws: defending red <- contested white -> attacking blue.
REACH_CMAP = LinearSegmentedColormap.from_list("reach", [DEF, "#ffffff", ATT])


# ── Typography ──────────────────────────────────────────────────────────
FONT_STACK = ["Chakra Petch", "Inter", "Helvetica Neue", "Arial", "DejaVu Sans"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": FONT_STACK,
    "font.size": 10,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": None,
    "axes.facecolor": BG,
    "figure.facecolor": BG,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
})

PE_S = [pe.withStroke(linewidth=2.2, foreground="black")]            # halo behind shirt numbers


# ── Pitch: vertical half, opta 0-100, attacking goal (x=100) at the TOP ──
VPKW = dict(
    pitch_type="opta",
    half=True,
    pitch_color=BG,
    line_color=PITCH,
    line_zorder=2,
    linewidth=1.2,
    pad_top=2.0,
    pad_bottom=2.0,        # past the halfway line, so nodes near it are not clipped
    pad_left=2.0,
    pad_right=2.0,
)


def new_half_pitch(**over: object) -> VerticalPitch:
    """Vertical half pitch in house style. `over` overrides any VPKW key."""
    return VerticalPitch(**{**VPKW, **over})


# ── Logo helpers ────────────────────────────────────────────────────────
_JO_LOGO_SCALE    = 0.478
_ESCUDO_TARGET_PX = 100


def _trim_transparent(img: np.ndarray, thr: int = 10) -> np.ndarray:
    """Crop outer fully-transparent rows/cols (alpha <= thr) so a crest fills its box."""
    if img.ndim != 3 or img.shape[2] < 4:
        return img
    ys, xs = np.where(img[..., 3] > thr)
    if len(ys) == 0:
        return img
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _load_rgba(path: Path | str, trim: bool = True) -> Optional[np.ndarray]:
    """Load an image as an RGBA array (trimmed of transparent borders), or None."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        img = np.asarray(Image.open(p).convert("RGBA"))
        return _trim_transparent(img) if trim else img
    except Exception:
        return None


def _normalize_img(img: np.ndarray, target_h: int = _ESCUDO_TARGET_PX) -> np.ndarray:
    """Resize to target_h px tall (keep aspect, LANCZOS) so all logos share one scale."""
    h = img.shape[0]
    if h == target_h:
        return img
    new_w = max(1, int(round(img.shape[1] * target_h / h)))
    return np.asarray(Image.fromarray(img).resize((new_w, target_h), Image.LANCZOS))


# ── Header: crest LEFT | title + subtitle | JO logo RIGHT ───────────────
_TEXT_FILL_FACTOR = 1.2


def draw_header(fig: plt.Figure, *, title: str, subtitle: str | list[str] | None = None,
                crest_path: Path | str | None = None,
                hdr_band: Tuple[float, float] = (0.85, 1.0),
                crest_x: float = 0.04, text_x: float = 0.13, jo_x: float = 0.96,
                logo_h_frac: float = 0.62, text_h_frac: float = 0.5,
                title_share: float = 2.0 / 3.0,
                crest_scale: float = 0.78) -> None:
    """Header band: optional crest on the left, title (+ N subtitle lines), JO logo on the right.

    Every size derives from the band height, so the header scales with the figure. The text
    block is split `title_share` / rest between title and subtitle, the crest is centred on the
    whole block, and `crest_scale` sizes the crest independently of the JO logo, which is
    centred on `jo_x` rather than anchored to the edge."""
    bot, top = hdr_band
    h = top - bot
    fig_h_in = fig.get_size_inches()[1]
    band_h_in = h * fig_h_in

    if subtitle is None or subtitle == "":
        sub_lines: list[str] = []
    elif isinstance(subtitle, (list, tuple)):
        sub_lines = [str(s) for s in subtitle if s]
    else:
        sub_lines = [str(subtitle)]
    n_sub = max(1, len(sub_lines))

    sub_share = 1.0 - title_share
    block_bot = 0.5 - text_h_frac / 2.0
    block_top = 0.5 + text_h_frac / 2.0
    boundary = block_bot + sub_share * text_h_frac
    title_y = bot + ((boundary + block_top) / 2.0) * h
    jo_y = bot + 0.5 * h

    title_size = (title_share * text_h_frac * band_h_in) * 72.0 / _TEXT_FILL_FACTOR + 2
    sub_size = (sub_share * text_h_frac * band_h_in) * 72.0 / _TEXT_FILL_FACTOR + 5

    if sub_lines:
        line_h_frac = (sub_size * _TEXT_FILL_FACTOR / 72.0) / fig_h_in
        last_sub_y = (title_y - (title_size * _TEXT_FILL_FACTOR / 72.0) / fig_h_in * 0.55
                      - (n_sub - 0.5) * line_h_frac)
        crest_y = (title_y + last_sub_y) / 2.0
    else:
        line_h_frac = 0.0
        crest_y = title_y

    target_logo_px = logo_h_frac * band_h_in * fig.dpi

    if crest_path is not None:
        img = _load_rgba(crest_path)
        if img is not None:
            img = _normalize_img(img)
            ab = AnnotationBbox(
                OffsetImage(img, zoom=(target_logo_px * crest_scale) / img.shape[0]),
                (crest_x, crest_y), frameon=False,
                xycoords="figure fraction", box_alignment=(0.0, 0.5))
            ab.set_clip_on(False)
            fig.add_artist(ab)

    fig.text(text_x, title_y, title, ha="left", va="center",
             color=TEXT, fontsize=title_size, fontweight=700)   # Chakra Petch tops out at Bold

    if sub_lines:
        sub_top = title_y - (title_size * _TEXT_FILL_FACTOR / 72.0) / fig_h_in * 0.55
        for i, line in enumerate(sub_lines):
            fig.text(text_x, sub_top - (i + 0.5) * line_h_frac, line,
                     ha="left", va="center", color=TEXT,
                     fontsize=sub_size, fontweight=500)

    img_jo = _load_rgba(_LOGO_PATH)
    if img_jo is not None:
        img_jo = _normalize_img(img_jo)
        ab = AnnotationBbox(
            OffsetImage(img_jo, zoom=(target_logo_px * _JO_LOGO_SCALE) / img_jo.shape[0]),
            (jo_x, jo_y), frameon=False,
            xycoords="figure fraction", box_alignment=(0.5, 0.5))
        ab.set_clip_on(False)
        fig.add_artist(ab)


# ── Side legend panel (right column) ────────────────────────────────────
def _block_axes(fig: plt.Figure, left: float, bottom: float, width: float,
                height: float) -> plt.Axes:
    """Clean 0-1 axes (no spines/ticks), so callers place a legend block by fraction."""
    ax = fig.add_axes([left, bottom, width, height])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def draw_side_legend(fig: plt.Figure, box: Tuple[float, float, float, float], *,
                     att_team_name: str = "Attacking",
                     def_team_name: str = "Defending",
                     att_color: str = ATT, def_color: str = DEF,
                     node_legend: list[tuple] | None = None,
                     stats: list[tuple] | None = None,
                     gradient: dict | None = None) -> None:
    """Right-column panel, the same height as the pitch.

    TOP half: either a vertical colour bar with a marker at `gradient["value"]` (the control
    surface), or up to 4 stat rows, each an icon with its number above its label. `stats` is a
    list of (kind, value, label) with kind in {"corner", "shot", "goal", "edge"} or a hex colour
    for a role chip. A grey divider sits at mid-height. BOTTOM half: `node_legend` as (color,
    label) rows (attacking/defending by default), then the ball.

    Laid out in axes fraction: every icon shares one column at x = 0.30, node labels start at
    0.50 and stat numbers centre on 0.64, so the content sits in the central half of the panel.
    Nodes and the legend ball share one marker size, which is also the goals icon's."""
    ax = _block_axes(fig, *box)

    icon_x, text_x, num_x = 0.30, 0.50, 0.64
    node_ms = 25
    ball_s = node_ms ** 2

    ax.plot([0.15, 0.90], [0.475, 0.475], color=GRID, lw=2.0, transform=ax.transAxes, zorder=2)

    if gradient is not None:
        bx0, bx1, by0, by1 = 0.20, 0.36, 0.56, 0.90
        ramp = np.linspace(1.0, 0.0, 128).reshape(-1, 1)
        ax.imshow(ramp, extent=[bx0, bx1, by0, by1], cmap=gradient["cmap"], vmin=0, vmax=1,
                  origin="upper", aspect="auto", zorder=3)
        ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, ec=TEXT, lw=1.0, zorder=4))
        ax.text((bx0 + bx1) / 2, by1 + 0.035, gradient.get("top_label", "Att"), ha="center", va="bottom",
                fontsize=15, fontweight="bold", color=LEGEND, transform=ax.transAxes)
        ax.text((bx0 + bx1) / 2, by0 - 0.035, gradient.get("bottom_label", "Def"), ha="center", va="top",
                fontsize=15, fontweight="bold", color=LEGEND, transform=ax.transAxes)
        v = float(gradient.get("value", 0.5))
        yv = by0 + v * (by1 - by0)
        ax.plot([bx0 - 0.035, bx1 + 0.035], [yv, yv], color=TEXT, lw=2.4, zorder=5, transform=ax.transAxes)
        ax.text(bx1 + 0.12, yv, f"{v:.0%}", ha="left", va="center", fontsize=20,
                fontweight="bold", color=LEGEND, transform=ax.transAxes)
    elif stats:
        rows_y = list(np.linspace(0.88, 0.55, len(stats[:4])))
        icon_h, num_sz, lab_sz = 0.055, 25, 16
        for (kind, value, label), y in zip(stats[:4], rows_y, strict=True):
            _draw_stat_icon(fig, ax, icon_x, y, kind, icon_h, node_ms)
            ax.text(num_x, y + 0.014, str(value), ha="center", va="center",
                    fontsize=num_sz, fontweight="bold", color=LEGEND, transform=ax.transAxes)
            ax.text(num_x, y - 0.018, label, ha="center", va="center",
                    fontsize=lab_sz, fontweight="bold", color=LEGEND, transform=ax.transAxes)

    items = node_legend if node_legend is not None else [
        (att_color, att_team_name), (def_color, def_team_name)]
    lab_sz = 17
    rows_y = list(np.linspace(0.40, 0.06, len(items) + 1))   # one row more than items: the ball
    for (color, name), y in zip(items, rows_y, strict=False):
        ax.plot(icon_x, y, "o", ms=node_ms, color=color, markeredgecolor=TEXT,
                markeredgewidth=1.2, alpha=0.95, transform=ax.transAxes, clip_on=False, zorder=5)
        ax.text(text_x, y, name, ha="left", va="center", fontsize=lab_sz,
                color=LEGEND, fontweight="bold", transform=ax.transAxes)
    yb = rows_y[-1]
    scatter_football([icon_x], [yb], ax=ax, s=ball_s, transform=ax.transAxes,
                     zorder=5, clip_on=False)
    ax.text(text_x, yb, "Ball", ha="left", va="center", fontsize=lab_sz,
            color=LEGEND, fontweight="bold", transform=ax.transAxes)


# ── Legend counts ───────────────────────────────────────────────────────
def corner_stats(events: pd.DataFrame, team: str) -> list:
    """Match-level corner outcomes for a team, shaped for `draw_side_legend(stats=...)`:
    corners taken, shots from corners, goals from corners.

    It reads `play_pattern.name`, which spans the whole possession the corner started and
    is a looser net than s01's per-delivery grammar: fine for a legend count, wrong for
    modelling. Nothing here may stand in for a segment."""
    is_team = events["team.name"] == team
    n_corn = int((is_team & (events["pass.type.name"] == "Corner")).sum())
    from_corner = is_team & (events["play_pattern.name"] == "From Corner")
    shots = from_corner & (events["type.name"] == "Shot")
    n_shots = int(shots.sum())
    n_goals = int((shots & (events["shot.outcome.name"] == "Goal")).sum())
    return [("corner", n_corn, "corners"), ("shot", n_shots, "shots"),
            ("goal", n_goals, "goals")]
