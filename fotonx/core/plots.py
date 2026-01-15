from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def _closed_pts(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts)
    if pts.size == 0:
        return pts
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
    return pts


def overlay_polys(
    ax,
    polys: Dict[str, List[np.ndarray]],
    colors: Optional[Dict[str, str]] = None,
    lw: float = 1.5,
    alpha: float = 1.0,
    linestyle: str = "-",
    zorder: int = 10,
):
    """
    Overlay polygon outlines on an existing axis (NO fill, no legend by default).
    Useful to overlay geometry over imshow/contour mode plots.

    polys: dict[name] -> list of (N,2) arrays
    colors: dict[name] -> matplotlib color (optional)
    """
    if colors is None:
        colors = {}

    for name, plist in polys.items():
        c = colors.get(name, "white")
        for poly in plist:
            pts = np.asarray(poly)
            if pts.size == 0:
                continue
            ptsc = _closed_pts(pts)
            ax.plot(
                ptsc[:, 0], ptsc[:, 1],
                color=c, lw=lw, alpha=alpha, linestyle=linestyle, zorder=zorder
            )


def plot_polys(
    polys: Dict[str, List[np.ndarray]],
    ax=None,
    *,
    fill: bool = True,
    alpha: float = 0.25,
    linewidth: float = 1.5,
    show_bbox: bool = True,
    title: str = "Extracted polygons",
    colors: Optional[Dict[str, str]] = None,
    show_legend: bool = True,
):
    """
    Plot extracted polygons (filled + outline) to verify GDS extraction.

    polys: dict[name] -> list of (N,2) arrays
    colors: optional dict[name]->matplotlib color string
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    colors = colors or {}

    # default colors (matplotlib cycle)
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    def get_color(i: int):
        return cycle[i % len(cycle)] if cycle else "C0"

    # global bbox
    xmin = +np.inf
    ymin = +np.inf
    xmax = -np.inf
    ymax = -np.inf

    i = 0
    for name, plist in polys.items():
        if not plist:
            continue

        c = colors.get(name, get_color(i))
        i += 1

        for k, pts in enumerate(plist):
            pts = np.asarray(pts)
            if pts.size == 0:
                continue

            xmin = min(xmin, float(pts[:, 0].min()))
            xmax = max(xmax, float(pts[:, 0].max()))
            ymin = min(ymin, float(pts[:, 1].min()))
            ymax = max(ymax, float(pts[:, 1].max()))

            ptsc = _closed_pts(pts)

            if fill:
                ax.fill(
                    ptsc[:, 0], ptsc[:, 1],
                    color=c, alpha=alpha,
                    label=name if k == 0 else None
                )
            ax.plot(ptsc[:, 0], ptsc[:, 1], color=c, lw=linewidth)

    if show_bbox and np.isfinite([xmin, ymin, xmax, ymax]).all():
        ax.plot(
            [xmin, xmax, xmax, xmin, xmin],
            [ymin, ymin, ymax, ymax, ymin],
            "--", lw=1.0, color="white"
        )

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (layout units)")
    ax.set_ylabel("y (layout units)")
    ax.grid(True, alpha=0.25)

    if show_legend:
        # Only show legend if at least one labeled artist exists
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) > 0:
            ax.legend(loc="best")

    return fig, ax
