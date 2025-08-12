#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Curvature Viewer
----------------
Python 3.8-compatible interactive viewer for XY trajectory + curvature.

Usage:
    python curvature_viewer.py /path/to/trajectory.csv

CSV requirements:
    - Columns 'x' and 'y' are expected.
    - Optional column 'curvature' will be used if present (can toggle vs computed).
    - Other columns are ignored.

Controls:
    - Left click on the XY plot: select nearest point and show curvature info.
    - Key 't': toggle between CSV curvature and computed curvature.
    - Key 's': save current figures as PNGs (xy.png, curvature.png).
    - Key 'q' or close window: quit.

Notes:
    - Curvature is computed using derivatives with respect to arc length s:
      kappa = (x' * y'' - y' * x'') / ( (x'^2 + y'^2)^(3/2) )
      with numerical derivatives via numpy.gradient.
"""
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# For Python 3.8 / non-interactive environments, use a standard backend.
# (You can comment the next line if you prefer your local default backend.)
# matplotlib.use("TkAgg")

def compute_curvature(x: np.ndarray, y: np.ndarray):
    """Compute signed curvature along a 2D path parameterized by arc length.

    Returns:
        s: arc length (same length as x, y)
        kappa: signed curvature
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Remove NaNs if any
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # Arc length parameter s
    dx = np.diff(x)
    dy = np.diff(y)
    ds = np.hypot(dx, dy)
    # Prevent zeros that break gradient (duplicate points)
    ds[ds == 0] = 1e-12
    s = np.concatenate([[0.0], np.cumsum(ds)])

    # Derivatives wrt s using numpy.gradient
    # gradient(y, s) handles nonuniform spacing if s is provided
    x_s = np.gradient(x, s, edge_order=2)
    y_s = np.gradient(y, s, edge_order=2)
    x_ss = np.gradient(x_s, s, edge_order=2)
    y_ss = np.gradient(y_s, s, edge_order=2)

    denom = (x_s * x_s + y_s * y_s) ** 1.5
    # Guard against division by zero
    denom[denom == 0] = np.finfo(float).eps
    kappa = (x_s * y_ss - y_s * x_ss) / denom

    # Restore original length by inserting NaNs where x,y were NaN
    full_s = np.full(mask.shape, np.nan, dtype=float)
    full_kappa = np.full(mask.shape, np.nan, dtype=float)
    full_s[mask] = s
    full_kappa[mask] = kappa
    return full_s, full_kappa


def load_xy_curvature(csv_path: str):
    df = pd.read_csv(csv_path)
    # Flexible column matching
    col_x = None
    col_y = None
    for c in df.columns:
        lc = c.lower()
        if col_x is None and (lc == "x" or lc.endswith(".x") or lc.endswith("_x")):
            col_x = c
        if col_y is None and (lc == "y" or lc.endswith(".y") or lc.endswith("_y")):
            col_y = c

    if col_x is None or col_y is None:
        raise ValueError("Could not find x/y columns. Expected columns named like 'x' and 'y'.")

    x = df[col_x].to_numpy(dtype=float)
    y = df[col_y].to_numpy(dtype=float)

    # CSV curvature if present
    csv_kappa = None
    for c in df.columns:
        if c.lower() == "curvature":
            csv_kappa = df[c].to_numpy(dtype=float)
            break

    s, comp_kappa = compute_curvature(x, y)
    return x, y, s, comp_kappa, csv_kappa, df

class Args:
    pass
def main():
    # parser = argparse.ArgumentParser(description="Interactive XY curvature viewer")
    # parser.add_argument("csv", help="Path to CSV with x,y,(optional curvature)")
    # parser.add_argument("--point-size", type=float, default=10.0, help="Scatter point size")
    # parser.add_argument("--line", action="store_true", help="Connect points with a thin line overlay")
    # args = parser.parse_args()

    args = Args()
    args.csv = "/aichallenge/workspace/src/aichallenge_submit/simple_trajectory_generator/data/trajectory_output.csv"
    args.point_size = 10.0
    args.line = True

    x, y, s, kappa_comp, kappa_csv, df = load_xy_curvature(args.csv)

    use_csv = kappa_csv is not None  # start with CSV curvature if present
    kappa = kappa_csv if use_csv else kappa_comp

    # Prepare figures: XY scatter colored by curvature, and curvature vs s
    fig_xy = plt.figure("XY Trajectory (colored by curvature)", figsize=(8, 7))
    ax_xy = fig_xy.add_subplot(111)
    sc = ax_xy.scatter(x, y, c=kappa, s=args.point_size, cmap="viridis")
    if args.line:
        ax_xy.plot(x, y, linewidth=0.5, alpha=0.5)
    cb = plt.colorbar(sc, ax=ax_xy)
    cb.set_label("Curvature [1/m]")

    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.set_xlabel("X")
    ax_xy.set_ylabel("Y")
    ax_xy.set_title("XY colored by curvature ({})".format("CSV" if use_csv else "Computed"))

    # Selection marker & annotation
    sel_marker, = ax_xy.plot([], [], marker="o", markersize=10, fillstyle="none", linestyle="None")
    annot = ax_xy.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
                           bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9))

    # Curvature vs s figure
    fig_k = plt.figure("Curvature vs arc length", figsize=(9, 4.8))
    ax_k = fig_k.add_subplot(111)
    curve_line, = ax_k.plot(s, kappa, linewidth=1.5)
    vline = ax_k.axvline(0.0, linestyle="--", linewidth=1.0)
    ax_k.set_xlabel("Arc length s [m]")
    ax_k.set_ylabel("Curvature [1/m]")
    ax_k.set_title("Curvature vs s ({})".format("CSV" if use_csv else "Computed"))
    ax_k.grid(True, alpha=0.3)

    # Console summary
    def summarize(tag, arr):
        finite = np.isfinite(arr)
        if not finite.any():
            return f"{tag}: no finite values"
        return f"{tag}: min={np.nanmin(arr):.6g}, max={np.nanmax(arr):.6g}, mean={np.nanmean(arr):.6g}"

    print(summarize("Computed curvature", kappa_comp))
    if kappa_csv is not None:
        print(summarize("CSV curvature", kappa_csv))

    # Nearest-point finder
    def nearest_index(event_x, event_y):
        # Work in data coordinates: find nearest Euclidean distance
        dx = x - event_x
        dy = y - event_y
        idx = np.nanargmin(dx*dx + dy*dy)
        return int(idx)

    # Update visuals for a selected index
    def update_selection(idx):
        if idx is None or not np.isfinite(idx):
            return
        # Marker
        sel_marker.set_data([x[idx]], [y[idx]])
        # Annotation text
        kc = kappa_comp[idx] if np.isfinite(kappa_comp[idx]) else np.nan
        kcsv = kappa_csv[idx] if (kappa_csv is not None and np.isfinite(kappa_csv[idx])) else np.nan
        kused = kappa[idx] if np.isfinite(kappa[idx]) else np.nan
        radius = (np.inf if (not np.isfinite(kused) or kused == 0) else 1.0/abs(kused))

        txt = "i = {idx}\nX = {x:.3f}\nY = {y:.3f}\ns = {s:.3f} m\ncurv_used = {ku:.6g} 1/m\nR_used = {R:.3f} m".format(
            idx=idx, x=x[idx], y=y[idx], s=s[idx], ku=kused, R=radius
        )
        if kappa_csv is not None:
            txt += "\ncurv_csv  = {kc:.6g} 1/m".format(kc=kcsv)
        txt += "\ncurv_comp = {kk:.6g} 1/m".format(kk=kc)

        annot.set_text(txt)
        annot.xy = (x[idx], y[idx])
        # Vertical line in curvature plot
        vline.set_xdata([s[idx]])
        # Redraw
        fig_xy.canvas.draw_idle()
        fig_k.canvas.draw_idle()

        # Also print to console
        print(txt.replace("\n", " | "))

    # Mouse click handler
    def on_click(event):
        if event.inaxes != ax_xy:
            return
        if event.xdata is None or event.ydata is None:
            return
        idx = nearest_index(event.xdata, event.ydata)
        update_selection(idx)

    # Key press handler
    def on_key(event):
        nonlocal use_csv, kappa, sc, cb, curve_line
        if event.key == 't':
            # Toggle between CSV and computed curvature
            if kappa_csv is None:
                print("No 'curvature' column in CSV to toggle to.")
                return
            use_csv = not use_csv
            kappa = kappa_csv if use_csv else kappa_comp
            sc.set_array(kappa)  # recolor scatter
            cb.update_normal(sc)
            ax_xy.set_title("XY colored by curvature ({})".format("CSV" if use_csv else "Computed"))
            # Update curvature vs s plot
            curve_line.set_ydata(kappa)
            fig_xy.canvas.draw_idle()
            fig_k.canvas.draw_idle()
            print(f"Toggled to {'CSV' if use_csv else 'Computed'} curvature.")
        elif event.key == 's':
            fig_xy.savefig("xy_curvature.png", dpi=150, bbox_inches="tight")
            fig_k.savefig("curvature_vs_s.png", dpi=150, bbox_inches="tight")
            print("Saved xy_curvature.png and curvature_vs_s.png")
        elif event.key in ('q', 'escape'):
            plt.close('all')

    # Connect events
    cid_click = fig_xy.canvas.mpl_connect('button_press_event', on_click)
    cid_key = fig_xy.canvas.mpl_connect('key_press_event', on_key)
    # Initialize selection at mid index
    if len(x) > 0:
        update_selection(len(x)//2)

    plt.show()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)
        sys.exit(1)
