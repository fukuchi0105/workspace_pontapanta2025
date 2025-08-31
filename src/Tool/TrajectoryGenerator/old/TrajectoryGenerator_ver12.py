import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog
import csv
import os
from scipy.interpolate import CubicHermiteSpline

# --- ベジェ曲線 ---
def bezier_curve(points, num=200):
    n = len(points) - 1
    t = np.linspace(0, 1, num)
    curve = np.zeros((num, 2))
    for i, p in enumerate(points):
        bernstein = (np.math.comb(n, i) * (t**i) * ((1-t)**(n-i)))[:, None]
        curve += bernstein * p
    return curve

# --- 曲率連続スプライン ---
def connect_fixed_curves(fixed_curves):
    if len(fixed_curves) < 2:
        return []
    connecting_segments = []
    for i in range(len(fixed_curves) - 1):
        end_point = fixed_curves[i][-1]
        start_point = fixed_curves[i+1][0]

        if len(fixed_curves[i]) > 2:
            v1 = fixed_curves[i][-1] - fixed_curves[i][-2]
        else:
            v1 = start_point - end_point
        if len(fixed_curves[i+1]) > 2:
            v2 = fixed_curves[i+1][1] - fixed_curves[i+1][0]
        else:
            v2 = start_point - end_point

        xs = [0, 1]
        ys_x = [end_point[0], start_point[0]]
        ys_y = [end_point[1], start_point[1]]
        m_x = [v1[0], v2[0]]
        m_y = [v1[1], v2[1]]

        t = np.linspace(0, 1, 50)
        cs_x = CubicHermiteSpline(xs, ys_x, m_x)
        cs_y = CubicHermiteSpline(xs, ys_y, m_y)
        connecting_segments.append(np.column_stack((cs_x(t), cs_y(t))))
    return connecting_segments

# --- CSV選択 ---
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
data = pd.read_csv(file_path)

# --- データ ---
left_points = data[['left_x', 'left_y']].values
right_points = data[['right_x', 'right_y']].values

# --- プロット初期化 ---
fig, ax = plt.subplots()
ax.scatter(left_points[:,0], left_points[:,1], c='b', s=20)
ax.scatter(right_points[:,0], right_points[:,1], c='b', s=20)
ax.set_title('クリックで点を選択 → ベジェ曲線')

selected_points = []
fixed_points = []
edit_mode = False
dragging_info = None

# --- 再描画 ---
def redraw_all():
    ax.clear()
    ax.scatter(left_points[:,0], left_points[:,1], c='b', s=20)
    ax.scatter(right_points[:,0], right_points[:,1], c='b', s=20)

    # 現在の点
    for p in selected_points:
        ax.scatter(p[0], p[1], c='r', s=40)
    if len(selected_points) >= 2:
        curve = bezier_curve(np.array(selected_points))
        ax.plot(curve[:,0], curve[:,1], 'r-', lw=2)

    # Fix済み点と曲線
    for fp in fixed_points:
        for p in fp:
            ax.scatter(p[0], p[1], c='r', s=20)
        if len(fp) >= 2:
            curve = bezier_curve(fp)
            ax.plot(curve[:,0], curve[:,1], 'r-', lw=2)

    # 接続スプライン
    connections = connect_fixed_curves([bezier_curve(fp) for fp in fixed_points])
    for seg in connections:
        ax.plot(seg[:,0], seg[:,1], 'r--', lw=1.5)

    fig.canvas.draw_idle()

# --- クリックイベント ---
def onclick(event):
    global dragging_info
    if event.inaxes != ax:
        return

    if edit_mode:
        # 編集モード → 全ての点から最近傍を選択
        all_points = [(selected_points, i) for i in range(len(selected_points))]
        for fp in fixed_points:
            for i in range(len(fp)):
                all_points.append((fp, i))

        if not all_points:
            return

        click_pos = np.array([event.xdata, event.ydata])
        min_dist = float('inf')
        closest = None
        for plist, idx in all_points:
            d = np.linalg.norm(plist[idx] - click_pos)
            if d < min_dist:
                min_dist = d
                closest = (plist, idx)
        if closest and min_dist < 0.3:
            dragging_info = closest
        return

    # 通常モード
    selected_points.append(np.array([event.xdata, event.ydata]))
    redraw_all()

# --- ドラッグ ---
def on_motion(event):
    global dragging_info
    if not edit_mode or dragging_info is None:
        return
    if event.inaxes != ax:
        return
    plist, idx = dragging_info
    plist[idx] = np.array([event.xdata, event.ydata])
    redraw_all()

# --- リリース ---
def on_release(event):
    global dragging_info
    dragging_info = None
    redraw_all()

# --- Fix ---
def fix(event):
    global selected_points
    if len(selected_points) < 2:
        return
    fixed_points.append(selected_points.copy())
    selected_points.clear()
    redraw_all()

# --- Reset ---
def reset(event):
    selected_points.clear()
    redraw_all()

# --- All Reset ---
def all_reset(event):
    selected_points.clear()
    fixed_points.clear()
    redraw_all()

# --- Editモード ---
def toggle_edit(event):
    global edit_mode
    edit_mode = not edit_mode
    button_edit.label.set_text('Editing' if edit_mode else 'Edit')
    redraw_all()

# --- Output ---
def output(event):
    base = os.path.splitext(file_path)[0]
    all_points = []
    for fp in fixed_points:
        all_points.extend(bezier_curve(fp))
    if len(selected_points) >= 2:
        all_points.extend(bezier_curve(np.array(selected_points)))
    if not all_points:
        print("出力データなし")
        return
    out_path = base + "_bezier_output.csv"
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x','y'])
        writer.writerows(all_points)
    print(f"出力: {out_path}")

# --- ボタン ---
output_ax = plt.axes([0.1, 0.01, 0.1, 0.05])
button_output = Button(output_ax, 'Output')
button_output.on_clicked(output)

edit_ax = plt.axes([0.25, 0.01, 0.1, 0.05])
button_edit = Button(edit_ax, 'Edit')
button_edit.on_clicked(toggle_edit)

reset_ax = plt.axes([0.45, 0.01, 0.1, 0.05])
button_reset = Button(reset_ax, 'Reset')
button_reset.on_clicked(reset)

all_reset_ax = plt.axes([0.57, 0.01, 0.1, 0.05])
button_all_reset = Button(all_reset_ax, 'All Reset')
button_all_reset.on_clicked(all_reset)

fix_ax = plt.axes([0.69, 0.01, 0.1, 0.05])
button_fix = Button(fix_ax, 'Fix')
button_fix.on_clicked(fix)

# --- イベント ---
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

plt.show()
