import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog
import csv
import os
from scipy.interpolate import CubicHermiteSpline

# --- 多次ベジェ曲線 ---
def bezier_curve(points, num=200):
    n = len(points) - 1
    t = np.linspace(0, 1, num)
    curve = np.zeros((num, 2))
    for i, p in enumerate(points):
        bernstein = (np.math.comb(n, i) * (t**i) * ((1-t)**(n-i)))[:, None]
        curve += bernstein * p
    return curve

# --- 補間関数 ---
def resample_points(points, split_div=0.5):
    new_points = [points[0]]
    for i in range(1, len(points)):
        p1 = points[i-1]
        p2 = points[i]
        seg_len = np.linalg.norm(p2 - p1)
        d = 0
        while d + split_div < seg_len:
            d += split_div
            ratio = d / seg_len
            new_points.append(p1 + ratio * (p2 - p1))
        new_points.append(p2)
    return np.array(new_points)

# --- 曲率連続補間 ---
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
        spline_points = np.column_stack((cs_x(t), cs_y(t)))
        connecting_segments.append(spline_points)
    return connecting_segments

# --- CSVファイル選択 ---
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
data = pd.read_csv(file_path)

# --- データ読み込み ---
left_points = data[['left_x', 'left_y']].values
right_points = data[['right_x', 'right_y']].values

# --- プロット初期化 ---
fig, ax = plt.subplots()
ax.scatter(left_points[:,0], left_points[:,1], c='b', s=20, label='Left')
ax.scatter(right_points[:,0], right_points[:,1], c='b', s=20, label='Right')
ax.set_title('クリックで点を選択 → ベジェ曲線表示')
ax.legend()

selected_points = []
click_artists = []
current_curve = None
output_points = []
fixed_curves = []
fixed_lines = []
fixed_points = []  # Fix済み点を保持
connecting_lines = []

edit_mode = False
dragging_info = None  # (リスト, インデックス)

# --- 再描画 ---
def redraw_current():
    global current_curve, output_points
    if current_curve:
        current_curve.remove()
        current_curve = None
    if len(selected_points) >= 2:
        curve = bezier_curve(np.array(selected_points))
        current_curve, = ax.plot(curve[:,0], curve[:,1], 'r-', lw=2)
        output_points = curve
    update_connections()
    fig.canvas.draw()

def update_connections():
    global connecting_lines
    for line in connecting_lines:
        line.remove()
    connecting_lines.clear()
    connections = connect_fixed_curves(fixed_curves)
    for seg in connections:
        connecting_lines.append(ax.plot(seg[:,0], seg[:,1], 'r--', lw=1.5)[0])

# --- クリックイベント ---
def onclick(event):
    global current_curve, output_points, dragging_info
    if event.inaxes != ax:
        return
    if edit_mode:
        all_points = [(selected_points, i) for i in range(len(selected_points))] + \
                     [(fp, i) for fp in fixed_points for i in range(len(fp))]
        if not all_points:
            return
        click_pos = np.array([event.xdata, event.ydata])
        closest = None
        min_dist = float('inf')
        for plist, idx in all_points:
            d = np.linalg.norm(plist[idx] - click_pos)
            if d < min_dist:
                min_dist = d
                closest = (plist, idx)
        if closest and min_dist < 0.3:
            dragging_info = closest
        return

    # 通常モード
    p = np.array([event.xdata, event.ydata])
    selected_points.append(p)
    click_artists.append(ax.scatter(p[0], p[1], c='r', s=40))
    redraw_current()

# --- ドラッグイベント ---
def on_motion(event):
    global dragging_info
    if not edit_mode or dragging_info is None:
        return
    if event.inaxes != ax:
        return
    plist, idx = dragging_info
    plist[idx] = np.array([event.xdata, event.ydata])
    for artist in click_artists:
        artist.remove()
    click_artists.clear()
    for p in selected_points:
        click_artists.append(ax.scatter(p[0], p[1], c='r', s=40))
    for fp in fixed_points:
        for p in fp:
            ax.scatter(p[0], p[1], c='r', s=20)
    # Fix済みベジェ再描画
    for line in fixed_lines:
        line.remove()
    fixed_lines.clear()
    for curve, fp in zip(fixed_curves, fixed_points):
        new_curve = bezier_curve(fp)
        fixed_lines.append(ax.plot(new_curve[:,0], new_curve[:,1], 'r-', lw=2)[0])
    redraw_current()

# --- マウスリリース ---
def on_release(event):
    global dragging_info
    dragging_info = None

# --- Fixボタン ---
def fix(event):
    global selected_points, click_artists, current_curve, output_points, fixed_curves, fixed_lines, fixed_points
    if len(output_points) == 0:
        return
    fixed_curves.append(output_points.copy())
    fixed_points.append(selected_points.copy())
    fixed_lines.append(ax.plot(output_points[:,0], output_points[:,1], 'r-', lw=2)[0])
    if current_curve:
        current_curve.remove()
    for artist in click_artists:
        artist.remove()
    click_artists.clear()
    selected_points.clear()
    current_curve = None
    output_points = []
    update_connections()
    fig.canvas.draw()

# --- Resetボタン ---
def reset(event):
    global selected_points, click_artists, current_curve, output_points
    for artist in click_artists:
        artist.remove()
    click_artists.clear()
    selected_points.clear()
    output_points = []
    if current_curve:
        current_curve.remove()
        current_curve = None
    fig.canvas.draw()

# --- All Reset ---
def all_reset(event):
    global selected_points, click_artists, current_curve, output_points, fixed_curves, fixed_lines, connecting_lines, fixed_points
    ax.clear()
    selected_points.clear()
    click_artists.clear()
    output_points = []
    current_curve = None
    fixed_curves.clear()
    fixed_points.clear()
    for line in fixed_lines:
        line.remove()
    fixed_lines.clear()
    for line in connecting_lines:
        line.remove()
    connecting_lines.clear()
    ax.scatter(left_points[:,0], left_points[:,1], c='b', s=20, label='Left')
    ax.scatter(right_points[:,0], right_points[:,1], c='b', s=20, label='Right')
    ax.set_title('クリックで点を選択 → ベジェ曲線表示')
    ax.legend()
    fig.canvas.draw()

# --- Reset Prev ---
def reset_prev(event):
    global selected_points, click_artists, current_curve, output_points, fixed_curves, fixed_lines, fixed_points
    for artist in click_artists:
        artist.remove()
    click_artists.clear()
    selected_points.clear()
    output_points = []
    if current_curve:
        current_curve.remove()
        current_curve = None
    if fixed_curves:
        fixed_curves.pop()
        fixed_points.pop()
        line = fixed_lines.pop()
        line.remove()
    update_connections()
    fig.canvas.draw()

# --- Editボタン ---
def toggle_edit(event):
    global edit_mode
    edit_mode = not edit_mode
    color = 'lightgreen' if edit_mode else 'lightgray'
    edit_ax.set_facecolor(color)
    fig.canvas.draw()
    if edit_mode:
        print("Edit Mode ON: ドラッグで点を移動")
    else:
        print("Edit Mode OFF: 編集完了")
        redraw_current()

# --- Outputボタン ---
def output(event):
    global fixed_curves, output_points
    all_points = []
    for curve in fixed_curves:
        all_points.extend(curve)
    if len(output_points) > 0:
        all_points.extend(output_points)
    if len(all_points) == 0:
        print("出力するデータがありません")
        return
    all_points = np.array(all_points)
    split_div = 0.5
    sampled = resample_points(all_points, split_div)
    out_path = os.path.splitext(file_path)[0] + "_bezier_output.csv"
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        writer.writerows(sampled)
    print(f"補間後データを出力: {out_path}")
    ax.scatter(sampled[:,0], sampled[:,1], c='b', s=30, label='Output Points')
    fig.canvas.draw()

# --- ボタン作成 ---
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

reset_prev_ax = plt.axes([0.69, 0.01, 0.1, 0.05])
button_reset_prev = Button(reset_prev_ax, 'Reset Prev')
button_reset_prev.on_clicked(reset_prev)

fix_ax = plt.axes([0.81, 0.01, 0.08, 0.05])
button_fix = Button(fix_ax, 'Fix')
button_fix.on_clicked(fix)

# --- イベント接続 ---
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

plt.show()
