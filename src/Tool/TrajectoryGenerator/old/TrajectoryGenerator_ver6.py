import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog
import csv
import os
from scipy.interpolate import CubicSpline

# --- 多次ベジェ曲線 ---
def bezier_curve(points, num=200):
    n = len(points) - 1
    t = np.linspace(0, 1, num)
    curve = np.zeros((num, 2))
    for i, p in enumerate(points):
        bernstein = (np.math.comb(n, i) * (t**i) * ((1-t)**(n-i)))[:, None]
        curve += bernstein * p
    return curve

# --- 補間関数 (split_div単位で座標を生成) ---
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

# --- Fix間を三次スプラインで接続 ---
def connect_fixed_curves(fixed_curves):
    if len(fixed_curves) < 2:
        return []
    connecting_segments = []
    for i in range(len(fixed_curves) - 1):
        end_point = fixed_curves[i][-1]
        start_point = fixed_curves[i+1][0]
        xs = [0, 1]
        ys_x = [end_point[0], start_point[0]]
        ys_y = [end_point[1], start_point[1]]
        cs_x = CubicSpline(xs, ys_x)
        cs_y = CubicSpline(xs, ys_y)
        t = np.linspace(0, 1, 50)
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
current_curve = None
output_points = []
fixed_curves = []
fixed_lines = []
connecting_lines = []

# --- クリックイベント ---
def onclick(event):
    global current_curve, output_points
    if event.inaxes != ax:
        return
    x, y = event.xdata, event.ydata
    selected_points.append(np.array([x, y]))
    ax.scatter(x, y, c='r', s=40)

    if len(selected_points) >= 2:
        if current_curve:
            current_curve.remove()
        curve = bezier_curve(np.array(selected_points))
        current_curve, = ax.plot(curve[:,0], curve[:,1], 'r-', lw=2)
        output_points = curve
    fig.canvas.draw()

# --- Fixボタン ---
def fix(event):
    global selected_points, current_curve, output_points, fixed_curves, fixed_lines, connecting_lines
    if len(output_points) == 0:
        return
    fixed_curves.append(output_points.copy())
    fixed_lines.append(ax.plot(output_points[:,0], output_points[:,1], 'r-', lw=2)[0])
    # 現在の編集中のデータをリセット
    if current_curve:
        current_curve.remove()
    selected_points.clear()
    current_curve = None
    output_points = []
    # Fix間スプラインを再計算
    for line in connecting_lines:
        line.remove()
    connecting_lines.clear()
    connections = connect_fixed_curves(fixed_curves)
    for seg in connections:
        connecting_lines.append(ax.plot(seg[:,0], seg[:,1], 'r--', lw=1.5)[0])
    fig.canvas.draw()

# --- Resetボタン (現在のクリック点だけ消去) ---
def reset(event):
    global selected_points, current_curve, output_points
    selected_points.clear()
    output_points = []
    if current_curve:
        current_curve.remove()
        current_curve = None
    fig.canvas.draw()

# --- All Resetボタン ---
def all_reset(event):
    global selected_points, current_curve, output_points, fixed_curves, fixed_lines, connecting_lines
    selected_points.clear()
    output_points = []
    current_curve = None
    fixed_curves.clear()
    for line in fixed_lines:
        line.remove()
    fixed_lines.clear()
    for line in connecting_lines:
        line.remove()
    connecting_lines.clear()
    ax.clear()
    ax.scatter(left_points[:,0], left_points[:,1], c='b', s=20, label='Left')
    ax.scatter(right_points[:,0], right_points[:,1], c='b', s=20, label='Right')
    ax.set_title('クリックで点を選択 → ベジェ曲線表示')
    ax.legend()
    fig.canvas.draw()

# --- Reset Prev Graphボタン ---
def reset_prev(event):
    global fixed_curves, fixed_lines, connecting_lines
    if fixed_curves:
        fixed_curves.pop()
        line = fixed_lines.pop()
        line.remove()
        # 接続ライン再計算
        for line in connecting_lines:
            line.remove()
        connecting_lines.clear()
        connections = connect_fixed_curves(fixed_curves)
        for seg in connections:
            connecting_lines.append(ax.plot(seg[:,0], seg[:,1], 'r--', lw=1.5)[0])
    fig.canvas.draw()

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
reset_ax = plt.axes([0.5, 0.01, 0.1, 0.05])
button_reset = Button(reset_ax, 'Reset')
button_reset.on_clicked(reset)

all_reset_ax = plt.axes([0.62, 0.01, 0.1, 0.05])
button_all_reset = Button(all_reset_ax, 'All Reset')
button_all_reset.on_clicked(all_reset)

reset_prev_ax = plt.axes([0.74, 0.01, 0.1, 0.05])
button_reset_prev = Button(reset_prev_ax, 'Reset Prev Graph')
button_reset_prev.on_clicked(reset_prev)

fix_ax = plt.axes([0.86, 0.01, 0.1, 0.05])
button_fix = Button(fix_ax, 'Fix')
button_fix.on_clicked(fix)

output_ax = plt.axes([0.1, 0.01, 0.1, 0.05])
button_output = Button(output_ax, 'Output')
button_output.on_clicked(output)

# --- イベント接続 ---
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
