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

# --- 曲率連続補間 (Hermite補間を使用) ---
def connect_fixed_curves(fixed_curves):
    if len(fixed_curves) < 2:
        return []
    connecting_segments = []
    for i in range(len(fixed_curves) - 1):
        end_point = fixed_curves[i][-1]
        start_point = fixed_curves[i+1][0]

        # 接続のための方向ベクトル（曲率連続性を考慮）
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
    if current_curve:
        current_curve.remove()
    selected_points.clear()
    current_curve = None
    output_points = []

    # 接続線を更新
    for line in connecting_lines:
        line.remove()
    connecting_lines.clear()
    connections = connect_fixed_curves(fixed_curves)
    for seg in connections:
        connecting_lines.append(ax.plot(seg[:,0], seg[:,1], 'r--', lw=1.5)[0])
    fig.canvas.draw()

# --- Resetボタン (現在のクリック点とグラフ削除) ---
def reset(event):
    global selected_points, current_curve, output_points
    selected_points.clear()
    output_points = []
    if current_curve:
        current_curve.remove()
        current_curve = None
    fig.canvas.draw()

# --- All Resetボタン (全削除) ---
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

# --- Reset Prev (今回と前回のFix削除) ---
def reset_prev(event):
    global selected_points, current_curve, output_points, fixed_curves, fixed_lines, connecting_lines
    # 現在の編集中の点と曲線を削除
    selected_points.clear()
    output_points = []
    if current_curve:
        current_curve.remove()
        current_curve = None
    # 前回のFixを削除
    if fixed_curves:
        fixed_curves.pop()
        line = fixed_lines.pop()
        line.remove()
    # 接続線更新
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

output_ax = plt.axes([0.1, 0.01, 0.1, 0.05])
button_output = Button(output_ax, 'Output')
button_output.on_clicked(output)

# --- イベント接続 ---
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
