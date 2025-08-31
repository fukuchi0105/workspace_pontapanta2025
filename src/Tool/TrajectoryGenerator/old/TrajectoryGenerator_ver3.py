import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog
import csv
import os

# --- 2点ベジェ関数 ---
def bezier_2point(p1, p2, num=50):
    t = np.linspace(0, 1, num)
    return np.outer(1 - t, p1) + np.outer(t, p2)

# --- 補間関数 (split_div単位で座標を生成) ---
def resample_points(points, split_div=0.5):
    new_points = [points[0]]
    total_dist = 0
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
ax.scatter(left_points[:,0], left_points[:,1], c='k', s=20, label='Left')
ax.scatter(right_points[:,0], right_points[:,1], c='k', s=20, label='Right')
ax.set_title('クリックで点を選択 → ベジェ曲線表示')
ax.legend()

selected_points = []
current_curve = None
output_points = []

# --- クリックイベント ---
def onclick(event):
    global current_curve
    if event.inaxes != ax:
        return
    x, y = event.xdata, event.ydata
    selected_points.append(np.array([x, y]))
    ax.scatter(x, y, c='g', s=40)  # 選択点を緑で表示
    
    # 2点目以降でベジェを描画（最新版のみ）
    if len(selected_points) >= 2:
        # 古い線を削除
        if current_curve:
            current_curve.remove()
        p1 = selected_points[-2]
        p2 = selected_points[-1]
        curve = bezier_2point(p1, p2)
        current_curve, = ax.plot(curve[:,0], curve[:,1], 'g-', lw=2)
        # 出力用に保存
        global output_points
        output_points = curve
    fig.canvas.draw()

# --- Resetボタン ---
def reset(event):
    global selected_points, current_curve, output_points
    selected_points = []
    output_points = []
    current_curve = None
    ax.clear()
    ax.scatter(left_points[:,0], left_points[:,1], c='k', s=20, label='Left')
    ax.scatter(right_points[:,0], right_points[:,1], c='k', s=20, label='Right')
    ax.set_title('クリックで点を選択 → ベジェ曲線表示')
    ax.legend()
    fig.canvas.draw()

# --- Outputボタン ---
def output(event):
    if len(output_points) == 0:
        print("出力するデータがありません")
        return
    # split_div間隔で再サンプリング
    split_div = 0.5
    sampled = resample_points(output_points, split_div)
    # CSV出力
    out_path = os.path.splitext(file_path)[0] + "_bezier_output.csv"
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        writer.writerows(sampled)
    print(f"補間後データを出力: {out_path}")
    # 補間点を青でプロット
    ax.scatter(sampled[:,0], sampled[:,1], c='b', s=30, label='Output Points')
    fig.canvas.draw()

# --- ボタン作成 ---
reset_ax = plt.axes([0.7, 0.01, 0.1, 0.05])
button_reset = Button(reset_ax, 'Reset')
button_reset.on_clicked(reset)

output_ax = plt.axes([0.82, 0.01, 0.1, 0.05])
button_output = Button(output_ax, 'Output')
button_output.on_clicked(output)

# --- イベント接続 ---
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
