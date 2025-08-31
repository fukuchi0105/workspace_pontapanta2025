import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.interpolate import make_interp_spline
import tkinter as tk
from tkinter import filedialog

# --- ベジェ曲線用関数 ---
def bezier_curve(points, num=100):
    n = len(points) - 1
    t = np.linspace(0, 1, num)
    curve = np.zeros((num, 2))
    for i, p in enumerate(points):
        bernstein = (np.math.comb(n, i) * (t**i) * ((1-t)**(n-i)))[:, None]
        curve += bernstein * p
    return curve

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
ax.plot(left_points[:,0], left_points[:,1], 'bo-', label='Left')
ax.plot(right_points[:,0], right_points[:,1], 'ro-', label='Right')
ax.set_title('クリックで点を選択 → ベジェ曲線表示')
ax.legend()

selected_points = []

# --- クリックイベント ---
def onclick(event):
    if event.inaxes != ax:
        return
    x, y = event.xdata, event.ydata
    selected_points.append([x, y])
    ax.plot(x, y, 'gx')  # 選択点を表示
    if len(selected_points) > 1:
        curve = bezier_curve(np.array(selected_points))
        ax.plot(curve[:,0], curve[:,1], 'g-')
    fig.canvas.draw()

# --- リセットボタン ---
def reset(event):
    global selected_points
    selected_points = []
    ax.clear()
    ax.plot(left_points[:,0], left_points[:,1], 'bo-', label='Left')
    ax.plot(right_points[:,0], right_points[:,1], 'ro-', label='Right')
    ax.set_title('クリックで点を選択 → ベジェ曲線表示')
    ax.legend()
    fig.canvas.draw()

# --- ボタン作成 ---
reset_ax = plt.axes([0.8, 0.01, 0.1, 0.05])
button = Button(reset_ax, 'Reset')
button.on_clicked(reset)

# --- イベント接続 ---
fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
