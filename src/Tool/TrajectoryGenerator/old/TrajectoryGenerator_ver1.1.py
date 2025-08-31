import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import tkinter as tk
from tkinter import filedialog
import csv
from datetime import datetime
from scipy.interpolate import CubicHermiteSpline


# ============================================
# Model: データ保持と数値計算ロジックを担当
# ============================================
class TrajectoryModel:
    """
    サーキットライン編集のデータモデル
    ・左ライン、右ラインの点群を保持
    ・編集中のポイント、固定されたポイント、比較用ポイントを管理
    ・ベジェ曲線生成やスプライン接続、CSV出力のロジックを実装
    """
    def __init__(self, left_points, right_points):
        # 入力データ
        self.left_points = left_points
        self.right_points = right_points

        # 編集状態
        self.selected_points = []      # 現在編集中のポイント
        self.fixed_points = []         # Fixされたポイントグループ
        self.comparison_points = []    # 比較用ポイント群
        self.edit_mode = False         # 編集モードフラグ
        self.dragging_info = None      # ドラッグ対象（ポイントとインデックス）
        
        # 初期点の設定
        self.initial_point = np.array([89630.067488, 43130.694559])
        self.selected_points.append(self.initial_point.copy())

    # --- 初期点がfixed_pointsに含まれているか判定 ---
    def is_initial_point_fixed(self):
        """
        初期点が fixed_points の中に含まれているか判定
        """
        for fp in self.fixed_points:
            for p in fp:
                if np.allclose(p, self.initial_point):
                    return True
        return False

    # --- ベジェ曲線生成 ---
    @staticmethod
    def bezier_curve(points, num=200):
        """
        指定した制御点からベジェ曲線を生成
        :param points: 制御点 (np.array)
        :param num: 分割数
        """
        n = len(points) - 1
        t = np.linspace(0, 1, num)
        curve = np.zeros((num, 2))
        for i, p in enumerate(points):
            bernstein = (math.comb(n, i) * (t**i) * ((1-t)**(n-i)))[:, None]
            curve += bernstein * p
        return curve

    # --- 曲率連続スプライン接続 ---
    @staticmethod
    def connect_fixed_curves(fixed_curves):
        """
        複数のベジェ曲線をCubicHermiteSplineで滑らかに接続
        :param fixed_curves: ベジェ曲線群
        """
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

    # --- CSV出力 ---
    def export_trajectory(self):
        """
        現在の編集結果を2種類のCSVで保存
        1. _Trajectory.csv: ベジェ曲線で補間した最終ライン
        2. _LogPoints.csv: 元の編集点（scene_id付き）
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # Trajectory生成
        all_points = []
        for fp in self.fixed_points:
            all_points.extend(self.bezier_curve(fp))
        if len(self.selected_points) >= 2:
            all_points.extend(self.bezier_curve(np.array(self.selected_points)))
        if not all_points:
            print("No Output Data")
            return None

        # Trajectory出力
        traj_path = f"{timestamp}_Trajectory.csv"
        with open(traj_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x','y'])
            writer.writerows(all_points)
        print(f"Trajectory Output: {traj_path}")

        # LogPoints出力
        log_path = f"{timestamp}_LogPoints.csv"
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['scene_id','x','y'])
            for sid, fp in enumerate(self.fixed_points, start=1):
                for p in fp:
                    writer.writerow([sid, p[0], p[1]])
            for p in self.selected_points:
                writer.writerow([0, p[0], p[1]])
        print(f"Log Output: {log_path}")

        return traj_path, log_path


# ============================================
# View: 描画・UI（Matplotlibベース）
# ============================================
class TrajectoryView:
    """
    Matplotlibを使ってデータを可視化し、
    ボタンや描画のUIを担当するクラス
    """
    def __init__(self, model):
        self.model = model
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal', adjustable='datalim')
        self.ax.set_title('Trajectory Generator')

        # ボタン作成
        self.create_buttons()

        # 初期描画
        self.redraw_all()

    # --- ボタンUI生成 ---
    def create_buttons(self):
        self.button_output = Button(plt.axes([0.02, 0.01, 0.1, 0.05]), 'Output')
        self.button_read = Button(plt.axes([0.14, 0.01, 0.1, 0.05]), 'Read Log')
        self.button_comp = Button(plt.axes([0.26, 0.01, 0.1, 0.05]), 'Comparison')
        self.button_edit = Button(plt.axes([0.38, 0.01, 0.1, 0.05]), 'Edit')
        self.button_reset = Button(plt.axes([0.50, 0.01, 0.1, 0.05]), 'Reset')
        self.button_reset_prev = Button(plt.axes([0.62, 0.01, 0.1, 0.05]), 'Reset Prev')
        self.button_all_reset = Button(plt.axes([0.74, 0.01, 0.1, 0.05]), 'All Reset')
        self.button_fix = Button(plt.axes([0.86, 0.01, 0.1, 0.05]), 'Fix')

    # --- 再描画 ---
    def redraw_all(self):
        """
        現在のモデルデータをすべて描画し直す
        """
        m = self.model
        self.ax.clear()

        # 左右ライン描画
        self.ax.scatter(m.left_points[:,0], m.left_points[:,1], c='b', s=20)
        self.ax.scatter(m.right_points[:,0], m.right_points[:,1], c='b', s=20)

        # 比較用ポイント描画
        for fp in m.comparison_points:
            for p in fp:
                self.ax.scatter(p[0], p[1], c='gray', s=30)
            if len(fp) >= 2:
                curve = m.bezier_curve(fp)
                self.ax.plot(curve[:,0], curve[:,1], color='gray', lw=2)

        if len(m.comparison_points) >= 2:
            comp_curves = m.connect_fixed_curves([m.bezier_curve(fp) for fp in m.comparison_points])
            for seg in comp_curves:
                self.ax.plot(seg[:,0], seg[:,1], '--', color='gray', lw=1.5)

        # 編集モードによる色分け
        current_color = 'orange' if m.edit_mode else 'r'

        # 現在編集中ポイント
        for p in m.selected_points:
            self.ax.scatter(p[0], p[1], c=current_color, s=40)
        if len(m.selected_points) >= 2:
            curve = m.bezier_curve(np.array(m.selected_points))
            self.ax.plot(curve[:,0], curve[:,1], color=current_color, lw=2)

        # Fix済みポイント
        for fp in m.fixed_points:
            for p in fp:
                self.ax.scatter(p[0], p[1], c=current_color, s=20)
            if len(fp) >= 2:
                curve = m.bezier_curve(fp)
                self.ax.plot(curve[:,0], curve[:,1], color=current_color, lw=2)

        # Fix間の接続
        connections = m.connect_fixed_curves([m.bezier_curve(fp) for fp in m.fixed_points])
        for seg in connections:
            self.ax.plot(seg[:,0], seg[:,1], 'r--', lw=1.5)

        self.fig.canvas.draw_idle()


# ============================================
# Controller: ユーザー操作とイベント処理
# ============================================
class TrajectoryController:
    """
    ユーザーのクリック・ドラッグ・ボタン操作を処理し、
    Model と View の仲介を行うクラス
    """
    def __init__(self, model, view):
        self.model = model
        self.view = view

        # Matplotlibイベント登録
        self.view.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.view.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.view.fig.canvas.mpl_connect('button_release_event', self.on_release)

        # ボタンイベント登録
        self.view.button_output.on_clicked(self.output)
        self.view.button_read.on_clicked(self.read_log)
        self.view.button_comp.on_clicked(self.comparison)
        self.view.button_edit.on_clicked(self.toggle_edit)
        self.view.button_reset.on_clicked(self.reset)
        self.view.button_reset_prev.on_clicked(self.reset_prev)
        self.view.button_all_reset.on_clicked(self.all_reset)
        self.view.button_fix.on_clicked(self.fix)

    # --- マウスクリック ---
    def onclick(self, event):
        m = self.model
        if event.inaxes != self.view.ax:
            return
        if m.edit_mode:
            # 既存ポイントから最も近い点を選択してドラッグ開始
            all_points = [(m.selected_points, i) for i in range(len(m.selected_points))]
            for fp in m.fixed_points:
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
                m.dragging_info = closest
            return

        # 編集モードでない場合は新規ポイントを追加
        m.selected_points.append(np.array([event.xdata, event.ydata]))
        self.view.redraw_all()

    # --- マウスドラッグ ---
    def on_motion(self, event):
        m = self.model
        if not m.edit_mode or m.dragging_info is None:
            return
        if event.inaxes != self.view.ax:
            return
        plist, idx = m.dragging_info
        plist[idx] = np.array([event.xdata, event.ydata])
        self.view.redraw_all()

    # --- マウスリリース ---
    def on_release(self, event):
        self.model.dragging_info = None
        self.view.redraw_all()

    # --- ボタン操作群 ---
    def fix(self, event):
        m = self.model
        if len(m.selected_points) < 2:
            return
        m.fixed_points.append(m.selected_points.copy())
        m.selected_points.clear()
        m.edit_mode = False
        self.view.button_edit.label.set_text('Edit')
        self.view.redraw_all()

    def reset(self, event):
        m = self.model
        if m.is_initial_point_fixed():
            # 初期点がFix点なら、selected_pointsを空に
            m.selected_points.clear()
        else:
            # 初期点が編集中なら、初期点だけ残す
            m.selected_points.clear()
            m.selected_points.append(m.initial_point.copy())
        self.view.redraw_all()

    def all_reset(self, event):
        m = self.model
        m.selected_points.clear()
        m.selected_points.append(m.initial_point.copy())
        m.fixed_points.clear()
        m.comparison_points.clear()
        self.view.redraw_all()

    def reset_prev(self, event):
        m = self.model
        if m.fixed_points:
            m.fixed_points.pop()
        if m.is_initial_point_fixed():
            # 初期点がFix点なら、selected_pointsを空に
            m.selected_points.clear()
        else:
            # 初期点が編集中なら、初期点だけ残す
            m.selected_points.clear()
            m.selected_points.append(m.initial_point.copy())
        self.view.redraw_all()

    def toggle_edit(self, event):
        m = self.model
        m.edit_mode = not m.edit_mode
        self.view.button_edit.label.set_text('Editing' if m.edit_mode else 'Edit')
        self.view.redraw_all()

    def output(self, event):
        self.model.export_trajectory()

    def read_log(self, event):
        m = self.model
        log_path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv")])
        if not log_path:
            return
        df = pd.read_csv(log_path)
        m.selected_points.clear()
        m.fixed_points.clear()
        for sid, group in df.groupby('scene_id'):
            pts = group[['x','y']].values.tolist()
            if sid == 0:
                m.selected_points = [np.array(p) for p in pts]
            else:
                m.fixed_points.append([np.array(p) for p in pts])
        print(f"ログ読み込み: {log_path}")
        self.view.redraw_all()

    def comparison(self, event):
        m = self.model
        comp_path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv")])
        if not comp_path:
            return
        df = pd.read_csv(comp_path)
        m.comparison_points.clear()
        for sid, group in df.groupby('scene_id'):
            pts = [np.array(p) for p in group[['x','y']].values.tolist()]
            m.comparison_points.append(pts)
        print(f"比較ログ読み込み: {comp_path}")
        self.view.redraw_all()


# ============================================
# Main: 起動処理
# ============================================
if __name__ == "__main__":
    # TkinterファイルダイアログでCSV選択
    root = tk.Tk()
    root.withdraw()
    # file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    file_path = "base_circuit_line.csv"
    data = pd.read_csv(file_path)

    # 左右ラインの読み込み
    left_points = data[['left_x', 'left_y']].values
    right_points = data[['right_x', 'right_y']].values

    # MVC初期化
    model = TrajectoryModel(left_points, right_points)
    view = TrajectoryView(model)
    TrajectoryController(model, view)

    plt.show()
