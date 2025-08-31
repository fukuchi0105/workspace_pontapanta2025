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
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline

##############################################
# Model: データ保持と数値計算ロジック
##############################################
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
        self.xq = 0.0 
        self.yq = 0.0
        self.zq = 0.890874
        self.wq = 0.454251

        # クオータニオン座標系モード ('right_hand' or 'vehicle')
        self.coord_mode = 'right_hand'

        # 出力間隔 (m)
        self.div_export = 0.5
        
        # ラップタイム保存
        self.lap_time_result = None

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

    # --- 車速計算用 ---
    def get_speed(self):
        """
        将来可変にするための車速計算関数
        今回は固定で 11 を返す
        """
        return 11.0

    # --- 姿勢計算用 ---
    def calculate_quaternion(self, p1, p2):
        """
        2点間から進行方向を計算し、クオータニオンを生成
        p1: 現在の点 [x, y]
        p2: 次の点 [x, y]

        将来的に3D対応するため yaw/pitch/roll を想定。
        現状は yaw のみ計算し、pitch=0, roll=0。

        coord_mode により右手系 or 車両座標系を切り替える
        """
        
        pitch = 0
        roll = 0

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        yaw = math.atan2(dy, dx)

        if self.coord_mode == 'right_hand':
            # 通常の右手系 (Z-up)
            r = R.from_euler('zyx', [yaw, pitch, roll])
        else:
            # 車両座標系 (X-forward, Y-left, Z-up)
            # 車両座標系の場合は yaw の符号や軸変換が異なる
            r = R.from_euler('zyx', [-yaw, pitch, roll])

        xq, yq, zq, wq = r.as_quat()
        return xq, yq, zq, wq

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

    # --- 生成された軌道を等間隔に再サンプル ---
    def resample_points(self, points):
        """
        入力された軌跡(points)を div_export 間隔で再サンプリング
        CubicSplineを使って平滑化した補間点を生成
        """
        if len(points) < 2:
            return points

        # 距離計算
        dist = np.zeros(len(points))
        for i in range(1, len(points)):
            dist[i] = dist[i-1] + np.linalg.norm(points[i] - points[i-1])

        # 新しいサンプリング距離
        new_dist = np.arange(0, dist[-1], self.div_export)

        # CubicSplineで補間
        cs_x = CubicSpline(dist, points[:, 0])
        cs_y = CubicSpline(dist, points[:, 1])

        new_points = np.column_stack((cs_x(new_dist), cs_y(new_dist)))
        return new_points

    # --- CSV出力 ---
    def export_trajectory(self):
        """
        現在の編集結果を2種類のCSVで保存
        1. _Trajectory.csv: ベジェ曲線で補間した最終ライン
        2. _LogPoints.csv: 元の編集点（scene_id付き）
        """
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")

        # --- ベジェ曲線を生成 ---
        all_points = []
        for fp in self.fixed_points:
            all_points.extend(self.bezier_curve(fp))
        if len(self.selected_points) >= 2:
            all_points.extend(self.bezier_curve(np.array(self.selected_points)))
        if not all_points:
            print("No Output Data")
            return None

        all_points = np.array(all_points)

        # 0.5m間隔で再サンプリング
        all_points = self.resample_points(all_points)

        # --- Trajectory 出力 ---
        traj_path = f"{timestamp}_Trajectory.csv"
        with open(traj_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x','y','z','x_quat','y_quat','z_quat','w_quat','speed'])

            for i, p in enumerate(all_points):
                if i < len(all_points)-1:
                    xq, yq, zq, wq = self.calculate_quaternion(p, all_points[i+1])
                else:
                    xq, yq, zq, wq = self.calculate_quaternion(all_points[i-1], p)

                # 初期点のクオータニオンは固定値
                if i == 0:
                    xq, yq, zq, wq = self.xq, self.yq, self.zq, self.wq

                writer.writerow([p[0], p[1], 0.0, xq, yq, zq, wq, self.get_speed()])

        print(f"Trajectory Output: {traj_path}")

        # --- LogPoints 出力 ---
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

    # --- 曲率半径の計算 ---
    def CalcCurvatureRad(self, x, y, max_rad=1000):
        """
        2D座標列から曲率半径を計算
        :param x: X座標配列
        :param y: Y座標配列
        :param max_rad: 最大半径（限界を設定）
        """
        curvature_rad = []
        for i in range(1, len(x)-1):
            p1 = np.array([x[i-1], y[i-1]])
            p2 = np.array([x[i], y[i]])
            p3 = np.array([x[i+1], y[i+1]])

            a = np.linalg.norm(p2 - p1)
            b = np.linalg.norm(p3 - p2)
            c = np.linalg.norm(p3 - p1)

            if a * b * c == 0:
                curvature_rad.append(max_rad)
                continue

            s = (a+b+c)/2
            area = max(s*(s-a)*(s-b)*(s-c), 0)
            if area == 0:
                curvature_rad.append(max_rad)
                continue

            radius = (a*b*c)/(4*np.sqrt(area))
            curvature_rad.append(min(radius, max_rad))

        # 最初と最後を補完
        curvature_rad.insert(0, curvature_rad[0])
        curvature_rad.append(curvature_rad[-1])
        return np.array(curvature_rad)
    
    # --- サーキットシミュレーション ---
    def simulate_lap_time(self):
        """
        サーキット走行タイムを計算するメイン関数
        """
        # パラメータ初期値
        self.vehicle_mass = 160  # 車両重量 [kg]
        self.max_radius_virtual = 1000  # 曲率半径の最大値 [m]
        self.max_horizontal_G = 0.9  # 横Gの上限 [G]
        self.max_accel_G = 3.2 / 9.8   # 最大加速度 [G]
        self.max_brake_G = 1   # 最大ブレーキG [G]
        self.CdA = 0.43  # 空気抵抗係数×前面投影面積の積 [m^2]
        self.air_density = 1.225 # 空気密度 [kg/m-3]
        self.u_r = 0.016 # 転がり抵抗係数 [kgf/kgf]
        # self.power = 160 #[kW]
        # self.power = 14.2 #[kW]
        self.power = 5 #モーターの最大出力 [kW]        
        self.max_speed_finish = 35 #最高速度 [kph]
        ini_vel = 30.0  # 初期速度 [kph]

        # 空力抵抗の係数を[kph]ベースで計算し、重力加速度で正規化してkgf単位に換算 [kgf/kph^2]
        self.u_aA = 1 / 2 * self.CdA * self.air_density / 3.6**2 / 9.8    

        div = self.div_export

        # 軌跡を取得して再サンプリング
        all_points = []
        for fp in self.fixed_points:
            all_points.extend(self.bezier_curve(fp))
        if len(self.selected_points) >= 2:
            all_points.extend(self.bezier_curve(np.array(self.selected_points)))
        if not all_points:
            return None

        all_points = np.array(all_points)
        all_points = self.resample_points(all_points)

        line_x = all_points[:,0]
        line_y = all_points[:,1]

        # 曲率半径計算
        self.curvature_rad = self.CalcCurvatureRad(line_x, line_y, max_rad=self.max_radius_virtual)

        # 旋回限界速度
        self.turning_limit_vel = []
        for r in self.curvature_rad:
            self.turning_limit_vel.append(np.sqrt(self.max_horizontal_G**2 * 9.8 * r) * 3.6)

        # 空力減速度・制動速度 (逆方向計算)
        self.air_resis = []
        self.brake_speed = [self.max_speed_finish]
        for i in range(len(self.curvature_rad)):
            self.air_resis.insert(0, self.u_aA * (self.brake_speed[0])**2 / self.vehicle_mass)
            self.brake_speed.insert(0, min([self.turning_limit_vel[-1 * i - 1],
                                            np.sqrt(2 * (self.max_brake_G + self.air_resis[0]) * 9.8 * div + (self.brake_speed[0] / 3.6)**2) * 3.6]))
        self.brake_speed.pop(-1)

        # 車速と加速度
        self.car_vel = [ini_vel]
        self.car_acc = [1]
        self.full_throttle_flag = [1]
        for i in range(len(self.curvature_rad)-1):
            self.car_acc.append(min([self.max_accel_G,
                                     self.power * 1000 / self.vehicle_mass / max(self.car_vel[-1] / 3.6,0.1) / 9.8 -
                                     (self.u_r + self.u_aA / self.vehicle_mass * (self.car_vel[-1]**2))]))
            compute_vel = min([np.sqrt(2 * self.car_acc[-1] * 9.8 * div + (self.car_vel[-1] / 3.6)**2) * 3.6, self.brake_speed[i]])
            if compute_vel > self.max_speed_finish:
                compute_vel = self.max_speed_finish
            self.car_vel.append(compute_vel)
            
            if self.car_vel[-1] >= self.brake_speed[i]:
                self.full_throttle_flag.append(0)
            else:
                self.full_throttle_flag.append(1)

        # 走行時間積分
        self.time = [0]
        for i in range(1, len(self.car_vel)):
            self.time.append(self.time[-1] + div / ((self.car_vel[i] + self.car_vel[i-1]) / 2 / 3.6))

        # 結果を保持して Time Output で使う
        self.line_x = line_x
        self.line_y = line_y
        return self.time[-1]

##############################################
# View: 描画・UI（Matplotlibベース）
##############################################
class TrajectoryView:
    """
    Matplotlibを使ってデータを可視化し、
    ボタンや描画のUIを担当するクラス
    """
    def __init__(self, model):
        self.model = model
        self.fig, self.ax = plt.subplots()

        # グラフの位置を左に寄せて右側にスペース確保
        self.fig.subplots_adjust(left=0.05, right=0.78, top=0.95, bottom=0.05)

        self.ax.set_aspect('equal', adjustable='datalim')
        # self.ax.set_title('Trajectory Generator')
        # self.ax.set_xlabel("x-axis") 
        # self.ax.set_ylabel("y-axis") 

        # ボタン生成
        self.create_buttons()
        self.redraw_all()

    # --- ボタンUI生成 ---
    def create_buttons(self):

        # 均等配置のため基準値設定
        base_y = 0.9
        step = 0.05

        # Trajectory タイトルとボタン
        self.fig.add_axes([0.82, base_y, 0.15, 0.03]).set_axis_off()
        self.fig.text(0.82, base_y+0.02, "~Trajectory~", fontsize=12, ha='left', fontweight='bold')
        self.button_fix = Button(plt.axes([0.82, base_y-1*step, 0.15, 0.04]), 'Fix')
        self.button_edit = Button(plt.axes([0.82, base_y-2*step, 0.15, 0.04]), 'Edit')
        self.button_reset = Button(plt.axes([0.82, base_y-3*step, 0.15, 0.04]), 'Reset')
        self.button_reset_prev = Button(plt.axes([0.82, base_y-4*step, 0.15, 0.04]), 'Prev Reset')
        self.button_all_reset = Button(plt.axes([0.82, base_y-5*step, 0.15, 0.04]), 'All Reset')

        # Data Reference タイトルとボタン
        ref_base = base_y-7*step
        self.fig.add_axes([0.82, ref_base, 0.15, 0.03]).set_axis_off()
        self.fig.text(0.82, ref_base+0.02, "~Data Reference~", fontsize=12, ha='left', fontweight='bold')
        self.button_read = Button(plt.axes([0.82, ref_base-1*step, 0.15, 0.04]), 'Read Log')
        self.button_comp = Button(plt.axes([0.82, ref_base-2*step, 0.15, 0.04]), 'Comparison')
        self.button_output = Button(plt.axes([0.82, ref_base-3*step, 0.15, 0.04]), 'Traj Output')
        self.button_save_fig = Button(plt.axes([0.82, ref_base-4*step, 0.15, 0.04]), 'Save Fig')

        # Simulation
        sim_base = ref_base-6*step
        self.fig.text(0.82, sim_base+0.02, "~Simulation~", fontsize=12, ha='left', fontweight='bold')
        self.button_calc_time = Button(plt.axes([0.82, sim_base-1*step, 0.15, 0.04]), 'Calc Time')
        self.button_time_output = Button(plt.axes([0.82, sim_base-2*step, 0.15, 0.04]), 'Time Output')
        self.button_time_output.ax.set_facecolor('lightgray')

    # --- 再描画 ---
    def redraw_all(self):
        """
        現在のモデルデータをすべて描画し直す
        """
        m = self.model
        self.ax.clear()
        self.ax.set_xlabel("x-axis")
        self.ax.set_ylabel("y-axis")
        # self.ax.grid(True)

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

        # ラップタイム凡例を再描画
        if m.lap_time_result is not None:
            self.ax.legend([f"Circuit Time = {m.lap_time_result:.3f} [s]"], loc='upper right', frameon=True)

        self.fig.canvas.draw_idle()


##############################################
# Controller: ユーザー操作とイベント処理
##############################################
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
        self.view.button_calc_time.on_clicked(self.calc_time)
        self.view.button_time_output.on_clicked(self.time_output)        
        self.view.button_save_fig.on_clicked(self.save_fig) 

        # Time Output 有効化フラグ
        self.time_data_ready = False
        
        # タイムスタンプを保持    
        self.lap_timestamp = None  

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

    # --- リセットボタン内容 ---
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

    # --- 全リセットボタン内容 ---
    def all_reset(self, event):
        m = self.model
        m.selected_points.clear()
        m.selected_points.append(m.initial_point.copy())
        m.fixed_points.clear()
        m.comparison_points.clear()
        self.view.redraw_all()

    # --- 前回ポイントのリセットボタン内容 ---
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

    # --- エディットモードの切り替え表示 ---
    def toggle_edit(self, event):
        m = self.model
        m.edit_mode = not m.edit_mode
        self.view.button_edit.label.set_text('Editing' if m.edit_mode else 'Edit')
        self.view.redraw_all()

    # --- 出力ボタンの内容 ---
    def output(self, event):
        self.model.export_trajectory()

    # --- ログ読み込みボタンの内容 ---
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
        print(f"Read Log : {log_path}")
        self.view.redraw_all()

    # --- 比較ボタンの内容 ---
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
        print(f"Read Comparison Data : {comp_path}")
        self.view.redraw_all()

    # --- シミュレーション計算の内容 ---
    def calc_time(self, event):
        lap_time = self.model.simulate_lap_time()
        print(f"Simulation Lap Time : {lap_time:.3f} [s]")
        
        if lap_time:
            # タイムスタンプ生成
            self.lap_timestamp = datetime.now().strftime("%Y%m%d_%H%M")

            # 凡例更新
            self.model.lap_time_result = lap_time  # ラップタイムを保存
            self.view.redraw_all()                 # 再描画時に凡例を反映
            
            # Time Output 有効化
            self.time_data_ready = True
            self.view.button_time_output.ax.set_facecolor('white')
            self.view.fig.canvas.draw_idle()

    # --- 画面保存ボタンの内容 ---        
    def save_fig(self, event):
        """
        現在のFigureを保存
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"{timestamp}_Fig.png"
        self.view.fig.savefig(filename, dpi=300)
        print(f"Figure saved: {filename}")    
    
    # --- シミュレーション結果の詳細出力 ---        
    def time_output(self, event):
        if not self.time_data_ready:
            print("Please run Calc Time first.")
            return

        timestamp = self.lap_timestamp
        output_path = f"{timestamp}_TimeSimulation.csv"
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # 出力
            writer.writerow(['time[s]','x','y','speed[kph]','brake_limit[kph]','turning_limit[kph]','full_throttle_fkag'])
            for i in range(len(self.model.car_vel)):
                writer.writerow([
                    self.model.time[i],
                    self.model.line_x[i],
                    self.model.line_y[i],
                    self.model.car_vel[i],
                    self.model.brake_speed[i],
                    self.model.turning_limit_vel[i],
                    self.model.full_throttle_flag[i]
                ])
        print(f"Time Simulation Output: {output_path}")

##############################################
# Main: 起動処理
##############################################
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
