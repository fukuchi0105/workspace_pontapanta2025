import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, Slerp
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 入力CSVファイル
    file_path = "raceline_konada.csv"  # 元データパス
    base_name = file_path.replace(".csv", "")

    # サンプリング間隔
    interval = 0.1  # [m]

    # CSV読み込み
    data = pd.read_csv(file_path)

    # 元の座標
    points = data[['x', 'y', 'z']].values

    # 累積距離計算
    distances = np.zeros(points.shape[0])
    distances[1:] = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))

    # スプライン補間設定（Cubic Spline）
    cs_x = CubicSpline(distances, data['x'])
    cs_y = CubicSpline(distances, data['y'])
    cs_z = CubicSpline(distances, data['z'])

    # intervalでサンプリング
    new_distances = np.arange(0, distances[-1], interval)
    x_new = cs_x(new_distances)
    y_new = cs_y(new_distances)
    z_new = cs_z(new_distances)

    # クオータニオン補間
    quat_orig = data[['x_quat', 'y_quat', 'z_quat', 'w_quat']].values
    rot_orig = R.from_quat(quat_orig)
    slerp_func = Slerp(distances, rot_orig)
    rot_new = slerp_func(new_distances)
    quat_new = rot_new.as_quat()

    # speed列を線形補間
    speed_orig = data['speed'].values
    speed_new = np.interp(new_distances, distances, speed_orig)

    # 結果をCSV保存
    output_df = pd.DataFrame({
        'x': x_new,
        'y': y_new,
        'z': z_new,
        'x_quat': quat_new[:,0],
        'y_quat': quat_new[:,1],
        'z_quat': quat_new[:,2],
        'w_quat': quat_new[:,3],
        'speed': speed_new
    })
    output_df.to_csv(base_name + f"_edit_{interval}m.csv", index=False)

    # プロット確認
    plt.figure(figsize=(8,6))
    plt.plot(data['x'], data['y'], 'o', label='Original Points', markersize=3)
    plt.plot(x_new, y_new, '-', label=f'Smoothed {interval}m Sampling')
    plt.legend()
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title(f'Smoothed Racing Line with {interval}m Interval')
    plt.axis('equal')
    plt.savefig(base_name + f"_edit_{interval}m.png")
    plt.show()
