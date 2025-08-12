#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimate steering actuator dynamics from ROS 2 bag:
- Closed-loop equivalent 1st-order (K_cl, tau, delay)
- FF-only equivalent (K*Kff, tau, delay)  -> product reported
- FF+P structure (K, tau, Kff, Kp, delay) -> weak identifiability; report covariances
"""

import argparse, os, sys, math, yaml, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from scipy.optimize import least_squares

def read_storage_id(bag_dir: str) -> str:
    meta = os.path.join(bag_dir, "metadata.yaml")
    if not os.path.isfile(meta):
        # try parent (some tools pass the db3 file path)
        parent = os.path.dirname(bag_dir)
        meta2 = os.path.join(parent, "metadata.yaml")
        if os.path.isfile(meta2):
            meta = meta2
        else:
            # default fallback
            return "sqlite3"
    with open(meta, "r") as f:
        data = yaml.safe_load(f)
    return data.get("rosbag2_bagfile_information", {}).get("storage_identifier", "sqlite3")

def open_reader(bag_dir: str):
    storage_id = read_storage_id(bag_dir)
    reader = rosbag2_py.SequentialReader()
    storage_opts = rosbag2_py.StorageOptions(uri=bag_dir, storage_id=storage_id)
    converter_opts = rosbag2_py.ConverterOptions(input_serialization_format="cdr",
                                                 output_serialization_format="cdr")
    reader.open(storage_opts, converter_opts)
    return reader, storage_id

def get_typemap(reader):
    typemap = {}
    for t in reader.get_all_topics_and_types():
        typemap[t.name] = t.type
    return typemap

def extract_series(bag_dir, cmd_topic, status_topic):
    reader, storage_id = open_reader(bag_dir)
    typemap = get_typemap(reader)

    if cmd_topic not in typemap:
        raise RuntimeError(f"Topic {cmd_topic} not in bag. Available: {list(typemap.keys())}")
    if status_topic not in typemap:
        raise RuntimeError(f"Topic {status_topic} not in bag. Available: {list(typemap.keys())}")

    cmd_msg_type = get_message(typemap[cmd_topic])
    status_msg_type = get_message(typemap[status_topic])

    t_cmd, r_cmd = [], []
    t_y, y_meas = [], []

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic == cmd_topic:
            msg = deserialize_message(data, cmd_msg_type)
            # AckermannControlCommand.lateral.steering_tire_angle  [rad]
            try:
                ang = float(msg.lateral.steering_tire_angle)
            except Exception:
                # some forks may nest differently; fail loudly
                raise
            t_cmd.append(t * 1e-9)  # ns -> s
            r_cmd.append(ang)
        elif topic == status_topic:
            msg = deserialize_message(data, status_msg_type)
            # SteeringReport.steering_tire_angle [rad]
            ang = float(msg.steering_tire_angle)
            t_y.append(t * 1e-9)
            y_meas.append(ang)

    r = pd.DataFrame({"t": np.array(t_cmd), "r": np.array(r_cmd)}).sort_values("t").dropna()
    y = pd.DataFrame({"t": np.array(t_y), "y": np.array(y_meas)}).sort_values("t").dropna()
    if len(r) < 5 or len(y) < 5:
        raise RuntimeError("Not enough samples for r or y.")
    return r, y, storage_id

def resample_align(r_df, y_df, dt=None):
    t0 = max(r_df["t"].min(), y_df["t"].min())
    t1 = min(r_df["t"].max(), y_df["t"].max())
    if t1 <= t0:
        raise RuntimeError("No time overlap between r and y.")
    if dt is None:
        # pick median dt among both
        dr = np.median(np.diff(r_df["t"].values[:min(len(r_df)-1, 1000)]))
        dy = np.median(np.diff(y_df["t"].values[:min(len(y_df)-1, 1000)]))
        dt = np.nanmedian([dr, dy])
        if not np.isfinite(dt) or dt <= 0:
            dt = 0.01
        dt = float(max(min(dt, 0.05), 0.001))  # clamp 1–50 ms
    t = np.arange(t0, t1, dt)
    r = np.interp(t, r_df["t"].values, r_df["r"].values)
    y = np.interp(t, y_df["t"].values, y_df["y"].values)
    return t, r, y, dt

def estimate_delay_xcorr(r, y, dt, max_delay=1.0):
    # zero-mean, window to finite lags
    r0 = r - np.mean(r)
    y0 = y - np.mean(y)
    max_lag = int(max_delay / dt)
    # use FFT convolution
    corr = np.fft.ifft(np.fft.fft(y0, n=2*len(y0)) * np.conj(np.fft.fft(r0, n=2*len(r0)))).real
    corr = corr[:len(y0)]
    lags = np.arange(len(corr))
    # translate to positive/negative around zero by circular shift
    # for simplicity: compute brute force local around +/- max_lag
    best_lag = 0
    best_val = -1e18
    for lag in range(-max_lag, max_lag+1):
        if lag >= 0:
            val = np.dot(y0[lag:], r0[:len(r0)-lag])
        else:
            val = np.dot(y0[:len(y0)+lag], r0[-lag:])
        if val > best_val:
            best_val = val
            best_lag = lag
    return best_lag * dt

def simulate_first_order_closedloop(t, r, Kcl, tau, delay, y0=0.0):
    dt = t[1]-t[0]
    y = np.zeros_like(r)
    y[0] = y0
    d_steps = int(max(0, round(delay/dt)))
    r_d = np.roll(r, d_steps)
    if d_steps > 0:
        r_d[:d_steps] = r_d[d_steps]  # hold-head
    a = 1.0 - dt/tau
    b = (dt/tau)*Kcl
    for k in range(1, len(r)):
        y[k] = a*y[k-1] + b*r_d[k-1]
    return y

def simulate_ff_only(t, r, K_times_Kff, tau, delay, y0=0.0):
    # tau dy/dt + y = (K*Kff) r_d
    return simulate_first_order_closedloop(t, r, K_times_Kff, tau, delay, y0=y0)

def simulate_ff_p(t, r, K, tau, Kff, Kp, delay, y0=0.0):
    dt = t[1]-t[0]
    y = np.zeros_like(r)
    y[0] = y0
    d_steps = int(max(0, round(delay/dt)))
    r_d = np.roll(r, d_steps)
    if d_steps > 0:
        r_d[:d_steps] = r_d[d_steps]
    for k in range(1, len(r)):
        u = Kff * r_d[k-1] + Kp * (r_d[k-1] - y[k-1])
        y[k] = y[k-1] + (dt/tau) * (K * u - y[k-1])
    return y

def fit_closedloop_first_order(t, r, y):
    # initial guesses
    dt = t[1]-t[0]
    delay0 = max(0.0, estimate_delay_xcorr(r, y, dt, max_delay=1.0))
    K0 = (np.std(y)+1e-6)/(np.std(r)+1e-6)
    tau0 = 0.2
    p0 = np.array([K0, tau0, delay0])

    def residual(p):
        Kcl, tau, delay = p
        Kcl = max(1e-6, Kcl)
        tau = max(1e-3, tau)
        delay = max(0.0, delay)
        yhat = simulate_first_order_closedloop(t, r, Kcl, tau, delay, y0=y[0])
        return yhat - y

    res = least_squares(residual, p0, bounds=([0.0, 1e-3, 0.0],[100.0, 10.0, 2.0]))
    return res

def fit_ff_only(t, r, y, delay_hint):
    # parameters: [K_times_Kff, tau, delay]
    p0 = np.array([(np.std(y)+1e-6)/(np.std(r)+1e-6), 0.2, max(0.0, delay_hint)])
    def residual(p):
        Kt, tau, delay = p
        Kt = max(1e-6, Kt); tau = max(1e-3, tau); delay = max(0.0, delay)
        yhat = simulate_ff_only(t, r, Kt, tau, delay, y0=y[0])
        return yhat - y
    res = least_squares(residual, p0, bounds=([0.0, 1e-3, 0.0],[100.0, 10.0, 2.0]))
    return res

def fit_ff_p(t, r, y, delay_hint):
    # parameters: [K, tau, Kff, Kp, delay]
    p0 = np.array([1.0, 0.2, 1.0, 1.0, max(0.0, delay_hint)])
    lb = [0.01, 1e-3, 0.0, 0.0, 0.0]
    ub = [100.0, 10.0, 10.0, 100.0, 2.0]
    def residual(p):
        K, tau, Kff, Kp, delay = p
        K = max(1e-6, K); tau = max(1e-3, tau); Kff = max(0.0, Kff); Kp = max(0.0, Kp); delay = max(0.0, delay)
        yhat = simulate_ff_p(t, r, K, tau, Kff, Kp, delay, y0=y[0])
        return yhat - y
    res = least_squares(residual, p0, bounds=(lb, ub))
    return res

def summarize(res, name, param_names):
    J = res.jac
    dof = max(1, len(res.fun) - len(res.x))
    sigma2 = np.sum(res.fun**2) / dof
    try:
        cov = np.linalg.inv(J.T @ J) * sigma2
        se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
        corr = cov / (se[:,None]*se[None,:] + 1e-12)
    except np.linalg.LinAlgError:
        cov = None; se = np.full_like(res.x, np.nan); corr = None
    print(f"\n[{name}] params:")
    for i, n in enumerate(param_names):
        print(f"  {n:>10s} = {res.x[i]: .6f}  ±{se[i]:.6f}")
    print(f"  RSS = {np.sum(res.fun**2):.6f},  RMSE = {math.sqrt(np.mean(res.fun**2)):.6f}")
    if corr is not None:
        print("  |corr| max (off-diagonal): {:.3f}".format(np.max(np.abs(corr - np.eye(len(res.x))))))
    return cov

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="rosbag2 directory (contains metadata.yaml)")
    ap.add_argument("--cmd-topic", default="/control/command/control_cmd")
    ap.add_argument("--status-topic", default="/vehicle/status/steering_status")
    ap.add_argument("--dt", type=float, default=None, help="resample step [s] (auto if None)")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    r_df, y_df, storage_id = extract_series(args.bag, args.cmd_topic, args.status_topic)
    print(f"Opened bag with storage_id='{storage_id}'. Samples: r={len(r_df)}, y={len(y_df)}")
    t, r, y, dt = resample_align(r_df, y_df, args.dt)
    print(f"Aligned length={len(t)}, dt={dt*1000:.1f} ms, duration={t[-1]-t[0]:.2f} s")

    # Closed-loop 1st-order fit
    res_cl = fit_closedloop_first_order(t, r, y)
    cov_cl = summarize(res_cl, "Closed-loop 1st-order", ["K_cl", "tau", "delay"])

    # FF-only equivalent fit (reports product K*Kff)
    res_ff = fit_ff_only(t, r, y, delay_hint=res_cl.x[2])
    cov_ff = summarize(res_ff, "FF-only (K*Kff product)", ["K_times_Kff", "tau", "delay"])

    # FF+P fit (weak identifiability; check correlations)
    res_ffp = fit_ff_p(t, r, y, delay_hint=res_cl.x[2])
    cov_ffp = summarize(res_ffp, "FF+P", ["K", "tau", "Kff", "Kp", "delay"])

    if args.plot:
        yhat_cl  = simulate_first_order_closedloop(t, r, *res_cl.x, y0=y[0])
        yhat_ff  = simulate_ff_only(t, r, *res_ff.x, y0=y[0])
        yhat_ffp = simulate_ff_p(t, r, *res_ffp.x, y0=y[0])

        fig1 = plt.figure()
        plt.title("Steering angle tracking")
        plt.plot(t, r, label="ref r [rad]", linewidth=1.0)
        plt.plot(t, y, label="meas y [rad]", linewidth=1.0)
        plt.plot(t, yhat_cl,  label="fit: 1st-order CL", linestyle="--")
        plt.plot(t, yhat_ff,  label="fit: FF-only", linestyle="--")
        plt.plot(t, yhat_ffp, label="fit: FF+P", linestyle="--")
        plt.xlabel("time [s]"); plt.ylabel("steering [rad]"); plt.legend(); plt.grid(True)

        fig2 = plt.figure()
        e_cl  = yhat_cl - y
        e_ff  = yhat_ff - y
        e_ffp = yhat_ffp - y
        plt.title("Residuals")
        plt.plot(t, e_cl,  label="CL residual")
        plt.plot(t, e_ff,  label="FF-only residual")
        plt.plot(t, e_ffp, label="FF+P residual")
        plt.xlabel("time [s]"); plt.ylabel("error [rad]"); plt.legend(); plt.grid(True)
        plt.show()

    # Practical notes
    print("\nNotes:")
    print("- モデル 2（FFのみ）は K と Kff が積でしか現れないため**分離不能**です。報告値は K*Kff です。")
    print("- モデル 3（FF+P）はパラメータ相関が高くなりがちです。出力された相関(|corr| max)が0.9近い場合、識別は弱いと判断してください。")
    print("- 可能ならステップ/PRBS等で励振し、静止と低速走行で再測定すると安定します。")

if __name__ == "__main__":
    main()
