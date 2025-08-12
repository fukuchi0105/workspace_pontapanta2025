#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read ROS2 bag and estimate steering dynamics:
  (A) FOPDT (closed-loop equivalent): K_cl, tau, delay
  (B) FF-only: product (K*Kff), tau, delay
  (C) FF+P: K, tau, Kff, Kp, delay  (weak identifiability; check correlations)

Assumptions:
- Plant:  tau * dy/dt + y = K * u
- FF-only:           u = Kff * r
- FF+P:              u = Kff * r + Kp * (r - y)
- Delay is modeled as pure input delay on r (discrete step-shift)

Input r is step-only (e.g., 0 -> 0.1 -> -0.1 -> 0.25 -> -0.25 -> 0.5 -> -0.5 -> 0).
"""

import argparse, os, math, yaml, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from scipy.optimize import least_squares

# -------------------- rosbag2 helpers --------------------

def read_storage_id(bag_dir: str) -> str:
    meta = os.path.join(bag_dir, "metadata.yaml")
    if not os.path.isfile(meta):
        # allow the user to pass a db3/0.mcap path; try parent
        p = os.path.dirname(bag_dir)
        meta2 = os.path.join(p, "metadata.yaml")
        if os.path.isfile(meta2):
            meta = meta2
        else:
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
    return {t.name: t.type for t in reader.get_all_topics_and_types()}

def _extract_control_cmd_angle(msg):
    """
    Works for:
      autoware_auto_control_msgs/msg/AckermannControlCommand
      autoware_control_msgs/msg/Control (with .lateral.steering_tire_angle)
    """
    # Common path
    if hasattr(msg, "lateral") and hasattr(msg.lateral, "steering_tire_angle"):
        return float(msg.lateral.steering_tire_angle)
    # Some forks may expose nested fields differently; add fallbacks if needed.
    raise AttributeError("Cannot find lateral.steering_tire_angle in control_cmd message.")

def _extract_steering_report_angle(msg):
    # autoware_auto_vehicle_msgs/msg/SteeringReport
    if hasattr(msg, "steering_tire_angle"):
        return float(msg.steering_tire_angle)
    raise AttributeError("Cannot find steering_tire_angle in steering_status message.")

def extract_series(bag_dir, cmd_topic, status_topic):
    reader, storage_id = open_reader(bag_dir)
    typemap = get_typemap(reader)

    if cmd_topic not in typemap:
        raise RuntimeError(f"Topic {cmd_topic} not in bag. Available: {list(typemap.keys())[:10]} ...")
    if status_topic not in typemap:
        raise RuntimeError(f"Topic {status_topic} not in bag. Available: {list(typemap.keys())[:10]} ...")

    cmd_type = get_message(typemap[cmd_topic])
    status_type = get_message(typemap[status_topic])

    t_cmd, r_cmd = [], []
    t_y, y_meas = [], []

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic == cmd_topic:
            msg = deserialize_message(data, cmd_type)
            ang = _extract_control_cmd_angle(msg)
            t_cmd.append(t_ns * 1e-9)  # ns -> s
            r_cmd.append(ang)
        elif topic == status_topic:
            msg = deserialize_message(data, status_type)
            ang = _extract_steering_report_angle(msg)
            t_y.append(t_ns * 1e-9)
            y_meas.append(ang)

    r = pd.DataFrame({"t": np.array(t_cmd), "r": np.array(r_cmd)}).sort_values("t")
    y = pd.DataFrame({"t": np.array(t_y), "y": np.array(y_meas)}).sort_values("t")
    if len(r) < 5 or len(y) < 5:
        raise RuntimeError("Not enough samples for r or y.")
    return r, y, storage_id

def resample_align(r_df, y_df, dt=None):
    t0 = max(r_df["t"].min(), y_df["t"].min())
    t1 = min(r_df["t"].max(), y_df["t"].max())
    if t1 <= t0:
        raise RuntimeError("No time overlap between r and y.")
    if dt is None:
        # robust median dt estimation
        def med_dt(arr):
            d = np.diff(arr[:min(len(arr)-1, 2000)])
            d = d[np.isfinite(d) & (d > 1e-6)]
            return np.median(d) if len(d) else np.nan
        dt_guess = np.nanmedian([med_dt(r_df["t"].values), med_dt(y_df["t"].values)])
        dt = float(np.clip(np.nan_to_num(dt_guess, nan=0.01), 0.001, 0.05))
    t = np.arange(t0, t1, dt)
    r = np.interp(t, r_df["t"].values, r_df["r"].values)
    y = np.interp(t, y_df["t"].values, y_df["y"].values)
    return t, r, y, dt

# -------------------- step detection & SK method --------------------

def detect_steps(r, t, min_step=0.02, min_hold=20):
    """
    Detect step indices where r changes by >= min_step [rad].
    min_hold: min samples to keep per plateau to avoid chattering.
    Returns list of (k_start, k_end, r0, r1)
    """
    dr = np.diff(r, prepend=r[0])
    # Slope-based detection
    idx = np.where(np.abs(dr) >= min_step)[0]
    if len(idx) == 0:
        return []
    # collapse near-duplicates
    picks = [idx[0]]
    for k in idx[1:]:
        if k - picks[-1] > min_hold:
            picks.append(k)
    segments = []
    prev = 0
    for k in picks + [len(r)-1]:
        if k - prev >= min_hold:
            r0 = r[prev]
            r1 = r[k]
            segments.append((prev, k, r0, r1))
            prev = k
    return segments

def sk_fopdt_single(t, r, y, k0, k1):
    """
    Sundaresan–Krishnaswamy method on a single step from k0->k1
    Returns (K, tau, theta) for y = K * [step response] with delay theta.
    """
    y0 = float(y[0]); yss = float(np.nanmedian(y[int(0.8*len(y)):]))  # final value robust
    r0 = float(r[0]); rss = float(r[-1])
    if abs(rss - r0) < 1e-6 or abs(yss - y0) < 1e-9:
        return None
    # normalized response wrt y
    yn = (y - y0) / (yss - y0 + 1e-12)
    # target levels
    y35, y85 = 0.353, 0.853
    try:
        t35 = np.interp(y35, yn, t)
        t85 = np.interp(y85, yn, t)
    except Exception:
        return None
    tau = 0.67 * (t85 - t35)
    theta = 1.3 * t35 - 0.29 * t85
    # gain K = Δy/Δr
    K = (yss - y0) / (rss - r0)
    if not (np.isfinite(K) and np.isfinite(tau) and np.isfinite(theta)):
        return None
    return K, max(tau, 1e-3), max(theta, 0.0)

def initial_guess_from_steps(t, r, y, steps):
    Ks, taus, thetas = [], [], []
    for (i0, i1, r0, r1) in steps:
        seg = slice(i0, min(len(t), i1 + 1 + int(0.5*(i1-i0))))
        est = sk_fopdt_single(t[seg], r[seg], y[seg], i0, i1)
        if est:
            K, tau, th = est
            Ks.append(K); taus.append(tau); thetas.append(th)
    if len(Ks) == 0:
        # fallback
        return 1.0, 0.2, 0.0
    return float(np.median(Ks)), float(np.median(taus)), float(np.median(thetas))

# -------------------- simulators --------------------

def simulate_fopdt_r_to_y(t, r, Kcl, tau, delay, y0=0.0, rate_limit=None):
    dt = t[1]-t[0]
    y = np.zeros_like(r); y[0] = y0
    d_steps = int(max(0, round(delay/dt)))
    r_d = np.roll(r, d_steps)
    if d_steps > 0:
        r_d[:d_steps] = r_d[d_steps]
    a = max(0.0, 1.0 - dt/tau)
    b = (dt/tau) * Kcl
    for k in range(1, len(r)):
        change_val = a*y[k-1] + b*r_d[k-1] - y[k-1]
        # Apply rate limit if specified
        if rate_limit is not None:
            print(f"change_val={change_val:.6f}, rate_limit={rate_limit:.6f}")
            change_val = np.clip(change_val, -rate_limit*dt, rate_limit*dt)
        y[k] = y[k-1] + change_val
        # y[k] = a*y[k-1] + b*r_d[k-1]
    return y

def simulate_ff_only(t, r, Kt, tau, delay, y0=0.0):
    # tau dy/dt + y = (K*Kff) * r_d
    return simulate_fopdt_r_to_y(t, r, Kt, tau, delay, y0=y0)

def simulate_ff_p(t, r, K, tau, Kff, Kp, delay, y0=0.0):
    dt = t[1]-t[0]
    y = np.zeros_like(r); y[0] = y0
    d_steps = int(max(0, round(delay/dt)))
    r_d = np.roll(r, d_steps)
    if d_steps > 0:
        r_d[:d_steps] = r_d[d_steps]
    for k in range(1, len(r)):
        u = Kff * r_d[k-1] + Kp * (r_d[k-1] - y[k-1])
        y[k] = y[k-1] + (dt/tau) * (K * u - y[k-1])
    return y

# -------------------- fitting --------------------

def estimate_delay_xcorr(r, y, dt, max_delay=1.0):
    r0 = r - np.mean(r); y0 = y - np.mean(y)
    max_lag = int(max_delay / dt)
    best_lag, best_val = 0, -1e18
    for lag in range(-max_lag, max_lag+1):
        if lag >= 0:
            val = np.dot(y0[lag:], r0[:len(r0)-lag])
        else:
            val = np.dot(y0[:len(y0)+lag], r0[-lag:])
        if val > best_val:
            best_val, best_lag = val, lag
    return best_lag * dt

def fit_fopdt_closedloop(t, r, y, p0):
    # params: [K_cl, tau, delay]
    def residual(p):
        Kcl, tau, delay = p
        Kcl = max(1e-6, Kcl); tau = max(1e-3, tau); delay = max(0.0, delay)
        yhat = simulate_fopdt_r_to_y(t, r, Kcl, tau, delay, y0=y[0])
        return yhat - y
    bounds = ([0.0, 1e-3, 0.0], [100.0, 10.0, 2.0])
    return least_squares(residual, p0, bounds=bounds)

def fit_ff_only(t, r, y, p0):
    # params: [K_times_Kff, tau, delay]
    def residual(p):
        Kt, tau, delay = p
        Kt = max(1e-6, Kt); tau = max(1e-3, tau); delay = max(0.0, delay)
        yhat = simulate_ff_only(t, r, Kt, tau, delay, y0=y[0])
        return yhat - y
    bounds = ([0.0, 1e-3, 0.0], [100.0, 10.0, 2.0])
    return least_squares(residual, p0, bounds=bounds)

def fit_ff_p(t, r, y, p0):
    # params: [K, tau, Kff, Kp, delay]
    def residual(p):
        K, tau, Kff, Kp, delay = p
        K = max(1e-6, K); tau = max(1e-3, tau); Kff = max(0.0, Kff); Kp = max(0.0, Kp); delay = max(0.0, delay)
        yhat = simulate_ff_p(t, r, K, tau, Kff, Kp, delay, y0=y[0])
        return yhat - y
    lb = [0.01, 1e-3, 0.0, 0.0, 0.0]
    ub = [100.0, 10.0, 10.0, 200.0, 2.0]
    return least_squares(residual, p0, bounds=(lb, ub))

def summarize(name, res):
    J = res.jac
    dof = max(1, len(res.fun) - len(res.x))
    sigma2 = float(np.sum(res.fun**2) / dof)
    try:
        cov = np.linalg.inv(J.T @ J) * sigma2
        se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
        corr = cov / (se[:,None]*se[None,:] + 1e-12)
        offdiag_max = float(np.max(np.abs(corr - np.eye(len(res.x)))))
    except Exception:
        cov = None; se = np.full_like(res.x, np.nan); offdiag_max = np.nan
    print(f"\n[{name}]")
    for n, v, s in zip(["K_cl/Kt or K","tau","delay","Kff","Kp"][:len(res.x)], res.x, se):
        print(f"  {n:>6s} = {v: .6f}  ±{s if np.isfinite(s) else float('nan'):.6f}")
    print(f"  RSS={np.sum(res.fun**2):.6e}, RMSE={math.sqrt(np.mean(res.fun**2)):.6e}")
    if np.isfinite(offdiag_max):
        print(f"  |corr|_max(off-diag) = {offdiag_max:.3f}")
    return res.x

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="rosbag2 directory (contains metadata.yaml)")
    ap.add_argument("--cmd-topic", default="/control/command/control_cmd")
    ap.add_argument("--status-topic", default="/vehicle/status/steering_status")
    ap.add_argument("--dt", type=float, default=None, help="resample step [s] (auto if None)")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--step-thresh", type=float, default=0.02, help="step detect threshold [rad]")
    args = ap.parse_args()

    r_df, y_df, storage_id = extract_series(args.bag, args.cmd_topic, args.status_topic)
    print(f"Opened bag storage='{storage_id}': r={len(r_df)}, y={len(y_df)}")

    t, r, y, dt = resample_align(r_df, y_df, args.dt)
    print(f"Aligned: N={len(t)}, dt={dt*1000:.1f} ms, duration={t[-1]-t[0]:.2f} s")

    # Step-based initial guess
    steps = detect_steps(r, t, min_step=args.step_thresh)
    print(f"Detected steps: {len(steps)}")
    K0, tau0, th0 = initial_guess_from_steps(t, r, y, steps)
    # Further refine delay by xcorr
    th0 = max(0.0, estimate_delay_xcorr(r, y, dt, max_delay=1.0))
    print(f"Initial guess ~ K={K0:.3f}, tau={tau0:.3f}, delay={th0:.3f}")

    # (A) Closed-loop equivalent FOPDT: y = (K_cl/(tau s + 1)) r_d
    resA = fit_fopdt_closedloop(t, r, y, p0=[max(1e-3, K0), max(1e-3, tau0), max(0.0, th0)])
    pa = summarize("A) FOPDT (closed-loop)", resA)  # [K_cl, tau, delay]

    # (B) FF-only equivalent: tau dy/dt + y = (K*Kff) r_d
    resB = fit_ff_only(t, r, y, p0=[max(1e-3, pa[0]), max(1e-3, pa[1]), max(0.0, pa[2])])
    pb = summarize("B) FF-only  (reports product K*Kff)", resB)

    # (C) FF+P: tau dy/dt + (1 + K*Kp) y = K (Kff + Kp) r_d
    # Start from closed-loop equivalent; pick K~1 as neutral init
    resC = fit_ff_p(t, r, y, p0=[1.0, max(1e-3, pa[1]), max(0.0, pa[0]), 0.5, max(0.0, pa[2])])
    pc = summarize("C) FF+P (weak identifiability)", resC)

    if args.plot:
        yA = simulate_fopdt_r_to_y(t, r, *resA.x, y0=y[0])
        yB = simulate_ff_only(t, r, *resB.x, y0=y[0])
        yC = simulate_ff_p(t, r, *resC.x, y0=y[0])
        ymyFit = simulate_fopdt_r_to_y(t, r, 0.7, 0.1, 0.2, y0=y[0], rate_limit=50*math.pi/180)

        plt.figure()
        plt.title("Steering tracking: ref vs measured vs fits")
        plt.plot(t, r, label="ref r [rad]", linewidth=1.0)
        plt.plot(t, y, label="meas y [rad]", linewidth=1.0)
        # plt.plot(t, yA, "--", label="fit A: FOPDT")
        # plt.plot(t, yB, "--", label="fit B: FF-only")
        # plt.plot(t, yC, "--", label="fit C: FF+P")
        plt.plot(t, ymyFit, "--", label="fit myfit")
        plt.xlabel("time [s]"); plt.ylabel("steering [rad]"); plt.grid(True); plt.legend()

        plt.figure()
        plt.title("Residuals")
        plt.plot(t, yA - y, label="A residual")
        plt.plot(t, yB - y, label="B residual")
        plt.plot(t, yC - y, label="C residual")
        plt.xlabel("time [s]"); plt.ylabel("error [rad]"); plt.grid(True); plt.legend()

        # quick visualization of detected steps
        # if len(steps):
        #     for (i0, i1, r0, r1) in steps:
        #         plt.figure()
        #         sl = slice(max(0, i0-20), min(len(t), i1+int(0.6*(i1-i0)+50)))
        #         plt.title(f"Step segment around k={i0} (r: {r0:.2f}->{r1:.2f})")
        #         plt.plot(t[sl], r[sl], label="r")
        #         plt.plot(t[sl], y[sl], label="y")
        #         plt.grid(True); plt.legend()

        plt.show()

    print("\nNotes:")
    print("- (B) は K と Kff が積でしか現れないため **不可識別**（分離不能）です。報告値は K*Kff。")
    print("- (C) はパラメータ間の相関が高くなりやすいので、|corr|_max を確認して下さい（0.9↑は要注意）。")
    print("- ステップ検出の閾値は --step-thresh で調整できます（デフォルト 0.02 rad）。")

if __name__ == "__main__":
    main()
