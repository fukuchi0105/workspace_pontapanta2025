#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read ROS2 bag and estimate steering dynamics with extra analyses and nonlinearities.

Adds:
  - Per-step local tau estimation (amplitude vs tau table/plot)
  - Histogram of |dy/dt| and percentiles (rate limit fingerprint)
  - 10–90% rise-time comparison for r and y per step
  - Nonlinear model: output slew-rate limit S [rad/s] and deadzone eps [rad]
    included in closed-loop FOPDT and FF+P models (estimable or fixable)

Usage examples:
  python3 estimate_steering_from_bag.py --bag /path/to/bag --plot
  # Fix nonlinear terms (e.g., after inspecting histograms)
  python3 estimate_steering_from_bag.py --bag /path/to/bag --fix-slew 1.5 --fix-deadzone 0.01 --plot
"""

import argparse, os, math, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
from scipy.optimize import least_squares

# -------------------- ROS bag helpers --------------------

def read_storage_id(bag_dir: str) -> str:
    meta = os.path.join(bag_dir, "metadata.yaml")
    if not os.path.isfile(meta):
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
    # autoware_auto_control_msgs/msg/AckermannControlCommand
    if hasattr(msg, "lateral") and hasattr(msg.lateral, "steering_tire_angle"):
        return float(msg.lateral.steering_tire_angle)
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
        raise RuntimeError(f"Topic {cmd_topic} not in bag.")
    if status_topic not in typemap:
        raise RuntimeError(f"Topic {status_topic} not in bag.")

    cmd_type = get_message(typemap[cmd_topic])
    status_type = get_message(typemap[status_topic])

    t_cmd, r_cmd = [], []
    t_y, y_meas = [], []

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic == cmd_topic:
            msg = deserialize_message(data, cmd_type)
            t_cmd.append(t_ns * 1e-9)
            r_cmd.append(_extract_control_cmd_angle(msg))
        elif topic == status_topic:
            msg = deserialize_message(data, status_type)
            t_y.append(t_ns * 1e-9)
            y_meas.append(_extract_steering_report_angle(msg))

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
        def med_dt(arr):
            d = np.diff(arr[:min(len(arr)-1, 4000)])
            d = d[(d > 1e-6) & np.isfinite(d)]
            return np.median(d) if len(d) else np.nan
        dt_guess = np.nanmedian([med_dt(r_df["t"].values), med_dt(y_df["t"].values)])
        dt = float(np.clip(np.nan_to_num(dt_guess, nan=0.01), 0.001, 0.05))
    t = np.arange(t0, t1, dt)
    r = np.interp(t, r_df["t"].values, r_df["r"].values)
    y = np.interp(t, y_df["t"].values, y_df["y"].values)
    return t, r, y, dt

# -------------------- utilities: steps, rise time, slopes --------------------

def detect_steps(r, t, min_step=0.02, min_hold=20):
    """Return list of (i_start, i_end, r0, r1) for each step plateau change."""
    dr = np.diff(r, prepend=r[0])
    idx = np.where(np.abs(dr) >= min_step)[0]
    if len(idx) == 0:
        return []
    picks = [idx[0]]
    for k in idx[1:]:
        if k - picks[-1] > min_hold:
            picks.append(k)
    segments = []
    prev = 0
    for k in picks + [len(r)-1]:
        if k - prev >= min_hold:
            segments.append((prev, k, r[prev], r[k]))
            prev = k
    return segments

def rise_time_10_90(t, y, y0=None, y1=None):
    """Return 10–90% rise time for a transition from y0 to y1."""
    if y0 is None: y0 = y[0]
    if y1 is None: y1 = np.median(y[int(0.8*len(y)):])
    if abs(y1 - y0) < 1e-9: return np.nan
    y10 = y0 + 0.1 * (y1 - y0)
    y90 = y0 + 0.9 * (y1 - y0)
    t10 = np.interp(y10, y, t)
    t90 = np.interp(y90, y, t)
    return max(0.0, t90 - t10)

def derivative(y, dt):
    dy = np.diff(y, prepend=y[0]) / max(dt, 1e-9)
    return dy

# -------------------- SK method for FOPDT initial guess --------------------

def sk_fopdt_single(t, r, y):
    y0 = float(y[0]); yss = float(np.nanmedian(y[int(0.8*len(y)):]))
    r0 = float(r[0]); rss = float(r[-1])
    if abs(rss - r0) < 1e-6 or abs(yss - y0) < 1e-9:
        return None
    yn = (y - y0) / (yss - y0 + 1e-12)
    y35, y85 = 0.353, 0.853
    try:
        t35 = np.interp(y35, yn, t); t85 = np.interp(y85, yn, t)
    except Exception:
        return None
    tau = 0.67 * (t85 - t35)
    theta = 1.3 * t35 - 0.29 * t85
    K = (yss - y0) / (rss - r0)
    if not (np.isfinite(K) and np.isfinite(tau) and np.isfinite(theta)): return None
    return float(K), float(max(tau, 1e-3)), float(max(theta, 0.0))

def initial_guess_from_steps(t, r, y, steps, dt):
    Ks, taus, thetas = [], [], []
    for (i0, i1, _, _) in steps:
        sl = slice(max(0, i0 - 5), min(len(t), i1 + int(0.6*(i1-i0)+5)))
        est = sk_fopdt_single(t[sl], r[sl], y[sl])
        if est: K, tau, th = est; Ks.append(K); taus.append(tau); thetas.append(th)
    if len(Ks) == 0:
        return 1.0, 0.2, 0.0
    # also refine delay by cross-corr around 1 s
    th0 = estimate_delay_xcorr(r, y, dt, max_delay=1.0)
    return float(np.median(Ks)), float(np.median(taus)), float(max(0.0, th0))

# -------------------- nonlinear building blocks --------------------

def apply_deadzone(x, eps):
    if eps <= 0.0: return x
    s = np.sign(x)
    a = np.maximum(np.abs(x) - eps, 0.0)
    return s * a

def clamp_slew(y_prev, y_candidate, dt, S):
    if S is None or S <= 0.0 or not np.isfinite(S):
        return y_candidate
    dmax = S * max(dt, 1e-9)
    dy = np.clip(y_candidate - y_prev, -dmax, dmax)
    return y_prev + dy

# -------------------- simulators (linear & nonlinear) --------------------

def simulate_fopdt_r_to_y(t, r, Kcl, tau, delay, y0=0.0, S=None, eps=0.0):
    """Closed-loop equivalent: y[k] = a*y[k-1] + b*r_d[k-1], with optional deadzone & slew on output."""
    dt = t[1]-t[0]
    y = np.zeros_like(r); y[0] = y0
    d_steps = int(max(0, round(delay/dt)))
    r_d = np.roll(r, d_steps)
    if d_steps > 0: r_d[:d_steps] = r_d[d_steps]
    a = max(0.0, 1.0 - dt/tau); b = (dt/tau) * Kcl
    for k in range(1, len(r)):
        inc = a*y[k-1] + b*r_d[k-1] - y[k-1]           # unconstrained increment
        inc = apply_deadzone(inc, eps)                  # deadzone on increment
        y_lin = y[k-1] + inc
        y[k] = clamp_slew(y[k-1], y_lin, dt, S)        # output slew-limit
    return y

def simulate_ff_p(t, r, K, tau, Kff, Kp, delay, y0=0.0, S=None, eps=0.0):
    """Plant: tau*dy/dt + y = K u, u = Kff r_d + Kp (r_d - y), with optional deadzone & output slew."""
    dt = t[1]-t[0]
    y = np.zeros_like(r); y[0] = y0
    d_steps = int(max(0, round(delay/dt)))
    r_d = np.roll(r, d_steps)
    if d_steps > 0: r_d[:d_steps] = r_d[d_steps]
    for k in range(1, len(r)):
        u = Kff * r_d[k-1] + Kp * (r_d[k-1] - y[k-1])
        inc = (dt/tau) * (K * u - y[k-1])              # unconstrained increment
        inc = apply_deadzone(inc, eps)
        y_lin = y[k-1] + inc
        y[k] = clamp_slew(y[k-1], y_lin, dt, S)
    return y

# -------------------- fitting --------------------

def estimate_delay_xcorr(r, y, dt, max_delay=1.0):
    r0 = r - np.mean(r); y0 = y - np.mean(y)
    max_lag = int(max_delay / dt)
    best_lag, best_val = 0, -1e18
    for lag in range(-max_lag, max_lag+1):
        if lag >= 0:   val = np.dot(y0[lag:], r0[:len(r0)-lag])
        else:          val = np.dot(y0[:len(y0)+lag], r0[-lag:])
        if val > best_val: best_val, best_lag = val, lag
    return best_lag * dt

def summarize(name, res, param_names):
    J = res.jac
    dof = max(1, len(res.fun) - len(res.x))
    sigma2 = float(np.sum(res.fun**2) / dof)
    try:
        cov = np.linalg.inv(J.T @ J) * sigma2
        se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
        corr = cov / (se[:,None]*se[None,:] + 1e-12)
        offdiag = float(np.max(np.abs(corr - np.eye(len(res.x)))))
    except Exception:
        se = np.full_like(res.x, np.nan); offdiag = np.nan
    print(f"\n[{name}]")
    for n, v, s in zip(param_names, res.x, se):
        print(f"  {n:>10s} = {v: .6f}  ±{s if np.isfinite(s) else float('nan'):.6f}")
    print(f"  RSS={np.sum(res.fun**2):.6e}, RMSE={math.sqrt(np.mean(res.fun**2)):.6e}")
    if np.isfinite(offdiag):
        print(f"  |corr|_max(off-diag) = {offdiag:.3f}")

# linear fits (as baseline)
def fit_fopdt_linear(t, r, y, p0):
    def residual(p):
        Kcl, tau, delay = p
        yhat = simulate_fopdt_r_to_y(t, r, max(1e-6,Kcl), max(1e-3,tau), max(0.0,delay), y0=y[0])
        return yhat - y
    return least_squares(residual, p0, bounds=([0.0,1e-3,0.0],[100.0,10.0,2.0]))

def fit_ffp_linear(t, r, y, p0):
    def residual(p):
        K, tau, Kff, Kp, delay = p
        yhat = simulate_ff_p(t, r, max(1e-6,K), max(1e-3,tau), max(0.0,Kff), max(0.0,Kp), max(0.0,delay), y0=y[0])
        return yhat - y
    lb = [0.01,1e-3,0.0,0.0,0.0]; ub=[100.0,10.0,10.0,200.0,2.0]
    return least_squares(residual, p0, bounds=(lb,ub))

# nonlinear fits (estimate S and eps)
def fit_fopdt_nl(t, r, y, p0):
    def residual(p):
        Kcl, tau, delay, S, eps = p
        yhat = simulate_fopdt_r_to_y(t, r, max(1e-6,Kcl), max(1e-3,tau), max(0.0,delay), y0=y[0],
                                     S=max(0.0,S), eps=max(0.0,eps))
        return yhat - y
    lb=[0.0,1e-3,0.0,0.0,0.0]; ub=[100.0,10.0,2.0,50.0,0.2]
    return least_squares(residual, p0, bounds=(lb,ub))

def fit_ffp_nl(t, r, y, p0):
    def residual(p):
        K, tau, Kff, Kp, delay, S, eps = p
        yhat = simulate_ff_p(t, r, max(1e-6,K), max(1e-3,tau), max(0.0,Kff), max(0.0,Kp),
                             max(0.0,delay), y0=y[0], S=max(0.0,S), eps=max(0.0,eps))
        return yhat - y
    lb=[0.01,1e-3,0.0,0.0,0.0,0.0,0.0]; ub=[100.0,10.0,10.0,200.0,2.0,50.0,0.2]
    return least_squares(residual, p0, bounds=(lb,ub))

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="rosbag2 directory")
    ap.add_argument("--cmd-topic", default="/control/command/control_cmd")
    ap.add_argument("--status-topic", default="/vehicle/status/steering_status")
    ap.add_argument("--dt", type=float, default=None, help="resample step [s]")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--step-thresh", type=float, default=0.02, help="step detect threshold [rad]")
    # Nonlinear options
    ap.add_argument("--enable-nonlinear", action="store_true", help="estimate slew rate and deadzone too")
    ap.add_argument("--fix-slew", type=float, default=None, help="fix slew rate S [rad/s] (skip estimation)")
    ap.add_argument("--fix-deadzone", type=float, default=None, help="fix deadzone eps [rad] (skip estimation)")
    args = ap.parse_args()

    r_df, y_df, storage_id = extract_series(args.bag, args.cmd_topic, args.status_topic)
    print(f"Opened bag storage='{storage_id}': r={len(r_df)}, y={len(y_df)}")
    t, r, y, dt = resample_align(r_df, y_df, args.dt)
    print(f"Aligned: N={len(t)}, dt={dt*1000:.1f} ms, duration={t[-1]-t[0]:.2f} s")

    # Step detection
    steps = detect_steps(r, t, min_step=args.step_thresh)
    print(f"Detected steps: {len(steps)}")

    # Initial guesses (SK + xcorr)
    K0, tau0, th0 = initial_guess_from_steps(t, r, y, steps, dt)
    print(f"Initial guess ~ K={K0:.3f}, tau={tau0:.3f}, delay={th0:.3f}")

    # --- Extra analysis #1: local tau per step (SK) ---
    rows = []
    for (i0, i1, r0, r1) in steps:
        sl = slice(max(0, i0-5), min(len(t), i1 + int(0.6*(i1-i0)+5)))
        est = sk_fopdt_single(t[sl], r[sl], y[sl])
        if est:
            K_s, tau_s, th_s = est
            rows.append({"k0": i0, "k1": i1, "dr": float(r1-r0), "tau_SK": tau_s, "theta_SK": th_s, "K_SK": K_s})
    if rows:
        df_tau = pd.DataFrame(rows)
        print("\n[Per-step SK estimates] (amplitude vs tau)")
        print(df_tau.to_string(index=False))
    else:
        df_tau = None
        print("\n[Per-step SK estimates] none")

    # --- Extra analysis #2: |dy/dt| histogram & percentiles ---
    dy = derivative(y, dt)
    abs_dy = np.abs(dy)
    pcts = np.percentile(abs_dy, [50, 90, 95, 99])
    print("\n[|dy/dt| percentiles] [rad/s]")
    print(f"  50%={pcts[0]:.4f}, 90%={pcts[1]:.4f}, 95%={pcts[2]:.4f}, 99%={pcts[3]:.4f}")

    # --- Extra analysis #3: rise time 10-90% for r and y per step ---
    rows_rt = []
    for (i0, i1, r0, r1) in steps:
        sl = slice(i0, min(len(t), i1 + int(0.6*(i1-i0)+1)))
        tr_r = rise_time_10_90(t[sl], r[sl], r0, r1)
        tr_y = rise_time_10_90(t[sl], y[sl], y[sl.start], np.median(y[sl.start:sl.stop]))
        rows_rt.append({"k0": i0, "dr": float(r1-r0), "rise_r": tr_r, "rise_y": tr_y, "ratio_y_over_r": (tr_y/tr_r) if (np.isfinite(tr_r) and tr_r>0) else np.nan})
    df_rt = pd.DataFrame(rows_rt)
    print("\n[Rise time 10–90% per step] [s]")
    print(df_rt.to_string(index=False, float_format=lambda v: f"{v:.4f}" if np.isfinite(v) else "nan"))

    # Heuristic initial values for nonlinearities
    S0 = float(np.clip(pcts[3], 0.1, 50.0))  # 99% slope as initial slew
    eps0 = 0.01                              # 0.01 rad (~0.57 deg) as default start
    if args.fix_slew is not None: S0 = float(args.fix_slew)
    if args.fix_deadzone is not None: eps0 = float(args.fix_deadzone)

    # --- Baseline linear fits ---
    resA_lin = fit_fopdt_linear(t, r, y, p0=[max(1e-3,K0), max(1e-3,tau0), max(0.0, th0)])
    summarize("A) FOPDT (linear)", resA_lin, ["K_cl","tau","delay"])

    resC_lin = fit_ffp_linear(t, r, y, p0=[1.0, max(1e-3,tau0), max(0.0,K0), 0.5, max(0.0,th0)])
    summarize("C) FF+P (linear)", resC_lin, ["K","tau","Kff","Kp","delay"])

    # --- Nonlinear fits (optional) ---
    if args.enable_nonlinear or (args.fix_slew is not None) or (args.fix_deadzone is not None):
        # If fixed values provided, keep them constant via tight bounds
        if args.fix_slew is not None or args.fix_deadzone is not None:
            S_lb = S_ub = S0 if args.fix_slew is not None else None
            eps_lb = eps_ub = eps0 if args.fix_deadzone is not None else None
        else:
            S_lb, S_ub, eps_lb, eps_ub = 0.0, 50.0, 0.0, 0.2

        def fit_fopdt_nl_with_bounds():
            p0=[max(1e-3,resA_lin.x[0]), max(1e-3,resA_lin.x[1]), max(0.0,resA_lin.x[2]), S0, eps0]
            lb=[0.0,1e-3,0.0, S_lb if S_lb is not None else 0.0, eps_lb if eps_lb is not None else 0.0]
            ub=[100.0,10.0,2.0, S_ub if S_ub is not None else 50.0, eps_ub if eps_ub is not None else 0.2]
            def residual(p):
                Kcl, tau, delay, S, eps = p
                yhat = simulate_fopdt_r_to_y(t, r, Kcl, tau, delay, y0=y[0], S=S, eps=eps)
                return yhat - y
            return least_squares(residual, p0, bounds=(lb,ub))

        def fit_ffp_nl_with_bounds():
            p0=[max(1e-3,resC_lin.x[0]), max(1e-3,resC_lin.x[1]), max(0.0,resC_lin.x[2]), max(0.0,resC_lin.x[3]), max(0.0,resC_lin.x[4]), S0, eps0]
            lb=[0.01,1e-3,0.0,0.0,0.0, S_lb if S_lb is not None else 0.0, eps_lb if eps_lb is not None else 0.0]
            ub=[100.0,10.0,10.0,200.0,2.0, S_ub if S_ub is not None else 50.0, eps_ub if eps_ub is not None else 0.2]
            def residual(p):
                K, tau, Kff, Kp, delay, S, eps = p
                yhat = simulate_ff_p(t, r, K, tau, Kff, Kp, delay, y0=y[0], S=S, eps=eps)
                return yhat - y
            return least_squares(residual, p0, bounds=(lb,ub))

        resA_nl = fit_fopdt_nl_with_bounds()
        summarize("A-nl) FOPDT + slew S + deadzone eps", resA_nl, ["K_cl","tau","delay","S","eps"])

        resC_nl = fit_ffp_nl_with_bounds()
        summarize("C-nl) FF+P + slew S + deadzone eps", resC_nl, ["K","tau","Kff","Kp","delay","S","eps"])
    else:
        resA_nl = resC_nl = None

    # --- Plots ---
    if args.plot:
        yA = simulate_fopdt_r_to_y(t, r, *resA_lin.x, y0=y[0])
        yC = simulate_ff_p(t, r, *resC_lin.x, y0=y[0])
        plt.figure(); plt.title("Tracking (linear fits)")
        plt.plot(t, r, label="r [rad]"); plt.plot(t, y, label="y [rad]")
        plt.plot(t, yA, "--", label="fit A: FOPDT")
        plt.plot(t, yC, "--", label="fit C: FF+P")
        plt.xlabel("time [s]"); plt.ylabel("steering [rad]"); plt.grid(True); plt.legend()

        if resA_nl is not None and resC_nl is not None:
            yA_nl = simulate_fopdt_r_to_y(t, r, *resA_nl.x[:3], y0=y[0], S=resA_nl.x[3], eps=resA_nl.x[4])
            yC_nl = simulate_ff_p(t, r, *resC_nl.x[:5], y0=y[0], S=resC_nl.x[5], eps=resC_nl.x[6])
            #simulate_ff_p(t, r, K, tau, Kff, Kp, delay, y0=0.0, S=None, eps=0.0):
            print(f"nonlinear parameters: A-nl={resA_nl.x}")
            ymyFit_nl = simulate_ff_p(t, r, 0.7, 0.1, 0.2, 0.88, 0, y0=y[0], S=0.881033, eps=0)
            plt.figure(); plt.title("Tracking (nonlinear fits)")
            plt.plot(t, r, label="r [rad]"); plt.plot(t, y, label="y [rad]")
            plt.plot(t, yA_nl, "--", label="fit A-nl: FOPDT+S+eps")
            # plt.plot(t, yC_nl, "--", label="fit C-nl: FF+P+S+eps")
            plt.plot(t, ymyFit_nl, "--", label="fit myfit-nl: FF+P+S")
            plt.xlabel("time [s]"); plt.ylabel("steering [rad]"); plt.grid(True); plt.legend()

        # step-wise views
        for (i0, i1, r0, r1) in steps:
            sl = slice(max(0, i0-20), min(len(t), i1+int(0.6*(i1-i0)+50)))
            plt.figure(); plt.title(f"Step segment k={i0} (r: {r0:.2f}->{r1:.2f})")
            plt.plot(t[sl], r[sl], label="r")
            plt.plot(t[sl], y[sl], label="y")
            plt.grid(True); plt.legend()

        # histogram of |dy/dt|
        plt.figure(); plt.title("|dy/dt| histogram")
        plt.hist(abs_dy, bins=60); plt.xlabel("|dy/dt| [rad/s]"); plt.grid(True)

        # amplitude vs tau (SK)
        if isinstance(df_tau, pd.DataFrame) and len(df_tau):
            plt.figure(); plt.title("Step amplitude vs local tau (SK)")
            plt.scatter(np.abs(df_tau["dr"]), df_tau["tau_SK"])
            plt.xlabel("|Δr| [rad]"); plt.ylabel("tau_SK [s]"); plt.grid(True)

        # rise-time scatter
        plt.figure(); plt.title("Rise time (10–90%) per step")
        plt.scatter(np.abs(df_rt["dr"]), df_rt["rise_y"], label="y rise")
        plt.scatter(np.abs(df_rt["dr"]), df_rt["rise_r"], label="r rise")
        plt.xlabel("|Δr| [rad]"); plt.ylabel("rise time [s]"); plt.grid(True); plt.legend()

        plt.show()

    print("\nNotes:")
    print("- FF-onlyモデルは省略（KとKffが積で不可識別のため）。必要なら従来版の fit を流用してください。")
    print("- 非線形同定は初期値に敏感です。収束しない場合は --fix-slew / --fix-deadzone を指定して線形パラメータのみ推定→段階的に開放が安定です。")
    print("- 立上り時間は10–90%定義を採用。S–K法での初期値は各ステップ片から算出しています。")

if __name__ == "__main__":
    main()
