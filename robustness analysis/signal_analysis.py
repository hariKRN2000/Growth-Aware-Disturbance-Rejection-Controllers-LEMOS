# signal_analysis.py

import numpy as np
import pandas as pd

def compute_rise_time(signal, time, final_value, tolerance_band):
    """
    Compute the rise time of a signal.
    Rise time = time from 0% to final_value - tolerance_band.
    """
    start_idx = np.where(signal >= 0 * final_value)[0]
    end_idx = np.where(signal >= final_value - tolerance_band)[0]
    if len(start_idx) == 0 or len(end_idx) == 0:
        return None
    return time[end_idx[0]] - time[start_idx[0]]



def compute_settling_time(signal, time, setpoint_value, tol, cap=960):
        """
        Settling time = earliest time t_k such that signal[t_k:] is entirely within
        [setpoint_value - tol, setpoint_value + tol]. If none, return cap.

        Parameters
        ----------
        signal : 1D np.array
        time   : 1D np.array (same length as signal)
        final_value : float (the setpoint)
        tol    : float (half-width of tolerance band around setpoint)
        cap    : float (returned if never persistently inside band)

        Returns
        -------
        float : settling time in same units as `time`
        """
        lower = setpoint_value - tol
        upper = setpoint_value + tol

        inside = (signal >= lower) & (signal <= upper)               # boolean vector
        # suffix_all[i] == True  <=> inside[i:] are all True
        suffix_all = np.flip(np.cumprod(np.flip(inside.astype(int))).astype(bool))

        idx_candidates = np.where(suffix_all)[0]
        if idx_candidates.size == 0:
            return cap
        return float(time[idx_candidates[0]])

def compute_itae_post_disturbance(signal, time_array, setpoint_value, perturbed_time):
    signal = np.asarray(signal)
    time_array = np.asarray(time_array)

    if signal.shape[0] != time_array.shape[0]:
        raise ValueError("signal and time_array must have the same length")

    # Find first index at or after the disturbance time
    start_idx = int(np.searchsorted(time_array, perturbed_time, side="left"))
    if start_idx >= len(time_array) - 1:
        return 0.0

    # Slice post-disturbance window
    sig = signal[start_idx:]
    t_abs = time_array[start_idx:]

    # Relative time starts at 0 at disturbance
    t = t_abs - t_abs[0]

    # Absolute tracking error (constant setpoint)
    e = np.abs(setpoint_value - sig)

    # Trapezoid integration of f(t) = t * |e(t)|
    itae = 0.0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        if dt < 0:
            raise ValueError("time_array must be non-decreasing")
        itae += 0.5 * ((t[i] * e[i]) + (t[i - 1] * e[i - 1])) * dt
        #itae += 0.5 * (e[i] + e[i - 1]) * dt

    return float(itae)

def analyze_signal(signal, time, setpoint_value, tolerance_band, max_dynamic_range, settle_time_cap = 960, perturbed_time=600):
    """
    Analyze a single signal:
    - rise time
    - settling time (capped if None)
    - overshoot error (%)
    - ITAE (Integrated Time Averaged Error)
    """
    # === Rise Time ===
    start_idx = np.where(signal >= 0 * setpoint_value)[0]
    end_idx = np.where(signal >= setpoint_value - tolerance_band)[0]
    rise_time = time[end_idx[0]] - time[start_idx[0]] if len(start_idx) > 0 and len(end_idx) > 0 else None

    # === Settling Time ===
    settling_time = compute_settling_time(signal, time, setpoint_value, tol=tolerance_band)
    if settling_time is None:
        settling_time = settle_time_cap  # cap if None

    # === Overshoot Error ===
    max_value = np.max(signal)
    overshoot_error = (max_value - setpoint_value) / max_dynamic_range * 100

    # === ITAE ===
    ITAE = compute_itae_post_disturbance(signal, time, setpoint_value, perturbed_time)

    return rise_time, settling_time, overshoot_error, ITAE

def analyze_simulation(
    P1_signal, P2_signal, 
    time1, time2, 
    st_pt_1, st_pt_2, 
    green_mean, red_mean, 
    tol_band_frac=0.1
):
    """
    Analyze single (1D) signals for rise time, overshoot error, and settling time.
    """
    max_dynamic_range = np.max(green_mean - red_mean)
    tolerance_band = tol_band_frac * max_dynamic_range

    # Analyze Set Point 1
    rise_1, settle_1, overshoot_1 = analyze_signal(P1_signal, time1, st_pt_1, tolerance_band, max_dynamic_range)

    # Analyze Set Point 2
    rise_2, settle_2, overshoot_2 = analyze_signal(P2_signal, time2, st_pt_2, tolerance_band, max_dynamic_range)

    # Create DataFrame
    metrics_df = pd.DataFrame({
        'Set Point': ['SP1', 'SP2'],
        'Rise Time': [rise_1, rise_2],
        'Settling Time': [settle_1, settle_2],
        'Overshoot Error (%)': [overshoot_1, overshoot_2]
    })

    return metrics_df

if __name__ == "__main__":
    # Example usage
    print("This module defines functions for analyzing signals for control metrics.")
