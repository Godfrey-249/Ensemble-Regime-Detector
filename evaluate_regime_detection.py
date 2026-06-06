# evaluate_regime_detection.py
import pandas as pd
import numpy as np
import os
from datetime import timedelta
from HMM_Regime_Detection import walk_forward_hmm  # assumes your script in same folder
import matplotlib.pyplot as plt

# ------------- PARAMETERS -------------
INPUT_CSV = "XAUUSD_2023_2025_Data.csv"   # replace with your file path
WINDOW_SIZE = 500
N_REGIMES = 4

# Ground-truth rule parameters (adjustable)
BREAKOUT_ATR_MULT = 2.0    # breakout if current 4H range > 2.0 * rolling ATR
RETRACE_DROP = -0.015      # retracement if 4H return < -1.5%
CONSOL_ATHR = 0.4          # normalization threshold for consolidation (norm_range < this)
# --------------------------------------

# run the walk-forward (this will return wf_df as in your script)
wf_df = walk_forward_hmm(INPUT_CSV, window_size=WINDOW_SIZE, n_regimes=N_REGIMES)

# ensure time is datetime
wf_df['time'] = pd.to_datetime(wf_df['time'])

# Build a simple ground-truth label by rules (per-bar)
def rule_label(row):
    # priority: extreme vol breakout -> breakout; big negative return -> retracement; low norm_range -> consolidation
    if row['norm_range'] >= BREAKOUT_ATR_MULT:
        return 'ATH_Breakout' if row['log_return'] > 0 else 'Retracement'
    if row['log_return'] <= RETRACE_DROP:
        return 'Retracement'
    if row['norm_range'] < CONSOL_ATHR:
        return 'Consolidation'
    return 'ATH_Grind'

wf_df['GT_Regime'] = wf_df.apply(rule_label, axis=1)

# Now compare GT_Regime to Stable_Regime (your detection)
# compute detection events where GT_Regime != GT_Regime.shift(1)
wf_df = wf_df.sort_values('time').reset_index(drop=True)
wf_df['gt_change'] = (wf_df['GT_Regime'] != wf_df['GT_Regime'].shift(1)).astype(int)
wf_df['det_change'] = (wf_df['Stable_Regime'] != wf_df['Stable_Regime'].shift(1)).astype(int)

# Build lists of change events: time and regime
gt_events = wf_df[wf_df['gt_change'] == 1][['time', 'GT_Regime']].reset_index()
det_events = wf_df[wf_df['det_change'] == 1][['time', 'Stable_Regime', 'regime_confidence']].reset_index()

# For each ground-truth event, find the first detection event of the same type within a window (e.g., next 48 bars)
MAX_LAG_BARS = 48
lags = []
missed = 0
false_alarms = 0

for idx, row in gt_events.iterrows():
    t0 = row['time']
    gt = row['GT_Regime']
    # define search window
    window = wf_df[(wf_df['time'] >= t0) & (wf_df['time'] <= t0 + pd.Timedelta(MAX_LAG_BARS * 4, unit='h'))]  # approximate, if bars are 4H
    # find first detection where Stable_Regime == gt within window
    found = window[window['Stable_Regime'] == gt]
    if not found.empty:
        t_detect = found['time'].iloc[0]
        lag_bars = (t_detect - t0) / pd.Timedelta(1, unit='h') / 4.0  # number of 4H bars approx
        lags.append({'gt_time': t0, 'gt': gt, 'det_time': t_detect, 'lag_bars': float(lag_bars)})
    else:
        missed += 1

# false alarms: detection events that are not followed by a GT change within -MAX_LAG..+MAX_LAG
false_alarms = 0
for idx, row in det_events.iterrows():
    t_detect = row['time']
    det_regime = row['Stable_Regime']
    # Find if a GT change to the *same regime* occurred recently or is about to occur.
    nearby_window = (gt_events['time'] >= t_detect - pd.Timedelta(MAX_LAG_BARS * 4, unit='h')) & (gt_events['time'] <= t_detect + pd.Timedelta(MAX_LAG_BARS * 4, unit='h'))
    nearby_and_correct_type = gt_events['GT_Regime'] == det_regime
    nearby = gt_events[nearby_window & nearby_and_correct_type]
    if nearby.empty:
        false_alarms += 1

# summary metrics
total_gt = len(gt_events)
tp = len(lags)
fn = missed
fp = false_alarms
precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
recall = tp / total_gt if total_gt > 0 else np.nan
mean_lag = np.mean([d['lag_bars'] for d in lags]) if lags else np.nan
median_lag = np.nanmedian([d['lag_bars'] for d in lags]) if lags else np.nan

print("=== Regime Detection Evaluation ===")
print(f"Total GT changes: {total_gt}")
print(f"True Positives (detected within window): {tp}")
print(f"False Negatives (missed): {fn}")
print(f"False Positives (detections w/no nearby GT change): {fp}")
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}")
print(f"Mean lag (bars): {mean_lag:.2f}, median lag (bars): {median_lag:.2f}")

# Plot lag histogram
lags_df = pd.DataFrame(lags)
if not lags_df.empty:
    plt.hist(lags_df['lag_bars'], bins=30)
    plt.title('Detection lag (in 4H bars)')
    plt.xlabel('Lag (4H bars)')
    plt.show()

# Save diagnostics
wf_df.to_csv("wf_regime_results.csv", index=False)
lags_df.to_csv("det_lags.csv", index=False)
print("Saved wf_regime_results.csv and det_lags.csv for inspection.")