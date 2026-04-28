# Code for Ghelasi et. al empirical test of M1 & Y1 on NO1 electricity price dataset
# Developed by [Nutan Gupta & Mathias Helseth] — [28/04/2026]
# University of Inland Norway
# All rights reserved. For academic use only.


"""
Empirical Test: Forward Price Maturity Selection on NO1 Market
==============================================================
Replicates the liquidity analysis of Ghelasi and Ziel (2025) on the
NO1 dataset using LassoLarsIC with BIC criterion.

For each forecasting horizon h, we fit a LassoLarsIC model with BIC
on a rolling expanding window and record how often M1 and Y1 are
selected.

This directly tests whether M1 or Y1 dominates at each horizon,
following the methodology of Ghelasi and Ziel (2025).
"""

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoLarsIC
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': '#f8f9fa',
    'axes.grid': True, 'grid.alpha': 0.35,
    'font.size': 11, 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.dpi': 110,
    'savefig.dpi': 150, 'savefig.bbox': 'tight',
})

# ── Load data ──────────────────────────────────────────────────────────────
DATA_PATH = 'master_clean_NO1.csv'
df = pd.read_csv(DATA_PATH, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

TRAIN_END = pd.Timestamp('2021-12-31')
VAL_START = pd.Timestamp('2022-01-01')
TEST_END  = pd.Timestamp('2024-12-31')

df['split'] = df['date'].apply(
    lambda d: 'train' if d <= TRAIN_END
    else ('eval' if d <= TEST_END else 'holdout'))

# ── Control features (shared across all horizons) ─────────────────────────
CONTROLS = [c for c in [
    'fuel_gas_eur_mwh', 'fuel_carbon_eur_t',
    'hydro_reserve_gwh', 'reservoir_deviation_gwh',
    'weather_temp_c', 'load_mw', 'macro_eur_nok',
    'cal_month_sin', 'cal_month_cos',
    'cal_day_of_year_sin', 'cal_day_of_year_cos',
    'season_winter', 'season_summer',
    'lag_price_365d',
] if c in df.columns]

# Forward candidates to compare
FWD_CANDIDATES = {
    'forward_M1':  'M1 (front-month)',
    'forward_Y1':  'Y1 (year-ahead)'
}

# ── Rolling LassoLarsIC selection ─────────────────────────────────────────
HORIZONS   = [30, 90, 180, 360]
ROLL_STEP  = 30
WINDOW     = 365 * 3   # 3-year rolling window

def run_selection(df, horizons, controls, fwd_candidates,
                  window=WINDOW, roll_step=ROLL_STEP):
    """
    For each horizon h and each rolling window step:
      1. Build X = controls + all forward candidates
      2. Build y = spot price h days ahead
      3. Fit LassoLarsIC (BIC)
      4. Record which forward candidates are selected (non-zero coef)
    Returns: DataFrame with selection results per horizon and candidate.
    """
    all_features = controls + list(fwd_candidates.keys())
    all_features = [c for c in all_features if c in df.columns]

    eval_mask  = (df['date'] >= VAL_START) & (df['date'] <= TEST_END)
    eval_dates = df.loc[eval_mask, 'date'].values[::roll_step]

    records = []

    for h in horizons:
        print(f"  Testing h={h}d...", end=' ')
        n_selected_m1 = 0
        n_selected_y1 = 0
        total_steps   = 0
        coef_sums     = {k: 0.0 for k in fwd_candidates}
        coef_counts   = {k: 0   for k in fwd_candidates}

        for fc_ts in eval_dates:
            fc_date  = pd.Timestamp(fc_ts)
            tgt_date = fc_date + pd.Timedelta(days=h)

            # Training window
            t_win = fc_date - pd.Timedelta(days=window)
            tr    = df[(df['date'] > t_win) & (df['date'] <= fc_date)].copy()

            # Target: price h days ahead from training window perspective
            # Use shifted target within training data
            tr = tr.copy()
            tr['target'] = tr['spot_price'].shift(-h)
            tr = tr.dropna(subset=['target'] + all_features)
            if len(tr) < 60:
                continue

            X = tr[all_features].fillna(0).values.astype(float)
            y = tr['target'].values.astype(float)

            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            try:
                model = LassoLarsIC(criterion='bic', max_iter=200)
                model.fit(Xs, y)
                coefs = model.coef_
            except Exception:
                continue

            total_steps += 1

            for j, feat in enumerate(all_features):
                if feat in fwd_candidates:
                    abs_c = abs(coefs[j])
                    if abs_c > 1e-8:
                        coef_sums[feat]   += abs_c
                        coef_counts[feat] += 1

        print(f"{total_steps} steps")

        for feat, label in fwd_candidates.items():
            if feat not in all_features:
                records.append({'horizon': h, 'feature': feat,
                                'label': label,
                                'selection_rate': np.nan,
                                'mean_abs_coef': np.nan})
                continue
            sel_rate = coef_counts[feat] / max(total_steps, 1)
            records.append({
                'horizon':       h,
                'feature':       feat,
                'label':         label,
                'selection_rate': round(sel_rate, 4),
                'n_selected':    coef_counts[feat],
                'n_total':       total_steps,
            })

    return pd.DataFrame(records)


print("Running LassoLarsIC (BIC) forward maturity selection test...")
print(f"Controls: {len(CONTROLS)} variables")
print(f"Forward candidates: {list(FWD_CANDIDATES.keys())}")
print(f"Horizons: {HORIZONS}")
print()

results = run_selection(df, HORIZONS, CONTROLS, FWD_CANDIDATES)

# ── Summary table ──────────────────────────────────────────────────────────
print()
print("="*70)
print("SELECTION RATE — fraction of rolling windows where feature is selected")
print("(LassoLarsIC BIC  |  higher = more consistently chosen)")
print("="*70)
pivot_rate = results.pivot_table(
    index='label', columns='horizon', values='selection_rate').round(3)
print(pivot_rate.to_string())

results.to_csv('forward_maturity_selection_NO1.csv', index=False)
print("\nSaved: forward_maturity_selection_NO1.csv")

fig, ax = plt.subplots(figsize=(10, 5))

fig.suptitle(
    'Forward Price Maturity Selection — NO1 Market\n'
    'LassoLarsIC with BIC',
    fontsize=13,
    fontweight='bold'
)

colors = {
    'forward_M1': '#2196F3',
    'forward_Y1': '#4CAF50',
    'epad_M1':    '#FF9800',
    'epad_Y1':    '#9C27B0'
}

x = np.arange(len(HORIZONS))
width = 0.2

# Plot bars
for i, (feat, label) in enumerate(FWD_CANDIDATES.items()):
    sub = results[results['feature'] == feat].sort_values('horizon')
    vals = sub['selection_rate'].values

    ax.bar(
        x + i * width,
        vals,
        width,
        color=colors.get(feat, 'gray'),
        alpha=0.82,
        label=label
    )

# Axis formatting
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([f'h={h}d' for h in HORIZONS])
ax.set_ylabel('Selection Rate (fraction of windows)')
ax.legend(fontsize=9)

# Crossover marker
ax.axvline(1.5 + width * 1.5, color='gray', ls='--', lw=1, alpha=0.6)
ax.text(
    1.5 + width * 1.5 + 0.05,
    ax.get_ylim()[1] * 0.95,
    'h=180/360\ncrossover zone',
    fontsize=8,
    color='gray',
    va='top'
)

plt.tight_layout()
plt.savefig('fig_forward_maturity_selection.png', dpi=300, bbox_inches='tight')
plt.show()




