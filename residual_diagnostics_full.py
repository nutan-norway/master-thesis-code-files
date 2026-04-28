# Code for residual diagnostic test for NO1 electricity price dataset
# Developed by [Nutan Gupta & Mathias Helseth] — [19/04/2026]
# University of Inland Norway
# All rights reserved. For academic use only.

"""
Residual Diagnostic Tests — All 5 Baseline Models
==================================================

Tests:
  1. Ljung-Box (10 lags)   — residual autocorrelation
  2. ARCH-LM  (5 lags)     — conditional heteroskedasticity
  3. Jarque-Bera            — normality

Models:  Naive, Seasonal-WD, SARIMA(2,1,1)(1,0,1)[7],
         ElasticNet-Unconstrained, ElasticNet-Constrained
Horizons: h = 1, 7, 30, 90 days
Period:   Jan 2023 — Dec 2024 (test set)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({
    'figure.facecolor': 'white',  'axes.facecolor': '#f8f9fa',
    'axes.grid': True,            'grid.alpha': 0.35,
    'font.size': 10,              'axes.spines.top': False,
    'axes.spines.right': False,   'figure.dpi': 110,
    'savefig.dpi': 150,           'savefig.bbox': 'tight',
})

# ══════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
DATA_PATH = 'master_clean_NO1.csv' 

df = pd.read_csv(DATA_PATH, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

TRAIN_END  = pd.Timestamp('2021-12-31')
TEST_START = pd.Timestamp('2023-01-01')
TEST_END   = pd.Timestamp('2024-12-31')

df['month']    = df['date'].dt.month
df['dow']      = df['date'].dt.dayofweek
df['iso_week'] = df['date'].dt.isocalendar().week.astype(int)

print(f"Data loaded: {len(df)} rows | {df['date'].min().date()} to {df['date'].max().date()}")

# ══════════════════════════════════════════════════════════════════════════
# BUILD FEATURES
# ══════════════════════════════════════════════════════════════════════════
train = df[df['date'] <= TRAIN_END].copy()

def make_profile(df_tr, col):
    p = (df_tr.groupby(['month', 'iso_week'])[col]
         .agg(s_mean='mean', s_std='std').reset_index())
    p['s_mean'] = p['s_mean'].rolling(3, center=True, min_periods=1).mean()
    return p

for col in ['spot_price', 'hydro_reserve_gwh']:
    prof = make_profile(train, col)
    m = df[['month', 'iso_week']].merge(
        prof[['month', 'iso_week', 's_mean', 's_std']],
        on=['month', 'iso_week'], how='left')
    df[f'seasonal_{col}_mean'] = m['s_mean'].values
    df[f'seasonal_{col}_std']  = m['s_std'].values

df['price_vs_seasonal'] = df['spot_price'] - df['seasonal_spot_price_mean']
df['price_seasonal_z']  = (df['price_vs_seasonal']
                            / df['seasonal_spot_price_std'].clip(lower=1))
df['hydro_seasonal_z']  = ((df['hydro_reserve_gwh']
                             - df['seasonal_hydro_reserve_gwh_mean'])
                            / df['seasonal_hydro_reserve_gwh_std'].clip(lower=1))
df['implied_spot_M1']   = df['forward_M1'] + df['epad_M1']
df['implied_spot_Y1']   = df['forward_Y1'] + df['epad_Y1']
df['epad_slope']        = df['epad_M1'] - df['epad_Y1']

for w in [7, 30]:
    df[f'roll_price_mean_{w}d'] = df['spot_price'].shift(1).rolling(w).mean()
    df[f'roll_price_std_{w}d']  = df['spot_price'].shift(1).rolling(w).std()

mc = df['date'].dt.month
df['season_winter'] = ((mc <= 2) | (mc == 12)).astype(int)
df['season_spring'] = ((mc >= 3) & (mc <= 5)).astype(int)
df['season_summer'] = ((mc >= 6) & (mc <= 8)).astype(int)
df['season_autumn'] = ((mc >= 9) & (mc <= 11)).astype(int)
df['month_num']     = mc
df['week_of_year']  = df['iso_week']

hz = (df['hydro_reserve_gwh'] - df['hydro_reserve_gwh'].mean()) / df['hydro_reserve_gwh'].std()
gz = (df['fuel_gas_eur_mwh']  - df['fuel_gas_eur_mwh'].mean())  / df['fuel_gas_eur_mwh'].std()
df['interaction_hydro_gas'] = hz * gz

FEATURES = [c for c in [
    'lag_price_1d', 'lag_price_7d', 'lag_price_30d', 'lag_price_365d',
    'ma_price_7d', 'ma_price_30d',
    'roll_price_mean_7d', 'roll_price_std_7d',
    'roll_price_mean_30d', 'roll_price_std_30d',
    'hydro_reserve_gwh', 'reservoir_deviation_gwh', 'hydro_seasonal_z',
    'price_vs_seasonal', 'price_seasonal_z',
    'seasonal_spot_price_mean', 'seasonal_hydro_reserve_gwh_mean',
    'fuel_gas_eur_mwh', 'fuel_carbon_eur_t',
    'forward_M1', 'forward_Y1', 'epad_M1', 'epad_Y1',
    'epad_slope', 'implied_spot_M1', 'implied_spot_Y1',
    'interaction_hydro_gas', 'macro_eur_nok',
    'cal_month_sin', 'cal_month_cos',
    'cal_day_of_year_sin', 'cal_day_of_year_cos',
    'season_winter', 'season_spring', 'season_summer', 'season_autumn',
    'month_num', 'week_of_year',
] if c in df.columns]

print(f"Feature columns available: {len(FEATURES)}")

# Sign constraints for constrained ElasticNet
# Positive direction: coefficient sign must match physical theory
COEF_BOUNDS_SIGN = {
    'hydro_reserve_gwh':       (-1, 0),   # more water -> lower price
    'reservoir_deviation_gwh': (-1, 0),
    'hydro_seasonal_z':        (-1, 0),
    'fuel_gas_eur_mwh':        ( 0, 1),   # higher gas -> higher price
    'fuel_carbon_eur_t':       ( 0, 1),
    'forward_M1':              ( 0, 1),
    'forward_Y1':              ( 0, 1),
    'epad_M1':                 ( 0, 1),
    'epad_Y1':                 ( 0, 1),
    'lag_price_1d':            ( 0, 1),
    'price_vs_seasonal':       ( 0, 1),
}

# ══════════════════════════════════════════════════════════════════════════
# ROLLING RESIDUAL COLLECTION
# ══════════════════════════════════════════════════════════════════════════
HORIZONS   = [1, 7, 30, 90]
WINDOW     = 365 * 3   # 3-year rolling window
ROLL_STEP  = 20        # evaluate every 20 days (gives ~37 steps)

test_dates = df.loc[
    (df['date'] >= TEST_START) & (df['date'] <= TEST_END),
    'date'
].values[::ROLL_STEP]

MODELS = [
    'Naive',
    'Seasonal-WD',
    'SARIMA',
    'ElasticNet-Unconstrained',
    'ElasticNet-Constrained',
]

all_resid = {m: {h: [] for h in HORIZONS} for m in MODELS}

def season_label(m):
    return {12:'W', 1:'W', 2:'W',
             3:'Sp', 4:'Sp', 5:'Sp',
             6:'Su', 7:'Su', 8:'Su',
             9:'A', 10:'A', 11:'A'}[m]

print(f"\nCollecting residuals over {len(test_dates)} rolling evaluation steps...")
print("This may take 5-10 minutes due to SARIMA fitting at each step.\n")

for step_i, fc_ts in enumerate(test_dates):
    fc = pd.Timestamp(fc_ts)
    tw = fc - pd.Timedelta(days=WINDOW)
    tr = df[(df['date'] > tw) & (df['date'] <= fc)].copy()
    if len(tr) < 90:
        continue

    print(f"  Step {step_i+1}/{len(test_dates)}  |  {fc.date()}", end='\r')

    # ── Weekday / season lookups ──────────────────────────────────────────
    wd_means = tr.groupby(tr['date'].dt.dayofweek)['spot_price'].mean().to_dict()
    last_p   = float(tr['spot_price'].iloc[-1])
    tr2      = tr.assign(
        dow_=tr['date'].dt.dayofweek,
        sea_=tr['date'].dt.month.map(season_label)
    )
    wd_sea   = tr2.groupby(['dow_', 'sea_'])['spot_price'].mean().to_dict()

    # ── Fit ElasticNet-Unconstrained ──────────────────────────────────────
    try:
        sc_u  = StandardScaler()
        Xs_u  = sc_u.fit_transform(tr[FEATURES].fillna(0).values.astype(float))
        net_u = ElasticNet(alpha=0.5, l1_ratio=0.5,
                           max_iter=1000, fit_intercept=True)
        net_u.fit(Xs_u, tr['spot_price'].values.astype(float))
        en_u_ok = True
    except Exception:
        en_u_ok = False

    # ── Fit ElasticNet-Constrained ────────────────────────────────────────
    # Uses same fit as unconstrained then clips coefficients to enforce
    # physical sign constraints from the Norwegian hydro merit-order
    try:
        sc_c  = StandardScaler()
        Xs_c  = sc_c.fit_transform(tr[FEATURES].fillna(0).values.astype(float))
        net_c = ElasticNet(alpha=0.5, l1_ratio=0.5,
                           max_iter=1000, fit_intercept=True)
        net_c.fit(Xs_c, tr['spot_price'].values.astype(float))
        for j, col in enumerate(FEATURES):
            if col in COEF_BOUNDS_SIGN:
                lo, hi = COEF_BOUNDS_SIGN[col]
                c = net_c.coef_[j]
                net_c.coef_[j] = np.clip(c, lo * abs(c), hi * abs(c))
        en_c_ok = True
    except Exception:
        en_c_ok = False

    # ── Fit SARIMA(2,1,1)(1,0,1)[7] ──────────────────────────────────────
    sarima_ok        = False
    sarima_forecasts = {}
    try:
        y_s  = tr['spot_price'].values.astype(float)
        mdl  = SARIMAX(y_s, order=(2, 1, 1),
                       seasonal_order=(1, 0, 1, 7),
                       enforce_stationarity=False,
                       enforce_invertibility=False)
        res_s = mdl.fit(disp=False, maxiter=50)
        fc_s  = res_s.forecast(steps=max(HORIZONS))
        for h in HORIZONS:
            sarima_forecasts[h] = float(fc_s[h - 1])
        sarima_ok = True
    except Exception:
        pass

    # ── Collect residuals for each horizon ────────────────────────────────
    for h in HORIZONS:
        tgt  = fc + pd.Timedelta(days=h)
        mask = df['date'] == tgt
        if not mask.any():
            continue
        y_true = float(df.loc[mask, 'spot_price'].values[0])

        # Naive
        y_naive = last_p if h <= 7 else float(wd_means.get(tgt.dayofweek, last_p))
        all_resid['Naive'][h].append(y_true - y_naive)

        # Seasonal-WD
        y_wd = float(wd_sea.get(
            (tgt.dayofweek, season_label(tgt.month)),
            float(tr['spot_price'].mean())
        ))
        all_resid['Seasonal-WD'][h].append(y_true - y_wd)

        # SARIMA
        if sarima_ok:
            all_resid['SARIMA'][h].append(y_true - sarima_forecasts[h])

        # ElasticNet-Unconstrained
        if en_u_ok:
            fr = df.loc[df['date'] == fc, FEATURES].fillna(0).values.astype(float)
            if len(fr) > 0:
                y_en_u = float(net_u.predict(sc_u.transform(fr))[0])
                all_resid['ElasticNet-Unconstrained'][h].append(y_true - y_en_u)

        # ElasticNet-Constrained
        if en_c_ok:
            fr = df.loc[df['date'] == fc, FEATURES].fillna(0).values.astype(float)
            if len(fr) > 0:
                y_en_c = float(net_c.predict(sc_c.transform(fr))[0])
                all_resid['ElasticNet-Constrained'][h].append(y_true - y_en_c)

print("\n\nResidual collection complete.")
print(f"{'Model':<26}  " + "  ".join([f"h={h}" for h in HORIZONS]))
print("-" * 55)
for m in MODELS:
    counts = [len(all_resid[m][h]) for h in HORIZONS]
    print(f"{m:<26}  " + "  ".join([f"{c:>4}" for c in counts]))

# ══════════════════════════════════════════════════════════════════════════
# RUN DIAGNOSTIC TESTS
# ══════════════════════════════════════════════════════════════════════════

def run_diagnostics(e_list):
    """
    Run Ljung-Box, ARCH-LM and Jarque-Bera on a residual list.
    Returns a dict of test statistics and p-values.
    """
    e = np.array(e_list, dtype=float)
    if len(e) < 8:
        return None

    # Ljung-Box (10 lags) — tests for residual autocorrelation
    lb     = acorr_ljungbox(e, lags=[10], return_df=True)
    lb_s   = round(float(lb['lb_stat'].iloc[0]),   2)
    lb_p   = round(float(lb['lb_pvalue'].iloc[0]), 4)

    # ARCH-LM (5 lags) — tests for conditional heteroskedasticity
    try:
        r       = het_arch(e, nlags=5)
        arch_s  = round(float(r[0]), 2)
        arch_p  = round(float(r[1]), 4)
    except Exception:
        arch_s, arch_p = np.nan, np.nan

    # Jarque-Bera — tests for normality
    jb_s, jb_p, skew, kurt = jarque_bera(e)

    return {
        'lb_stat':   lb_s,
        'lb_p':      lb_p,
        'arch_stat': arch_s,
        'arch_p':    arch_p,
        'jb_stat':   round(float(jb_s),   2),
        'jb_p':      round(float(jb_p),   4),
        'skew':      round(float(skew),   3),
        'kurt':      round(float(kurt),   3),
        'rmse':      round(float(np.sqrt(np.mean(e**2))), 2),
        'bias':      round(float(np.mean(e)),             2),
        'n':         int(len(e)),
    }


rows = []
for model in MODELS:
    for h in HORIZONS:
        d = run_diagnostics(all_resid[model][h])
        if d:
            d['model']   = model
            d['horizon'] = h
            rows.append(d)

diag_df = pd.DataFrame(rows)
diag_df.to_csv('residual_diagnostics_NO1.csv', index=False)

# ══════════════════════════════════════════════════════════════════════════
# PRINT RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════

def sig(p):
    """Return significance stars."""
    if np.isnan(p): return '   '
    if p < 0.01:    return '** '
    if p < 0.05:    return '*  '
    return '   '

print()
print("=" * 100)
print("RESIDUAL DIAGNOSTIC TESTS — ALL 5 BASELINE MODELS")
print("Evaluation period: Jan 2023 — Dec 2024 (test set)")
print("* p < 0.05   ** p < 0.01")
print()
print("LB   = Ljung-Box portmanteau test (lags=10)  — H0: no residual autocorrelation")
print("ARCH = ARCH-LM test (lags=5)                 — H0: no conditional heteroskedasticity")
print("JB   = Jarque-Bera test                      — H0: residuals are normally distributed")
print("=" * 100)

header = (f"{'Model':<26} {'h':>4} {'n':>4}  "
          f"{'LB stat':>8} {'LB p':>8}  "
          f"{'ARCH stat':>10} {'ARCH p':>8}  "
          f"{'JB stat':>8} {'JB p':>8}  "
          f"{'Skew':>7} {'Kurt':>7}  {'RMSE':>8}")
print(header)
print("-" * 100)

prev_model = None
for _, row in diag_df.sort_values(['model', 'horizon']).iterrows():
    if prev_model and row['model'] != prev_model:
        print()
    prev_model = row['model']
    print(
        f"{row['model']:<26} {int(row['horizon']):>4} {int(row['n']):>4}  "
        f"{row['lb_stat']:>8.2f} {row['lb_p']:>7.4f}{sig(row['lb_p'])} "
        f"{row['arch_stat']:>10.2f} {row['arch_p']:>7.4f}{sig(row['arch_p'])} "
        f"{row['jb_stat']:>8.2f} {row['jb_p']:>7.4f}{sig(row['jb_p'])} "
        f"{row['skew']:>7.3f} {row['kurt']:>7.3f}  {row['rmse']:>8.2f}"
    )

print()
print("Saved: residual_diagnostics_NO1.csv")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE: RESIDUAL DIAGNOSTICS AT h=1d FOR ALL 5 MODELS
# ══════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(5, 3, figsize=(17, 21))
fig.suptitle(
    'Residual Diagnostics — All 5 Baseline Models  |  h = 1 day  |  Test Set 2023-2024\n'
    'Left: Residual time series  |  Centre: ACF  |  Right: Normal Q-Q plot',
    fontsize=12, fontweight='bold'
)

for i, model in enumerate(MODELS):
    e = np.array(all_resid[model][1], dtype=float)

    if len(e) < 5:
        for j in range(3):
            axes[i, j].text(0.5, 0.5, 'Insufficient data',
                             ha='center', va='center',
                             transform=axes[i, j].transAxes)
            axes[i, j].set_title(model, fontsize=10, fontweight='bold')
        continue

    d = run_diagnostics(e)

    # ── Panel 1: Residual time series ──────────────────────────────────
    ax1 = axes[i, 0]
    ax1.plot(e, lw=0.9, color='steelblue', alpha=0.85)
    ax1.axhline(0, color='black', lw=1.2)
    ax1.axhline( 2 * e.std(), color='red', ls='--', lw=1, alpha=0.7, label='±2σ')
    ax1.axhline(-2 * e.std(), color='red', ls='--', lw=1, alpha=0.7)
    ax1.set_title(model, fontsize=10, fontweight='bold')
    ax1.set_ylabel('Residual (EUR/MWh)')
    ax1.set_xlabel('Evaluation step')
    ax1.legend(fontsize=8)

    # Annotate with test statistics
    def star(p): return '**' if p < 0.01 else '*' if p < 0.05 else ''
    ann = (
        f"LB p = {d['lb_p']:.3f}{star(d['lb_p'])}"
        f"   ARCH p = {d['arch_p']:.3f}{star(d['arch_p'])}\n"
        f"JB p = {d['jb_p']:.3f}{star(d['jb_p'])}"
        f"   Skew = {d['skew']:.2f}   Kurt = {d['kurt']:.2f}\n"
        f"RMSE = {d['rmse']:.1f} EUR/MWh   Bias = {d['bias']:.1f}"
    )
    ax1.text(0.02, 0.03, ann, transform=ax1.transAxes, fontsize=8,
             bbox=dict(fc='white', ec='gray', alpha=0.85), va='bottom')

    # ── Panel 2: ACF ───────────────────────────────────────────────────
    ax2 = axes[i, 1]
    plot_acf(e, lags=min(15, len(e) // 2 - 1),
             ax=ax2, zero=False, alpha=0.05)
    ax2.set_title('ACF of Residuals', fontsize=9)
    ax2.set_xlabel('Lag (evaluation steps)')

    # ── Panel 3: Normal Q-Q plot ───────────────────────────────────────
    ax3 = axes[i, 2]
    stats.probplot(e, dist='norm', plot=ax3)
    ax3.set_title('Normal Q-Q Plot', fontsize=9)
    ax3.get_lines()[0].set(markersize=5, alpha=0.65, color='steelblue')
    ax3.get_lines()[1].set(color='red', lw=1.5)

plt.tight_layout()
plt.savefig('fig_residual_diagnostics.png')
plt.show()
