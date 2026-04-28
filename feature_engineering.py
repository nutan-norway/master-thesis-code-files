
# Code for feature engineering of NO1 electricity price dataset
# Developed by [Nutan Gupta & Mathias Helseth] — [28/04/2026]
# University of Inland Norway
# All rights reserved. For academic use only.

# Feature Engineering
# Norwegian Electricity Price Forecasting | Region NO1

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import pearsonr, spearmanr

plt.rcParams.update({
    'figure.facecolor': 'white',  'axes.facecolor': '#f8f9fa',
    'axes.grid': True,            'grid.alpha': 0.35,
    'font.size': 11,              'axes.spines.top': False,
    'axes.spines.right': False,   'axes.labelsize': 12,
    'axes.titlesize': 13,         'figure.dpi': 110,
    'savefig.dpi': 150,           'savefig.bbox': 'tight',
})

COLORS = {
    'pre':    '#2196F3', 'crisis': '#F44336', 'post': '#4CAF50',
    'hydro':  '#00BCD4', 'gas':    '#FF9800', 'fwd':  '#9C27B0',
    'short':  '#1976D2', 'medium': '#F57C00', 'long': '#388E3C',
}
CRISIS_START = pd.Timestamp('2021-07-01')
CRISIS_END   = pd.Timestamp('2023-01-01')
TRAIN_END    = pd.Timestamp('2021-12-31')   # end of training window
VAL_START    = pd.Timestamp('2022-01-01')   # validation = crisis stress-test
VAL_END      = pd.Timestamp('2022-12-31')
TEST_START   = pd.Timestamp('2023-01-01')
TEST_END     = pd.Timestamp('2024-12-31')

MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
print("Setup complete.")


#  Load Data and Define Train/Val/Test Split
DATA_PATH = 'master_clean_NO1.csv'   

df = pd.read_csv(DATA_PATH, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

# Add regime and split labels
def get_regime(d):
    if d < CRISIS_START:   return 'Pre-Crisis (2018-mid2021)'
    elif d < CRISIS_END:   return 'Crisis (mid2021-2022)'
    return 'Post-Crisis (2023-2025)'

def get_split(d):
    if d <= TRAIN_END:                        return 'train'
    elif VAL_START <= d <= VAL_END:           return 'val'
    elif TEST_START <= d <= TEST_END:         return 'test'
    return 'holdout'

df['regime'] = df['date'].apply(get_regime)
df['split']  = df['date'].apply(get_split)
df['year']   = df['date'].dt.year
df['month']  = df['date'].dt.month
df['dow']    = df['date'].dt.dayofweek
df['iso_week'] = df['date'].dt.isocalendar().week.astype(int)

# Split counts
split_counts = df['split'].value_counts()
print("Train / Validation / Test / Holdout Split:")
print("="*50)
for split_name, dates in [
    ('TRAIN',   (df['date'].min(), TRAIN_END)),
    ('VAL',     (VAL_START, VAL_END)),
    ('TEST',    (TEST_START, TEST_END)),
    ('HOLDOUT', (df['date'][df['split']=='holdout'].min() if (df['split']=='holdout').any() else None,
                 df['date'].max())),
]:
    n = (df['split'] == split_name.lower()).sum()
    if dates[0] is not None:
        print(f"  {split_name:<8}: {str(dates[0])[:10]} to {str(dates[1])[:10]}  ({n} days)")
print("="*50)
print()

fig, ax = plt.subplots(figsize=(15, 5))

split_colors = {
    'train':   '#2196F3',
    'val':     '#F44336',
    'test':    '#4CAF50',
    'holdout': '#9C27B0',
}

for split_name, color in split_colors.items():
    mask = df['split'] == split_name
    if mask.any():
        ax.fill_between(df.loc[mask, 'date'], df.loc[mask, 'spot_price'],
                        alpha=0.55, color=color, label=split_name.upper())
        ax.plot(df.loc[mask, 'date'], df.loc[mask, 'spot_price'],
                color=color, lw=0.6, alpha=0.8)

ax.axvline(TRAIN_END,  color='black', ls='--', lw=1.5, alpha=0.8)
ax.axvline(VAL_END,    color='black', ls='--', lw=1.5, alpha=0.8)
ax.axvline(TEST_END,   color='black', ls='--', lw=1.5, alpha=0.8)
ax.text(pd.Timestamp('2020-01-01'), 580, 'TRAIN', fontsize=11,
        fontweight='bold', color='#1565C0')
ax.text(pd.Timestamp('2022-02-01'), 580, 'VAL\n(stress)', fontsize=11,
        fontweight='bold', color='#B71C1C')
ax.text(pd.Timestamp('2023-04-01'), 580, 'TEST', fontsize=11,
        fontweight='bold', color='#1B5E20')
ax.text(pd.Timestamp('2025-02-01'), 580, 'HOLDOUT', fontsize=9,
        fontweight='bold', color='#4A148C')

ax.set_title('Figure FE1 — Train / Validation / Test / Holdout Split\n'
             '(Blue=Train, Red=Validation/Crisis, Green=Test, Purple=Holdout)',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Spot Price (EUR/MWh)')
ax.legend(fontsize=9, ncol=4)

plt.savefig('fig_fe1_data_split.png')
plt.show()

print("Figure FE1 saved as fig_fe1_data_split.png")
print()


# Seasonal Profiles (The Medium/Long-Term Regressor Solution)
def build_seasonal_profile(df, col, train_end=TRAIN_END):
    """
    Compute seasonal median and std by (month, iso_week)
    using ONLY training data (data before train_end).
    
    This is the Norwegian analog of the GAM-based RES/load forecasts
    in Ghelasi & Ziel (2025) — we use the seasonal mean as a proxy
    for the expected value at medium/long horizons, avoiding lookahead bias.
    
    Parameters
    ----------
    df       : DataFrame with 'date', 'month', 'iso_week', and col columns
    col      : Column name to compute seasonal profile for
    train_end: Only use data up to this date (no lookahead)
    
    Returns
    -------
    DataFrame with columns: month, iso_week, seasonal_mean, seasonal_std,
                            seasonal_median, seasonal_p10, seasonal_p90
    """
    train = df[df['date'] <= train_end].copy()
    profile = (
        train.groupby(['month', 'iso_week'])[col]
        .agg(
            seasonal_mean   = 'mean',
            seasonal_median = 'median',
            seasonal_std    = 'std',
            seasonal_p10    = lambda x: np.percentile(x, 10),
            seasonal_p90    = lambda x: np.percentile(x, 90),
            n_obs           = 'count',
        )
        .reset_index()
    )
    # Smooth with 3-period rolling window to reduce noise
    for col_s in ['seasonal_mean', 'seasonal_median', 'seasonal_std']:
        profile[col_s] = profile[col_s].rolling(3, center=True, min_periods=1).mean()
    return profile


# Build seasonal profiles for key variables
print("Building seasonal profiles from TRAINING DATA ONLY")
print(f"(using data up to {TRAIN_END.date()} to avoid lookahead bias)")
print()

cols_to_profile = {
    'spot_price':             'Spot Price (EUR/MWh)',
    'hydro_reserve_gwh':      'Hydro Reserve (GWh)',
    'inflow_hbv_gwh':         'Inflow HBV (GWh/week)',
    'reservoir_deviation_gwh':'Reservoir Deviation (GWh)',
    'fuel_gas_eur_mwh':       'Gas Price (EUR/MWh)',
    'load_mw':                'Load (MW)',
}

seasonal_profiles = {}
for col, label in cols_to_profile.items():
    if col not in df.columns:
        print(f"  SKIP: {col} not in dataset")
        continue
    profile = build_seasonal_profile(df, col)
    seasonal_profiles[col] = profile
    n_weeks = len(profile)
    print(f"  {label:<30}  {n_weeks} (month, week) combinations built")

print()
print(f"Total profiles built: {len(seasonal_profiles)}")
print()
print("These profiles will be used as feature values at medium/long horizons")
print("because the actual future values are not available at forecast time.")


fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Figure FE2 — Seasonal Profiles (Built from Training Data Only)\n'
             'These replace actual values as regressors at medium/long horizons',
             fontsize=13, fontweight='bold')

plot_configs = [
    ('spot_price',             'Spot Price (EUR/MWh)',     COLORS['crisis']),
    ('hydro_reserve_gwh',      'Hydro Reserve (GWh)',      COLORS['hydro']),
    ('inflow_hbv_gwh',         'Inflow HBV (GWh/week)',    COLORS['pre']),
    ('reservoir_deviation_gwh','Reservoir Deviation (GWh)',COLORS['gas']),
    ('fuel_gas_eur_mwh',       'Gas Price (EUR/MWh)',      COLORS['gas']),
    ('load_mw',                'Load (MW)',                COLORS['fwd']),
]

for ax, (col, ylabel, color) in zip(axes.flat, plot_configs):
    if col not in seasonal_profiles:
        ax.set_visible(False)
        continue
    prof = seasonal_profiles[col]
    # Average by month for a clean monthly view
    monthly = prof.groupby('month').agg(
        mean   = ('seasonal_mean',   'mean'),
        std_lo = ('seasonal_mean',   lambda x: x.mean() - x.std()),
        std_hi = ('seasonal_mean',   lambda x: x.mean() + x.std()),
    ).reset_index()
    ax.fill_between(monthly['month'], monthly['std_lo'], monthly['std_hi'],
                    alpha=0.22, color=color)
    ax.plot(monthly['month'], monthly['mean'], 'o-', color=color, lw=2, ms=6)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([m[0] for m in zip(MONTHS)], fontsize=8)
    ax.set_title(ylabel, fontsize=10)
    ax.set_xlabel('Month')

plt.tight_layout()
plt.savefig('fig_fe2_seasonal_profiles.png')
plt.show()


def attach_seasonal_features(df, seasonal_profiles):
    """
    Join seasonal profile statistics onto the full DataFrame.
    Adds columns: seasonal_{col}_mean, seasonal_{col}_std,
                  seasonal_{col}_p10, seasonal_{col}_p90
    for each variable in seasonal_profiles.
    
    IMPORTANT: These seasonal values are computed from training data only
    (no lookahead bias). They can be used as regressors at ALL horizons
    because they represent 'what is the typical value at this time of year'
    — information that is always available at forecast time.
    """
    df = df.copy()
    for col, profile in seasonal_profiles.items():
        merged = df[['month', 'iso_week']].merge(
            profile[['month', 'iso_week',
                      'seasonal_mean', 'seasonal_std',
                      'seasonal_p10', 'seasonal_p90']],
            on=['month', 'iso_week'],
            how='left',
        )
        df[f'seasonal_{col}_mean'] = merged['seasonal_mean'].values
        df[f'seasonal_{col}_std']  = merged['seasonal_std'].values
        df[f'seasonal_{col}_p10']  = merged['seasonal_p10'].values
        df[f'seasonal_{col}_p90']  = merged['seasonal_p90'].values
    return df

df = attach_seasonal_features(df, seasonal_profiles)

# Show what was added
new_cols = [c for c in df.columns if c.startswith('seasonal_')]
print(f"Added {len(new_cols)} seasonal feature columns:")
for c in new_cols:
    non_null = df[c].notna().sum()
    print(f"  {c:<45}  {non_null} non-null values")
print()
print("Key derived features from seasonal profiles:")
print()
# Price vs seasonal mean (the 'current model' deviation signal)
df['price_vs_seasonal']  = df['spot_price'] - df['seasonal_spot_price_mean']
df['price_seasonal_z']   = df['price_vs_seasonal'] / df['seasonal_spot_price_std'].clip(lower=1)
df['hydro_seasonal_z']   = ((df['hydro_reserve_gwh'] - df['seasonal_hydro_reserve_gwh_mean'])
                             / df['seasonal_hydro_reserve_gwh_std'].clip(lower=1))
df['inflow_seasonal_z']  = ((df['inflow_hbv_gwh'] - df['seasonal_inflow_hbv_gwh_mean'])
                             / df['seasonal_inflow_hbv_gwh_std'].clip(lower=1))

print("  price_vs_seasonal : current spot minus seasonal expectation")
print(f"    range: {df['price_vs_seasonal'].min():.1f} to {df['price_vs_seasonal'].max():.1f} EUR/MWh")
print()
print("  price_seasonal_z  : standardised deviation from seasonal mean")
print(f"    range: {df['price_seasonal_z'].min():.2f} to {df['price_seasonal_z'].max():.2f} sigma")
print()
print("  hydro_seasonal_z  : how far current reservoir is from seasonal normal")
print(f"    range: {df['hydro_seasonal_z'].min():.2f} to {df['hydro_seasonal_z'].max():.2f} sigma")


# Engineer Short-Term Features
# At short horizons (h ≤ 30 days) all features are valid and available.

def engineer_short_term(df):
    """
    SHORT-TERM feature engineering (h <= 30 days).
    
    At short horizons ALL features are valid:
      - Autoregressive lags (ACF still > 0.6 at h=30d)
      - Forward prices (M1, Y1)
      - Actual hydro levels (known at forecast time via data feeds)
      - Weather forecasts (1-day-ahead available from met services)
      - Fuel prices (current front-month futures)
    
    Additional engineered features:
      - Rolling price statistics at multiple windows
      - Hydro-gas interaction (key Norwegian non-linearity)
      - Forward term structure (contango/backwardation signal)
      - Price momentum (deviation from recent average)
    """
    df = df.copy()
    
    # ── Rolling price statistics (shift(1) to avoid lookahead) ───────────
    for w in [7, 14, 30, 90]:
        df[f'roll_price_mean_{w}d'] = df['spot_price'].shift(1).rolling(w).mean()
        df[f'roll_price_std_{w}d']  = df['spot_price'].shift(1).rolling(w).std()
    
    # ── Rolling hydro and gas stats ───────────────────────────────────────
    for w in [7, 30]:
        df[f'roll_hydro_mean_{w}d'] = df['hydro_reserve_gwh'].shift(1).rolling(w).mean()
        df[f'roll_gas_mean_{w}d']   = df['fuel_gas_eur_mwh'].shift(1).rolling(w).mean()
    
    # ── Price momentum (current price vs rolling mean) ────────────────────
    for w in [7, 30]:
        df[f'price_momentum_{w}d'] = df['lag_price_1d'] - df[f'roll_price_mean_{w}d']
    
    # ── Hydro-gas interaction (key Norwegian non-linearity) ───────────────
    # When hydro is LOW and gas is HIGH => extreme price spikes
    # Capture this interaction as a product of Z-scores
    hydro_z = ((df['hydro_reserve_gwh'] - df['hydro_reserve_gwh'].mean())
                / df['hydro_reserve_gwh'].std())
    gas_z   = ((df['fuel_gas_eur_mwh'] - df['fuel_gas_eur_mwh'].mean())
                / df['fuel_gas_eur_mwh'].std())
    df['interaction_hydro_gas'] = hydro_z * gas_z
    df['ratio_gas_to_hydro']    = (df['fuel_gas_eur_mwh']
                                   / (df['hydro_reserve_gwh'].clip(lower=1000) / 1000))
    
    # ── Reservoir fill rate ───────────────────────────────────────────────
    df['hydro_fill_rate'] = (df['hydro_reserve_gwh'].diff()
                              / df['inflow_hbv_gwh'].clip(lower=1))
    df['hydro_fill_rate'] = df['hydro_fill_rate'].clip(-5, 5)
    
    # ── Snow fraction (snow inflow / total inflow) ────────────────────────
    df['snow_fraction'] = df['inflow_snow_gwh'] / df['inflow_hbv_gwh'].clip(lower=1)
    
    # ── Forward market features ───────────────────────────────────────────
    df['epad_slope']      = df['epad_M1'] - df['epad_Y1']   # EPAD term structure
    df['implied_spot_M1'] = df['forward_M1'] + df['epad_M1']  # implied NO1 price
    df['implied_spot_Y1'] = df['forward_Y1'] + df['epad_Y1']
    df['spot_fwd_basis']  = df['spot_price'] - df['implied_spot_M1']
    df['gas_co2_ratio']   = df['fuel_gas_eur_mwh'] / df['fuel_carbon_eur_t'].clip(lower=1)
    
    # ── Calendar dummies ──────────────────────────────────────────────────
    m = df['date'].dt.month
    df['season_winter'] = ((m <= 2) | (m == 12)).astype(int)
    df['season_spring'] = ((m >= 3) & (m <= 5)).astype(int)
    df['season_summer'] = ((m >= 6) & (m <= 8)).astype(int)
    df['season_autumn'] = ((m >= 9) & (m <= 11)).astype(int)
    df['is_weekend']    = (df['date'].dt.dayofweek >= 5).astype(int)
    df['month_num']     = df['date'].dt.month
    df['week_of_year']  = df['date'].dt.isocalendar().week.astype(int)
    
    return df

df = engineer_short_term(df)

new_feats = [
    'roll_price_mean_7d', 'roll_price_std_7d', 'roll_price_mean_30d',
    'roll_hydro_mean_30d', 'roll_gas_mean_30d',
    'price_momentum_7d', 'price_momentum_30d',
    'interaction_hydro_gas', 'ratio_gas_to_hydro',
    'hydro_fill_rate', 'snow_fraction',
    'epad_slope', 'implied_spot_M1', 'implied_spot_Y1', 'spot_fwd_basis',
    'gas_co2_ratio', 'season_winter', 'is_weekend',
]
print("Short-term engineered features added:")
for f in new_feats:
    if f in df.columns:
        print(f"  {f:<30}  min={df[f].min():.2f}  max={df[f].max():.2f}")
print()
print(f"Total columns now: {df.shape[1]}")


#  Define Horizon-Specific Feature Sets
FEATURES_SHORT = [
    # Autoregressive (valid at short horizon — ACF > 0.6 at h=30d)
    'lag_price_1d', 'lag_price_7d', 'lag_price_30d', 'lag_price_365d',
    'ma_price_7d',  'ma_price_30d',
    'std_price_7d', 'std_price_30d',
    # Rolling statistics
    'roll_price_mean_7d',  'roll_price_std_7d',
    'roll_price_mean_14d', 'roll_price_std_14d',
    'roll_price_mean_30d', 'roll_price_std_30d',
    'roll_price_mean_90d', 'roll_price_std_90d',
    'roll_hydro_mean_7d',  'roll_hydro_mean_30d',
    'roll_gas_mean_7d',    'roll_gas_mean_30d',
    'price_momentum_7d',   'price_momentum_30d',
    # Hydro fundamentals (actual, available at forecast time)
    'hydro_reserve_gwh', 'inflow_hbv_gwh', 'inflow_snow_gwh',
    'reservoir_deviation_gwh', 'derived_inflow_z_score',
    'derived_hydro_change', 'hydro_fill_rate', 'snow_fraction',
    # Interactions
    'interaction_hydro_gas', 'ratio_gas_to_hydro',
    # Fuel prices (current front-month)
    'fuel_gas_eur_mwh', 'fuel_carbon_eur_t', 'fuel_brent_usd_bbl',
    'derived_gas_volatility_30d', 'gas_co2_ratio',
    # Forward / market expectations
    'forward_M1', 'forward_Y1',
    'epad_M1',    'epad_Y1',
    'derived_forward_slope', 'epad_slope',
    'implied_spot_M1', 'implied_spot_Y1', 'spot_fwd_basis',
    # Weather (1-day-ahead forecast available)
    'weather_temp_c', 'weather_wind_ms', 'weather_precip_mm',
    # Load and macro
    'load_mw', 'macro_eur_nok',
    # Calendar
    'cal_month_sin', 'cal_month_cos',
    'cal_day_of_year_sin', 'cal_day_of_year_cos',
    'season_winter', 'season_spring', 'season_summer', 'season_autumn',
    'is_weekend', 'month_num', 'week_of_year',
]

FEATURES_MEDIUM = [
    # Seasonal expectations (REPLACE raw AR lags at medium horizon)
    'seasonal_spot_price_mean',  'seasonal_spot_price_std',
    'seasonal_spot_price_p10',   'seasonal_spot_price_p90',
    'price_vs_seasonal',         'price_seasonal_z',
    # Hydro seasonal signals
    'seasonal_hydro_reserve_gwh_mean', 'seasonal_hydro_reserve_gwh_std',
    'seasonal_hydro_reserve_gwh_p10',  'seasonal_hydro_reserve_gwh_p90',
    'seasonal_inflow_hbv_gwh_mean',    'seasonal_inflow_hbv_gwh_std',
    # Current hydro levels (still known at forecast time)
    'hydro_reserve_gwh', 'reservoir_deviation_gwh',
    'derived_inflow_z_score', 'inflow_hbv_gwh',
    'hydro_seasonal_z', 'inflow_seasonal_z',
    # Forward market (still informative at 1-6 month horizon)
    'forward_M1', 'forward_Y1',
    'epad_M1',    'epad_Y1',
    'derived_forward_slope', 'epad_slope',
    'implied_spot_M1', 'implied_spot_Y1',
    # Fuel (front-month futures — informative up to ~6 months)
    'fuel_gas_eur_mwh', 'fuel_carbon_eur_t',
    'derived_gas_volatility_30d',
    'roll_gas_mean_30d',
    # Interaction
    'interaction_hydro_gas',
    # Macro
    'macro_eur_nok',
    # Calendar (now more important as horizon grows)
    'cal_month_sin', 'cal_month_cos',
    'cal_day_of_year_sin', 'cal_day_of_year_cos',
    'season_winter', 'season_spring', 'season_summer', 'season_autumn',
    'month_num', 'week_of_year',
    # Weak long-memory signal (flagged — may still help)
    'lag_price_365d',
    # NOTE: lag_price_1d, lag_price_7d, ma_price_7d are EXCLUDED
    #       (ACF < 0.3 at h=90d, unit-root spurious regression risk)
]

FEATURES_LONG = [
    # Forward Y+1 only (matched to long horizon, per Paper 3 finding)
    'forward_Y1', 'epad_Y1', 'implied_spot_Y1',
    # Seasonal hydro expectations (what we expect at this time of year)
    'seasonal_hydro_reserve_gwh_mean', 'seasonal_hydro_reserve_gwh_std',
    'seasonal_inflow_hbv_gwh_mean',
    # Seasonal price expectations
    'seasonal_spot_price_mean', 'seasonal_spot_price_std',
    'seasonal_spot_price_p10',  'seasonal_spot_price_p90',
    # Macro (long-run structural)
    'macro_eur_nok',
    # Calendar (primary stabilising signal at this horizon)
    'cal_month_sin', 'cal_month_cos',
    'cal_day_of_year_sin', 'cal_day_of_year_cos',
    'season_winter', 'season_spring', 'season_summer', 'season_autumn',
    'month_num', 'week_of_year',
    # NOTE: ALL autoregressive features EXCLUDED
    #       forward_M1 EXCLUDED (too short horizon, becomes spurious)
    #       fuel_gas, fuel_carbon EXCLUDED (unit root, spurious at 6-12mo)
]

# Filter to only columns that actually exist in df
FEATURES_SHORT  = [f for f in FEATURES_SHORT  if f in df.columns]
FEATURES_MEDIUM = [f for f in FEATURES_MEDIUM if f in df.columns]
FEATURES_LONG   = [f for f in FEATURES_LONG   if f in df.columns]

print("Horizon-Specific Feature Sets:")
print("="*60)
print(f"  SHORT-TERM  (h <= 30 days)  :  {len(FEATURES_SHORT):>3} features")
print(f"  MEDIUM-TERM (h = 1-6 months):  {len(FEATURES_MEDIUM):>3} features")
print(f"  LONG-TERM   (h = 6-12 months): {len(FEATURES_LONG):>3} features")
print("="*60)
print()
print("Key features EXCLUDED at each tier:")
print()
print("MEDIUM excludes:")
excluded_med = set(FEATURES_SHORT) - set(FEATURES_MEDIUM)
for f in sorted(excluded_med)[:15]:
    print(f"  - {f}")
print()
print("LONG additionally excludes:")
excluded_long = set(FEATURES_MEDIUM) - set(FEATURES_LONG)
for f in sorted(excluded_long):
    print(f"  - {f}")



ALL_HORIZONS = [1, 7, 14, 30, 60, 90, 180, 360]

# Add forward-shifted target columns (future spot price at each horizon)
for h in ALL_HORIZONS:
    df[f'target_h{h}'] = df['spot_price'].shift(-h)

target_cols = [f'target_h{h}' for h in ALL_HORIZONS]
meta_cols   = ['date', 'spot_price', 'regime', 'split',
               'year', 'month', 'iso_week', 'dow']

# Build final feature matrices for each horizon tier
def build_matrix(df, feature_cols, label):
    keep = meta_cols + feature_cols + target_cols
    keep = [c for c in keep if c in df.columns]
    mat = df[keep].copy()
    # Drop rows where ALL targets are NaN (end of series)
    mat = mat[mat[target_cols].notna().any(axis=1)]
    print(f"  {label:<18}: {mat.shape[0]} rows x {mat.shape[1]} columns")
    return mat

print("Building feature matrices...")
print()
df_short  = build_matrix(df, FEATURES_SHORT,  'Short-term')
df_medium = build_matrix(df, FEATURES_MEDIUM, 'Medium-term')
df_long   = build_matrix(df, FEATURES_LONG,   'Long-term')

print()
print("Checking for missing values in each matrix:")
for name, mat, feats in [
    ('Short',  df_short,  FEATURES_SHORT),
    ('Medium', df_medium, FEATURES_MEDIUM),
    ('Long',   df_long,   FEATURES_LONG),
]:
    feat_missing = mat[feats].isnull().sum()
    n_missing    = (feat_missing > 0).sum()
    print(f"  {name}: {n_missing} features with missing values")
    if n_missing > 0:
        print(feat_missing[feat_missing > 0].to_string())

print()


#  Feature Importance Analysis
def compute_importance(df, feature_cols, target='spot_price'):
    """
    Compute Spearman correlation of each feature with the target.
    Spearman is preferred over Pearson because:
      - It is rank-based and more robust to outliers (the extreme 2022 prices)
      - It captures monotone non-linear relationships
    """
    results = []
    y = df[target].dropna()
    for col in feature_cols:
        if col not in df.columns:
            continue
        x     = df.loc[y.index, col]
        valid = ~(x.isna() | y.isna())
        if valid.sum() < 10:
            continue
        rho, p = spearmanr(x[valid], y[valid])
        results.append({
            'feature':   col,
            'rho':       round(rho,  4),
            'abs_rho':   round(abs(rho), 4),
            'p_value':   round(p, 4),
            'direction': 'positive' if rho > 0 else 'negative',
        })
    out = (pd.DataFrame(results)
             .sort_values('abs_rho', ascending=False)
             .reset_index(drop=True))
    out.index += 1   # rank starts at 1
    return out

print("Computing feature importance (Spearman rho with spot_price)...")
print()

imp_short  = compute_importance(df_short,  FEATURES_SHORT)
imp_medium = compute_importance(df_medium, FEATURES_MEDIUM)
imp_long   = compute_importance(df_long,   FEATURES_LONG)

print("TOP 15 — SHORT-TERM features:")
print(imp_short[['feature','rho','abs_rho','direction']].head(15).to_string())
print()
print("TOP 15 — MEDIUM-TERM features:")
print(imp_medium[['feature','rho','abs_rho','direction']].head(15).to_string())
print()
print("TOP 15 — LONG-TERM features:")
print(imp_long[['feature','rho','abs_rho','direction']].head(15).to_string())



fig, axes = plt.subplots(3, 1, figsize=(10, 16))
fig.suptitle('Feature Importance by Horizon Tier\n'
             '(Spearman rho with spot_price)',
             fontsize=13, fontweight='bold')

tier_info = [
    (axes[0], imp_short,  'SHORT-TERM (h <= 30d)',    COLORS['short'],  20),
    (axes[1], imp_medium, 'MEDIUM-TERM (h = 1-6mo)',  COLORS['medium'], 20),
    (axes[2], imp_long,   'LONG-TERM (h = 6-12mo)',   COLORS['long'],   15),
]

for ax, imp_df, title, base_color, n_top in tier_info:
    top = imp_df.head(n_top).sort_values('rho')   # ascending so highest is at top
    bar_colors = [COLORS['crisis'] if v < 0 else base_color for v in top['rho']]
    bars = ax.barh(range(len(top)), top['rho'], color=bar_colors, alpha=0.82)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['feature'], fontsize=7.5)
    ax.axvline(0, color='k', lw=1)
    ax.set_xlabel('Spearman rho')
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlim(-1, 1)
    # Add value labels
    for bar, val in zip(bars, top['rho']):
        offset = 0.02 if val >= 0 else -0.02
        ha = 'left' if val >= 0 else 'right'
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', ha=ha, va='center', fontsize=7)

plt.tight_layout()
plt.savefig('fig_fe3_feature_importance.png')
plt.show()



#  Hydro-Gas Interaction Visualisation
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr

fig = plt.figure(figsize=(13, 10), constrained_layout=False)

fig.suptitle(
    'Hydro-Gas Interaction: Norwegian Non-Linearity\n',
    fontsize=14,
    fontweight='bold',
    y=0.97
)

# Top row only: 2 columns, normal spacing
gs = gridspec.GridSpec(
    1, 2,
    figure=fig,
    left=0.07, right=0.95, top=0.89, bottom=0.57,
    wspace=0.28
)

ax  = fig.add_subplot(gs[0, 0])   # Panel A
ax2 = fig.add_subplot(gs[0, 1])   # Panel B

# Panel C manually centered below with SAME width/height as A/B
posA = ax.get_position()
panel_width  = posA.width
panel_height = posA.height

x_center = 0.5 - panel_width / 2
y_bottom = 0.18   # adjust if needed

ax3 = fig.add_axes([x_center, y_bottom, panel_width, panel_height])

# -------------------------
# Panel A: Gas x Hydro => Spot Price
# -------------------------
sc = ax.scatter(
    df['fuel_gas_eur_mwh'],
    df['hydro_reserve_gwh'] / 1000,
    c=df['spot_price'],
    cmap='RdYlGn_r',
    s=10,
    alpha=0.6,
    vmin=0,
    vmax=300
)

ax.axhline(
    df['hydro_reserve_gwh'].quantile(0.2) / 1000,
    color='red', ls='--', lw=1.2, label='P20 hydro'
)
ax.axvline(
    df['fuel_gas_eur_mwh'].quantile(0.8),
    color='red', ls=':', lw=1.2, label='P80 gas'
)

ax.set_xlabel('Gas Price (EUR/MWh)')
ax.set_ylabel('Hydro Reserve (TWh)')
ax.set_title('Panel A — Gas × Hydro and Spot Price', fontsize=12, fontweight='bold')
ax.legend(fontsize=8, loc='upper right')

cbar1 = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
cbar1.set_label('Spot Price (EUR/MWh)')

# -------------------------
# Panel B: Interaction feature vs Spot
# -------------------------
valid = df[['interaction_hydro_gas', 'spot_price']].dropna()
rho_int, _ = spearmanr(valid['interaction_hydro_gas'], valid['spot_price'])

sc2 = ax2.scatter(
    df['interaction_hydro_gas'],
    df['spot_price'],
    c=df['derived_gas_volatility_30d'],
    cmap='Reds',
    s=10,
    alpha=0.55
)

ax2.axvline(0, color='gray', ls='--', lw=1)

ax2.set_xlabel('Interaction Feature (Z hydro × Z gas)')
ax2.set_ylabel('Spot Price (EUR/MWh)')
ax2.set_title(
    f'Panel B — Interaction Feature vs Spot\nSpearman rho = {rho_int:.3f}',
    fontsize=12,
    fontweight='bold'
)

ax2.text(
    0.03, 0.97,
    'Negative values = high hydro\n(low price expected)',
    transform=ax2.transAxes,
    fontsize=8,
    va='top',
    bbox=dict(fc='white', ec='gray', alpha=0.85)
)
ax2.text(
    0.62, 0.97,
    'Positive = low hydro\n(high price expected)',
    transform=ax2.transAxes,
    fontsize=8,
    va='top',
    bbox=dict(fc='white', ec='gray', alpha=0.85)
)

cbar2 = fig.colorbar(sc2, ax=ax2, fraction=0.04, pad=0.02)
cbar2.set_label('Gas Volatility (30d)')

# -------------------------
# Panel C: Hydro deviation from seasonal normal by regime
# -------------------------
regime_colors = {
    'Pre-Crisis (2018-mid2021)': COLORS['pre'],
    'Crisis (mid2021-2022)': COLORS['crisis'],
    'Post-Crisis (2023-2025)': COLORS['post'],
}

for regime, grp in df.groupby('regime', sort=False):
    if 'hydro_seasonal_z' not in grp.columns:
        continue

    ax3.scatter(
        grp['hydro_seasonal_z'],
        grp['spot_price'],
        color=regime_colors.get(regime, 'gray'),
        alpha=0.35,
        s=10,
        label=regime.split('(')[0].strip()
    )

ax3.axvline(0, color='gray', ls='--', lw=1, label='Seasonal normal')
ax3.set_xlabel('Hydro Z-score (sigma from seasonal mean)')
ax3.set_ylabel('Spot Price (EUR/MWh)')
ax3.set_title('Panel C — Hydro Deviation from Seasonal Normal', fontsize=12, fontweight='bold')
ax3.legend(fontsize=8, loc='upper right')

for a in [ax, ax2, ax3]:
    a.grid(True, alpha=0.25)

plt.savefig('fig_fe4_hydro_gas_interaction.png', dpi=300, bbox_inches='tight')
plt.show()



fig, ax = plt.subplots(figsize=(14, 7))
fig.suptitle('Engineered Feature Relevance Decay\n',
             fontsize=13, fontweight='bold')

horizons = [1, 7, 14, 30, 60, 90, 120, 180, 270, 360]

# Select the most representative features from each tier
engineered_features = {
    # Short-tier leaders
    'lag_price_1d':           (COLORS['crisis'],  '--'),
    'implied_spot_M1':        (COLORS['pre'],     '-'),
    'interaction_hydro_gas':  (COLORS['hydro'],   '-'),
    'price_vs_seasonal':      (COLORS['gas'],     '-'),
    # Medium-tier leaders
    'forward_M1':             ('#9C27B0',          '-'),
    'forward_Y1':             ('#4CAF50',          '-'),
    'hydro_seasonal_z':       ('#00BCD4',          '-'),
    # Long-tier leaders
    'implied_spot_Y1':        ('#795548',          '-'),
    'seasonal_spot_price_mean':('#E91E63',         ':'),
}

for feat, (color, ls) in engineered_features.items():
    if feat not in df.columns:
        continue
    corrs = []
    for h in horizons:
        if h >= len(df):
            corrs.append(np.nan)
            continue
        shifted = df['spot_price'].shift(-h)
        valid   = ~(shifted.isna() | df[feat].isna())
        if valid.sum() < 20:
            corrs.append(np.nan)
            continue
        rho, _ = spearmanr(df.loc[valid, feat], shifted[valid])
        corrs.append(abs(rho))
    ax.plot(horizons, corrs, 'o-', lw=2, ms=5, ls=ls,
            color=color, label=feat)

# Shade the three horizon zones
ax.axvspan(0,   30,  alpha=0.06, color='blue',   label='Short zone')
ax.axvspan(30,  180, alpha=0.06, color='orange', label='Medium zone')
ax.axvspan(180, 370, alpha=0.06, color='green',  label='Long zone')
ax.axvline(30,  color='gray', ls=':', lw=1.2)
ax.axvline(180, color='gray', ls=':', lw=1.2)
ax.text(15,  0.02, 'SHORT',  fontsize=9, ha='center', color='navy')
ax.text(100, 0.02, 'MEDIUM', fontsize=9, ha='center', color='darkorange')
ax.text(270, 0.02, 'LONG',   fontsize=9, ha='center', color='darkgreen')
ax.axhline(0.3, color='green',  ls='--', lw=1.0, alpha=0.7, label='0.3 threshold')
ax.axhline(0.1, color='orange', ls='--', lw=1.0, alpha=0.7, label='0.1 threshold')

ax.set_xlabel('Forecasting Horizon (days)')
ax.set_ylabel('|Spearman rho| with future spot price')
ax.legend(fontsize=8, ncol=3, loc='upper right')
ax.set_ylim(0, 1)
ax.set_xticks(horizons)
ax.set_xticklabels([f'{h}d' for h in horizons])

plt.tight_layout()
plt.savefig('fig_fe5_engineered_decay.png')
plt.show()


#  Coefficient Bounds from Merit-Order Theory

import matplotlib.patches as mpatches
COEF_BOUNDS = {
    # Continental fuel linkages (via interconnectors to DK/DE/NL/UK)
    'fuel_gas_eur_mwh':        (0,      4.00),   # max heat rate = 1/eta_gas = 1/0.25
    'fuel_carbon_eur_t':       (0,      1.33),   # max eps/eta  = 0.4/0.3 (lignite)
    'fuel_brent_usd_bbl':      (0,      0.588),  # max oil heat rate

    # Norwegian hydro fundamentals (NEGATIVE: more water => lower price)
    'hydro_reserve_gwh':       (None,   0),
    'inflow_hbv_gwh':          (None,   0),
    'reservoir_deviation_gwh': (None,   0),
    'derived_inflow_z_score':  (None,   0),
    'hydro_seasonal_z':        (None,   0),
    'inflow_seasonal_z':       (None,   0),

    # Demand (POSITIVE: higher demand => higher price)
    'load_mw':                 (0,      None),

    # Forward prices (POSITIVE contribution to expected price)
    'forward_M1':              (0,      None),
    'forward_Y1':              (0,      None),
    'epad_M1':                 (0,      None),   # area premium >= 0
    'epad_Y1':                 (0,      None),
    'implied_spot_M1':         (0,      None),
    'implied_spot_Y1':         (0,      None),

    # Autoregressive (POSITIVE: price persistence)
    'lag_price_1d':            (0,      None),
    'lag_price_7d':            (0,      None),
    'lag_price_30d':           (0,      None),
    'lag_price_365d':          (0,      None),
    'price_vs_seasonal':       (0,      None),   # deviation above seasonal => higher price
}

# Display the constraint table
print("Coefficient Bounds — Norwegian Hydro Merit-Order Model")
print("(Analogous to Table 3 in Ghelasi & Ziel 2025)")
print()
print(f"{'Feature':<32}  {'Lower':<10}  {'Upper':<10}  {'Reasoning'}")
print("-"*75)
bounds_explained = {
    'fuel_gas_eur_mwh':        '1/eta_gas (max heat rate = 4.0)',
    'fuel_carbon_eur_t':       'eps/eta_lignite = 0.4/0.3 = 1.33',
    'fuel_brent_usd_bbl':      '1/eta_oil * conv = 0.588',
    'hydro_reserve_gwh':       'More water => lower price (merit order)',
    'inflow_hbv_gwh':          'More inflow => lower future water value',
    'reservoir_deviation_gwh': 'Above seasonal normal => lower scarcity',
    'derived_inflow_z_score':  'Higher inflow z => more surplus',
    'load_mw':                 'Higher demand => higher equilibrium price',
    'forward_M1':              'Market expectation (positive contribution)',
    'forward_Y1':              'Market expectation (positive contribution)',
    'epad_M1':                 'Area premium >= 0 by definition',
    'lag_price_1d':            'Price persistence (positive autocorrelation)',
    'price_vs_seasonal':       'Above seasonal expectation => price boost',
}
for feat, (lo, hi) in COEF_BOUNDS.items():
    lo_str = str(lo)   if lo is not None else '-inf'
    hi_str = str(hi)   if hi is not None else '+inf'
    reason = bounds_explained.get(feat, '')
    print(f"  {feat:<30}  [{lo_str:<8}, {hi_str:<8}]  {reason}")

print()





