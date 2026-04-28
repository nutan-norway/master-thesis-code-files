# Code for EDA and feature analysis of NO1 electricity price dataset
# Developed by [Nutan Gupta & Mathias Helseth] — [28/04/2026]
# University of Inland Norway
# All rights reserved. For academic use only.

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import pearsonr, spearmanr, johnsonsu
from statsmodels.tsa.stattools import acf, pacf, adfuller

plt.rcParams.update({
    'figure.facecolor': 'white',  'axes.facecolor': '#f8f9fa',
    'axes.grid': True,            'grid.alpha': 0.35,
    'font.size': 11,              'axes.spines.top': False,
    'axes.spines.right': False,   'axes.labelsize': 12,
    'axes.titlesize': 13,         'figure.dpi': 110,
    'savefig.dpi': 150,           'savefig.bbox': 'tight',
})

COLORS = {
    'pre':    '#2196F3',   # blue  – pre-crisis
    'crisis': '#F44336',   # red   – crisis 2021-22
    'post':   '#4CAF50',   # green – post-crisis
    'hydro':  '#00BCD4',   'gas':  '#FF9800',   'fwd': '#9C27B0',
}
CRISIS_START = pd.Timestamp('2021-07-01')
CRISIS_END   = pd.Timestamp('2023-01-01')
MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

print("Setup complete. All libraries loaded.")

DATA_PATH = 'master_clean_NO1.csv'

df = pd.read_csv(DATA_PATH, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")
print(f"From    : {df['date'].min().date()}")
print(f"To      : {df['date'].max().date()}")
print(f"Missing : {df.isnull().sum().sum()} values  (0 = clean)")
print()

# Feature groups
groups = {
    'Target':         ['spot_price'],
    'Hydro':          ['hydro_reserve_gwh','inflow_hbv_gwh','inflow_snow_gwh',
                       'reservoir_deviation_gwh','derived_inflow_z_score','derived_hydro_change'],
    'Forwards/EPAD':  ['forward_M1','forward_Y1','epad_M1','epad_Y1','derived_forward_slope'],
    'Fuel Prices':    ['fuel_gas_eur_mwh','fuel_carbon_eur_t','fuel_brent_usd_bbl',
                       'derived_gas_volatility_30d'],
    'Weather':        ['weather_temp_c','weather_wind_ms','weather_precip_mm'],
    'Load/Macro':     ['load_mw','macro_eur_nok'],
    'AR Lags':        ['lag_price_1d','lag_price_7d','lag_price_30d','lag_price_365d',
                       'ma_price_7d','ma_price_30d','std_price_7d','std_price_30d'],
    'Calendar':       ['cal_month_sin','cal_month_cos','cal_day_of_year_sin','cal_day_of_year_cos'],
}
print("Feature groups:")
print("-"*40)
for name, cols in groups.items():
    n = len([c for c in cols if c in df.columns])
    print(f"  {name:<18}  {n} columns")
print("-"*40)
print(f"  {'TOTAL (features)':<18}  {df.shape[1]-2}")

print("Spot Price (EUR/MWh) — Summary Statistics:")
print("-"*45)
print(df['spot_price'].describe().round(2).to_string())
print()
print(f"Negative prices : {(df['spot_price'] < 0).sum()} days")
print(f"Prices > 300    : {(df['spot_price'] > 300).sum()} days  (extreme crisis spikes)")
print(f"Prices > 500    : {(df['spot_price'] > 500).sum()} days")
print()
print("Note: The wide range (0.9 to 660 EUR/MWh) means models must handle")
print("      very different price regimes. A single static model will struggle.")


#Price Time Series and Regime Analysis

def get_regime(d):
    if d < CRISIS_START:   return 'Pre-Crisis (2018-mid2021)'
    elif d < CRISIS_END:   return 'Crisis (mid2021-2022)'
    return 'Post-Crisis (2023-2025)'

df['regime'] = df['date'].apply(get_regime)
df['year']   = df['date'].dt.year
df['month']  = df['date'].dt.month

REGIME_COLORS = {
    'Pre-Crisis (2018-mid2021)': COLORS['pre'],
    'Crisis (mid2021-2022)':     COLORS['crisis'],
    'Post-Crisis (2023-2025)':   COLORS['post'],
}

# Annual statistics
annual = df.groupby('year')['spot_price'].agg(
    Mean='mean', Std='std', Min='min', Max='max',
    Neg_days=lambda x: (x < 0).sum()
).round(2)
print("Table 1 — Annual Spot Price Statistics (EUR/MWh):")
print("="*60)
print(annual.to_string())
print("="*60)
print()
print("Key observations:")
print("  2020: unusually LOW prices (COVID demand drop + high hydro surplus)")
print("  2021: onset of European gas crisis in H2")
print("  2022: peak crisis — mean 193, max 660 EUR/MWh")
print("  2023: partial normalisation begins")

fig, axes = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={'hspace': 0.45})
fig.suptitle('Figure 1 — NO1 Spot Price Dynamics and Regime Analysis',
             fontsize=15, fontweight='bold', y=0.98)

# Panel A: daily prices coloured by regime
ax = axes[0]
for regime, grp in df.groupby('regime', sort=False):
    ax.plot(grp['date'], grp['spot_price'],
            color=REGIME_COLORS[regime], lw=0.75, label=regime)
ax.axvspan(CRISIS_START, CRISIS_END, alpha=0.08, color='red', label='Crisis Window')
mean_p = df['spot_price'].mean()
ax.axhline(mean_p, color='k', ls='--', lw=1.0,
           label=f'Full-sample mean = {mean_p:.1f} EUR/MWh')
ax.set_title('Panel A — Daily Spot Price (Three Distinct Regimes)')
ax.set_ylabel('EUR/MWh')
ax.legend(fontsize=9, ncol=3)

# Panel B: monthly mean ± 1 std
ax = axes[1]
mon = df.resample('ME', on='date')['spot_price'].agg(['mean', 'std']).reset_index()
ax.fill_between(mon['date'], mon['mean'] - mon['std'], mon['mean'] + mon['std'],
                alpha=0.25, color='steelblue', label='±1 Std Dev')
ax.plot(mon['date'], mon['mean'], color='steelblue', lw=2, label='Monthly Mean')
ax.axvspan(CRISIS_START, CRISIS_END, alpha=0.08, color='red')
ax.set_title('Panel B — Monthly Average Price ± 1 Standard Deviation')
ax.set_ylabel('EUR/MWh')
ax.legend(fontsize=9)

# Panel C: annual box plots
ax = axes[2]
years = sorted(df['year'].unique())
data_by_year = [df[df['year'] == y]['spot_price'].values for y in years]
bp = ax.boxplot(data_by_year, patch_artist=True,
                medianprops=dict(color='k', lw=2),
                flierprops=dict(marker='.', ms=3, alpha=0.3))
for patch, yr in zip(bp['boxes'], years):
    if yr in {2021, 2022}: c = COLORS['crisis']
    elif yr < 2021:        c = COLORS['pre']
    else:                  c = COLORS['post']
    patch.set_facecolor(c)
    patch.set_alpha(0.75)
ax.set_xticklabels(years)
ax.set_title('Panel C — Annual Distribution (Crisis Years in Red)')
ax.set_ylabel('EUR/MWh')
ax.set_xlabel('Year')
ax.legend(handles=[
    Patch(fc=COLORS['pre'],    alpha=0.75, label='Pre-Crisis'),
    Patch(fc=COLORS['crisis'], alpha=0.75, label='Crisis 2021-22'),
    Patch(fc=COLORS['post'],   alpha=0.75, label='Post-Crisis'),
], fontsize=9)

plt.savefig('fig1_price_dynamics.png')
plt.show()

print("Figure 1 saved as fig1_price_dynamics.png")
print()


# Seasonality Patterns
df['iso_week'] = df['date'].dt.isocalendar().week.astype(int)
df['dow']      = df['date'].dt.dayofweek

fig = plt.figure(figsize=(16, 12))
fig.suptitle('Figure 2 — Seasonality Patterns — NO1 Electricity Price',
             fontsize=14, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# Panel A: year x month heatmap
ax = fig.add_subplot(gs[0, :2])
pivot = df.groupby(['year', 'month'])['spot_price'].mean().unstack()
pivot.columns = MONTHS[:len(pivot.columns)]
im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=200)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, fontsize=9)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=9)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        v = pivot.values[i, j]
        if not np.isnan(v):
            color = 'white' if v > 130 else 'black'
            ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                    fontsize=7.5, color=color, fontweight='bold')
plt.colorbar(im, ax=ax, label='EUR/MWh', shrink=0.85)
ax.set_title('Panel A — Monthly Average Spot Price Heatmap (EUR/MWh)')

# Panel B: monthly box plots
ax2 = fig.add_subplot(gs[0, 2])
bp = ax2.boxplot([df[df['month'] == m]['spot_price'].values for m in range(1, 13)],
                 patch_artist=True,
                 medianprops=dict(color='k', lw=1.5),
                 flierprops=dict(marker='.', ms=2, alpha=0.3))
for i, patch in enumerate(bp['boxes']):
    patch.set_facecolor('#2196F3' if (i + 1) in {1, 2, 3, 11, 12} else '#FF9800')
    patch.set_alpha(0.65)
ax2.set_xticklabels([m[0] for m in zip(MONTHS)], fontsize=8)
ax2.set_title('Panel B — Distribution by Month')
ax2.set_ylabel('EUR/MWh')

# Panel C: weekly seasonality (ISO week average ± 1 std)
ax3 = fig.add_subplot(gs[1, :2])
wm = df.groupby('iso_week')['spot_price'].mean()
ws = df.groupby('iso_week')['spot_price'].std()
ax3.fill_between(wm.index, wm - ws, wm + ws, alpha=0.22, color='steelblue')
ax3.plot(wm.index, wm.values, color='steelblue', lw=2)
for s, e, c, lbl in [
    (1, 12,  '#1565C0', 'Winter'),
    (13, 25, '#388E3C', 'Spring'),
    (26, 38, '#E65100', 'Summer'),
    (39, 52, '#6A1B9A', 'Autumn'),
]:
    ax3.axvspan(s, e, alpha=0.07, color=c, label=lbl)
ax3.set_title('Panel C — Average Price by Week of Year ± 1 Std Dev')
ax3.set_xlabel('ISO Week Number')
ax3.set_ylabel('EUR/MWh')
ax3.legend(fontsize=9, ncol=4)

# Panel D: day-of-week
ax4 = fig.add_subplot(gs[1, 2])
bp2 = ax4.boxplot([df[df['dow'] == d]['spot_price'].values for d in range(7)],
                  patch_artist=True,
                  medianprops=dict(color='k', lw=1.5),
                  flierprops=dict(marker='.', ms=2, alpha=0.3))
for i, patch in enumerate(bp2['boxes']):
    patch.set_facecolor('#F44336' if i >= 5 else '#4CAF50')
    patch.set_alpha(0.65)
ax4.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], fontsize=9)
ax4.set_title('Panel D — Distribution by Day of Week')
ax4.set_ylabel('EUR/MWh')

plt.savefig('fig2_seasonality.png')
plt.show()

print("Figure 2 saved as fig2_seasonality.png")
print()
print("Monthly average prices (EUR/MWh) — for thesis table:")
monthly_mean = df.groupby('month')['spot_price'].mean().round(1)
for m, v in zip(MONTHS, monthly_mean.values):
    bar = '|' * int(v / 10)
    print(f"  {m:>3}:  {v:>6.1f}  {bar}")
print()

# Hydro Fundamentals (Norway-Specific)
fig = plt.figure(figsize=(16, 13))
fig.suptitle('Figure 3 — Hydro Fundamentals — The Norwegian Price Driver',
             fontsize=14, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.35)

# Panel A: reservoir level vs spot price (dual axis, full timeline)
ax = fig.add_subplot(gs[0, :])
ax_r = ax.twinx()
ax.fill_between(df['date'], df['hydro_reserve_gwh'] / 1000,
                alpha=0.35, color=COLORS['hydro'], label='Hydro Reserve (TWh)')
ax.plot(df['date'], df['hydro_reserve_gwh'] / 1000, color=COLORS['hydro'], lw=0.8)
ax_r.plot(df['date'], df['spot_price'], color=COLORS['crisis'],
          lw=0.9, alpha=0.85, label='Spot Price')
ax.set_ylabel('Hydro Reserve (TWh)', color=COLORS['hydro'])
ax_r.set_ylabel('Spot Price (EUR/MWh)', color=COLORS['crisis'])
ax.tick_params(axis='y', colors=COLORS['hydro'])
ax_r.tick_params(axis='y', colors=COLORS['crisis'])
l1, n1 = ax.get_legend_handles_labels()
l2, n2 = ax_r.get_legend_handles_labels()
ax.legend(l1 + l2, n1 + n2, fontsize=9, loc='upper left')
ax.set_title('Panel A — Hydro Reservoir Level vs Spot Price (Inverse Relationship)')

# Panel B: reservoir deviation vs spot price scatter (coloured by month)
ax2 = fig.add_subplot(gs[1, 0])
sc = ax2.scatter(df['reservoir_deviation_gwh'] / 1000, df['spot_price'],
                 c=df['month'], cmap='hsv', alpha=0.35, s=8)
plt.colorbar(sc, ax=ax2, label='Month')
rho_dev, _ = spearmanr(df['reservoir_deviation_gwh'], df['spot_price'])
z = np.polyfit(df['reservoir_deviation_gwh'] / 1000, df['spot_price'], 1)
xl = np.linspace(df['reservoir_deviation_gwh'].min() / 1000,
                 df['reservoir_deviation_gwh'].max() / 1000, 100)
ax2.plot(xl, np.poly1d(z)(xl), 'k--', lw=1.5, label='Linear fit')
ax2.axvline(0, color='gray', ls=':', lw=1, label='Seasonal normal')
ax2.set_title(f'Panel B — Reservoir Deviation vs Price (Spearman rho = {rho_dev:.3f})')
ax2.set_xlabel('Deviation from Seasonal Normal (TWh)')
ax2.set_ylabel('Spot Price (EUR/MWh)')
ax2.legend(fontsize=8)

# Panel C: seasonal hydro patterns by month (dual axis)
ax3 = fig.add_subplot(gs[1, 1])
mh  = df.groupby('month')['hydro_reserve_gwh'].mean() / 1000
mi  = df.groupby('month')['inflow_hbv_gwh'].mean()
ax3r = ax3.twinx()
ax3.bar(range(1, 13), mh.values, alpha=0.6, color=COLORS['hydro'],
        label='Avg Reserve (TWh)')
ax3r.plot(range(1, 13), mi.values, 'o-', color=COLORS['gas'],
          lw=2, label='Avg Inflow (GWh/wk)')
ax3.set_xticks(range(1, 13))
ax3.set_xticklabels([m[0] for m in zip(MONTHS)], fontsize=8)
ax3.set_ylabel('Reservoir (TWh)', color=COLORS['hydro'])
ax3r.set_ylabel('Inflow (GWh/week)', color=COLORS['gas'])
ax3.tick_params(axis='y', colors=COLORS['hydro'])
ax3r.tick_params(axis='y', colors=COLORS['gas'])
l1, n1 = ax3.get_legend_handles_labels()
l2, n2 = ax3r.get_legend_handles_labels()
ax3.legend(l1 + l2, n1 + n2, fontsize=8)
ax3.set_title('Panel C — Seasonal Hydro Patterns (snowmelt peak May-Jun)')

#Panel D: hydro statistics by regime (table)
ax4 = fig.add_subplot(gs[2, 0])
ax4.axis('off')
rows = []
rc   = ['#BBDEFB', '#FFCDD2', '#C8E6C9']
for regime, grp in df.groupby('regime', sort=False):
    rh, _ = spearmanr(grp['hydro_reserve_gwh'], grp['spot_price'])
    rows.append([
        regime.split('(')[0].strip(),
        f"{grp['hydro_reserve_gwh'].mean() / 1000:.1f} TWh",
        f"{grp['inflow_hbv_gwh'].mean():.0f} GWh/wk",
        f"{rh:.3f}",
    ])
tbl = ax4.table(
    cellText=rows,
    colLabels=['Regime', 'Avg Reserve', 'Avg Inflow', 'Hydro-Price rho'],
    loc='center', cellLoc='center',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 2.0)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#1565C0')
        cell.set_text_props(color='white', fontweight='bold')
    elif r <= 3:
        cell.set_facecolor(rc[r - 1])
ax4.set_title('Panel D — Hydro Stats by Regime', fontsize=10, pad=8)

plt.savefig('fig3_hydro_fundamentals.png')
plt.show()

print("Figure 3 saved as fig3_hydro_fundamentals.png")
print()
print(f"Reservoir deviation Spearman rho (full sample): {rho_dev:.3f}")
print(f"Hydro range: {df['hydro_reserve_gwh'].min()/1000:.0f} to "
      f"{df['hydro_reserve_gwh'].max()/1000:.0f} TWh")
print()


# Fuel Prices and Forward Markets
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('Figure 4 — Fuel Prices and Forward Markets vs Spot Price',
             fontsize=14, fontweight='bold', y=0.98)
plt.subplots_adjust(hspace=0.40, wspace=0.32)

# Panel A: normalised time-series co-movement
ax = axes[0, 0]
series_to_plot = [
    ('spot_price',       'Spot Price',   COLORS['crisis']),
    ('fuel_gas_eur_mwh', 'Gas Price',    COLORS['gas']),
    ('forward_M1',       'Forward M+1',  COLORS['pre']),
    ('fuel_carbon_eur_t','CO2 Price',    COLORS['fwd']),
]
for col, label, color in series_to_plot:
    norm = (df[col] - df[col].mean()) / df[col].std()
    ax.plot(df['date'], norm, lw=0.9, alpha=0.85, label=label, color=color)
ax.axvspan(CRISIS_START, CRISIS_END, alpha=0.08, color='red')
ax.set_title('Panel A — Normalised Series: Co-movement During Crisis')
ax.set_ylabel('Z-Score')
ax.legend(fontsize=9, ncol=2)
ax.text(pd.Timestamp('2021-10-01'), 3.5, 'Crisis onset', fontsize=8, color='red')

# Panel B: gas vs spot scatter — coloured by regime, separate trend lines
ax2 = axes[0, 1]
for regime, grp in df.groupby('regime', sort=False):
    ax2.scatter(grp['fuel_gas_eur_mwh'], grp['spot_price'],
                c=REGIME_COLORS[regime], alpha=0.38, s=8,
                label=regime.split('(')[0].strip())
    z  = np.polyfit(grp['fuel_gas_eur_mwh'], grp['spot_price'], 1)
    xl = np.linspace(grp['fuel_gas_eur_mwh'].min(),
                     grp['fuel_gas_eur_mwh'].max(), 50)
    ax2.plot(xl, np.poly1d(z)(xl), color=REGIME_COLORS[regime], lw=2, ls='--')
r_gas, _ = pearsonr(df['fuel_gas_eur_mwh'], df['spot_price'])
ax2.set_title(f'Panel B — Gas vs Spot (r = {r_gas:.3f})\nDifferent slope per regime = non-stationarity')
ax2.set_xlabel('Gas Price (EUR/MWh)')
ax2.set_ylabel('Spot Price (EUR/MWh)')
ax2.legend(fontsize=8)

# Panel C: Forward M+1 vs spot
ax3 = axes[1, 0]
for regime, grp in df.groupby('regime', sort=False):
    ax3.scatter(grp['forward_M1'], grp['spot_price'],
                c=REGIME_COLORS[regime], alpha=0.38, s=8,
                label=regime.split('(')[0].strip())
r_fwd, _ = pearsonr(df['forward_M1'], df['spot_price'])
z2  = np.polyfit(df['forward_M1'], df['spot_price'], 1)
xl2 = np.linspace(df['forward_M1'].min(), df['forward_M1'].max(), 100)
ax3.plot(xl2, np.poly1d(z2)(xl2), 'k--', lw=1.5, label='Overall trend')
ax3.set_title(f'Panel C — Forward M+1 vs Spot (r = {r_fwd:.3f})\nKey medium-term predictor')
ax3.set_xlabel('Forward M+1 (EUR/MWh)')
ax3.set_ylabel('Spot Price (EUR/MWh)')
ax3.legend(fontsize=8)

# Panel D: EPAD M+1 vs spot (Norway-specific area premium)
ax4 = axes[1, 1]
for regime, grp in df.groupby('regime', sort=False):
    ax4.scatter(grp['epad_M1'], grp['spot_price'],
                c=REGIME_COLORS[regime], alpha=0.38, s=8,
                label=regime.split('(')[0].strip())
r_epad, _ = pearsonr(df['epad_M1'], df['spot_price'])
ax4.set_title(f'Panel D — EPAD M+1 vs Spot (r = {r_epad:.3f})\nNO1 area premium above Nordic system price')
ax4.set_xlabel('EPAD M+1 (EUR/MWh)')
ax4.set_ylabel('Spot Price (EUR/MWh)')
ax4.legend(fontsize=8)

plt.savefig('fig4_fuel_forward.png')
plt.show()

print("Figure 4 saved as fig4_fuel_forward.png")
print()
print("Pearson correlation with spot price (key for feature selection):")
print("-"*50)
feature_corr = [
    ('lag_price_1d',           'Lag Price 1d (AR)'),
    ('forward_M1',             'Forward M+1'),
    ('fuel_gas_eur_mwh',       'Gas Price'),
    ('forward_Y1',             'Forward Y+1'),
    ('epad_M1',                'EPAD M+1'),
    ('epad_Y1',                'EPAD Y+1'),
    ('hydro_reserve_gwh',      'Hydro Reserve'),
    ('reservoir_deviation_gwh','Reservoir Deviation'),
    ('fuel_carbon_eur_t',      'CO2 Price'),
    ('macro_eur_nok',          'EUR/NOK'),
]
for col, label in feature_corr:
    if col not in df.columns:
        continue
    r, _ = pearsonr(df[col].dropna(),
                    df.loc[df[col].notna(), 'spot_price'])
    bar = '#' * int(abs(r) * 20)
    print(f"  {label:<28}  r = {r:+.3f}  {bar}")
print()


# Time-Series Properties
price = df['spot_price'].values

fig = plt.figure(figsize=(16, 12))
fig.suptitle('Figure 5 — Time-Series Properties: ACF, Distribution, Unit Root Tests',
             fontsize=14, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

# Panel A: ACF up to 90 lags
ax1 = fig.add_subplot(gs[0, 0])
acf_vals, conf = acf(price, nlags=90, alpha=0.05)
ax1.bar(range(len(acf_vals)), acf_vals, color='steelblue', alpha=0.75, width=0.85)
ax1.fill_between(range(len(conf)),
                 conf[:, 0] - acf_vals, conf[:, 1] - acf_vals,
                 alpha=0.25, color='red', label='95% CI')
ax1.axhline(0, color='k', lw=0.8)
for lag, lbl in [(7, '1wk'), (30, '1mo'), (60, '2mo'), (90, '3mo')]:
    ax1.axvline(lag, color='orange', ls='--', lw=0.9, alpha=0.8)
    ax1.text(lag, max(acf_vals) * 0.90, lbl, fontsize=7.5,
             ha='center', color='darkorange')
ax1.set_title('Panel A — ACF (90 lags)\nStrong autocorrelation persists well beyond 30 days')
ax1.set_xlabel('Lag (days)')
ax1.set_ylabel('Autocorrelation')
ax1.legend(fontsize=9)

# Panel B: PACF up to 30 lags
ax2 = fig.add_subplot(gs[0, 1])
pacf_vals, conf_p = pacf(price, nlags=30, alpha=0.05, method='ywm')
ax2.bar(range(len(pacf_vals)), pacf_vals, color=COLORS['gas'], alpha=0.75, width=0.85)
ax2.fill_between(range(len(conf_p)),
                 conf_p[:, 0] - pacf_vals, conf_p[:, 1] - pacf_vals,
                 alpha=0.25, color='red')
ax2.axhline(0, color='k', lw=0.8)
ax2.set_title('Panel B — PACF (30 lags)\nSignificant at lags 1, 2, 7 => AR(1,2,7)')
ax2.set_xlabel('Lag (days)')
ax2.set_ylabel('Partial Autocorrelation')
ax2.text(0.97, 0.95, 'Lags 1, 2, 7 significant\n=> SARIMA(2,1,1)(1,0,1)[7]',
         transform=ax2.transAxes, fontsize=9, ha='right', va='top',
         bbox=dict(fc='white', ec='gray', alpha=0.85))

# Panel C: price distribution + Johnson SU fit
ax3 = fig.add_subplot(gs[1, 0])
for regime, grp in df.groupby('regime', sort=False):
    ax3.hist(grp['spot_price'], bins=40, density=True, alpha=0.5,
             color=REGIME_COLORS[regime], label=regime.split('(')[0].strip())
params = johnsonsu.fit(df['spot_price'])
x_range = np.linspace(df['spot_price'].min() - 10,
                      df['spot_price'].max() + 10, 300)
ax3.plot(x_range, johnsonsu.pdf(x_range, *params), 'k-', lw=2,
         label="Johnson's SU fit (Paper 1)")
ax3.set_title("Panel C — Price Distribution + Johnson's SU Fit\n"
              "(Used in the DMLP probabilistic model)")
ax3.set_xlabel('Spot Price (EUR/MWh)')
ax3.set_ylabel('Density')
ax3.legend(fontsize=8)
ax3.set_xlim(-30, 450)

# Panel D: ACF at key forecasting horizons (bar chart — key for model design)
ax4 = fig.add_subplot(gs[1, 1])
acf_full     = acf(price, nlags=365)
h_labels     = [1, 7, 14, 30, 60, 90, 120, 180, 270, 365]
acf_at_h     = [float(acf_full[min(h, len(acf_full) - 1)]) for h in h_labels]
bar_colors   = ['#4CAF50' if v > 0.3 else '#FFC107' if v > 0.1 else '#F44336'
                for v in acf_at_h]
bars = ax4.bar(range(len(h_labels)), acf_at_h, color=bar_colors, alpha=0.85)
ax4.set_xticks(range(len(h_labels)))
ax4.set_xticklabels([f'{h}d' for h in h_labels], fontsize=9)
ax4.axhline(0.3, color='green',  ls='--', lw=1.2, label='0.3 = useful')
ax4.axhline(0.1, color='orange', ls='--', lw=1.2, label='0.1 = marginal')
ax4.axhline(0,   color='k', lw=0.8)
for bar, val in zip(bars, acf_at_h):
    ax4.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.01, f'{val:.3f}',
             ha='center', va='bottom', fontsize=8)
ax4.set_title('Panel D — ACF at Key Horizons\n'
              'Green = useful, Orange = marginal, Red = spurious risk')
ax4.set_ylabel('Autocorrelation')
ax4.legend(fontsize=9)

# Panel E: ADF unit root test table
ax5 = fig.add_subplot(gs[2, :])
ax5.axis('off')
adf_rows = []
series_to_test = [
    ('Spot Price',          df['spot_price']),
    ('Gas Price',           df['fuel_gas_eur_mwh']),
    ('CO2 Price',           df['fuel_carbon_eur_t']),
    ('Forward M1',          df['forward_M1']),
    ('Forward Y1',          df['forward_Y1']),
    ('Hydro Reserve',       df['hydro_reserve_gwh']),
    ('Reservoir Deviation', df['reservoir_deviation_gwh']),
    ('EUR/NOK',             df['macro_eur_nok']),
]
for name, series in series_to_test:
    try:
        stat, p, _, _, crit, _ = adfuller(series.dropna(), maxlag=10)
        result = 'UNIT ROOT' if p > 0.05 else 'Stationary'
        impl   = ('Spurious regression risk at long horizons'
                  if p > 0.05 else 'Safe to use as regressor')
        adf_rows.append([name, f'{stat:.3f}', f'{p:.4f}',
                         f'{crit["1%"]:.3f}', result, impl])
    except Exception as e:
        adf_rows.append([name, 'N/A', 'N/A', 'N/A', 'N/A', str(e)])

tbl = ax5.table(
    cellText=adf_rows,
    colLabels=['Series', 'ADF Stat', 'p-value', 'Crit(1%)',
               'Conclusion', 'Implication for Long-Horizon Models'],
    loc='center', cellLoc='center',
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.65)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#1565C0')
        cell.set_text_props(color='white', fontweight='bold')
    elif 'UNIT ROOT' in str(cell.get_text().get_text()):
        cell.set_facecolor('#FFCDD2')
    elif 'Stationary' in str(cell.get_text().get_text()):
        cell.set_facecolor('#C8E6C9')
    elif r % 2 == 0:
        cell.set_facecolor('#f5f5f5')
ax5.set_title('Panel E — ADF Unit Root Tests (H0: series has a unit root)\n'
              'Critical finding: spot, gas, CO2, forwards all have unit roots',
              fontsize=11, pad=8)

plt.savefig('fig5_ts_properties.png')
plt.show()

print("Figure 5 saved as fig5_ts_properties.png")
print()
print("ACF values at key horizons (Table for thesis):")
print("-"*55)
for h, v in zip(h_labels, acf_at_h):
    if v > 0.3:   flag = "USEFUL — include AR lags"
    elif v > 0.1: flag = "marginal"
    else:         flag = "near-zero — spurious regression risk"
    print(f"  h = {h:3d}d:  ACF = {v:.4f}   {flag}")
print()


# Feature Relevance by Forecasting Horizon
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Figure 6 — Feature Relevance Decay Across Forecasting Horizons\n'
             '(Critical for deciding which features to include at each horizon tier)',
             fontsize=13, fontweight='bold')

horizons = [1, 7, 14, 30, 60, 90, 120, 180, 270, 360]

# Panel A: fundamental features
ax = axes[0]
fundamentals = [
    ('Gas Price',            'fuel_gas_eur_mwh',        COLORS['gas']),
    ('Forward M+1',          'forward_M1',               COLORS['pre']),
    ('Forward Y+1',          'forward_Y1',               COLORS['fwd']),
    ('Hydro Reserve',        'hydro_reserve_gwh',        COLORS['hydro']),
    ('Reservoir Deviation',  'reservoir_deviation_gwh',  '#009688'),
    ('CO2 Price',            'fuel_carbon_eur_t',        '#E91E63'),
    ('EUR/NOK',              'macro_eur_nok',            '#795548'),
]
for label, col, color in fundamentals:
    corrs = []
    for h in horizons:
        if h >= len(df):
            corrs.append(np.nan)
            continue
        shifted = df['spot_price'].shift(-h)
        valid   = ~(shifted.isna() | df[col].isna())
        r, _    = pearsonr(df.loc[valid, col], shifted[valid])
        corrs.append(abs(r))
    ax.plot(horizons, corrs, 'o-', lw=1.8, ms=5, label=label, color=color)

ax.axvline(30,  color='gray', ls=':', lw=1.2)
ax.axvline(180, color='gray', ls=':', lw=1.2)
ax.text(15,  0.02, '1mo', fontsize=8, ha='center', color='gray')
ax.text(150, 0.02, '6mo', fontsize=8, ha='center', color='gray')
ax.axhline(0.3, color='green',  ls='--', lw=1, alpha=0.7)
ax.axhline(0.1, color='orange', ls='--', lw=1, alpha=0.7)
ax.set_title('Panel A — Fundamental Features\n(Stay informative at medium/long horizons)')
ax.set_xlabel('Forecasting Horizon (days)')
ax.set_ylabel('|Pearson r| with future spot price')
ax.legend(fontsize=8, ncol=2)
ax.set_ylim(0, 1)
ax.set_xticks(horizons)

# Panel B: AR features — showing rapid decay
ax2 = axes[1]
ar_features = [
    ('ACF: Spot Price (auto)', 'spot_price',    'steelblue'),
    ('Lag 1d',                 'lag_price_1d',  COLORS['crisis']),
    ('Lag 7d',                 'lag_price_7d',  COLORS['gas']),
    ('Lag 30d',                'lag_price_30d', COLORS['pre']),
    ('MA 7d',                  'ma_price_7d',   '#9C27B0'),
    ('MA 30d',                 'ma_price_30d',  '#607D8B'),
]
for label, col, color in ar_features:
    corrs = []
    for h in horizons:
        if h >= len(df):
            corrs.append(np.nan)
            continue
        if col == 'spot_price':
            corrs.append(float(acf_full[min(h, len(acf_full) - 1)]))
        else:
            shifted = df['spot_price'].shift(-h)
            valid   = ~(shifted.isna() | df[col].isna())
            r, _    = pearsonr(df.loc[valid, col], shifted[valid])
            corrs.append(abs(r))
    ax2.plot(horizons, corrs, 'o-', lw=1.8, ms=5, label=label, color=color)

ax2.axvline(30,  color='gray', ls=':', lw=1.2)
ax2.axvline(180, color='gray', ls=':', lw=1.2)
ax2.axhline(0.3, color='green',  ls='--', lw=1, alpha=0.7, label='0.3 threshold')
ax2.axhline(0.1, color='orange', ls='--', lw=1, alpha=0.7, label='0.1 threshold')
ax2.set_title('Panel B — AR/Lag Features: Rapid Decay\n'
              '=> MUST be dropped at medium/long horizons')
ax2.set_xlabel('Forecasting Horizon (days)')
ax2.set_ylabel('|Correlation| with future spot')
ax2.legend(fontsize=8, ncol=2)
ax2.set_ylim(0, 1)
ax2.set_xticks(horizons)

plt.tight_layout()
plt.savefig('fig6_feature_horizon_relevance.png')
plt.show()


#  Regime Analysis and Structural Breaks
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Figure 7 — Regime Analysis: Structural Breaks in Key Relationships',
             fontsize=14, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

# Panel A: rolling 30-day mean and std
ax1 = fig.add_subplot(gs[0, :])
rm30 = df['spot_price'].rolling(30).mean()
rs30 = df['spot_price'].rolling(30).std()
ax1.fill_between(df['date'], rm30 - rs30, rm30 + rs30,
                 alpha=0.22, color='steelblue', label='+-1 sigma (30d window)')
ax1.plot(df['date'], rm30, color='steelblue', lw=2, label='30d Rolling Mean')
ax1.plot(df['date'], rs30, color=COLORS['gas'],
         lw=1.5, ls='--', label='30d Rolling Std (volatility)')
ax1.axvspan(CRISIS_START, CRISIS_END, alpha=0.10, color='red', label='Crisis Period')
ax1.axvline(CRISIS_START, color='red',   ls='--', lw=1.5, alpha=0.8)
ax1.axvline(CRISIS_END,   color='green', ls='--', lw=1.5, alpha=0.8)
ax1.set_title('Panel A — Rolling Mean and Volatility (30d window) — Three Regimes Visible')
ax1.set_ylabel('EUR/MWh')
ax1.legend(fontsize=9, ncol=4)

# Panel B: gas vs spot — regime-specific slopes show non-stationarity
ax2 = fig.add_subplot(gs[1, 0])
for regime, grp in df.groupby('regime', sort=False):
    ax2.scatter(grp['fuel_gas_eur_mwh'], grp['spot_price'],
                c=REGIME_COLORS[regime], alpha=0.35, s=8,
                label=f"{regime.split('(')[0].strip()} (n={len(grp)})")
    z  = np.polyfit(grp['fuel_gas_eur_mwh'], grp['spot_price'], 1)
    xl = np.linspace(grp['fuel_gas_eur_mwh'].min(),
                     grp['fuel_gas_eur_mwh'].max(), 50)
    ax2.plot(xl, np.poly1d(z)(xl), color=REGIME_COLORS[regime], lw=2.5, ls='--')
ax2.set_title('Panel B — Gas vs Spot\nDifferent slope per regime confirms non-stationarity')
ax2.set_xlabel('Gas Price (EUR/MWh)')
ax2.set_ylabel('Spot Price (EUR/MWh)')
ax2.legend(fontsize=8)

# Panel C: hydro vs spot — regime-specific slopes
ax3 = fig.add_subplot(gs[1, 1])
for regime, grp in df.groupby('regime', sort=False):
    ax3.scatter(grp['hydro_reserve_gwh'] / 1000, grp['spot_price'],
                c=REGIME_COLORS[regime], alpha=0.35, s=8,
                label=regime.split('(')[0].strip())
    z  = np.polyfit(grp['hydro_reserve_gwh'] / 1000, grp['spot_price'], 1)
    xl = np.linspace(grp['hydro_reserve_gwh'].min() / 1000,
                     grp['hydro_reserve_gwh'].max() / 1000, 50)
    ax3.plot(xl, np.poly1d(z)(xl), color=REGIME_COLORS[regime], lw=2.5, ls='--')
ax3.set_title('Panel C — Hydro Reserve vs Spot\nCrisis amplifies hydro-price sensitivity')
ax3.set_xlabel('Hydro Reserve (TWh)')
ax3.set_ylabel('Spot Price (EUR/MWh)')
ax3.legend(fontsize=8)

plt.savefig('fig7_regime_analysis.png')
plt.show()

print()
print("Regime comparison:")
print("="*70)
for regime, grp in df.groupby('regime', sort=False):
    rg, _ = pearsonr(grp['fuel_gas_eur_mwh'], grp['spot_price'])
    rh, _ = spearmanr(grp['hydro_reserve_gwh'], grp['spot_price'])
    g = np.polyfit(grp['fuel_gas_eur_mwh'], grp['spot_price'], 1)
    print(f"\n{regime}")
    print(f"  Observations     : {len(grp)} days")
    print(f"  Price mean+-std  : {grp['spot_price'].mean():.1f} +- {grp['spot_price'].std():.1f} EUR/MWh")
    print(f"  Price min / max  : {grp['spot_price'].min():.1f} / {grp['spot_price'].max():.1f} EUR/MWh")
    print(f"  Gas slope        : {g[0]:.2f} EUR/MWh per EUR/MWh gas")
    print(f"  Gas-Price r      : {rg:.3f}")
    print(f"  Hydro-Price rho  : {rh:.3f}")
print()







