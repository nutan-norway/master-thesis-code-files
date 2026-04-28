# Code for data cleaning, feature engineering and building NO1 electricity price dataset
# Developed by [Nutan Gupta & Mathias Helseth] — [28/04/2026]
# University of Inland Norway
# All rights reserved. For academic use only.

# ============================================================
# Norwegian Electricity Price Forecasting
# Script : master_clean_NO1.py
# Purpose: Clean all raw data files, merge, engineer features,
#          and produce master_clean_NO1.csv for modelling
#
# Input files:
#   - norway_day_ahead_prices_long.csv
#   - hydro_reserves_clean.csv
#   - water_inflow.csv
#   - nordic_system_forwards_clean.csv
#   - nordic_epads_clean.csv
#   - TTF_Daily_.csv
#   - EEX_EUA_Daily_Continuous_2012_2025.csv
#   - brent_front_month_m1.csv
#   - norway_weather_datasetunchanged.csv
#   - load_NO1_raw.csv
#   - eur_nok_clean.csv
#
# Output files (saved to DATA_FOLDER):
#   - master_dataset_NO1.csv   (2881 rows — with NaN warmup)
#   - master_clean_NO1.csv     (2516 rows — model ready)
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from functools import reduce

# ── SET DATA FOLDER ─────────────────────────────────────────────────────
DATA_FOLDER = Path('/Users/nutan/Desktop/Master_Thesis/Jupyter Notebook/RECENT DATA')

# Study period start — determined by earliest TTF gas data
STUDY_START = '2017-11-24'


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def to_daily_ffill(df, date_col='date', start=STUDY_START):
    """Resample any dataframe to daily frequency using forward-fill."""
    df = df.set_index(date_col).resample('D').ffill().reset_index()
    df = df[df[date_col] >= start].reset_index(drop=True)
    return df

def assert_clean(df, name):
    """Assert no nulls and no inf values in a dataframe."""
    nulls = df.isnull().sum().sum()
    infs  = np.isinf(df.select_dtypes(include=np.number)).sum().sum()
    assert nulls == 0, f"{name}: {nulls} null values found!"
    assert infs  == 0, f"{name}: {infs} inf values found!"
    print(f"  {name:<40}: shape={df.shape}  "
          f"date={df.iloc[:,0].min().date()} → {df.iloc[:,0].max().date()}  ✅")


# ============================================================
# SHARED FILES (same for all 5 zones)
# ============================================================
print("=" * 60)
print("SECTION 1 — CLEANING SHARED FILES")
print("=" * 60)

# ── FILE 1: HYDRO RESERVOIR LEVELS ───────────────────────────────────────────
# Source: NVE via hydro_reserves_clean.csv
# Raw   : Weekly, multiple zones (NO, NO1-NO5)
# Keep  : National NO level only
# Action: Weekly → daily via forward-fill
print("\n--- File 1: Hydro Reservoir ---")

df_hydro = pd.read_csv(DATA_FOLDER / 'hydro_reserves_clean.csv')
df_hydro['date'] = pd.to_datetime(df_hydro['datetime'])

# Keep national level only
df_hydro = df_hydro[df_hydro['zone'] == 'NO'][['date', 'hydro_reserve_gwh']].copy()

# Weekly → daily forward-fill
df_hydro = to_daily_ffill(df_hydro)
df_hydro['hydro_reserve_gwh'] = df_hydro['hydro_reserve_gwh'].ffill().bfill()

assert_clean(df_hydro, "hydro_reserves")


# ── FILE 2: WATER INFLOW AND SNOWPACK ─────────────────────────────────────────
# Source: NVE via water_inflow.csv
# Raw   : Weekly, Norwegian column names, data from 1958 onwards
# Keep  : National NO level only
# Action: Translate columns, convert year+week → date, weekly → daily
print("\n--- File 2: Water Inflow ---")

df_inflow = pd.read_csv(DATA_FOLDER / 'water_inflow.csv', sep=';', encoding='utf-8')

# Rename Norwegian columns to English
df_inflow = df_inflow.rename(columns={
    'Område' : 'zone',
    'År'     : 'year',
    'Uke'    : 'week',
    'Nyttbart tilsig HBV (brukes før 2015 og til min/maks/gj.snitt)' : 'inflow_hbv_gwh',
    'Nyttbart tilsig (produksjonsdata og magasinstatistikk, brukes f.o.m. 2015)' : 'inflow_prod_gwh',
    'Magasinavvik' : 'reservoir_deviation_gwh',
    'Snø'          : 'inflow_snow_gwh',
})

# Keep national NO level, drop metadata rows
df_inflow = df_inflow[df_inflow['zone'] == 'NO'][
    ['year', 'week', 'inflow_hbv_gwh', 'inflow_snow_gwh', 'reservoir_deviation_gwh']
].copy()

# Convert to numeric — removes any non-numeric metadata rows
df_inflow['year'] = pd.to_numeric(df_inflow['year'], errors='coerce')
df_inflow['week'] = pd.to_numeric(df_inflow['week'], errors='coerce')
df_inflow = df_inflow.dropna(subset=['year', 'week'])

# Convert ISO year + week number → calendar date (Monday of that week)
df_inflow['date'] = pd.to_datetime(
    df_inflow['year'].astype(int).astype(str)
    + '-W'
    + df_inflow['week'].astype(int).astype(str).str.zfill(2)
    + '-1',
    format='%G-W%V-%u'
)

# Keep only the columns we need, sort by date
df_inflow = (df_inflow[['date', 'inflow_hbv_gwh', 'inflow_snow_gwh', 'reservoir_deviation_gwh']]
             .sort_values('date')
             .reset_index(drop=True))

# Weekly → daily forward-fill
df_inflow = to_daily_ffill(df_inflow)

# Convert value columns to numeric and fill any remaining nulls
for col in ['inflow_hbv_gwh', 'inflow_snow_gwh', 'reservoir_deviation_gwh']:
    df_inflow[col] = pd.to_numeric(df_inflow[col], errors='coerce').ffill().bfill()

assert_clean(df_inflow, "water_inflow")

# Store full history for z-score calculation later
# (use full 1958-2026 record, not just study period)
inflow_hist = pd.read_csv(DATA_FOLDER / 'water_inflow.csv', sep=';', encoding='utf-8')
inflow_hist = inflow_hist.rename(columns={
    'Område' : 'zone',
    'Uke'    : 'week',
    'Nyttbart tilsig HBV (brukes før 2015 og til min/maks/gj.snitt)' : 'inflow_hbv_gwh',
})
inflow_hist = inflow_hist[inflow_hist['zone'] == 'NO'][['week', 'inflow_hbv_gwh']].dropna()
inflow_hist['week']          = pd.to_numeric(inflow_hist['week'],          errors='coerce')
inflow_hist['inflow_hbv_gwh'] = pd.to_numeric(inflow_hist['inflow_hbv_gwh'], errors='coerce')
inflow_hist = inflow_hist.dropna()
weekly_norm = inflow_hist.groupby('week')['inflow_hbv_gwh'].mean()
weekly_std  = inflow_hist.groupby('week')['inflow_hbv_gwh'].std().replace(0, np.nan)
print(f"  Inflow historical norm: {len(inflow_hist)} weekly obs used (full 1958–2026 record)")


# ── FILE 3: NORDIC SYSTEM FORWARD PRICES ──────────────────────────────────────
# Source: Nordic exchange via nordic_system_forwards_clean.csv
# Raw   : Daily, multiple tenors (M1-M6, Q1-Q8, Y1-Y10), SYSTEM zone
# Keep  : M1 and Y1 only (M2/M3/Q1 have r>0.89 with M1 — redundant)
# Action: Pivot to wide format, daily forward-fill weekends
print("\n--- File 3: System Forward Prices ---")

df_fwd = pd.read_csv(DATA_FOLDER / 'nordic_system_forwards_clean.csv')
df_fwd['date'] = pd.to_datetime(df_fwd['datetime'])

# Keep M1 and Y1 only
df_fwd = df_fwd[
    ((df_fwd['tenor_type'] == 'M') & (df_fwd['tenor_num'] == 1)) |
    ((df_fwd['tenor_type'] == 'Y') & (df_fwd['tenor_num'] == 1))
].copy()

# Remove duplicates — two contract codes for same tenor
df_fwd = df_fwd.drop_duplicates(subset=['date', 'tenor_type', 'tenor_num'])

# Pivot: one row per date, one column per tenor
pivot_fwd = df_fwd.pivot_table(
    index   = 'date',
    columns = ['tenor_type', 'tenor_num'],
    values  = 'settlement'
)
pivot_fwd.columns = [f"forward_{t}{n}" for t, n in pivot_fwd.columns]
df_fwd_clean = pivot_fwd.reset_index()

# Daily forward-fill weekends
df_fwd_clean = to_daily_ffill(df_fwd_clean)
for col in ['forward_M1', 'forward_Y1']:
    df_fwd_clean[col] = df_fwd_clean[col].ffill().bfill()

assert_clean(df_fwd_clean, "system_forwards")


# ── FILE 4: TTF NATURAL GAS PRICES ────────────────────────────────────────────
# Source: PEGAS/ICE via TTF_Daily_.csv
# Raw   : Semicolon-delimited, UTF-8 BOM header, European decimal notation
# Action: Parse format, extract settlement price, daily forward-fill
print("\n--- File 4: TTF Natural Gas ---")

df_ttf = pd.read_csv(DATA_FOLDER / 'TTF_Daily_.csv', sep=';', encoding='utf-8-sig')

# Rename columns (first column contains BOM artifact #NAME?)
df_ttf.columns = [
    'drop', 'Open', 'High', 'Low', 'Settlement',
    'Close', 'Volume', 'OI', 'Date', 'ContractName'
]

# Parse date
df_ttf['date'] = pd.to_datetime(df_ttf['Date'], format='%d/%m/%Y %H:%M', errors='coerce')

# Parse settlement — replace European decimal comma with period
df_ttf['fuel_gas_eur_mwh'] = (df_ttf['Settlement']
                               .astype(str)
                               .str.replace(',', '.', regex=False)
                               .apply(pd.to_numeric, errors='coerce'))

df_ttf = (df_ttf[['date', 'fuel_gas_eur_mwh']]
          .dropna()
          .sort_values('date')
          .reset_index(drop=True))

# Daily forward-fill weekends
df_ttf = to_daily_ffill(df_ttf)
df_ttf['fuel_gas_eur_mwh'] = df_ttf['fuel_gas_eur_mwh'].ffill().bfill()

assert_clean(df_ttf, "TTF_gas")


# ── FILE 5: EU CARBON ALLOWANCE PRICE ─────────────────────────────────────────
# Source: EEX via EEX_EUA_Daily_Continuous_2012_2025.csv
# Raw   : Daily, no gaps, standard date format
# Action: Rename, parse date, resample for consistency
print("\n--- File 5: EU Carbon Price ---")

df_carbon = pd.read_csv(DATA_FOLDER / 'EEX_EUA_Daily_Continuous_2012_2025.csv')
df_carbon = df_carbon.rename(columns={
    'Date'             : 'date',
    'AuctionPrice_EUR' : 'fuel_carbon_eur_t'
})
df_carbon['date'] = pd.to_datetime(df_carbon['date'])
df_carbon = to_daily_ffill(df_carbon)
df_carbon['fuel_carbon_eur_t'] = df_carbon['fuel_carbon_eur_t'].ffill().bfill()

assert_clean(df_carbon, "EUA_carbon")


# ── FILE 6: BRENT CRUDE OIL PRICE ─────────────────────────────────────────────
# Source: ICE via brent_front_month_m1.csv
# Raw   : Trading days only, dd/mm/yyyy date format
# Action: Parse dayfirst format, daily forward-fill weekends
print("\n--- File 6: Brent Crude Oil ---")

df_brent = pd.read_csv(DATA_FOLDER / 'brent_front_month_m1.csv')
df_brent['date'] = pd.to_datetime(df_brent['datetime'], dayfirst=True)
df_brent = df_brent.rename(columns={'brent_close': 'fuel_brent_usd_bbl'})[
    ['date', 'fuel_brent_usd_bbl']
]
df_brent = to_daily_ffill(df_brent)
df_brent['fuel_brent_usd_bbl'] = df_brent['fuel_brent_usd_bbl'].ffill().bfill()

assert_clean(df_brent, "Brent_crude")


# ── FILE 7: EUR/NOK EXCHANGE RATE ─────────────────────────────────────────────
# Source: Central bank data via eur_nok_clean.csv
# Raw   : Trading days only
# Action: Daily forward-fill weekends
print("\n--- File 7: EUR/NOK Exchange Rate ---")

df_fx = pd.read_csv(DATA_FOLDER / 'eur_nok_clean.csv')
df_fx['date'] = pd.to_datetime(df_fx['datetime'])
df_fx = df_fx.rename(columns={'eur_nok': 'macro_eur_nok'})[['date', 'macro_eur_nok']]
df_fx = to_daily_ffill(df_fx)
df_fx['macro_eur_nok'] = df_fx['macro_eur_nok'].ffill().bfill()

assert_clean(df_fx, "EUR_NOK")


# ── MERGE ALL SHARED FILES ────────────────────────────────────────────────────
print("\n--- Merging shared files ---")
shared_files = [df_hydro, df_inflow, df_fwd_clean, df_ttf, df_carbon, df_brent, df_fx]
df_shared = reduce(
    lambda left, right: pd.merge(left, right, on='date', how='inner'),
    shared_files
)
df_shared = df_shared.sort_values('date').reset_index(drop=True)
assert_clean(df_shared, "SHARED_MASTER")


# ============================================================
# ZONE-SPECIFIC FILES FOR NO1
# ============================================================
print("\n" + "=" * 60)
print("SECTION 2 — CLEANING ZONE-SPECIFIC FILES (NO1)")
print("=" * 60)

# ── FILE 8: DAY-AHEAD SPOT PRICES — NO1 (TARGET VARIABLE) ────────────────────
# Source: NordPool via norway_day_ahead_prices_long.csv
# Raw   : Hourly, all 5 zones (NO1-NO5), format dd/mm/yyyy HH:MM
# Keep  : NO1 only
# Action: Aggregate 24 hourly prices → 1 daily mean
print("\n--- File 8: Spot Prices (NO1 target) ---")

df_spot = pd.read_csv(DATA_FOLDER / 'norway_day_ahead_prices_long.csv')
df_spot['datetime'] = pd.to_datetime(df_spot['datetime'], format='%d/%m/%Y %H:%M')
df_spot['date']     = df_spot['datetime'].dt.floor('D')

# Keep NO1 zone only
df_spot = df_spot[df_spot['zone'] == 'NO1'].copy()

# Aggregate hourly → daily mean
# Note: DST days have 23 or 25 hours — mean handles this correctly
df_spot = (df_spot
           .groupby('date')['price']
           .mean()
           .reset_index()
           .rename(columns={'price': 'spot_price'}))

df_spot = df_spot[df_spot['date'] >= pd.Timestamp(STUDY_START)].reset_index(drop=True)

# Negative prices are KEPT — valid market phenomenon (excess supply)
# Extreme prices (up to 660 EUR/MWh) are KEPT — 2021-2022 energy crisis
print(f"  Spot_prices (NO1): shape={df_spot.shape}  "
      f"min={df_spot['spot_price'].min():.2f}  max={df_spot['spot_price'].max():.2f}  "
      f"negative_days={(df_spot['spot_price'] < 0).sum()}")


# ── FILE 9: AREA PRICE DIFFERENTIALS (EPAD) — NO1 ────────────────────────────
# Source: Nordic exchange via nordic_epads_clean.csv
# Raw   : Multiple zones, multiple tenors
# Keep  : Oslo zone (SYOSLAFUTBL = NO1), M1 and Y1 only
print("\n--- File 9: NO1 EPADs ---")

df_epad = pd.read_csv(DATA_FOLDER / 'nordic_epads_clean.csv')
df_epad['date'] = pd.to_datetime(df_epad['datetime'], dayfirst=True, errors='coerce')

# Keep Oslo zone only (= NO1)
df_epad = df_epad[df_epad['zone'] == 'SYOSLAFUTBL'].copy()

# Keep M1 and Y1 tenors only
df_epad = df_epad[
    ((df_epad['tenor_type'] == 'M') & (df_epad['tenor_num'] == 1)) |
    ((df_epad['tenor_type'] == 'Y') & (df_epad['tenor_num'] == 1))
].copy()

# Remove duplicates
df_epad = df_epad.drop_duplicates(subset=['date', 'tenor_type', 'tenor_num'])

# Pivot to wide format
pivot_epad = df_epad.pivot_table(
    index   = 'date',
    columns = ['tenor_type', 'tenor_num'],
    values  = 'settlement'
)
pivot_epad.columns = [f"epad_{t}{n}" for t, n in pivot_epad.columns]
df_epad_clean = pivot_epad.reset_index()

# Daily forward-fill weekends
df_epad_clean = to_daily_ffill(df_epad_clean)
for col in ['epad_M1', 'epad_Y1']:
    df_epad_clean[col] = df_epad_clean[col].ffill().bfill()

assert_clean(df_epad_clean, "EPAD_NO1")


# ── FILE 10: ELECTRICITY LOAD — NO1 ──────────────────────────────────────────
# Source: ENTSO-E via load_NO1_raw.csv
# Raw   : Hourly, UTC timestamps with timezone offset
# Action: Convert UTC → Oslo local time, aggregate hourly → daily mean
print("\n--- File 10: Electricity Load (NO1) ---")

df_load = pd.read_csv(DATA_FOLDER / 'load_NO1_raw.csv')

# Parse UTC timestamps
df_load['datetime'] = pd.to_datetime(df_load['datetime'], utc=True)

# Convert to Oslo local time (handles CET/CEST DST automatically)
df_load['date'] = (df_load['datetime']
                   .dt.tz_convert('Europe/Oslo')
                   .dt.floor('D')
                   .dt.tz_localize(None))

# Aggregate hourly → daily mean
# DST transition days (23h or 25h) are handled correctly by mean
df_load = (df_load
           .groupby('date')['Actual Load']
           .mean()
           .reset_index()
           .rename(columns={'Actual Load': 'load_mw'}))

df_load = df_load[df_load['date'] >= pd.Timestamp(STUDY_START)].reset_index(drop=True)
print(f"  Load_NO1: shape={df_load.shape}  "
      f"mean={df_load['load_mw'].mean():.0f} MW ")


# ── FILE 11: WEATHER DATA — NO1 ───────────────────────────────────────────────
# Source: MET Norway via norway_weather_datasetunchanged.csv
# Raw   : All 5 zones, 14 columns (7 completely empty)
# Keep  : NO1 zone, temp/wind/precip only
# Drop  : air_pressure, snow_depth, relative_humidity, cloud_area_fraction,
#         dew_point, shortwave_radiation, wind_speed_of_gust (all 100% empty)
print("\n--- File 11: Weather Data (NO1) ---")

df_wx = pd.read_csv(DATA_FOLDER / 'norway_weather_datasetunchanged.csv')
df_wx['date'] = pd.to_datetime(df_wx['date'])

# Keep NO1 zone, keep only 3 non-empty columns
df_wx = df_wx[df_wx['zone'] == 'NO1'][['date', 'temp', 'wind', 'precip']].copy()

# Rename clearly
df_wx = df_wx.rename(columns={
    'temp'   : 'weather_temp_c',
    'wind'   : 'weather_wind_ms',
    'precip' : 'weather_precip_mm',
})

df_wx = df_wx[df_wx['date'] >= pd.Timestamp(STUDY_START)].reset_index(drop=True)

# Fill minor nulls
for col in ['weather_temp_c', 'weather_wind_ms', 'weather_precip_mm']:
    df_wx[col] = df_wx[col].ffill().bfill()

# Negative temperatures are KEPT — valid Norwegian winter values
print(f"  Weather_NO1: shape={df_wx.shape}  "
      f"temp_min={df_wx['weather_temp_c'].min():.1f}°C  "
      f"temp_max={df_wx['weather_temp_c'].max():.1f}°C")


# ============================================================
# MERGE ALL FILES (NO1)
# ============================================================
print("\n" + "=" * 60)
print("SECTION 3 — MERGING ALL FILES (NO1)")
print("=" * 60)

# Zone-specific files for NO1
zone_files = [df_spot, df_epad_clean, df_load, df_wx]

# Merge zone-specific files together
df_zone = reduce(
    lambda left, right: pd.merge(left, right, on='date', how='inner'),
    zone_files
)

# Merge zone files with shared files
df_master = pd.merge(df_zone, df_shared, on='date', how='inner')
df_master = df_master.sort_values('date').reset_index(drop=True)

print(f"\n  Raw master shape  : {df_master.shape}")
print(f"  Date range        : {df_master['date'].min().date()} → {df_master['date'].max().date()}")
print(f"  Nulls             : {df_master.isnull().sum().sum()}")

# Verify expected row count
assert len(df_master) == 2881, f"Expected 2881 rows, got {len(df_master)}"
print(f"  Row count check   : 2881")


# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 60)
print("SECTION 4 — FEATURE ENGINEERING")
print("=" * 60)

df = df_master.copy()
df = df.sort_values('date').reset_index(drop=True)


# PRICE LAG AND ROLLING FEATURES ───────────────────────────────────
# shift(1) or more ensures no leakage — model never sees today's price
# min_periods = window size — only compute when full window is available

df['lag_price_1d']   = df['spot_price'].shift(1)    # yesterday
df['lag_price_7d']   = df['spot_price'].shift(7)    # 1 week ago
df['lag_price_30d']  = df['spot_price'].shift(30)   # 1 month ago
df['lag_price_365d'] = df['spot_price'].shift(365)  # same day last year

# Moving averages — shift(1) first so today's price is not included
df['ma_price_7d']  = df['spot_price'].shift(1).rolling(7,  min_periods=7).mean()
df['ma_price_30d'] = df['spot_price'].shift(1).rolling(30, min_periods=30).mean()
df['std_price_7d']  = df['spot_price'].shift(1).rolling(7,  min_periods=7).std()
df['std_price_30d'] = df['spot_price'].shift(1).rolling(30, min_periods=30).std()

print("  Group A — Price lag & rolling")


# ── HYDRO DERIVED FEATURES ──────────────────────────────────────────

# Week number (used for z-score calculation only — not a model feature)
df['week_num'] = df['date'].dt.isocalendar().week.astype(int)

# Inflow z-score: how abnormal is current inflow vs 68-year history?
# Positive = above norm (surplus water), Negative = below norm (drought → higher prices)
# weekly_norm and weekly_std were computed from the full 1958-2026 NVE record
df['derived_inflow_z_score'] = (
    (df['inflow_hbv_gwh'] - df['week_num'].map(weekly_norm)) /
    df['week_num'].map(weekly_std)
)

# Daily reservoir change: positive = filling, negative = draining
df['derived_hydro_change'] = df['hydro_reserve_gwh'].diff(1)

print("  Group B — Hydro derived")


# ── FORWARD CURVE DERIVED ────────────────────────────────────────────
# Curve slope: Y1 - M1
# Positive (contango) = market expects prices to rise
# Negative (backwardation) = market expects prices to fall
df['derived_forward_slope'] = df['forward_Y1'] - df['forward_M1']

print("  Group C — Forward derived")


# ── GROUP D: FUEL DERIVED ─────────────────────────────────────────────────────
# Gas price volatility: 30-day rolling standard deviation of TTF
# Captures uncertainty in marginal cost — important for probabilistic forecasting
# min_periods=30 so only valid when full 30-day window is available
df['derived_gas_volatility_30d'] = (
    df['fuel_gas_eur_mwh'].rolling(30, min_periods=30).std()
)

print("  Group D — Fuel derived")


# ── GROUP E: CALENDAR CYCLICAL ENCODING ───────────────────────────────────────
# sin/cos encoding avoids the discontinuity at year-end (Dec → Jan would jump 12 → 1)
# 365.25 accounts for leap years
# cal_week_sin/cos excluded — r=0.9996 with cal_day_of_year (identical information)

df['cal_month_sin']       = np.sin(2 * np.pi * df['date'].dt.month / 12)
df['cal_month_cos']       = np.cos(2 * np.pi * df['date'].dt.month / 12)
df['cal_day_of_year_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofyear / 365.25)
df['cal_day_of_year_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofyear / 365.25)

print("  Group E — Calendar cyclical")


# ============================================================
# SELECT FINAL COLUMNS
# ============================================================
print("\n" + "=" * 60)
print("SECTION 5 — SELECTING FINAL 34 COLUMNS")
print("=" * 60)

# Columns REMOVED (with reason):
#   reservoir_deviation_gwh  → NVE official measure — KEPT (better than derived alternative)
#   forward_M2/M3, Q1        → r > 0.89 with forward_M1 (redundant)
#   epad_Q1                  → r = 0.97 with epad_M1 (redundant)
#   weather_temp_max/min     → r > 0.95 with weather_temp_c (redundant)
#   cal_week_sin/cos         → r = 0.9996 with cal_day_of_year (identical)
#   no1_spot_std             → subtle leakage risk
#   week_num                 → internal variable only, not a model feature

final_cols = [
    'date',
    'spot_price',                   # TARGET — daily mean NO1 price (EUR/MWh)

    # Raw hydro (4)
    'hydro_reserve_gwh',            # national reservoir level (GWh)
    'inflow_hbv_gwh',               # weekly water inflow (GWh)
    'inflow_snow_gwh',              # snowpack energy (GWh)
    'reservoir_deviation_gwh',      # NVE official deviation from long-run norm (GWh)

    # Raw forward (2)
    'forward_M1',                   # 1-month system forward price (EUR/MWh)
    'forward_Y1',                   # 1-year system forward price (EUR/MWh)

    # Raw EPAD (2)
    'epad_M1',                      # NO1 monthly area differential (EUR/MWh)
    'epad_Y1',                      # NO1 yearly area differential (EUR/MWh)

    # Raw fuel (3)
    'fuel_gas_eur_mwh',             # TTF gas price (EUR/MWh)
    'fuel_carbon_eur_t',            # EU carbon allowance (EUR/tonne CO2)
    'fuel_brent_usd_bbl',           # Brent crude oil (USD/barrel)

    # Raw weather (3) — max/min removed (r > 0.95 with temp_c)
    'weather_temp_c',               # daily average temperature (°C)
    'weather_wind_ms',              # daily wind speed (m/s)
    'weather_precip_mm',            # daily precipitation (mm)

    # Raw demand + macro (2)
    'load_mw',                      # NO1 electricity consumption (MW)
    'macro_eur_nok',                # EUR/NOK exchange rate

    # Price lags (4)
    'lag_price_1d',
    'lag_price_7d',
    'lag_price_30d',
    'lag_price_365d',

    # Rolling statistics (4)
    'ma_price_7d',
    'ma_price_30d',
    'std_price_7d',
    'std_price_30d',

    # Hydro derived (2)
    'derived_inflow_z_score',
    'derived_hydro_change',

    # Forward derived (1)
    'derived_forward_slope',

    # Fuel derived (1)
    'derived_gas_volatility_30d',

    # Calendar (4)
    'cal_month_sin',
    'cal_month_cos',
    'cal_day_of_year_sin',
    'cal_day_of_year_cos',
]

# Verify all columns exist
missing = [c for c in final_cols if c not in df.columns]
assert not missing, f"Missing columns: {missing}"

df_final = df[final_cols].copy()
print(f"  Selected {df_final.shape[1]} columns  ({df_final.shape[1]-2} features + date + target)")


# ============================================================
# CLEAN INF VALUES, PRESERVE LAG NaN WARMUP
# ============================================================
print("\n" + "=" * 60)
print("SECTION 6 — CLEANING AND PRESERVING WARMUP")
print("=" * 60)

# Lag columns must keep their NaN values — these are the warmup rows
# that will be removed by dropna() at the end
lag_cols = [
    'lag_price_1d', 'lag_price_7d', 'lag_price_30d', 'lag_price_365d',
    'ma_price_7d',  'ma_price_30d', 'std_price_7d',  'std_price_30d',
    'derived_hydro_change', 'derived_gas_volatility_30d'
]

# Non-lag columns: replace inf → ffill → bfill (safe to fill)
non_lag = [c for c in df_final.columns if c not in lag_cols and c != 'date']
df_final[non_lag]  = df_final[non_lag].replace([np.inf, -np.inf], np.nan).ffill().bfill()

# Lag columns: replace inf → NaN only (preserve NaN warmup)
df_final[lag_cols] = df_final[lag_cols].replace([np.inf, -np.inf], np.nan)

print(f"  Inf values removed  : 0 ")
print(f"  Lag NaN warmup rows : {df_final[lag_cols].isnull().any(axis=1).sum()} (will be removed by dropna)")


# ============================================================
# SECTION 7 — ASSERT CHECKS AND SAVE
# ============================================================
print("\n" + "=" * 60)
print("SECTION 7 — FINAL CHECKS AND SAVING")
print("=" * 60)

# Checks on full dataset (with warmup)
assert len(df_final)    == 2881,   f"Expected 2881 rows, got {len(df_final)}"
assert df_final.shape[1] == 34,    f"Expected 34 columns, got {df_final.shape[1]}"
assert np.isinf(df_final.select_dtypes(include=np.number)).sum().sum() == 0, "Inf found!"
assert pd.isna(df_final.loc[0,   'lag_price_365d']), "lag NaN not preserved on row 0!"
assert pd.notna(df_final.loc[365, 'lag_price_365d']), "lag should be valid on row 365!"

# Save master with warmup (reference file)
df_final.to_csv(DATA_FOLDER / 'master_dataset_NO1.csv', index=False)

# ── DROP WARMUP AND SAVE CLEAN DATASET ───────────────────────────────────────
df_clean = df_final.dropna().reset_index(drop=True)

# Checks on clean dataset
assert len(df_clean)    == 2516,   f"Expected 2516 rows, got {len(df_clean)}"
assert df_clean.shape[1] == 34,    f"Expected 34 columns, got {df_clean.shape[1]}"
assert df_clean.isnull().sum().sum()  == 0,  "Nulls in clean dataset!"
assert np.isinf(df_clean.select_dtypes(include=np.number)).sum().sum() == 0, "Inf in clean!"
assert df_clean.duplicated(subset=['date']).sum() == 0, "Duplicate dates!"
assert df_clean['date'].min() == pd.Timestamp('2018-11-24'), "Wrong start date!"
assert df_clean['date'].max() == pd.Timestamp('2025-10-13'), "Wrong end date!"

df_clean.to_csv(DATA_FOLDER / 'master_clean_NO1.csv', index=False)

print(f"\n  master_dataset_NO1.csv  : {df_final.shape}  (includes NaN warmup rows)")
print(f"  master_clean_NO1.csv    : {df_clean.shape}   (model ready — warmup dropped)")
print(f"\n  Date range (clean)      : {df_clean['date'].min().date()} → {df_clean['date'].max().date()}")
print(f"  Target mean             : {df_clean['spot_price'].mean():.2f} EUR/MWh")
print(f"  Target std              : {df_clean['spot_price'].std():.2f} EUR/MWh")
print(f"  Target min              : {df_clean['spot_price'].min():.2f} EUR/MWh")
print(f"  Target max              : {df_clean['spot_price'].max():.2f} EUR/MWh")
print(f"  Negative price days     : {(df_clean['spot_price'] < 0).sum()}")
print(f"\n  All assert checks passed")
print(f"\n{'='*60}")
print(f"  master_clean_NO1.csv READY FOR MODELLING")
print(f"{'='*60}")
