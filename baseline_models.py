# Code for baseline models and analysis of NO1 electricity price dataset
# Developed by [Nutan Gupta & Mathias Helseth] — [28/04/2026]
# University of Inland Norway
# All rights reserved. For academic use only.

#  Baseline Models
#  Norwegian Electricity Price Forecasting | Region NO1

import warnings; warnings.filterwarnings('ignore')
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats
from scipy.stats import pearsonr, spearmanr

# Scikit-learn
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf

plt.rcParams.update({
    'figure.facecolor': 'white',  'axes.facecolor': '#f8f9fa',
    'axes.grid': True,            'grid.alpha': 0.35,
    'font.size': 11,              'axes.spines.top': False,
    'axes.spines.right': False,   'axes.labelsize': 12,
    'axes.titlesize': 13,         'figure.dpi': 110,
    'savefig.dpi': 150,           'savefig.bbox': 'tight',
})

COLORS = {
    'pre':    '#2196F3', 'crisis': '#F44336', 'post':   '#4CAF50',
    'hydro':  '#00BCD4', 'gas':    '#FF9800', 'fwd':    '#9C27B0',
    'short':  '#1976D2', 'medium': '#F57C00', 'long':   '#388E3C',
}

# Regime / split boundaries
CRISIS_START = pd.Timestamp('2021-07-01')
CRISIS_END   = pd.Timestamp('2023-01-01')
TRAIN_END    = pd.Timestamp('2021-12-31')
VAL_START    = pd.Timestamp('2022-01-01')
VAL_END      = pd.Timestamp('2022-12-31')
TEST_START   = pd.Timestamp('2023-01-01')
TEST_END     = pd.Timestamp('2024-12-31')

# Model colour palette (consistent across all plots)
MODEL_COLORS = {
    'Naive':                    '#9E9E9E',
    'Seasonal-WD':              '#607D8B',
    'SARIMA':                   '#2196F3',
    'ElasticNet-Unconstrained': '#FF9800',
    'ElasticNet-Constrained':   '#4CAF50',
}
MODEL_MARKERS = {
    'Naive': 'o', 'Seasonal-WD': 's', 'SARIMA': '^',
    'ElasticNet-Unconstrained': 'D', 'ElasticNet-Constrained': '*',
}
MONTHS = ['Jan','Feb','Mar','Apr','May','Jun',
          'Jul','Aug','Sep','Oct','Nov','Dec']

print("Setup complete. All libraries loaded.")
print(f"  scikit-learn : {__import__('sklearn').__version__}")
print(f"  statsmodels  : {__import__('statsmodels').__version__}")


# Load Data and Rebuild Features
DATA_PATH = 'master_clean_NO1.csv' 

df = pd.read_csv(DATA_PATH, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

# ── Regime and split labels ───────────────────────────────────────────────
def get_regime(d):
    if d < CRISIS_START:  return 'Pre-Crisis'
    elif d < CRISIS_END:  return 'Crisis'
    return 'Post-Crisis'

def get_split(d):
    if d <= TRAIN_END:              return 'train'
    elif VAL_START <= d <= VAL_END: return 'val'
    elif TEST_START <= d <= TEST_END: return 'test'
    return 'holdout'

df['regime']   = df['date'].apply(get_regime)
df['split']    = df['date'].apply(get_split)
df['year']     = df['date'].dt.year
df['month']    = df['date'].dt.month
df['dow']      = df['date'].dt.dayofweek
df['iso_week'] = df['date'].dt.isocalendar().week.astype(int)

# ── Seasonal profiles (from training data only) ──────────────────────────
train = df[df['date'] <= TRAIN_END].copy()

def seasonal_profile(df_train, col):
    prof = (df_train.groupby(['month','iso_week'])[col]
            .agg(s_mean='mean', s_std='std',
                 s_p10=lambda x: np.percentile(x,10),
                 s_p90=lambda x: np.percentile(x,90))
            .reset_index())
    prof['s_mean'] = prof['s_mean'].rolling(3, center=True, min_periods=1).mean()
    return prof

prof_spot  = seasonal_profile(train, 'spot_price')
prof_hydro = seasonal_profile(train, 'hydro_reserve_gwh')
prof_inflow= seasonal_profile(train, 'inflow_hbv_gwh')

for col, prof, prefix in [
    ('spot_price',        prof_spot,   'sp'),
    ('hydro_reserve_gwh', prof_hydro,  'hy'),
    ('inflow_hbv_gwh',    prof_inflow, 'if'),
]:
    merged = df[['month','iso_week']].merge(
        prof[['month','iso_week','s_mean','s_std','s_p10','s_p90']],
        on=['month','iso_week'], how='left')
    df[f'seasonal_{col}_mean'] = merged['s_mean'].values
    df[f'seasonal_{col}_std']  = merged['s_std'].values
    df[f'seasonal_{col}_p10']  = merged['s_p10'].values
    df[f'seasonal_{col}_p90']  = merged['s_p90'].values

# ── Derived features ──────────────────────────────────────────────────────
df['price_vs_seasonal']  = df['spot_price'] - df['seasonal_spot_price_mean']
df['price_seasonal_z']   = (df['price_vs_seasonal']
                             / df['seasonal_spot_price_std'].clip(lower=1))
df['hydro_seasonal_z']   = ((df['hydro_reserve_gwh']
                             - df['seasonal_hydro_reserve_gwh_mean'])
                             / df['seasonal_hydro_reserve_gwh_std'].clip(lower=1))
df['implied_spot_M1']    = df['forward_M1'] + df['epad_M1']
df['implied_spot_Y1']    = df['forward_Y1'] + df['epad_Y1']
df['epad_slope']         = df['epad_M1'] - df['epad_Y1']

hydro_z = (df['hydro_reserve_gwh'] - df['hydro_reserve_gwh'].mean()) / df['hydro_reserve_gwh'].std()
gas_z   = (df['fuel_gas_eur_mwh']  - df['fuel_gas_eur_mwh'].mean())  / df['fuel_gas_eur_mwh'].std()
df['interaction_hydro_gas'] = hydro_z * gas_z

for w in [7, 14, 30, 90]:
    df[f'roll_price_mean_{w}d'] = df['spot_price'].shift(1).rolling(w).mean()
    df[f'roll_price_std_{w}d']  = df['spot_price'].shift(1).rolling(w).std()

m = df['date'].dt.month
df['season_winter'] = ((m<=2)|(m==12)).astype(int)
df['season_spring'] = ((m>=3)&(m<=5)).astype(int)
df['season_summer'] = ((m>=6)&(m<=8)).astype(int)
df['season_autumn'] = ((m>=9)&(m<=11)).astype(int)
df['is_weekend']    = (df['dow']>=5).astype(int)
df['month_num']     = df['date'].dt.month
df['week_of_year']  = df['iso_week']

print(f"Data loaded: {len(df)} rows | {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Train: {(df['split']=='train').sum()} | Val: {(df['split']=='val').sum()} | "
      f"Test: {(df['split']=='test').sum()} | Holdout: {(df['split']=='holdout').sum()}")

# ── Horizon-specific feature columns ─────────────────────────────────────
FEATURES_SHORT = [c for c in [
    'lag_price_1d','lag_price_7d','lag_price_30d','lag_price_365d',
    'ma_price_7d','ma_price_30d','std_price_7d','std_price_30d',
    'roll_price_mean_7d','roll_price_std_7d','roll_price_mean_30d','roll_price_std_30d',
    'roll_price_mean_90d','roll_price_std_90d',
    'hydro_reserve_gwh','inflow_hbv_gwh','reservoir_deviation_gwh',
    'derived_inflow_z_score','interaction_hydro_gas',
    'fuel_gas_eur_mwh','fuel_carbon_eur_t','derived_gas_volatility_30d',
    'forward_M1','forward_Y1','epad_M1','epad_Y1',
    'derived_forward_slope','epad_slope','implied_spot_M1','implied_spot_Y1',
    'weather_temp_c','weather_wind_ms','weather_precip_mm','load_mw',
    'macro_eur_nok','cal_month_sin','cal_month_cos',
    'cal_day_of_year_sin','cal_day_of_year_cos',
    'season_winter','season_spring','season_summer','season_autumn',
    'is_weekend','month_num','week_of_year',
] if c in df.columns]

FEATURES_MEDIUM = [c for c in [
    'seasonal_spot_price_mean','seasonal_spot_price_std',
    'seasonal_spot_price_p10','seasonal_spot_price_p90',
    'price_vs_seasonal','price_seasonal_z',
    'seasonal_hydro_reserve_gwh_mean','seasonal_hydro_reserve_gwh_std',
    'seasonal_inflow_hbv_gwh_mean',
    'hydro_reserve_gwh','reservoir_deviation_gwh',
    'derived_inflow_z_score','hydro_seasonal_z',
    'forward_M1','forward_Y1','epad_M1','epad_Y1',
    'epad_slope','implied_spot_M1','implied_spot_Y1',
    'fuel_gas_eur_mwh','fuel_carbon_eur_t','derived_gas_volatility_30d',
    'interaction_hydro_gas','macro_eur_nok','lag_price_365d',
    'cal_month_sin','cal_month_cos','cal_day_of_year_sin','cal_day_of_year_cos',
    'season_winter','season_spring','season_summer','season_autumn',
    'month_num','week_of_year',
] if c in df.columns]

FEATURES_LONG = [c for c in [
    'forward_Y1','epad_Y1','implied_spot_Y1',
    'seasonal_hydro_reserve_gwh_mean','seasonal_hydro_reserve_gwh_std',
    'seasonal_inflow_hbv_gwh_mean',
    'seasonal_spot_price_mean','seasonal_spot_price_std',
    'seasonal_spot_price_p10','seasonal_spot_price_p90',
    'macro_eur_nok','cal_month_sin','cal_month_cos',
    'cal_day_of_year_sin','cal_day_of_year_cos',
    'season_winter','season_spring','season_summer','season_autumn',
    'month_num','week_of_year',
] if c in df.columns]

# Coefficient bounds (Norwegian hydro merit-order)
COEF_BOUNDS = {
    'fuel_gas_eur_mwh':        (0, 4.0),
    'fuel_carbon_eur_t':       (0, 1.33),
    'fuel_brent_usd_bbl':      (0, 0.588),
    'hydro_reserve_gwh':       (None, 0),
    'inflow_hbv_gwh':          (None, 0),
    'reservoir_deviation_gwh': (None, 0),
    'derived_inflow_z_score':  (None, 0),
    'hydro_seasonal_z':        (None, 0),
    'load_mw':                 (0, None),
    'forward_M1':              (0, None),
    'forward_Y1':              (0, None),
    'epad_M1':                 (0, None),
    'epad_Y1':                 (0, None),
    'implied_spot_M1':         (0, None),
    'implied_spot_Y1':         (0, None),
    'lag_price_1d':            (0, None),
    'lag_price_7d':            (0, None),
    'lag_price_30d':           (0, None),
    'lag_price_365d':          (0, None),
    'price_vs_seasonal':       (0, None),
    'price_seasonal_z':        (0, None),
}

print(f"\nFeature set sizes:")
print(f"  Short  : {len(FEATURES_SHORT)} features")
print(f"  Medium : {len(FEATURES_MEDIUM)} features")
print(f"  Long   : {len(FEATURES_LONG)} features")


#  Baseline Models: Implementation
class NaiveModel:
    """
    Naive Benchmark — Weekday-Preserving Random Walk

    Forecast logic:
      h = 1  day  : predict last known price (pure random walk)
      h = 2-7 days: predict last known price (carry forward)
      h > 7  days : predict average price for the same weekday
                    observed over the past 90 training days

    This is the standard EPF naive benchmark used in Ghelasi & Ziel (2025)
    Table 5 and Agakishiev et al. (2025) Table 1.
    """

    name = 'Naive'

    def fit(self, train_df):
        self.last_price    = float(train_df['spot_price'].iloc[-1])
        self.weekday_means = (train_df
                              .groupby(train_df['date'].dt.dayofweek)['spot_price']
                              .mean().to_dict())
        self.overall_mean  = float(train_df['spot_price'].mean())
        # Residual std from 1-day naive errors on training data
        naive_1d = train_df['spot_price'].shift(1)
        resids   = train_df['spot_price'] - naive_1d
        self.resid_std = float(resids.dropna().std())
        return self

    def predict(self, horizon, forecast_date):
        if horizon <= 7:
            return self.last_price
        target_dow = (forecast_date + pd.Timedelta(days=horizon)).dayofweek
        return self.weekday_means.get(target_dow, self.overall_mean)


# ── Quick demonstration ───────────────────────────────────────────────────
train_demo = df[df['split'] == 'train'].copy()
m = NaiveModel().fit(train_demo)

print("Naive model fitted on training data.")
print(f"  Last known price : {m.last_price:.2f} EUR/MWh")
print(f"  Weekday means (Mon-Sun):")
for dow, name in enumerate(['Mon','Tue','Wed','Thu','Fri','Sat','Sun']):
    print(f"    {name}: {m.weekday_means.get(dow, 0):.2f} EUR/MWh")
print(f"  Training residual std : {m.resid_std:.2f} EUR/MWh")
print()
print("Prediction examples:")
fc_date = pd.Timestamp('2023-06-01')
for h in [1, 7, 30, 90, 180, 360]:
    pred = m.predict(h, fc_date)
    print(f"  h={h:3d}d ahead from {fc_date.date()} => {pred:.2f} EUR/MWh")


#  Model 2: Seasonal Weekday-Season (WD)

class SeasonalWDModel:
    """
    Seasonal Weekday-Season Benchmark (WD model)

    Predicts the historical average price for the (weekday, season) pair
    of the target date, computed from the training window.

    Seasons: Winter=DJF, Spring=MAM, Summer=JJA, Autumn=SON

    This is a pure seasonality model. It knows nothing about
    recent price levels or fundamental drivers. Its purpose is to
    establish a lower bound: any useful model must beat this
    by capturing price variation beyond seasonal patterns.
    """

    name = 'Seasonal-WD'

    @staticmethod
    def _season(month):
        return {12:'W',1:'W',2:'W', 3:'Sp',4:'Sp',5:'Sp',
                6:'Su',7:'Su',8:'Su', 9:'A',10:'A',11:'A'}[month]

    def fit(self, train_df):
        df_t = train_df.copy()
        df_t['dow']    = df_t['date'].dt.dayofweek
        df_t['season'] = df_t['date'].dt.month.map(self._season)
        self.lookup    = (df_t.groupby(['dow','season'])['spot_price']
                          .mean().to_dict())
        self.fallback  = float(train_df['spot_price'].mean())
        self.resid_std = float(train_df['spot_price'].std())
        return self

    def predict(self, horizon, forecast_date):
        target_date = forecast_date + pd.Timedelta(days=horizon)
        key = (target_date.dayofweek, self._season(target_date.month))
        return self.lookup.get(key, self.fallback)


# ── Quick demonstration ───────────────────────────────────────────────────
m_wd = SeasonalWDModel().fit(train_demo)

print("Seasonal-WD model fitted.")
print()
print("Seasonal averages by (weekday, season) — first 12 entries:")
items = list(m_wd.lookup.items())[:12]
for (dow, season), price in items:
    day_name = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][dow]
    print(f"  {day_name} / {season:<3} : {price:.2f} EUR/MWh")
print()

# Show that WD gives identical answer for any horizon with same weekday/season
print("WD predictions (same weekday+season = same prediction at any horizon):")
fc_date = pd.Timestamp('2023-06-01')
for h in [1, 7, 30, 90, 180, 360]:
    pred = m_wd.predict(h, fc_date)
    target = fc_date + pd.Timedelta(days=h)
    print(f"  h={h:3d}d => target {target.date()} "
          f"({['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][target.dayofweek]}, "
          f"{SeasonalWDModel._season(target.month)}) => {pred:.2f} EUR/MWh")

print()

#  Model 3: SARIMA(2,1,1)(1,0,1)[7]
class SARIMAModel:
    """
    SARIMA(2,1,1)(1,0,1)[7] — Seasonal AutoRegressive Integrated Moving Average

    Order selection rationale (from NB1 PACF analysis):
      - PACF significant at lags 1, 2, 7  => AR(2) + seasonal AR(1) at s=7
      - Series needs differencing (unit root confirmed in NB1) => d=1
      - MA(1) + seasonal MA(1) for residual structure => MA(1), Q=1
      - Weekly seasonality period s=7 (daily data)

    This is the standard EPF benchmark for time-series-only models.
    It captures:
      - Short-term autocorrelation (AR and MA terms)
      - Weekly seasonal pattern (SAR and SMA at s=7)
      - Trend removal via first differencing

    Limitation: SARIMA only uses the price series itself.
    It ignores all fundamental drivers (gas, hydro, forwards).
    SARIMAX would add exogenous variables, but SARIMA gives a clean
    baseline of what autocorrelation structure alone can achieve.
    """

    name = 'SARIMA'

    def __init__(self, order=(2,1,1), seasonal_order=(1,0,1,7)):
        self.order          = order
        self.seasonal_order = seasonal_order
        self.fitted         = None
        self.resid_std      = None

    def fit(self, train_df):
        price = (train_df.set_index('date')['spot_price']
                 .asfreq('D').ffill())
        model = SARIMAX(price,
                        order=self.order,
                        seasonal_order=self.seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False)
        self.fitted    = model.fit(disp=False, maxiter=80)
        self.resid_std = float(self.fitted.resid.std())
        return self

    def predict(self, horizon):
        if self.fitted is None:
            raise RuntimeError("Call fit() first")
        fc = self.fitted.get_forecast(steps=horizon)
        return float(fc.predicted_mean.iloc[-1])

    def predict_interval(self, horizon, alpha=0.05):
        fc  = self.fitted.get_forecast(steps=horizon)
        ci  = fc.conf_int(alpha=alpha)
        return (float(fc.predicted_mean.iloc[-1]),
                float(ci.iloc[-1, 0]),
                float(ci.iloc[-1, 1]))


# ── Fit on training data and show diagnostics ────────────────────────────
print("Fitting SARIMA(2,1,1)(1,0,1)[7] on training data...")
print("(This takes ~20-30 seconds)")
t0 = time.time()
m_sarima = SARIMAModel().fit(train_demo)
print(f"Fitted in {time.time()-t0:.1f}s")
print()
print(f"AIC  : {m_sarima.fitted.aic:.2f}")
print(f"BIC  : {m_sarima.fitted.bic:.2f}")
print(f"Resid std : {m_sarima.resid_std:.2f} EUR/MWh")
print()
print("SARIMA predictions from end of training (h=1 to 360):")
for h in [1, 7, 30, 90, 180, 360]:
    pt, lo, hi = m_sarima.predict_interval(h, alpha=0.05)
    print(f"  h={h:3d}d : {pt:.2f} EUR/MWh  [95% PI: {lo:.1f}, {hi:.1f}]")
print()
print("Key: SARIMA intervals widen with horizon as uncertainty compounds.")
print("At h=360d the interval will be very wide — this is correct behaviour.")


from scipy.optimize import minimize

class ElasticNetModel:
    """
    Elastic Net Regression — Linear model with L1 + L2 regularisation.

    Objective:
        min_{beta} ||y - X*beta||^2 + alpha*l1_ratio*||beta||_1
                                    + alpha*(1-l1_ratio)/2*||beta||^2

    Two modes:
    ─────────
    mode='unconstrained':
        Standard sklearn ElasticNet. No physical constraints.
        Can produce negative gas coefficients or positive hydro coefficients
        if the training data is noisy or multicollinear.

    mode='constrained':
        Adds the Norwegian hydro merit-order coefficient bounds from NB2.
        Solved via L-BFGS-B optimisation (handles box constraints).
        Prevents economically nonsensical coefficients.

    Why does constraining help?
        At longer horizons, multicollinearity and unit-root spurious regression
        can push coefficients in the wrong direction. Physical bounds act as
        a regulariser that keeps the model economically interpretable.

    Feature set selection (automatic by horizon):
        h <= 30  days : FEATURES_SHORT
        h = 31-180 days : FEATURES_MEDIUM
        h > 180  days : FEATURES_LONG
    """

    def __init__(self, alpha=0.5, l1_ratio=0.5, mode='unconstrained'):
        self.alpha    = alpha
        self.l1_ratio = l1_ratio
        self.mode     = mode
        self.name     = f'ElasticNet-{mode.capitalize()}'
        self.scaler   = StandardScaler()
        self.coef_    = None
        self.intercept_ = 0.0
        self.feature_cols_ = []
        self.resid_std_    = 1.0

    def _get_bounds(self, feature_cols, scales):
        bounds = []
        for col, s in zip(feature_cols, scales):
            lo, hi = COEF_BOUNDS.get(col, (None, None))
            lo_s = (-np.inf if lo is None else lo * s)
            hi_s = ( np.inf if hi is None else hi * s)
            bounds.append((lo_s, hi_s))
        return bounds

    def fit(self, X, y):
        self.feature_cols_ = list(X.columns)
        Xn = X.fillna(0).values.astype(float)
        yn = y.fillna(float(y.median())).values.astype(float)
        Xs = self.scaler.fit_transform(Xn)

        if self.mode == 'unconstrained':
            net = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                             max_iter=2000, fit_intercept=True)
            net.fit(Xs, yn)
            self.coef_      = net.coef_
            self.intercept_ = net.intercept_

        else:  # constrained — L-BFGS-B with physical bounds
            bounds = self._get_bounds(self.feature_cols_,
                                      self.scaler.scale_)

            # Warm start: unconstrained solution clipped to bounds
            net0 = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                              max_iter=2000, fit_intercept=True)
            net0.fit(Xs, yn)
            coef0 = np.clip(net0.coef_,
                            [b[0] if b[0] != -np.inf else -1e6 for b in bounds],
                            [b[1] if b[1] !=  np.inf else  1e6 for b in bounds])

            lam, rho = self.alpha, self.l1_ratio

            def objective(params):
                coef, intercept = params[:-1], params[-1]
                resid = yn - (Xs @ coef + intercept)
                mse   = np.mean(resid**2)
                l1    = lam * rho       * np.sum(np.abs(coef))
                l2    = lam*(1-rho)/2   * np.sum(coef**2)
                return mse + l1 + l2

            x0 = np.append(coef0, net0.intercept_)
            res = minimize(objective, x0, method='L-BFGS-B',
                           bounds=bounds + [(None, None)],
                           options={'maxiter': 500, 'ftol': 1e-9})
            self.coef_      = res.x[:-1]
            self.intercept_ = res.x[-1]

        yhat = Xs @ self.coef_ + self.intercept_
        self.resid_std_ = float(np.std(yn - yhat))
        return self

    def predict(self, X):
        Xn = X[self.feature_cols_].fillna(0).values.astype(float)
        Xs = self.scaler.transform(Xn)
        return Xs @ self.coef_ + self.intercept_

    def coef_table(self):
        """Return a DataFrame of feature coefficients sorted by |coef|."""
        # Convert scaled coef back to original units
        coef_orig = self.coef_ / self.scaler.scale_
        return (pd.DataFrame({
                    'feature': self.feature_cols_,
                    'coef_orig': coef_orig,
                    'coef_scaled': self.coef_,
                    'bound_lo': [COEF_BOUNDS.get(c,(None,None))[0]
                                 for c in self.feature_cols_],
                    'bound_hi': [COEF_BOUNDS.get(c,(None,None))[1]
                                 for c in self.feature_cols_],
                })
                .assign(abs_coef=lambda d: d['coef_orig'].abs())
                .sort_values('abs_coef', ascending=False)
                .reset_index(drop=True))


# ── Fit both models and compare coefficients ─────────────────────────────
print("Fitting Elastic Net models on SHORT-TERM features...")
print()

X_train = df.loc[df['split']=='train', FEATURES_SHORT].fillna(0)
y_train = df.loc[df['split']=='train', 'spot_price']

print("  Fitting unconstrained model...")
m_enet_unc = ElasticNetModel(mode='unconstrained').fit(X_train, y_train)

print("  Fitting constrained model (L-BFGS-B)...")
m_enet_con = ElasticNetModel(mode='constrained').fit(X_train, y_train)

# Show top coefficients for both models
ct_unc = m_enet_unc.coef_table().head(12)
ct_con = m_enet_con.coef_table().head(12)

print()
print("Top 12 features — UNCONSTRAINED ElasticNet:")
print(ct_unc[['feature','coef_orig','bound_lo','bound_hi']].to_string(index=False))
print()
print("Top 12 features — CONSTRAINED ElasticNet:")
print(ct_con[['feature','coef_orig','bound_lo','bound_hi']].to_string(index=False))

# Check for constraint violations in unconstrained model
print()
print("Constraint violations in UNCONSTRAINED model:")
n_violations = 0
for _, row in ct_unc.iterrows():
    lo = row['bound_lo']
    hi = row['bound_hi']
    coef = row['coef_orig']
    if lo is not None and coef < lo:
        print(f"  {row['feature']:<30}: coef={coef:.4f} < bound_lo={lo}")
        n_violations += 1
    if hi is not None and coef > hi:
        print(f"  {row['feature']:<30}: coef={coef:.4f} > bound_hi={hi}")
        n_violations += 1
if n_violations == 0:
    print("  None in top-12 features (may exist in lower-ranked features)")
print()
print("The constrained model enforces all bounds by construction.")


# Rolling Window Evaluation Engine

def rolling_evaluate(df, model_name, model_factory,
                     horizons, feature_cols=None,
                     window_days=365*3, roll_step=30,
                     verbose=True):
    """
    Expanding-Window Forecast Evaluation Engine

    Design:
      - Evaluation period: VAL_START to TEST_END (2022-2024)
      - Training window: last 3 years (1095 days) of data before forecast date
      - Window expands by roll_step (30 days) at each step
      - For each evaluation date T and each horizon h:
          1. Train on data from [T - window_days, T]
          2. Predict price at T + h
          3. Compare with actual price at T + h

    Parameters
    ----------
    df           : full DataFrame with features and spot_price
    model_name   : string label ('Naive', 'SARIMA', etc.)
    model_factory: callable that returns a fresh unfitted model instance
    horizons     : list of integer forecast horizons (days)
    feature_cols : feature columns for linear models (None for time-series)
    window_days  : training window length in days (default 3 years)
    roll_step    : re-estimation frequency in days (default 30)
    verbose      : print progress

    Returns
    -------
    DataFrame with columns: forecast_date, target_date, horizon,
                            y_true, y_pred, error, abs_error, sq_error,
                            year, regime, split, model
    """
    df  = df.sort_values('date').reset_index(drop=True)
    # Evaluation dates: VAL_START to TEST_END, stepped by roll_step
    eval_mask  = ((df['date'] >= VAL_START) & (df['date'] <= TEST_END))
    eval_dates = df.loc[eval_mask, 'date'].values[::roll_step]
    n_steps    = len(eval_dates)

    records = []
    t_start = time.time()

    for step_i, fc_ts in enumerate(eval_dates):
        fc_date = pd.Timestamp(fc_ts)

        if verbose and step_i % max(1, n_steps//8) == 0:
            elapsed = time.time() - t_start
            pct     = (step_i+1) / n_steps * 100
            print(f"    [{model_name}] {step_i+1}/{n_steps} "
                  f"({pct:.0f}%)  {fc_date.date()}  {elapsed:.0f}s", end='\r')

        # Training window
        t_start_win = fc_date - pd.Timedelta(days=window_days)
        train_mask  = (df['date'] > t_start_win) & (df['date'] <= fc_date)
        train_df    = df[train_mask].copy()
        if len(train_df) < 90:
            continue

        # Fit model
        try:
            model = model_factory()
            if feature_cols is not None:   # linear models
                X_tr = train_df[feature_cols].fillna(0)
                y_tr = train_df['spot_price']
                model.fit(X_tr, y_tr)
            else:                           # time-series models
                model.fit(train_df)
        except Exception as e:
            if verbose:
                print(f"\n    [WARN] fit failed at {fc_date.date()}: {e}")
            continue

        # Predict each horizon
        for h in horizons:
            tgt_date  = fc_date + pd.Timedelta(days=h)
            tgt_mask  = df['date'] == tgt_date
            if not tgt_mask.any():
                continue
            y_true = float(df.loc[tgt_mask, 'spot_price'].iloc[0])

            try:
                if feature_cols is not None:
                    # Use FORECAST DATE features (no lookahead)
                    feat_row = df.loc[df['date'] == fc_date, feature_cols]
                    if len(feat_row) == 0:
                        continue
                    y_pred = float(model.predict(feat_row)[0])
                elif model_name == 'Naive':
                    y_pred = model.predict(h, fc_date)
                elif model_name == 'Seasonal-WD':
                    y_pred = model.predict(h, fc_date)
                elif model_name == 'SARIMA':
                    y_pred = model.predict(h)
                else:
                    continue
            except Exception:
                continue

            meta = df.loc[tgt_mask].iloc[0]
            records.append({
                'forecast_date': fc_date,
                'target_date':   tgt_date,
                'horizon':       h,
                'y_true':        y_true,
                'y_pred':        y_pred,
                'error':         y_pred - y_true,
                'abs_error':     abs(y_pred - y_true),
                'sq_error':      (y_pred - y_true)**2,
                'year':          tgt_date.year,
                'regime':        meta.get('regime', ''),
                'split':         meta.get('split',  ''),
                'model':         model_name,
            })

    if verbose:
        print()   # end carriage-return line
    return pd.DataFrame(records)


print("Rolling evaluation engine defined.")
print()
print("Configuration:")
print(f"  Evaluation period : {VAL_START.date()} to {TEST_END.date()}")
print(f"  Training window   : 3 years (1095 days)")

# ── Evaluation horizons ───────────────────────────────────────────────────
Full set: [1, 7, 14, 30, 60, 90, 180, 360]
EVAL_HORIZONS = [1, 7, 14, 30, 90, 180, 360]

# Feature set selection by horizon (used for linear models)
def get_features_for_horizon(h):
    if h <= 30:
        return FEATURES_SHORT
    elif h <= 180:
        return FEATURES_MEDIUM
    else:
        return FEATURES_LONG

print("="*55)
print("RUNNING ROLLING-WINDOW EVALUATION")
print(f"Horizons: {EVAL_HORIZONS}")
print("="*55)
print()

all_results = []

# 1. Naive ──────────────────────────────────────────────────────────────
print("Running Naive...")
res_naive = rolling_evaluate(
    df, 'Naive', NaiveModel,
    horizons=EVAL_HORIZONS, feature_cols=None, verbose=True
)
all_results.append(res_naive)
print(f"  Done: {len(res_naive)} records")

# 2. Seasonal-WD ────────────────────────────────────────────────────────
print("Running Seasonal-WD...")
res_wd = rolling_evaluate(
    df, 'Seasonal-WD', SeasonalWDModel,
    horizons=EVAL_HORIZONS, feature_cols=None, verbose=True
)
all_results.append(res_wd)
print(f"  Done: {len(res_wd)} records")

# 3. SARIMA  ────────────────────────────
print("Running SARIMA (this takes a few minutes)...")
res_sarima = rolling_evaluate(
    df, 'SARIMA',
    lambda: SARIMAModel(order=(2,1,1), seasonal_order=(1,0,1,7)),
    horizons=EVAL_HORIZONS, feature_cols=None, verbose=True
)
all_results.append(res_sarima)
print(f"  Done: {len(res_sarima)} records")

#  4. ElasticNet Unconstrained ───────────────────────────────────────────
print("Running ElasticNet-Unconstrained...")
res_unc = rolling_evaluate(
    df, 'ElasticNet-Unconstrained',
    lambda: ElasticNetModel(alpha=0.5, l1_ratio=0.5, mode='unconstrained'),
    horizons=EVAL_HORIZONS,
    feature_cols=FEATURES_SHORT,   
    verbose=True
)
all_results.append(res_unc)
print(f"  Done: {len(res_unc)} records")

#  5. ElasticNet Constrained ─────────────────────────────────────────────
print("Running ElasticNet-Constrained...")
res_con = rolling_evaluate(
    df, 'ElasticNet-Constrained',
    lambda: ElasticNetModel(alpha=0.5, l1_ratio=0.5, mode='constrained'),
    horizons=EVAL_HORIZONS,
    feature_cols=FEATURES_SHORT,
    verbose=True
)
all_results.append(res_con)
print(f"  Done: {len(res_con)} records")

# Combine all results ───────────────────────────────────────────────────
results = pd.concat(all_results, ignore_index=True)
results['forecast_date'] = pd.to_datetime(results['forecast_date'])
results['target_date']   = pd.to_datetime(results['target_date'])

print()
print(f"Total evaluation records: {len(results)}")
print(f"Models evaluated: {results['model'].unique().tolist()}")
print(f"Horizons covered: {sorted(results['horizon'].unique().tolist())}")



# Results: RMSE and MAE Tables

def summary_table(results, metric='rmse'):
    """
    Build RMSE or MAE table: rows=models, columns=horizons.
    Mirrors Table 5 in Ghelasi & Ziel (2025).
    """
    fn = (lambda g: np.sqrt(g['sq_error'].mean())) if metric == 'rmse' \
         else (lambda g: g['abs_error'].mean())

    models   = list(results['model'].unique())
    horizons = sorted(results['horizon'].unique())
    rows     = []
    for model in models:
        sub = results[results['model'] == model]
        row = {'Model': model}
        for h in horizons:
            h_sub = sub[sub['horizon'] == h]
            row[f'h={h}d'] = round(fn(h_sub), 2) if len(h_sub) > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index('Model')


rmse_table = summary_table(results, 'rmse')
mae_table  = summary_table(results, 'mae')

print("="*70)
print("TABLE — RMSE by Model and Horizon (EUR/MWh)")
print("(Mirrors Table 5 in Ghelasi & Ziel 2025)")
print("="*70)
print(rmse_table.to_string())
print()
print("="*70)
print("TABLE — MAE by Model and Horizon (EUR/MWh)")
print("="*70)
print(mae_table.to_string())

rmse_table.to_csv('summary_rmse_NO1.csv')
mae_table.to_csv('summary_mae_NO1.csv')
print()
print("Tables saved: summary_rmse_NO1.csv / summary_mae_NO1.csv")

# ── Improvement over Naive ─────────────────────────────────────────────────
print()
print("="*70)
print("IMPROVEMENT OVER NAIVE BENCHMARK (RMSE reduction %)")
print("="*70)
naive_rmse = rmse_table.loc['Naive'] if 'Naive' in rmse_table.index else None
if naive_rmse is not None:
    for model in rmse_table.index:
        if model == 'Naive':
            continue
        delta = ((naive_rmse - rmse_table.loc[model]) / naive_rmse * 100).round(1)
        print(f"\n  {model}:")
        for col, d in delta.items():
            sign   = 'better' if d > 0 else 'worse'
            arrow  = '+' if d > 0 else ''
            print(f"    {col:<10}: {arrow}{d:.1f}%  ({sign} than Naive)")


#  Figure BM1: Rolling RMSE Over Time
h_plot = 1   # show h=1 day rolling performance over time

fig, ax = plt.subplots(figsize=(15, 6))
fig.suptitle(f'Rolling 30-Day RMSE Over Time  (h={h_plot} day ahead)\n'
             'Red Shaded area = 2022 Crisis (stress-test validation period)',
             fontsize=13, fontweight='bold')

h1_data = results[results['horizon'] == h_plot].copy()

for model in results['model'].unique():
    sub = (h1_data[h1_data['model'] == model]
           .sort_values('target_date'))
    rolling_rmse = (sub['sq_error']
                    .rolling(30, min_periods=10)
                    .mean()
                    .pipe(np.sqrt))
    ax.plot(sub['target_date'], rolling_rmse,
            lw=1.8, alpha=0.9,
            color=MODEL_COLORS.get(model, 'black'),
            label=model)

ax.axvspan(VAL_START, VAL_END, alpha=0.12, color='red', label='Crisis 2022')
ax.axvline(TEST_START, color='green', ls='--', lw=1.2, alpha=0.7,
           label='Test period start')

ax.set_ylabel('30-Day Rolling RMSE (EUR/MWh)')
ax.set_xlabel('Date')
ax.legend(fontsize=9, ncol=3, loc='upper right')

plt.tight_layout()
plt.savefig('fig_bm1_rolling_rmse.png')
plt.show()

print("Figure BM1 saved: fig_bm1_rolling_rmse.png")
print()


# Figure BM2: RMSE and MAE by Horizon
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Forecast Accuracy by Horizon',
             fontsize=13, fontweight='bold')

horizons = sorted(results['horizon'].unique())

for ax, metric, ylabel in [
    (axes[0], 'rmse', 'RMSE (EUR/MWh)'),
    (axes[1], 'mae',  'MAE (EUR/MWh)'),
]:
    fn = (lambda g: np.sqrt(g['sq_error'].mean())) if metric == 'rmse' \
         else (lambda g: g['abs_error'].mean())

    for model in results['model'].unique():
        sub  = results[results['model'] == model]
        vals = [fn(sub[sub['horizon']==h]) if (sub['horizon']==h).any()
                else np.nan for h in horizons]
        ax.plot(horizons, vals,
                marker=MODEL_MARKERS.get(model, 'o'),
                lw=2.0, ms=8,
                color=MODEL_COLORS.get(model, 'k'),
                label=model)

    # Shade the three horizon zones
    ax.axvspan(0,   30,  alpha=0.06, color='steelblue')
    ax.axvspan(30,  180, alpha=0.06, color='orange')
    ax.axvspan(180, 370, alpha=0.06, color='green')
    ax.axvline(30,  color='gray', ls=':', lw=1.2)
    ax.axvline(180, color='gray', ls=':', lw=1.2)

    ymax = ax.get_ylim()[1]
    ax.text(15,  ymax*0.97, 'Short',  ha='center', fontsize=9,
            color='steelblue', fontweight='bold')
    ax.text(100, ymax*0.97, 'Medium', ha='center', fontsize=9,
            color='darkorange', fontweight='bold')
    ax.text(270, ymax*0.97, 'Long',   ha='center', fontsize=9,
            color='darkgreen', fontweight='bold')

    ax.set_xlabel('Forecasting Horizon (days)')
    ax.set_ylabel(ylabel)
    ax.set_xticks(horizons)
    ax.set_xticklabels([f'{h}d' for h in horizons])
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('fig_bm2_rmse_by_horizon.png')
plt.show()


# Figure BM3: Annual RMSE Breakdown
horizons_to_show = [h for h in [1, 7, 30, 90] if h in results['horizon'].unique()]
models = list(results['model'].unique())
years = sorted(results['year'].unique())

fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=False)
axes = axes.flatten()  # makes indexing easy

fig.suptitle(
    'Annual RMSE Breakdown by Horizon\n'
    '(Red shading = crisis year 2022 — the stress-test validation)',
    fontsize=13,
    fontweight='bold'
)

x_pos = np.arange(len(years))
width = 0.75 / len(models)

for ax, h in zip(axes, horizons_to_show):
    h_df = results[results['horizon'] == h]

    for i, model in enumerate(models):
        vals = []
        for yr in years:
            sub = h_df[(h_df['model'] == model) & (h_df['year'] == yr)]
            vals.append(np.sqrt(sub['sq_error'].mean()) if len(sub) > 0 else np.nan)

        offset = (i - len(models)/2 + 0.5) * width
        ax.bar(
            x_pos + offset,
            vals,
            width,
            color=MODEL_COLORS.get(model, 'gray'),
            alpha=0.82,
            label=model if h == horizons_to_show[0] else ''
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(years, rotation=45, fontsize=8)
    ax.set_title(f'h = {h} days', fontsize=11)
    ax.set_ylabel('RMSE (EUR/MWh)')

    # Highlight 2022 crisis year
    if 2022 in years:
        idx_2022 = years.index(2022)
        ax.axvspan(
            idx_2022 - 0.45,
            idx_2022 + 0.45,
            alpha=0.12,
            color='red',
            zorder=0
        )

# Hide any unused axes if fewer than 4 horizons
for ax in axes[len(horizons_to_show):]:
    ax.set_visible(False)

handles = [
    Patch(color=MODEL_COLORS.get(m, 'gray'), alpha=0.82, label=m)
    for m in models
]

fig.legend(
    handles=handles,
    loc='lower center',
    ncol=min(len(models), 4),
    fontsize=9,
    bbox_to_anchor=(0.5, 0.01)
)

plt.tight_layout(rect=[0, 0.06, 1, 0.95])
plt.savefig('fig_bm3_annual_breakdown.png', dpi=300, bbox_inches='tight')
plt.show()

print("Figure BM3 saved: fig_bm3_annual_breakdown.png")
print()
print("Annual RMSE breakdown (RMSE per year):")
for h in horizons_to_show[:3]:
    print(f"\n  Horizon h={h}d:")
    h_df = results[results['horizon'] == h]
    pivot = (
        h_df.groupby(['model', 'year'])['sq_error']
        .apply(lambda x: round(np.sqrt(x.mean()), 1))
        .unstack('year')
    )
    print(pivot.to_string())
print()


#  Figure BM4: Coefficient Stability
train_X_s = df.loc[df['split']=='train', FEATURES_SHORT].fillna(0)
train_X_m = df.loc[df['split']=='train', FEATURES_MEDIUM].fillna(0)
train_X_l = df.loc[df['split']=='train', FEATURES_LONG].fillna(0)
train_y   = df.loc[df['split']=='train', 'spot_price']

print("Fitting constrained ElasticNet on all three feature tiers...")
m_short  = ElasticNetModel(mode='constrained').fit(train_X_s, train_y)
m_medium = ElasticNetModel(mode='constrained').fit(train_X_m, train_y)
m_long   = ElasticNetModel(mode='constrained').fit(train_X_l, train_y)

# Key features that appear in multiple tiers
shared_features = list(set(FEATURES_SHORT) &
                        set(FEATURES_MEDIUM) &
                        set(FEATURES_LONG))
interest_feats  = [
    'forward_Y1', 'epad_Y1', 'hydro_reserve_gwh',
    'fuel_gas_eur_mwh', 'macro_eur_nok',
    'season_winter', 'month_num',
]
interest_feats  = [f for f in interest_feats if f in shared_features]

fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# Left panel: bar chart of shared feature coefficients across tiers
ax = axes[0]
if interest_feats:
    n_f = len(interest_feats)
    x   = np.arange(n_f)
    w   = 0.25

    def get_coef(model, feat):
        ct = model.coef_table()
        row = ct[ct['feature'] == feat]
        return float(row['coef_orig'].iloc[0]) if len(row) > 0 else 0.0

    coefs_s = [get_coef(m_short,  f) for f in interest_feats]
    coefs_m = [get_coef(m_medium, f) for f in interest_feats]
    coefs_l = [get_coef(m_long,   f) for f in interest_feats]

    ax.bar(x - w, coefs_s, w, color=COLORS['short'],  alpha=0.82, label='Short tier')
    ax.bar(x,     coefs_m, w, color=COLORS['medium'], alpha=0.82, label='Medium tier')
    ax.bar(x + w, coefs_l, w, color=COLORS['long'],   alpha=0.82, label='Long tier')
    ax.set_xticks(x)
    ax.set_xticklabels(interest_feats, rotation=45, ha='right', fontsize=9)
    ax.axhline(0, color='k', lw=0.8)
    #ax.set_title('Shared Feature Coefficients Across Tiers\n'
    #             '(physical constraints ensure correct signs)')
    ax.set_ylabel('Original-scale coefficient')
    ax.legend(fontsize=9)

# Right panel: top-10 coefficients for each tier
ax2 = axes[1]
for i, (model, label, color) in enumerate([
    (m_short,  'Short (h<=30d)',    COLORS['short']),
    (m_medium, 'Medium (h=1-6mo)', COLORS['medium']),
    (m_long,   'Long (h=6-12mo)',  COLORS['long']),
]):
    ct = model.coef_table().head(10)
    ct = ct.sort_values('coef_orig')
    y_pos = np.arange(len(ct)) + i * 12
    ax2.barh(y_pos, ct['coef_orig'].values, 0.8,
             color=color, alpha=0.75, label=label)
    for y, feat in zip(y_pos, ct['feature'].values):
        ax2.text(0.02, y, feat, va='center', fontsize=7,
                 color='black', ha='left')
ax2.axvline(0, color='k', lw=0.8)
ax2.set_title('Top 10 Features per Tier\n(most influential coefficients)')
ax2.set_xlabel('Original-scale coefficient')
ax2.legend(fontsize=9, loc='lower right')
ax2.set_yticks([])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('fig_bm4_coef_stability.png')
plt.show()

print("Figure BM4 saved: fig_bm4_coef_stability.png")
print()
print("Full constrained coefficient table — SHORT tier (top 15):")
print(m_short.coef_table()[['feature','coef_orig','bound_lo','bound_hi']]
      .head(15).to_string(index=False))
print()
print("Physical constraint check:")
ct_full = m_short.coef_table()
for _, row in ct_full.iterrows():
    lo, hi = row['bound_lo'], row['bound_hi']
    c = row['coef_orig']
    if lo is not None and not np.isnan(c) and c < lo - 0.001:
        print(f"  VIOLATION: {row['feature']} coef={c:.4f} < bound={lo}")
    if hi is not None and not np.isnan(c) and c > hi + 0.001:
        print(f"  VIOLATION: {row['feature']} coef={c:.4f} > bound={hi}")
print("  (no output = all constraints satisfied)")


# Diebold-Mariano Test and Figure BM5
def diebold_mariano(errors_a, errors_b, h=1):
    """
    Diebold-Mariano (1995) test for equal predictive accuracy.

    H0: Equal predictive accuracy (d_t = loss_a_t - loss_b_t has mean zero)
    H1: Model B significantly outperforms Model A (d_t mean < 0)

    Uses squared error loss. Positive DM statistic => model_b better.
    Newey-West variance correction for serial correlation.

    Returns: {'stat': DM statistic, 'p_value': two-sided p-value,
              'conclusion': string summary}
    """
    d  = errors_a**2 - errors_b**2        # loss differential
    n  = len(d)
    if n < 10:
        return {'stat': np.nan, 'p_value': np.nan, 'conclusion': 'n/a'}
    d_bar  = d.mean()
    gamma0 = ((d - d_bar)**2).mean()
    bw     = max(1, h - 1)
    nw_var = gamma0
    for j in range(1, bw + 1):
        if j < n:
            d1, d2  = d[j:], d[:-j]
            gamma_j = float(np.cov(d1.astype(float), d2.astype(float))[0, 1])
            nw_var += 2 * (1 - j / (bw + 1)) * gamma_j
    nw_var = max(nw_var, 1e-12)
    dm_stat = d_bar / np.sqrt(nw_var / n)
    p_val   = float(2 * (1 - stats.norm.cdf(abs(dm_stat))))
    conc = ('Reject H0 (sig diff)' if p_val < 0.05
            else 'Fail to reject H0')
    return {'stat': round(dm_stat, 3),
            'p_value': round(p_val, 4),
            'conclusion': conc}


# Run DM tests: each model vs Naive ─────────────────────────────────────
print("Computing Diebold-Mariano tests (each model vs Naive)...")
naive_df = results[results['model'] == 'Naive']
dm_rows  = []
for model in results['model'].unique():
    if model == 'Naive':
        continue
    for h in sorted(results['horizon'].unique()):
        naive_h = (naive_df[naive_df['horizon'] == h]
                   .sort_values('target_date'))
        model_h = (results[(results['model'] == model) &
                            (results['horizon'] == h)]
                   .sort_values('target_date'))
        merged  = naive_h[['target_date','error']].merge(
            model_h[['target_date','error']]
                   .rename(columns={'error':'error_m'}),
            on='target_date', how='inner')
        if len(merged) < 20:
            continue
        dm = diebold_mariano(merged['error'].values,
                             merged['error_m'].values, h=h)
        dm_rows.append({'model': model, 'horizon': h,
                        'dm_stat': dm['stat'],
                        'p_value': dm['p_value'],
                        'significant': dm['p_value'] < 0.05,
                        'conclusion': dm['conclusion']})

dm_df = pd.DataFrame(dm_rows)
dm_df.to_csv('dm_tests_NO1.csv', index=False)

print("DM test results (p < 0.05 = significantly better than Naive):")
print()
pivot_pval = dm_df.pivot_table(index='model', columns='horizon',
                                values='p_value').round(4)
print(pivot_pval.to_string())
print()
pivot_sig = dm_df.pivot_table(index='model', columns='horizon',
                               values='significant', aggfunc='first')
print("\nSignificant (True = p < 0.05):")
print(pivot_sig.to_string())

# ── Figure BM5: DM p-value heatmap ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
models_dm  = dm_df['model'].unique()
horizons_dm= sorted(dm_df['horizon'].unique())
pval_mat   = np.ones((len(models_dm), len(horizons_dm)))

for i, model in enumerate(models_dm):
    for j, h in enumerate(horizons_dm):
        sub = dm_df[(dm_df['model']==model) & (dm_df['horizon']==h)]
        if len(sub) > 0:
            pval_mat[i, j] = sub['p_value'].iloc[0]

im = ax.imshow(pval_mat, cmap='RdYlGn', vmin=0, vmax=0.10, aspect='auto')
plt.colorbar(im, ax=ax,
             label='DM p-value  (green < 0.05 = significantly better than Naive)')
ax.set_xticks(range(len(horizons_dm)))
ax.set_xticklabels([f'h={h}d' for h in horizons_dm], fontsize=9)
ax.set_yticks(range(len(models_dm)))
ax.set_yticklabels(models_dm, fontsize=9)
for i in range(len(models_dm)):
    for j in range(len(horizons_dm)):
        pv   = pval_mat[i, j]
        star = '**' if pv < 0.01 else ('*' if pv < 0.05 else 'ns')
        ax.text(j, i, f'{pv:.3f}\n{star}',
                ha='center', va='center', fontsize=8)
ax.set_title('Figure BM5 — Diebold-Mariano Test p-values vs Naive Benchmark\n'
             '(**p<0.01, *p<0.05, ns=not significant)',
             fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('fig_bm5_dm_heatmap.png')
plt.show()


import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': '#f8f9fa',
    'axes.grid': True, 'grid.alpha': 0.35, 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 110, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
})

# ── Constrained Elastic Net (reuse from NB3 / refit here) ─────────────────
COEF_BOUNDS = {
    'fuel_gas_eur_mwh':        (0, 4.0),
    'fuel_carbon_eur_t':       (0, 1.33),
    'hydro_reserve_gwh':       (None, 0),
    'inflow_hbv_gwh':          (None, 0),
    'reservoir_deviation_gwh': (None, 0),
    'derived_inflow_z_score':  (None, 0),
    'forward_M1':              (0, None),
    'forward_Y1':              (0, None),
    'epad_M1':                 (0, None),
    'epad_Y1':                 (0, None),
    'implied_spot_M1':         (0, None),
    'implied_spot_Y1':         (0, None),
    'lag_price_1d':            (0, None),
    'lag_price_365d':          (0, None),
    'price_vs_seasonal':       (0, None),
}

class ConstrainedEN:
    def __init__(self, alpha=0.5, l1_ratio=0.5):
        self.alpha    = alpha
        self.l1_ratio = l1_ratio
        self.scaler   = StandardScaler()
        self.coef_    = None
        self.intercept_ = 0.0
        self.feature_cols_ = []

    def fit(self, X, y):
        self.feature_cols_ = list(X.columns)
        Xn = X.fillna(0).values.astype(float)
        yn = y.values.astype(float)
        Xs = self.scaler.fit_transform(Xn)

        # Warm start with unconstrained EN
        net0 = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio,
                          max_iter=2000, fit_intercept=True)
        net0.fit(Xs, yn)

        bounds = []
        for col, sc in zip(self.feature_cols_, self.scaler.scale_):
            lo, hi = COEF_BOUNDS.get(col, (None, None))
            bounds.append((
                (-np.inf if lo is None else lo * sc),
                ( np.inf if hi is None else hi * sc),
            ))
        bounds.append((None, None))   # intercept

        coef0 = np.clip(net0.coef_,
                        [b[0] if b[0] != -np.inf else -1e6 for b in bounds[:-1]],
                        [b[1] if b[1] !=  np.inf else  1e6 for b in bounds[:-1]])
        x0 = np.append(coef0, net0.intercept_)

        lam, rho = self.alpha, self.l1_ratio
        def obj(p):
            c, ic = p[:-1], p[-1]
            r = yn - (Xs @ c + ic)
            return np.mean(r**2) + lam*rho*np.sum(np.abs(c)) + lam*(1-rho)/2*np.sum(c**2)

        res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-9})
        self.coef_      = res.x[:-1]
        self.intercept_ = res.x[-1]
        return self

    def predict(self, X):
        Xn = X[self.feature_cols_].fillna(0).values.astype(float)
        return self.scaler.transform(Xn) @ self.coef_ + self.intercept_


# ── Choose a representative forecast date ────────────────────────────────
# Use a mid-2023 date: model is fully post-crisis, gas/hydro at normal levels
FC_DATE   = pd.Timestamp('2023-06-01')
WIN_DAYS  = 365 * 3

train_mask = (df['date'] > FC_DATE - pd.Timedelta(days=WIN_DAYS)) & (df['date'] <= FC_DATE)
train_win  = df[train_mask].copy()
print(f"Forecast date: {FC_DATE.date()}")
print(f"Training window: {train_win['date'].min().date()} to {train_win['date'].max().date()}")
print(f"Training observations: {len(train_win)}")

# ── Sensitivity variables and their historical percentiles ────────────────
sens_vars = {
    'fuel_gas_eur_mwh': 'Gas Price (EUR/MWh)',
    'forward_Y1':       'Forward Y1 (EUR/MWh)',
    'fuel_carbon_eur_t':'CO2 Price (EUR/t)',
    'hydro_reserve_gwh':'Hydro Reserve (GWh)',
}
PERCENTILES = [10, 25, 50, 75, 90]

# Compute percentile values from training window
print()
print("Input variable percentiles (training window):")
pct_vals = {}
for var, label in sens_vars.items():
    if var not in df.columns:
        continue
    pcts = np.percentile(train_win[var].dropna(), PERCENTILES)
    pct_vals[var] = pcts
    print(f"  {label:<28} P10={pcts[0]:.1f}  P50={pcts[2]:.1f}  P90={pcts[4]:.1f}")

# ── Fit model on FEATURES_LONG for h=360, FEATURES_MEDIUM for h=180 ───────
results_sens = {}

for h, feat_cols in [(180, FEATURES_MEDIUM), (360, FEATURES_LONG)]:
    feat_cols = [c for c in feat_cols if c in df.columns]
    X_tr = train_win[feat_cols].fillna(0)
    y_tr = train_win['spot_price']

    model = ConstrainedEN().fit(X_tr, y_tr)

    # Baseline: actual values on forecast date
    base_row = df.loc[df['date'] == FC_DATE, feat_cols].fillna(0).copy()
    if len(base_row) == 0:
        print(f"No data for forecast date {FC_DATE.date()}")
        continue
    base_pred = float(model.predict(base_row)[0])

    sens_results = []
    for var in pct_vals:
        if var not in feat_cols:
            continue
        for pct_i, pct_val in zip(PERCENTILES, pct_vals[var]):
            row_mod = base_row.copy()
            row_mod[var] = pct_val
            pred_mod = float(model.predict(row_mod)[0])
            sens_results.append({
                'variable':   var,
                'label':      sens_vars[var],
                'percentile': pct_i,
                'value':      pct_val,
                'pred':       pred_mod,
                'delta':      pred_mod - base_pred,
            })

    results_sens[h] = {
        'base_pred': base_pred,
        'results':   pd.DataFrame(sens_results),
    }
    print(f"\nh={h}d baseline forecast: {base_pred:.1f} EUR/MWh")

#  Figure: Sensitivity tornado charts ────────────────────────────────────
n_horizons = len(results_sens)
if n_horizons == 0:
    print("No sensitivity results to plot — check forecast date and feature columns.")
else:
    fig, axes = plt.subplots(1, n_horizons, figsize=(8*n_horizons, 7))
    if n_horizons == 1:
        axes = [axes]

    fig.suptitle(f'Sensitivity Analysis — Input Uncertainty Effect on Forecast\n'
                 f'Forecast date: {FC_DATE.date()}  |  Inputs varied P10 to P90',
                 fontsize=13, fontweight='bold')

    for ax, (h, res_dict) in zip(axes, results_sens.items()):
        base_pred = res_dict['base_pred']
        sens_df   = res_dict['results']
        tier      = 'MEDIUM' if h <= 180 else 'LONG'

        # For each variable: range of impact = max(delta) - min(delta)
        summary = (sens_df.groupby(['variable', 'label'])['delta']
                   .agg(d_min='min', d_max='max')
                   .reset_index()
                   .assign(range=lambda x: x['d_max'] - x['d_min'])
                   .sort_values('range', ascending=True))

        y_pos = np.arange(len(summary))
        ax.barh(y_pos, summary['d_min'], left=0,
                color='#2196F3', alpha=0.75, label='Below median (P10-P40)')
        ax.barh(y_pos, summary['d_max'], left=0,
                color='#F44336', alpha=0.75, label='Above median (P60-P90)')
        ax.axvline(0, color='black', lw=1.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(summary['label'], fontsize=10)
        ax.set_xlabel('Change in predicted price vs baseline (EUR/MWh)')
        ax.set_title(f'h = {h} days  |  Feature tier: {tier}\n'
                     f'Baseline forecast = {base_pred:.1f} EUR/MWh')
        ax.legend(fontsize=9)

        # Add range annotation
        for j, (_, row) in enumerate(summary.iterrows()):
            ax.text(max(abs(row['d_min']), abs(row['d_max'])) + 0.5,
                    j, f'±{row["range"]/2:.1f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('fig_sensitivity_analysis.png')
    plt.show()
    print("\nSaved: fig_sensitivity_analysis.png")

    # ── Print summary table ────────────────────────────────────────────────
    print()
    print("="*65)
    print("SENSITIVITY SUMMARY — impact of P10 to P90 variation")
    print("="*65)
    for h, res_dict in results_sens.items():
        print(f"\nh = {h} days (baseline = {res_dict['base_pred']:.1f} EUR/MWh):")
        sens_df = res_dict['results']
        for var, label in sens_vars.items():
            sub = sens_df[sens_df['variable'] == var]
            if len(sub) == 0:
                continue
            d_lo  = sub[sub['percentile'] == 10]['delta'].values
            d_hi  = sub[sub['percentile'] == 90]['delta'].values
            if len(d_lo) == 0 or len(d_hi) == 0:
                continue
            print(f"  {label:<30}: P10 effect = {d_lo[0]:+.1f}  "
                  f"P90 effect = {d_hi[0]:+.1f}  "
                  f"Range = {d_hi[0]-d_lo[0]:.1f} EUR/MWh")

    print()





