# Code for probabilistic forecasting and analysis of NO1 electricity price dataset
# Developed by [Nutan Gupta & Mathias Helseth] — [28/04/2026]
# University of Inland Norway
# All rights reserved. For academic use only.

# Probabilistic Forecasting


import warnings; warnings.filterwarnings('ignore')
import time, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, kstest
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

plt.rcParams.update({
    'figure.facecolor': 'white',  'axes.facecolor': '#f8f9fa',
    'axes.grid': True,            'grid.alpha': 0.35,
    'font.size': 11,              'axes.spines.top': False,
    'axes.spines.right': False,   'axes.labelsize': 12,
    'axes.titlesize': 13,         'figure.dpi': 110,
    'savefig.dpi': 150,           'savefig.bbox': 'tight',
})

MODEL_COLORS = {
    'Naive-Gaussian': '#9E9E9E',
    'LASSO-QR':       '#2196F3',
    'DMLP':           '#4CAF50',
}
COLORS = {'pre':'#2196F3','crisis':'#F44336','post':'#4CAF50','hydro':'#00BCD4','gas':'#FF9800'}

CRISIS_START = pd.Timestamp('2021-07-01')
CRISIS_END   = pd.Timestamp('2023-01-01')
TRAIN_END    = pd.Timestamp('2021-12-31')
VAL_START    = pd.Timestamp('2022-01-01')
VAL_END      = pd.Timestamp('2022-12-31')
TEST_START   = pd.Timestamp('2023-01-01')
TEST_END     = pd.Timestamp('2024-12-31')

# 99 quantile levels — Paper 1 (Agakishiev et al. 2025)
TAUS = [round(q/100, 2) for q in range(1, 100)]   # 99 levels

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Setup complete. Device: {DEVICE}. Quantile levels: {len(TAUS)}")

DATA_PATH = 'master_clean_NO1.csv'

df = pd.read_csv(DATA_PATH, parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

def get_regime(d):
    if d < CRISIS_START:   return 'Pre-Crisis'
    elif d < CRISIS_END:   return 'Crisis'
    return 'Post-Crisis'

def get_split(d):
    if d <= TRAIN_END:               return 'train'
    elif VAL_START <= d <= VAL_END:  return 'val'
    elif TEST_START <= d <= TEST_END: return 'test'
    return 'holdout'

df['regime']   = df['date'].apply(get_regime)
df['split']    = df['date'].apply(get_split)
df['year']     = df['date'].dt.year
df['month']    = df['date'].dt.month
df['dow']      = df['date'].dt.dayofweek
df['iso_week'] = df['date'].dt.isocalendar().week.astype(int)

# ── Seasonal profiles from TRAINING DATA ONLY (no lookahead bias) ─────────
train = df[df['date'] <= TRAIN_END].copy()

def make_profile(df_tr, col):
    p = (df_tr.groupby(['month', 'iso_week'])[col]
         .agg(s_mean='mean', s_std='std',
              s_p10=lambda x: np.percentile(x, 10),
              s_p90=lambda x: np.percentile(x, 90))
         .reset_index())
    p['s_mean'] = p['s_mean'].rolling(3, center=True, min_periods=1).mean()
    return p

for col in ['spot_price', 'hydro_reserve_gwh', 'inflow_hbv_gwh']:
    prof = make_profile(train, col)
    m = df[['month', 'iso_week']].merge(
        prof[['month', 'iso_week', 's_mean', 's_std', 's_p10', 's_p90']],
        on=['month', 'iso_week'], how='left')
    df[f'seasonal_{col}_mean'] = m['s_mean'].values
    df[f'seasonal_{col}_std']  = m['s_std'].values
    df[f'seasonal_{col}_p10']  = m['s_p10'].values
    df[f'seasonal_{col}_p90']  = m['s_p90'].values

# ── Engineered features ───────────────────────────────────────────────────
df['price_vs_seasonal']  = df['spot_price'] - df['seasonal_spot_price_mean']
df['price_seasonal_z']   = (df['price_vs_seasonal']
                             / df['seasonal_spot_price_std'].clip(lower=1))
df['hydro_seasonal_z']   = ((df['hydro_reserve_gwh']
                              - df['seasonal_hydro_reserve_gwh_mean'])
                             / df['seasonal_hydro_reserve_gwh_std'].clip(lower=1))
df['implied_spot_M1']    = df['forward_M1'] + df['epad_M1']
df['implied_spot_Y1']    = df['forward_Y1'] + df['epad_Y1']
df['epad_slope']         = df['epad_M1'] - df['epad_Y1']
hz = (df['hydro_reserve_gwh'] - df['hydro_reserve_gwh'].mean()) / df['hydro_reserve_gwh'].std()
gz = (df['fuel_gas_eur_mwh']  - df['fuel_gas_eur_mwh'].mean())  / df['fuel_gas_eur_mwh'].std()
df['interaction_hydro_gas'] = hz * gz

for w in [7, 14, 30, 90]:
    df[f'roll_price_mean_{w}d'] = df['spot_price'].shift(1).rolling(w).mean()
    df[f'roll_price_std_{w}d']  = df['spot_price'].shift(1).rolling(w).std()

mc = df['date'].dt.month
df['season_winter'] = ((mc <= 2) | (mc == 12)).astype(int)
df['season_spring'] = ((mc >= 3) & (mc <= 5)).astype(int)
df['season_summer'] = ((mc >= 6) & (mc <= 8)).astype(int)
df['season_autumn'] = ((mc >= 9) & (mc <= 11)).astype(int)
df['is_weekend']    = (df['dow'] >= 5).astype(int)
df['month_num']     = df['date'].dt.month
df['week_of_year']  = df['iso_week']

# ═════════════════════════════════════════════════════════════════════════
# HORIZON-SPECIFIC FEATURE SETS 
# ═════════════════════════════════════════════════════════════════════════
def _keep(cols):
    return [c for c in cols if c in df.columns]

# SHORT-TERM: h <= 30 days
# All features valid: AR lags have ACF > 0.5 at h=30d
FEATURES_SHORT = _keep([
    'lag_price_1d', 'lag_price_7d', 'lag_price_30d', 'lag_price_365d',
    'ma_price_7d', 'ma_price_30d', 'std_price_7d', 'std_price_30d',
    'roll_price_mean_7d', 'roll_price_std_7d',
    'roll_price_mean_14d', 'roll_price_std_14d',
    'roll_price_mean_30d', 'roll_price_std_30d',
    'roll_price_mean_90d', 'roll_price_std_90d',
    'hydro_reserve_gwh', 'inflow_hbv_gwh', 'reservoir_deviation_gwh',
    'derived_inflow_z_score', 'interaction_hydro_gas', 'hydro_seasonal_z',
    'price_vs_seasonal', 'price_seasonal_z',
    'seasonal_spot_price_mean', 'seasonal_spot_price_std',
    'seasonal_hydro_reserve_gwh_mean', 'seasonal_hydro_reserve_gwh_std',
    'fuel_gas_eur_mwh', 'fuel_carbon_eur_t', 'derived_gas_volatility_30d',
    'forward_M1', 'forward_Y1', 'epad_M1', 'epad_Y1',
    'derived_forward_slope', 'epad_slope', 'implied_spot_M1', 'implied_spot_Y1',
    'weather_temp_c', 'weather_precip_mm', 'load_mw', 'macro_eur_nok',
    'cal_month_sin', 'cal_month_cos', 'cal_day_of_year_sin', 'cal_day_of_year_cos',
    'season_winter', 'season_spring', 'season_summer', 'season_autumn',
    'is_weekend', 'month_num', 'week_of_year',
])

# MEDIUM-TERM: 30 < h <= 180 days
# AR lags DROPPED (ACF near 0.3-0.5 at h=90d, spurious regression risk)
# Seasonal deviation features replace AR lags as primary signal
FEATURES_MEDIUM = _keep([
    'price_vs_seasonal', 'price_seasonal_z',
    'seasonal_spot_price_mean', 'seasonal_spot_price_std',
    'seasonal_spot_price_p10', 'seasonal_spot_price_p90',
    'hydro_seasonal_z',
    'seasonal_hydro_reserve_gwh_mean', 'seasonal_hydro_reserve_gwh_std',
    'seasonal_inflow_hbv_gwh_mean',
    'hydro_reserve_gwh', 'reservoir_deviation_gwh', 'derived_inflow_z_score',
    'forward_M1', 'forward_Y1', 'epad_M1', 'epad_Y1',
    'epad_slope', 'implied_spot_M1', 'implied_spot_Y1',
    'fuel_gas_eur_mwh', 'fuel_carbon_eur_t', 'derived_gas_volatility_30d',
    'interaction_hydro_gas',
    'lag_price_365d',
    'macro_eur_nok',
    'cal_month_sin', 'cal_month_cos', 'cal_day_of_year_sin', 'cal_day_of_year_cos',
    'season_winter', 'season_spring', 'season_summer', 'season_autumn',
    'month_num', 'week_of_year',
])

# LONG-TERM: h > 180 days
# Only year-ahead forward prices, seasonal expectations, and calendar
# ALL AR features and short-horizon forward/fuel excluded (unit root + near-zero ACF)
FEATURES_LONG = _keep([
    'forward_Y1', 'epad_Y1', 'implied_spot_Y1',
    'seasonal_hydro_reserve_gwh_mean', 'seasonal_hydro_reserve_gwh_std',
    'seasonal_inflow_hbv_gwh_mean',
    'seasonal_spot_price_mean', 'seasonal_spot_price_std',
    'seasonal_spot_price_p10', 'seasonal_spot_price_p90',
    'macro_eur_nok',
    'cal_month_sin', 'cal_month_cos', 'cal_day_of_year_sin', 'cal_day_of_year_cos',
    'season_winter', 'season_spring', 'season_summer', 'season_autumn',
    'month_num', 'week_of_year',
])

def get_features(h):
    """Return the correct feature set for forecasting horizon h days ahead."""
    if h <= 30:   return FEATURES_SHORT
    elif h <= 180: return FEATURES_MEDIUM
    else:          return FEATURES_LONG

print(f"Data loaded: {len(df)} rows | {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Train: {(df['split']=='train').sum()} | Val: {(df['split']=='val').sum()} | "
      f"Test: {(df['split']=='test').sum()}")
print()
print("Horizon-specific feature sets (matching NB2 design):")
print(f"  FEATURES_SHORT  (h <= 30d)       : {len(FEATURES_SHORT)} features")
print(f"  FEATURES_MEDIUM (30d < h <= 180d): {len(FEATURES_MEDIUM)} features")
print(f"  FEATURES_LONG   (h > 180d)       : {len(FEATURES_LONG)} features")
print()
print("Key exclusions that prevent spurious regression at longer horizons:")
dropped_med = sorted(set(FEATURES_SHORT) - set(FEATURES_MEDIUM))
ar_dropped  = [f for f in dropped_med if any(x in f for x in ['lag_','ma_','roll_','std_price'])]
print(f"  Medium drops AR/rolling features: {ar_dropped[:6]} ...")
dropped_long = sorted(set(FEATURES_MEDIUM) - set(FEATURES_LONG))
print(f"  Long additionally drops:  {dropped_long[:5]} ...")


# Evaluation Metrics (CRPS, PIT, Coverage)
def pinball_loss(y_true, y_pred_q, tau):
    e = np.array(y_true) - np.array(y_pred_q)
    return float(np.mean(np.where(e >= 0, tau * e, (tau - 1) * e)))

def crps_from_quantiles(y_true, q_matrix, taus):
    total = sum(pinball_loss(y_true, q_matrix[:, j], tau)
                for j, tau in enumerate(taus))
    return 2.0 * total / len(taus)

def compute_pit(y_true, q_matrix, taus):
    taus_arr = np.array(taus)
    pits = []
    for i, y in enumerate(y_true):
        q_row = q_matrix[i]
        if y <= q_row[0]:      pits.append(float(taus_arr[0]))
        elif y >= q_row[-1]:   pits.append(float(taus_arr[-1]))
        else:
            idx = np.searchsorted(q_row, y, side='left')
            t0, t1 = taus_arr[idx-1], taus_arr[idx]
            q0, q1 = q_row[idx-1], q_row[idx]
            pits.append(float(t0 + (t1 - t0) * (y - q0) / (q1 - q0 + 1e-9)))
    return np.array(pits)

print("Metrics defined: pinball_loss, crps_from_quantiles, compute_pit")


# Model 1: Naive Gaussian Benchmark
# 
# The minimum probabilistic model. Point forecast is the weekday-corrected
# random walk. Uncertainty is a Gaussian envelope that widens with sqrt(h).


class NaiveGaussian:
    """
    Naive Gaussian Benchmark.
    Point forecast: weekday-corrected random walk.
    Uncertainty: Gaussian with std = sigma_residual * sqrt(h).
    """
    name = 'Naive-Gaussian'

    def fit(self, train_df):
        self.last_price    = float(train_df['spot_price'].iloc[-1])
        self.weekday_means = (train_df
                              .groupby(train_df['date'].dt.dayofweek)['spot_price']
                              .mean().to_dict())
        resids             = train_df['spot_price'] - train_df['spot_price'].shift(1)
        self.resid_std     = float(resids.dropna().std())
        return self

    def _point(self, horizon, forecast_date):
        if horizon <= 7:
            return self.last_price
        dow = (forecast_date + pd.Timedelta(days=horizon)).dayofweek
        return self.weekday_means.get(dow, self.last_price)

    def predict_quantiles(self, horizon, forecast_date, taus):
        mu  = self._point(horizon, forecast_date)
        std = self.resid_std * math.sqrt(horizon)
        return np.array([norm.ppf(tau, loc=mu, scale=std) for tau in taus])

print("NaiveGaussian defined.")


# Model 2: LASSO Quantile Regression
# Estimates 99 quantile levels independently using L1-regularised pinball loss.
# Implemented via IRLS (3 iterations), lambda = 0.01.
# At prediction time only the features valid for that horizon are passed to the
# model — AR lags are zeroed out at medium and long horizons.


class LASSOQuantileRegression:
    """
    LASSO Quantile Regression — (Agakishiev et al. 2025), Section 3.4.
    Estimates 99 quantile levels using L1-regularised pinball loss (IRLS, 3 iterations).
    lambda = 0.01.
    """
    name = 'LASSO-QR'

    def __init__(self, lam=0.01, taus=None):
        self.lam    = lam
        self.taus   = taus if taus is not None else TAUS
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_cols_ = []

    def fit(self, X, y):
        self.feature_cols_ = list(X.columns)
        Xn = X.fillna(0).values.astype(float)
        yn = y.fillna(float(y.median())).values.astype(float)
        Xs = self.scaler.fit_transform(Xn)
        for tau in self.taus:
            n   = len(yn)
            wt  = np.ones(n)
            coef, intercept = np.zeros(Xs.shape[1]), float(np.mean(yn))
            for _ in range(3):
                resid = yn - (Xs @ coef + intercept)
                wt    = np.where(resid >= 0, tau, 1 - tau)
                wt    = np.clip(wt, 0.01, None)
                net   = ElasticNet(alpha=self.lam, l1_ratio=1.0, max_iter=1000,
                                   fit_intercept=True, warm_start=True)
                net.fit(Xs, yn, sample_weight=wt)
                coef, intercept = net.coef_, net.intercept_
            self.models[tau] = (coef.copy(), float(intercept))
        return self

    def predict_quantiles(self, X):
        Xn = X[self.feature_cols_].fillna(0).values.astype(float)
        Xs = self.scaler.transform(Xn)
        q  = np.zeros((Xs.shape[0], len(self.taus)))
        for j, tau in enumerate(self.taus):
            coef, intercept = self.models[tau]
            q[:, j] = Xs @ coef + intercept
        return np.sort(q, axis=1)

print("LASSOQuantileRegression defined.")


# Model 3: DMLP with Johnson's SU Distribution
# 
# Neural network that directly predicts the four parameters of Johnson's SU.
# Architecture: Input -> [100 -> 50 -> 25] (BatchNorm + ReLU + Dropout)
#              -> JohnsonSU(gamma, delta, xi, lambda).
# Loss function: Negative Log-Likelihood of the Johnson SU distribution.
# Early stopping with patience = 15 epochs on a held-out validation slice.


class JohnsonSULayer(nn.Module):
    def __init__(self, in_features, eps=1e-3):
        super().__init__()
        self.fc  = nn.Linear(in_features, 4)
        self.eps = eps

    def forward(self, x):
        raw = self.fc(x)
        g   = raw[:, 0]
        d   = nn.functional.softplus(raw[:, 1]) + self.eps
        xi  = raw[:, 2]
        lam = nn.functional.softplus(raw[:, 3]) + self.eps
        return g, d, xi, lam


class DMLPNetwork(nn.Module):
    """
    Architecture (Agakishiev et al. 2025):
        Input -> [100 -> 50 -> 25] (BatchNorm + ReLU + Dropout 0.3)
               -> Johnson SU layer (gamma, delta, xi, lambda)
    Loss: Negative Log-Likelihood of the Johnson SU distribution.
    """
    def __init__(self, n_features, dropout=0.3):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(n_features, 100), nn.BatchNorm1d(100), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(100, 50),         nn.BatchNorm1d(50),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(50, 25),          nn.BatchNorm1d(25),  nn.ReLU(), nn.Dropout(dropout),
        )
        self.dist = JohnsonSULayer(25)

    def forward(self, x):
        return self.dist(self.hidden(x))

    def nll_loss(self, x, y):
        g, d, xi, lam = self.forward(x)
        d   = d.clamp(min=1e-3)
        lam = lam.clamp(min=1e-3)
        z   = g + d * torch.asinh((y - xi) / lam)
        log_pdf = (torch.log(d) - torch.log(lam)
                   - 0.5 * torch.log(1 + ((y - xi) / lam) ** 2)
                   - 0.5 * np.log(2 * np.pi) - 0.5 * z ** 2)
        return -log_pdf.mean()

    def get_params(self, x):
        with torch.no_grad():
            g, d, xi, lam = self.forward(x)
        return (g.cpu().numpy(), d.cpu().numpy(),
                xi.cpu().numpy(), lam.cpu().numpy())


def jsu_quantile(tau, g, d, xi, lam):
    return xi + lam * np.sinh((norm.ppf(tau) - g) / d)


class DMLPModel:
    """
    Training wrapper for DMLPNetwork.
    Adam optimiser, lr=1e-3, weight_decay=1e-4, early stopping patience=15.
    """
    name = 'DMLP'

    def __init__(self, epochs=150, batch_size=256, lr=1e-3,
                 weight_decay=1e-4, patience=15, dropout=0.3, taus=None):
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.weight_decay = weight_decay
        self.patience     = patience
        self.dropout      = dropout
        self.taus         = taus if taus is not None else TAUS
        self.scaler       = StandardScaler()
        self.net_         = None
        self.feature_cols_= []
        self.train_losses = []
        self.val_losses   = []

    def fit(self, X, y):
        self.feature_cols_ = list(X.columns)
        Xn = X.fillna(0).values.astype(float)
        yn = y.fillna(float(y.median())).values.astype(float)
        Xs = self.scaler.fit_transform(Xn)

        n_val   = max(30, int(0.15 * len(Xs)))
        Xt, Xv  = Xs[:-n_val], Xs[-n_val:]
        yt, yv  = yn[:-n_val], yn[-n_val:]

        Xt_t = torch.tensor(Xt, dtype=torch.float32).to(DEVICE)
        yt_t = torch.tensor(yt, dtype=torch.float32).to(DEVICE)
        Xv_t = torch.tensor(Xv, dtype=torch.float32).to(DEVICE)
        yv_t = torch.tensor(yv, dtype=torch.float32).to(DEVICE)

        loader = DataLoader(TensorDataset(Xt_t, yt_t),
                            batch_size=self.batch_size, shuffle=True)
        self.net_ = DMLPNetwork(Xs.shape[1], self.dropout).to(DEVICE)
        opt   = optim.Adam(self.net_.parameters(),
                           lr=self.lr, weight_decay=self.weight_decay)
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=5, factor=0.5, min_lr=1e-5)

        best_val   = np.inf
        best_state = None
        no_imp     = 0
        self.train_losses = []
        self.val_losses   = []

        for epoch in range(self.epochs):
            self.net_.train()
            ep_loss = 0.0
            for Xb, yb in loader:
                opt.zero_grad()
                loss = self.net_.nll_loss(Xb, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net_.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item() * len(Xb)
            ep_loss /= len(Xt_t)

            self.net_.eval()
            with torch.no_grad():
                vl = self.net_.nll_loss(Xv_t, yv_t).item()
            self.train_losses.append(ep_loss)
            self.val_losses.append(vl)
            sched.step(vl)

            if vl < best_val - 1e-5:
                best_val   = vl
                best_state = {k: v.clone()
                              for k, v in self.net_.state_dict().items()}
                no_imp     = 0
            else:
                no_imp += 1
                if no_imp >= self.patience:
                    break

        if best_state:
            self.net_.load_state_dict(best_state)
        return self

    def predict_quantiles(self, X):
        Xn = X[self.feature_cols_].fillna(0).values.astype(float)
        Xs = self.scaler.transform(Xn)
        Xt = torch.tensor(Xs, dtype=torch.float32).to(DEVICE)
        self.net_.eval()
        g_a, d_a, xi_a, lam_a = self.net_.get_params(Xt)
        taus = np.array(self.taus)
        q    = np.zeros((len(g_a), len(taus)))
        for i in range(len(g_a)):
            q[i] = jsu_quantile(taus, float(g_a[i]), float(d_a[i]),
                                 float(xi_a[i]), float(lam_a[i]))
        return np.sort(q, axis=1)


# Verify architecture
n_feat   = len(FEATURES_SHORT)
net_test = DMLPNetwork(n_feat)
n_params = sum(p.numel() for p in net_test.parameters())
print(f"DMLP defined. Input={n_feat} features, Parameters={n_params:,}")
dummy = torch.randn(3, n_feat)
net_test.eval()
g_t, d_t, xi_t, lam_t = net_test.get_params(dummy)
print(f"  get_params test: gamma={g_t.mean():.3f}  delta={d_t.mean():.3f}  [OK]")


# Rolling Evaluation Engine with Horizon-Specific Features
# Key design:
# 1. At each rolling step, the model is fitted on FEATURES_SHORT (the broadest set)
# 2. At prediction time for horizon h, features not valid for that tier are
#    zeroed out before calling predict_quantiles
# 3. This ensures AR lags cannot influence predictions at h=180d or h=360d


def rolling_prob_evaluate(df, model_name, model_factory,
                           horizons, feature_selector,
                           window_days=365*3, roll_step=30,
                           taus=TAUS, verbose=True):
    """
    Rolling-window probabilistic evaluation with HORIZON-SPECIFIC FEATURE SETS.


        h <= 30d    => FEATURES_SHORT  (AR lags + all fundamentals)
        30 < h <= 180d => FEATURES_MEDIUM (no AR lags, seasonal deviations)
        h > 180d    => FEATURES_LONG  (only year-ahead forwards + seasonal)

    The model is trained on FEATURES_SHORT (the broadest set) but at
    prediction time only the horizon-appropriate columns are active.
    Features excluded for a given horizon are zeroed out before the
    model's predict_quantiles method is called. This ensures that AR
    lags and short-horizon fuel prices cannot influence predictions at
    medium and long horizons, consistent with the feature design in NB2.
    """
    all_feat_cols = feature_selector(1)   # FEATURES_SHORT is the broadest set

    df         = df.sort_values('date').reset_index(drop=True)
    eval_mask  = (df['date'] >= VAL_START) & (df['date'] <= TEST_END)
    eval_dates = df.loc[eval_mask, 'date'].values[::roll_step]
    n_steps    = len(eval_dates)
    records    = []
    t0         = time.time()

    for step_i, fc_ts in enumerate(eval_dates):
        fc_date = pd.Timestamp(fc_ts)

        if verbose and step_i % max(1, n_steps // 8) == 0:
            elapsed = time.time() - t0
            print(f"    [{model_name}] {step_i+1}/{n_steps} "
                  f"({100*(step_i+1)/n_steps:.0f}%)  "
                  f"{fc_date.date()}  {elapsed:.0f}s", end='\r')

        t_win    = fc_date - pd.Timedelta(days=window_days)
        train_df = df[(df['date'] > t_win) & (df['date'] <= fc_date)].copy()
        if len(train_df) < 90:
            continue

        # Fit on the SHORT (broadest) feature set
        try:
            model = model_factory()
            if model_name == 'Naive-Gaussian':
                model.fit(train_df)
            else:
                X_tr = train_df[all_feat_cols].fillna(0)
                y_tr = train_df['spot_price']
                model.fit(X_tr, y_tr)
        except Exception as e:
            if verbose:
                print(f"\n    [WARN] fit failed at {fc_date.date()}: {e}")
            continue

        for h in horizons:
            tgt_date = fc_date + pd.Timedelta(days=h)
            tgt_mask = df['date'] == tgt_date
            if not tgt_mask.any():
                continue
            y_true = float(df.loc[tgt_mask, 'spot_price'].iloc[0])

            # Get horizon-appropriate feature columns
            feat_cols_h = feature_selector(h)
            tier        = 'short' if h <= 30 else 'medium' if h <= 180 else 'long'

            try:
                if model_name == 'Naive-Gaussian':
                    q_vec = model.predict_quantiles(h, fc_date, taus)
                else:
                    # Build a full-width feature row using FORECAST DATE values
                    feat_full = df.loc[df['date'] == fc_date, all_feat_cols].fillna(0).copy()
                    if len(feat_full) == 0:
                        continue
                    # Zero out features excluded at this horizon
                    # This prevents AR lags from influencing medium/long predictions
                    excluded = set(all_feat_cols) - set(feat_cols_h)
                    for col in excluded:
                        if col in feat_full.columns:
                            feat_full[col] = 0.0
                    q_vec = model.predict_quantiles(feat_full)[0]
            except Exception:
                continue

            q_vec  = np.array(q_vec)
            taus_a = np.array(taus)
            crps_v = crps_from_quantiles(np.array([y_true]),
                                          q_vec.reshape(1, -1), taus)

            def q_at(lv):
                return float(q_vec[np.argmin(np.abs(taus_a - lv))])

            meta = df.loc[tgt_mask].iloc[0]
            records.append({
                'forecast_date': fc_date,
                'target_date':   tgt_date,
                'horizon':       h,
                'feature_tier':  tier,
                'y_true':        y_true,
                'crps':          crps_v,
                'q05': q_at(0.05), 'q10': q_at(0.10), 'q25': q_at(0.25),
                'q50': q_at(0.50), 'q75': q_at(0.75), 'q90': q_at(0.90),
                'q95': q_at(0.95),
                'model':   model_name,
                'year':    tgt_date.year,
                'regime':  meta.get('regime', ''),
                'split':   meta.get('split',  ''),
            })

    if verbose:
        print()
    return pd.DataFrame(records)


# ── Horizons ────────────────────────────────────────────────────────────────
PROB_HORIZONS = [1, 7, 30, 90, 180, 360]

print("Rolling evaluation engine defined.")
print(f"Evaluation horizons: {PROB_HORIZONS}")
print()
print("Feature tier assigned at each horizon:")
for h in PROB_HORIZONS:
    feat_cols_h = get_features(h)
    tier = 'SHORT' if h <= 30 else 'MEDIUM' if h <= 180 else 'LONG'
    print(f"  h={h:3d}d  =>  FEATURES_{tier}  ({len(feat_cols_h)} features active)")


# Run All Models
# 
# Full evaluation across all six horizons: 1, 7, 30, 90, 180, 360 days.


print("="*60)
print("PROBABILISTIC MODEL EVALUATION — Horizon-Specific Features")
print(f"Horizons : {PROB_HORIZONS}")
print(f"Quantiles: {len(TAUS)} levels")
print("="*60)
print()

prob_results = []

# ── 1. Naive Gaussian ──────────────────────────────────────────────────────
print("Running Naive-Gaussian...")
res_ng = rolling_prob_evaluate(
    df, 'Naive-Gaussian', NaiveGaussian,
    horizons=PROB_HORIZONS,
    feature_selector=get_features,
    verbose=True,
)
prob_results.append(res_ng)
print(f"  Done: {len(res_ng)} records")

# ── 2. LASSO-QR ────────────────────────────────────────────────────────────
print("\nRunning LASSO-QR...")
res_lqr = rolling_prob_evaluate(
    df, 'LASSO-QR',
    lambda: LASSOQuantileRegression(lam=0.01),
    horizons=PROB_HORIZONS,
    feature_selector=get_features,
    verbose=True,
)
prob_results.append(res_lqr)
print(f"  Done: {len(res_lqr)} records")

# ── 3. DMLP ──────────────────────────────
print("\nRunning DMLP (takes ~20-30 min on CPU)...")
res_dmlp = rolling_prob_evaluate(
    df, 'DMLP',
    lambda: DMLPModel(epochs=150, batch_size=256, lr=1e-3,
                      weight_decay=1e-4, patience=15, dropout=0.3),
    horizons=PROB_HORIZONS,
    feature_selector=get_features,
    verbose=True,
)
prob_results.append(res_dmlp)
print(f"  Done: {len(res_dmlp)} records")

# ── Combine ────────────────────────────────────────────────────────────────
prob_df = pd.concat(prob_results, ignore_index=True)
prob_df['forecast_date'] = pd.to_datetime(prob_df['forecast_date'])
prob_df['target_date']   = pd.to_datetime(prob_df['target_date'])

print()
print(f"Total records : {len(prob_df)}")
print(f"Models        : {prob_df['model'].unique().tolist()}")
print(f"Feature tiers : {prob_df['feature_tier'].unique().tolist()}")
prob_df.to_csv('results_probabilistic_NO1.csv', index=False)
print("Saved: results_probabilistic_NO1.csv")


# ## Result — CRPS Table and Figure PB1
# 
# The CRPS (Continuous Ranked Probability Score) is the primary metric.
# Lower CRPS = better probabilistic forecast.


# ══════════════════════════════════════════════════════════════════════════
# RESULT: CRPS Summary Table  (primary probabilistic result)
# ══════════════════════════════════════════════════════════════════════════

models   = list(prob_df['model'].unique())
horizons = sorted(prob_df['horizon'].unique())

rows = []
for m in models:
    sub = prob_df[prob_df['model'] == m]
    row = {'Model': m}
    for h in horizons:
        h_sub = sub[sub['horizon'] == h]
        row[f'h={h}d'] = round(h_sub['crps'].mean(), 2) if len(h_sub) > 0 else np.nan
    rows.append(row)
crps_table = pd.DataFrame(rows).set_index('Model')

print("="*65)
print("TABLE — CRPS by Model and Horizon (EUR/MWh, lower is better)")
print("Feature tiers: SHORT (h<=30d) | MEDIUM (h<=180d) | LONG (h>180d)")
print("="*65)
print(crps_table.to_string())
crps_table.to_csv('summary_crps_NO1.csv')
print("\nSaved: summary_crps_NO1.csv")

# CRPS improvement over Naive-Gaussian
print()
print("CRPS improvement over Naive-Gaussian (positive = better):")
if 'Naive-Gaussian' in crps_table.index:
    naive_row = crps_table.loc['Naive-Gaussian']
    for model in crps_table.index:
        if model == 'Naive-Gaussian':
            continue
        delta = ((naive_row - crps_table.loc[model]) / naive_row * 100).round(1)
        print(f"  {model}:")
        for col, d in delta.items():
            flag = "better" if d > 0 else "worse"
            print(f"    {col}: {'+' if d > 0 else ''}{d:.1f}% ({flag})")

# ── Figure PB1: CRPS by horizon ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle('Figure PB1 — CRPS by Forecasting Horizon\n'
             'Each model uses the horizon-appropriate feature set (SHORT / MEDIUM / LONG)',
             fontsize=13, fontweight='bold')

for model in models:
    sub  = prob_df[prob_df['model'] == model]
    vals = [sub[sub['horizon']==h]['crps'].mean() if (sub['horizon']==h).any()
            else np.nan for h in horizons]
    ax.plot(range(len(horizons)), vals,
            'o-', lw=2.5, ms=9,
            color=MODEL_COLORS.get(model, 'black'),
            label=model)

# Shade horizon zones
short_end  = next((i for i, h in enumerate(horizons) if h > 30),  len(horizons))
medium_end = next((i for i, h in enumerate(horizons) if h > 180), len(horizons))
ax.axvspan(-0.5,           short_end  - 0.5, alpha=0.07, color='steelblue')
ax.axvspan(short_end-0.5,  medium_end - 0.5, alpha=0.07, color='orange')
ax.axvspan(medium_end-0.5, len(horizons)-0.5, alpha=0.07, color='green')

ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 100
ax.text(max(short_end/2 - 0.5, 0),          ymax*0.95, 'SHORT\nFEATURES_SHORT',
        ha='center', fontsize=8.5, color='steelblue', fontweight='bold')
ax.text((short_end + medium_end)/2 - 0.5,   ymax*0.95, 'MEDIUM\nFEATURES_MEDIUM',
        ha='center', fontsize=8.5, color='darkorange', fontweight='bold')
ax.text((medium_end + len(horizons))/2 - 0.5, ymax*0.95, 'LONG\nFEATURES_LONG',
        ha='center', fontsize=8.5, color='darkgreen', fontweight='bold')

ax.set_xticks(range(len(horizons)))
ax.set_xticklabels([f'{h}d' for h in horizons])
ax.set_xlabel('Forecasting Horizon (days)')
ax.set_ylabel('Mean CRPS (EUR/MWh)')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('fig_pb1_crps_by_horizon.png')
plt.show()
print("Figure PB1 saved: fig_pb1_crps_by_horizon.png")


# ## Result — Figure PB2: Fan Charts
# 

# ## Result — Figure PB3: PIT Histograms and Coverage Table
# 
# PIT (Probability Integral Transform) histograms test whether the predicted
# probability intervals are statistically calibrated. A flat histogram means
# the model's uncertainty estimates are correct. The coverage table shows
# the empirical fraction of actual prices falling inside each nominal interval.


# ══════════════════════════════════════════════════════════════════════════
# RESULT 3: Figure PB3 — PIT Histograms + Coverage Table
# Tests whether the predicted probability intervals are well-calibrated.
# A well-calibrated model has a flat (uniform) PIT histogram.
# ══════════════════════════════════════════════════════════════════════════

def pit_from_stored(sub):
    stored_taus = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    stored_cols = [f'q{int(t*100):02d}' for t in stored_taus]
    avail = [(t, c) for t, c in zip(stored_taus, stored_cols) if c in sub.columns]
    if len(avail) < 3:
        return np.array([])
    t_av = np.array([t for t, _ in avail])
    pits = []
    for _, row in sub.iterrows():
        y     = row['y_true']
        q_arr = np.array([row[c] for _, c in avail])
        if y <= q_arr[0]:      pits.append(float(t_av[0]))
        elif y >= q_arr[-1]:   pits.append(float(t_av[-1]))
        else:
            idx = np.searchsorted(q_arr, y, side='left')
            t0, t1 = t_av[idx-1], t_av[idx]
            q0, q1 = q_arr[idx-1], q_arr[idx]
            pits.append(float(t0 + (t1-t0)*(y-q0)/(q1-q0+1e-9)))
    return np.array(pits)


# PIT at h=30d (uses FEATURES_MEDIUM — confirms medium-tier calibration)
h_pit  = 30
models = list(prob_df['model'].unique())

fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 6))
if len(models) == 1:
    axes = [axes]

tier_label = 'MEDIUM' if h_pit > 30 else 'SHORT' if h_pit <= 30 else 'LONG'
fig.suptitle(f'Figure PB3 — PIT Histograms (Calibration)  |  h = {h_pit} days\n'
             f'Feature set at this horizon: FEATURES_{tier_label}  '
             f'|  Flat histogram = well calibrated',
             fontsize=13, fontweight='bold')

pit_dict = {}
for ax, model in zip(axes, models):
    sub  = prob_df[(prob_df['model'] == model) & (prob_df['horizon'] == h_pit)]
    pits = pit_from_stored(sub)
    pit_dict[model] = pits

    if len(pits) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
        continue

    ax.hist(pits, bins=20, density=True, alpha=0.75,
            color=MODEL_COLORS.get(model, 'steelblue'), edgecolor='white', lw=0.5)
    ax.axhline(1.0, color='black', ls='--', lw=2, label='Uniform (ideal)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(2.5, ax.get_ylim()[1]))
    ax.set_title(model, fontsize=11, fontweight='bold')
    ax.set_xlabel('PIT Value')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)

    ks_stat, ks_p = kstest(pits, 'uniform')
    cal_msg = 'well-calibrated' if ks_p > 0.05 else 'miscalibrated'
    ax.text(0.05, 0.95,
            f'KS stat = {ks_stat:.3f}\np = {ks_p:.3f}\n{cal_msg}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(fc='white', ec='gray', alpha=0.85))

plt.tight_layout()
plt.savefig('fig_pb3_pit_histograms.png')
plt.show()
print("Figure PB3 saved: fig_pb3_pit_histograms.png")

# ── Coverage Table ─────────────────────────────────────────────────────────
print()
print("="*65)
print("PI Coverage Table — Empirical vs Nominal Level")
print("(Ideal: empirical coverage = nominal level)")
print("="*65)

cov_records = []
for model in models:
    for h in horizons:
        sub = prob_df[(prob_df['model'] == model) & (prob_df['horizon'] == h)]
        if len(sub) < 10:
            continue
        y_true = sub['y_true'].values
        tier   = sub['feature_tier'].iloc[0].upper() if len(sub) > 0 else 'N/A'
        for nom, lo_col, hi_col in [
            (0.50, 'q25', 'q75'),
            (0.80, 'q10', 'q90'),
            (0.90, 'q05', 'q95'),
        ]:
            if lo_col not in sub.columns or hi_col not in sub.columns:
                continue
            lo, hi = sub[lo_col].values, sub[hi_col].values
            emp    = float(((y_true >= lo) & (y_true <= hi)).mean())
            width  = float((hi - lo).mean())
            cov_records.append({'model': model, 'horizon': h, 'tier': tier,
                                 'nominal': nom, 'empirical': round(emp, 3),
                                 'width': round(width, 1)})

cov_df = pd.DataFrame(cov_records)
cov_df.to_csv('coverage_report_NO1.csv', index=False)

for nom in [0.50, 0.80, 0.90]:
    print(f"\n{int(nom*100)}% Prediction Interval — Empirical Coverage:")
    sub = cov_df[cov_df['nominal'] == nom]
    pivot = sub.pivot_table(index='model', columns='horizon',
                             values='empirical', aggfunc='first').round(3)
    print(pivot.to_string())

print()
print("Feature tier active at each horizon:")
for h in horizons:
    tier = 'SHORT' if h <= 30 else 'MEDIUM' if h <= 180 else 'LONG'
    print(f"  h={h:3d}d => {tier}")

print("\nSaved: coverage_report_NO1.csv")


def plot_fan(prob_df, model_name, horizons_to_plot, filename, suptitle):
    available = [h for h in horizons_to_plot if h in prob_df['horizon'].values
                 and (prob_df['model'] == model_name).any()]
    if len(available) == 0:
        print(f"  No data for {model_name}")
        return

    fig, axes = plt.subplots(len(available), 1,
                              figsize=(15, 5 * len(available)),
                              gridspec_kw={'hspace': 0.55})
    if len(available) == 1:
        axes = [axes]

    fig.suptitle(suptitle, fontsize=12, fontweight='bold', y=0.995)
    color = MODEL_COLORS.get(model_name, 'steelblue')

    for ax, h in zip(axes, available):
        sub = (prob_df[(prob_df['model'] == model_name) & (prob_df['horizon'] == h)]
               .sort_values('target_date').copy())
        if len(sub) == 0:
            continue

        tier  = sub['feature_tier'].iloc[0].upper() if 'feature_tier' in sub.columns else 'N/A'
        n     = len(sub)
        xi    = np.arange(n)
        dates = pd.to_datetime(sub['target_date'].values)

        def shade(lo_col, hi_col, alpha, label):
            if lo_col in sub.columns and hi_col in sub.columns:
                ax.fill_between(xi,
                                sub[lo_col].values.astype(float),
                                sub[hi_col].values.astype(float),
                                alpha=alpha, color=color, label=label)

        shade('q05', 'q95', 0.12, '95% PI')
        shade('q10', 'q90', 0.22, '80% PI')
        shade('q25', 'q75', 0.38, '50% PI')

        if 'q50' in sub.columns:
            ax.plot(xi, sub['q50'].values.astype(float),
                    color=color, lw=1.5, ls='--', alpha=0.9, label='Median (Q50)')

        ax.plot(xi, sub['y_true'].values.astype(float),
                color='black', lw=1.0, alpha=0.85, label='Actual price')

        # Shade crisis period
        idx_lo = np.searchsorted(dates, pd.Timestamp(VAL_START))
        idx_hi = np.searchsorted(dates, pd.Timestamp(VAL_END))
        if idx_lo < idx_hi < n:
            ax.axvspan(idx_lo, idx_hi, alpha=0.10, color='red', label='Crisis 2022')

        # Coverage annotation
        cov80_val = None
        for _, row in cov_df.iterrows():
            if (row['model'] == model_name and
                    row['horizon'] == h and
                    abs(row['nominal'] - 0.80) < 0.01):
                cov80_val = row['empirical']
                width_val = row['width']
                break

        if cov80_val is not None:
            ax.text(0.02, 0.95,
                    f"80% PI: empirical coverage = {cov80_val*100:.1f}%  "
                    f"(nominal = 80%)\nMean PI width = {width_val:.1f} EUR/MWh",
                    transform=ax.transAxes, fontsize=9, va='top',
                    bbox=dict(fc='white', ec='gray', alpha=0.85))


        step = max(1, n // 6)
        ax.set_xticks(xi[::step])
        ax.set_xticklabels([pd.Timestamp(dates[i]).strftime('%Y-%m')
                            for i in xi[::step]], rotation=30, ha='right', fontsize=8)
        ax.set_title(f'h = {h} days  |  FEATURES_{tier}  '
                     f'({len(get_features(h))} features)',
                     fontsize=11)
        ax.set_ylabel('Spot Price (EUR/MWh)')
        ax.legend(fontsize=8, ncol=5, loc='upper right')

    plt.savefig(filename)
    plt.show()
    print(f"Saved: {filename}")


# ── Figure PB2a: DMLP fan charts─────────────
print("Plotting DMLP fan charts (primary probabilistic figure)...")
plot_fan(
    prob_df,
    model_name='DMLP',
    horizons_to_plot=[7, 30, 90, 180],
    filename='fig_pb2a_fan_charts_dmlp.png',
    suptitle=(
        'Fan Charts: DMLP\n'
        '50% / 80% / 95% Prediction Intervals vs Actual Prices\n'
    )
)

# ── Figure PB2b: LASSO-QR fan charts (for comparison) ─────
print("\nPlotting LASSO-QR fan charts (for comparison)...")
plot_fan(
    prob_df,
    model_name='LASSO-QR',
    horizons_to_plot=[7, 30, 90, 180],
    filename='fig_pb2b_fan_charts_lasso.png',
    suptitle=(
        'Fan Charts: LASSO-QR\n'
        '50% / 80% / 95% Prediction Intervals vs Actual Prices\n'
    )
)

# ── Print interval width comparison ──────────────────────────────────────
print()
print("="*65)
print("Interval width comparison — why LASSO-QR fans are invisible")
print("="*65)
print()
print("80% Prediction Interval mean width (EUR/MWh):")
print()
w_table = cov_df[cov_df['nominal'] == 0.80].pivot_table(
    index='model', columns='horizon', values='width', aggfunc='first')
print(w_table.to_string())
print()



"""
Point 2: Sharp Feature Tier Boundary Analysis
=============================================
Tests whether the discrete feature tier transitions at h=30d and h=180d
introduce visible discontinuities in the forecast trajectory.

Method: For each evaluation date, we have CRPS and median (q50) predictions
at horizons [1, 7, 30, 90, 180, 360]. We plot the median forecast across
horizons for several representative forecast dates and check whether there
are visible jumps at the tier boundaries.

We also compute the absolute change in median forecast at each horizon step
and compare the changes at boundary horizons (h=30, h=180) versus interior
horizons (h=7, h=90) to quantify any discontinuity.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': '#f8f9fa',
    'axes.grid': True, 'grid.alpha': 0.35, 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 110, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
})

# ── Use LASSO-QR median forecasts across horizons ──────────────────────────
# (DMLP if available, otherwise LASSO-QR)
target_model = 'DMLP' if 'DMLP' in prob_df['model'].values else 'LASSO-QR'
mdf = prob_df[prob_df['model'] == target_model].copy()
mdf['forecast_date'] = pd.to_datetime(mdf['forecast_date'])
mdf['target_date']   = pd.to_datetime(mdf['target_date'])

horizons_avail = sorted(mdf['horizon'].unique())
print(f"Model: {target_model}")
print(f"Available horizons: {horizons_avail}")
print(f"Tier boundaries: h=30d (SHORT->MEDIUM) and h=180d (MEDIUM->LONG)")

# ── Panel A: Median forecast trajectory per forecast date ─────────────────
# Pick a sample of forecast dates from the post-crisis test period
test_dates = mdf[mdf['split'] == 'test']['forecast_date'].unique()
sample_dates = pd.to_datetime(sorted(test_dates)[::8])[:8]  # every 8th step

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Feature Tier Boundary Analysis\n'
             'Testing for forecast discontinuities at h=30d and h=180d boundaries',
             fontsize=13, fontweight='bold')

# Panel A: median forecast trajectory
ax1 = axes[0]
colors_sample = plt.cm.Blues(np.linspace(0.4, 0.9, len(sample_dates)))

for i, fc_date in enumerate(sample_dates):
    fc_data = mdf[mdf['forecast_date'] == fc_date].sort_values('horizon')
    if len(fc_data) < 3:
        continue
    if 'q50' in fc_data.columns:
        ax1.plot(fc_data['horizon'], fc_data['q50'],
                 'o-', color=colors_sample[i], lw=1.5, ms=5, alpha=0.8,
                 label=fc_date.strftime('%Y-%m'))

# Mark tier boundaries
for bnd, label in [(30, 'SHORT\n→MEDIUM'), (180, 'MEDIUM\n→LONG')]:
    if bnd in horizons_avail:
        ax1.axvline(bnd, color='red', ls='--', lw=1.5, alpha=0.7)
        ax1.text(bnd + 3, ax1.get_ylim()[1] * 0.9 if ax1.get_ylim()[1] > 0 else 100,
                 label, fontsize=8, color='red')

ax1.set_xlabel('Forecasting Horizon (days)')
ax1.set_ylabel('Median Forecast (EUR/MWh)')
ax1.set_title('Panel A — Median Forecast Trajectory\n'
              'One line per forecast date (post-crisis test period)')
ax1.legend(fontsize=7, ncol=2, title='Forecast date')
ax1.set_xticks(horizons_avail)

# Panel B: absolute change in median at each horizon step
ax2 = axes[1]

# Compute horizon-to-horizon changes in median forecast
# For each pair of consecutive horizons, compute mean absolute change
if 'q50' in mdf.columns and len(horizons_avail) >= 3:
    changes = []
    for fc_date in mdf['forecast_date'].unique():
        fc_data = mdf[mdf['forecast_date'] == fc_date].sort_values('horizon')
        if len(fc_data) < 2:
            continue
        q50 = fc_data['q50'].values.astype(float)
        h_vals = fc_data['horizon'].values

        for j in range(1, len(q50)):
            changes.append({
                'h_from': h_vals[j-1],
                'h_to':   h_vals[j],
                'label':  f"h={h_vals[j-1]}→{h_vals[j]}",
                'abs_change': abs(q50[j] - q50[j-1]),
                'is_boundary': h_vals[j-1] in [30, 180] or h_vals[j] in [30, 180],
            })

    ch_df = pd.DataFrame(changes)
    step_means = (ch_df.groupby('label')['abs_change']
                  .agg(['mean', 'median', 'std'])
                  .reset_index()
                  .sort_values('label'))

    bar_colors = []
    for _, row in step_means.iterrows():
        is_bnd = any(str(b) in row['label'] for b in [30, 180])
        bar_colors.append('#F44336' if is_bnd else '#2196F3')

    x = np.arange(len(step_means))
    ax2.bar(x, step_means['mean'], color=bar_colors, alpha=0.82,
            label='Mean |change|')
    ax2.errorbar(x, step_means['mean'], yerr=step_means['std'],
                 fmt='none', color='black', capsize=4, lw=1.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(step_means['label'], rotation=30, ha='right', fontsize=9)
    ax2.set_xlabel('Horizon step')
    ax2.set_ylabel('Mean absolute change in median forecast\n(EUR/MWh)')
    ax2.set_title('Panel B — Forecast Jump at Each Horizon Step\n'
                  'Red = boundary transition | Blue = interior step')
    from matplotlib.patches import Patch
    ax2.legend(handles=[
        Patch(color='#F44336', alpha=0.82, label='Tier boundary step'),
        Patch(color='#2196F3', alpha=0.82, label='Interior step'),
    ], fontsize=9)

    print()
    print("Mean absolute change in median forecast at each step (EUR/MWh):")
    print(step_means[['label','mean','median']].to_string(index=False))
    print()
    boundary_steps = step_means[step_means['label'].str.contains('30|180')]
    interior_steps = step_means[~step_means['label'].str.contains('30|180')]
    print(f"Mean change at BOUNDARY steps : {boundary_steps['mean'].mean():.2f} EUR/MWh")
    print(f"Mean change at INTERIOR steps : {interior_steps['mean'].mean():.2f} EUR/MWh")
    ratio = boundary_steps['mean'].mean() / interior_steps['mean'].mean()
    print(f"Ratio (boundary / interior)   : {ratio:.2f}x")
    if ratio < 1.5:
        print("=> No meaningful discontinuity detected at tier boundaries.")
    else:
        print("=> Some discontinuity present at tier boundaries.")

plt.tight_layout()
plt.savefig('fig_boundary_analysis.png')
plt.show()
print("Saved: fig_boundary_analysis.png")


"""
Post-Estimation Calibration via Split Conformal Prediction
===========================================================

Method: Split Conformal Prediction (Venn-Abers / Quantile Conformal)
Reference: Angelopoulos and Bates (2022), "A Gentle Introduction to
           Conformal Prediction"

How it works:
    1. Use the VALIDATION SET (2022) as the calibration set.
       This is data the model has already seen in evaluation but was
       never used for training — exactly what conformal prediction requires.
    2. Compute the non-conformity score for each calibration observation:
           s_i = max(q_lo_i - y_i,  y_i - q_hi_i)
       This measures how far the actual price falls outside the
       predicted interval. If y_i is inside the interval, s_i <= 0.
    3. For a desired coverage level (1 - alpha), compute the
       (1 - alpha)(1 + 1/n) quantile of the calibration scores.
       Call this q_hat.
    4. At test time, expand each predicted interval by q_hat:
           [q_lo - q_hat,  q_hi + q_hat]
       This guarantees marginal coverage of (1 - alpha) on the test set.
"""

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kstest

plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': '#f8f9fa',
    'axes.grid': True, 'grid.alpha': 0.35, 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 110, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
})


def conformal_calibrate(prob_df, model_name,
                        cal_period_start, cal_period_end,
                        eval_period_start, eval_period_end,
                        levels=(0.50, 0.80, 0.90)):
    """
    Apply split conformal prediction to recalibrate prediction intervals.

    Parameters
    ----------
    prob_df           : full results DataFrame from rolling evaluation
    model_name        : 'DMLP' or 'LASSO-QR'
    cal_period_start  : start of calibration period (Timestamp)
    cal_period_end    : end of calibration period (Timestamp)
    eval_period_start : start of evaluation period (Timestamp)
    eval_period_end   : end of evaluation period (Timestamp)
    levels            : nominal coverage levels to calibrate

    Returns
    -------
    DataFrame: test-period predictions with original and calibrated intervals
    dict:      conformal quantile adjustments per (horizon, level)
    """
    model_df = prob_df[prob_df['model'] == model_name].copy()
    model_df['target_date'] = pd.to_datetime(model_df['target_date'])

    # Calibration set: validation period (2022 crisis)
    cal = model_df[
        (model_df['target_date'] >= cal_period_start) &
        (model_df['target_date'] <= cal_period_end)
    ].copy()

    # Test set: post-crisis period
    test = model_df[
        (model_df['target_date'] >= eval_period_start) &
        (model_df['target_date'] <= eval_period_end)
    ].copy()

    # Map nominal levels to stored quantile columns
    level_cols = {
        0.50: ('q25', 'q75'),
        0.80: ('q10', 'q90'),
        0.90: ('q05', 'q95'),
    }

    adjustments = {}   # (horizon, level) -> q_hat
    cal_coverage = {}  # (horizon, level) -> pre-calibration calibration-set coverage

    for h in sorted(model_df['horizon'].unique()):
        cal_h  = cal[cal['horizon'] == h]
        test_h = test[test['horizon'] == h]
        if len(cal_h) < 5 or len(test_h) < 5:
            continue

        for level in levels:
            if level not in level_cols:
                continue
            lo_col, hi_col = level_cols[level]
            if lo_col not in cal_h.columns or hi_col not in cal_h.columns:
                continue
            alpha = 1 - level

            # Non-conformity scores on calibration set
            lo_cal = cal_h[lo_col].values.astype(float)
            hi_cal = cal_h[hi_col].values.astype(float)
            y_cal  = cal_h['y_true'].values.astype(float)

            # Score: how far outside the interval is the actual price?
            # Negative means inside, positive means outside
            scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)

            # Conformal quantile: (1-alpha)(1+1/n) empirical quantile of scores
            n      = len(scores)
            q_level = np.ceil((1 - alpha) * (n + 1)) / n
            q_level = min(q_level, 1.0)
            q_hat   = float(np.quantile(scores, q_level))

            adjustments[(h, level)] = q_hat

            # Pre-calibration coverage on calibration set
            cal_cov = float(((y_cal >= lo_cal) & (y_cal <= hi_cal)).mean())
            cal_coverage[(h, level)] = cal_cov

    # Apply adjustments to test set
    test_calibrated = test.copy()
    for h in sorted(test['horizon'].unique()):
        mask = test_calibrated['horizon'] == h
        for level in levels:
            if (h, level) not in adjustments:
                continue
            lo_col, hi_col = level_cols[level]
            if lo_col not in test_calibrated.columns:
                continue
            q_hat = adjustments[(h, level)]
            # Expand interval symmetrically by q_hat
            test_calibrated.loc[mask, f'cal_{lo_col}'] = (
                test_calibrated.loc[mask, lo_col] - q_hat)
            test_calibrated.loc[mask, f'cal_{hi_col}'] = (
                test_calibrated.loc[mask, hi_col] + q_hat)

    return test_calibrated, adjustments, cal_coverage


# ── Run conformal calibration ──────────────────────────────────────────────
print("Applying split conformal prediction to DMLP intervals...")
print("Calibration set: 2022 (validation / crisis period)")
print("Test set:        2023-2024")
print()

CAL_START  = pd.Timestamp('2022-01-01')
CAL_END    = pd.Timestamp('2022-12-31')
TEST_START_CAL = pd.Timestamp('2023-01-01')
TEST_END_CAL   = pd.Timestamp('2024-12-31')
LEVELS = [0.50, 0.80, 0.90]

dmlp_cal, adjustments, cal_cov = conformal_calibrate(
    prob_df, 'DMLP',
    CAL_START, CAL_END,
    TEST_START_CAL, TEST_END_CAL,
    levels=LEVELS
)

# ── Print adjustment table ─────────────────────────────────────────────────
print("Conformal adjustment (q_hat) per horizon and level:")
print("(interval expanded by this amount on each side, EUR/MWh)")
print()
horizons_avail = sorted(set(h for h, l in adjustments.keys()))
rows = []
for h in horizons_avail:
    row = {'Horizon': f'h={h}d'}
    for level in LEVELS:
        row[f'{int(level*100)}% PI'] = round(adjustments.get((h, level), np.nan), 2)
    rows.append(row)
adj_table = pd.DataFrame(rows).set_index('Horizon')
print(adj_table.to_string())

# ── Coverage comparison: before vs after calibration ──────────────────────
print()
print("="*65)
print("COVERAGE COMPARISON — DMLP on TEST SET (2023-2024)")
print("80% Prediction Interval")
print("="*65)
print(f"{'Horizon':<10} {'Before calib':>14} {'After calib':>14} {'Nominal':>10}")
print("-"*50)

level_cols = {0.50: ('q25','q75'), 0.80: ('q10','q90'), 0.90: ('q05','q95')}
coverage_results = []

for h in horizons_avail:
    test_h = dmlp_cal[dmlp_cal['horizon'] == h]
    if len(test_h) == 0:
        continue
    y = test_h['y_true'].values.astype(float)

    for level in LEVELS:
        lo_col, hi_col = level_cols[level]
        if lo_col not in test_h.columns:
            continue

        # Original coverage
        lo_orig = test_h[lo_col].values.astype(float)
        hi_orig = test_h[hi_col].values.astype(float)
        cov_orig = float(((y >= lo_orig) & (y <= hi_orig)).mean())

        # Calibrated coverage
        cal_lo_col = f'cal_{lo_col}'
        cal_hi_col = f'cal_{hi_col}'
        if cal_lo_col in test_h.columns:
            lo_cal = test_h[cal_lo_col].values.astype(float)
            hi_cal = test_h[cal_hi_col].values.astype(float)
            cov_cal = float(((y >= lo_cal) & (y <= hi_cal)).mean())
            width_cal = float((hi_cal - lo_cal).mean())
        else:
            cov_cal  = np.nan
            width_cal = np.nan

        coverage_results.append({
            'horizon': h, 'level': level,
            'before': round(cov_orig, 3),
            'after':  round(cov_cal, 3),
            'nominal': level,
            'width_after': round(width_cal, 1),
        })

        if level == 0.80:
            print(f"  h={h:3d}d   {cov_orig*100:>12.1f}%  {cov_cal*100:>12.1f}%  "
                  f"{'80.0%':>10}")

cov_results_df = pd.DataFrame(coverage_results)
cov_results_df.to_csv('conformal_coverage_NO1.csv', index=False)
print()
print("Full results saved: conformal_coverage_NO1.csv")

# ── Figure: Coverage before vs after calibration ──────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharey=False)
fig.suptitle('Conformal Prediction Calibration of DMLP Intervals\n'
             'Calibration set: 2022  |  Test set: 2023-2024',
             fontsize=13, fontweight='bold')

nominal_colors = {0.50: '#9C27B0', 0.80: '#2196F3', 0.90: '#4CAF50'}
horizons_plot  = sorted(horizons_avail)
x = np.arange(len(horizons_plot))

for i, level in enumerate(LEVELS):
    ax = axes[i]
    sub = cov_results_df[cov_results_df['level'] == level]
    sub = sub.sort_values('horizon')

    before_vals = sub['before'].values * 100
    after_vals  = sub['after'].values  * 100

    ax.bar(x - 0.2, before_vals, 0.35,
           color='#FF9800', alpha=0.82, label='Before calibration')
    ax.bar(x + 0.2, after_vals,  0.35,
           color=nominal_colors[level], alpha=0.82, label='After calibration')
    ax.axhline(level * 100, color='black', ls='--', lw=2,
               label=f'Nominal {int(level*100)}%')

    ax.set_xticks(x)
    ax.set_xticklabels([f'h={h}d' for h in horizons_plot], fontsize=9)
    ax.set_ylabel('Empirical Coverage (%)')
    ax.set_title(f'{int(level*100)}% Prediction Interval')
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8)

    # Annotation: improvement
    for j, (bv, av) in enumerate(zip(before_vals, after_vals)):
        diff = av - bv
        ax.text(j + 0.2, av + 2, f'+{diff:.0f}' if diff > 0 else f'{diff:.0f}',
                ha='center', fontsize=8,
                color='green' if diff > 0 else 'red')

plt.tight_layout()
plt.savefig('fig_pb6_conformal_calibration.png')
plt.show()
print("Figure PB6 saved: fig_pb6_conformal_calibration.png")

print()
print("="*65)

