# Medium and Long-Term Norwegian Electricity Price Forecasting Using Probabilistic Machine Learning and Fundamental Drivers 

> **Master Thesis** — University of Inland Norway 
> **Author:** Nutan Gupta & Mathias Helseth
> **Year:** 2026

---

## Overview

This repository contains the complete source code, data pipeline, and results for a master thesis investigating **Medium and Long-Term Norwegian Electricity Price Forecasting Using Probabilistic Machine Learning and Fundamental Drivers**.

Norway's electricity market is unique due to its near-complete dependence on hydropower (~90% of generation). This makes traditional short-term forecasting methods insufficient for medium- to long-term horizons, where hydro reservoir conditions, water inflow patterns, and energy market fundamentals play a dominant role.

This thesis bridges the gap between short-term and medium/long-term forecasting by:
- Systematically integrating fundamental market drivers (reservoir levels, inflow, forward prices, fuel costs) with machine learning models
- Applying probabilistic forecasting methods to quantify forecast uncertainty
- Evaluating model performance across Norwegian price zones

---

## Research Questions

1. What are the main challenges in extending short-term ML-based forecasting methods to medium and long-term horizons?
2. What types of information are necessary for improving forecasting accuracy and quantifying uncertainty?
3. How can ML/DL models be used to estimate medium-term water values, and how effective are these estimates in predicting electricity prices?
4. How can fundamental aspects, such as water value simulations or other market factors, be incorporated into ML-based models to improve forecast reliability?

---

## Repository Structure

```
Master-Thesis-Norwegian-Electricity-Forecasting/
│
├── README.md
│
├── build_master_clean_dataset_NO1.py    # Complete data pipeline: raw → master_clean_NO1.csv
├── exploratory_data_analysis.py         # EDA: correlation analysis, time series plots, distributions
├── feature_engineering.py               # Feature engineering pipeline
├── baseline_models.py                   # Baseline Models
├── probabilistic_forecasting.py         # probabilistic models - LASSO Quantile regression, Distributed Machine Learning Perceptron (DMLP)
├── empirical_test_Ghelasi_m1_y1.py      # Empirical test of Ghelasi et al. forward price model
├── residual_diagnostics_full.py         # Residual analysis, error diagnostics, model evaluation
│
└── results/                             # All model outputs, plots, and saved models
```

---

## Data Sources

The dataset integrates **11 data sources** covering the period 2015–2025. All raw data files should be placed in the same directory as the scripts.

| # | File | Source | Description |
|---|---|---|---|
| 1 | `norway_day_ahead_prices_long.csv` | Montel | Hourly day-ahead spot prices — all 5 zones (NO1–NO5) |
| 2 | `hydro_reserves_clean.csv` | NVE | Weekly reservoir storage levels (NO national + zones) |
| 3 | `water_inflow.csv` | NVE | Weekly water inflow + snowpack, 1958–2026 |
| 4 | `nordic_system_forwards_clean.csv` | Nordic Exchange | Daily Nordic system forward prices (M1, Y1) |
| 5 | `nordic_epads_clean.csv` | Montel | Daily area price differentials for all zones |
| 6 | `TTF_Daily_.csv` | Montel | TTF natural gas front-month price (EUR/MWh) |
| 7 | `EEX_EUA_Daily_Continuous_2012_2025.csv` | Montel | EU carbon allowance auction price (EUR/tonne CO₂) |
| 8 | `brent_front_month_m1.csv` | Montel | Brent crude oil front-month price (USD/barrel) |
| 9 | `norway_weather_datasetunchanged.csv` | MET Norway | Daily temperature, wind speed, precipitation — all zones |
| 10 | `load_NO1_raw.csv` … `load_NO5_raw.csv` | ENTSO-E | Hourly actual electricity consumption — per zone |
| 11 | `eur_nok_clean.csv` | Montel | EUR/NOK daily exchange rate |

> **Note:** Raw data files are not included in this repository due to data licensing. Contact the author or the respective sources to obtain the data.

---

## Final Dataset

The data pipeline produces one clean dataset per zone:

| Zone | File | Rows | Columns | Period |
|---|---|---|---|---|
| NO1 (Oslo) | `master_clean_NO1.csv` | 2,516 | 34 | 2018-11-24 → 2025-10-13 |
| NO2 (Kristiansand) | `master_clean_NO2.csv` | 1,106 | 34 | 2022-10-04 → 2025-10-13 |
| NO3 (Trondheim) | `master_clean_NO3.csv` | 1,944 | 34 | 2020-06-18 → 2025-10-13 |
| NO4 (Tromsø) | `master_clean_NO4.csv` | 2,516 | 34 | 2018-11-24 → 2025-10-13 |
| NO5 (Bergen) | `master_clean_NO5.csv` | 1,057 | 34 | 2022-11-22 → 2025-10-13 |

Each dataset contains **32 features** across 8 groups:

| Feature Group | Features | Count |
|---|---|---|
| Raw hydro | `hydro_reserve_gwh`, `inflow_hbv_gwh`, `inflow_snow_gwh`, `reservoir_deviation_gwh` | 4 |
| Forward market | `forward_M1`, `forward_Y1`, `epad_M1`, `epad_Y1` | 4 |
| Fuel prices | `fuel_gas_eur_mwh`, `fuel_carbon_eur_t`, `fuel_brent_usd_bbl` | 3 |
| Weather & demand | `weather_temp_c`, `weather_wind_ms`, `weather_precip_mm`, `load_mw` | 4 |
| Macroeconomic | `macro_eur_nok` | 1 |
| Price lags | `lag_price_1d`, `lag_price_7d`, `lag_price_30d`, `lag_price_365d` | 4 |
| Rolling statistics | `ma_price_7d`, `ma_price_30d`, `std_price_7d`, `std_price_30d` | 4 |
| Derived features | `derived_inflow_z_score`, `derived_hydro_change`, `derived_forward_slope`, `derived_gas_volatility_30d` | 4 |
| Calendar | `cal_month_sin`, `cal_month_cos`, `cal_day_of_year_sin`, `cal_day_of_year_cos` | 4 |

---

## Train / Validation / Test Split

All models use a strict **chronological split** — never random:

| Split | Period | Rows | Share |
|---|---|---|---|
| **Train** | 2018-11-24 → 2022-12-31 | 1134 | 45.07% |
| **Validation** | 2022-01-01 → 2023-12-31 | 365 | 14.51% |
| **Test** | 2023-01-01 → 2024-12-31 | 731 | 29.05% |
| **Holdout** | 2025-01-01 → 2025-10-13 | 286 | 11.37% |

The training set deliberately includes the 2021–2022 European energy crisis (TTF gas peaked at 317.65 EUR/MWh; NO1 spot peaked at 660.06 EUR/MWh) so the model learns from extreme market regimes.

---

## How to Run

### 1. Install dependencies

```bash
pip install pandas numpy tqdm scikit-learn matplotlib seaborn joblib pytz
```

### 2. Set your data folder path

In each script, update the `DATA_FOLDER` variable at the top:

```python
DATA_FOLDER = Path('/your/path/to/data/folder')
```

### 3. Build the master dataset (run first)

```bash
python build_master_clean_dataset_NO1.py
```

This will:
- Clean all 11 raw files individually
- Merge them on date (inner join)
- Engineer 16 derived features
- Run 10 assert checks
- Save `master_dataset_NO1.csv` (2,881 rows — with warmup) and `master_clean_NO1.csv` (2,516 rows — model ready)

### 4. Run exploratory data analysis

```bash
python exploratory_data_analysis.py
```

### 5. Run baseline models

```bash
python baseline_models.py
```

### 6. Run probabilistic forecasting

```bash
python probabilistic_forecasting.py
```

### 7. Run residual diagnostics

```bash
python residual_diagnostics_full.py
```

---

## Key Technical Decisions

| Decision | Choice | Reason |
|---|---|---|
| Study period start | 2017-11-24 | TTF gas data availability |
| Hydro reference history | 1958–2026 (68 years) | Long-run seasonal norm from NVE |
| Rolling `min_periods` | Equal to window size | Only compute when full window available |
| Calendar encoding | sin/cos with 365.25 | Avoids year-boundary discontinuity, handles leap years |
| Coal prices | Excluded | Norway has no coal plants; gas is the marginal technology |
| `reservoir_deviation_gwh` | NVE official measure kept | More accurate than 8-year derived alternative |
| Data leakage prevention | `shift(1)` on all lag features | Model never sees current day's price as input |
| Split method | Chronological | Time series must preserve temporal order |

---

## Target Variable Statistics (NO1)

| Statistic | Value |
|---|---|
| Mean | 68.70 EUR/MWh |
| Standard deviation | 72.83 EUR/MWh |
| Minimum | −6.81 EUR/MWh |
| Maximum | 660.06 EUR/MWh |
| Negative price days | 17 |

---

## Dependencies

```
Python       >= 3.10
pandas       >= 2.0
numpy        >= 1.24
xgboost      >= 2.0
scikit-learn >= 1.3
matplotlib   >= 3.7
seaborn      >= 0.12
joblib       >= 1.3
pytz         >= 2023.3
```

---

## Reproducibility

All models use `random_state=42`. The data pipeline produces identical output on every run — verified by assert checks on row count, date range, null values, and inf values. The expected outputs are:

```
master_clean_NO1.csv   →  2516 rows × 34 columns  (2018-11-24 → 2025-10-13)
master_clean_NO2.csv   →  1106 rows × 34 columns  (2022-10-04 → 2025-10-13)
master_clean_NO3.csv   →  1944 rows × 34 columns  (2020-06-18 → 2025-10-13)
master_clean_NO4.csv   →  2516 rows × 34 columns  (2018-11-24 → 2025-10-13)
master_clean_NO5.csv   →  1057 rows × 34 columns  (2022-11-22 → 2025-10-13)
```

---

## License

This repository is made available for academic purposes. If you use any part of this code or methodology in your own research, please cite this thesis appropriately.

---

## Contact

For questions about the methodology, data pipeline, or results, please open an issue in this repository.
