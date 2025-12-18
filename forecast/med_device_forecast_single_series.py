# created: 12/17/2025
# last upated: 12/18/2025
# clean code to estimate models and forecast for one series

# packages
from pathlib import Path
import pandas as pd
import numpy as np

#---------------------------------------------------------------------
# part 0: prep med device data
#---------------------------------------------------------------------

# 0. load HST codes of med goods

# Original project folder
path_origin=Path("C:/Users/xiang/Documents/backup_T580/Videos/Videos/Dropbox/tariff_medgoods_2025/")

# load HS codes of medical devices
df1 = pd.read_csv(Path.cwd().parent/'utilities'/'app_search_hscode_df.csv', usecols=["HTS22"])
df1["med_device"] = 1

# load country iso
df0 = pd.read_csv(path_origin/'replication_package_09_16_2025'/'04_us_monthly_import_data'/'census_country_code.csv', usecols=["name","iso", "code"])

# load sourcing data up to July 2025 (monthly)
df3=pd.read_csv(path_origin/'replication_package_09_16_2025'/'04_us_monthly_import_data'/'med_imports_from_2017_01_to_2025_07_all_data.csv')

#df3=df3[df3["time"]=="2025-07"].copy().reset_index(drop=True)

df3_1 = df3.rename(columns={'I_COMMODITY': 'HTS22', 'CTY_CODE': 'code'})
df3_2 = df3_1[~df3_1['code'].str.strip().str.startswith('0')]
df3_2 = df3_2[(df3_2['code'].str.strip() != '-')]
df3_2 = df3_2[~df3_2['code'].str.endswith('X')]

# 1. cleaning
df3_2['code'] = pd.to_numeric(df3_2['code'])
df3_2=df3_2[['HTS22','time','code', 'CAL_DUT_MO', 'GEN_CIF_MO', 'GEN_VAL_MO', 'GEN_QY1_MO', 'DUT_VAL_MO' ]].reset_index(drop=True)
df3_3 = df3_2.merge(df1, on="HTS22", how="left").dropna(subset=['med_device']) # keep only med device
df3_4 = df3_3.merge(df0,on="code", how="left")
df3_4 = df3_4.drop(['med_device'], axis=1)

df3_4['time'] = pd.to_datetime(df3_4['time'], format='%Y-%m') # formating date time
df3_4['time'] =df3_4['time'].dt.strftime('%Y-%m-%d') # removing minutes

# 2. define the full date range and unique groups
all_dates = pd.date_range(start=df3_4['time'].min(), end=df3_4['time'].max(), freq='MS')
all_groups = df3_4[['HTS22', 'code']].drop_duplicates()

# 3. create a reference DataFrame with all combinations
template = all_groups.assign(key=1).merge(pd.DataFrame({'time': all_dates, 'key': 1}), on='key').drop('key', axis=1)
template['time'] = pd.to_datetime(template['time'], format='%Y-%m')
template['time'] = template['time'].dt.strftime('%Y-%m-%d') 

# 4. merge with original data to find missing entries
df_complete = pd.merge(template, df3_4, on=['HTS22', 'code', 'time'], how='left')
df_complete['GEN_CIF_MO'] = df_complete['GEN_CIF_MO'].replace(0.0, np.nan)

# delete if country does no export the good at all (i.e. no churns)
df_filtered = df_complete.groupby(['HTS22', 'code']).filter(lambda g: g['GEN_CIF_MO'].notnull().any())
df_filtered = df_complete.drop(['name', 'iso'], axis=1)
df_filtered = df_filtered[df_filtered['code']!=7370]
df_filtered = df_filtered.merge(df0, on='code', how='left')
df_filtered[['GEN_CIF_MO', 'CAL_DUT_MO', 'GEN_VAL_MO','GEN_QY1_MO','DUT_VAL_MO']] = df_filtered[['GEN_CIF_MO', 'CAL_DUT_MO', 'GEN_VAL_MO','GEN_QY1_MO','DUT_VAL_MO']].fillna(0)


#---------------------------------------------------------------------
#---------------------------------------------------------------------
#  one series for time series
#---------------------------------------------------------------------
#---------------------------------------------------------------------

import statsmodels.formula.api as smf
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from xgboost import XGBRegressor

cutoff = pd.Timestamp("2024-01-01") # data used for testing the model
x=6                                 # forecast period after the last obs

# part 1: linear model to remove trend
g1=df_filtered[(df_filtered['HTS22']==3002120090) & (df_filtered['iso']=='CA')]
g1['ln_Y']=np.log(g1['GEN_QY1_MO']+1)
g1["time"] = pd.to_datetime(g1["time"])
g1["quarter"] = g1["time"].dt.quarter 


g1 = g1.sort_values("time").reset_index(drop=True)
g1["t"] = np.arange(len(g1)) # time trend starting from 0
g1['t2']= g1['t']**2
g1['t3']= g1['t']**3

res = smf.ols(formula='ln_Y~C(quarter, Treatment(reference=4))+t+t2+t3', data=g1).fit()

g1 = g1.sort_values("time").copy()
g1["ln_Y_hat"] = res.predict(g1)
g1["y"] = g1['ln_Y']-g1['ln_Y_hat']

# part 2: ARIMA model
g1['ds'] =g1['time']
g1['unique_id']=1

Y_train_df = g1.loc[ g1['time']<cutoff,['ds','y','unique_id']]
Y_test_df = g1.loc[g1['time']>=cutoff,['ds','y','unique_id']]

season_length = 12       # Monthly data
horizon = len(Y_test_df) # number of predictions for testing

h_test = len(Y_test_df)

sf = StatsForecast(models=[AutoARIMA(season_length=season_length)], freq='MS')
sf.fit(df=Y_train_df)

Yhat_test = sf.forecast(df=Y_train_df, h=h_test).rename(columns={'AutoARIMA': "resid_hat"}) # forecast residuals for the test horizon

test_merge = Y_test_df.merge(Yhat_test, on=["unique_id", "ds"], how="left") # merge and evaluate MSE on residuals
mse_resid=round(float(np.mean((test_merge['y'] - test_merge['resid_hat']) ** 2)),4)


# part 2.1: ARIMA model forecast

Y_all_df=pd.concat([Y_train_df, Y_test_df])

sf_full = StatsForecast(models=[AutoARIMA(season_length=season_length)], freq='MS')
sf_full.fit(df=Y_all_df)

resid_fcst = sf_full.forecast(df=Y_all_df, h=x).rename(columns={'AutoARIMA': "resid_hat"})

# build future regressors for regression component
last_time = g1["time"].max()
future_time = pd.date_range(last_time + pd.offsets.MonthBegin(1), periods=x, freq='MS')

future = pd.DataFrame({"time": future_time})
future["quarter"] = future["time"].dt.quarter

t_last = g1["t"].iloc[-1]            # last observed t = n-1
future["t"]  = np.arange(t_last + 1, t_last + 1 + x)
future["t2"] = future["t"]**2
future["t3"] = future["t"]**3

future["reg_hat"] = res.predict(future)

# combine: ln forecast = regression forecast + residual forecast
out = future.merge(
    resid_fcst[["ds", "resid_hat"]].rename(columns={"ds": "time"}),
    on="time",
    how="left"
)

out["ln_forecast"] = out["reg_hat"] + out["resid_hat"]
out["forecast_level"] = np.exp(out["ln_forecast"])  # optional

out  # contains time, reg_hat, resid_hat, ln_forecast, forecast_level

#----------------------------------------------
# using xgboost
#----------------------------------------------


# ----------------------------
# Helper: feature engineering
# ----------------------------
lags = [1, 2, 3, 6, 12]
# y is the residual from the regression
def make_ts_features(df, ycol='y', timecol="time", lags=lags):
    df = df.sort_values(timecol).copy()
    #df["quarter"] = pd.to_datetime(df[timecol]).dt.quarter

    # lag features
    for L in lags:
        df[f"lag{L}"] = df[ycol].shift(L)

    return df

feature_cols = ( [f"lag{L}" for L in lags])


g1_feature = make_ts_features(g1, ycol='y', timecol="time")
g1_feature = g1_feature.dropna(subset=feature_cols + ['y']).copy()  # drop rows without full features

X_train = g1_feature.loc[ g1_feature['time']<cutoff, feature_cols]
y_train = g1_feature.loc[ g1_feature['time']<cutoff, 'y']

X_test = g1_feature.loc[ g1_feature['time']>=cutoff, feature_cols]
y_test = g1_feature.loc[ g1_feature['time']>=cutoff, 'y']


# ----------------------------
# Part B: Fit XGBoost on train residuals + test MSE
# ----------------------------
xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=0,
)
xgb.fit(X_train, y_train)

resid_hat_test_xgb = xgb.predict(X_test)
mse_resid_xgb = round(float(np.mean((y_test.values - resid_hat_test_xgb) ** 2)),4)




resid_series = g1.set_index("time")["y"].copy()
last_time = resid_series.index.max()
future_dates = pd.date_range(last_time + pd.offsets.MonthBegin(1), periods=x, freq='MS')

def xgb_predict_one_step(series):
    # build one-row feature vector from current series history
    row = {}
    for L in lags:
        row[f"lag{L}"] = series.iloc[-L] if len(series) >= L else np.nan

    Xrow = pd.DataFrame([row])[feature_cols]
    return float(xgb.predict(Xrow)[0])

resid_fcst_xgb = []
series_work = resid_series.copy()

# ensure we have enough history to compute features
min_needed = max(lags)
if len(series_work) < min_needed:
    raise ValueError(f"Not enough history to forecast: need at least {min_needed} observations.")

for dt in future_dates:
    yhat = xgb_predict_one_step(series_work)
    resid_fcst_xgb.append({"time": dt, "resid_hat_xgb": yhat})
    series_work.loc[dt] = yhat  # append prediction for next-step lags/rolls

resid_fcst_xgb = pd.DataFrame(resid_fcst_xgb)


forecast_df = (
    out.merge(resid_fcst_xgb,   on="time", how="left")
)

# final forecasts in log
forecast_df["ln_forecast_xgb"]   = forecast_df["reg_hat"] + forecast_df["resid_hat_xgb"]

# level forecasts
forecast_df["forecast_level_xgb"]   = np.exp(forecast_df["ln_forecast_xgb"])

# add MSE columns (constants repeated for convenience)
forecast_df["mse_resid_arima"] = round(float(mse_resid['resid_hat'].iloc[0]),4)
forecast_df["mse_resid_xgb"]   = mse_resid_xgb

forecast_df


