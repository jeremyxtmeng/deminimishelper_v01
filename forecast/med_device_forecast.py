# created: 12/13/2025
# last upated: 12/18/2025
# prep data of med devices and developing forecasting models

# 12/18/2025
# saving CIF value a csv file

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

# divide country goods parts into two groups
# time series without zero
df_final1=df_filtered.groupby(['HTS22', 'code']).filter(lambda g: g['GEN_CIF_MO'].notnull().all())

# time series with zero
df_final2=df_filtered.groupby(['HTS22', 'code']).filter(lambda g: g['GEN_CIF_MO'].isnull().any())


# final version saved for the database
# delete if country does no export the good at all (i.e. no churns)
df_filtered2 = df_complete.groupby(['HTS22', 'code']).filter(lambda g: g['GEN_CIF_MO'].notnull().any())
df_filtered2 = df_complete.drop(['name', 'iso'], axis=1)
df_filtered2 = df_filtered2[df_filtered2['code']!=7370]
df_filtered2 = df_filtered2.merge(df0, on='code', how='left')
df_filtered2[['GEN_CIF_MO']] = df_filtered2[['GEN_CIF_MO']].fillna(0)
df_filtered2=df_filtered2[['HTS22', 'time','iso', 'GEN_CIF_MO']]


df_filtered2.to_csv('raw_p_q_data.csv', index=False)
#---------------------------------------------------------------------
#---------------------------------------------------------------------
# part 1.1: testing one series for time series without zero
#---------------------------------------------------------------------
#---------------------------------------------------------------------
import scipy.stats as stats
import seaborn as sns

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.arima import arima_string

#----------------------------------------
# part 1.1.1: visual inspection
#----------------------------------------

#print(df_final1['HTS22'].dtype)
g1=df_final1[(df_final1['HTS22']==3002120090) & (df_final1['iso']=='NL')]
#print(f"\nThe date range: {g1['time'].min()} to { g1['time'].max()}")

g1=g1.set_index('time')
g1['unit_price']=g1['GEN_CIF_MO']/g1['GEN_QY1_MO']
g1['y']=np.log(g1['GEN_QY1_MO'])


#------------------
# way 1
#------------------

# Adjusting the figure size
fig = plt.subplots(figsize=(16, 5))

# Creating a plot with proper ticks
#plt.plot(g1.index, g1['GEN_QY1_MO'])
plt.plot(g1.index, g1['y'])

plt.title('GEN_QY1_MO', fontsize=20)
plt.xlabel('Date', fontsize=15)
plt.ylabel('Quantity', fontsize=15)

#plt.xlim(g1.index.min(), g1.index.max())
g1_ticks = list(g1.index)
g1_tick_label=g1_ticks[0::5]
plt.xticks(g1_tick_label, rotation=45)

plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
from pylab import rcParams
rcParams['figure.figsize'] = (18,7)


#------------------
# way 2
#------------------
g1['ds'] =g1.index
g1['unique_id']=1

StatsForecast.plot(g1)

# plotting acf and pacf-----------
fig, axs = plt.subplots(nrows=1, ncols=2)
plot_acf(g1["y"],  lags=24, ax=axs[0],color="fuchsia")
axs[0].set_title("Autocorrelation");
plot_pacf(g1["y"],  lags=24, ax=axs[1],color="lime")
axs[1].set_title('Partial Autocorrelation')
plt.show()

#----------------------------------------
# part 1.1.2: training model
#----------------------------------------

Y_train_df = g1.loc[(g1.index>'2018-06-01') & (g1.index<'2024-01-01'),['ds','y','unique_id']]
Y_train_df['ds']=pd.to_datetime(Y_train_df["ds"])


Y_test_df = g1.loc[g1.ds>='2024-01-01',['ds','y','unique_id']]
Y_test_df['ds']=pd.to_datetime(Y_test_df["ds"])

Y_train_df.shape, Y_test_df.shape

sns.lineplot(Y_train_df,x="ds", y="y", label="Train")
sns.lineplot(Y_test_df, x="ds", y="y", label="Test")
plt.show()


season_length = 12 # Monthly data
horizon = len(Y_test_df) # number of predictions

models = [AutoARIMA(season_length=season_length)]
sf = StatsForecast(models=models, freq='MS')

sf.fit(df=Y_train_df)
StatsForecast(models=[AutoARIMA],freq='MS')

arima_string(sf.fitted_[0,0].model_)

result=sf.fitted_[0,0].model_
print(result.keys())
print(result['arma'])

residual=pd.DataFrame(result.get("residuals"), columns=["residual Model"])
residual

fig, axs = plt.subplots(nrows=2, ncols=2)

# plot[1,1]
residual.plot(ax=axs[0,0])
axs[0,0].set_title("Residuals");

# plot
sns.histplot(residual, ax=axs[0,1]);
axs[0,1].set_title("Density plot - Residual");

# plot
stats.probplot(residual["residual Model"], dist="norm", plot=axs[1,0])
axs[1,0].set_title('Plot Q-Q')

# plot
plot_acf(residual,  lags=35, ax=axs[1,1],color="fuchsia")
axs[1,1].set_title("Autocorrelation");

plt.show();


#------------------
Y_hat_df = sf.forecast(df=Y_train_df, h=horizon, fitted=True)
Y_hat_df.head()

values=sf.forecast_fitted_values()
values

Y_hat_df = Y_test_df.merge(Y_hat_df, how='left', on=['unique_id', 'ds'])

fig, ax = plt.subplots(1, 1, figsize = (18, 7))
plot_df = pd.concat([Y_train_df, Y_hat_df]).set_index('ds')
plot_df[['y', 'AutoARIMA']].plot(ax=ax, linewidth=2)
ax.set_title(' Forecast', fontsize=22)
ax.set_ylabel('Monthly ', fontsize=20)
ax.set_xlabel('Timestamp [t]', fontsize=20)
ax.legend(prop={'size': 15})
ax.grid()

from functools import partial

import utilsforecast.losses as ufl
from utilsforecast.evaluation import evaluate

evaluate(
    Y_test_df.merge(Y_hat_df),
    metrics=[ufl.mae, ufl.mape, partial(ufl.mase, seasonality=season_length), ufl.rmse, ufl.smape],
    train_df=Y_train_df,
)

#---------------------------------------------------------------------
# part 3: LSTM (not suitable)
#---------------------------------------------------------------------

#---------------------------------------------------------------------
#---------------------------------------------------------------------
# Linear regression+ARIMA/XGboost
#---------------------------------------------------------------------
#---------------------------------------------------------------------

g1=df_final1[(df_final1['HTS22']==3002120090) & (df_final1['iso']=='NL')]
g1['ln_GEN_QY1_MO']=np.log(g1['GEN_QY1_MO'])
g1["time"] = pd.to_datetime(g1["time"])
g1["quarter"] = g1["time"].dt.quarter 

p = 3  # set to 1 for linear, 2 for quadratic, 3 for cubic, etc.
g1 = g1.sort_values("time").reset_index(drop=True)
g1["t"] = np.arange(len(g1)) # time trend starting from 0
g1['t2']= g1['t']**2
g1['t3']= g1['t']**3

# --- 3) Fit y on Q1,Q2,Q3 (Q4 baseline), constant, and polynomial trend ---
import statsmodels.formula.api as smf


res = smf.ols(formula='ln_GEN_QY1_MO~C(quarter, Treatment(reference=4))+t+t2+t3', data=g1).fit()
print(res.summary())

g1 = g1.sort_values("time").copy()
g1["ln_GEN_QY1_MO_hat"] = res.predict(g1)   # or: res.fittedvalues (if same row order)
g1["y"] = g1['ln_GEN_QY1_MO']-g1['ln_GEN_QY1_MO_hat']

plt.figure(figsize=(10, 4))
plt.plot(g1["time"], g1["ln_GEN_QY1_MO"], label="Actual")
plt.plot(g1["time"], g1["ln_GEN_QY1_MO_hat"], label="Fitted")
plt.xlabel("Time")
plt.ylabel("Q")
plt.legend()
plt.tight_layout()
plt.show()

#------------------
# way 2
#------------------
g1['ds'] =g1['time']
g1['unique_id']=1

StatsForecast.plot(g1)

# plotting acf and pacf-----------
fig, axs = plt.subplots(nrows=1, ncols=2)
plot_acf(g1["y"],  lags=24, ax=axs[0],color="fuchsia")
axs[0].set_title("Autocorrelation");
plot_pacf(g1["y"],  lags=24, ax=axs[1],color="lime")
axs[1].set_title('Partial Autocorrelation')
plt.show()

#----------------------------------------
# part 1.1.2: training model
#----------------------------------------

Y_train_df = g1.loc[ g1['time']<'2024-01-01',['ds','y','unique_id']]
Y_train_df['ds']=pd.to_datetime(Y_train_df["ds"])


Y_test_df = g1.loc[g1['time']>='2024-01-01',['ds','y','unique_id']]
Y_test_df['ds']=pd.to_datetime(Y_test_df["ds"])

Y_train_df.shape, Y_test_df.shape

sns.lineplot(Y_train_df,x="ds", y="y", label="Train")
sns.lineplot(Y_test_df, x="ds", y="y", label="Test")
plt.show()


season_length = 12 # Monthly data
horizon = len(Y_test_df) # number of predictions

models = [AutoARIMA(season_length=season_length)]
sf = StatsForecast(models=models, freq='MS')

sf.fit(df=Y_train_df)
StatsForecast(models=[AutoARIMA],freq='MS')

arima_string(sf.fitted_[0,0].model_)

Y_hat_df = sf.forecast(df=Y_train_df, h=horizon, fitted=True)
Y_hat_df.head()

values=sf.forecast_fitted_values()
values

Y_hat_df = Y_test_df.merge(Y_hat_df, how='left', on=['unique_id', 'ds'])


from functools import partial

import utilsforecast.losses as ufl
from utilsforecast.evaluation import evaluate

evaluate(
    Y_test_df.merge(Y_hat_df),
    metrics=[ufl.mse],
    train_df=Y_train_df,
)

#----------------------------------------------------------
#-------------- redo no plotting ---------------
import statsmodels.formula.api as smf
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mse


cutoff = pd.Timestamp("2024-01-01")

# part 1: linear model to remove trend
g1=df_final1[(df_final1['HTS22']==3002120090) & (df_final1['iso']=='CA')]
g1['ln_Y']=np.log(g1['GEN_QY1_MO'])
g1["time"] = pd.to_datetime(g1["time"])
g1["quarter"] = g1["time"].dt.quarter 

p = 3 
g1 = g1.sort_values("time").reset_index(drop=True)
g1["t"] = np.arange(len(g1)) # time trend starting from 0
g1['t2']= g1['t']**2
g1['t3']= g1['t']**3

# Fit y on Q1,Q2,Q3 (Q4 baseline), constant, and polynomial trend ---

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
horizon = len(Y_test_df) # number of predictions

h_test = len(Y_test_df)

sf = StatsForecast(models=[AutoARIMA(season_length=season_length)], freq=freq)
sf.fit(df=Y_train_df)

# forecast residuals for the test horizon
Yhat_test = sf.forecast(df=Y_train_df, h=h_test)

# (robustly) detect the forecast column name (often "AutoARIMA")
fcst_col = [c for c in Yhat_test.columns if c not in ["unique_id", "ds"]][0]
Yhat_test = Yhat_test.rename(columns={fcst_col: "resid_hat"})

# merge and evaluate MSE on residuals
test_merge = Y_test_df.merge(Yhat_test, on=["unique_id", "ds"], how="left")
mse_resid = evaluate(test_merge, metrics=[mse], train_df=Y_train_df)
print(mse_resid)

# ----------------------------
# Part 3: Forecast x periods ahead of ORIGINAL series
#   1) forecast residuals with ARIMA
#   2) forecast regression component for future dates
#   3) add them back
# ----------------------------

x=6

Y_all_df=pd.concat([Y_train_df, Y_test_df])

sf_full = StatsForecast(models=[AutoARIMA(season_length=season_length)], freq='MS')
sf_full.fit(df=Y_all_df)

resid_fcst = sf_full.forecast(df=Y_all_df, h=x)
fcst_col2 = [c for c in resid_fcst.columns if c not in ["unique_id", "ds"]][0]
resid_fcst = resid_fcst.rename(columns={fcst_col2: "resid_hat"})

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



#------- ploting----------
_ = sf_full.forecast(df=Y_all_df, h=x, fitted=True)


resid_fit_df = None
for meth in ["forecast_fitted_values", "fitted_values", "get_fitted_values"]:
    if hasattr(sf_full, meth):
        resid_fit_df = getattr(sf_full, meth)()
        break
g1['reg_hat']=g1['ln_GEN_QY1_MO_hat']

# Build a historical dataframe with actual + fitted
hist = g1[["time", "ln_GEN_QY1_MO", "reg_hat"]].copy()
hist = hist.rename(columns={"time": "ds", "ln_GEN_QY1_MO": "ln_actual"})

if resid_fit_df is not None:
    # detect the residual fitted column (typically "AutoARIMA")
    fit_col = [c for c in resid_fit_df.columns if c not in ["unique_id", "ds", "y"]][0]
    resid_fit_df = resid_fit_df.rename(columns={fit_col: "resid_fit"})
    hist = hist.merge(resid_fit_df[["ds", "resid_fit"]], on="ds", how="left")
    hist["ln_fitted"] = hist["reg_hat"] + hist["resid_fit"]
else:
    # fallback: only regression fitted (still useful)
    hist["ln_fitted"] = hist["reg_hat"]

#------- ploting----------
# # 2) Prepare forecast dataframe
fcst = out[["time", "ln_forecast", "forecast_level"]].copy()
fcst = fcst.rename(columns={"time": "ds"})

#------- ploting----------
# 3) Plot in LOG
plt.figure(figsize=(11, 4))
plt.plot(hist["ds"], hist["ln_actual"], label="Actual (log)")
plt.plot(hist["ds"], hist["ln_fitted"], label="Fitted (log)")
plt.plot(fcst["ds"], fcst["ln_forecast"], label="Forecast (log)")
plt.axvline(hist["ds"].max(), linestyle="--", label="Forecast starts")
plt.xlabel("Time")
plt.ylabel("log(y)")
plt.legend()
plt.tight_layout()
plt.show()

#------- ploting----------
# 4) plot in level
hist["level_actual"] = np.exp(hist["ln_actual"])
hist["level_fitted"] = np.exp(hist["ln_fitted"])

plt.figure(figsize=(11, 4))
plt.plot(hist["ds"], hist["level_actual"], label="Actual (level)")
plt.plot(hist["ds"], hist["level_fitted"], label="Fitted (level)")
plt.plot(fcst["ds"], fcst["forecast_level"], label="Forecast (level)")
plt.axvline(hist["ds"].max(), linestyle="--", label="Forecast starts")
plt.xlabel("Time")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()

#----------------------------------------------
# using xgboost
#----------------------------------------------
from xgboost import XGBRegressor

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

print("Residual MSE (XGBoost):", mse_resid_xgb)


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


