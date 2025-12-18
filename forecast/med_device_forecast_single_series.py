# created: 12/17/2025
# last upated: 12/18/2025
# clean code to estimate models and forecast for one series

# packages
import pandas as pd
import numpy as np

import statsmodels.formula.api as smf
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from xgboost import XGBRegressor

import io, os
import joblib
from google.cloud import storage
#---------------------------------------------------------------------
# part 0: prep med device data
#---------------------------------------------------------------------
cutoff = pd.Timestamp("2024-01-01") # data used for testing the model
x=6                                 # forecast period after the last obs


df_filtered = pd.read_csv('raw_p_q_data.csv') # only GEN_CIF_MO
all_groups = df_filtered[['HTS22', 'code']].drop_duplicates()

g1=df_filtered[(df_filtered['HTS22']==3002120090) & (df_filtered['iso']=='CA')]
g1['ln_Y']=np.log(g1['GEN_CIF_MO']+1)

#---------------------------------------------------------------------
#---------------------------------------------------------------------
# part 1: one series for time series
#---------------------------------------------------------------------
#---------------------------------------------------------------------

# part 1: linear model to remove trend
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

h_test = len(Y_test_df)

sf = StatsForecast(models=[AutoARIMA(season_length=12)], freq='MS') # Monthly data season_length = 12 
sf.fit(df=Y_train_df)

Yhat_test = sf.forecast(df=Y_train_df, h=h_test).rename(columns={'AutoARIMA': "resid_hat"}) # forecast residuals for the test horizon

test_merge = Y_test_df.merge(Yhat_test, on=["unique_id", "ds"], how="left") # merge and evaluate MSE on residuals
mse_resid=round(float(np.mean((test_merge['y'] - test_merge['resid_hat']) ** 2)),4)


# part 2.1: ARIMA model forecast

Y_all_df=g1[['ds','y','unique_id']].copy() # all sample

resid_fcst = sf.forecast(df=Y_all_df, h=x).rename(columns={'AutoARIMA': "resid_hat"})

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
future["forecast_level"] = np.exp(future["reg_hat"] + resid_fcst["resid_hat"])-1  # optional


#----------------------------------------------
# using xgboost
#----------------------------------------------

lags = [1, 2, 3, 6, 12] # lag features

def make_ts_features(df, ycol='y', timecol="time", lags=lags): # y is the residual from the regression
    df = df.sort_values(timecol).copy()
    
    for L in lags:
        df[f"lag{L}"] = df[ycol].shift(L) # lag features

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

for dt in future_dates:
    yhat = xgb_predict_one_step(series_work)
    resid_fcst_xgb.append({"time": dt, "resid_hat_xgb": yhat})
    series_work.loc[dt] = yhat  # append prediction for next-step lags/rolls

resid_fcst_xgb = pd.DataFrame(resid_fcst_xgb)

future["forecast_level_xgb"]   = np.exp(future["reg_hat"] + resid_fcst_xgb["resid_hat_xgb"])+1 # level forecasts


future["mse_resid_arima"] = mse_resid # add MSE columns (constants repeated for convenience)
future["mse_resid_xgb"]   = mse_resid_xgb

future=future.drop(['quarter', 't', 't2', 't3', 'reg_hat' ], axis=1)


#---------------------------------------
# saving results
#---------------------------------------

# google cloud local setting
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'deminimishelper-53e328982b5b.json'

BUCKET_NAME = "deminimishelper"
PREFIX = "models"
series_id = f"{int(g1['HTS22'].iloc[0])}_{g1['iso'].iloc[0]}"

storage_client = storage.Client()

bucket = storage_client.bucket(BUCKET_NAME)

def gcs_upload_bytes(path: str, data: bytes, content_type: str):
    blob = bucket.blob(path)
    blob.upload_from_string(data, content_type=content_type)

# save regression results
reg_buf = io.BytesIO()
joblib.dump(res, reg_buf)

gcs_upload_bytes(
    f"{PREFIX}/reg/{series_id}.joblib",
    reg_buf.getvalue(),
    content_type="application/octet-stream",
)
print("Uploaded:", f"gs://{BUCKET_NAME}/{PREFIX}/reg/{series_id}.joblib")

# save arima
arima_buf = io.BytesIO()
joblib.dump(sf, arima_buf)  # results from AutoARIMA is saved into sf
gcs_upload_bytes(
    f"{PREFIX}/arima/{series_id}.joblib",
    arima_buf.getvalue(),
    content_type="application/octet-stream",
)

# save xgboos
model_bytes = xgb.get_booster().save_raw(raw_format="ubj")

gcs_upload_bytes(
    f"{PREFIX}/xgb/{series_id}.ubj",
    model_bytes,
    content_type="application/octet-stream",
)

print(f"Saved to: gs://{BUCKET_NAME}/{f"{PREFIX}/xgb/{series_id}.ubj"}")