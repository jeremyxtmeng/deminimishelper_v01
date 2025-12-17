# created: 12/13/2025
# last upated: 12/17/2025
# prep data of med devices and developing forecasting models

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


#---------------------------------------------------------------------
#---------------------------------------------------------------------
# part 1.1: testing one series for time series without zero
#---------------------------------------------------------------------
#---------------------------------------------------------------------
import scipy.stats as stats
import seaborn as sns

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf

import matplotlib
import matplotlib.pyplot as plt

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from statsforecast.arima import arima_string

#----------------------------------------
# part 1.1.1: visual inspection
#----------------------------------------

print(df_final1['HTS22'].dtype)

g1=df_final1[(df_final1['HTS22']==3002120090) & (df_final1['iso']=='NL')]
print(f"\nThe date range: {g1['time'].min()} to { g1['time'].max()}")

g1=g1.set_index('time')
g1['unit_price']=g1['GEN_CIF_MO']/g1['GEN_QY1_MO']


# Adjusting the figure size
fig = plt.subplots(figsize=(16, 5))

# Creating a plot
#plt.plot(g1.index, g1['GEN_QY1_MO'])
plt.plot(g1.index, g1['unit_price'])

# Adding a plot title and customizing its font size
plt.title('GEN_QY1_MO', fontsize=20)

# Adding axis labels and customizing their font size
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


g1['ds'] =g1.index
g1['y']=np.log(g1['GEN_QY1_MO'])
g1['unique_id']=1

StatsForecast.plot(g1)

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
# part 2.1: XGboost
#---------------------------------------------------------------------
#---------------------------------------------------------------------

# Assume 'your_data.csv' has a column 'value' with your time series
# Or directly use a list/array
# Example:
# df = pd.read_csv('your_data.csv', parse_dates=['date'], index_col='date')
# series = df['value']  # or whatever your target column is

# For this example, replace with your actual series
series = pd.Series(your_time_series_values)  # e.g., list or array of ~90 floats

def create_lagged_dataset(series, n_lags=20):
    """
    Creates X (features) and y (target) using lagged values.
    n_lags: how many past observations to use (recommend 10-30 for your size)
    """
    df = pd.DataFrame({'target': series})
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df['target'].shift(lag)
    
    # Optional: add more features like rolling stats
    # df['rolling_mean_5'] = df['target'].shift(1).rolling(5).mean()
    # df['rolling_std_5'] = df['target'].shift(1).rolling(5).std()
    
    df = df.dropna().reset_index(drop=True)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    return X, y

# Recommended starting point
n_lags = 20  # Gives ~70 samples (90 - 20), adjust 10-30 based on experiments
X, y = create_lagged_dataset(series, n_lags=n_lags)

print(f"Samples after lagging: {len(X)}")  # Should be around 70
print(X.head())





import xgboost as xgb
 
num_days = 50
observations_per_day = 20
start_p = 0.9
end_p = 0.4
 
data = []
delta_p = (end_p - start_p) / (num_days - 1)
 
for day in range(num_days):
    # Simulate daily probability
    current_p = start_p + (day * delta_p)
    # Add some random daily component
    observations = np.random.binomial(1, current_p, observations_per_day)
    # Let's make the price based on past observations to simulate a drift
    price = current_p * 1000
    for obs in observations:
        data.append([day + 1, obs, price])
 
# Convert to a dataframe
df = pd.DataFrame(data, columns=['Day', 'Target', 'Price'])
df['Intercept'] = 1


test=df.groupby('Day')['Target'].mean()


train_df = df[df['Day'] <= 20]
test_df = df[df['Day'] > 20]
 
# Features and labels
X_train = train_df.drop(columns=['Target', 'Price','Day'])
y_train = train_df['Target']
X_test = test_df.drop(columns=['Target', 'Price', 'Day'])
y_test = test_df['Target']
 
print(X_train.head())
 
# Train the XGBoost model
model = xgb.XGBClassifier(eval_metric='logloss')
model.fit(X_train, y_train)
 
# Predict probabilities for the training set
y_train_pred_proba = model.predict_proba(X_train)[:, 1]
train_df['Predicted_Prob'] = y_train_pred_proba
 
# Predict probabilities for the test set
y_test_pred_proba = model.predict_proba(X_test)[:, 1]
test_df['Predicted_Prob'] = y_test_pred_proba
 
# Calculate the mean of the target variable and predicted probabilities for each day in both sets
daily_means_train = train_df.groupby('Day')['Target'].mean()
daily_probs_train = train_df.groupby('Day')['Predicted_Prob'].mean()
daily_means_test = test_df.groupby('Day')['Target'].mean()
daily_probs_test = test_df.groupby('Day')['Predicted_Prob'].mean()
