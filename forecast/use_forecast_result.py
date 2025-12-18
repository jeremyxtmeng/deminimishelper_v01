# created : 12/18/2025
# updated: 12/18/2025
# testing forecasting using results from Google Cloud Storage

import os, io
import numpy as np
from google.cloud import storage
import xgboost as xgb
import joblib


BUCKET_NAME = "deminimishelper"
PREFIX = "models"
series_id = "3002120090_NL"  # example

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

gcs_path = f"{PREFIX}/{series_id}.ubj"
blob = bucket.blob(gcs_path)
model_bytes = blob.download_as_bytes()

booster = xgb.Booster()
booster.load_model(bytearray(model_bytes))   # works with UBJ bytes


dmat = xgb.DMatrix(X, feature_names=feature_cols)
booster.predict(dmat)



def gcs_download_bytes(gcs_path: str) -> bytes:
    return bucket.blob(gcs_path).download_as_bytes()

def load_joblib_from_gcs(gcs_path: str):
    b = gcs_download_bytes(gcs_path)
    return joblib.load(io.BytesIO(b))

# --- Load regression (statsmodels results object) ---
ols_res = load_joblib_from_gcs(f"{PREFIX}/reg/{series_id}.joblib")

# Example usage:
# future_df must have the same columns used in training (quarter, t, t2, t3, etc.)
# reg_hat = ols_res.predict(future_df)

# --- Load ARIMA (StatsForecast object) ---
sf_full = load_joblib_from_gcs(f"{PREFIX}/arima/{series_id}.joblib")

# Example usage:
# Y_all_df must be a StatsForecast-format dataframe: columns [unique_id, ds, y]
# resid_fcst = sf_full.forecast(df=Y_all_df, h=12)

