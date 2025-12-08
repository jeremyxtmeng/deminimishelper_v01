# created: 12/07/2025
# last updated: 12/07/2025
# calculate the embeddings of the descriptions of medical devices

import os
import numpy as np
import pandas as pd
from typing import List, Dict

# for embeddings
from sentence_transformers import SentenceTransformer

# part 1
# loading data
CSV_PATH = "med_goods_hts22_final.csv"  # adjust path if needed

df = pd.read_csv(CSV_PATH)

# selecting medical devices
df = df[
    (df["cat"] == "Medical device") &
    df["product"].notna() &               # only use HS codes in the latest revisions       
    (df["product"].astype(str).str.strip() != "")].copy()

df = df[['cat','product', 'HTS22']].reset_index(drop=True)
df['HTS22'] =df['HTS22'].astype(pd.StringDtype())

# part 2 
# loading the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# part 3
# calculate embedding of the medical devices
products: List[str] = df["product"].astype(str).tolist()

catalog_embeddings = model.encode(
    products,
    batch_size=32,
    show_progress_bar=False,
    normalize_embeddings=True
)
catalog_embeddings = np.asarray(catalog_embeddings)
#print("Embedding matrix shape:", catalog_embeddings.shape)
np.save('app_search_hscode_embeddings.npy', catalog_embeddings)
df.to_csv('app_search_hscode_df.csv', index=False)

