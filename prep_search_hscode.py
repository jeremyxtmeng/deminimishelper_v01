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




# alternative of using genai to calculate embedding

import google.generativeai as genai

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("no GEMINI_API_KEY environment variable.")

genai.configure(api_key=GEMINI_API_KEY)

def embed_with_gemini(text: str) -> np.ndarray:
    """
    Get a normalized embedding vector for `text` using Gemini embeddings.
    """
    resp = genai.embed_content(
        model="models/text-embedding-004",  # Gemini embedding model
        content=text,
    )
    vec = np.array(resp["embedding"], dtype=np.float32)  # shape: (dim,)

    # L2-normalize to mimic `normalize_embeddings=True`
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec

catalog_texts = df["product"].tolist()
emb_list = [embed_with_gemini(t) for t in catalog_texts]
catalog_embeddings = np.vstack(emb_list)  

np.save('app_search_hscode_embeddings_genai.npy', catalog_embeddings)
