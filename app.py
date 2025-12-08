# created: 12/07/2025
# last updated: 12/07/2025
# app for searching hs code

import os
import numpy as np
import pandas as pd
from typing import List, Dict
from flask_cors import CORS

# for embeddings
from sentence_transformers import SentenceTransformer

# for using gemini api
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

import google.generativeai as genai
import re

# packages for postgre
import psycopg2
from psycopg2.extras import RealDictCursor

# load product info and embeddings
catalog_embeddings = np.load('app_search_hscode_embeddings.npy')
df = pd.read_csv('app_search_hscode_df.csv')
df['HTS22'] =df['HTS22'].astype(pd.StringDtype())

#-----------------------------------------------------------------
# classify goods
#-----------------------------------------------------------------
# loading the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def classify_goods(user_des: str)-> Dict: 
    user_emb = model.encode(
        [user_des],
        normalize_embeddings=True)  # shape: (1, dim)
    
    user_emb = np.asarray(user_emb)[0]  # shape: (dim,)
    #print("Embedding matrix shape:", user_emb.shape)
    similarities = catalog_embeddings @ user_emb  # dot product 
                                              # (# product, 1)
    # return top 5 indices
    top_indices = np.argsort(-similarities)[:5]

    candidates = []
    for idx in top_indices:
        cos_sim = float(similarities[idx])
        conf = float((cos_sim + 1.0) / 2.0)
        row = df.iloc[idx]
        candidates.append({
            "hs10": row["HTS22"],
            "product": row["product"],
            "similarity": cos_sim,
            "confidence": conf})
    best = candidates[0]

    return {
        "input": user_des,
        "hs10": best["hs10"],
        "product":best["product"],
        "confidence":best["confidence"]
    }

# test classify function
#test_hs=classify_goods("Wheelchair")
#print(test_hs.get("hs10"))

#-----------------------------------------------------------------
# Configure Gemini
#-----------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("no GEMINI_API_KEY environment variable.")

genai.configure(api_key=GEMINI_API_KEY)

# pick a fast model; adjust as needed

#from google import genai
#client = genai.Client()
#print("List of models that support generateContent:\n")
#for m in client.models.list():
#    for action in m.supported_actions:
#        if action == "generateContent":
#            print(m.name)

GEMINI_MODEL_NAME = "gemma-3-4b-it"

# other options
# gemini-2.5-flash
#models/gemma-3-1b-it
#models/gemma-3-4b-it
#models/gemma-3-12b-it
#models/gemma-3-27b-it

gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

#-----------------------------------------------------------------
# functions on the prompt to gemini
#-----------------------------------------------------------------


#---------------1: determining valid description-------------------
def ask_gemini(prompt: str) -> str:
    response = gemini_model.generate_content(prompt)
    return (response.text or "").strip()

def validity_with_gemini(text: str)-> dict:
    validation_prompt = f"""
    You are a validator to determine true or false

    Task: Decide if the following texts are the description of a product or the name of a product. Say "true" when the likelihood is likelihood is not zero.

    Return ONLY valid JSON with a single key "is_a_product" whose value is either true or false.

    Example of the required format:
    {{"is_a_product": true}}

    Text:
    \"\"\"{text}\"\"\"
    """
    raw = ask_gemini(validation_prompt)

    s = raw.strip()

    # Remove leading ```... language fences
    # e.g. ```json\n or ```\n
    s = re.sub(r"^```[a-zA-Z]*\s*", "", s)

    # Remove trailing ```
    s = re.sub(r"\s*```$", "", s)

    return json.loads(s)


#--------------2: gemini as a fallback strategy--------------------
LOW_CONF_THRESHOLD = 0.76

def classify_with_gemini(text: str) -> dict:
    prompt = f"""
    You are helping with a rough HS classification at 10 digit level when the internal database has low confidence.

    Given the following product description, suggest:
        - "hs10_guess": a plausible HS10 (10-digit) or 6-digit HS code as a string
        - "label": a concise English description of the product

    Return ONLY valid JSON in this exact format:
        {{"hs10_guess": "<string>", "label": "<string>"}}

    Description:
        \"\"\"{text}\"\"\"
    """
    raw = ask_gemini(prompt)
    s = raw.strip()

    # Remove leading ```... language fences
    # e.g. ```json\n or ```\n
    s = re.sub(r"^```[a-zA-Z]*\s*", "", s)

    # Remove trailing ```
    s = re.sub(r"\s*```$", "", s)

    return json.loads(s)


app = Flask(__name__)

CORS(
    app,
    resources={r"/api/*": {"origins": "*"}},
    supports_credentials=False,
)


#-----------------------------------------------------------------
# setting up limits to user requests
#-----------------------------------------------------------------
RATE_LIMIT = 50          # max prompts per IP
WINDOW_SECONDS = 300    # 5-minute window

POSTGRES_DSN = os.environ.get(
    "POSTGRES_DSN",
    "dbname=v0_db user=postgres password=lemontree host=localhost port=5432",
)

def get_db_conn():
    conn = psycopg2.connect(POSTGRES_DSN, cursor_factory=RealDictCursor) # one connection per request
    return conn

def is_rate_limited(ip: str) -> bool:
    conn = get_db_conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT COUNT(*) AS cnt
                FROM user_requests
                WHERE ip = %s
                  AND "time" >= NOW() - INTERVAL '{WINDOW_SECONDS} seconds'
                """,
                (ip,),
            )
            row = cur.fetchone()
            count = row["cnt"] if row is not None else 0
    finally:
        conn.close()

    return count >= RATE_LIMIT


def log_request(ip: str, prompt: str) -> None:
    conn = get_db_conn()
    try:
        with conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_requests (ip, "time", prompt)
                VALUES (%s, NOW(), %s)
                """,
                (ip, prompt),
            )
        conn.commit()
    finally:
        conn.close()
##################################################################
#-----------------------------------------------------------------
# classification api
#-----------------------------------------------------------------
@app.route("/api/medical-classify", methods=["POST"])

def main_app():
    body = request.get_json(force=True)
    text = (body.get("prompt") or "").strip()

    # 0) start logging
    # get IP and enforce rate limit
    ip = (request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip())

    if is_rate_limited(ip):
        return jsonify({
            "ok": False,
            "message": "Your limit is reached and please come back in a few minutes."
        }), 429

    # Log this prompt in user_requests
    log_request(ip, text)

    if not text:
        return jsonify({"ok": False, "message": "Prompt is empty."}), 400

    # 1) Use Gemini to validate if this is a medical-device description
    try:
        parsed=validity_with_gemini(text)
    # 2) If Gemini output isn't valid JSON, return error
    except Exception as e:
        return jsonify({
            "ok": False,  
            "message": "Your description is not validated. Please describe ONE medical device or equipment."
        }), 200

    is_med = bool(parsed.get("is_a_product"))

    if not is_med:
    # 3) Not a valid device description -> tell user to try again
        return jsonify({
            "ok": False,
            "message": "Your description cannot be validated to be describing a product. Please describe ONE medical device or equipment."
        }), 200

    # 4) Valid device description -> run the HS10 classifier
    try:
        hs_result = classify_goods(text) or {}
    except Exception as e:
        return jsonify({
            "ok": False,
            "message": f"Internal error while classifying HS10: {e}"
        }), 500
    
    confidence = None
    try:
        confidence = float(hs_result.get("confidence"))
    except (TypeError, ValueError):
        confidence = None

    # --- 5a) Low confidence: use Gemini as fallback classifier ---
    if confidence is None or confidence < LOW_CONF_THRESHOLD:
        fallback_msg = (
            "Your description is not in the pilot version of the database, which only contains medical devices. "
            "I will let a Gen AI help you with a rough classification, though it may not be accurate. "
            "Please enter a new product description afterwards.")

        try:
            gem_cls = classify_with_gemini(text)
            hs10_guess = gem_cls.get("hs10_guess")
            label_guess = gem_cls.get("label")
        except Exception:
            return jsonify({
                "ok": True,
                "hs10": None,
                "label": None,
                "extra": {"confidence": confidence, "fallback": "gemini_error"},
                "message": fallback_msg + " (However, the AI classifier encountered an error.)", 
                # "validator_reason": reason,
            }), 200

        return jsonify({
            "ok": True,
            "hs10": hs10_guess,
            "label": label_guess,
            "extra": {
                "confidence": confidence,
                "fallback": "Gemini 3"
            },
            "message": fallback_msg, 
            #"validator_reason": reason,
        }), 200

    # --- 5b) High confidence: normal classification + follow-up guidance ---
    followup_prompt = (
        "What would you like to know? If you tell me the sourcing country, I can tell you the latest "
        "information on trade policy and supply chain."
    )

    return jsonify({
        "ok": True,
        "hs10": hs_result.get("hs10"),
        "label": hs_result.get("product"),
        #"extra": extra,
        # "validator_reason": reason,
        "followup_prompt": followup_prompt,
        "message": None,  # you can also put a generic message here if you like
    }), 200

    #return jsonify({
    #    "ok": True,
    #    "hs10": hs_result.get("hs10"),
    #    "label": hs_result.get("product"),
    #    }), 200

##############################################################3
#---------------------------------
COUNTRY_NAMES = [
    "Bangladesh","Indonesia","Venezuela","Kiribati","Cameroon","Luxembourg",
    "Czech Republic","Sweden","Montenegro","Uganda","Jordan","Dominican Republic",
    "Saint Helena","Cambodia","Ireland","Macedonia","Singapore","Sri Lanka",
    "San Marino","Brunei","Uzbekistan","Portugal","Finland","Malta","Colombia",
    "Albania","Cayman Islands","Saudi Arabia","Ukraine","Cote d'Ivoire","Latvia",
    "Kyrgyzstan","France","Maldives","Slovakia","Israel","Ghana","Kenya","Senegal",
    "Malaysia","Iceland","Madagascar","Hong Kong","Sierra Leone","Philippines",
    "Guinea","Cyprus","Turkey","Nigeria","Cocos (Keeling) Islands",
    "Laos (Lao People's Democratic Republic)","China","Bosnia and Hercegovina",
    "Armenia","Belarus","Qatar","Netherlands","Gabon","Paraguay","Martinique",
    "Australia","Serbia","Mauritius","Angola","Libya","Bahrain","Spain",
    "United Arab Emirates","Georgia","Malawi","Belgium","Monaco","Curacao","Taiwan",
    "Solomon Islands","Thailand","Germany (Federal Republic of Germany)","Togo",
    "Niue","El Salvador","Italy","Uruguay","Oman","Congo","Republic of the Congo",
    "Eswatini","Fiji","United Kingdom","South Korea (Republic of Korea)","Canada",
    "Barbados","Bermuda","Marshall Islands","Argentina","Liechtenstein",
    "Azerbaijan","Slovenia","Egypt","Greece","Bahamas","Afghanistan","Denmark, except Greenland",
    "India","Saint Kitts and Nevis","French Polynesia","Chile","Estonia","Vietnam",
    "Suriname","South Africa","Peru","Kazakhstan","Guadeloupe","Japan","Macao",
    "Jamaica","Trinidad and Tobago","Mongolia","Mozambique","Seychelles","Switzerland",
    "Ecuador","New Zealand","Hungary","Russia","Belize","Norway","Honduras",
    "Botswana","Pakistan","Romania","Brazil","Austria","Guatemala","Bolivia",
    "Ethiopia","Niger","Panama","Lithuania","Bulgaria","Croatia","Tunisia","Aruba",
    "Mali","Morocco","Moldova","Myanmar","Nicaragua","Mexico","Nepal","Tonga",
    "Guyana","Tanzania","Poland","Greenland","Lebanon","Costa Rica","Haiti","Andorra"
]

def remove_parentheses_content(text):
  return re.sub(r'\([^)]*\)', '', text)

def extract_country(text: str) -> str | None:
    tl = text.lower()
    best_match = None
    for name in COUNTRY_NAMES:
        nl = remove_parentheses_content(name.lower())
        n2 = nl.strip()
        if n2 in tl:
            best_match = name
    return best_match

from sqlalchemy import create_engine, text

db_url = "postgresql+psycopg2://postgres:lemontree@127.0.0.1:5432/v0_db"
engine = create_engine(db_url)

def get_tariffs_by_country(cntry: str, hs10: int) -> float:
    """
    Return a pandas Series of tariff values for a given country_name.
    """
    query = text(f"""
        SELECT col1_duty + tariff_temp_total AS tariff
        FROM {"tariff_rate_2025_08"}
        WHERE name = :cntry 
            AND "HTS22"=:hs10
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"cntry": cntry, "hs10":hs10})

    if df.empty:
        return None  # or raise ValueError("No match found")

    # returns just the tariff column (Series)
    return df.iat[0, 0]

def get_price_by_country(cntry: str, hs10: int) -> float:
    """
    Return a pandas Series of unit prices for a given country_name.
    """
    query = text(f"""
       SELECT
         COALESCE(
           MAX(
             CASE
               WHEN "GEN_QY1_MO" = 0 THEN 0
               ELSE ("DUT_VAL_MO" + "GEN_CIF_MO")::numeric / "GEN_QY1_MO"
             END
           ),
           0
         ) AS unit_price
        FROM {"trade_flow_2025_07"}
        WHERE name = :cntry 
            AND "HTS22"=:hs10
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"cntry": cntry, "hs10":hs10})

    if df.empty:
        return None  # or raise ValueError("No match found")

    # returns just the tariff column (Series)
    return df.iat[0, 0]

DASHBOARD_URL = "https://v.0.1.deminimishelper.com/"

@app.post("/api/trade-info")
def trade_info():
    body = request.get_json(force=True)
    text = (body.get("prompt") or "").strip()
    hs10 = (body.get("hs10") or "").strip()

    #---------------start logging-----------------------------
    # get IP and enforce rate limit
    ip = (request.headers.get("X-Forwarded-For", request.remote_addr or "unknown").split(",")[0].strip())

    if is_rate_limited(ip):
        return jsonify({
            "ok": False,
            "message": "Your limit is reached and please come back in a few minutes.",
        }), 429

    # Log this prompt in user_requests
    log_request(ip, text)
    # -------------- end of logging block --------------------

    if not text or not hs10:
        return jsonify({
            "ok": False,
            "message": "Missing prompt or HS10 to continue."
        }), 400

    country = extract_country(text)
    if not country:
        return jsonify({
            "ok": False,
            "message": "I could not detect a country name from your input. "
                       "Please enter the description of another product. I also have a dashboard on supply chain forecasting using the link below",
            "dashboard_url": DASHBOARD_URL,
            "reset_to_product": True # go back to product description stage
        }), 200

    # Call your backend functions (e.g., query PostgreSQL, etc.)
    try:
        a_arr = get_tariffs_by_country(country,hs10)
        b_arr = get_price_by_country(country,hs10)
    except Exception as e:
        return jsonify({
            "ok": False,
            "message": f"Error while computing trade info: {e}",
            #"dashboard_url": DASHBOARD_URL,
            "reset_to_product": True
        }), 500

    # Convert numpy arrays (or other containers) to scalars
    def to_scalar(x):
        if isinstance(x, (np.ndarray, list, tuple)) and len(x) > 0:
            return float(x[0])
        return float(x)

    try:
        a_val = to_scalar(a_arr)
        b_val = round(to_scalar(b_arr),3)
    except Exception as e:
        return jsonify({
            "ok": False,
            "message": f"I encounter an error in retrieving information. Do you want to try another country (e.g. Canada, China)?",
            #"message": f"Error converting trade values: {e}",
            "reset_to_product": False
        }), 500

    msg = (
        f"The tariff rate of goods ({hs10}) imported from {country} is {a_val} "
        f"in August, 2025 with a unit price of ${b_val} in July 2025. "
        #f"I also have a dashboard on supply chain forecasting using the link below. "
        f"Please enter the description of another product."
    )

    return jsonify({
        "ok": True,
        "message": msg,
        "hs10": hs10,
        "country": country,
        "a": a_val,
        "b": b_val,
        #"dashboard_url": DASHBOARD_URL,
        "reset_to_product": True  # frontend should go back to initial stage
    }), 200


if __name__ == "__main__":
    # Dev server on http://localhost:5000
    app.run(host="0.0.0.0", port=5000, debug=True)




