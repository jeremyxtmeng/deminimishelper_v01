# created: 12/07/2025
# last updated: 12/18/2025
# app for searching hs code
# adding the forecasting

import os
import requests
import numpy as np
from typing import List, Dict, Any
from flask_cors import CORS

# for using gemini api
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from google import genai
from google.genai import types
import re

# packages for postgre
from datetime import datetime, timezone, timedelta
from supabase import create_client, Client
from io import BytesIO

# packages for forecasting
from google.cloud import storage
from typing import Optional
from google.oauth2 import service_account

#---------------------------------------------------------------------------------------------------
# Configure Gemini
#---------------------------------------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("no GEMINI_API_KEY environment variable.")

client = genai.Client(api_key=GEMINI_API_KEY)  # Gemini Developer API :contentReference[oaicite:1]{index=1}


GEMINI_MODEL_NAME = "gemma-3-4b-it"
#---------------------------------------------------------------------------------------------------
# Configure database
#---------------------------------------------------------------------------------------------------

# code for Vercel
supabase_url: str = os.environ.get("SUPABASE_URL")
supabase_key: str = os.environ.get("SUPABASE_ANON_KEY")
supabase: Client = create_client(supabase_url, supabase_key)


#---------------------------------------------------------------------------------------------------
# Configure google cloud storage
#---------------------------------------------------------------------------------------------------

def get_storage_client() -> storage.Client:
    try:
        # Works locally if ADC is configured
        return storage.Client()
    except Exception:
        project_id = os.environ["GCP_PROJECT_ID"]
        client_email = os.environ["GCP_SERVICE_ACCOUNT_EMAIL"]
        private_key = os.environ["GCP_PRIVATE_KEY"].replace("\\n", "\n")

        info = {
            "type": "service_account",
            "project_id": project_id,
            "client_email": client_email,
            "private_key": private_key,
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        creds = service_account.Credentials.from_service_account_info(info)
        return storage.Client(project=project_id, credentials=creds)
    
#-----------------------------------------------------------------------------------------------------
# functions used to classify goods
#------------------------------------------------------------------------------------------------------
EMBED_URL = os.environ.get(
    "EMBED_URL",
    "https://qacmfkbhcngbmewhxihm.supabase.co/storage/v1/object/public/static_file/app_search_hscode_embeddings_genai.npy",
)
CATALOG_URL = os.environ.get(
    "CATALOG_URL",
    "https://qacmfkbhcngbmewhxihm.supabase.co/storage/v1/object/public/static_file/med_goods_hts22_final.json",
)

catalog_embeddings: np.ndarray | None = None
CATALOG: List[Dict[str, Any]] = []  # [{"hs10": int, "product": str}, ...]

def load_embeddings_from_cloud() -> None:
    global catalog_embeddings
    if catalog_embeddings is not None:
        return  # already loaded in this cold start

    try:
        print(f"[init] Downloading embeddings from {EMBED_URL}")
        resp = requests.get(EMBED_URL, timeout=10)
        resp.raise_for_status()

        bio = BytesIO(resp.content)
        catalog_embeddings_local = np.load(bio, allow_pickle=True, encoding="latin1")

        catalog_embeddings = catalog_embeddings_local.astype(np.float32)
        print(f"[init] Loaded embeddings, shape={catalog_embeddings.shape}")
    except Exception as e:
        print(f"[init] ERROR loading embeddings from cloud: {e}")
        catalog_embeddings = np.zeros((0, 0), dtype=np.float32)


def load_catalog_from_cloud() -> None:
    global CATALOG
    if CATALOG:
        return  # already loaded

    try:
        print(f"[init] Downloading catalog from {CATALOG_URL}")
        resp = requests.get(CATALOG_URL, timeout=10)
        resp.raise_for_status()

        CATALOG = json.loads(resp.text)
        print(f"[init] Loaded {len(CATALOG)} catalog rows from cloud")
    except Exception as e:
        print(f"[init] ERROR loading catalog from cloud: {e}")
        CATALOG = []


# loading the embedding model
def embed_with_gemini(text: str) -> np.ndarray:
    resp = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config={"output_dimensionality": 768},
    )

    vec = np.array(resp.embeddings[0].values, dtype=np.float32)

    # L2-normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm

    return vec

def classify_goods(user_des: str) -> Dict[str, Any]:
    global CATALOG, catalog_embeddings

    load_embeddings_from_cloud()
    load_catalog_from_cloud()

    if catalog_embeddings is None or catalog_embeddings.size == 0 or not CATALOG:
        raise RuntimeError("Catalog or embeddings not loaded on the server.")

    user_emb = embed_with_gemini(user_des)      # your existing embedding function
    similarities = catalog_embeddings @ user_emb  # (N,)

    top_indices = np.argsort(-similarities)[:5]

    candidates = []
    for idx in top_indices:
        cos_sim = float(similarities[idx])
        conf = float((cos_sim + 1.0) / 2.0)
        item = CATALOG[idx]  # {"hs10": int, "product": str}
        candidates.append({
            "hs10": item["hs10"],
            "product": item["product"],
            "similarity": cos_sim,
            "confidence": conf,
        })

    best = candidates[0]
    return {
        "input": user_des,
        "hs10": best["hs10"],
        "product": best["product"],
        "confidence": best["confidence"],
    }

#---------------------------------------------------------------------------------------------------
# functions to work with gemini
#---------------------------------------------------------------------------------------------------

#---------------1: determining valid description-------------------
def ask_gemini(prompt: str) -> str:
    response = client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=prompt,
    )
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



#---------------------------------------------------------------------------------------------------
# functions for setting up limits to user requests
#---------------------------------------------------------------------------------------------------
RATE_LIMIT = 10             # max prompts per IP
WINDOW_SECONDS = 300        # 5-minute window
WINDOW_SECONDS_STAGE2 = 60  # 5-minute window
STAGE2_RATE_LIMIT=3         # how many times to remind users 

def is_rate_limited(ip: str) -> bool:
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=WINDOW_SECONDS)
    cutoff_iso = cutoff.isoformat()

    resp = (
        supabase
        .table("user_requests")
        .select("id", count="exact") 
        .eq("ip", ip)
        .gte("time", cutoff_iso)
        .execute()
    )

    count = resp.count or 0
    return count >= RATE_LIMIT

def is_stage2_rate_limited(ip: str) -> bool:
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=WINDOW_SECONDS_STAGE2)
    cutoff_iso = cutoff.isoformat()

    resp = (
        supabase
        .table("user_requests")
        .select("id", count="exact")
        .eq("ip", ip)
        .eq("stage2", True)
        .gte("time", cutoff_iso)
        .execute()
    )

    count = resp.count or 0
    return count > STAGE2_RATE_LIMIT

def log_request(ip: str, prompt: str, hs10: Optional[int] = None) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()

    stage2 = hs10 is not None

    payload = {
        "ip": ip,
        "time": now_iso,
        "prompt": prompt,
        "stage2": stage2,
    }
    supabase.table("user_requests").insert(payload).execute()


#---------------------------------------------------------------------------------------------------
# functions to extract country names
#---------------------------------------------------------------------------------------------------

COUNTRY_NAMES = {'Albania':'AL','Armenia':'AM','Australia':'AU','Azerbaijan':'AZ','Barbados':'BB','Botswana':'BW','Czech Republic':'CZ','Ecuador':'EC','Eswatini':'SZ','Ethiopia':'ET','Guinea':'GN','Guyana':'GY','India':'IN','Jamaica':'JM','Kyrgyzstan':'KG','Laos':'LA','Malaysia':'MY','Mali':'ML','Niue':'NU','Saint Helena':'SH','San Marino':'SM','Solomon Islands':'SB','Suriname':'SR','United Arab Emirates':'AE','Afghanistan':'AF','Andorra':'AD','Angola':'AO','Argentina':'AR','Aruba':'AW','Austria':'AT','Bahamas':'BS','Bahrain':'BH','Bangladesh':'BD','Belarus':'BY','Belgium':'BE','Belize':'BZ','Bermuda':'BM','Bolivia':'BO','Bosnia and Hercegovina':'BA','Brazil':'BR','Brunei':'BN','Bulgaria':'BG','Cambodia':'KH','Cameroon':'CM','Canada':'CA','Cayman Islands':'KY','Chile':'CL','China':'CN','Cocos Keeling Islands':'CC','Colombia':'CO','Congo Democratic Zaire':'CD','Costa Rica':'CR','Cote dIvoire':'CI','Croatia':'HR','Curacao':'CW','Cyprus':'CY','Denmark except Greenland':'DK','Dominican Republic':'DO','Egypt':'EG','El Salvador':'SV','Estonia':'EE','Fiji':'FJ','Finland':'FI','France':'FR','French Polynesia':'PF','Gabon':'GA','Georgia':'GE','Germany':'DE','Ghana':'GH','Greece':'GR','Greenland':'GL','Guadeloupe':'GP','Guatemala':'GT','Haiti':'HT','Honduras':'HN','Hong Kong':'HK','Hungary':'HU','Iceland':'IS','Indonesia':'ID','Ireland':'IE','Israel':'IL','Italy':'IT','Japan':'JP','Jordan':'JO','Kazakhstan':'KZ','Kenya':'KE','Kiribati':'KI','Latvia':'LV','Lebanon':'LB','Libya':'LY','Liechtenstein':'LI','Lithuania':'LT','Luxembourg':'LU','Macao':'MO','Macedonia':'MK','Madagascar':'MG','Malawi':'MW','Maldives':'MV','Malta':'MT','Marshall Islands':'MH','Martinique':'MQ','Mauritius':'MU','Mexico':'MX','Moldova':'MD','Monaco':'MC','Mongolia':'MN','Montenegro':'ME','Morocco':'MA','Mozambique':'MZ','Myanmar':'MM','Nepal':'NP','Netherlands':'NL','New Zealand':'NZ','Nicaragua':'NI','Niger':'NE','Nigeria':'NG','Norway':'NO','Oman':'OM','Pakistan':'PK','Panama':'PA','Paraguay':'PY','Peru':'PE','Philippines':'PH','Poland':'PL','Portugal':'PT','Qatar':'QA','Republic of the Congo':'CG','Romania':'RO','Russia':'RU','Saint Kitts and Nevis':'KN','Saudi Arabia':'SA','Senegal':'SN','Serbia':'RS','Seychelles':'SC','Sierra Leone':'SL','Singapore':'SG','Slovakia':'SK','Slovenia':'SI','South Africa':'ZA','South Korea':'KR','Spain':'ES','Sri Lanka':'LK','Sweden':'SE','Switzerland':'CH','Taiwan':'TW','Tanzania':'TZ','Thailand':'TH','Togo':'TG','Tonga':'TO','Trinidad and Tobago':'TT','Tunisia':'TN','Turkey':'TR','Uganda':'UG','Ukraine':'UA','United Kingdom':'GB','Uruguay':'UY','Uzbekistan':'UZ','Venezuela':'VE','Vietnam':'VN'}

def remove_parentheses_content(text):
  return re.sub(r'\([^)]*\)', '', text)

def extract_country(text: str) -> str | None:
    tl = text.lower()
    best_match = None
    for name,_ in COUNTRY_NAMES.items():
        nl = remove_parentheses_content(name.lower())
        n2 = nl.strip()
        if n2 in tl:
            best_match = name
    return best_match


#---------------------------------------------------------------------------------------------------
# functions to extract tariffs and prices from the DB
#---------------------------------------------------------------------------------------------------

def get_tariffs_by_country(cntry: str, hs10: int) -> float:
    cntry = (cntry or "").strip()
    resp = (
        supabase
        .table("tariff_rate_2025_08")
        .select("col1_duty, tariff_temp_total, hts22, name")  # extra cols for debugging
        .eq("name", cntry)
        .eq("hts22", hs10)
        .limit(1)
        .execute()
    )

    rows = resp.data or []

    if not rows:
        return 0.0

    row = rows[0]
    col1_duty = row.get("col1_duty") or 0
    tariff_temp_total = row.get("tariff_temp_total") or 0

    return float(col1_duty) + float(tariff_temp_total)

def get_price_by_country(country: str, hs10: int) -> float:
    country = (country or "").strip()
    resp = (
        supabase
        .table("trade_flow_2025_07")
        .select("DUT_VAL_MO, GEN_CIF_MO, GEN_QY1_MO, hts22, name")
        .eq("name", country)
        .eq("hts22", hs10)
        .limit(1)
        .execute()
    )

    rows = resp.data or []
    if not rows:
        return 0.0

    row = rows[0]
    dut_val = row.get("DUT_VAL_MO")
    cif_val = row.get("GEN_CIF_MO") 
    qty = row.get("GEN_QY1_MO") 

    try:
        unit_price=(float(dut_val) + float(cif_val)) / float(qty)
        return unit_price
    except Exception:
        return 0.0 


#---------------------------------------------------------------------------------------------------
# functions to forecast
#---------------------------------------------------------------------------------------------------

def gemini_date_extraction(text: str): 
    prompt =f"""
    You are extracting information on year and month from a description. Determine the year and month. For the month, determine the quarter.
    Given the description, extract the following as integers
        - "valid_date": True if the description contains information of a date with descriptions of month and year after (inclusive) August, 01, 2025
        - "user_q": the quarter of the month in the description; if no available information then return 1
        - "user_m": the month in the description; if no avaialble information then return 1
        - "user_y": the year in the description; if no available information then return 2026

    Return ONLY valid JSON in this exact format:
        {{"valid_date": "<bool>", "user_q": "<int>", "user_m": "<int>", "user_y": "<int>"}}

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

    user_date=json.loads(s)

    user_q=int(user_date.get('user_q'))
    user_m=int(user_date.get('user_m'))
    user_y=int(user_date.get('user_y'))

    return user_q, user_m, user_y

# use existing forecast if t<=108, then directly pull forecasts
def forecast_pre_computed(t: int, hs10: int, iso: str) -> float:
    resp = (
        supabase
        .table("forecast_all_results")
        .select("t, forecast_level_xgb, hts22, iso")  # extra cols for debugging
        .eq("t", t)
        .eq("hts22", hs10)
        .eq("iso",iso)
        .limit(1)
        .execute()
    )

    rows = resp.data
    row = rows[0]
    return round(float(row.get("forecast_level_xgb")),4)


def forecast_resid_dict(hs10: int, iso: str) -> Dict[int, float]:
    resp = (
        supabase
        .table("forecast_all_results")
        .select("t, resid_hat")
        .eq("hts22", hs10)
        .eq("iso", iso)
        .execute()
    )

    rows: List[dict] = resp.data or []
    return {row["t"]: row["resid_hat"] for row in rows}


def main_forecast(user_q:int, user_m: int, user_y: int, hs10:int, iso: str)-> str:
    user_t=12*(user_y-2017)+(user_m-1)
    if user_t<103:
        y_hat=forecast_pre_computed(108, hs10, iso)
        meg_forecast=f'There are already stats for the date you entered. But the import demand is projected to be ${y_hat} in January 2026 using a ML (XGBoost) model.'
    elif 104<=user_t<=108:
        y_hat=forecast_pre_computed(user_t, hs10, iso)
        meg_forecast=f'The import demand is projected to be ${y_hat} in {user_y}-{user_m} using a ML (XGBoost) model.'
    else:
        BUCKET_NAME = "deminimishelper"
        PREFIX = "models"
        series_id = f'{hs10}_{iso}' 
        storage_client = get_storage_client()
        bucket = storage_client.bucket(BUCKET_NAME)

        gcs_path = f"{PREFIX}/{series_id}.json"
        blob = bucket.blob(gcs_path)
        json_string = blob.download_as_bytes().decode('utf-8') # Download the blob content as a string of bytes, then decode to utf-8
        pm= json.loads(json_string) # Parse the JSON string into a Python dictionary

        reg_hat=pm['constant']+int(user_q==1)*pm['fe_q1']+int(user_q==2)*pm['fe_q2']+int(user_q==3)*pm['fe_q3']+pm['t']*user_t+pm['t2']*(user_t**2)+pm['t3']*(user_t**3)

        i=109 # the period not forecasted
        rhat=forecast_resid_dict(hs10, iso)

        while i<=user_t:
            resid_hat_one_step=pm['ar1']*rhat[i-1]+pm['ar2']*rhat[i-2]+pm['ar3']*rhat[i-3]+pm['ar4']*rhat[i-4]+pm['ar5']*rhat[i-5]
            rhat[i]=resid_hat_one_step
            i+=1
        y_hat=round(np.exp(reg_hat+rhat[user_t])-1,4)
        meg_forecast=f'The import demand is projected to be ${y_hat} in {user_y}-{user_m} using an ARIMA model.'
    return meg_forecast

















####################################################################################################################################
####################################################################################################################################
#-----------------------------------------------------------------#-----------------------------------------------------------------
# start of the app
#-----------------------------------------------------------------#-----------------------------------------------------------------
####################################################################################################################################
####################################################################################################################################
app = Flask(__name__, template_folder="templates")

CORS(
    app,
    resources={r"/api/*": {"origins": "*"}},
    supports_credentials=False,
)

# ---------- Error handler so you always return JSON ----------
@app.errorhandler(Exception)
def handle_exception(e):
    print(f"[ERROR] {e.__class__.__name__}: {e}")
    return jsonify({
        "ok": False,
        "message": f"Server error: {e.__class__.__name__}: {e}",
    }), 500


@app.route("/")
def home():
    return render_template("index.html")

####################################################################################################################################
####################################################################################################################################
#-----------------------------------------------------------------#-----------------------------------------------------------------
# classification api
#-----------------------------------------------------------------#-----------------------------------------------------------------
####################################################################################################################################
####################################################################################################################################

@app.post("/api/medical-classify")

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
        hs_result = classify_goods(text)
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
        fallback_msg = ("Your description is not in the pilot version of the database, which only contains medical devices. I will let a Gen AI help you with a rough classification, though it may not be accurate. Please enter a new product description afterwards.")

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
        "I can forecast the import demand of this product and tell you the latest information of trade policy."
        " Please tell me one sourcing country (e.g. Canada, Germany, Thailand...) and a date (the exact format does not matter) after July 31, 2025 of your interest (e.g. July 2026)." 
    )
   
    # --- 5b) High confidence: normal classification + follow-up guidance ---
    #log_request(ip, hs_result.get("hs10"))

    return jsonify({
        "ok": True,
        "hs10": int(hs_result.get("hs10")),
        "label": hs_result.get("product"),
        "followup_prompt": followup_prompt}), 200


####################################################################################################################################
####################################################################################################################################
#-----------------------------------------------------------------#-----------------------------------------------------------------
# forecast api
#-----------------------------------------------------------------#-----------------------------------------------------------------
####################################################################################################################################
####################################################################################################################################

@app.post("/api/trade-info")

def trade_info():
    body = request.get_json(force=True)
    text = (body.get("prompt") or "").strip()
    hs10 = int(body.get("hs10") )

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
                       "Please enter the description of another product.",
            "reset_to_product": True # go back to product description stage
        }), 200

    # Call your backend functions (e.g., query PostgreSQL, etc.)
    try:
        iso=COUNTRY_NAMES[country]
        a_rr = get_tariffs_by_country(country,hs10)
        b_rr = get_price_by_country(country,hs10)
        user_q, user_m, user_y =gemini_date_extraction(text)
        try:
            meg_forecast=main_forecast(user_q, user_m, user_y, hs10, iso)
        except Exception:
            meg_forecast=f"The U.S. isn't expected to import from {country} in {user_y}-{user_m}."
    except Exception as e:
        return jsonify({
            "ok": False,
            "message": f"Error while computing trade info: {e}",
            "reset_to_product": True
        }), 500
    # Convert numpy arrays (or other containers) to scalars
    def to_scalar(x):
        if isinstance(x, (np.ndarray, list, tuple)) and len(x) > 0:
            return float(x[0])
        return float(x)

    try:
        a_val = to_scalar(a_rr)
        b_val = round(to_scalar(b_rr),3)
    except Exception as e:
        return jsonify({
            "ok": False,
            "message": f"I encounter an error in retrieving information. Do you want to try another country (e.g. Canada, China)?",
            "reset_to_product": False
        }), 500

    if b_val>0:
        msg = (
            f"The tariff rate of goods ({hs10}) imported from {country}({iso}) is {a_val} "
            f"in August, 2025 with a unit price of ${b_val} in July 2025. "
            f"{meg_forecast}"
            f"Please enter the description of another product."
        )
    else: 
        msg = (
            f"The tariff rate of goods ({hs10}) imported from {country}({iso}) is {a_val} "
            f"in August, 2025. However, no imports in July 2025. "
            f"{meg_forecast} "
            f"Please enter the description of another product."
        )

    return jsonify({
        "ok": True,
        "message": msg,
        "hs10": hs10,
        "country": country,
        "a": a_val,
        "b": b_val,
        "reset_to_product": True  # frontend back to the initial stage
    }), 200


if __name__ == "__main__":
    # Dev server on http://localhost:5000
    app.run(host="0.0.0.0", port=5000, debug=True)