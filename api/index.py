# created: 12/07/2025
# last updated: 12/07/2025
# app for searching hs code

import os
import requests
import numpy as np
from typing import List, Dict, Any
from flask_cors import CORS

# for using gemini api
import json
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

import google.generativeai as genai
import re

# packages for postgre
from datetime import datetime, timezone, timedelta
from supabase import create_client, Client
from io import BytesIO


#-----------------------------------------------------------------
# Configure Gemini
#-----------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("no GEMINI_API_KEY environment variable.")

genai.configure(api_key=GEMINI_API_KEY)

GEMINI_MODEL_NAME = "gemma-3-4b-it"

gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

#----------------------------------------
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


#app = Flask(__name__)

#CORS(
#    app,
#    resources={r"/api/*": {"origins": "*"}},
#    supports_credentials=False,
#)


#-----------------------------------------------------------------
# setting up limits to user requests
#-----------------------------------------------------------------
RATE_LIMIT = 50          # max prompts per IP
WINDOW_SECONDS = 300    # 5-minute window

# Supabase HTTP client (keep this if you still use Supabase auth/storage/etc.)
supabase_url: str = os.environ.get("SUPABASE_URL")
supabase_key: str = os.environ.get("SUPABASE_ANON_KEY")
supabase: Client = create_client(supabase_url, supabase_key)


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

def log_request(ip: str, prompt: str) -> None:

    now_iso = datetime.now(timezone.utc).isoformat()

    payload = {
        "ip": ip,
        "time": now_iso,
        "prompt": prompt,
    }
    supabase.table("user_requests").insert(payload).execute()

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


@app.post("/api/medical-classify")
##################################################################
#-----------------------------------------------------------------
# classification api
#-----------------------------------------------------------------
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
   
    # --- 5b) High confidence: normal classification + follow-up guidance ---
    #log_request(ip, hs_result.get("hs10"))

    return jsonify({
        "ok": True,
        "hs10": int(hs_result.get("hs10")),
        "label": hs_result.get("product"),
        "followup_prompt": followup_prompt}), 200


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

def get_tariffs_by_country(cntry: str, hs10: int) -> float:
    resp = (
        supabase
        .table("tariff_rate_2025_08")
        .select("col1_duty, tariff_temp_total")
        .eq("name", cntry)
        .eq("hts22",  hs10 )
        .limit(1)
        .execute()
    )

    rows = resp.data 

    row = rows[0]
    col1_duty = row.get("col1_duty") 
    tariff_temp_total = row.get("tariff_temp_total") 

    return float(col1_duty) + float(tariff_temp_total)

def get_price_by_country(country: str, hs10: int) -> float:
    resp = (
        supabase
        .table("trade_flow_2025_07")
        .select('DUT_VAL_MO, GEN_CIF_MO, GEN_QY1_MO')
        .eq('name', country)
        .eq('hts22', hs10)
        .limit(1)              # just one row is enough
        .execute()
    )

    rows = resp.data or []

    row = rows[0]
    dut_val = row.get("DUT_VAL_MO") 
    cif_val = row.get("GEN_CIF_MO") 
    qty     = row.get("GEN_QY1_MO") 

    if not qty or qty == 0:
        return 0.0
    total = dut_val + cif_val
    return float(total) / float(qty)

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
        a_val = get_tariffs_by_country(country,hs10)
        b_val = get_price_by_country(country,hs10)
    

    # Convert numpy arrays (or other containers) to scalars
    #def to_scalar(x):
    #    if isinstance(x, (np.ndarray, list, tuple)) and len(x) > 0:
    #        return float(x[0])
    #    return float(x)

    #try:
    #    a_val = to_scalar(a_arr)
    #    b_val = round(to_scalar(b_arr),3)
    #except Exception as e:
    #    return jsonify({
    #        "ok": False,
    #        "message": f"I encounter an error in retrieving information. Do you want to try another country (e.g. Canada, China)?",
    #        "reset_to_product": False
    #    }), 500

        msg = (
            f"The tariff rate of goods ({hs10}) imported from {country} is {a_val} "
            f"in August, 2025 with a unit price of ${b_val} in July 2025. "
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

    except Exception as e:
        return jsonify({
            "ok": False,
            "message": f"Error while computing trade info: {e}",
            "reset_to_product": True
        }), 500


if __name__ == "__main__":
    # Dev server on http://localhost:5000
    app.run(host="0.0.0.0", port=5000, debug=True)