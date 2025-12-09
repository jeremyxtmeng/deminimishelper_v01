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

#-------------------------------------------------------------
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

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/medical-classify", methods=["POST"])
    
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
        #----------------------------------------
        EMBED_URL = os.environ.get(
            "EMBED_URL",
            "https://qacmfkbhcngbmewhxihm.supabase.co/storage/v1/object/public/static_file/app_search_hscode_embeddings_genai.npy",
        )

        #------------------------------------------------------------------------------------
        resp = requests.get(EMBED_URL, timeout=10)
        resp.raise_for_status()

        bio = BytesIO(resp.content)
        catalog_embeddings_local = np.load(bio, allow_pickle=True, encoding="latin1")
        catalog_embeddings = catalog_embeddings_local.astype(np.float32)
 

        CATALOG = [{"hs10": 3002120010, "product": "Antisera and other blood fractions"}, {"hs10": 3002120020, "product": "Antisera and other blood fractions"}, {"hs10": 3002120030, "product": "Antisera and other blood fractions"}, {"hs10": 3002120040, "product": "Antisera and other blood fractions"}, {"hs10": 3002120090, "product": "Antisera and other blood fractions"}, {"hs10": 3006100100, "product": "Sterile surgical catgut, suture materials, tissue adhesives for wound closure, laminaria, laminaria tents, and absorbable hemostatics"}, {"hs10": 3006301000, "product": "Opacifying preparations for X-ray examinations; diagnostic reagents designed to be administered to the patient"}, {"hs10": 3006305000, "product": "Opacifying preparations for X-ray examinations; diagnostic reagents designed to be administered to the patient"}, {"hs10": 3006400000, "product": "Dental cements and other dental fillings etc"}, {"hs10": 3006700000, "product": "Gel preparations designed to be used in human or veterinary medicine as a lubricant for parts of the body for surgical operations or physical examinations or as a coupling agent between the body and medical instruments"}, {"hs10": 3407004000, "product": "Other modeling pastes (incl/ for dental impressions)"}, {"hs10": 3822110000, "product": "Malaria Diagnostic Test Kits"}, {"hs10": 3926201010, "product": "Surigcal and medical seamlesss gloves"}, {"hs10": 4015121000, "product": "Surgical gloves of vulcanized rubber other than hard rubber"}, {"hs10": 4015129000, "product": "dental/veterinary gloves, mittens, mitts"}, {"hs10": 7010100000, "product": "ampoules, glass, for conveyance or packng of goods"}, {"hs10": 7010900510, "product": "glass pharmaceutical containers, over 1 liter"}, {"hs10": 7010900520, "product": "glass pharmaceutical contnrs, capacity 033>=1 l"}, {"hs10": 7010900530, "product": "glass pharmaceutical containers, 015=> 033 l"}, {"hs10": 7010900540, "product": "glass pharmaceutical containers, <=015 liter"}, {"hs10": 7015905000, "product": "clock a noncorrctve spectcl glasses hlw sphrs etc"}, {"hs10": 7418201000, "product": "sanitary ware and parts thereof of brass"}, {"hs10": 7418205000, "product": "sanitary ware and parts thereof of copper ex brass"}, {"hs10": 8419200010, "product": "Medical, surgical or laboratory sterilizers"}, {"hs10": 8419200020, "product": "Medical, surgical or laboratory sterilizers"}, {"hs10": 8419905040, "product": "Parts of medical, surgical or laboratory sterilizers"}, {"hs10": 8539292000, "product": "lamps with glass lt=635 mm diameter lt=100 v"}, {"hs10": 8543100000, "product": "particle accelerators, nesoi"}, {"hs10": 8543309040, "product": "magnesium andes fr elctrpltng, elctrlsis/electphor"}, {"hs10": 8543309080, "product": "electroplat/electrolysis/electrophores mach nesoi"}, {"hs10": 8543708500, "product": "Electric Nerve Stimulation Machines & Apparatus"}, {"hs10": 8543709700, "product": "plasma cleaners that remove organic contaminants"}, {"hs10": 8543908845, "product": "particle accelerator parts"}, {"hs10": 8713100000, "product": "Carriages for disabled persons not mechanically propelled"}, {"hs10": 8714200000, "product": "Parts and accessories of carriages for disabled persons"}, {"hs10": 9001200000, "product": "sheets and plates of polarizing material"}, {"hs10": 9002190000, "product": "objective lenses and parts, nesoi"}, {"hs10": 9005100020, "product": "prism binoculars for use with infrared light"}, {"hs10": 9005804020, "product": "optical telescopes for use with infrared light"}, {"hs10": 9005804040, "product": "optical telescopes, nesoi"}, {"hs10": 9005806000, "product": "astronomical instruments and mounting, nesoi"}, {"hs10": 9005904000, "product": "binoc & telescope parts,incl goods of 9001 or 9002"}, {"hs10": 9005908001, "product": "binoculars, other optical telescopes parts,nesoi"}, {"hs10": 9006300000, "product": "cameras for underwater, aerial survey, medical etc"}, {"hs10": 9015808080, "product": "surveying, hydrographic, etc inst &appln, nesoi"}, {"hs10": 9015900150, "product": "parts and accessories of seismographs"}, {"hs10": 9015900160, "product": "parts & accessorie of oth geophysical inst & appln"}, {"hs10": 9015900190, "product": "pts for surveying inst & appln,exc compasses,nesoi"}, {"hs10": 9018113000, "product": "Electrocardiographs, and parts and accessories"}, {"hs10": 9018116000, "product": "Electrocardiographs, and parts and accessories"}, {"hs10": 9018119000, "product": "Electrocardiographs, and parts and accessories"}, {"hs10": 9018120000, "product": "Ultrasonic scanning apparatus"}, {"hs10": 9018130000, "product": "Magnetic resonance imaging apparatus"}, {"hs10": 9018140000, "product": "Scintigraphic electro-diagnostic apparatus used in medical, surgical, dental or veterinary sciences"}, {"hs10": 9018194000, "product": "Other electro-diagnostic apparatus and parts"}, {"hs10": 9018195500, "product": "Other electro-diagnostic apparatus and parts"}, {"hs10": 9018197500, "product": "Other electro-diagnostic apparatus and parts"}, {"hs10": 9018199530, "product": "Other electro-diagnostic apparatus and parts"}, {"hs10": 9018199535, "product": "Other electro-diagnostic apparatus and parts"}, {"hs10": 9018199550, "product": "Other electro-diagnostic apparatus and parts"}, {"hs10": 9018199560, "product": "Other electro-diagnostic apparatus and parts"}, {"hs10": 9018200040, "product": "Ultraviolet or infrared ray apparatus used in medical, surgical, dental or veterinary sciences, and parts and accessories thereof"}, {"hs10": 9018200080, "product": "Ultraviolet or infrared ray apparatus used in medical, surgical, dental or veterinary sciences, and parts and accessories thereof"}, {"hs10": 9018310040, "product": "Syringes, with or without needles; parts and accesssories"}, {"hs10": 9018310080, "product": "Syringes, with or without needles; parts and accesssories"}, {"hs10": 9018310090, "product": "Syringes, with or without needles; parts and accesssories"}, {"hs10": 9018320000, "product": "Tubular metal needles and needles for sutures and parts"}, {"hs10": 9018390020, "product": "Bougies, catheters, drains, and parts and accessories"}, {"hs10": 9018390040, "product": "Bougies, catheters, drains, and parts and accessories"}, {"hs10": 9018390050, "product": "Bougies, catheters, drains, and parts and accessories"}, {"hs10": 9018410000, "product": "Dental drill engines; parts and accessories"}, {"hs10": 9018494000, "product": "Instruments and appliances for dental science; parts and accessories, nesoi"}, {"hs10": 9018498040, "product": "Instruments and appliances for dental science; parts and accessories, nesoi"}, {"hs10": 9018498080, "product": "Instruments and appliances for dental science; parts and accessories, nesoi"}, {"hs10": 9018500000, "product": "Other Ophthalmic Instruments & Appliances & Parts"}, {"hs10": 9018901000, "product": "Optical instruments and appliances and parts and accessories thereof"}, {"hs10": 9018902000, "product": "Optical instruments and appliances and parts and accessories thereof"}, {"hs10": 9018903000, "product": "Optical instruments and appliances and parts and accessories thereof"}, {"hs10": 9018904000, "product": "Optical instruments and appliances and parts and accessories thereof"}, {"hs10": 9018905040, "product": "Optical instruments and appliances and parts and accessories thereof"}, {"hs10": 9018905080, "product": "Optical instruments and appliances and parts and accessories thereof"}, {"hs10": 9018906000, "product": "Optical instruments and appliances and parts and accessories thereof"}, {"hs10": 9018906400, "product": "Optical instruments and appliances and parts and accessories thereof"}, {"hs10": 9018906800, "product": "Optical instruments and appliances and parts and accessories thereof"}, {"hs10": 9018907520, "product": "Optical instruments and appliances and parts and accessories thereof"}, {"hs10": 9018907540, "product": "Optical instruments and appliances and parts and accessories thereof"}, {"hs10": 9018907560, "product": "Optical instruments and appliances and parts and accessories thereof"}, {"hs10": 9018907570, "product": "Optical instruments and appliances and parts and accessories thereof"}, {"hs10": 9018907580, "product": "Optical instruments and appliances and parts and accessories thereof"}, {"hs10": 9018908000, "product": "Optical instruments and appliances and parts and accessories thereof"}, {"hs10": 9019102010, "product": "Mechano-therapy appliances; massage apparatus; psychological aptitude-testing apparatus; parts and accessories thereof:"}, {"hs10": 9019102020, "product": "Mechano-therapy appliances; massage apparatus; psychological aptitude-testing apparatus; parts and accessories thereof:"}, {"hs10": 9019102030, "product": "Mechano-therapy appliances; massage apparatus; psychological aptitude-testing apparatus; parts and accessories thereof:"}, {"hs10": 9019102035, "product": "Mechano-therapy appliances; massage apparatus; psychological aptitude-testing apparatus; parts and accessories thereof:"}, {"hs10": 9019102045, "product": "Mechano-therapy appliances; massage apparatus; psychological aptitude-testing apparatus; parts and accessories thereof:"}, {"hs10": 9019102050, "product": "Mechano-therapy appliances; massage apparatus; psychological aptitude-testing apparatus; parts and accessories thereof:"}, {"hs10": 9019102090, "product": "Mechano-therapy appliances; massage apparatus; psychological aptitude-testing apparatus; parts and accessories thereof:"}, {"hs10": 9019104000, "product": "Mechano-therapy appliances; massage apparatus; psychological aptitude-testing apparatus; parts and accessories thereof:"}, {"hs10": 9019106000, "product": "Mechano-therapy appliances; massage apparatus; psychological aptitude-testing apparatus; parts and accessories thereof:"}, {"hs10": 9019200000, "product": "Ozone therapy, oxygen therapy, aerosol therapy, artificial respiration or other therapeutic respiration apparatus; parts and accessories thereof"}, {"hs10": 9021294000, "product": "Dental Fittings and Parts and Accessories"}, {"hs10": 9021298000, "product": "Dental Fittings and Parts and Accessories"}, {"hs10": 9022120000, "product": "Computed tomography apparatus"}, {"hs10": 9022130000, "product": "Apparatus base on X-ray for dental, uses, nesoi"}, {"hs10": 9022140000, "product": "Apparatus based on the use of X-rays for medical, surgical or veterinary uses (other than computed tomography apparatus)"}, {"hs10": 9022210000, "product": "Apparatus based on the use of alpha, beta or gamma radiations, for medical, surgical, dental or veterinary use"}, {"hs10": 9022300000, "product": "X-Ray Tubes"}, {"hs10": 9022904000, "product": "X-ray generators, high tension generators, desks, screens, examination or treatment tables, chairs and similar apparatus, nesoi"}, {"hs10": 9022906000, "product": "X-ray generators, high tension generators, desks, screens, examination or treatment tables, chairs and similar apparatus, nesoi"}, {"hs10": 9025112000, "product": "Clinical thermometers, liquid-filled"}, {"hs10": 9025198010, "product": "Clinical Thermometer, non-liquid filled"}, {"hs10": 9025198020, "product": "Clinical Thermometer, non-liquid filled"}, {"hs10": 9025801000, "product": "elec hydrometers,therometers,pyrometers, etc"}, {"hs10": 9025801500, "product": "barometers, not combined w/ oth instruments, nesoi"}, {"hs10": 9025802000, "product": "hydrometers and similar floating instruments"}, {"hs10": 9025803500, "product": "hygrometers and psychrometers, non recording"}, {"hs10": 9025804000, "product": "thermographs, barographs, hygrographs"}, {"hs10": 9025805000, "product": "oth inst,recording or not,any combination of these"}, {"hs10": 9025900600, "product": "pts, hydrometers,therometers,pyrometers, etc,nesoi"}, {"hs10": 9027106000, "product": "gas or smoke analysis apparatus, nesoi"}, {"hs10": 9027205030, "product": "electrical electrophoresis instruments"}, {"hs10": 9027205050, "product": "electrical gas chromatographs"}, {"hs10": 9027205060, "product": "electrical liquid chromatographs"}, {"hs10": 9027205080, "product": "electrl chromatograph & electrophores inst, nesoi"}, {"hs10": 9027208030, "product": "gas chromatographs, except electrical"}, {"hs10": 9027208060, "product": "liquid chromatographs, except electrical"}, {"hs10": 9027208090, "product": "chromatographs&electrophoresis inst,exc elec,nesoi"}, {"hs10": 9027504015, "product": "Electrical instruments and apparatus using optical radiations (ultraviolet, visible, infrared), nesoi"}, {"hs10": 9027504020, "product": "Electrical instruments and apparatus using optical radiations (ultraviolet, visible, infrared), nesoi"}, {"hs10": 9027504060, "product": "Electrical instruments and apparatus using optical radiations (ultraviolet, visible, infrared), nesoi"}, {"hs10": 9027508015, "product": "Other instruments and apparatus using optical radiations (ultraviolet, visible, infrared); Chemical analysis instruments and apparatus; Thermal analysis instruments and apparatus"}, {"hs10": 9027508020, "product": "Other instruments and apparatus using optical radiations (ultraviolet, visible, infrared); Chemical analysis instruments and apparatus; Thermal analysis instruments and apparatus"}, {"hs10": 9027508060, "product": "Other instruments and apparatus using optical radiations (ultraviolet, visible, infrared); Chemical analysis instruments and apparatus; Thermal analysis instruments and apparatus"}, {"hs10": 9027894530, "product": "Electrical instruments and apparatus for physical or chemical analysis, measuring viscosity, checking heat, sound, light, etc, nesoi"}, {"hs10": 9027894560, "product": "Electrical instruments and apparatus for physical or chemical analysis, measuring viscosity, checking heat, sound, light, etc, nesoi"}, {"hs10": 9027894590, "product": "Electrical instruments and apparatus for physical or chemical analysis, measuring viscosity, checking heat, sound, light, etc, nesoi"}, {"hs10": 9027898030, "product": "other chemical analysis instruments, nesoi"}, {"hs10": 9027898060, "product": "other physical analysis instruments, nesoi"}, {"hs10": 9027898090, "product": "oth inst, measuring/checking viscosity etc,nesoi"}, {"hs10": 9027902000, "product": "Microtomes"}, {"hs10": 9027904500, "product": "Printed circuit assemblies for the goods of subheading 902780"}, {"hs10": 9027905650, "product": "Parts and accessories of electrical instruments and apparatus of subheading 902720, 902730, 902750 or 902780"}, {"hs10": 9027905695, "product": "Parts and accessories of electrical instruments and apparatus of subheading 902720, 902730, 902750 or 902780"}, {"hs10": 9027905995, "product": "Other parts and accessories of other electrical instruments and apparatus of heading 9027, nesoi"}, {"hs10": 9027906400, "product": "pts non elec optic inst 0f 902720,30,40,50,80"}, {"hs10": 9030100000, "product": "inst for measuring/detecting ionizing radiations"}, {"hs10": 9030902500, "product": "printed circuit assemblies for articles of 903010"}, {"hs10": 9030904600, "product": "parts for articles of subheading 903010, nesoi"}, {"hs10": 9030906800, "product": "printed circuit assemblies excpt for 903010,nesoi"}, {"hs10": 9402100000, "product": "Dentists', barbers' or similar chairs and parts thereof"}, {"hs10": 9402900010, "product": "Other medical, surigcal, and dental furniture"}, {"hs10": 9402900020, "product": "Other medical, surigcal, and dental furniture"}, {"hs10": 9620001500, "product": "mono- bi- tripods etc as accessories for head 9005"}, {"hs10": 9620003050, "product": "mono- bi- tripods etc accessories for seismographs"}, {"hs10": 9620003060, "product": "mono- bi- tripods etc for other geophysical instru"}, {"hs10": 9620003090, "product": "mono- bi- tripods etc accessor for head 9015 nesoi"}, {"hs10": 3005101000, "product": "Adhesive dressings and other articles WITH an adhesive layer"}, {"hs10": 3005105000, "product": "Adhesive dressings and other articles WITH an adhesive layer"}, {"hs10": 3005901000, "product": "Wadding, gauze and bandages, etc WITHOUT an adhesive layer"}, {"hs10": 3005905010, "product": "Wadding, gauze and bandages, etc WITHOUT an adhesive layer"}, {"hs10": 3005905090, "product": "Wadding, gauze and bandages, etc WITHOUT an adhesive layer"}, {"hs10": 6115100500, "product": "Graduated Compression Hosiery"}, {"hs10": 6115101000, "product": "Graduated Compression Hosiery"}, {"hs10": 6115101510, "product": "Graduated Compression Hosiery"}, {"hs10": 6115101520, "product": "Graduated Compression Hosiery"}, {"hs10": 6115101540, "product": "Graduated Compression Hosiery"}, {"hs10": 6115103000, "product": "Graduated Compression Hosiery"}, {"hs10": 6115104000, "product": "Graduated Compression Hosiery"}, {"hs10": 6115105500, "product": "Graduated Compression Hosiery"}, {"hs10": 6115106000, "product": "Graduated Compression Hosiery"}, {"hs10": 9021214000, "product": "Artificial Teeth and Parts and Accessories"}, {"hs10": 9021218000, "product": "Artificial Teeth and Parts and Accessories"}, {"hs10": 9021904040, "product": "Appliances Worn, Carried, Implanted In Body & Parts,Nesoi"}, {"hs10": 9021904080, "product": "Appliances Worn, Carried, Implanted In Body & Parts,Nesoi"}, {"hs10": 9021908100, "product": "Appliances Worn, Carried, Implanted In Body & Parts,Nesoi"}, {"hs10": 9022900500, "product": "X-ray generators, high tension generators, desks, screens, examination or treatment tables, chairs and similar apparatus, nesoi"}, {"hs10": 9022901500, "product": "X-ray generators, high tension generators, desks, screens, examination or treatment tables, chairs and similar apparatus, nesoi"}, {"hs10": 9022902500, "product": "X-ray generators, high tension generators, desks, screens, examination or treatment tables, chairs and similar apparatus, nesoi"}, {"hs10": 9022907000, "product": "X-ray generators, high tension generators, desks, screens, examination or treatment tables, chairs and similar apparatus, nesoi"}, {"hs10": 9022909500, "product": "X-ray generators, high tension generators, desks, screens, examination or treatment tables, chairs and similar apparatus, nesoi"}, {"hs10": 3821000010, "product": "Prepared culture media for development or maintenance of micro-organisms (including viruses and the like) or of plant, human or animal cells"}, {"hs10": 3821000090, "product": "Prepared culture media for development or maintenance of micro-organisms (including viruses and the like) or of plant, human or animal cells"}, {"hs10": 3917400010, "product": "Fittings of plastics, for plastic tubes, pipes and hoses"}, {"hs10": 3917400090, "product": "Fittings of plastics, for plastic tubes, pipes and hoses"}, {"hs10": 6210105010, "product": "Nonwoven disposable apparel designed for use in hospitals, clinics, laboratories or contaminated areas, made up of fab of 5602/5603, n/formed or lined w paper, not k/c"}, {"hs10": 6210105090, "product": "Nonwoven disposable apparel designed for use in hospitals, clinics, laboratories or contaminated areas, made up of fab of 5602/5603, n/formed or lined w paper, not k/c"}, {"hs10": 6307906010, "product": "Surgical drapes of fabric formed on a base of paper or covered or lined with paper"}, {"hs10": 6307906090, "product": "Surgical drapes of fabric formed on a base of paper or covered or lined with paper"}, {"hs10": 6307908910, "product": "Surgical towels; cotton towels of pile/tufted const; pillow shells, of cotton; shells for quilts etc, and similar articles of cotton"}, {"hs10": 6307908940, "product": "Surgical towels; cotton towels of pile/tufted const; pillow shells, of cotton; shells for quilts etc, and similar articles of cotton"}, {"hs10": 6307908945, "product": "Surgical towels; cotton towels of pile/tufted const; pillow shells, of cotton; shells for quilts etc, and similar articles of cotton"}, {"hs10": 6307908950, "product": "Surgical towels; cotton towels of pile/tufted const; pillow shells, of cotton; shells for quilts etc, and similar articles of cotton"}, {"hs10": 6307908985, "product": "Surgical towels; cotton towels of pile/tufted const; pillow shells, of cotton; shells for quilts etc, and similar articles of cotton"}, {"hs10": 6307908995, "product": "Surgical towels; cotton towels of pile/tufted const; pillow shells, of cotton; shells for quilts etc, and similar articles of cotton"}, {"hs10": 8713900030, "product": "Carriages for disabled persons mecahnically propelled"}, {"hs10": 8713900060, "product": "Carriages for disabled persons mecahnically propelled"}, {"hs10": 9021100050, "product": "Orthopedic or fracture appliances, and parts and accessories thereof"}, {"hs10": 9021100090, "product": "Orthopedic or fracture appliances, and parts and accessories thereof"}, {"hs10": 9025198060, "product": "Clinical Thermometer, non-liquid filled"}, {"hs10": 9025198085, "product": "Clinical Thermometer, non-liquid filled"}, {"hs10": 9027504050, "product": "Electrical instruments and apparatus using optical radiations (ultraviolet, visible, infrared), nesoi"}, {"hs10": 9027905625, "product": "Parts and accessories of electrical instruments and apparatus of subheading 902720, 902730, 902750 or 902780"}, {"hs10": 9027905630, "product": "Parts and accessories of electrical instruments and apparatus of subheading 902720, 902730, 902750 or 902780"}, {"hs10": 9027905640, "product": "Parts and accessories of electrical instruments and apparatus of subheading 902720, 902730, 902750 or 902780"}, {"hs10": 9027905910, "product": "Other parts and accessories of other electrical instruments and apparatus of heading 9027, nesoi"}, {"hs10": 3006500000, "product": "First-aid boxes and kits"}, {"hs10": 3006910000, "product": "Appliances Identifiable For Ostomy Use"}, {"hs10": 3822190040, "product": "3822 containing methyl chloroform (1,1,1-trichloro- ethane) or carbon tetrachloride"}, {"hs10": 3822190010, "product": "3822 containing methyl chloroform (1,1,1-trichloro- ethane) or carbon tetrachloride"}, {"hs10": 3822190030, "product": "3822 containing methyl chloroform (1,1,1-trichloro- ethane) or carbon tetrachloride"}, {"hs10": 3822190080, "product": "3822 containing methyl chloroform (1,1,1-trichloro- ethane) or carbon tetrachloride"}, {"hs10": 3822900000, "product": "Certfied Reference Materials (As Def In Note 2, Chap 38)"}, {"hs10": 3824993900, "product": "DENTAL MATERIALS"}, {"hs10": 3926909910, "product": "Laboratory Ware"}, {"hs10": 3926909950, "product": "Other (ie, plastic parts of medical devices)"}, {"hs10": 4206001300, "product": "Articles of catgut if imported for use in the manufacture of sterile surgical sutures"}, {"hs10": 6307906800, "product": "Surgical drapes of spunlaced or bonded fiber fabric disposable surgical drapes of manâ€made fibers"}, {"hs10": 6307907200, "product": "Surgical drapes, nesoi, not spunlaced or bonded fiber fabric"}, {"hs10": 8421916000, "product": "Parts of centrifuges, including centrifugal dryers, nesoi"}, {"hs10": 8528520000, "product": "Monitors for used in or with diagnostics and medical devices"}, {"hs10": 9001300000, "product": "Contact Lenses"}, {"hs10": 9001400000, "product": "Spectacle lenses of glass"}, {"hs10": 9001500000, "product": "Spectacle lenses of mterials other than glass"}, {"hs10": 9003110000, "product": "Frames and mountings for spectacles etc, plastic"}, {"hs10": 9003190000, "product": "Frames and mountings for spectacles etc, Nesoi"}, {"hs10": 9003900000, "product": "Parts For frames and mountings, spectacles, etc"}, {"hs10": 9004100000, "product": "Sunglasses"}, {"hs10": 9011800000, "product": "Compound optical microscopes other than stereoscopic or those for microphotography, microcinematography or microprojection"}, {"hs10": 9020004000, "product": "Underwater breathing devices designed as a complete unit to be carried on the person and not requiring attendants"}, {"hs10": 9020006000, "product": "Other Breathing Appliances and Gas Masks"}, {"hs10": 9020009000, "product": "Parts and accessories of breathing appliances and gas masks, nesoi"}, {"hs10": 9021310000, "product": "Artificial Joints and Parts and Accessories Therof"}, {"hs10": 9021390000, "product": "Artificial parts of the body (other than artificial joints) and parts and accessories thereof, nesoi"}, {"hs10": 9021400000, "product": "Hearing Aids, Excluding Parts and Accessories Thereof"}, {"hs10": 9021500000, "product": "Pacemakers for stimulating heart muscles, excluding parts and accessories thereof"}, {"hs10": 9022190000, "product": "Apparatus based on the use of alpha, beta or gamma radiations, whether or not for medical, surgical, dental or veterinary uses, including radiography or radiotherapy apparatus:"}]


        user_emb = embed_with_gemini(text)      #  existing embedding function
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

        hs_result = candidates[0]
#-----------------------------------------------------------------
    except Exception as e:
        return jsonify({
            "ok": False,
            "message": f"Internal error while classifying HS10: {e}"
        }), 500
    
    confidence = float(hs_result.get("confidence"))

    followup_prompt = (
        "What would you like to know? If you tell me the sourcing country, I can tell you the latest "
        "information on trade policy and supply chain.")

    # --- 5a) Low confidence: use Gemini as fallback classifier ---
    if  confidence > LOW_CONF_THRESHOLD:
        hs_a=hs_result.get("hs10")
        hs_b=hs_result.get("product")
        return jsonify({
            "ok": True,
            "hs10": hs_a,
            "label": hs_b,
            "message": followup_prompt,
            #"message": None,  # you can also put a generic message here if you like
            }), 200
    else:    
        fallback_msg = (
            "Your description is not in the pilot version of the database, which only contains medical devices. "
            "I will let a Gen AI help you with a rough classification, though it may not be accurate. "
            "Please enter a new product description afterwards.")

        try:
            gem_cls = classify_with_gemini(text)
            hs10_guess = gem_cls.get("hs10_guess")
            label_guess = hs_result.get("label")

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

def get_tariffs_by_country(cntry: str, hs10: int) -> float | None:
    resp = (
        supabase
        .table("tariff_rate_2025_08")
        .select("col1_duty, tariff_temp_total")
        .eq("name", cntry)
        .eq("HTS22",  hs10 )
        .limit(1)
        .execute()
    )

    rows = resp.data or []
    if not rows:
        return 0  
    
    row = rows[0]
    col1_duty = row.get("col1_duty") or 0
    tariff_temp_total = row.get("tariff_temp_total") or 0

    return float(col1_duty) + float(tariff_temp_total)

def get_price_by_country(country: str, hs10: int) -> float:
    resp = (
        supabase
        .table("trade_flow_2025_07")
        .select('DUT_VAL_MO, GEN_CIF_MO, GEN_QY1_MO')
        .eq('name', country)
        .eq('HTS22', hs10)
        .limit(1)              # just one row is enough
        .execute()
    )

    rows = resp.data or []
    if not rows:
        return 0.0

    row = rows[0]
    dut_val = row.get("DUT_VAL_MO") or 0
    cif_val = row.get("GEN_CIF_MO") or 0
    qty     = row.get("GEN_QY1_MO") or 0

    if not qty or qty == 0:
        return 0.0
    total = dut_val + cif_val
    return float(total) / float(qty)

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
                       "Please enter the description of another product.",
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
            "reset_to_product": False
        }), 500

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


if __name__ == "__main__":
    # Dev server on http://localhost:5000
    app.run(host="0.0.0.0", port=5000, debug=True)




