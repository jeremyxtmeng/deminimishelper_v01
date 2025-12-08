# app.py
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# Allow the frontend (on 5500) to call this API (on 5000)
CORS(
    app,
    resources={r"/api/*": {"origins": "*"}},
    supports_credentials=False,
)

@app.route("/api/medical-classify", methods=["POST"])
def medical_classify():
    # Just echo back for now so we can debug
    data = request.get_json(force=True)
    text = (data.get("prompt") or "").strip()
    print("Received POST /api/medical-classify with:", text)

    if not text:
        return jsonify({"ok": False, "message": "Prompt is empty."}), 400

    return jsonify({
        "ok": True,
        "hs10": "TEST123456",
        "label": "Test device classification",
        "extra": {"echo": text},
        "validator_reason": "Temporary echo backend (no Gemini yet).",
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)