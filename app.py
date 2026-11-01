"""Smart Health Assistant - Flask server.
Reuses backend_code.py exactly as it is; only the frontend changed (static/index.html).
Run:  pip install flask  &&  python app.py   ->  http://localhost:5000
"""
import os, json, tempfile

# Deploy (Render/Vercel): Firebase key can come from env var FIREBASE_JSON.
# Written to a temp file BEFORE backend_code is imported (it reads the path at import).
_fj = os.getenv("FIREBASE_JSON")
if _fj:
    _p = os.path.join(tempfile.gettempdir(), "firebase-key.json")
    with open(_p, "w") as _f:
        _f.write(_fj)
    os.environ["FIREBASE_CONFIG"] = _p

import hmac
from flask import Flask, request, jsonify, send_from_directory
from backend_code import (
    interpret_command_with_gpt, process_interpreted_command, addDataToDatabase,
    retrieveFormDataFromDatabase, backfill_final_disease, delete_all_records,
    check_disease_rules, extract_symptoms_from_text, disease_info,
)

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.join(BASE, "static"))
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")  # set in .env (local) or host dashboard; never hardcode


def analyze(prompt, rules_text, none_msg):
    """Same flow as the Streamlit app: AI first, rule-based fallback on any error."""
    try:
        return process_interpreted_command(interpret_command_with_gpt(prompt)), False
    except Exception:
        matched = check_disease_rules(extract_symptoms_from_text(rules_text))
        return (f"You may have: {', '.join(matched)}" if matched else none_msg), True


def details(text):
    return [{"name": d, **info} for d, info in disease_info.items() if d in text]


def authed():
    if not ADMIN_PASSWORD:  # no password configured -> admin stays locked
        return False
    return hmac.compare_digest(request.headers.get("X-Admin-Password", ""), ADMIN_PASSWORD)


@app.get("/")
def index():
    return send_from_directory(os.path.join(BASE, "static"), "index.html")


@app.get("/api/health")
def health():
    """TEMPORARY diagnostics: open /api/health in the browser. Remove once admin works."""
    import backend_code as bc
    out = {"api_key_set": bool(bc.api_key), "admin_password_set": bool(ADMIN_PASSWORD),
           "firebase_json_env_set": bool(_fj),
           "key_file_exists": bool(bc.firebase_config and os.path.isfile(bc.firebase_config))}
    if _fj:
        try:
            json.loads(_fj); out["firebase_json_valid"] = True
        except Exception as e:
            out["firebase_json_valid"] = False; out["json_error"] = str(e)[:120]
    try:
        out["connected"] = bool(bc._init_firebase())
        if out["connected"]:
            out["disease_form_has_docs"] = len(bc.db.collection("disease_form").limit(1).get()) > 0
    except Exception as e:
        out["connected"] = False; out["error"] = str(e)[:200]
    return jsonify(out)


@app.post("/api/predict")
def predict():
    j = request.get_json(force=True)
    name, symptoms = (j.get("name") or "").strip(), (j.get("symptoms") or "").strip()
    if not name or not symptoms:
        return jsonify(error="Please enter both name and symptoms."), 400
    resp, fb = analyze(f"My name is {name} and I have {symptoms}", symptoms,
                       "Could not detect disease. Please consult a doctor.")
    return jsonify(response=resp, fallback=fb, diseases=details(resp))


@app.post("/api/chat")
def chat():
    msg = (request.get_json(force=True).get("message") or "").strip()
    if not msg:
        return jsonify(error="Empty message."), 400
    resp, fb = analyze(msg, msg, "I'm not sure about the condition. Please consult a doctor.")
    addDataToDatabase(msg, resp)
    return jsonify(response=resp, fallback=fb, diseases=details(resp))


@app.get("/api/predictions")
def predictions():
    if not authed():
        return jsonify(error="Incorrect password."), 401
    rows = retrieveFormDataFromDatabase()
    for r in rows:
        if hasattr(r.get("time"), "isoformat"):
            r["time"] = r["time"].isoformat()
    return jsonify(rows=rows[::-1])


@app.post("/api/backfill")
def backfill():
    return (jsonify(count=backfill_final_disease()) if authed()
            else (jsonify(error="Incorrect password."), 401))


@app.post("/api/delete")
def delete():
    return (jsonify(count=delete_all_records()) if authed()
            else (jsonify(error="Incorrect password."), 401))


if __name__ == "__main__":
    app.run(debug=True, port=int(os.getenv("PORT", 5000)))
