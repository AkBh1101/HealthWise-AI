import os
import google.generativeai as genai
import json
import datetime
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, firestore
import streamlit as st
import time
import re

# ----------------------------
# Firebase + Environment Setup
# ----------------------------
# Resolve .env relative to this file so it works regardless of where you run streamlit from
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(_BASE_DIR, ".env"))

api_key = os.getenv("API_KEY")
_firebase_config_raw = os.getenv("FIREBASE_CONFIG")  # e.g. firebase_config/key.json

# Resolve to absolute path (handles both relative and already-absolute paths)
firebase_config = (
    os.path.join(_BASE_DIR, _firebase_config_raw)
    if _firebase_config_raw and not os.path.isabs(_firebase_config_raw)
    else _firebase_config_raw
)

# ── Gemini setup ──────────────────────────────────────────────────────────────
if api_key:
    genai.configure(api_key=api_key)
else:
    st.warning("⚠️ API_KEY is missing in .env — AI features will use fallback mode.")

# ── Firebase setup (safe, won't crash on import) ──────────────────────────────
db = None

def _init_firebase():
    """Initialize Firebase once. Returns True on success, False on failure."""
    global db
    if db is not None:
        return True

    if firebase_admin._apps:
        db = firestore.client()
        return True

    if not firebase_config:
        return False  # No config provided — DB features disabled

    try:
        # firebase_config must be the PATH to the service-account JSON file.
        # e.g.  FIREBASE_CONFIG=./serviceAccountKey.json
        if not os.path.isfile(firebase_config):
            st.error(
                f"FIREBASE_CONFIG points to '{firebase_config}' but that file was not found. "
                "Check your .env file."
            )
            return False

        cred = credentials.Certificate(firebase_config)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        return True

    except Exception as e:
        st.error(f"Firebase initialization failed: {e}")
        return False


# ----------------------------
# Database Helpers
# ----------------------------
def addDataToDatabase(prompt, response):
    if not _init_firebase():
        return  # Silently skip if Firebase is unavailable
    data = {"user": prompt, "bot": response, "time": datetime.datetime.now()}
    db.collection("chat").add(data)


def addFormDataToDatabase(name, symptoms, prediction, final_disease=None):
    if not _init_firebase():
        return
    data = {
        "name": name,
        "symptoms": symptoms,
        "prediction": prediction,
        "final_disease": final_disease if final_disease else "Not confirmed",
        "time": datetime.datetime.now()
    }
    db.collection("disease_form").add(data)


def retrieveDataFromDatabase():
    if not _init_firebase():
        return []
    history = db.collection("chat").order_by("time").get()
    return [chat.to_dict() for chat in history]


def retrieveFormDataFromDatabase():
    if not _init_firebase():
        return []
    history = db.collection("disease_form").order_by("time").get()
    return [entry.to_dict() for entry in history]


# ----------------------------
# Rule-based Disease Detection
# ----------------------------
def check_disease_rules(symptoms_list):
    disease_rules = {
        "Cholera": ["diarrhea", "watery stool", "vomiting", "dehydration", "sunken eyes", "dry mouth", "cramps"],
        "Typhoid": ["fever", "headache", "abdominal pain", "constipation", "diarrhea"],
        "Dysentery": ["bloody diarrhea", "abdominal cramps", "nausea", "vomiting"],
        "Hepatitis A/E": ["jaundice", "yellow eyes", "dark urine", "fatigue", "loss of appetite"]
    }

    matched = []
    for disease, indicators in disease_rules.items():
        for s in symptoms_list:
            if any(keyword in s.lower() for keyword in indicators):
                matched.append(disease)

    return list(set(matched))


# ----------------------------
# Disease Info
# ----------------------------
disease_info = {
    "Cholera": {
        "precautions": "Drink clean water, maintain hygiene, avoid contaminated food.",
        "medicines": "ORS, antibiotics (only if prescribed), rehydration therapy."
    },
    "Typhoid": {
        "precautions": "Avoid outside food, drink boiled water, maintain sanitation.",
        "medicines": "Antibiotics (doctor prescribed), paracetamol for fever."
    },
    "Dysentery": {
        "precautions": "Wash hands, avoid contaminated food/water.",
        "medicines": "ORS, antibiotics (if severe), anti-diarrheal meds."
    },
    "Hepatitis A/E": {
        "precautions": "Avoid oily food, maintain hygiene, drink safe water.",
        "medicines": "Rest, hydration, liver-friendly diet."
    }
}


# ----------------------------
# Utility Functions
# ----------------------------
def extract_json(text):
    """Extract the first JSON object found in a string."""
    match = re.search(r'\{.*\}', text, re.DOTALL)
    return match.group(0) if match else "{}"


def extract_symptoms_from_text(text):
    text = text.lower()

    phrases = ["i have", "i am having", "i feel", "since", "from"]
    for p in phrases:
        text = text.replace(p, "")

    text = text.replace(" and ", ",")
    text = text.replace(" with ", ",")

    parts = re.split(r'[,.]', text)

    cleaned = []
    for p in parts:
        p = p.strip()
        if len(p) > 2:
            cleaned.append(p)

    return cleaned


def extract_name_and_symptoms(text):
    text = text.strip()

    name_match = re.search(r"my name is (\w+)", text.lower())
    if name_match:
        name = name_match.group(1).capitalize()
    else:
        name = "User"

    cleaned_text = re.sub(r"my name is \w+", "", text.lower())
    symptoms_list = extract_symptoms_from_text(cleaned_text)

    return name, ", ".join(symptoms_list)


# ----------------------------
# AI + Processing Functions
# ----------------------------
def interpret_command_with_gpt(command):
    """
    Ask Gemini to classify the user's intent and return structured JSON.
    'command' must be ONLY the current user message (not a context blob).
    Falls back to rule-based disease prediction if AI is unavailable.
    """
    if not api_key:
        name, symptoms = extract_name_and_symptoms(command)
        return json.dumps({
            "kind": "disease_prediction",
            "parameters": {"name": name, "symptoms": symptoms}
        })

    # Safety: strip context prefix if caller accidentally passes the full blob
    if "current message:" in command.lower():
        command = re.split(r"current message:", command, flags=re.IGNORECASE)[-1].strip()

    now_date = datetime.datetime.now().strftime('%Y-%m-%d')
    now_time = datetime.datetime.now().strftime('%H:%M')

    prompt = (
        "You are a medical assistant AI. Classify the user message and return ONLY valid JSON "
        "with no markdown, no extra text, no explanation.\n\n"
        "Return format:\n"
        '{{"kind": "<type>", "parameters": {{...}}}}\n\n'
        "Kinds:\n"
        "- \"disease_prediction\": use whenever the user mentions ANY health symptoms (fever, diarrhea, jaundice, vomiting, pain, etc.)\n"
        "  parameters: {\"name\": \"<name if mentioned, else User>\", \"symptoms\": \"<comma-separated symptoms>\"}\n"
        "- \"general_chat\": use for greetings, general questions, anything NOT symptom-related\n"
        "  parameters: {\"response\": \"<short helpful reply>\"}\n"
        "- \"mathematics\": use for math calculations\n"
        "  parameters: {\"result\": \"<answer>\"}\n"
        f"- \"date_time\": use for date/time questions. Today is {now_date}, time is {now_time}\n"
        f"  parameters: {{\"response\": \"<answer>\"}}\n\n"
        "RULE: If ANY symptom words appear, always use disease_prediction.\n\n"
        f"User message: {command}"
    )

    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        raw = response.text.strip()
        cleaned = extract_json(raw)
        json.loads(cleaned)  # validate
        return cleaned

    except Exception:
        name, symptoms = extract_name_and_symptoms(command)
        return json.dumps({
            "kind": "disease_prediction",
            "parameters": {"name": name, "symptoms": symptoms}
        })


def process_interpreted_command(interpreted_command):
    try:
        command_data = json.loads(interpreted_command)
    except Exception:
        return "Sorry, I couldn't process that request. Please try again."

    kind = command_data.get("kind", "")
    parameters = command_data.get("parameters", {})

    if kind == "general_chat":
        reply = parameters.get("response", "").strip()
        # If Gemini returned an empty response field, ask it directly
        if not reply:
            reply = _ask_gemini_directly(
                interpreted_command,
                fallback="I'm here to help! Could you describe your symptoms?"
            )
        return reply

    elif kind == "mathematics":
        return parameters.get("result", "Sorry, I couldn't evaluate that expression.")

    elif kind == "date_time":
        reply = parameters.get("response", "").strip()
        if not reply:
            current_date = datetime.datetime.now().strftime("%Y-%m-%d")
            current_time = datetime.datetime.now().strftime("%H:%M")
            reply = f"Today's date is {current_date} and the current time is {current_time}."
        return reply

    elif kind == "disease_prediction":
        name = parameters.get("name", "User")
        symptoms = parameters.get("symptoms", "")

        if not symptoms.strip():
            return f"Hi {name}, I didn't catch any symptoms. Could you describe what you're feeling?"

        symptoms_list = extract_symptoms_from_text(symptoms)
        matched_diseases = check_disease_rules(symptoms_list)

        if matched_diseases:
            final_disease = ", ".join(matched_diseases)

            details = ""
            for d in matched_diseases:
                info = disease_info.get(d, {})
                precautions = info.get("precautions", "N/A")
                medicines = info.get("medicines", "N/A")
                details += f"\n\n🔹 **{d}**\n- Precautions: {precautions}\n- Medicines: {medicines}"

            prediction = f"You may have: **{final_disease}**.{details}\n\n⚠️ Please consult a doctor."
        else:
            final_disease = None
            prediction = "Symptoms are unclear or don't match known water-borne diseases. Please consult a doctor."

        addFormDataToDatabase(name, symptoms, prediction, final_disease)

        return f"Hi {name}, based on your symptoms ({symptoms}):\n\n{prediction}"

    elif kind == "generate_code":
        return parameters.get("code", "No code was generated.")

    else:
        return "I couldn't quite understand that. Could you describe your symptoms or ask a health-related question?"


def _ask_gemini_directly(original_input, fallback="I'm here to help!"):
    """
    Fallback: ask Gemini for a plain text reply when structured JSON had empty content.
    """
    if not api_key:
        return fallback
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(
            f"You are a helpful health assistant. Reply concisely to: {original_input}"
        )
        return response.text.strip() or fallback
    except Exception:
        return fallback


# ----------------------------
# UI Response
# ----------------------------
def writeAssistantResponse(response):
    with st.chat_message("assistant"):
        placeholder = st.empty()
        typed_message = ""

        for char in response:
            typed_message += char
            placeholder.markdown(typed_message)
            time.sleep(0.02)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })


def delete_all_records():
    if not _init_firebase():
        return 0
    docs = db.collection("disease_form").get()
    deleted_count = 0
    for doc in docs:
        db.collection("disease_form").document(doc.id).delete()
        deleted_count += 1
    return deleted_count


def backfill_final_disease():
    """Update old records that are missing the final_disease field."""
    if not _init_firebase():
        return 0
    docs = db.collection("disease_form").get()
    updated_count = 0

    for doc in docs:
        data = doc.to_dict()
        if "final_disease" not in data:
            symptoms_list = extract_symptoms_from_text(data.get("symptoms", ""))
            matched = check_disease_rules(symptoms_list)
            final_disease = ", ".join(matched) if matched else "Not confirmed"

            db.collection("disease_form").document(doc.id).update({
                "final_disease": final_disease
            })
            updated_count += 1

    return updated_count
