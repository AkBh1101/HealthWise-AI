import streamlit as st
from backend_code import (
    interpret_command_with_gpt,
    process_interpreted_command,
    writeAssistantResponse,
    retrieveDataFromDatabase,
    addDataToDatabase,
    retrieveFormDataFromDatabase,
    backfill_final_disease,
    delete_all_records,
    check_disease_rules,
    extract_symptoms_from_text,
    extract_name_and_symptoms
)

# ------------------------------
# Session Initialization
# ------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_chatbot" not in st.session_state:
    st.session_state.show_chatbot = False
if "context" not in st.session_state:
    st.session_state.context = ""

# ------------------------------
# App Title
# ------------------------------
st.markdown("""
<h1 style='text-align: center; color:#38bdf8;'>💧 Smart Health Assistant</h1>
<p style='text-align: center; color:#cbd5f5; font-size:16px;'>
AI + Rule-based Water-borne Disease Detection
</p>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
.stTextInput>div>div>input,
.stTextArea textarea {
    background-color: #1e293b;
    color: white;
    border-radius: 10px;
    border: 1px solid #38bdf8;
}
.stButton>button {
    background-color: #38bdf8;
    color: black;
    border-radius: 10px;
    font-weight: bold;
}
h1, h2, h3 { color: #38bdf8; }
div[data-testid="stChatMessage"] {
    background-color: #1e293b !important;
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Mode Selection
# ------------------------------
mode = st.radio("Choose Mode:", ["Disease Form", "Chatbot", "View Predictions"])

# ------------------------------
# FORM MODE
# ------------------------------
if mode == "Disease Form":
    st.subheader("Predict Disease from Symptoms")
    with st.form("disease_form"):
        name = st.text_input("Enter your name")
        symptoms = st.text_area("Enter your symptoms")
        submitted = st.form_submit_button("Predict Disease")

        if submitted:
            if name.strip() == "" or symptoms.strip() == "":
                st.warning("⚠️ Please enter both name and symptoms.")
            else:
                prompt = f"My name is {name} and I have {symptoms}"

                try:
                    interpreted_command = interpret_command_with_gpt(prompt)
                    response = process_interpreted_command(interpreted_command)
                except Exception as e:
                    # Hard fallback: rule-based only
                    symptoms_list = extract_symptoms_from_text(symptoms)
                    matched = check_disease_rules(symptoms_list)
                    response = (
                        f"You may have: {', '.join(matched)}"
                        if matched
                        else "Could not detect disease. Please consult a doctor."
                    )
                    st.warning(f"AI unavailable, using rule-based fallback. ({e})")

                st.markdown(f"""
                    <div style='
                        padding:15px;
                        border-radius:12px;
                        background:linear-gradient(135deg, #1e293b, #0f172a);
                        color:white;
                        border:1px solid #38bdf8;
                        box-shadow: 0 0 10px rgba(56,189,248,0.3);
                    '>
                    {response}
                    </div>
                    """, unsafe_allow_html=True)

# ------------------------------
# CHAT MODE
# ------------------------------
elif mode == "Chatbot":
    st.subheader("Chat with the Health Assistant")

    if st.button("Start Chatting"):
        st.session_state.show_chatbot = True
        st.session_state.messages = []

        greeting = "Hi there! How can I assist you with your health concerns today?"
        st.session_state.context += f"Bot: {greeting}\n"
        st.session_state.messages.append({"role": "assistant", "content": greeting})

    if st.session_state.show_chatbot:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Type your symptoms or query..."):
            with st.chat_message("user"):
                st.markdown(prompt)

            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.context += f"User: {prompt}\n"
            st.session_state.context = st.session_state.context[-1000:]

            # Pass only the current message to Gemini for classification.
            # Context is stored in session state for display but NOT sent to the classifier
            # (sending the full blob caused Gemini to echo it back as the response).
            try:
                interpreted_command = interpret_command_with_gpt(prompt)
                response = process_interpreted_command(interpreted_command)
            except Exception as e:
                symptoms_list = extract_symptoms_from_text(prompt)
                matched = check_disease_rules(symptoms_list)
                response = (
                    f"You may have: {', '.join(matched)}"
                    if matched
                    else "I'm not sure about the condition. Please consult a doctor."
                )

            st.session_state.context += f"Bot: {response}\n"
            addDataToDatabase(prompt, response)
            writeAssistantResponse(response)

# ------------------------------
# ADMIN PANEL
# ------------------------------
elif mode == "View Predictions":
    st.subheader("Admin Panel - Disease Predictions 📝")

    admin_pass = st.text_input("Enter admin password", type="password")

    if admin_pass == "YOUR_PASSWORD_HERE":
        data = retrieveFormDataFromDatabase()

        if not data:
            st.info("No predictions stored yet.")
        else:
            import pandas as pd
            df = pd.DataFrame(data)

            if "time" in df.columns:
                df["time"] = pd.to_datetime(df["time"])

            cols_to_show = ["name", "symptoms", "final_disease", "prediction", "time"]
            available_cols = [c for c in cols_to_show if c in df.columns]
            st.dataframe(df[available_cols])

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download as CSV", csv, "predictions.csv", "text/csv")

            if st.button("Backfill old records with Final Disease"):
                count = backfill_final_disease()
                st.success(f"✅ Updated {count} old records.")

            if st.button("Delete ALL old records from database"):
                count = delete_all_records()
                st.success(f"🗑️ Deleted {count} records.")

    elif admin_pass:
        st.error("❌ Incorrect password.")