import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import auth
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import pickle
import streamlit as st
import speech_recognition as sr
import ollama
import requests
import sqlite3
import base64
from rapidfuzz import process
from auth import login, signup, logout
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit.components.v1 import html
from fpdf import FPDF


# ✅ Database Connection for Authentication
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE, password TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ✅ Set random seed for reproducibility
seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
random.seed(seed_value)

# ✅ Model Paths
MODEL_PATH = "D:/FINAL MEDICAL/frontend/lstm_model.keras"
ENCODER_PATH = "D:/FINAL MEDICAL/frontend/lstm_encoder.pkl"
SCALER_PATH = "D:/FINAL MEDICAL/frontend/scaler.pkl"

# ✅ Load Pickle Files
def load_pickle(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            return pickle.load(file)
    else:
        st.error(f"⚠️ Error: File not found - {file_path}")
        return None

# ✅ Load Model
model = load_model(MODEL_PATH, compile=False, safe_mode=False) if os.path.exists(MODEL_PATH) else None
y_encoder = load_pickle(ENCODER_PATH)
scaler = load_pickle(SCALER_PATH)

# ✅ Load CSV files
def load_csv(file_name):
    return pd.read_csv(file_name, encoding="ISO-8859-1") if os.path.exists(file_name) else pd.DataFrame()

training_df = load_csv("D:/FINAL MEDICAL/frontend/training.csv")
description_df = load_csv("D:/FINAL MEDICAL/frontend/description.csv")
precaution_df = load_csv("D:/FINAL MEDICAL/frontend/precautions.csv")
medications_df = load_csv("D:/FINAL MEDICAL/frontend/Medication.csv")
diet_df = load_csv("D:/FINAL MEDICAL/frontend/Diets.csv")
workout_df = load_csv("D:/FINAL MEDICAL/frontend/workouts.csv")
future_risk_df  = load_csv("D:/FINAL MEDICAL/frontend/Forcast Risk.csv")
first_aid = pd.read_csv("D:/FINAL MEDICAL/frontend/first_aid_instructions.csv", encoding="latin1")

# ✅ Extract symptom columns
symptom_columns = list(training_df.drop(columns=["Disease"]).columns) if not training_df.empty else []

# ✅ Initialize session state
for key in ["user_symptoms", "predicted_disease", "show_risk_button"]:
    if key not in st.session_state:
        st.session_state[key] = "" if key == "user_symptoms" else None

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "user_symptoms" not in st.session_state:
    st.session_state.user_symptoms = []
if "show_future_risk" not in st.session_state:
    st.session_state.show_future_risk = False    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "show_chat_history" not in st.session_state:
    st.session_state.show_chat_history = False 
if "show_first_aid" not in st.session_state:
    st.session_state.show_first_aid = False    

# ✅ Authentication UI
def show_auth_page():
    st.title("🔐 Welcome to AI-Driven Disease Diagnosis System")
    menu = ["Sign Up", "Login"]
    choice = st.sidebar.selectbox("Select", menu)
    
    if choice == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            authenticated = login(username, password)
            if authenticated:
                st.session_state["authenticated"] = True
                st.session_state["username"] = username
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    elif choice == "Sign Up":
        # 🛠 FIXED INDENTATION
        username = st.text_input("Username")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Sign Up"):
            success, message = signup(username, email, password)
            if success:
                st.success(message)  # ✅ User should log in manually
            else:
                st.error(message)

# ✅ Show login/signup page if not authenticated
if not st.session_state["authenticated"]:
    show_auth_page()
    st.stop()

# ✅ Show Logout Button in Sidebar when logged in
st.sidebar.write(f"👤 Logged in as: {st.session_state['username']}")
if st.sidebar.button("Logout"):
    logout()

# Initialize session state variables if they don't exist
session_defaults = {
    "desc": "",  # Initialize desc as an empty string
    "pre": [],
    "med": [],
    "die": [],
    "wrkout": [],
    
}
for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ✅ Encode Symptoms
def encode_symptoms(user_symptoms):
    if model is None or scaler is None:
        return None, ["Error: Model or scaler not loaded"]
    
    X_input = np.zeros((1, len(symptom_columns)), dtype=np.float32)
    not_found = []

    for symptom in user_symptoms:
        symptom = symptom.lower().strip()
        if symptom in symptom_columns:
            X_input[0, symptom_columns.index(symptom)] = 1
        else:
            not_found.append(symptom)

    X_input = scaler.transform(X_input)
    X_input = np.expand_dims(X_input, axis=0)
    return X_input, not_found

# ✅ Predict Disease
def get_predicted_disease(user_symptoms):
    X_input, not_found = encode_symptoms(user_symptoms)
    if X_input is None:
        return None, None, not_found
    
    y_pred = model.predict(X_input)
    predicted_class = np.argmax(y_pred)
    confidence_score = y_pred[0][predicted_class]
    
    predicted_disease = y_encoder.inverse_transform([predicted_class])[0] if y_encoder else "Error: Encoder not loaded"
    return predicted_disease, confidence_score, not_found

# ✅ Speech Recognition (Fixed)
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening... Speak now!")
        try:
            audio = recognizer.listen(source, timeout=5)
            recognized_text = recognizer.recognize_google(audio)
            
            # ✅ Convert space-separated words to a list
            spoken_symptoms = [s.strip().lower() for s in recognized_text.split()]
            
            # ✅ Store formatted symptoms in session state
            st.session_state.user_symptoms.extend(spoken_symptoms)
            st.success(f"✅ Recognized Symptoms: {', '.join(spoken_symptoms)}")

        except sr.UnknownValueError:
            st.error("❌ Could not understand speech.")
        except sr.RequestError:
            st.error("❌ Could not process request. Check internet connection.")

# ✅ Fetch Disease Details
def fetch_disease_details(disease):
    def get_value(df, column):
        values = df.loc[df['Disease'] == disease, column].values
        return values[0] if len(values) > 0 else f"No {column} available"

    description = get_value(description_df, 'Description')
    precautions = precaution_df.loc[precaution_df['Disease'] == disease].values[:, 1:].flatten().tolist()
    
    medications = medications_df.loc[medications_df['Disease'] == disease, 'Medications']
    medications = [med.strip(" '") for med in medications.iloc[0].strip("[]").split(",")] if not medications.empty else ["No medications available"]

    diet = diet_df.loc[diet_df['Disease'] == disease, 'Diet'].tolist()
    workout = workout_df.loc[workout_df['Disease'] == disease, 'workout'].tolist()

    return description, precautions, medications, diet, workout

# ✅ Function to create styled cards
def display_card(title, icon, items):
    st.markdown(
        f"""
        <div style="border: 2px solid #444; padding: 15px; border-radius: 10px; background-color: #222; margin-bottom: 10px;">
            <h3 style="color: #fff;">{icon} {title}</h3>
            <ul style="color: #ddd; font-size: 16px;">
                {''.join(f'<li>{item}</li>' for item in items)}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ✅ Function to display disease description in a styled card
def display_description_card(title, description):
    st.markdown(
        f"""
        <div style="border: 2px solid #444; padding: 15px; border-radius: 10px; background-color: #333; margin-bottom: 10px;">
            <h3 style="color: #fff;">📖 {title}</h3>
            <p style="color: #ddd; font-size: 16px;">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ✅ Fetch Disease Details
def fetch_disease_details(disease):
    def get_value(df, column):
        values = df.loc[df['Disease'] == disease, column].values
        return values[0] if len(values) > 0 else f"No {column} available"

    description = get_value(description_df, 'Description')
    precautions = precaution_df.loc[precaution_df['Disease'] == disease].values[:, 1:].flatten().tolist()
    medications = medications_df.loc[medications_df['Disease'] == disease, 'Medications']
    medications = [med.strip(" '") for med in medications.iloc[0].strip("[]").split(",")] if not medications.empty else ["No medications available"]
    diet = diet_df.loc[diet_df['Disease'] == disease, 'Diet'].tolist()
    workout = workout_df.loc[workout_df['Disease'] == disease, 'workout'].tolist()
    return description, precautions, medications, diet, workout

# ✅ Fetch Future Risk
def fetch_future_risk(disease):
    risk_data = future_risk_df[future_risk_df['Disease'] == disease]
    if risk_data.empty:
        return None, None, "No future risk data available"
    return risk_data['Disease Risk Score'].values[0], risk_data['Symptoms Severity Score'].values[0], risk_data['Future Risks'].values[0]

# ✅ Display Future Risk with Stacked Bar Chart
def display_forecast_chart(disease):
    future_risk, future_severity, risk_name = fetch_future_risk(disease)
    if future_risk is None:
        st.warning("No future risk data available for this disease.")
        return
    
    def get_risk_category(score):
        if score < 4:
            return "Low Risk", "#4CAF50"
        elif score < 7:
            return "Medium Risk", "#1E90FF"
        else:
            return "High Risk", "#F44336"

    risk_label, risk_color = get_risk_category(future_risk)
    severity_label, severity_color = get_risk_category(future_severity)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar("Disease Risk Score", future_risk, color=risk_color, label=f"Risk: {risk_label}")
    ax.bar("Symptoms Severity Score", future_severity, color=severity_color, label=f"Severity: {severity_label}")

    plt.ylabel("Score")
    plt.title(f"Future Risk Forecast for {disease}")
    plt.ylim(0, 10)
    plt.legend()
    
    st.pyplot(fig)
    st.info(f"⚠️ **Future Risk Name:** {risk_name}")
    
# ✅ Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load unique symptoms list from CSV
symptoms_df = pd.read_csv("D:/FINAL MEDICAL/frontend/Unique_symptoms.csv")  # Ensure file exists
all_symptoms = symptoms_df["symptoms"].tolist()  # Convert column to a list

# ✅ Streamlit UI
st.title("🩺 AI-Driven Disease Diagnosis System")

# Select symptoms using dropdown (predefined list)
selected_symptoms = st.multiselect(
    "Select symptoms",
    options=all_symptoms,  
    default=[],
    placeholder="Start typing or select symptoms...",
)

# ✅ Manual input for symptoms
manual_input = st.text_input("Or type symptoms manually (comma-separated)")
manual_symptoms = [s.strip() for s in manual_input.split(",") if s.strip()]

# ✅ Speech Recognition Button
if st.button("🎙️ Speak Symptoms"):
    recognized_text = recognize_speech()  # Assume this function returns symptoms as text
    if recognized_text:
        manual_symptoms.extend([s.strip() for s in recognized_text.split(",") if s.strip()])

# Merge dropdown + manual input and remove duplicates
final_symptoms = list(set(selected_symptoms + manual_symptoms))

# ✅ Predict Button
if st.button("Predict"):
    if final_symptoms:  
        try:
            symptoms_list = [s.strip().lower() for s in final_symptoms]
            predicted_disease, confidence_score, not_found = get_predicted_disease(symptoms_list)

            if predicted_disease:
                # ✅ Store results in session state to persist across reruns
                st.session_state.predicted_disease = predicted_disease  
                st.session_state.confidence_score = confidence_score
                st.session_state.not_found = not_found
                st.session_state.desc, st.session_state.pre, st.session_state.med, st.session_state.die, st.session_state.wrkout = fetch_disease_details(predicted_disease)
                st.session_state.show_risk_button = True  # ✅ Enable Future Risk button

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
    else:
        st.warning("⚠️ Please enter symptoms before predicting.")

# ✅ Display Prediction Results and Recommendations Persistently
if st.session_state.predicted_disease:
    st.success(f"✅ **Predicted Disease:** {st.session_state.predicted_disease}")

    # ✅ Show Description Card
    display_description_card("Disease Description", st.session_state.desc)

    # ✅ Display other details in two columns
    col1, col2 = st.columns(2)
    with col1:
        display_card("Precautions", "🛡️", st.session_state.pre)
        display_card("Diet Recommendations", "🥗", st.session_state.die)
    with col2:
        display_card("Medications", "💊", st.session_state.med)
        display_card("Workout Recommendations", "🏋️", st.session_state.wrkout)

# First Aid Button
predicted_disease = st.session_state.predicted_disease
if predicted_disease in first_aid['Disease'].values:
    if st.button('🆘 Show First Aid Instructions'):
        st.session_state.show_first_aid = True

# ✅ Display First Aid Instructions with formatted numbering
if st.session_state.show_first_aid:
    first_aid_text = first_aid[first_aid['Disease'] == predicted_disease]['First Aid Instructions'].values[0]
    
    # Convert plain text into a numbered list with emojis
    formatted_text = ""
    points = first_aid_text.split("\n")  # Split by new line if exists
    emoji_numbers = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]  # Emoji numbers for styling

    for i, point in enumerate(points):
        if i < len(emoji_numbers):
            formatted_text += f"{emoji_numbers[i]} {point}\n\n"
        else:
            formatted_text += f"➡️ {point}\n\n"  # Use arrow if more than 5 points

    st.warning(f'**🚑 Emergency First Aid:**\n\n{formatted_text}')

if st.session_state.show_risk_button:
    if st.button("📊 Future Risk"):
        st.session_state.show_future_risk = True  # ✅ Preserve Future Risk visibility

# ✅ Display Future Risk Chart persistently
if st.session_state.show_future_risk:
    display_forecast_chart(st.session_state.predicted_disease)
    # ✅ Save the displayed chart
    chart_filename = "risk_chart.png"
    plt.savefig(chart_filename)  # Save the currently displayed figure
    plt.close()

# ✅ Chatbot UI
st.markdown("### 💬 Chat with AI")

user_message = st.text_input("Type your message:")
if st.button("Send"):
    response = ollama.chat(model="gemma:2b", messages=[{"role": "user", "content": user_message}])
    bot_reply = response['message']['content'] if 'message' in response else "⚠️ Unable to respond."
    
    # ✅ Store the latest chatbot response in session state
    st.session_state.latest_chat = {"user": user_message, "bot": bot_reply}
    
    # ✅ Append to chat history for later viewing
    st.session_state.chat_history.append(st.session_state.latest_chat)

# ✅ Only show chatbot response if a message is sent (but not history)
if "latest_chat" in st.session_state:
    st.markdown(f"🤖 : {st.session_state.latest_chat['bot']}")

# ✅ Toggle chat history visibility
if st.button("👀 View Chat History"):
    st.session_state.show_chat_history = not st.session_state.show_chat_history

# ✅ Display full chat history when enabled
if st.session_state.show_chat_history:
    for msg in st.session_state.chat_history:
        st.markdown(f"You : {msg['user']}")
        st.markdown(f"🤖 : {msg['bot']}")

from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", style="B", size=16)
        self.cell(200, 10, "AI-Driven Disease Diagnosis Report", ln=True, align="C")
        self.ln(4)  # Small space after the title
        
    def section_title(self, title):
        self.ln(5)  # **Adds space before the title**
        self.set_font("Arial", style="B", size=13)
        self.cell(0, 6, title, ln=True)
        self.ln(1)  # **Only 1 line space after the title**

    def section_content(self, content):
        if content is None:
            content = ""
        self.set_font("Arial", size=11)
        self.multi_cell(0, 5, str(content).replace("•", "-"))  # **Reduced line height**
        self.ln(1)  # **Smaller space after content**

pdf = PDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()

# Predicted Disease
pdf.section_title("Predicted Disease:")
pdf.section_content(st.session_state.get("predicted_disease", "N/A"))  # Handle missing data

# Description
pdf.section_title("Description:")
pdf.section_content(st.session_state.get("desc", "No description available."))

# Precautions
pdf.section_title("Precautions:")
pdf.section_content("\n- " + "\n- ".join(st.session_state.get("pre", [])))  # Handle list safely

# Medications
pdf.section_title("Medications:")
pdf.section_content("\n- " + "\n- ".join(st.session_state.get("med", [])))

# Diet Recommendations
pdf.section_title("Diet Recommendations:")
pdf.section_content("\n- " + "\n- ".join(st.session_state.get("die", [])))

# Workout Recommendations
pdf.section_title("Workout Recommendations:")
pdf.section_content("\n- " + "\n- ".join(st.session_state.get("wrkout", [])))

# ✅ First Aid Instructions (if available)
if st.session_state.get("predicted_disease") in first_aid['Disease'].values:
    first_aid_text = first_aid[first_aid['Disease'] == st.session_state.predicted_disease]['First Aid Instructions'].values[0]
    
    pdf.section_title("First Aid Instructions:")

    if first_aid_text:  # Check if data exists
        points = first_aid_text.split("\n")  # Split by line

        pdf.set_font("Arial", size=11)
        for i, point in enumerate(points, start=1):
            pdf.multi_cell(0, 8, f"{i}. {point}")  # Numbered list formatting
        pdf.ln(5)
    else:
        pdf.section_content("No first aid instructions available.")

# ✅ Future Risk Analysis (if applicable)
if st.session_state.get("show_future_risk", False):
    pdf.section_title("Future Risk Analysis:")
    future_risk, future_severity, risk_name = fetch_future_risk(st.session_state.get("predicted_disease", ""))
    pdf.section_content(f"Future Risk Score: {future_risk}\nSymptoms Severity Score: {future_severity}\nFuture Risk Name: {risk_name}")
    pdf.image("risk_chart.png", x=10, y=None, w=180)

# ✅ Save and Download the PDF
pdf_file = "diagnosis_report.pdf"
pdf.output(pdf_file)

with open(pdf_file, "rb") as f:
    st.download_button("📥 Download Report", f, file_name=pdf_file, mime="application/pdf")

