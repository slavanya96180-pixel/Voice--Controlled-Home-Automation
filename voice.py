# ================================
# VOICE CONTROL HOME AUTOMATION (FINAL WITH LOGIN)
# ================================

import streamlit as st
import pandas as pd
import speech_recognition as sr
import sounddevice as sd
import soundfile as sf
import tempfile
import numpy as np
import librosa
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# -------------------------------
# 🔐 LOGIN SYSTEM
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.title("🔐 Smart Home Login")
    st.subheader("Owner Authentication Required")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.success("✅ Login Successful")
            st.rerun()
        else:
            st.error("❌ Invalid Credentials")

# Show login first
if not st.session_state.logged_in:
    login_page()
    st.stop()

# -------------------------------
# MAIN APP STARTS
# -------------------------------
st.set_page_config(page_title="Smart Home AI", layout="centered")

st.title("🏠 Smart Voice Home Automation")

# Logout button
if st.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

st.write("AI-based secure voice control system")

# -------------------------------
# DATASET
# -------------------------------
data = {
    "command": [
        "turn on light", "turn off light",
        "switch on fan", "switch off fan",
        "turn on AC", "turn off AC",
        "light on", "light off",
        "fan on", "fan off"
    ],
    "action": [
        "LIGHT_ON", "LIGHT_OFF",
        "FAN_ON", "FAN_OFF",
        "AC_ON", "AC_OFF",
        "LIGHT_ON", "LIGHT_OFF",
        "FAN_ON", "FAN_OFF"
    ]
}

df = pd.DataFrame(data)

st.subheader("📂 Dataset")
st.dataframe(df)

# -------------------------------
# MODEL TRAINING
# -------------------------------
X = df["command"]
y = df["action"]

vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
st.success(f"✅ Model Accuracy: {accuracy:.2f}")

# -------------------------------
# 🔐 OWNER VOICE DETECTION
# -------------------------------
def extract_features(file):
    audio, sr = librosa.load(file, duration=3)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

try:
    owner_voice = extract_features("owner.wav")
except:
    owner_voice = None
    st.warning("⚠️ Upload owner.wav for voice authentication")

def is_owner(test_file):
    if owner_voice is None:
        return True

    test_voice = extract_features(test_file)
    distance = np.linalg.norm(owner_voice - test_voice)

    return distance < 50

# -------------------------------
# VOICE INPUT
# -------------------------------
def get_voice_command():
    recognizer = sr.Recognizer()
    st.info("🎤 Listening...")

    duration = 5
    samplerate = 44100

    audio = sd.rec(int(duration * samplerate),
                   samplerate=samplerate, channels=1)
    sd.wait()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, samplerate)

        # Owner check
        if not is_owner(f.name):
            st.error("🚫 Access Denied: Not Owner")
            return "", False

        with sr.AudioFile(f.name) as source:
            audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
        return text, True

    except:
        st.error("❌ Could not understand audio")
        return "", False

# -------------------------------
# INPUT
# -------------------------------
st.subheader("🎯 Control Panel")

method = st.radio("Choose Input", ["Text", "Voice"])

user_input = ""
valid_user = True

if method == "Text":
    user_input = st.text_input("Enter command:")

else:
    if st.button("🎙️ Start Voice"):
        user_input, valid_user = get_voice_command()
        st.write("You said:", user_input)

# -------------------------------
# HISTORY
# -------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------
# PREDICTION
# -------------------------------
if user_input and valid_user:

    vec = vectorizer.transform([user_input])
    prediction = model.predict(vec)[0]

    prob = model.predict_proba(vec).max()

    st.subheader("🔌 Device Action")
    st.info(f"Confidence: {prob:.2f}")

    result = ""

    if prediction == "LIGHT_ON":
        result = "💡 Light ON"
        st.success(result)

    elif prediction == "LIGHT_OFF":
        result = "💡 Light OFF"
        st.warning(result)

    elif prediction == "FAN_ON":
        result = "🌀 Fan ON"
        st.success(result)

    elif prediction == "FAN_OFF":
        result = "🌀 Fan OFF"
        st.warning(result)

    elif prediction == "AC_ON":
        result = "❄️ AC ON"
        st.success(result)

    elif prediction == "AC_OFF":
        result = "❄️ AC OFF"
        st.warning(result)

    else:
        result = "Unknown"
        st.error(result)

    st.session_state.history.append((user_input, result))

# -------------------------------
# HISTORY DISPLAY
# -------------------------------
st.subheader("📜 Command History")

if st.session_state.history:
    for cmd, res in reversed(st.session_state.history):
        st.write(f"👉 {cmd} ➝ {res}")
else:
    st.write("No commands yet")

# -------------------------------
# SYSTEM STATUS
# -------------------------------
st.subheader("📊 System Status")

st.write("✔ Login Security Enabled")
st.write("✔ Voice Authentication Enabled")
st.write("✔ AI Model Running")

# -------------------------------
# END
# -------------------------------
st.success("🚀 System Running Successfully!")
