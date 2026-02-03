import pandas as pd
import os
import joblib
import streamlit as st
import plotly.graph_objects as go
from collections import Counter
import base64

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from langdetect import detect
from googletrans import Translator

# ---------------------- Page Config ----------------------
st.set_page_config(
    page_title="📧 Email Spam Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

translator = Translator()

# ---------------------- Background Wallpaper ----------------------
def set_bg_image(image_file):
    if os.path.exists(image_file):
        with open(image_file, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

set_bg_image("background.jpg")

# ---------------------- PREMIUM UI STYLING ----------------------
st.markdown("""
<style>

/* ---------- Sidebar Glass ---------- */
[data-testid="stSidebar"] {
    background: rgba(0, 0, 0, 0.65);
    backdrop-filter: blur(12px);
}

[data-testid="stSidebar"] * {
    color: white;
}

/* ---------- Main Glass Container ---------- */
.main-box {
    background: rgba(0, 0, 0, 0.50);
    backdrop-filter: blur(14px);
    padding: 30px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0px 8px 32px rgba(0,0,0,0.35);
}

/* ---------- Textarea Premium ---------- */
textarea {
    background: rgba(0,0,0,0.55) !important;
    color: white !important;
    border-radius: 12px !important;
}

/* ---------- File Uploader ---------- */
[data-testid="stFileUploader"] {
    background: rgba(0,0,0,0.55);
    border-radius: 12px;
    padding: 10px;
}

/* ---------- Button Premium ---------- */
.stButton>button {
    background: linear-gradient(45deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 30px;
    height: 50px;
    font-size: 16px;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0px 0px 15px #00c6ff;
}

/* ---------- Labels ---------- */
.spam { color: #ff4b4b; font-size: 28px; font-weight: bold; }
.ham { color: #2ecc71; font-size: 28px; font-weight: bold; }

</style>
""", unsafe_allow_html=True)

# ---------------------- File Paths ----------------------
csv_path = r"C:\Users\mukesh\email_spam_detection\spam.csv"
model_path = r"C:\Users\mukesh\email_spam_detection\spam_model.pkl"
vectorizer_path = r"C:\Users\mukesh\email_spam_detection\vectorizer.pkl"

# ---------------------- Load or Train Model ----------------------
@st.cache_resource
def load_model():

    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer, None

    data = pd.read_csv(csv_path, encoding="latin-1")
    X = data["text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_vec))

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    return model, vectorizer, accuracy

model, vectorizer, accuracy = load_model()

# ---------------------- Explainable AI ----------------------
def explain_prediction(text, model, vectorizer, top_n=8):

    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    tfidf_vector = vectorizer.transform([text])
    indices = tfidf_vector.nonzero()[1]

    word_scores = []

    for idx in indices:
        word = feature_names[idx]
        score = coefficients[idx] * tfidf_vector[0, idx]
        word_scores.append((word, score))

    word_scores = sorted(word_scores, key=lambda x: abs(x[1]), reverse=True)
    return word_scores[:top_n]

# ---------------------- Session State ----------------------
if 'email_history' not in st.session_state:
    st.session_state.email_history = []

# ---------------------- Sidebar ----------------------
st.sidebar.title("📊 Project Info")

if accuracy:
    st.sidebar.success(f"Model Accuracy: {accuracy:.2%}")
else:
    st.sidebar.info("Using pre-trained model")

st.sidebar.markdown("""
### ✉ Description
Multilingual Spam Detection using Logistic Regression & TF-IDF.

### 🚀 Features
• Any language support  
• Spam probability gauge  
• Explainable AI  
• File upload + manual input  
• Admin analytics dashboard  
""")

# -------- Sidebar Prediction Stats --------
st.sidebar.subheader("📈 Prediction Stats")

spam_count = sum(1 for e in st.session_state.email_history if e['prediction']=="SPAM")
ham_count = sum(1 for e in st.session_state.email_history if e['prediction']=="HAM")

st.sidebar.write(f"🚨 SPAM: {spam_count}")
st.sidebar.write(f"✅ HAM: {ham_count}")

# ---------------------- Navigation ----------------------
page = st.sidebar.radio("Navigate", ["Spam Detection", "Admin Dashboard"])

# ---------------------- Spam Detection ----------------------
if page == "Spam Detection":

    st.title("📧 Email Spam Detection System")
    st.caption("Premium ML Powered Spam Classifier")

    if accuracy:
        st.success(f"Model trained successfully (Accuracy: {accuracy:.2%})")
    else:
        st.info("Loaded trained model")

    st.markdown("<div class='main-box'>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Email (.txt)", type=["txt"])

    file_text = ""
    if uploaded_file:
        file_text = uploaded_file.read().decode("utf-8")
        st.success("File Loaded Successfully")

    email_input = st.text_area("Enter Email Content", height=180, value=file_text)

    if st.button("🔍 Detect Spam"):

        if email_input.strip():

            detected_lang = detect(email_input)
            translated_text = email_input

            if detected_lang != "en":
                translated_text = translator.translate(email_input, src=detected_lang, dest="en").text

                # -------- Display Translation --------
                st.subheader("🌐 Translated Content (English)")
                st.info(translated_text)

            email_vec = vectorizer.transform([translated_text])
            prediction = model.predict(email_vec)[0]
            probability = model.predict_proba(email_vec)[0][1]

            label = "SPAM 🚨" if prediction else "HAM ✅"
            css = "spam" if prediction else "ham"

            st.markdown(f"<p class='{css}'>{label}</p>", unsafe_allow_html=True)

            # Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Spam Probability"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red"},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},
                        {'range': [30, 70], 'color': "orange"},
                        {'range': [70, 100], 'color': "red"}
                    ]
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

            # Explainability
            st.subheader("🔍 Prediction Explanation")

            explanations = explain_prediction(translated_text, model, vectorizer)

            for word, score in explanations:
                icon = "🔴" if score > 0 else "🟢"
                direction = "SPAM" if score > 0 else "HAM"
                st.write(f"{icon} {word} → pushes towards {direction}")

            st.session_state.email_history.append({
                "prediction": "SPAM" if prediction else "HAM",
                "probability": probability,
                "translated": translated_text
            })

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------------- Admin Dashboard ----------------------
else:

    st.title("📊 Admin Dashboard")

    history = st.session_state.email_history

    if not history:
        st.info("No emails processed yet.")
    else:

        df = pd.DataFrame(history)

        st.metric("Total Emails", len(df))
        st.metric("SPAM Emails", (df['prediction'] == "SPAM").sum())
        st.metric("HAM Emails", (df['prediction'] == "HAM").sum())

        pie = go.Figure(data=[go.Pie(
            labels=["HAM", "SPAM"],
            values=[
                (df['prediction'] == "HAM").sum(),
                (df['prediction'] == "SPAM").sum()
            ],
            hole=0.4
        )])

        st.plotly_chart(pie, use_container_width=True)

        line = go.Figure()
        line.add_trace(go.Scatter(y=df['probability'], mode='lines+markers'))

        st.plotly_chart(line, use_container_width=True)
