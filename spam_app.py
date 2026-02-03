import streamlit as st
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Paths
model_path = r"C:\Users\mukesh\email_spam_detection\spam_model.pkl"
vectorizer_path = r"C:\Users\mukesh\email_spam_detection\vectorizer.pkl"
csv_path = r"C:\Users\mukesh\email_spam_detection\spam.csv"

# Load model & vectorizer
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
else:
    st.error("❌ Model or vectorizer not found! Train and save them first.")
    st.stop()

# App title
st.title("📧 Email Spam Detection with Logistic Regression")
st.write("A simple app to classify emails as **Spam** or **Ham (Not Spam)**")

# ================================
# 1. Show Accuracy & Confusion Matrix
# ================================
if os.path.exists(csv_path):
    # Load dataset
    data = pd.read_csv(csv_path, encoding="latin-1")

    X = data['text']
    y = data['label']

    # Train-test split (same as training time)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Vectorize
    X_test_vec = vectorizer.transform(X_test)

    # Predictions
    y_pred = model.predict(X_test_vec)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    st.subheader("📊 Model Evaluation")
    st.write(f"✅ **Accuracy:** {accuracy:.2%}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    st.pyplot(fig)

else:
    st.warning("⚠️ Dataset not found. Skipping accuracy & confusion matrix.")

# ================================
# 2. Email Prediction
# ================================
st.subheader("🔍 Test an Email")

email_text = st.text_area("✍️ Enter email text here:")

if st.button("Predict"):
    if email_text.strip():
        email_vec = vectorizer.transform([email_text])
        prediction = model.predict(email_vec)[0]
        probability = model.predict_proba(email_vec)[0][1]

        result = "🚨 SPAM" if prediction == 1 else "✅ HAM (Not Spam)"
        st.write(f"### Prediction: {result}")
        st.write(f"**Spam Probability:** {probability:.3f}")

        if probability > 0.7:
            st.warning("⚠️ High chance of Spam!")
        elif probability < 0.3:
            st.success("✅ Likely a safe email.")
        else:
            st.info("⚖️ Uncertain – could go either way.")
    else:
        st.error("Please enter some text to analyze.")

# Footer
st.markdown("---")
st.caption("Built with Logistic Regression + TF-IDF")
