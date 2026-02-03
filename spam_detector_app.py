import streamlit as st
import joblib
import os

# File paths
model_path = r"C:\Users\mukesh\email_spam_detection\spam_model.pkl"
vectorizer_path = r"C:\Users\mukesh\email_spam_detection\vectorizer.pkl"

# Load model and vectorizer
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
else:
    st.error("❌ Model or vectorizer not found! Please train and save them first.")
    st.stop()

# Streamlit app
st.title("📧 Email Spam Detection")
st.write("Enter an email message below to check if it's **Spam or Ham**")

# Text input
email_text = st.text_area("✍️ Type your email text here:")

if st.button("🔍 Predict"):
    if email_text.strip():
        # Transform text
        email_vec = vectorizer.transform([email_text])
        prediction = model.predict(email_vec)[0]
        probability = model.predict_proba(email_vec)[0][1]

        # Result
        result = "🚨 SPAM" if prediction == 1 else "✅ HAM (Not Spam)"
        st.subheader("Prediction Result")
        st.write(result)

        # Probability
        st.write(f"**Spam Probability:** {probability:.3f}")

        # Extra info
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
