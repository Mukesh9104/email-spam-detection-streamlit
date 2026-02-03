Multilingual Email Spam Detection System
Project Overview

The Multilingual Email Spam Detection System is a Machine Learning–based web application developed using Streamlit. The system classifies email content as Spam or Ham (legitimate message).

The application supports multiple languages by detecting the input language and translating it into English before classification. It also provides explainable prediction insights, real-time analytics, and probability-based visualization.

Objectives

Detect spam emails using Machine Learning

Support multilingual email inputs

Provide explainable prediction insights

Display real-time analytics dashboard

Deliver a professional and interactive user interface

Technologies Used
Machine Learning

Logistic Regression

TF-IDF Vectorization

Scikit-learn

Natural Language Processing

LangDetect (Language Detection)

GoogleTrans (Translation)

Frontend and Visualization

Streamlit

Plotly

Custom CSS Styling

Backend and Model Handling

Joblib

Pandas

Key Features
Multilingual Support

Detects language of email input automatically

Translates content to English before classification

Spam Probability Visualization

Displays spam confidence using gauge visualization

Explainable AI

Highlights words influencing prediction

Shows direction of influence toward Spam or Ham

File Upload Support

Accepts text-based email files

Admin Analytics Dashboard

Displays total predictions

Shows Spam vs Ham distribution

Displays probability trend visualization

User Interface

Glassmorphism-based design

Responsive layout

Background wallpaper support

Project Structure
email-spam-detection-streamlit/
│
├── app.py
├── spam_model.pkl
├── vectorizer.pkl
├── background.jpg
├── requirements.txt
├── README.md
└── .gitignore

Installation and Setup
Clone Repository
git clone https://github.com/Mukesh9104/email-spam-detection-streamlit.git
cd email-spam-detection-streamlit

Install Dependencies
pip install -r requirements.txt

Run Application
streamlit run app.py

Machine Learning Workflow

Data Collection

Data Preprocessing

Feature Extraction using TF-IDF

Model Training using Logistic Regression

Model Evaluation

Deployment using Streamlit

Real-Time Prediction and Analytics

Model Performance

Algorithm: Logistic Regression

Feature Engineering: TF-IDF

Dataset: Email Spam Dataset

Evaluation Metric: Accuracy Score

Explainable AI Implementation

The system extracts influential words using Logistic Regression coefficients and TF-IDF feature weights. These words are displayed to explain prediction results.

Multilingual Processing Flow
User Input
   ↓
Language Detection
   ↓
Translation to English
   ↓
Spam Classification
   ↓
Prediction Visualization

Future Enhancements

Deep learning-based spam detection models

Email attachment scanning

Email server integration

User authentication and role-based access

Cloud deployment

Feedback-based model retraining

Author

Mukesh Kanna B
MSc Software Systems
Sri Krishna Arts and Science College, Coimbatore

License

This project is developed for academic and educational purposes.

Contributions

Contributions, issues, and feature requests are welcome.

Suggested Requirements File

If not already created, use:

streamlit
pandas
scikit-learn
plotly
joblib
langdetect
googletrans==4.0.0-rc1
