# SMS Spam Detector (Streamlit Web App)

A Machine Learning web application that classifies SMS messages as **Spam** or **Ham (Not Spam)** using a trained model and text vectorization.

## Live Demo
https://sms-spam-detector-jyr2hkmsniafqmhrstysyw.streamlit.app/

## Problem Statement
Spam SMS messages can contain scams, phishing links, or unwanted promotions.  
This project builds an AI model that automatically detects spam to improve user safety and reduce fraud.

## Features
- Predicts whether an SMS is **Spam** or **Ham**
- Simple, user-friendly Streamlit interface
- Model + vectorizer saved and reused for fast predictions

## Project Workflow (End-to-End)
1. Defined the problem (Spam vs Ham classification)
2. Acquired and loaded the SMS dataset
3. Cleaned/preprocessed text data
4. Converted text to numeric features using a vectorizer
5. Trained and evaluated a Machine Learning classifier
6. Deployed the model as a free Streamlit web app

## Project Structure
- `app.py` → Streamlit application
- `spam_model.pkl` → trained ML model
- `vectorizer.pkl` → text vectorizer used in training
- `requirements.txt` → dependencies for deployment

## How to Run Locally
```bash
pip install -r requirements.txt
python -m streamlit run app.py
