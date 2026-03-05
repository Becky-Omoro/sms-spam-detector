import re
import pandas as pd
import streamlit as st
import joblib
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ---------- Page setup ----------
st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")

# ---------- Load model ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_artifacts()

# ---------- Helpers ----------
def count_urls(text: str) -> int:
    return len(re.findall(r"(https?://\S+|www\.\S+)", text.lower()))

def message_stats(text: str):
    return {
        "length": len(text),
        "digits": sum(ch.isdigit() for ch in text),
        "exclaims": text.count("!"),
        "urls": count_urls(text),
    }

def predict_one(text: str):
    vec = vectorizer.transform([text])
    pred = int(model.predict(vec)[0])
    proba = model.predict_proba(vec)[0]  # [ham, spam]
    return pred, float(proba[0]), float(proba[1])

# ---------- Session state (history) ----------
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    show_keywords = st.checkbox("Show suspicious keywords", value=True)
    show_stats = st.checkbox("Show message statistics", value=True)
    show_prob = st.checkbox("Show probability details", value=True)

    st.divider()
    if st.button("🧹 Clear history"):
        st.session_state.history = []
        st.success("History cleared.")

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["🧠 Predict", "📜 History", "📊 Model Evaluation"])

# =========================
# TAB 1: PREDICT
# =========================
with tab1:
    st.title("📩 SMS Spam Detector")
    st.caption("Paste an SMS message and the model will classify it as Spam or Ham.")

    # Demo buttons
    c1, c2 = st.columns(2)
    with c1:
        demo_spam = st.button("Try SPAM example", use_container_width=True)
    with c2:
        demo_ham = st.button("Try HAM example", use_container_width=True)

    if demo_spam:
        st.session_state.demo_text = "CONGRATULATIONS! You have WON a FREE prize. Click http://bit.ly/win-now to claim urgently!!!"
    if demo_ham:
        st.session_state.demo_text = "Hey, are we still meeting at 4pm? Let me know when you're on your way."

    default_text = st.session_state.get("demo_text", "")

    message = st.text_area("Message", value=default_text, placeholder="Type or paste your SMS here...")

    if st.button("Predict"):
        if message.strip() == "":
            st.warning("Please type an SMS message first.")
        else:
            pred, ham_prob, spam_prob = predict_one(message)

            # Result
            if pred == 1:
                st.error("🚨 Prediction: SPAM")
            else:
                st.success("✅ Prediction: HAM (Not Spam)")

            # Confidence bar
            st.subheader("Confidence")
            st.progress(spam_prob)
            st.write(f"Spam probability: **{spam_prob*100:.2f}%** | Ham probability: **{ham_prob*100:.2f}%**")

            if show_prob:
                m1, m2 = st.columns(2)
                m1.metric("Spam %", f"{spam_prob*100:.2f}%")
                m2.metric("Ham %", f"{ham_prob*100:.2f}%")

            # Stats
            stats = message_stats(message)
            if show_stats:
                st.subheader("Message statistics")
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Length", stats["length"])
                s2.metric("Digits", stats["digits"])
                s3.metric("Exclamation", stats["exclaims"])
                s4.metric("URLs", stats["urls"])

            # Keywords
            found = []
            if show_keywords:
                st.subheader("Suspicious keyword flags")
                spam_keywords = [
                    "free", "win", "winner", "prize", "claim", "urgent", "offer", "bonus",
                    "click", "limited", "money", "cash", "reward", "credit", "loan"
                ]
                found = [w for w in spam_keywords if w in message.lower()]

                if found:
                    st.warning("⚠ Found common spam-related keywords:")
                    st.write(", ".join([w.upper() for w in found]))
                else:
                    st.info("No common spam keywords detected.")

            # Save to history
            st.session_state.history.append({
                "message": message,
                "prediction": "spam" if pred == 1 else "ham",
                "spam_probability": round(spam_prob, 4),
                "ham_probability": round(ham_prob, 4),
                "length": stats["length"],
                "digits": stats["digits"],
                "exclaims": stats["exclaims"],
                "urls": stats["urls"],
                "keywords_found": ", ".join(found).upper() if found else ""
            })

            st.caption("Tip: Spam messages often include urgency, links, prizes, or money-related words.")

# =========================
# TAB 2: HISTORY
# =========================
with tab2:
    st.subheader("📜 Prediction History")
    if len(st.session_state.history) == 0:
        st.info("No predictions yet. Go to the Predict tab and test a message.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True)

        st.download_button(
            "⬇️ Download history as CSV",
            data=hist_df.to_csv(index=False).encode("utf-8"),
            file_name="spam_detector_history.csv",
            mime="text/csv"
        )

# =========================
# TAB 3: MODEL EVALUATION
# =========================
with tab3:
    st.subheader("📊 Model Evaluation (Quick Demo)")
    st.write("This section demonstrates evaluation on a small built-in sample set. "
             "In your report, you will show full test-set results from Colab.")

    sample = pd.DataFrame({
        "message": [
            "WIN a FREE prize now!!! Click http://bit.ly/claim",
            "Hey, are you coming today?",
            "URGENT! Your account has won cash reward",
            "Let's meet at 6pm at the mall",
            "Claim your bonus offer now!!!",
            "Please call me when you are free"
        ],
        "true_label": [1, 0, 1, 0, 1, 0]  # 1=spam, 0=ham
    })

    # Predict on sample
    preds = []
    for msg in sample["message"]:
        pred, _, _ = predict_one(msg)
        preds.append(pred)

    y_true = sample["true_label"].tolist()
    y_pred = preds

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    st.write(f"Accuracy on sample set: **{acc*100:.2f}%**")
    st.write("Confusion Matrix (rows=true, cols=pred):")
    st.write(cm)

    st.text("Classification Report:")
    st.text(classification_report(y_true, y_pred, target_names=["ham", "spam"]))