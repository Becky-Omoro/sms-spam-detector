import re
import pandas as pd
import streamlit as st
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ---------------- Page config ----------------
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Session state ----------------
if "history" not in st.session_state:
    st.session_state.history = []
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# ---------------- Theme CSS ----------------
LIGHT_CSS = """
<style>
.stApp {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);}
.main > div {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.25);
}
h1 {
    background: linear-gradient(120deg, #667eea, #764ba2, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-size: 200% 200%;
    animation: gradient 3s ease infinite;
    font-size: 3rem !important;
    font-weight: 800 !important;
}
@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.prediction-card {
    background: white;
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.10);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
.prediction-card:hover {transform: translateY(-4px); box-shadow: 0 18px 55px rgba(0,0,0,0.14);}
.stButton > button {
    background: linear-gradient(45deg, #667eea, #764ba2);
    color: white; border: none; border-radius: 50px;
    padding: 0.75rem 2rem; font-weight: 650;
    text-transform: uppercase; letter-spacing: 1px;
    box-shadow: 0 4px 15px rgba(102,126,234,0.35);
    transition: all 0.25s ease;
}
.stButton > button:hover {transform: scale(1.03); box-shadow: 0 7px 22px rgba(102,126,234,0.55);}
.keyword-cloud {display:flex; flex-wrap:wrap; gap:10px; margin: 1rem 0;}
.keyword-tag {
    background: linear-gradient(135deg, #667eea22, #764ba222);
    border: 2px solid #667eea;
    color: #333;
    padding: 8px 16px;
    border-radius: 25px;
    font-size: 0.9rem;
    font-weight: 650;
}
.metric-card {
    background: linear-gradient(135deg, #f6f9fc, #e6f0f5);
    border-radius: 15px;
    padding: 1.25rem;
    text-align: center;
    border: 1px solid rgba(102, 126, 234, 0.20);
}
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
    background: rgba(102, 126, 234, 0.12);
    padding: 0.75rem;
    border-radius: 50px;
}
.stTabs [data-baseweb="tab"] {border-radius: 30px; padding: 0.75rem 1.25rem; font-weight: 650;}
small {color: #666;}
</style>
"""

DARK_CSS = """
<style>
.stApp {background: linear-gradient(135deg, #0f172a 0%, #1f2937 100%);}
.main > div {
    background: rgba(17, 24, 39, 0.92);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.35);
}
h1 {
    background: linear-gradient(120deg, #60a5fa, #a78bfa, #fb7185);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-size: 200% 200%;
    animation: gradient 3s ease infinite;
    font-size: 3rem !important;
    font-weight: 800 !important;
}
@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
.prediction-card {
    background: rgba(31, 41, 55, 0.9);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.35);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    color: #e5e7eb;
}
.prediction-card:hover {transform: translateY(-4px); box-shadow: 0 18px 55px rgba(0,0,0,0.45);}
.stButton > button {
    background: linear-gradient(45deg, #60a5fa, #a78bfa);
    color: white; border: none; border-radius: 50px;
    padding: 0.75rem 2rem; font-weight: 650;
    text-transform: uppercase; letter-spacing: 1px;
    box-shadow: 0 4px 15px rgba(96,165,250,0.25);
    transition: all 0.25s ease;
}
.stButton > button:hover {transform: scale(1.03); box-shadow: 0 7px 22px rgba(167,139,250,0.35);}
.keyword-cloud {display:flex; flex-wrap:wrap; gap:10px; margin: 1rem 0;}
.keyword-tag {
    background: rgba(96, 165, 250, 0.12);
    border: 2px solid rgba(167, 139, 250, 0.85);
    color: #e5e7eb;
    padding: 8px 16px;
    border-radius: 25px;
    font-size: 0.9rem;
    font-weight: 650;
}
.metric-card {
    background: rgba(31, 41, 55, 0.85);
    border-radius: 15px;
    padding: 1.25rem;
    text-align: center;
    border: 1px solid rgba(148, 163, 184, 0.25);
    color: #e5e7eb;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
    background: rgba(96, 165, 250, 0.12);
    padding: 0.75rem;
    border-radius: 50px;
}
.stTabs [data-baseweb="tab"] {border-radius: 30px; padding: 0.75rem 1.25rem; font-weight: 650; color: #e5e7eb;}
small {color: #cbd5e1;}
</style>
"""

st.markdown(LIGHT_CSS if st.session_state.theme == "light" else DARK_CSS, unsafe_allow_html=True)

# ---------------- Load model ----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_artifacts()

# ---------------- Helpers ----------------
def count_urls(text: str) -> int:
    return len(re.findall(r"(https?://\S+|www\.\S+)", text.lower()))

def stats(text: str):
    return {
        "Length": len(text),
        "Digits": sum(ch.isdigit() for ch in text),
        "Exclamation": text.count("!"),
        "Question": text.count("?"),
        "Capital Words": len(re.findall(r"\b[A-Z]{2,}\b", text)),
        "URLs": count_urls(text),
    }

def predict_message(text: str):
    vec = vectorizer.transform([text])
    pred = int(model.predict(vec)[0])  # 1=spam, 0=ham

    # Safe proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]  # [ham, spam]
        ham_prob, spam_prob = float(proba[0]), float(proba[1])
    else:
        ham_prob, spam_prob = 0.5, 0.5

    return pred, ham_prob, spam_prob

def get_spam_keywords():
    return [
        "free", "win", "winner", "prize", "claim", "urgent", "offer", "bonus",
        "click", "limited", "money", "cash", "reward", "credit", "loan",
        "congratulations", "selected", "won", "lottery", "jackpot",
        "expires", "action", "verify"
    ]

def clean_words_for_freq(text: str):
    # Lowercase + remove punctuation for better word counting
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    words = [w for w in cleaned.split() if len(w) > 1]
    return words

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("<h2 style='text-align:center;'> SMS SPAM DETECTOR</h2>", unsafe_allow_html=True)

    total_predictions = len(st.session_state.history)
    spam_count = sum(1 for h in st.session_state.history if h["prediction"] == "spam")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Predictions", total_predictions)
    with c2:
        st.metric("Spam Detected", spam_count)

    st.markdown("---")
    st.markdown("###  Project Overview")
    st.write(" **Task:** Classification (Spam vs Ham)")
    st.write(" **Model:** Naive Bayes + TF-IDF")
    st.write(" **Metrics:** See evaluation in notebook/report")
    st.write(" **Response:** Real-time prediction")

    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.link_button("🌐 Live App", "https://sms-spam-detector-jyr2hkmsniafqmhrstysyw.streamlit.app/")
    st.link_button("💻 GitHub Repo", "https://github.com/Becky-Omoro/sms-spam-detector")
    st.link_button("📄 README / Documentation", "https://github.com/Becky-Omoro/sms-spam-detector#readme")

    st.markdown("---")
    colA, colB = st.columns(2)
    with colA:
        if st.button(" Clear History", use_container_width=True):
            st.session_state.history = []
            st.success("History cleared.")
            st.rerun()
    with colB:
        if st.button(" Theme", use_container_width=True):
            st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
            st.rerun()

# ---------------- Header ----------------
st.markdown("""
<div style='text-align:center; padding: 1.5rem 0 0.5rem 0;'>
    <h1>📩 SMS Spam Detector</h1>
    <p style='font-size: 1.1rem; margin-top: 0.3rem;'>
        Powered by AI · Real-time Analysis · Clean UI
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([" Predict", " Analytics", " History", " About"])

# ---------------- Predict tab ----------------
with tab1:
    st.markdown("###  Quick Test Messages")
    b1, b2, b3, b4 = st.columns(4)

    with b1:
        if st.button(" Prize SPAM", use_container_width=True):
            st.session_state.example = "CONGRATULATIONS! You've WON $1,000,000! Click http://bit.ly/claim-now to claim your prize TODAY!!!"
    with b2:
        if st.button(" Loan SPAM", use_container_width=True):
            st.session_state.example = "URGENT: Low interest loans available! Get cash now! No credit check! Call 0800 123 4567 today!"
    with b3:
        if st.button(" Normal HAM", use_container_width=True):
            st.session_state.example = "Hey Mom, just letting you know I'll be home for dinner around 7pm. Love you!"
    with b4:
        if st.button(" Meeting HAM", use_container_width=True):
            st.session_state.example = "Hi team, reminder about our meeting tomorrow at 10am. Please bring your reports."

    st.markdown("")
    colL, colR = st.columns([3, 1])

    with colL:
        message = st.text_area(
            "✏️ Enter your message:",
            value=st.session_state.get("example", ""),
            placeholder="Type or paste your SMS message here...",
            height=160
        )

    with colR:
        if message:
            chars_left = 160 - len(message)
            st.metric("Characters", len(message))
            st.caption(f"{chars_left} left (SMS ~160 chars)")
            preview = stats(message)
            st.caption(f"Digits: {preview['Digits']}  |  URLs: {preview['URLs']}")

    st.markdown("")
    colX, colY, colZ = st.columns([1, 2, 1])
    with colY:
        run = st.button("🔮 Analyze Message", type="primary", use_container_width=True)

    if run:
        if not message.strip():
            st.warning("⚠️ Please enter a message to analyze.")
        else:
            with st.spinner("🤖 Analyzing..."):
                pred, ham_prob, spam_prob = predict_message(message)
                s = stats(message)

            # Save history
            st.session_state.history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "message": message[:50] + "..." if len(message) > 50 else message,
                "full_message": message,
                "prediction": "spam" if pred == 1 else "ham",
                "spam_prob": round(spam_prob, 4),
                "ham_prob": round(ham_prob, 4),
                "length": s["Length"],
                "digits": s["Digits"],
                "exclamation": s["Exclamation"],
                "urls": s["URLs"],
                "capital_words": s["Capital Words"]
            })

            st.markdown("### 🎯 Result")
            if pred == 1:
                st.markdown(
                    "<div class='prediction-card' style='border-left: 10px solid #ff6b6b;'>"
                    "<h2 style='margin:0;'>🚨 SPAM DETECTED</h2>"
                    "<p>Be cautious. This message looks like spam.</p>"
                    "</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div class='prediction-card' style='border-left: 10px solid #48dbfb;'>"
                    "<h2 style='margin:0;'>✅ LEGITIMATE (HAM)</h2>"
                    "<p>This message looks safe.</p>"
                    "</div>",
                    unsafe_allow_html=True
                )

            st.markdown("### 📊 Confidence")
            cA, cB = st.columns(2)
            with cA:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=spam_prob * 100,
                    title={'text': "Spam Confidence (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#ff6b6b" if spam_prob >= 0.5 else "#48dbfb"},
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 50}
                    }
                ))
                fig.update_layout(height=260, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)
            with cB:
                prob_df = pd.DataFrame({
                    "Category": ["Spam", "Ham"],
                    "Probability": [spam_prob * 100, ham_prob * 100]
                })
                fig2 = px.bar(prob_df, x="Category", y="Probability", text="Probability")
                fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                fig2.update_layout(showlegend=False, height=260, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("### 🔍 Message Details")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Length", s["Length"])
            m2.metric("Digits", s["Digits"])
            m3.metric("!", s["Exclamation"])
            m4.metric("?", s["Question"])
            m5.metric("URLs", s["URLs"])

            # Keywords
            spam_keywords = get_spam_keywords()
            found = [w.upper() for w in spam_keywords if w in message.lower()]
            if found:
                st.markdown("#### ⚠️ Suspicious Keywords")
                html = '<div class="keyword-cloud">' + "".join([f'<span class="keyword-tag">⚠️ {k}</span>' for k in found[:12]]) + "</div>"
                st.markdown(html, unsafe_allow_html=True)

            # Word frequency chart
            st.markdown("#### 📝 Word Frequency (Top 10)")
            words = clean_words_for_freq(message)
            if words:
                freq = pd.Series(words).value_counts().head(10)
                fig3 = px.bar(x=freq.values, y=freq.index, orientation="h",
                              labels={"x": "Frequency", "y": "Word"},
                              title="Top 10 Words in the Message")
                fig3.update_layout(height=320)
                st.plotly_chart(fig3, use_container_width=True)

# ---------------- Analytics tab ----------------
with tab2:
    st.markdown("### 📊 Analytics (from your session history)")
    if not st.session_state.history:
        st.info("No predictions yet. Make predictions to see analytics.")
    else:
        df = pd.DataFrame(st.session_state.history)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total", len(df))
        c2.metric("Spam", (df["prediction"] == "spam").sum())
        c3.metric("Ham", (df["prediction"] == "ham").sum())
        c4.metric("Avg Length", f"{df['length'].mean():.0f}")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(df, names="prediction", title="Prediction Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(df, x="length", color="prediction", nbins=20, title="Message Length Distribution")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📈 Feature Comparison")
        feature = st.selectbox("Select feature:", ["length", "digits", "exclamation", "urls", "capital_words"])
        fig = px.box(df, x="prediction", y=feature, color="prediction", title=f"{feature.title()} by Prediction")
        st.plotly_chart(fig, use_container_width=True)

# ---------------- History tab ----------------
with tab3:
    st.markdown("### 📜 Prediction History")
    if not st.session_state.history:
        st.info("No history yet. Go to Predict tab.")
    else:
        df = pd.DataFrame(st.session_state.history)

        search = st.text_input("🔍 Search messages", "")
        filt = st.selectbox("Filter", ["All", "Spam Only", "Ham Only"])

        fdf = df.copy()
        if search:
            fdf = fdf[fdf["full_message"].str.contains(search, case=False, na=False)]
        if filt == "Spam Only":
            fdf = fdf[fdf["prediction"] == "spam"]
        elif filt == "Ham Only":
            fdf = fdf[fdf["prediction"] == "ham"]

        st.dataframe(fdf[["timestamp", "prediction", "spam_prob", "ham_prob", "message"]], use_container_width=True)

        st.download_button(
            "⬇️ Download CSV",
            data=fdf.to_csv(index=False).encode("utf-8"),
            file_name="spam_detector_history.csv",
            mime="text/csv"
        )

# ---------------- About tab ----------------
with tab4:
    st.markdown("### ℹ️ About")
    st.write("""
This project is a **machine learning classification** system that detects SMS spam.

**Steps:**
1) Clean the dataset  
2) Convert text to numbers using **TF-IDF**  
3) Train a classifier (e.g., Naive Bayes)  
4) Evaluate using accuracy, precision, recall, and confusion matrix  
5) Deploy using Streamlit
""")
    st.write("**Live App:** https://sms-spam-detector-jyr2hkmsniafqmhrstysyw.streamlit.app/")
    st.write("**GitHub:** https://github.com/Becky-Omoro/sms-spam-detector")


