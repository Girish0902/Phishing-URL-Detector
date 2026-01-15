import streamlit as st
import joblib
import numpy as np
from feature_extractor import URLFeatureExtractor

st.set_page_config(page_title="AI Phishing URL Detector", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("phishing_url_detector.pkl")

model = load_model()
extractor = URLFeatureExtractor()

st.title("🛡️ AI-Based Phishing & Malicious URL Detection System")

url = st.text_input("Enter URL to analyze:")

if st.button("Analyze URL"):
    if not url.strip():
        st.warning("Please enter a URL")
    else:
        try:
            features = extractor.extract_features(url)
            features = np.array(features).reshape(1, -1)

            phishing_prob = model.predict_proba(features)[0][1]
            risk_score = int(phishing_prob * 100)

            st.subheader("🔍 Risk Score")
            st.metric("Phishing Probability", f"{risk_score}%")

            if risk_score >= 70:
                st.error(f"🚨 HIGH RISK ({risk_score}%) — PHISHING URL")
            elif risk_score >= 40:
                st.warning(f"⚠️ MEDIUM RISK ({risk_score}%) — SUSPICIOUS")
            else:
                st.success(f"✅ LOW RISK ({risk_score}%) — LEGITIMATE")

            # Explanation
            st.subheader("📋 Analysis Details")
            reasons = []

            if "login" in url.lower():
                reasons.append("Contains login-related keyword")
            if not url.startswith("https"):
                reasons.append("Does not use HTTPS")
            if "-" in url:
                reasons.append("Hyphen detected in domain")
            if extractor.extract_features(url)[17] > 0:
                reasons.append("Suspicious keywords detected")

            if reasons:
                for r in reasons:
                    st.write(f"• {r}")
            else:
                st.write("No strong phishing indicators detected")

        except Exception as e:
            st.error(f"Error analyzing URL: {e}")

            st.caption(
    "⚠️ This tool provides probabilistic risk analysis and should not be used as the sole security decision mechanism."
)
