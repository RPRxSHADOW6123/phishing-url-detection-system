import streamlit as st
import pandas as pd
import joblib

from feature_extraction import extract_features
st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="centered"
)
MODEL_PATH = "model/phishing_model.pkl"

model = joblib.load(MODEL_PATH)
st.title("🛡️ Phishing URL Detection System")
st.write(
    "Enter a URL below to analyze its characteristics "
    "and predict whether it is likely to be legitimate or phishing."
)
url = st.text_input(
    "Enter URL",
    placeholder="https://example.com/login"
)
if st.button("🔍 Check URL"):

    if not url.strip():

        st.warning("Please enter a URL.")

    else:
        features = extract_features(url)
        feature_df = pd.DataFrame([features])
        prediction = model.predict(feature_df)[0]
        probability = model.predict_proba(feature_df)[0]
        phishing_probability = probability[1] * 100
        benign_probability = probability[0] * 100

        st.subheader("Prediction")

        if prediction == 1:

            st.error("⚠️ PHISHING URL")

            st.write(
                f"Phishing Probability: **{phishing_probability:.2f}%**"
            )

        else:

            st.success("✅ LEGITIMATE URL")

            st.write(
                f"Legitimate Probability: **{benign_probability:.2f}%**"
            )

        st.subheader("Extracted URL Features")

        feature_display = pd.DataFrame(
            {
                "Feature": features.keys(),
                "Value": features.values()
            }
        )

        st.dataframe(
            feature_display,
            use_container_width=True,
            hide_index=True
        )