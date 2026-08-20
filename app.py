import streamlit as st
import pandas as pd
import joblib

from feature_extraction import extract_features


# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="centered"
)


# -----------------------------------
# Load Trained Model
# -----------------------------------

MODEL_PATH = "model/phishing_model.pkl"

model = joblib.load(MODEL_PATH)


# -----------------------------------
# Title
# -----------------------------------

st.title("🛡️ Phishing URL Detection System")

st.write(
    "Enter a URL below to analyze its characteristics "
    "and predict whether it is likely to be legitimate or phishing."
)


# -----------------------------------
# URL Input
# -----------------------------------

url = st.text_input(
    "Enter URL",
    placeholder="https://example.com/login"
)


# -----------------------------------
# Prediction
# -----------------------------------

if st.button("🔍 Check URL"):

    if not url.strip():

        st.warning("Please enter a URL.")

    else:

        # Extract features
        features = extract_features(url)

        # Convert features into DataFrame
        feature_df = pd.DataFrame([features])

        # Make prediction
        prediction = model.predict(feature_df)[0]

        # Get prediction probability
        probability = model.predict_proba(feature_df)[0]

        # Probability of phishing
        phishing_probability = probability[1] * 100

        # Probability of benign
        benign_probability = probability[0] * 100


        # -----------------------------------
        # Display Result
        # -----------------------------------

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


        # -----------------------------------
        # Display Features
        # -----------------------------------

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