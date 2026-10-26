import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="AI Crop Advisor",
    page_icon="🌱",
    layout="centered"
)

saved = joblib.load("crop_model.pkl")
model = saved["model"]
features = saved["features"]

st.title("🌱 AI-Based Crop Recommendation System")
st.write(
    "Enter soil and weather information to see "
    "the top three crops predicted by the AI model."
)

with st.form("crop_inputs"):
    st.subheader("Soil nutrient values")

    col1, col2, col3 = st.columns(3)

    with col1:
        N = st.number_input(
            "Nitrogen (N)", 0.0, 200.0, 50.0
        )
    with col2:
        P = st.number_input(
            "Phosphorus (P)", 0.0, 200.0, 40.0
        )
    with col3:
        K = st.number_input(
            "Potassium (K)", 0.0, 250.0, 40.0
        )

    st.subheader("Weather and soil conditions")

    temperature = st.number_input(
        "Temperature (°C)", -5.0, 60.0, 25.0
    )
    humidity = st.number_input(
        "Humidity (%)", 0.0, 100.0, 70.0
    )
    ph = st.number_input(
        "Soil pH", 0.0, 14.0, 6.5
    )
    rainfall = st.number_input(
        "Rainfall (mm)", 0.0, 1000.0, 100.0
    )

    submitted = st.form_submit_button(
        "Recommend Crops"
    )

if submitted:
    values = {
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }

    inputs = pd.DataFrame([values])[features]

    scores = model.predict_proba(inputs)[0]
    ranking = sorted(
        zip(model.classes_, scores),
        key=lambda item: item[1],
        reverse=True
    )[:3]

    st.subheader("Your top three crop suggestions")

    for rank, (crop, score) in enumerate(ranking, 1):
        st.markdown(f"### {rank}. {crop.title()}")
        st.progress(float(score))
        st.caption(f"Model score: {score:.1%}")

    st.warning(
        "This is a model-based suggestion, not a "
        "guarantee of crop yield. Check the local "
        "season, soil test, irrigation, and advice "
        "from an agricultural expert before planting."
    )

st.divider()
st.caption(
    "Educational prototype | AI Crop Advisor"
)
