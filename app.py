import streamlit as st
import pandas as pd
import joblib 
import numpy as np


# Fire and Smoke Gradient Background
st.markdown(
    """
    <style>
    /* Background gradient for fire-theme */
    .stApp {
        background: linear-gradient(-45deg, #1c1c1c, #8B0000, #FF4500, #FFA500, #1c1c1c);
        background-size: 500% 500%;
        animation: gradientBG 20s ease infinite;
        color: white;
        font-size: 18px;
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Title styling */
    h1 {
        font-size: 40px !important;
        color: #FFD700;
        margin-bottom: 20px;
    }

    /* Button styling */
    .stButton > button {
        background-color: #e25822;
        color: white;
        border: none;
        padding: 0.5em 1em;
        border-radius: 6px;
        font-weight: bold;
        font-size: 18px;
    }

    .stButton > button:hover {
        background-color: #ff6600;
        color: black;
    }

    /* Make X and Y inputs same style as month/day (dark transparent bg, white text) */
    div.row-widget.stNumberInput:nth-child(1) input,
    div.row-widget.stNumberInput:nth-child(2) input {
        background-color: rgba(0, 0, 0, 0.6) !important;
        color: white !important;
        font-size: 18px !important;
        padding: 0 !important;
        margin: 0 !important;
        border-radius: 0 !important;
        border: 1px solid white !important;
    }

    /* All other number, select and slider inputs */
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"],
    .stTextInput input {
        background-color: rgba(0, 0, 0, 0.6) !important;
        color: white !important;
        font-size: 18px !important;
        padding: 0 !important;
        margin: 0 !important;
        border-radius: 0 !important;
        border: 1px solid white !important;
    }

    /* Sliders styling */
    .stSlider > div > div {
        padding: 0 !important;
        margin: 0 !important;
    }

    input[type="range"] {
        accent-color: black !important;
    }

    .stSlider {
        background-color: transparent !important;
        border: none !important;
    }

    /* Labels */
    label, .stMarkdown {
        font-size: 20px !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)



#load model
regressor = joblib.load('regressor.pkl')
classifier = joblib.load('classifier.pkl')
le_month = joblib.load('le_month.pkl')
le_day = joblib.load('le_day.pkl')

st.title("🔥 Forest Fire Burn Area Prediction System")
st.markdown("Enter environmental conditions to predict fire risk and area (in ha) (if any).")

#input features
X = st.number_input("X (spatial coordinate)", 1,9,4)
Y = st.number_input("Y (spatial coordinate)", 2,9,4)
month = st.selectbox("Month", le_month.classes_)
day = st.selectbox("Day", le_day.classes_)
FFMC = st.slider("FFMC", 18.7, 96.20, 86.0)
DMC = st.slider("DMC", 1.1, 291.3, 26.2)
DC = st.slider("DC", 7.9, 860.6, 94.3)
ISI = st.slider("ISI", 0.0, 56.1, 5.1)
temp = st.slider("Temperature (°C)", 2.2, 33.3, 18.0)
RH = st.slider("Relative Humidity (%)", 15, 100, 45)
wind = st.slider("Wind speed (km/h)", 0.4, 9.4, 4.0)
rain = st.slider("Rain (mm)", 0.0, 6.4, 0.0)

#converted categorical
month_encoded = le_month.transform([month])[0]
day_encoded = le_day.transform([day])[0]

input_data = pd.DataFrame([[X,Y, month_encoded, day_encoded, FFMC, DMC, DC, ISI, temp, RH, wind, rain]])

#predict
if st.button("Predict "):
    fire_class = classifier.predict(input_data)[0]

    if fire_class == 0:
        st.success("No Forest Fire Predicted. ")
    else:
        area_pred = regressor.predict(input_data)[0]
        st.error("🔥 Forest Fire Predicted!")
        st.write(f"🌲 Estimated Burn Area: **{area_pred:.2f} hectares**")



