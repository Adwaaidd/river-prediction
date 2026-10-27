import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from streamlit_autorefresh import st_autorefresh

# Automatically refresh the UI every 5 minutes
st_autorefresh(interval=5 * 60 * 1000, key="rainfall_refresh")
# ------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------

st.set_page_config(
    page_title="Nila River Prediction",
    page_icon="🌊",
    layout="wide"
)

# ------------------------------------------
# LOAD THE TRAINED MODEL
# ------------------------------------------

MODEL_PATH = Path(__file__).parent / "nila_river_mlr.pkl"

@st.cache_resource
def load_model(path):
    return joblib.load(path)

st.title("🌊 Nila River Water Level Prediction")
st.write(
    "Machine Learning Based River Water Level "
    "Prediction System"
)

st.markdown("---")

if not MODEL_PATH.exists():
    st.error("Model file not found!")
    st.write("Expected file location:")
    st.code(str(MODEL_PATH))
    st.info(
        "Place nila_river_mlr.pkl in the same folder "
        "as app.py."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as e:
    st.error("Unable to load the trained model.")
    st.write("Error details:")
    st.code(str(e))
    st.info(
        "Check that nila_river_mlr.pkl is the model "
        "downloaded from your Google Colab notebook."
    )
    st.stop()

# ------------------------------------------
# SIDEBAR
# ------------------------------------------

st.sidebar.title("🌧️ Project Information")

st.sidebar.write(
    "This application predicts the next day's "
    "river water level using Multiple Linear Regression."
)

st.sidebar.markdown("---")

st.sidebar.write("**Model:** Multiple Linear Regression")
st.sidebar.write("**Target:** Next-day river level")
st.sidebar.write("**Output unit:** Metres")

# ------------------------------------------
# INPUT SECTION
# ------------------------------------------

st.header("📋 Enter Environmental Measurements")

st.write(
    "Enter the rainfall, river and dam measurements "
    "to predict the next day's river level."
)

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌧️ Rainfall Data")

        rainfall = st.number_input(
            "Today's rainfall (mm)",
            min_value=0.0,
            value=80.0,
            step=1.0
        )

        rainfall_3d = st.number_input(
            "Total rainfall over 3 days (mm)",
            min_value=0.0,
            value=150.0,
            step=1.0
        )

        rainfall_7d = st.number_input(
            "Total rainfall over 7 days (mm)",
            min_value=0.0,
            value=220.0,
            step=1.0
        )

        river_level = st.number_input(
            "Current river level (m)",
            min_value=0.0,
            value=4.5,
            step=0.1
        )

        level_change = st.number_input(
            "River level change (m)",
            value=0.4,
            step=0.1
        )

    with col2:
        st.subheader("🏞️ Dam Data")

        dam_storage = st.number_input(
            "Dam storage (%)",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=1.0
        )

        dam_inflow = st.number_input(
            "Dam inflow (m³/s)",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

        dam_outflow = st.number_input(
            "Dam outflow (m³/s)",
            min_value=0.0,
            value=60.0,
            step=1.0
        )

    submitted = st.form_submit_button(
        "🔍 Predict Tomorrow's River Level",
        use_container_width=True
    )

# ------------------------------------------
# PREDICTION
# ------------------------------------------

if submitted:

    try:
        # Input order and names must match model training
        features = [
            "rainfall_mm",
            "rainfall_3d_total_mm",
            "rainfall_7d_total_mm",
            "river_level_m",
            "river_level_change_m",
            "dam_storage_pct",
            "dam_inflow_m3s",
            "dam_outflow_m3s"
        ]

        input_data = pd.DataFrame(
            [[
                rainfall,
                rainfall_3d,
                rainfall_7d,
                river_level,
                level_change,
                dam_storage,
                dam_inflow,
                dam_outflow
            ]],
            columns=features
        )

        prediction = float(model.predict(input_data)[0])

        if not pd.notna(prediction):
            st.error("The model returned an invalid prediction.")

        else:
            st.markdown("---")
            st.header("📊 Prediction Results")

            increase = prediction - river_level

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "Current River Level",
                f"{river_level:.2f} m"
            )

            metric2.metric(
                "Predicted River Level",
                f"{prediction:.2f} m",
                delta=f"{increase:+.2f} m"
            )

            metric3.metric(
                "Dam Storage",
                f"{dam_storage:.1f}%"
            )

            # ----------------------------------
            # GRAPH
            # ----------------------------------

            st.subheader("📈 River Level Comparison")

            fig, ax = plt.subplots(figsize=(9, 4))

            labels = ["Current Level", "Predicted Level"]
            values = [river_level, prediction]

            bars = ax.bar(labels, values)

            ax.set_ylabel("River Level (metres)")
            ax.set_title("Current vs Predicted River Level")
            ax.axhline(
                river_level,
                linestyle="--",
                alpha=0.6,
                label="Current level"
            )

            ax.bar_label(
                bars,
                labels=[f"{v:.2f} m" for v in values],
                padding=3
            )

            ax.legend()
            ax.grid(axis="y", linestyle="--", alpha=0.3)

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # ----------------------------------
            # RESULT INTERPRETATION
            # ----------------------------------

            st.subheader("📝 Interpretation")

            if increase > 0.01:
                st.warning(
                    f"The model predicts an increase of "
                    f"{increase:.2f} metres by tomorrow."
                )

            elif increase < -0.01:
                st.info(
                    f"The model predicts a decrease of "
                    f"{abs(increase):.2f} metres by tomorrow."
                )

            else:
                st.info(
                    "The model predicts approximately "
                    "the same river level."
                )

            # ----------------------------------
            # DOWNLOAD PREDICTION
            # ----------------------------------

            result = pd.DataFrame({
                "Measurement": [
                    "Current river level (m)",
                    "Predicted next-day river level (m)",
                    "Predicted level change (m)",
                    "Today's rainfall (mm)",
                    "3-day rainfall total (mm)",
                    "7-day rainfall total (mm)",
                    "Dam storage (%)",
                    "Dam inflow (m3/s)",
                    "Dam outflow (m3/s)"
                ],
                "Value": [
                    river_level,
                    prediction,
                    increase,
                    rainfall,
                    rainfall_3d,
                    rainfall_7d,
                    dam_storage,
                    dam_inflow,
                    dam_outflow
                ]
            })

            st.download_button(
                "📥 Download Prediction Report",
                data=result.to_csv(index=False),
                file_name="nila_river_prediction.csv",
                mime="text/csv"
            )

            st.caption(
                "Prediction generated using the trained "
                "Multiple Linear Regression model."
            )

            st.warning(
                "This is an experimental model prediction, "
                "not an official flood warning. The model's "
                "accuracy depends on the quality and "
                "representativeness of its training data."
            )

    except Exception as e:
        st.error("Something went wrong while predicting.")
        st.write("Error details:")
        st.code(str(e))

# ------------------------------------------
# FOOTER
# ------------------------------------------

st.markdown("---")

st.caption(
    "Nila River Water Level Prediction | "
    "Machine Learning Mini Project"
)
import streamlit as st
import requests
from datetime import datetime

# Nila River region coordinates (approximate)
LATITUDE = 10.77
LONGITUDE = 76.38

def get_live_rainfall():
    """Fetch the latest hourly rainfall estimate from Open-Meteo."""

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "current": "precipitation",
        "hourly": "precipitation",
        "timezone": "Asia/Kolkata",
        "forecast_days": 1
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()

    data = response.json()

    rainfall = data["current"]["precipitation"]
    updated_time = data["current"]["time"]

    return rainfall, updated_time


st.title("🌧️ Nila River Water Level Prediction")
st.subheader("Live Rainfall Monitoring")

if st.button("🔄 Refresh Rainfall"):
    st.cache_data.clear()

try:
    rainfall, updated_time = get_live_rainfall()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Current Rainfall",
            value=f"{rainfall} mm"
        )

    with col2:
        st.metric(
            label="Last Weather Data Time",
            value=updated_time
        )

    st.caption(
        "Rainfall is an estimate from a weather model, "
        "not a direct rain-gauge measurement."
    )

except Exception as e:
    st.error(f"Unable to fetch rainfall data: {e}")


