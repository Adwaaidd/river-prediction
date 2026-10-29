import streamlit as st
import streamlit_autorefresh
import requests
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime
import json


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=" River Water Level Prediction",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# AUTOMATIC REFRESH - EVERY 5 MINUTES
# ============================================================

streamlit_autorefresh.st_autorefresh(
    interval=5 * 60 * 1000,
    key="river_dashboard_refresh"
)


# ============================================================
# BLACK BACKGROUND AND DARK UI
# ============================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #000000;
    color: #FFFFFF;
}

[data-testid="stAppViewContainer"] {
    background-color: #000000;
}

[data-testid="stHeader"] {
    background-color: #000000;
}

[data-testid="stMain"] {
    background-color: #000000;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #080808;
    border-right: 1px solid #202020;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

/* General text */
.stApp p,
.stApp label,
.stApp li {
    color: #D1D5DB;
}

/* Headings */
h1, h2, h3, h4, h5 {
    color: #FFFFFF !important;
}

/* Hero section */
.hero {
    background: linear-gradient(135deg, #080808, #001925);
    border: 1px solid #073B4C;
    border-radius: 18px;
    padding: 28px;
    margin-bottom: 24px;
}

.hero-title {
    font-size: 34px;
    font-weight: 800;
    color: #FFFFFF;
    margin-bottom: 8px;
}

.hero-subtitle {
    font-size: 15px;
    color: #9CA3AF;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, #111111, #080808);
    border: 1px solid #262626;
    border-radius: 14px;
    padding: 20px;
    min-height: 120px;
}

.metric-title {
    color: #9CA3AF;
    font-size: 13px;
    margin-bottom: 10px;
}

.metric-value {
    color: #00D4FF;
    font-size: 28px;
    font-weight: 800;
}

.metric-caption {
    color: #737373;
    font-size: 12px;
    margin-top: 8px;
}

/* Section heading */
.section-heading {
    font-size: 21px;
    font-weight: 700;
    color: #00D4FF;
    margin-top: 24px;
    margin-bottom: 12px;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button {
    background-color: #007F99;
    color: #FFFFFF;
    border: 1px solid #00B8D9;
    border-radius: 9px;
    font-weight: 600;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #00A6CC;
    color: #FFFFFF;
    border: 1px solid #00D4FF;
}

/* Input boxes */
.stNumberInput input,
.stTextInput input {
    background-color: #111111;
    color: #FFFFFF;
}

/* Select boxes */
[data-baseweb="select"] > div {
    background-color: #111111;
    color: #FFFFFF;
}

/* Sliders */
.stSlider [data-baseweb="slider"] {
    color: #00D4FF;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: #080808;
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    color: #D1D5DB;
}

.stTabs [aria-selected="true"] {
    color: #00D4FF !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #0A0A0A;
    border: 1px solid #292929;
    border-radius: 10px;
}

/* Alerts */
[data-testid="stAlert"] {
    background-color: #111111;
    border: 1px solid #333333;
}

/* Divider */
hr {
    border-color: #252525;
}

/* Footer */
.footer {
    text-align: center;
    color: #666666;
    font-size: 12px;
    padding: 20px;
}

/* Mobile responsiveness */
@media (max-width: 640px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
        padding-top: 1rem;
    }

    .hero {
        padding: 18px;
    }

    .hero-title {
        font-size: 25px;
    }

    .metric-value {
        font-size: 23px;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = Path(__file__).parent / "nila_river_mlr.pkl"

# Approximate location near the Nila River in Kerala.
# Change these coordinates if you have a specific monitoring
# station whose location is known.
LATITUDE = 10.77
LONGITUDE = 76.38

FEATURES = [
    "rainfall_mm",
    "rainfall_3d_total_mm",
    "rainfall_7d_total_mm",
    "river_level_m",
    "river_level_change_m",
    "dam_storage_pct",
    "dam_inflow_m3s",
    "dam_outflow_m3s"
]


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model(path, modified_time):
    return joblib.load(path)


if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Please upload "
        "'nila_river_mlr.pkl' to the same GitHub folder as app.py."
    )
    st.stop()

try:
    model = load_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime
    )
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.stop()


# ============================================================
# GET WEATHER DATA FROM OPEN-METEO
# ============================================================

@st.cache_data(ttl=300)
def get_weather_data():

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "current": "precipitation",
        "hourly": "precipitation",
        "past_days": 7,
        "forecast_days": 1,
        "timezone": "Asia/Kolkata"
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        current_rainfall = data.get(
            "current", {}
        ).get("precipitation", 0)

        hourly = data.get("hourly", {})

        times = hourly.get("time", [])
        rainfall = hourly.get("precipitation", [])

        weather_df = pd.DataFrame({
            "time": times,
            "rainfall_mm": rainfall
        })

        weather_df["time"] = pd.to_datetime(
            weather_df["time"]
        )

        weather_df["rainfall_mm"] = pd.to_numeric(
            weather_df["rainfall_mm"],
            errors="coerce"
        ).fillna(0)

        today = datetime.now().date()

        daily_df = weather_df[
            weather_df["time"].dt.date == today
        ].copy()

        return (
            float(current_rainfall or 0),
            weather_df,
            daily_df,
            None
        )

    except Exception as error:
        return 0.0, pd.DataFrame(), pd.DataFrame(), str(error)


current_rainfall, weather_df, daily_df, weather_error = (
    get_weather_data()
)


# ============================================================
# DASHBOARD HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-title">
        🌊  RIVER PREDICTION
    </div>
    <div class="hero-subtitle">
         River Water level prediction dashboard
        | Kerala, India
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌊  River")
st.sidebar.caption(" Water Level Monitoring")

st.sidebar.markdown("---")

st.sidebar.subheader("Dashboard Information")

st.sidebar.info(
    "Enter the latest river and dam measurements "
    "to predict the next day's river level."
)

st.sidebar.markdown("### Weather Information")

if weather_error:
    st.sidebar.warning(
        "Live weather data is unavailable. "
        "You can still enter rainfall manually."
    )
else:
    st.sidebar.metric(
        "Current Precipitation",
        f"{current_rainfall:.1f} mm"
    )

st.sidebar.markdown("---")

st.sidebar.caption(
    "Weather source: Open-Meteo"
)

st.sidebar.caption(
    "Prediction: Multiple Linear Regression"
)

st.sidebar.caption(
    "Refresh interval: 5 minutes"
)


# ============================================================
# INPUTS
# ============================================================

st.markdown(
    '<div class="section-heading">📊 River Parameters</div>',
    unsafe_allow_html=True
)

st.caption(
    "Enter measurements for the river and dam. "
    "The rainfall value is prefilled from the weather API "
    "when available."
)

col1, col2, col3 = st.columns(3)

with col1:
    rainfall_mm = st.number_input(
        "Current Rainfall (mm)",
        min_value=0.0,
        max_value=1000.0,
        value=float(max(0, current_rainfall)),
        step=1.0,
        help="Current precipitation estimate in millimetres."
    )

    rainfall_3d_total_mm = st.number_input(
        "3-Day Rainfall Total (mm)",
        min_value=0.0,
        max_value=5000.0,
        value=0.0,
        step=1.0
    )

    rainfall_7d_total_mm = st.number_input(
        "7-Day Rainfall Total (mm)",
        min_value=0.0,
        max_value=10000.0,
        value=0.0,
        step=1.0
    )

with col2:
    river_level_m = st.number_input(
        "Current River Level (m)",
        min_value=0.0,
        max_value=100.0,
        value=2.0,
        step=0.1
    )

    river_level_change_m = st.number_input(
        "Recent River Level Change (m)",
        min_value=-20.0,
        max_value=20.0,
        value=0.0,
        step=0.1,
        help="Positive means rising; negative means falling."
    )

    dam_storage_pct = st.slider(
        "Dam Storage (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0
    )

with col3:
    dam_inflow_m3s = st.number_input(
        "Dam Inflow (m³/s)",
        min_value=0.0,
        max_value=100000.0,
        value=100.0,
        step=10.0
    )

    dam_outflow_m3s = st.number_input(
        "Dam Outflow (m³/s)",
        min_value=0.0,
        max_value=100000.0,
        value=100.0,
        step=10.0
    )


# ============================================================
# CREATE MODEL INPUT
# ============================================================

input_data = pd.DataFrame([{
    "rainfall_mm": rainfall_mm,
    "rainfall_3d_total_mm": rainfall_3d_total_mm,
    "rainfall_7d_total_mm": rainfall_7d_total_mm,
    "river_level_m": river_level_m,
    "river_level_change_m": river_level_change_m,
    "dam_storage_pct": dam_storage_pct,
    "dam_inflow_m3s": dam_inflow_m3s,
    "dam_outflow_m3s": dam_outflow_m3s
}])

input_data = input_data[FEATURES]


# ============================================================
# PREDICTION
# ============================================================

st.markdown("---")

predict_col, reset_col = st.columns([2, 1])

with predict_col:
    predict_button = st.button(
        "🔮 Predict Next-Day River Level",
        use_container_width=True
    )

with reset_col:
    st.caption(
        "Predictions depend on the values entered above."
    )


if "prediction" not in st.session_state:
    st.session_state.prediction = None

if predict_button:

    try:
        prediction = model.predict(input_data)

        prediction_value = float(
            np.asarray(prediction).reshape(-1)[0]
        )

        if not np.isfinite(prediction_value):
            raise ValueError(
                "The model returned an invalid prediction."
            )

        st.session_state.prediction = prediction_value

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# ============================================================
# DISPLAY PREDICTION
# ============================================================

prediction_value = st.session_state.prediction

if prediction_value is not None:

    st.markdown("---")

    st.markdown(
        '<div class="section-heading">'
        '🎯 Prediction Results'
        '</div>',
        unsafe_allow_html=True
    )

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">
                PREDICTED NEXT-DAY LEVEL
            </div>
            <div class="metric-value">
                {prediction_value:.2f} m
            </div>
            <div class="metric-caption">
                Model estimate
            </div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        difference = prediction_value - river_level_m

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">
                CHANGE FROM ENTERED LEVEL
            </div>
            <div class="metric-value">
                {difference:+.2f} m
            </div>
            <div class="metric-caption">
                Predicted level minus current input
            </div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        if difference > 0:
            trend = "Rising"
        elif difference < 0:
            trend = "Falling"
        else:
            trend = "Unchanged"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">
                PREDICTED TREND
            </div>
            <div class="metric-value">
                {trend}
            </div>
            <div class="metric-caption">
                Based on the entered current level
            </div>
        </div>
        """, unsafe_allow_html=True)


    # ========================================================
    # GAUGE CHART
    # ========================================================

    st.markdown(
        '<div class="section-heading">'
        '🌊 Predicted Water Level'
        '</div>',
        unsafe_allow_html=True
    )

    gauge_max = max(
        10.0,
        abs(prediction_value) * 1.5,
        abs(river_level_m) * 1.5
    )

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prediction_value,
            number={
                "suffix": " m",
                "font": {
                    "color": "#00D4FF",
                    "size": 35
                }
            },
            title={
                "text": "Predicted River Level",
                "font": {
                    "color": "white"
                }
            },
            gauge={
                "axis": {
                    "range": [0, gauge_max],
                    "tickcolor": "white"
                },
                "bar": {
                    "color": "#00D4FF"
                },
                "bgcolor": "#111111",
                "bordercolor": "#333333",
                "steps": [
                    {
                        "range": [0, gauge_max * 0.5],
                        "color": "#10251D"
                    },
                    {
                        "range": [
                            gauge_max * 0.5,
                            gauge_max * 0.8
                        ],
                        "color": "#332A12"
                    },
                    {
                        "range": [
                            gauge_max * 0.8,
                            gauge_max
                        ],
                        "color": "#351414"
                    }
                ]
            }
        )
    )

    gauge.update_layout(
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        height=350,
        margin=dict(l=25, r=25, t=60, b=20)
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )


    # ========================================================
    # CURRENT VS PREDICTED LEVEL
    # ========================================================

    st.markdown(
        '<div class="section-heading">'
        '📈 Current vs Predicted Level'
        '</div>',
        unsafe_allow_html=True
    )

    comparison_df = pd.DataFrame({
        "Measurement": [
            "Entered Current Level",
            "Predicted Next-Day Level"
        ],
        "Water Level (m)": [
            river_level_m,
            prediction_value
        ]
    })

    fig_comparison = px.bar(
        comparison_df,
        x="Measurement",
        y="Water Level (m)",
        color="Measurement",
        color_discrete_sequence=[
            "#007F99",
            "#00D4FF"
        ],
        text="Water Level (m)"
    )

    fig_comparison.update_traces(
        texttemplate="%{text:.2f} m",
        textposition="outside"
    )

    fig_comparison.update_layout(
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font_color="white",
        showlegend=False,
        yaxis_title="Water Level (m)",
        xaxis_title="",
        margin=dict(l=20, r=20, t=30, b=20)
    )

    st.plotly_chart(
        fig_comparison,
        use_container_width=True
    )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-heading">'
        '📋 Prediction Input Summary'
        '</div>',
        unsafe_allow_html=True
    )

    summary_df = pd.DataFrame({
        "Parameter": FEATURES,
        "Entered Value": [
            rainfall_mm,
            rainfall_3d_total_mm,
            rainfall_7d_total_mm,
            river_level_m,
            river_level_change_m,
            dam_storage_pct,
            dam_inflow_m3s,
            dam_outflow_m3s
        ]
    })

    st.dataframe(
        summary_df,
        hide_index=True,
        use_container_width=True
    )


    # ========================================================
    # DOWNLOAD REPORT
    # ========================================================

    report = {
        "project": "Nila River Water Level Prediction",
        "generated_at": datetime.now().isoformat(),
        "prediction_target": "Next-day river level",
        "predicted_river_level_m": prediction_value,
        "entered_current_level_m": river_level_m,
        "predicted_change_m": difference,
        "inputs": {
            feature: float(input_data.iloc[0][feature])
            for feature in FEATURES
        },
        "model": "Multiple Linear Regression"
    }

    st.download_button(
        label="📥 Download Prediction Report (JSON)",
        data=json.dumps(report, indent=4),
        file_name="nila_river_prediction.json",
        mime="application/json",
        use_container_width=True
    )


# ============================================================
# RAINFALL CHART
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-heading">'
    '🌧️ Rainfall Monitoring'
    '</div>',
    unsafe_allow_html=True
)

if not weather_df.empty:

    weather_df["date"] = (
        weather_df["time"].dt.strftime("%d %b")
    )

    daily_rainfall = (
        weather_df
        .groupby("date", sort=False)["rainfall_mm"]
        .sum()
        .reset_index()
    )

    fig_rain = px.bar(
        daily_rainfall,
        x="date",
        y="rainfall_mm",
        labels={
            "date": "Date",
            "rainfall_mm": "Rainfall (mm)"
        },
        title="Hourly Weather Data Aggregated by Day",
        color_discrete_sequence=["#00A6CC"]
    )

    fig_rain.update_layout(
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font_color="white",
        xaxis_title="Date",
        yaxis_title="Rainfall (mm)",
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(
        fig_rain,
        use_container_width=True
    )

    st.caption(
        "Rainfall is estimated weather data for the configured "
        "coordinates. It is not a verified river-gauge measurement."
    )

else:

    st.warning(
        "Rainfall chart unavailable because live weather data "
        "could not be retrieved. Please check your internet "
        "connection and try again."
    )


# ============================================================
# IMPORTANT DISCLAIMER
# ============================================================

st.markdown("---")

st.warning(
    "Research prototype only: This model uses your trained "
    "machine-learning model and the values entered by the user. "
    "The model's accuracy depends on its training data. "
    "Do not use its predictions for real flood warnings or "
    "emergency decisions without validation against reliable "
    "river-gauge measurements and official warnings."
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    Nila River Water Level Prediction |
    AI & Machine Learning Mini Project
    <br>
    Built with Python, Streamlit and Multiple Linear Regression
</div>
""", unsafe_allow_html=True)


