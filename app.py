import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import plotly.graph_objects as go
import plotly.express as px
import json

from pathlib import Path
from datetime import datetime
from streamlit_autorefresh import st_autorefresh


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Nila River Prediction",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Refresh every 5 minutes
st_autorefresh(
    interval=5 * 60 * 1000,
    key="river_refresh"
)


# =========================================================
# BLACK THEME & DYNAMIC ANIMATED CSS
# =========================================================

st.markdown("""
<style>

/* Main application background */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background-color: #000000;
    color: #FFFFFF;
}

/* Header */
[data-testid="stHeader"] {
    background-color: #000000;
}

/* Left sidebar */
[data-testid="stSidebar"] {
    background-color: #080808;
    border-right: 1px solid #222222;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

/* Main headings */
h1, h2, h3, h4, h5 {
    color: #FFFFFF !important;
}

/* General text */
.stApp p,
.stApp label {
    color: #D1D5DB;
}

/* Main animated flowing river hero banner */
.hero {
    position: relative;
    background: linear-gradient(135deg, #020813, #001B2B);
    border: 1px solid #06445A;
    border-radius: 18px;
    padding: 32px 28px;
    margin-bottom: 22px;
    overflow: hidden;
    z-index: 1;
}

.hero-content {
    position: relative;
    z-index: 10;
}

.hero-title {
    font-size: 34px;
    font-weight: 800;
    color: #FFFFFF !important;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.8);
}

.hero-subtitle {
    color: #E2E8F0;
    font-size: 15px;
    margin-top: 8px;
    text-shadow: 0 1px 4px rgba(0, 0, 0, 0.8);
}

/* Animated Water Waves Layered Background */
.hero::after {
    content: "";
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 100%;
    background: url('data:image/svg+xml;utf8,<svg viewBox="0 0 1200 120" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none"><path d="M0,0 C150,90 350,-40 500,40 C650,120 900,10 1200,50 L1200,120 L0,120 Z" fill="%2300D4FF" fill-opacity="0.12"/></svg>') repeat-x;
    background-size: 1000px 100%;
    animation: waveAnimation 14s linear infinite;
    z-index: 2;
    pointer-events: none;
}

.hero::before {
    content: "";
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 85%;
    background: url('data:image/svg+xml;utf8,<svg viewBox="0 0 1200 120" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="none"><path d="M0,30 C200,110 400,0 600,60 C800,120 1000,20 1200,60 L1200,120 L0,120 Z" fill="%23007F99" fill-opacity="0.18"/></svg>') repeat-x;
    background-size: 800px 100%;
    animation: waveAnimationFast 8s linear infinite;
    z-index: 3;
    pointer-events: none;
}

@keyframes waveAnimation {
    0% { background-position-x: 0px; }
    100% { background-position-x: 1000px; }
}

@keyframes waveAnimationFast {
    0% { background-position-x: 0px; }
    100% { background-position-x: -800px; }
}

/* Pulsing Emergency Warning Banner */
@keyframes pulseGlow {
    0% {
        box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.6);
        border-color: rgba(239, 68, 68, 0.8);
    }
    70% {
        box-shadow: 0 0 0 14px rgba(239, 68, 68, 0);
        border-color: rgba(239, 68, 68, 1);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(239, 68, 68, 0);
        border-color: rgba(239, 68, 68, 0.8);
    }
}

.emergency-alert {
    background: linear-gradient(135deg, #2b0c0c, #1a0505);
    border: 2px solid #ef4444;
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 20px;
    margin-bottom: 20px;
    animation: pulseGlow 2s infinite;
}

.emergency-title {
    color: #fca5a5;
    font-weight: 800;
    font-size: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.emergency-text {
    color: #fca5a5;
    font-size: 13px;
    margin-top: 6px;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, #111111, #080808);
    border: 1px solid #292929;
    border-radius: 14px;
    padding: 20px;
    min-height: 115px;
}

.metric-title {
    color: #A3A3A3;
    font-size: 13px;
}

.metric-value {
    color: #00D4FF;
    font-size: 28px;
    font-weight: 800;
    margin-top: 8px;
}

.metric-caption {
    color: #737373;
    font-size: 12px;
    margin-top: 6px;
}

/* Section headings */
.section-title {
    color: #00D4FF;
    font-size: 22px;
    font-weight: 700;
    margin-top: 22px;
    margin-bottom: 12px;
}

/* Input fields */
.stNumberInput input,
.stTextInput input {
    background-color: #151515;
    color: #FFFFFF;
}

/* Dropdowns */
[data-baseweb="select"] > div {
    background-color: #151515;
    color: #FFFFFF;
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

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background-color: #080808;
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

/* Divider */
hr {
    border-color: #292929;
}

/* Footer */
.footer {
    text-align: center;
    color: #737373;
    font-size: 12px;
    padding: 20px;
}

/* Mobile layout */
@media (max-width: 640px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
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


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = Path(__file__).parent / "nila_river_mlr.pkl"

LATITUDE = 11.8745
LONGITUDE = 75.3704

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


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model(model_path, modified_time):
    return joblib.load(model_path)


if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Please make sure "
        "'nila_river_mlr.pkl' is in the same GitHub folder as app.py."
    )
    st.stop()

try:
    model = load_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime
    )
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()


# =========================================================
# FETCH LIVE WEATHER DATA
# =========================================================

@st.cache_data(ttl=300)
def get_weather_data(latitude, longitude):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "precipitation",
        "hourly": "precipitation",
        "past_days": 7,
        "forecast_days": 1,
        "timezone": "Asia/Kolkata"
    }

    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()

        current = data.get("current", {})
        hourly = data.get("hourly", {})

        current_rainfall = float(current.get("precipitation") or 0)

        weather_df = pd.DataFrame({
            "time": hourly.get("time", []),
            "rainfall_mm": hourly.get("precipitation", [])
        })

        if not weather_df.empty:
            weather_df["time"] = pd.to_datetime(weather_df["time"])
            weather_df["rainfall_mm"] = pd.to_numeric(
                weather_df["rainfall_mm"], errors="coerce"
            ).fillna(0)

        return current_rainfall, weather_df, None

    except Exception as e:
        return 0.0, pd.DataFrame(), str(e)


current_rainfall, weather_df, weather_error = get_weather_data(
    LATITUDE,
    LONGITUDE
)


# =========================================================
# LEFT SIDEBAR
# =========================================================

st.sidebar.markdown("# 🌊  RIVER")
st.sidebar.caption("Water Level Prediction System")

st.sidebar.markdown("---")

st.sidebar.markdown("### 📍 Location")
st.sidebar.write("Latitude")
st.sidebar.code(str(LATITUDE))
st.sidebar.write("Longitude")
st.sidebar.code(str(LONGITUDE))

st.sidebar.markdown("---")

st.sidebar.markdown("### 🌧️ Live Rainfall")

if weather_error:
    st.sidebar.warning("Live weather data unavailable.")
else:
    st.sidebar.metric(
        "Current precipitation",
        f"{current_rainfall:.1f} mm"
    )

st.sidebar.markdown("---")

st.sidebar.markdown("### 📊 River Inputs")

rainfall_mm = st.sidebar.number_input(
    "Current rainfall (mm)",
    min_value=0.0,
    max_value=1000.0,
    value=float(max(0, current_rainfall)),
    step=1.0
)

rainfall_3d_total_mm = st.sidebar.number_input(
    "3-day rainfall total (mm)",
    min_value=0.0,
    max_value=5000.0,
    value=0.0,
    step=1.0
)

rainfall_7d_total_mm = st.sidebar.number_input(
    "7-day rainfall total (mm)",
    min_value=0.0,
    max_value=10000.0,
    value=0.0,
    step=1.0
)

river_level_m = st.sidebar.number_input(
    "Current river level (m)",
    min_value=0.0,
    max_value=100.0,
    value=2.0,
    step=0.1
)

river_level_change_m = st.sidebar.number_input(
    "Recent level change (m)",
    min_value=-20.0,
    max_value=20.0,
    value=0.0,
    step=0.1
)

dam_storage_pct = st.sidebar.slider(
    "Dam storage (%)",
    min_value=0.0,
    max_value=100.0,
    value=50.0,
    step=1.0
)

dam_inflow_m3s = st.sidebar.number_input(
    "Dam inflow (m³/s)",
    min_value=0.0,
    max_value=100000.0,
    value=100.0,
    step=10.0
)

dam_outflow_m3s = st.sidebar.number_input(
    "Dam outflow (m³/s)",
    min_value=0.0,
    max_value=100000.0,
    value=100.0,
    step=10.0
)

st.sidebar.markdown("---")
st.sidebar.caption("Model: Multiple Linear Regression")


# =========================================================
# MAIN DASHBOARD HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <div class="hero-content">
        <div class="hero-title">
            🌊  RIVER WATER LEVEL PREDICTION
        </div>
        <div class="hero-subtitle">
            Intelligent Water Level Monitoring & Next-Day Prediction
            <br>
            Kerala, India | Machine Learning Dashboard
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# DYNAMIC STATUS & WEATHER SEVERITY BADGE
# =========================================================

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    st.markdown(
        f"**🕒 Dashboard time:** "
        f"{datetime.now().strftime('%d %b %Y, %I:%M:%S %p')}"
    )

with status_col2:
    if weather_error:
        st.markdown("**🌧️ Weather API:** ⚠️ Unavailable")
    else:
        st.markdown("**🌧️ Weather API:** 🟢 Connected")

with status_col3:
    if current_rainfall > 20:
        st.markdown("**⚡ Severity:** 🔴 Heavy Rainfall Alert")
    elif current_rainfall > 5:
        st.markdown("**⚡ Severity:** 🟡 Moderate Rain")
    else:
        st.markdown("**⚡ Severity:** 🟢 Normal Conditions")


# =========================================================
# BUILD MODEL INPUT
# =========================================================

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


# =========================================================
# PREDICTION BUTTON & LOGIC
# =========================================================

st.markdown(
    '<div class="section-title">🎯 Water Level Prediction</div>',
    unsafe_allow_html=True
)

if st.button(
    "🔮 Predict Next-Day Water Level",
    use_container_width=True
):
    try:
        result = model.predict(input_data)
        prediction = float(np.asarray(result).reshape(-1)[0])

        if not np.isfinite(prediction):
            raise ValueError("The model returned an invalid value.")

        st.session_state["river_prediction"] = prediction
        st.session_state["river_prediction_inputs"] = input_data.copy()

    except Exception as e:
        st.error(f"Prediction error: {e}")


# =========================================================
# DISPLAY PREDICTION RESULTS
# =========================================================

prediction = st.session_state.get("river_prediction")

if prediction is not None:
    previous_inputs = st.session_state.get(
        "river_prediction_inputs",
        input_data
    )

    previous_level = float(previous_inputs.iloc[0]["river_level_m"])
    difference = prediction - previous_level

    if difference > 0.05:
        trend = "Rising Fast 🚀" if difference > 0.2 else "Rising ↗"
    elif difference < -0.05:
        trend = "Falling ↘"
    else:
        trend = "Stable →"

    st.markdown("---")

    st.markdown(
        '<div class="section-title">📊 Prediction Summary</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">PREDICTED NEXT-DAY LEVEL</div>
            <div class="metric-value">{prediction:.2f} m</div>
            <div class="metric-caption">Machine-learning estimate</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">ESTIMATED LEVEL CHANGE</div>
            <div class="metric-value">{difference:+.2f} m</div>
            <div class="metric-caption">Compared with current level</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">PREDICTED TREND</div>
            <div class="metric-value">{trend}</div>
            <div class="metric-caption">Based on model projection</div>
        </div>
        """, unsafe_allow_html=True)

    # Dynamic Water Hazard Warning Signal Banner (>= 4.0 m triggers Red Alert)
    if prediction >= 4.0:
        st.markdown("""
        <div class="emergency-alert">
            <div class="emergency-title">
                🚨 CRITICAL FLOOD WARNING SIGNAL ACTIVE
            </div>
            <div class="emergency-text">
                Predicted water level exceeds critical safety thresholds (≥ 4.0m). Immediate downstream monitoring advised.
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif prediction >= 3.0:
        st.markdown("""
        <div class="emergency-alert" style="border-color: #f59e0b; background: linear-gradient(135deg, #2b1d0c, #1a1005);">
            <div class="emergency-title" style="color: #fde68a;">
                ⚠️ ELEVATED WATER LEVEL ADVISORY
            </div>
            <div class="emergency-text" style="color: #fde68a;">
                Water level is projected to rise significantly. Keep continuous watch on local inflows.
            </div>
        </div>
        """, unsafe_allow_html=True)


    # =====================================================
    # GAUGE (FIXED COLOR THRESHOLDS: 4m+ IS RED)
    # =====================================================

    st.markdown(
        '<div class="section-title">🌊 Water Level Gauge</div>',
        unsafe_allow_html=True
    )

    gauge_max = max(10.0, prediction * 1.5, previous_level * 1.5)

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prediction,
            number={
                "suffix": " m",
                "font": {"color": "#00D4FF", "size": 36}
            },
            title={"text": "Predicted River Level", "font": {"color": "white"}},
            gauge={
                "axis": {"range": [0, gauge_max], "tickcolor": "white"},
                "bar": {"color": "#00D4FF"},
                "bgcolor": "#111111",
                "bordercolor": "#333333",
                "steps": [
                    {"range": [0, 3.0], "color": "#10251D"},       # Safe Zone (0 - 3m)
                    {"range": [3.0, 4.0], "color": "#332A12"},     # Caution Zone (3 - 4m)
                    {"range": [4.0, gauge_max], "color": "#351414"}  # Critical Red Zone (4m+)
                ]
            }
        )
    )

    gauge.update_layout(
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        height=330,
        margin=dict(l=25, r=25, t=60, b=20)
    )

    st.plotly_chart(gauge, use_container_width=True)


    # =====================================================
    # CURRENT VS PREDICTED GRAPH
    # =====================================================

    st.markdown(
        '<div class="section-title">📈 Current vs Predicted River Level</div>',
        unsafe_allow_html=True
    )

    comparison_df = pd.DataFrame({
        "Measurement": ["Current Level", "Predicted Level"],
        "Water Level (m)": [previous_level, prediction]
    })

    comparison_fig = px.bar(
        comparison_df,
        x="Measurement",
        y="Water Level (m)",
        color="Measurement",
        text="Water Level (m)",
        color_discrete_sequence=["#007F99", "#00D4FF"]
    )

    comparison_fig.update_traces(
        texttemplate="%{text:.2f} m",
        textposition="outside"
    )

    comparison_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font_color="white",
        showlegend=False,
        yaxis_title="Water Level (m)",
        xaxis_title="",
        margin=dict(l=20, r=20, t=30, b=20)
    )

    st.plotly_chart(comparison_fig, use_container_width=True)


    # =====================================================
    # INTERACTIVE WHAT-IF SCENARIO SIMULATOR
    # =====================================================

    st.markdown(
        '<div class="section-title">🎛️ Interactive Scenario Simulator</div>',
        unsafe_allow_html=True
    )
    st.caption("Quickly test how different rainfall spikes or dam releases impact the water level prediction.")

    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        sim_rainfall = st.slider("Simulated Rainfall (mm)", 0.0, 300.0, float(rainfall_mm), 5.0)
    with sim_col2:
        sim_dam_pct = st.slider("Simulated Dam Storage (%)", 0.0, 100.0, float(dam_storage_pct), 1.0)
    with sim_col3:
        sim_outflow = st.slider("Simulated Dam Outflow (m³/s)", 0.0, 5000.0, float(dam_outflow_m3s), 50.0)

    sim_input = previous_inputs.copy()
    sim_input.loc[0, "rainfall_mm"] = sim_rainfall
    sim_input.loc[0, "dam_storage_pct"] = sim_dam_pct
    sim_input.loc[0, "dam_outflow_m3s"] = sim_outflow
    sim_prediction = float(np.asarray(model.predict(sim_input[FEATURES])).reshape(-1)[0])

    st.info(f"💡 **Simulated Next-Day Level:** `{sim_prediction:.2f} m` (Compared to baseline prediction of `{prediction:.2f} m`)")


    # =====================================================
    # SENSITIVITY EXPLORER
    # =====================================================

    st.markdown(
        '<div class="section-title">🧪 Rainfall Sensitivity Explorer</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Explore how changing current rainfall affects predictions while keeping other factors constant."
    )

    rainfall_range = st.slider(
        "Explore rainfall range (mm)",
        min_value=0,
        max_value=300,
        value=(0, 100),
        step=5
    )

    rainfall_values = np.linspace(rainfall_range[0], rainfall_range[1], 20)
    sensitivity_predictions = []

    for rainfall_value in rainfall_values:
        test_input = previous_inputs.copy()
        test_input.loc[test_input.index[0], "rainfall_mm"] = rainfall_value
        test_prediction = float(
            np.asarray(model.predict(test_input[FEATURES])).reshape(-1)[0]
        )
        sensitivity_predictions.append(test_prediction)

    sensitivity_df = pd.DataFrame({
        "Rainfall (mm)": rainfall_values,
        "Predicted Level (m)": sensitivity_predictions
    })

    sensitivity_fig = px.line(
        sensitivity_df,
        x="Rainfall (mm)",
        y="Predicted Level (m)",
        markers=True
    )

    sensitivity_fig.update_traces(
        line=dict(color="#00D4FF", width=3),
        marker=dict(size=6)
    )

    sensitivity_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font_color="white"
    )

    st.plotly_chart(sensitivity_fig, use_container_width=True)


    # =====================================================
    # DOWNLOAD REPORT
    # =====================================================

    report = {
        "project": "Nila River Water Level Prediction",
        "generated_at": datetime.now().isoformat(),
        "predicted_next_day_level_m": prediction,
        "entered_current_level_m": previous_level,
        "estimated_change_m": difference,
        "model": "Multiple Linear Regression",
        "inputs": {
            feature: float(previous_inputs.iloc[0][feature])
            for feature in FEATURES
        }
    }

    st.download_button(
        "📥 Download Prediction Report",
        data=json.dumps(report, indent=4),
        file_name="nila_river_prediction.json",
        mime="application/json",
        use_container_width=True
    )


# =========================================================
# RAINFALL GRAPH
# =========================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">🌧️ Rainfall Monitoring</div>',
    unsafe_allow_html=True
)

if not weather_df.empty:
    chart_df = weather_df.copy()
    chart_df["Date"] = chart_df["time"].dt.date
    daily_rainfall = chart_df.groupby("Date", as_index=False)["rainfall_mm"].sum()

    rainfall_fig = px.bar(
        daily_rainfall,
        x="Date",
        y="rainfall_mm",
        title="Rainfall Over Time",
        labels={"rainfall_mm": "Rainfall (mm)"},
        color_discrete_sequence=["#00A6CC"]
    )

    rainfall_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        font_color="white",
        xaxis_title="Date",
        yaxis_title="Rainfall (mm)"
    )

    st.plotly_chart(rainfall_fig, use_container_width=True)
else:
    st.warning("Live rainfall graph is unavailable.")


# =========================================================
# COMMUNITY OBSERVATIONS & FIELD REPORTS
# =========================================================

st.markdown("---")
st.markdown('<div class="section-title">🗣️ Community & Field Observations</div>', unsafe_allow_html=True)
st.caption("Notice something at the riverbank? Log a quick field report for other dashboard viewers.")

if "community_reports" not in st.session_state:
    st.session_state["community_reports"] = []

with st.form("community_form"):
    reporter_name = st.text_input("Your Name / Station ID")
    observation_text = st.text_input("Observation (e.g., Water level rising near Shoranur bridge)")
    report_submitted = st.form_submit_button("Submit Field Report")
    
    if report_submitted and observation_text:
        st.session_state["community_reports"].insert(0, {
            "time": datetime.now().strftime("%I:%M %p, %d %b"),
            "name": reporter_name if reporter_name else "Anonymous",
            "text": observation_text
        })
        st.success("Report added successfully!")

if st.session_state["community_reports"]:
    st.markdown("**Recent Field Reports:**")
    for rep in st.session_state["community_reports"][:3]:
        st.markdown(f"> *{rep['time']}* — **{rep['name']}**: {rep['text']}")


# =========================================================
# EMERGENCY GUIDELINES EXPANDER
# =========================================================

with st.expander("🚨 Emergency Guidelines & Helpline Contacts"):
    st.markdown("""
    * **State Emergency Operations Center:** 1077 / 1070
    * **District Disaster Management Authority:** Keep local taluk office contacts handy.
    * **Safety Checklist:**
      1. Move electronic appliances and valuables to higher floors if flood alerts trigger.
      2. Keep emergency kits (drinking water, flashlight, first aid, essential medicines) ready.
      3. Avoid crossing overflowing bridges or streams.
    """)


# =========================================================
# INPUT SUMMARY & DISCLAIMER
# =========================================================

with st.expander("📋 View Model Input Values"):
    st.dataframe(input_data, use_container_width=True, hide_index=True)

st.markdown("---")

st.warning(
    "Research prototype only. Weather data is an estimate for "
    "the configured coordinates, not a verified river-gauge reading. "
    "The model's accuracy depends on its training data. Do not use "
    "this prototype for actual flood warnings or emergency decisions."
)

st.markdown("""
<div class="footer">
     River Water Level Prediction System
    <br>
    Built with Python, Streamlit, Plotly and Machine Learning
</div>
""", unsafe_allow_html=True)
