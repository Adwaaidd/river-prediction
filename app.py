import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import plotly.graph_objects as go
import plotly.express as px
import json

from pathlib import Path
from datetime import datetime
from streamlit_autorefresh import st_autorefresh


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title=" River Predictor",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
    st.markdown("""
<style>
/* Main app background */
.stApp {
    background-color: #000000;
    color: #FFFFFF;
}

/* Main content area */
[data-testid="stAppViewContainer"] {
    background-color: #000000;
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #0A0A0A;
}

/* Headings and text */
h1, h2, h3, h4, p, label, span {
    color: #FFFFFF;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #111111;
    border: 1px solid #292929;
    padding: 15px;
    border-radius: 12px;
}

/* Input fields */
.stNumberInput input,
.stTextInput input {
    background-color: #151515;
    color: #FFFFFF;
}

/* Dropdowns */
.stSelectbox div[data-baseweb="select"] {
    background-color: #151515;
    color: #FFFFFF;
}

/* Buttons */
.stButton > button {
    background-color: #1DB954;
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
}

.stButton > button:hover {
    background-color: #169C46;
    color: #FFFFFF;
    border: 1px solid #1DB954;
}

/* Dividers */
hr {
    border-color: #333333;
}
</style>
""", unsafe_allow_html=True)
)

# Refresh the dashboard every 5 minutes
st_autorefresh(
    interval=5 * 60 * 1000,
    key="nila_refresh"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #f3f7fc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(120deg, #073b66, #087eae);
        padding: 28px;
        border-radius: 18px;
        color: white;
        margin-bottom: 22px;
    }

    .hero h1 {
        color: white;
        margin-bottom: 5px;
    }

    .hero p {
        color: #e1f5ff;
        font-size: 16px;
    }

    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #dce8f4;
        box-shadow: 0 3px 12px rgba(0, 40, 80, 0.04);
        min-height: 130px;
    }

    .metric-title {
        color: #60758a;
        font-size: 14px;
    }

    .metric-value {
        color: #073b66;
        font-size: 27px;
        font-weight: 750;
        margin-top: 8px;
    }

    .metric-caption {
        color: #718096;
        font-size: 12px;
        margin-top: 5px;
    }

    .section-title {
        color: #073b66;
        font-weight: 700;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }

    div[data-testid="stSidebar"] {
        background-color: #e8f2fb;
    }

    @media (max-width: 640px) {
        .hero {
            padding: 18px;
        }

        .metric-value {
            font-size: 22px;
        }
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# CONSTANTS
# --------------------------------------------------

LATITUDE = 10.77
LONGITUDE = 76.38
TIMEZONE = "Asia/Kolkata"

MODEL_PATH = Path(__file__).resolve().parent / "nila_river_mlr.pkl"

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


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Upload nila_river_mlr.pkl to the repository."
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except Exception as error:
    st.error("The machine learning model could not be loaded.")
    st.code(str(error))
    st.info(
        "Check that nila_river_mlr.pkl is in the same "
        "GitHub folder as app.py."
    )
    st.stop()


# --------------------------------------------------
# FETCH WEATHER / RAINFALL ESTIMATES
# --------------------------------------------------

@st.cache_data(ttl=300)
def get_weather():
    endpoint = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "current": "precipitation",
        "hourly": "precipitation",
        "past_days": 7,
        "forecast_days": 1,
        "timezone": TIMEZONE
    }

    response = requests.get(
        endpoint,
        params=params,
        timeout=15
    )

    response.raise_for_status()
    data = response.json()

    current = data.get("current", {})
    hourly = data.get("hourly", {})

    times = hourly.get("time", [])
    rainfall = hourly.get("precipitation", [])

    weather_df = pd.DataFrame({
        "time": pd.to_datetime(times),
        "rainfall": pd.to_numeric(
            pd.Series(rainfall),
            errors="coerce"
        )
    }).dropna()

    now = pd.Timestamp.now(tz=TIMEZONE).tz_localize(None)

    past = weather_df[weather_df["time"] <= now]

    # Exclude the current, potentially incomplete hour
    completed = past[past["time"] < now.floor("h")]

    rainfall_3d = completed.tail(72)["rainfall"].sum()
    rainfall_7d = completed.tail(168)["rainfall"].sum()

    current_rainfall = float(
        current.get("precipitation", 0) or 0
    )

    return {
        "current_rainfall": current_rainfall,
        "rainfall_3d": float(rainfall_3d),
        "rainfall_7d": float(rainfall_7d),
        "updated": current.get("time", "Unknown"),
        "hourly_df": weather_df
    }


try:
    weather = get_weather()
    weather_available = True

except Exception as error:
    weather = None
    weather_available = False
    weather_error = str(error)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>🌊 Nila River Predictor</h1>
    <p>
        Smart river-level exploration using rainfall,
        river conditions and dam parameters.
    </p>
    <span>📍 Approximate monitoring coordinates:
    10.77° N, 76.38° E</span>
</div>
""", unsafe_allow_html=True)


st.caption(
    "Experimental ML dashboard · Weather data is model-estimated, "
    "not a direct river-gauge measurement."
)


# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------

st.sidebar.title("🎛️ Prediction Controls")

st.sidebar.write(
    "Adjust the inputs to explore how the model's "
    "predicted next-day river level changes."
)

if weather_available:
    default_rain = weather["current_rainfall"]
    default_3d = weather["rainfall_3d"]
    default_7d = weather["rainfall_7d"]
else:
    default_rain = 0.0
    default_3d = 0.0
    default_7d = 0.0

st.sidebar.subheader("🌧️ Rainfall")

rainfall_mm = st.sidebar.slider(
    "Current hourly rainfall estimate (mm)",
    min_value=0.0,
    max_value=200.0,
    value=float(np.clip(default_rain, 0, 200)),
    step=0.5
)

rainfall_3d_total_mm = st.sidebar.slider(
    "Rainfall over the last 3 days (mm)",
    min_value=0.0,
    max_value=1000.0,
    value=float(np.clip(default_3d, 0, 1000)),
    step=1.0
)

rainfall_7d_total_mm = st.sidebar.slider(
    "Rainfall over the last 7 days (mm)",
    min_value=0.0,
    max_value=2000.0,
    value=float(np.clip(default_7d, 0, 2000)),
    step=1.0
)

st.sidebar.subheader("🌊 River Conditions")

river_level_m = st.sidebar.slider(
    "Current river level (m)",
    min_value=0.0,
    max_value=20.0,
    value=3.2,
    step=0.1
)

river_level_change_m = st.sidebar.slider(
    "Recent river-level change (m)",
    min_value=-5.0,
    max_value=5.0,
    value=0.1,
    step=0.1,
    help="Positive means rising; negative means falling."
)

st.sidebar.subheader("🏞️ Dam Parameters")

dam_storage_pct = st.sidebar.slider(
    "Dam storage (%)",
    min_value=0.0,
    max_value=100.0,
    value=65.0,
    step=1.0
)

dam_inflow_m3s = st.sidebar.slider(
    "Dam inflow (m³/s)",
    min_value=0.0,
    max_value=10000.0,
    value=250.0,
    step=10.0
)

dam_outflow_m3s = st.sidebar.slider(
    "Dam outflow (m³/s)",
    min_value=0.0,
    max_value=10000.0,
    value=200.0,
    step=10.0
)

st.sidebar.divider()

if st.sidebar.button(
    "🔄 Refresh rainfall data",
    use_container_width=True
):
    get_weather.clear()
    st.rerun()


# --------------------------------------------------
# WEATHER STATUS
# --------------------------------------------------

if weather_available:
    st.success(
        f"🌦️ Weather API connected · "
        f"Latest reported time: {weather['updated']}"
    )
else:
    st.warning(
        "Weather data is currently unavailable. "
        "Rainfall defaults are set to zero; adjust them "
        "manually or try refreshing the data."
    )

    with st.expander("Weather connection details"):
        st.code(weather_error)


# --------------------------------------------------
# MODEL PREDICTION
# --------------------------------------------------

input_values = {
    "rainfall_mm": rainfall_mm,
    "rainfall_3d_total_mm": rainfall_3d_total_mm,
    "rainfall_7d_total_mm": rainfall_7d_total_mm,
    "river_level_m": river_level_m,
    "river_level_change_m": river_level_change_m,
    "dam_storage_pct": dam_storage_pct,
    "dam_inflow_m3s": dam_inflow_m3s,
    "dam_outflow_m3s": dam_outflow_m3s
}

input_df = pd.DataFrame(
    [[input_values[feature] for feature in FEATURES]],
    columns=FEATURES
)

try:
    prediction = float(model.predict(input_df)[0])

    if not np.isfinite(prediction):
        raise ValueError("The model returned an invalid number.")

except Exception as error:
    st.error("Prediction failed. Check the model and its input features.")
    st.code(str(error))
    st.stop()


level_change = prediction - river_level_m


# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------

def metric_card(title, value, caption):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown(
    '<h3 class="section-title">📊 Monitoring Overview</h3>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    metric_card(
        "🌧️ Current rainfall",
        f"{rainfall_mm:.1f} mm",
        "Current-hour input"
    )

with c2:
    metric_card(
        "🌊 Current river level",
        f"{river_level_m:.2f} m",
        "User-provided input"
    )

with c3:
    metric_card(
        "🔮 Predicted next-day level",
        f"{prediction:.2f} m",
        "ML model estimate"
    )

with c4:
    metric_card(
        "📈 Predicted change",
        f"{level_change:+.2f} m",
        "Predicted minus current level"
    )


# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "🌊 Water Gauge",
    "📈 Graphs",
    "🌧️ Rainfall",
    "📋 Input Details"
])


# --------------------------------------------------
# TAB 1: WATER GAUGE
# --------------------------------------------------

with tab1:

    st.subheader("Next-Day Water-Level Gauge")

    st.caption(
        "The gauge is a visual representation of the model output, "
        "not an official flood threshold."
    )

    gauge_max = max(10.0, prediction * 1.25, river_level_m * 1.25)

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=prediction,
            number={"suffix": " m", "valueformat": ".2f"},
            delta={
                "reference": river_level_m,
                "valueformat": ".2f",
                "suffix": " m"
            },
            title={"text": "Predicted Next-Day River Level"},
            gauge={
                "axis": {
                    "range": [0, gauge_max]
                },
                "bar": {
                    "color": "#087eae"
                },
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#dce8f4"
            }
        )
    )

    gauge.update_layout(
        height=330,
        margin=dict(l=25, r=25, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#073b66"}
    )

    st.plotly_chart(gauge, use_container_width=True)

    if level_change > 0:
        st.info(
            f"The model predicts a rise of {level_change:.2f} m "
            "relative to the entered current level."
        )
    elif level_change < 0:
        st.info(
            f"The model predicts a fall of "
            f"{abs(level_change):.2f} m relative to the "
            "entered current level."
        )
    else:
        st.info("The predicted level equals the entered current level.")

    st.warning(
        "This is an experimental prediction from a model trained "
        "on synthetic data. It is not a validated flood forecast. "
        "Do not use it for emergency decisions."
    )


# --------------------------------------------------
# TAB 2: GRAPHS
# --------------------------------------------------

with tab2:

    st.subheader("Current vs Predicted Level")

    comparison_df = pd.DataFrame({
        "Measurement": [
            "Entered current level",
            "Predicted next-day level"
        ],
        "River level (m)": [
            river_level_m,
            prediction
        ]
    })

    bar_fig = px.bar(
        comparison_df,
        x="Measurement",
        y="River level (m)",
        color="Measurement",
        text="River level (m)",
        title="Current and Predicted River Levels"
    )

    bar_fig.update_traces(
        texttemplate="%{text:.2f} m",
        textposition="outside"
    )

    bar_fig.update_layout(
        showlegend=False,
        yaxis_title="River level (m)",
        xaxis_title="",
        margin=dict(l=15, r=15, t=60, b=20)
    )

    st.plotly_chart(bar_fig, use_container_width=True)

    st.subheader("Input Sensitivity Explorer")

    st.write(
        "Explore how rainfall affects the prediction while keeping "
        "all other inputs fixed."
    )

    rainfall_range = np.linspace(0, 200, 41)

    sensitivity_df = pd.DataFrame({
        "rainfall_mm": rainfall_range,
        "rainfall_3d_total_mm": rainfall_3d_total_mm,
        "rainfall_7d_total_mm": rainfall_7d_total_mm,
        "river_level_m": river_level_m,
        "river_level_change_m": river_level_change_m,
        "dam_storage_pct": dam_storage_pct,
        "dam_inflow_m3s": dam_inflow_m3s,
        "dam_outflow_m3s": dam_outflow_m3s
    })[FEATURES]

    try:
        sensitivity_df["Predicted level (m)"] = model.predict(
            sensitivity_df[FEATURES]
        )

        sensitivity_fig = px.line(
            sensitivity_df,
            x="rainfall_mm",
            y="Predicted level (m)",
            markers=True,
            title="Model Output vs Current Rainfall Input"
        )

        sensitivity_fig.add_vline(
            x=rainfall_mm,
            line_dash="dash",
            annotation_text="Selected rainfall"
        )

        sensitivity_fig.update_layout(
            xaxis_title="Current rainfall input (mm)",
            yaxis_title="Predicted next-day level (m)"
        )

        st.plotly_chart(
            sensitivity_fig,
            use_container_width=True
        )

    except Exception as error:
        st.error("Could not generate the sensitivity graph.")
        st.code(str(error))


# --------------------------------------------------
# TAB 3: RAINFALL
# --------------------------------------------------

with tab3:

    st.subheader("Rainfall Information")

    if weather_available:

        st.write(
            "Source: Open-Meteo weather model. "
            "These are weather estimates, not local rain-gauge readings."
        )

        r1, r2, r3 = st.columns(3)

        with r1:
            metric_card(
                "Current-hour precipitation",
                f"{weather['current_rainfall']:.1f} mm",
                "Weather API estimate"
            )

        with r2:
            metric_card(
                "Last 3 days",
                f"{weather['rainfall_3d']:.1f} mm",
                "Sum of completed hourly estimates"
            )

        with r3:
            metric_card(
                "Last 7 days",
                f"{weather['rainfall_7d']:.1f} mm",
                "Sum of completed hourly estimates"
            )

        weather_df = weather["hourly_df"].copy()

        if not weather_df.empty:

            now = pd.Timestamp.now(tz=TIMEZONE).tz_localize(None)

            recent_df = weather_df[
                weather_df["time"] <= now
            ].tail(168)

            rain_fig = px.area(
                recent_df,
                x="time",
                y="rainfall",
                title="Hourly Precipitation Estimates"
            )

            rain_fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Precipitation (mm)"
            )

            st.plotly_chart(
                rain_fig,
                use_container_width=True
            )

            st.download_button(
                "📥 Download rainfall data",
                data=recent_df.to_csv(index=False).encode("utf-8"),
                file_name="nila_rainfall_data.csv",
                mime="text/csv"
            )

    else:
        st.info(
            "Rainfall charts will appear when the weather API "
            "becomes available."
        )


# --------------------------------------------------
# TAB 4: INPUT DETAILS
# --------------------------------------------------

with tab4:

    st.subheader("Model Input Summary")

    input_display = pd.DataFrame({
        "Feature": FEATURES,
        "Input value": [
            input_values[name] for name in FEATURES
        ]
    })

    st.dataframe(
        input_display,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Prediction Report")

    report = {
        "project": "Nila River Water Level Prediction",
        "generated_at": datetime.now().isoformat(),
        "input_features": input_values,
        "predicted_next_day_river_level_m": prediction,
        "predicted_change_m": level_change,
        "weather_source": (
            "Open-Meteo weather model estimate"
            if weather_available
            else "Unavailable"
        ),
        "model_status": (
            "Experimental; trained on synthetic data"
        )
    }

    st.download_button(
        "📥 Download prediction report (JSON)",
        data=json.dumps(report, indent=2),
        file_name="nila_prediction_report.json",
        mime="application/json",
        use_container_width=True
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown("""
<div style="text-align:center;color:#60758a;padding:12px">
    <strong>🌊 Nila River Predictor</strong><br>
    Experimental machine learning project<br>
    <small>
        Predictions are not validated flood warnings.
        Consult official river-gauge data and local authorities
        for safety decisions.
    </small>
</div>
""", unsafe_allow_html=True)



