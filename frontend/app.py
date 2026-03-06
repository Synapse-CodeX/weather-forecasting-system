import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime

BACKEND_URL = "http://localhost:5000"

st.set_page_config(
    page_title="Bakkhali Weather Prediction",
    page_icon="🌊",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #0066cc, #0099ff);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🌊 Bakkhali Weather Prediction</h1><p>Real-time 24-hour weather forecasts for Bakkhali Beach, West Bengal</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Bakkhali_sea_beach_3.jpg/800px-Bakkhali_sea_beach_3.jpg", 
             caption="Bakkhali Beach")

    st.header("⚙️ Controls")

    
    st.info("📊 Displaying Next 24 Hours Forecast")

    st.markdown("---")

    if st.button("🔄 Get Latest Predictions", type="primary", use_container_width=True):
        with st.spinner("Fetching 24-hour predictions..."):
            try:
                
                response = requests.get(f"{BACKEND_URL}/api/predict/24h", timeout=10)

                if response.status_code == 200:
                    data = response.json()
                    if data["success"]:
                        st.session_state.predictions = data["predictions"]
                        st.session_state.last_update = datetime.datetime.now()
                        st.success("✅ Predictions updated!")
                    else:
                        st.error(f"❌ Error: {data.get('error', 'Unknown error')}")
                else:
                    st.error(f"❌ Failed to connect to backend (Status: {response.status_code})")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Make sure the server is running at " + BACKEND_URL)
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    if "last_update" in st.session_state:
        st.info(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")


if "predictions" in st.session_state and st.session_state.predictions:
    predictions = st.session_state.predictions

    st.header("📊 Next 24 Hours Forecast")

    df = pd.DataFrame(predictions)

    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Temp", f"{df['Temperature (°C)'].iloc[0]:.1f}°C")
    with col2:
        st.metric("Current Humidity", f"{df['Humidity (%)'].iloc[0]:.1f}%")
    with col3:
        st.metric("Wind Speed", f"{df['Wind Speed (m/s)'].iloc[0]:.1f} m/s")
    with col4:
        st.metric("Pressure", f"{df['Pressure (kPa)'].iloc[0]:.1f} kPa")

    st.markdown("---")

    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Temperature', 'Humidity & Precipitation', 'Wind Speed & Radiation'),
        vertical_spacing=0.12
    )

    
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['Temperature (°C)'],
                  mode='lines+markers', name='Temperature',
                  line=dict(color='red', width=2),
                  marker=dict(size=6)),
        row=1, col=1
    )

    
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['Humidity (%)'],
                  mode='lines+markers', name='Humidity',
                  line=dict(color='blue', width=2),
                  marker=dict(size=6)),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(x=df['datetime'], y=df['Precipitation (mm/hr)'],
              name='Precipitation', marker_color='lightblue',
              opacity=0.5),
        row=2, col=1
    )

    
    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['Wind Speed (m/s)'],
                  mode='lines+markers', name='Wind Speed',
                  line=dict(color='green', width=2),
                  marker=dict(size=6)),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(x=df['datetime'], y=df['Radiation (W/m²)'],
                  mode='lines+markers', name='Radiation',
                  line=dict(color='orange', width=2),
                  marker=dict(size=6)),
        row=3, col=1
    )

    fig.update_layout(height=900, showlegend=True, hovermode='x unified')
    fig.update_xaxes(title_text="Time", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

    
    with st.expander("📋 View Hourly Breakdown"):
        st.dataframe(
            df.style.format({
                'Temperature (°C)': '{:.1f}°C',
                'Humidity (%)': '{:.1f}%',
                'Wind Speed (m/s)': '{:.1f} m/s',
                'Pressure (kPa)': '{:.1f} kPa',
                'Precipitation (mm/hr)': '{:.2f} mm',
                'Cloud Coverage (%)': '{:.1f}%',
                'Radiation (W/m²)': '{:.1f} W/m²'
            }),
            use_container_width=True
        )

else:
    st.info("👈 Click 'Get Latest Predictions' in the sidebar to view 24-hour weather forecast")

    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background: #f0f2f6; padding: 2rem; border-radius: 10px; text-align: center;">
            <h3>🌡️ Temperature</h3>
            <p>Hourly temperature forecasts for the next 24 hours</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background: #f0f2f6; padding: 2rem; border-radius: 10px; text-align: center;">
            <h3>💧 Humidity & Rain</h3>
            <p>Track humidity levels and precipitation</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background: #f0f2f6; padding: 2rem; border-radius: 10px; text-align: center;">
            <h3>💨 Wind & Radiation</h3>
            <p>Monitor wind conditions and solar radiation</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "🌊 Bakkhali 24-Hour Weather Prediction System | Data: Open-Meteo API | Model: XGBoost"
    "</p>", 
    unsafe_allow_html=True
)