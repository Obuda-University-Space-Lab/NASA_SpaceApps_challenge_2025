import streamlit as st
import folium
from streamlit_folium import st_folium
from datetime import date
import numpy as np
import logic

st.set_page_config(layout="wide", page_title="Wildfire Prediction Model", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        font-size: 16px;
        padding: 0.75rem 1.5rem;
        border: none;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    section[data-testid="stSidebar"] {
        display: none;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }
    iframe {
        height: calc(100vh - 120px) !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Wildfire Prediction Model")

col1, col2 = st.columns([1, 4])

with col1:
    st.subheader("Input Parameters")
    
    longitude = st.number_input(
        "Longitude",
        min_value=-180.0,
        max_value=180.0,
        value=-120.0,
        step=0.1,
        format="%.4f"
    )
    
    latitude = st.number_input(
        "Latitude",
        min_value=-90.0,
        max_value=90.0,
        value=40.0,
        step=0.1,
        format="%.4f"
    )
    
    date_to_predict = st.date_input( # esetleg date-range?
        "Date to Predict",
        value=date.today(),
        format="YYYY-MM-DD"
    )
    
    rectangle_offset = st.slider(
        "Area Offset (km)",
        min_value=30,
        max_value=200,
        value=30,
        step=10
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    

    predict_button = st.button("Predict", type="primary", use_container_width=True)
    
    if predict_button:


        offset_deg = rectangle_offset / 111.0
    
        bounds = [
            [latitude - offset_deg, longitude - offset_deg],
            [latitude + offset_deg, longitude + offset_deg]
        ]
        out_image =logic.logic_func(latitude,longitude,date_to_predict,bounds)
        st.session_state.prediction_made = True
        st.session_state.pred_long = longitude
        st.session_state.pred_lat = latitude
        st.session_state.pred_date = date_to_predict
        st.session_state.pred_offset = rectangle_offset

with col2:
    m = folium.Map(
        location=[latitude, longitude],
        zoom_start=8,
        tiles="OpenStreetMap",
        scrollWheelZoom=True,
        dragging=True,
        zoomControl=True
    )
    
    offset_deg = rectangle_offset / 111.0
    
    bounds = [
        [latitude - offset_deg, longitude - offset_deg],
        [latitude + offset_deg, longitude + offset_deg]
    ]
    
    folium.Rectangle(
        bounds=bounds,
        color="#6d25d1",
        fill=True,
        fillColor="#c498dd",
        fillOpacity=0.2,
        weight=2,
        popup=f"Prediction Area: {rectangle_offset}km offset"
    ).add_to(m)
    
    folium.Marker(
        [latitude, longitude],
        popup=f"Center: ({latitude:.4f}, {longitude:.4f})",
        tooltip="Prediction Center",
        icon=folium.Icon(color="red", icon="fire", prefix='fa')
    ).add_to(m)
    
    if 'prediction_made' in st.session_state and st.session_state.prediction_made:
        
        np.random.seed(42)
        sample_points = []
        for _ in range(20):
            rand_lat = st.session_state.pred_lat + np.random.uniform(-offset_deg, offset_deg)
            rand_lon = st.session_state.pred_long + np.random.uniform(-offset_deg, offset_deg)
            sample_points.append([rand_lat, rand_lon])
        
        #TODO: KRIGING
        out_image.add_to(m)
        
        for point in sample_points:
            risk_level = np.random.uniform(0, 1)
            
            if risk_level < 0.3:
                color = 'green'
                risk_text = 'Low'
            elif risk_level < 0.7:
                color = 'orange'
                risk_text = 'Medium'
            else:
                color = 'red'
                risk_text = 'High'
            
            folium.CircleMarker(
                point,
                radius=8,
                popup=f"Risk: {risk_text} ({risk_level:.2%})",
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
        
        

    st_folium(m, width=None, height=800, returned_objects=[])

if 'prediction_made' in st.session_state and st.session_state.prediction_made:
    st.success(f"Prediction complete for {st.session_state.pred_date} at ({st.session_state.pred_lat:.4f}, {st.session_state.pred_long:.4f})") # nem mentjük sehova