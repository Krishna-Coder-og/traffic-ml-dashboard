import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import time

# -- Page Configuration --
st.set_page_config(page_title="Traffic Insight Pro", page_icon="🚗", layout="wide", initial_sidebar_state="expanded")

# -- Premium CSS Injection --
st.markdown("""
<style>
    /* Gradient Backgrounds & Fonts */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    h1 {
        color: #00F2FE;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 2px 2px 8px rgba(0,242,254,0.3);
    }
    h2, h3 {
        color: #4FACFE;
        font-weight: 600;
    }

    /* DataFrame & Widget Borders */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Prediction Box Cards */
    .pred-box {
        background: linear-gradient(135deg, #1C2331 0%, #0E1117 100%);
        padding: 30px;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.05);
        box-shadow: 0 8px 32px 0 rgba(0,0,0,0.3);
        text-align: center;
        margin-top: 20px;
        transition: transform 0.3s ease;
    }
    .pred-box:hover {
        transform: translateY(-5px);
    }
    
    .sev-1 { border-left: 5px solid #00E676; } /* Low Severity: Green */
    .sev-2 { border-left: 5px solid #FFEA00; } /* Medium: Yellow */
    .sev-3 { border-left: 5px solid #FF1744; } /* High: Red */
    
    /* Sleek buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #4FACFE 0%, #00F2FE 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        height: 50px;
        transition: opacity 0.3s ease;
    }
    .stButton>button:hover {
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("45_traffic_accidents.csv")
    return df

df = load_data()

# -- Sidebar Navigation --
st.sidebar.title("🚗 Menu")
page = st.sidebar.radio("Navigate", ["📊 Data Explorer", "🤖 Accident Predictor"])

st.sidebar.markdown("---")
st.sidebar.info(
    "**Traffic Insight Pro**\n\n"
    "Harnessing machine learning to predict accident severity based on real-world conditions."
)

if page == "📊 Data Explorer":
    st.title("Traffic Accident Data Explorer")
    st.markdown("Explore historical traffic accident data to uncover patterns and risk factors.")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", len(df))
    col2.metric("Avg Speed (km/h)", round(df['speed_kmh'].mean(), 1))
    col3.metric("Most Frequent Weather", df['weather'].mode()[0].title())
    col4.metric("High Severity Cases", len(df[df['severity_1to3'] == 3]))
    
    st.markdown("---")
    
    # Visualizations
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Severity vs Speed Distribution")
        fig1 = px.violin(df, x="severity_1to3", y="speed_kmh", color="severity_1to3", 
                         box=True, template="plotly_dark", 
                         title="Speed by Severity Level")
        st.plotly_chart(fig1, use_container_width=True)
        
    with c2:
        st.subheader("Accidents by Weather")
        weather_counts = df['weather'].value_counts().reset_index()
        weather_counts.columns = ['Weather', 'Count']
        fig2 = px.bar(weather_counts, x="Weather", y="Count", color="Weather", 
                      template="plotly_dark", title="Total Accidents per Weather Condition")
        st.plotly_chart(fig2, use_container_width=True)
        
    st.subheader("Severity Ratio by Road Type")
    fig3 = px.sunburst(df, path=['road_type', 'severity_1to3'], template="plotly_dark",
                       color='severity_1to3', color_continuous_scale='Reds')
    st.plotly_chart(fig3, use_container_width=True)
    
    st.subheader("Raw Dataset Preview")
    st.dataframe(df.head(100), use_container_width=True)

elif page == "🤖 Accident Predictor":
    st.title("AI Accident Predictor")
    st.markdown("Enter simulated environmental and vehicular conditions to predict potential accident severity using our Random Forest AI.")
    
    st.markdown("---")
    
    with st.form("prediction_form"):
        st.subheader("Simulation Parameters")
        
        c1, c2 = st.columns(2)
        with c1:
            speed = st.slider("🚗 Vehicle Speed (km/h)", min_value=0, max_value=200, value=60)
            vehicles = st.number_input("🚙 Vehicles Involved", min_value=1, max_value=10, value=2)
            road = st.selectbox("🛣️ Road Type", df['road_type'].unique())
            
        with c2:
            weather = st.selectbox("🌧️ Weather Condition", df['weather'].unique())
            light = st.selectbox("💡 Light Condition", df['light_condition'].unique())
            
        submitted = st.form_submit_button("Engage AI Predictor")
        
    if submitted:
        payload = {
            "speed_kmh": speed,
            "vehicles_involved": vehicles,
            "weather": weather,
            "light_condition": light,
            "road_type": road
        }
        
        with st.spinner("Analyzing neural pathways..."):
            time.sleep(0.8) # Artificial delay for aesthetic UX
            
            sev = None
            try:
                # 1. Try to connect to Flask Backend
                res = requests.post("http://127.0.0.1:5000/predict", json=payload, timeout=2)
                if res.status_code == 200:
                    sev = res.json().get('prediction')
            except Exception:
                # 2. Fallback for Streamlit Cloud Deployments: Load Model directly
                import joblib
                try:
                    model = joblib.load('model.pkl')
                    model_columns = joblib.load('model_columns.pkl')
                    
                    # Process exactly like Flask does
                    df_input = pd.DataFrame([payload])
                    df_input = pd.get_dummies(df_input)
                    df_input = df_input.reindex(columns=model_columns, fill_value=0)
                    
                    sev = int(model.predict(df_input)[0])
                except FileNotFoundError:
                    st.error("No model files found. Please ensure `model.pkl` is uploaded to your repo.")
            
            if sev:
                if sev == 1:
                    sev_text = "Minor Impact"
                    css_class = "sev-1"
                    icon = "🟢"
                elif sev == 2:
                    sev_text = "Moderate Damage"
                    css_class = "sev-2"
                    icon = "🟡"
                else:
                    sev_text = "CRITICAL SEVERITY"
                    css_class = "sev-3"
                    icon = "🔴"
                    
                st.markdown(f"""
                    <div class="pred-box {css_class}">
                        <h2 style="margin:0; font-size:1.5rem; color:#8E9BAE;">Predicted Severity Level</h2>
                        <h1 style="margin-top:10px; font-size:4rem;">{icon} {sev}</h1>
                        <h3 style="color:#FFF;">{sev_text}</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                st.success("Target analysis complete. (Used direct model fallback or Flask API).")
