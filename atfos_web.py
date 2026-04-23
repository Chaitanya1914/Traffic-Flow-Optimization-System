import os
# Absolutely critical to stop PyTorch thread crashing on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import google.generativeai as genai
from ultralytics import YOLO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- PAGE CONFIG ---
st.set_page_config(page_title="ATFOS Master Dashboard", layout="wide", page_icon="🚦")

# --- CUSTOM CSS FOR PREMIUM AESTHETICS ---
st.markdown("""
    <style>
    .metric-container {
        background-color: #1E1E2E;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-bottom: 20px;
        border: 1px solid #333344;
    }
    .main-title {
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3em;
        margin-bottom: 0px;
    }
    .stChatFloatingInputContainer {
        padding-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🚦 ATFOS</h1>", unsafe_allow_html=True)
st.markdown("### AI-Driven Traffic Optimization System")
st.markdown("*Bridging Historical Machine Learning with Live Computer Vision to automate civic infrastructure.*")
st.divider()

# --- SIDEBAR: API CONFIGURATION ---
st.sidebar.header("⚙️ API Configuration")

# Safely fetch the key from .env file
env_api_key = os.getenv("GEMINI_API_KEY")

if env_api_key:
    api_key = env_api_key
    st.sidebar.success("✅ API Key securely loaded from .env")
else:
    api_key = st.sidebar.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API key to activate the Smart Chatbot.")
    
st.sidebar.markdown("""
*If you do not have an API key, get one for free at [Google AI Studio](https://aistudio.google.com/).* \n
*The rest of the CV/ML pipeline works entirely locally!*
""")

# --- LOAD MODELS EFFICIENTLY ---
@st.cache_resource
def load_models():
    brain = joblib.load('atfos_model.pkl')
    model_cols = joblib.load('atfos_features.pkl')
    eyes = YOLO('yolov8n.pt')
    return brain, model_cols, eyes

with st.spinner("Initializing Deep Learning Models (PyTorch & Scikit-learn)..."):
    brain, model_cols, eyes = load_models()

# --- DEFINE CORE LOGIC BRIDGE ---
def make_decision(live_count, hist_speed):
    if live_count > 15 or hist_speed < 20:
        return "HEAVY TRAFFIC: 60s Green", "#FF4B4B" # Red
    elif live_count > 5:
        return "MODERATE TRAFFIC: 45s Green", "#FFA500" # Orange
    else:
        return "LOW TRAFFIC: 20s Green", "#00C853" # Green

# --- LAYOUT SETUP ---
col1, col2 = st.columns([1.8, 1.2])

with col1:
    st.markdown("#### 🎥 Live Vision Gateway (Phase 2)")
    # Streamlit Video Frame Placeholder
    stframe = st.empty()
    run_video = st.button("🟢 Start System & Process Video", use_container_width=True)

with col2:
    st.markdown("#### 🧠 System Metrics & Data Fusion (Phase 1 + 3)")
    
    # We will use placeholders for metrics so they update cleanly
    metric_area = st.empty()
    
    st.divider()
    
    st.markdown("#### 🤖 AI Command Center")
    # Chatbot Area
    chat_container = st.container(height=380)
    user_prompt = st.chat_input("Ask the system why it changed the lights...")

# --- INIT STATE FOR PROMPT ---
if "live_cars" not in st.session_state:
    st.session_state.live_cars = 0
if "pred_speed" not in st.session_state:
    st.session_state.pred_speed = 0.0
if "last_decision" not in st.session_state:
    st.session_state.last_decision = "Waiting for data..."
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello operator! I am the ATFOS AI Assistant. Enter your Gemini API key, then ask me anything about the current intersection timings."}
    ]

# --- DISPLAY CHAT HISTORY ---
with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# --- LLM CHATBOT LOGIC ---
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_prompt)
        
        with st.chat_message("assistant"):
            if api_key:
                with st.spinner("ATFOS is thinking..."):
                    try:
                        # Configure API
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-flash-latest')
                        
                        system_context = f"""
                        You are ATFOS (AI-Driven Traffic Optimization System). 
                        You are an advanced traffic automation controller.
                        Currently, your cameras detect {st.session_state.live_cars} vehicles.
                        Your historical Machine Learning models expect traffic to crawl at {st.session_state.pred_speed:.1f} km/h right now.
                        Because of this, your overarching 'Logic Bridge' decided to output: {st.session_state.last_decision}.
                        
                        Give the user a short, incredibly smart-sounding, engineering-focused explanation for your decision based on the live count vs historical data constraint logic.
                        Be extremely professional and brief (Under 3 concise sentences). Answer directly.
                        """
                        
                        # Call API
                        response = model.generate_content(system_context + "\nUser asked: " + user_prompt)
                        bot_reply = response.text
                    except Exception as e:
                        bot_reply = f"⚠️ **API Error:** Check the key! ({str(e)})"
            else:
                bot_reply = f"🔑 **Activation Required:** Please enter a Gemini API Key in the sidebar. I cannot dynamically generate explanations without it! \n\n*Current Debug State: [Live Cars: {st.session_state.live_cars}] | [Hist Speed: {st.session_state.pred_speed:.1f} km/h].*"
            
            st.markdown(bot_reply)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})


# --- REAL-TIME VIDEO PROCESSING PIPELINE ---
if run_video:
    cap = cv2.VideoCapture('videoplayback.mp4')
    
    # Static Data Load for Master Context (Phase 1)
    sample_data = pd.DataFrame([{
        'time_of_day': 'Morning Peak',
        'day_of_week': 'Monday',
        'weather_condition': 'Clear',
        'road_type': 'Main Road'
    }])
    test_dummies = pd.get_dummies(sample_data)
    final_input = test_dummies.reindex(columns=model_cols, fill_value=0)
    
    # Calculate Base Prediction Rate
    predicted_speed = brain.predict(final_input)[0]
    st.session_state.pred_speed = predicted_speed
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # If video ends, loop it seamlessly
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        # 1. Vision Processing
        # Resize frame slightly to make processing instantly responsive on laptops
        frame = cv2.resize(frame, (640, 360))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Streamlit constraint
        
        results = eyes(frame_rgb, classes=[2, 5, 7], verbose=False)
        live_cars = len(results[0].boxes)
        
        annotated_frame = results[0].plot()
        
        # 2. Logic Fusion Execution
        decision_text, color = make_decision(live_cars, predicted_speed)
        
        # 3. State Management Sync
        st.session_state.live_cars = live_cars
        st.session_state.last_decision = decision_text
        
        # 4. Streamlit Render
        stframe.image(annotated_frame, channels="RGB", use_container_width=True)
        
        with metric_area.container():
            st.markdown(f"""
            <div class="metric-container">
                <h4 style="color:#aaa; margin:0;">Live Vehicle Count: <span style="color:white; font-size:24px;">{live_cars}</span></h4>
                <p style="color:#aaa; font-size:12px; margin-bottom:10px;">Instantaneous reading from YOLOv8 Vision Layer.</p>
                <h4 style="color:#aaa; margin:0;">Historical Expected Flow: <span style="color:white; font-size:24px;">{predicted_speed:.1f} km/h</span></h4>
                <p style="color:#aaa; font-size:12px; margin-bottom:10px;">Predicted by Scikit-learn Phase 1 Model constraint.</p>
                <hr style="border-color:#444;">
                <div style="background-color: {color}; padding: 10px; border-radius: 8px; color: #fff; text-align: center; box-shadow: 0 0 10px {color}88;">
                    <strong>AI ISSUED TIMING:</strong><br><span style="font-size:1.2em;">{decision_text}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    cap.release()
