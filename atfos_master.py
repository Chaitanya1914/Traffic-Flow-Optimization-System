import os
# Fix for common Windows OpenMP and PyTorch threading conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import joblib
import pandas as pd
import numpy as np
from ultralytics import YOLO

# 1. LOAD THE BRAIN (ML Model & Features)
print("Loading AI Brain...")
brain = joblib.load('atfos_model.pkl')
model_cols = joblib.load('atfos_features.pkl')

# 2. LOAD THE EYES (YOLO)
print("Loading AI Eyes...")
eyes = YOLO('yolov8n.pt')

# 3. SET UP THE VIDEO
cap = cv2.VideoCapture('videoplayback.mp4') 

if not cap.isOpened():
    print("Error: Could not open video source 'videoplayback.mp4'.")


def make_decision(live_count, hist_speed):
    """ The Core Logic: Fusion of ML and CV """
    # Logic: If count is high OR historical speed is very low (peak hour)
    if live_count > 15 or hist_speed < 20:
        return "HEAVY TRAFFIC: Green Light for 60s", (0, 0, 255) # Red-ish/Alert
    elif live_count > 5:
        return "MODERATE TRAFFIC: Green Light for 45s", (0, 255, 255) # Yellow
    else:
        return "LOW TRAFFIC: Green Light for 20s", (0, 255, 0) # Green

print("Master System Active. Press 'q' to stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # --- STEP A: VISION (Live Count) ---
    results = eyes(frame, classes=[2, 5, 7], verbose=False)
    live_cars = len(results[0].boxes)
    
    # --- STEP B: PREDICTION (Historical Context) ---
    # We create a 'test case' that matches your Delhi Dataset
    sample_data = pd.DataFrame([{
        'time_of_day': 'Morning Peak',
        'day_of_week': 'Monday',
        'weather_condition': 'Clear',
        'road_type': 'Main Road'
    }])
    
    # The Fix: Match columns with your saved feature list
    test_dummies = pd.get_dummies(sample_data)
    final_input = test_dummies.reindex(columns=model_cols, fill_value=0)
    
    predicted_speed = brain.predict(final_input)[0]

    # --- STEP C: THE DECISION ---
    decision_text, color = make_decision(live_cars, predicted_speed)

    # --- STEP D: UI OVERLAY ---
    annotated_frame = results[0].plot()
    
    # Top Dashboard Bar
    cv2.rectangle(annotated_frame, (0, 0), (640, 100), (0, 0, 0), -1)
    
    cv2.putText(annotated_frame, f"LIVE COUNT: {live_cars} | HIST SPEED: {predicted_speed:.1f} km/h", 
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.putText(annotated_frame, f"AI DECISION: {decision_text}", 
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("ATFOS MASTER SYSTEM", annotated_frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()