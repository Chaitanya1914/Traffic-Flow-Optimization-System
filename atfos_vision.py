import os
# Fix for common Windows OpenMP and PyTorch threading conflict that causes abrupt crashes/interrupts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
from ultralytics import YOLO

def main():
    # 1. Initialize YOLOv8
    # 'yolov8n.pt' is the "Nano" version—fastest for laptops
    print("Initializing AI Eyes...")
    try:
        model = YOLO('yolov8n.pt') 
    except KeyboardInterrupt:
        print("Initialization interrupted.")
        return

    # 2. SELECT YOUR INPUT SOURCE
    # Set USE_WEBCAM to True for a live demo with your camera
    # Set USE_WEBCAM to False to use your 'traffic_video.mp4' file
    USE_WEBCAM = False

    if USE_WEBCAM:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Use CAP_DSHOW for faster startup on Windows
        print("Source: Laptop Webcam")
    else:
        video_path = 'videoplayback.mp4'
        cap = cv2.VideoCapture(video_path)
        print(f"Source: {video_path}")

    # Check if source is opened correctly
    if not cap.isOpened():
        print("Error: Could not open video source. Check path or camera connection!")
        return

    print("Vision System Active. Press 'q' to quit.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 3. Detect Vehicles
            # We filter for COCO classes: 2 (car), 5 (bus), 7 (truck)
            results = model(frame, classes=[2, 5, 7], verbose=False)

            # 4. Get the Live Count
            live_count = len(results[0].boxes)

            # 5. UI Overlays
            annotated_frame = results[0].plot() # Draws the boxes automatically
            
            # Add a clean text overlay for your presentation
            cv2.rectangle(annotated_frame, (10, 10), (350, 60), (0, 0, 0), -1) # Black background bar
            cv2.putText(annotated_frame, f"ATFOS COUNT: {live_count} Vehicles", (20, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # 6. Show the result
            cv2.imshow("ATFOS Phase 2 - Vehicle Detection", annotated_frame)

            # Adjust waitKey: 1 for webcam, 30 for video~ to keep speed normal
            delay = 1 if USE_WEBCAM else 30
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        # Prevent the random Windows/PyTorch interrupt from showing an ugly traceback
        print("\n[Notice] System interrupt signal received. Closing cleanly...")
    except Exception as e:
        print(f"\n[Error] An unexpected error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Vision System Shutdown.")

if __name__ == '__main__':
    # Wrapping everything to ensure multiprocessing works cleanly on Windows
    try:
        main()
    except KeyboardInterrupt:
        pass