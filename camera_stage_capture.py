# camera_stage_capture.py
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from ultralytics import YOLO
import time
import json
import os
from datetime import datetime

# Initialize models
print("Loading models...")
detector = MTCNN()
cert_model = YOLO("yolov8n.pt")
print("Models loaded!")

# ADD HAND CASCADE
try:
    hand_cascade = cv2.CascadeClassifier('haarcascade_hand.xml')
    print("Hand cascade loaded (gesture detection enabled)")
except:
    hand_cascade = None
    print("Hand cascade not available (gesture detection disabled)")

print("Models loaded!")

# Initialize variables
countdown_started = False
countdown_start_time = None
countdown_duration = 3
photo_captured = False
captured_photo_path = None
no_faces_count = 0
no_faces_threshold = 60  # 2 seconds at 30fps
auto_reset_timer_start = None
auto_reset_duration = 15  # 15 seconds display time

# ADD GESTURE VARIABLES
gesture_detected = False
gesture_start_time = None
gesture_duration_required = 2  # Wave for 2 seconds

def detect_hand_gesture(img):
    """Detect hand wave gesture in upper portion of screen"""
    if hand_cascade is None:
        return False
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hands = hand_cascade.detectMultiScale(gray, 1.1, 5)
    
    # If hand detected in upper portion of screen (waving)
    for (x, y, w, h) in hands:
        if y < img.shape[0] // 2:  # Upper half of screen
            # Draw hand detection box for visual feedback
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
            cv2.putText(img, "Hand Detected", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            return True
    return False

def main():
    global countdown_started, countdown_start_time, photo_captured, captured_photo_path
    global no_faces_count, auto_reset_timer_start
    global gesture_detected, gesture_start_time
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        result = {"error": "Camera not available"}
        with open('stage_capture_result.json', 'w') as f:
            json.dump(result, f)
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    cv2.namedWindow('Stage Photo Capture', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Stage Photo Capture', 1280, 720)
    
    os.makedirs("stage_photos", exist_ok=True)
    
    print("Camera opened. Press 'Q' to quit")
    print("Auto-capture will trigger when 2 faces + certificate detected")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        img = frame.copy()
        
        # 1. Certificate detection
        results = cert_model(img, verbose=False)
        certificate_found = False
        
        for r in results:
            for box in r.boxes:
                cls = r.names[int(box.cls)]
                if cls in ["book", "folder", "paper", "document"]:
                    certificate_found = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
                    cv2.putText(img, f"Certificate ({cls})", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        # 2. Face detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)
        faces_count = len(faces)
        
        # Count faces looking forward
        faces_looking_forward = 0
        for face in faces:
            confidence = face['confidence']
            if confidence > 0.95:
                faces_looking_forward += 1
            
            x, y, w, h = face['box']
            color = (0, 255, 0) if confidence > 0.95 else (255, 165, 0)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
            cv2.putText(img, f"{confidence:.2f}", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Check if ready
        capture_ready = (certificate_found and faces_looking_forward == 2)
        
        # 3. Auto-countdown and capture logic
        if not photo_captured:
            if capture_ready:
                # Start countdown
                if not countdown_started:
                    countdown_started = True
                    countdown_start_time = time.time()
                    print("Countdown started!")
                
                elapsed = time.time() - countdown_start_time
                remaining = countdown_duration - elapsed
                
                if remaining > 0:
                    # Show countdown
                    countdown_text = f"SMILE! Capturing in {int(remaining) + 1}..."
                    cv2.putText(img, countdown_text, (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    
                    # Draw countdown circle
                    center_x, center_y = img.shape[1] // 2, 150
                    radius = 60
                    progress = 1 - (remaining / countdown_duration)
                    angle = int(360 * progress)
                    cv2.ellipse(img, (center_x, center_y), (radius, radius),
                               -90, 0, angle, (0, 255, 0), 8)
                    cv2.putText(img, str(int(remaining) + 1), (center_x - 30, center_y + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 4)
                else:
                    # Capture!
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"stage_photos/stage_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    
                    photo_captured = True
                    captured_photo_path = filename
                    auto_reset_timer_start = time.time()
                    
                    print(f"Photo captured: {filename}")
                    
                    # Show flash effect
                    cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), 
                                 (255, 255, 255), 50)
            else:
                # Reset countdown if not ready
                if countdown_started:
                    countdown_started = False
                    countdown_start_time = None
        
        # 4. Handle post-capture state
        if photo_captured:
            # Start auto-reset timer
            if auto_reset_timer_start is None:
                auto_reset_timer_start = time.time()
            
            elapsed_display = time.time() - auto_reset_timer_start
            
            # Check if people left (faster reset)
            if faces_count == 0:
                no_faces_count += 1
            else:
                no_faces_count = 0
            
            # HAND GESTURE DETECTION FOR RESET
            gesture_reset_triggered = False
            if hand_cascade is not None:
                if detect_hand_gesture(img):
                    if not gesture_detected:
                        gesture_detected = True
                        gesture_start_time = time.time()
                        print("👋 Hand gesture detected! Keep waving for 2 seconds to reset")
                    
                    elapsed_gesture = time.time() - gesture_start_time
                    
                    if elapsed_gesture >= gesture_duration_required:
                        # Reset triggered by gesture!
                        gesture_reset_triggered = True
                    else:
                        # Show progress
                        remaining = gesture_duration_required - elapsed_gesture
                        progress_bar_width = int(300 * (elapsed_gesture / gesture_duration_required))
                        cv2.rectangle(img, (50, 220), (350, 250), (100, 100, 100), -1)
                        cv2.rectangle(img, (50, 220), (50 + progress_bar_width, 250), (255, 0, 255), -1)
                        cv2.putText(img, f"Keep waving... {remaining:.1f}s", (50, 210),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                else:
                    # Hand not detected, reset gesture timer
                    gesture_detected = False
                    gesture_start_time = None
            
            # Perform gesture reset
            if gesture_reset_triggered:
                print("✅ Gesture reset triggered!")
                photo_captured = False
                captured_photo_path = None
                countdown_started = False
                no_faces_count = 0
                auto_reset_timer_start = None
                gesture_detected = False
                gesture_start_time = None
                
                # Show reset message with animation
                for i in range(3):
                    temp_img = img.copy()
                    cv2.putText(temp_img, "GESTURE RESET!", (img.shape[1]//2 - 200, img.shape[0]//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 4)
                    cv2.imshow('Stage Photo Capture', temp_img)
                    cv2.waitKey(300)
                continue
            
            # AUTO-RESET CONDITIONS (only if no gesture in progress)
            if not gesture_detected:
                if no_faces_count >= no_faces_threshold or elapsed_display >= auto_reset_duration:
                    # Auto-reset
                    photo_captured = False
                    captured_photo_path = None
                    countdown_started = False
                    no_faces_count = 0
                    auto_reset_timer_start = None
                    
                    reset_reason = "People left frame" if no_faces_count >= no_faces_threshold else "15-second timer"
                    print(f"🔄 Auto-reset: {reset_reason}")
                    
                    cv2.putText(img, f"AUTO-RESET: {reset_reason}", (50, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3)
                    cv2.imshow('Stage Photo Capture', img)
                    cv2.waitKey(1000)
                    continue
            
            # DISPLAY CAPTURED STATUS
            remaining_time = int(auto_reset_duration - elapsed_display)
            cv2.putText(img, "PHOTO CAPTURED!", (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            
            if hand_cascade is not None:
                cv2.putText(img, "Wave hand to reset immediately", (50, 160),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(img, f"OR auto-reset in {remaining_time}s / leave frame", (50, 190),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            else:
                cv2.putText(img, f"Auto-reset in {remaining_time}s or leave frame", (50, 160),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
        # Display status bar
        status_text = f"Faces: {faces_looking_forward}/2 | Cert: {'YES' if certificate_found else 'NO'}"
        status_color = (0, 255, 0) if capture_ready else (0, 0, 255)
        cv2.putText(img, status_text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        if capture_ready and not countdown_started and not photo_captured:
            cv2.putText(img, "READY! Hold still...", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        # Show instructions
        instructions_y = img.shape[0] - 100
        cv2.putText(img, "Press 'Q' to quit", (50, instructions_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if hand_cascade is not None:
            cv2.putText(img, "Wave hand (upper half) to reset after capture", (50, instructions_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Stage Photo Capture', img)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            print("User quit")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save result
    if captured_photo_path:
        result = {
            "status": "success",
            "photo_path": captured_photo_path,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    else:
        result = {"status": "cancelled"}
    
    with open('stage_capture_result.json', 'w') as f:
        json.dump(result, f)

if __name__ == "__main__":
    main()