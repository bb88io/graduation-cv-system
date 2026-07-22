# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import csv
import qrcode
import os
import smtplib
from email.message import EmailMessage
import pandas as pd
import av
from scipy.spatial.distance import euclidean
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import threading
import time
from datetime import datetime
from gtts import gTTS
import base64
from io import BytesIO
import threading
import json
import subprocess
import re
import sys
import serial
from ultralytics import YOLO



def text_to_speech(text):
    """Convert text to speech and play it"""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio_base64 = base64.b64encode(fp.read()).decode()
        audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        print(f"TTS Error: {e}")

# -------------------------------------------------
# AUTO CAPTURE TRANSFORMER WITH MIRROR FLIP
# -------------------------------------------------
class AutoCaptureTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None
        self.frame_original = None  # Store original unflipped frame

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Store original unflipped frame for saving
        self.frame_original = img.copy()
        
        # Flip horizontally for mirror effect display
        img_flipped = cv2.flip(img, 1)
        self.frame = img_flipped
        
        return img_flipped
        
# -------------------------
# Config / Paths
# -------------------------
os.makedirs("student_qr_codes", exist_ok=True)
os.makedirs("student_csv", exist_ok=True)
os.makedirs("face_database", exist_ok=True)

CSV_FILE = "student_csv/students.csv"
EMBEDDED_CSV = "student_csv/student_with_embeddings.csv"
ATTENDANCE_CSV = "student_csv/attendance.csv"
MASK_MODEL_PATH = "mask_detectorV2.h5"

# CSV header
CSV_FIELDS = ['student_id', 'student_name', 'email', 'programme', 'faculty','fingerprint_id',
              'img1', 'img2', 'img3', 'img4', 'img5']

ATTENDANCE_FIELDS = ['student_id', 'student_name', 'programme', 'faculty', 
                     'status', 'verification_time']

# Create CSV if missing
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

if not os.path.exists(ATTENDANCE_CSV):
    with open(ATTENDANCE_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=ATTENDANCE_FIELDS)
        writer.writeheader()

# -------------------------
# Utilities: Extract & Update Embeddings
# -------------------------
def extract_embedding(image_path):
    if not os.path.exists(image_path):
        st.warning(f"Image not found: {image_path}")
        return None
    img = cv2.imread(image_path)

    if img is None:
        st.warning(f"Failed to read image: {image_path}")
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = DETECTOR.detect_faces(img_rgb)
    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]['box']
    x, y = max(0, x-10), max(0, y-10)
    w, h = w+20, h+20
    face = img_rgb[y:y+h, x:x+w]

    face = cv2.resize(face, (160, 160))
    face = face.astype('float32')
    face = (face - face.mean()) / (face.std() + 1e-6)

    embedding = EMBEDDER.embeddings([face])[0]
    return embedding

def save_student_embedding(student_id, student_name, email, programme, faculty, fingerprint, image_files):
    # Extract embeddings from 5 images
    all_embeddings = []

    for img_path in image_files:
        emb = extract_embedding(img_path)
        if emb is not None:
            all_embeddings.append(emb)
        else:
            st.warning(f"❌ No face detected in {img_path}")

    if len(all_embeddings) == 0:
        return False, "No face detected in the images."

    # Average embedding (stronger representation)
    final_embedding = np.mean(np.array(all_embeddings), axis=0)

    row = {
        "student_id": student_id,
        "student_name": student_name,
        "email": email,
        "programme": programme,
        "faculty": faculty,
        "fingerprint_id": fingerprint,
        "img1": image_files[0],
        "img2": image_files[1],
        "img3": image_files[2],
        "img4": image_files[3],
        "img5": image_files[4],
        "embedding": final_embedding.tolist()
    }

    # Save or append
    emb_csv = "student_csv/student_with_embeddings.csv"
    df_row = pd.DataFrame([row])

    if os.path.exists(emb_csv):
        df_row.to_csv(emb_csv, mode="a", header=False, index=False)
    else:
        df_row.to_csv(emb_csv, index=False)

    return True, "Embedding saved."

# -------------------------
# Utilities: QR decoding
# -------------------------
qr_detector = cv2.QRCodeDetector()

def decode_qr_from_image(img_array):
    data, bbox, _ = qr_detector.detectAndDecode(img_array)
    if data:
        return data
    try:
        ok, decoded_infos, points, _ = qr_detector.detectAndDecodeMulti(img_array)
        if ok and decoded_infos:
            for d in decoded_infos:
                if d:
                    return d
    except Exception:
        pass
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    data2, _, _ = qr_detector.detectAndDecode(enhanced)
    if data2:
        return data2
    return None

def log_attendance(student_data, status):
    """Log attendance to CSV"""
    try:
        attendance_record = {
            'student_id': student_data.get('student_id', ''),
            'student_name': student_data.get('student_name', ''),
            'programme': student_data.get('programme', ''),
            'faculty': student_data.get('faculty', ''),
            'status': status,
            'verification_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(ATTENDANCE_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=ATTENDANCE_FIELDS)
            writer.writerow(attendance_record)
        return True
    except Exception as e:
        st.error(f"Failed to log attendance: {e}")
        return False

# -------------------------
# Session state defaults
# -------------------------
if "page" not in st.session_state:
    st.session_state.page = "register"
if "photo_captured" not in st.session_state:
    st.session_state.photo_captured = False
if "captured_photo_path" not in st.session_state:
    st.session_state.captured_photo_path = None
if "current_student_id" not in st.session_state:
    st.session_state.current_student_id = None
if "scanned_id" not in st.session_state:
    st.session_state.scanned_id = None
if "captured_images" not in st.session_state:
    st.session_state.captured_images = []
if "capture_started" not in st.session_state:
    st.session_state.capture_started = False
if "face_match_data" not in st.session_state:
    st.session_state.face_match_data = None
if "status_messages" not in st.session_state:
    st.session_state.status_messages = []

# -------------------------
# Sidebar navigation with scrollable status
# -------------------------
st.sidebar.title("Navigation")
if st.sidebar.button("Register Student"):
    st.session_state.page = "register"
if st.sidebar.button("Scan QR"):
    st.session_state.page = "scan"
if st.sidebar.button("Face Recognition"):
    st.session_state.page = "face"
if st.sidebar.button("Attendance Dashboard"):
    st.session_state.page = "dashboard"
if st.sidebar.button("📸 Capture Stage Photo"):
    st.session_state.page = "capture"
if st.sidebar.button("🎓 Graduation Photo Capture"):
    st.session_state.page = "graduation_capture"
st.sidebar.markdown("---")
st.sidebar.write("**Current page:**", st.session_state.page)

# Status Messages Section - Fixed height, latest at top
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Recent Activity")

# Display messages with fixed container
if st.session_state.status_messages:
    # Get last 10 messages in reverse order (latest first)
    recent_messages = list(reversed(st.session_state.status_messages[-10:]))
    
    for msg in recent_messages:
        timestamp = msg.get('time', '')
        text = msg.get('text', '')
        msg_type = msg.get('type', 'info')
        
        # Color coding based on message type
        if msg_type == 'success':
            icon = '✅'
            st.sidebar.success(f"**{timestamp}** {icon} {text}")
        elif msg_type == 'error':
            icon = '❌'
            st.sidebar.error(f"**{timestamp}** {icon} {text}")
        elif msg_type == 'warning':
            icon = '⚠️'
            st.sidebar.warning(f"**{timestamp}** {icon} {text}")
        else:
            icon = 'ℹ️'
            st.sidebar.info(f"**{timestamp}** {icon} {text}")
    
    if st.sidebar.button("🗑️ Clear Messages"):
        st.session_state.status_messages = []
        st.rerun()
else:
    st.sidebar.info("No status messages yet")

def add_status_message(text, msg_type='info'):
    """Add a status message to the sidebar"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    st.session_state.status_messages.append({
        'time': timestamp,
        'text': text,
        'type': msg_type
    })

# -------------------------
# Input Validation
# -------------------------
def validate_student_id(student_id):
    """
    Format: 24PMRXXXXX (X = digit)
    Example: 24PMR12345
    """
    pattern = r"^24PMR\d{5}$"
    if re.match(pattern, student_id):
        return True, ""
    return False, "Student ID must be in the format: 24PMRXXXXX (X = digit)"


def validate_email(email):
    """
    Must be: name-pm24@student.tarc.edu.my
    Example: johndoe-pm24@student.tarc.edu.my
    """

    pattern = r"^[a-zA-Z]+-pm24@student\.tarc\.edu\.my$"

    if re.match(pattern, email):
        return True, ""
    return False, "Email must be in the format: name-pm24@student.tarc.edu.my"


def validate_text_only(value, field_name):
    """
    Programme and faculty: letters + spaces only
    """
    pattern = r"^[A-Za-z ()]+$"
    if re.match(pattern, value):
        return True, ""
    return False, f"{field_name} must contain only letters (no numbers)"

# --- AUTO UPPERCASE INPUT FUNCTION ---
def uppercase_input(label, key):
    # If value exists, convert to uppercase
    if key in st.session_state and st.session_state[key]:
        st.session_state[key] = st.session_state[key].upper()
    return st.text_input(label, key=key)


# --- VALIDATION FUNCTIONS ---
def validate_student_name(student_name):
    """
    Only letters + spaces allowed (case doesn't matter)
    """
    pattern = r"^[A-Za-z ]+$"
    if re.match(pattern, student_name):
        return True, ""
    return False, f"{student_name} must contain only letters (no numbers)"

# --- FINGERPRINT CONFIGURATION --- 
SERIAL_PORT = 'COM3'  # <--- CHANGE THIS to your Arduino Port
BAUD_RATE = 9600

def get_next_fingerprint_id(csv_file):
    """Calculates the next available numeric ID (1, 2, 3...) for the sensor."""
    if not os.path.exists(csv_file):
        return 1
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            # Look for a column named 'fingerprint_id'. If missing, return 1.
            ids = []
            for row in reader:
                if 'fingerprint_id' in row and row['fingerprint_id']:
                    try:
                        ids.append(int(row['fingerprint_id']))
                    except ValueError:
                        pass
            return max(ids) + 1 if ids else 1
    except Exception:
        return 1

def enroll_finger_process(student_id_text, next_fid):
    """Handles the Serial communication with Arduino for enrollment."""
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # Open Connection
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # Wait for reboot
        
        # Send Command
        arduino.reset_input_buffer()
        command = f"ENROLL:{next_fid}\n"
        arduino.write(command.encode())
        
        status_text.info(f"Connecting to sensor... ")
        
        start_time = time.time()
        while True:
            # Timeout safety (30 seconds)
            if time.time() - start_time > 30:
                arduino.close()
                return False, "Timeout: Process took too long."

            if arduino.in_waiting > 0:
                line = arduino.readline().decode('utf-8', errors='ignore').strip()
                
                # --- PARSE ARDUINO RESPONSES ---
                if "PLACE_THUMB" in line:
                    status_text.warning("PLACE THUMB on the sensor now.")
                    progress_bar.progress(20)
                elif "REMOVE_THUMB" in line:
                    status_text.info("REMOVE THUMB.")
                    progress_bar.progress(50)
                elif "PLACE_AGAIN" in line:
                    status_text.warning("PLACE SAME THUMB again to confirm.")
                    progress_bar.progress(70)
                elif "SUCCESS_ENROLLED_ID" in line:
                    arduino.close()
                    progress_bar.progress(100)
                    return True, next_fid
                elif "ERROR" in line:
                    arduino.close()
                    return False, f"Sensor Error: {line}"
                    
    except serial.SerialException:
        return False, f"Could not connect to {SERIAL_PORT}. Is Arduino plugged in?"
    except Exception as e:
        return False, str(e)

# -------------------------
# Page: Register Student
# -------------------------
def page_register():
    st.header("Register Student & Auto-Capture 5 Images")
     # --- Initialize Session State for Fingerprint ---
    if 'fingerprint_success' not in st.session_state:
        st.session_state.fingerprint_success = False
    if 'fingerprint_id_assigned' not in st.session_state:
        st.session_state.fingerprint_id_assigned = None

    all_valid = True  # final flag

    student_id = st.text_input("Student ID", key="reg_id")
    if student_id:
        valid, msg = validate_student_id(student_id)
        if not valid:
            st.error(f"❌ {msg}")
            all_valid = False
        else:
            st.success("✅ Student ID format correct")

    student_name = uppercase_input("Student Name", "reg_name")
    if student_name:
        valid, msg = validate_student_name(student_name)
        if not valid:
            st.error(f"❌ {msg}")
            all_valid = False
        else:
            st.success("✅ Student name format correct")

    email = st.text_input("Email", key="reg_email")
    if email:
        valid, msg = validate_email(email)
        if not valid:
            st.error(f"❌ {msg}")
            all_valid = False
        else:
            st.success("✅ Email format correct")

    programme = st.text_input("Programme", key="reg_programme")
    if programme:
        valid, msg = validate_text_only(programme, "Programme")
        if not valid:
            st.error(f"❌ {msg}")
            all_valid = False
        else:
            st.success("✅ Programme format correct")

    faculty = st.text_input("Faculty", key="reg_faculty")
    if faculty:
        valid, msg = validate_text_only(faculty, "Faculty")
        if not valid:
            st.error(f"❌ {msg}")
            all_valid = False
        else:
            st.success("✅ Faculty format correct")

    # --- FINGERPRINT REGISTRATION --------  
    st.subheader("👆 Fingerprint Registration")
    
    # define a container to hold the fingerprint status
    fp_container = st.container()
    
    with fp_container:
        # Check if already enrolled in this session
        if st.session_state.fingerprint_success:
            st.success(f"Fingerprint Enrolled!")
            if st.button("Redo Fingerprint"):
                st.session_state.fingerprint_success = False
                st.session_state.fingerprint_id_assigned = None
                st.rerun()
        else:
            st.info("Click below to start fingerprint registration.")
            if st.button("Start Fingerprint Scan", type="primary"):
                # 1. Get next ID
                next_fid = get_next_fingerprint_id(CSV_FILE)
                
                # 2. Run Enrollment Process
                success, result = enroll_finger_process(student_id, next_fid)
                
                if success:
                    st.session_state.fingerprint_success = True
                    st.session_state.fingerprint_id_assigned = result
                    st.balloons()
                    st.rerun()
                else:
                    st.error(f"Enrollment Failed: {result}")

    st.write("---")
    st.subheader("📸 Student Face Registration")

    mode = st.radio("Select Input Method:", ["Upload 5 Images", "Capture Now"])
    
    if mode == "Upload 5 Images":
        st.info("Please upload 5 face images (different angles).")
        
        uploaded_files = st.file_uploader(
            "Upload Images",
            accept_multiple_files=True,
            type=["jpg", "jpeg", "png"],
            key="image_uploader"
        )
    
        if uploaded_files and len(uploaded_files) != 5:
            st.warning(f"You must upload exactly 5 images. Currently uploaded: {len(uploaded_files)}")
        elif uploaded_files and len(uploaded_files) == 5:
            st.success("All 5 images uploaded successfully!")
            
            # Clear previous captured images
            st.session_state.captured_images = []
            
            # Save files and add to captured_images
            for i, file in enumerate(uploaded_files):
                img = Image.open(file)
                filename = f"{student_id}_{i+1}.jpg"
                filepath = os.path.join("face_database", filename)
                img.save(filepath)
                
                # Add to session state
                st.session_state.captured_images.append(filename)
                st.write(f"✓ Saved: {filename}")
            
            st.success("Images saved to face_database/")
            add_status_message(f"5 images uploaded for {student_id}", 'success')
            if all_valid:
                st.balloons()

    
    elif mode == "Capture Now":
    
        # Use container to expand width
        st.markdown("""
        <style>
        .block-container {
            max-width: 95%;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Wider two-column layout: 65% camera, 35% instructions
        col1, col2 = st.columns([65, 35], gap="medium")
        
        with col1:
            st.info("📹 Live Camera Feed")
            ctx = webrtc_streamer(
                key="auto-capture",
                video_processor_factory=AutoCaptureTransformer,
                media_stream_constraints={
                    "video": {
                        "width": {"ideal": 1280},
                        "height": {"ideal": 720}
                    }, 
                    "audio": False
                },
            )
        
        with col2:
            st.info("📸 Capture Progress")
            
            instructions = [
                "1️⃣ Look straight at the camera",
                "2️⃣ Turn slightly left",
                "3️⃣ Turn slightly right", 
                "4️⃣ Look slightly upward",
                "5️⃣ Look slightly downward"
            ]
            
            # Create a container for all content
            with st.container():
                # Display instructions
                st.markdown("**Instructions:**")
                for instruction in instructions:
                    st.markdown(f"- {instruction}")
                
                st.markdown("---")
                
                # Show captured images status
                if st.session_state.captured_images:
                    st.success(f"✅ Captured: {len(st.session_state.captured_images)}/5 images")
                    for i, fname in enumerate(st.session_state.captured_images, 1):
                        st.markdown(f"✓ Image {i}: `{fname}`")
                else:
                    st.info("No images captured yet")
                
                st.markdown("---")
    
                if st.button("🎬 Start Auto Capture", use_container_width=True):
                    if not student_id:
                        st.error("Please enter Student ID first.")
                        add_status_message("Registration failed: No student ID", 'error')
                        text_to_speech("Please enter Student ID first")
                    elif not ctx.video_processor:
                        st.error("Camera not started yet.")
                        add_status_message("Camera not ready", 'error')
                        text_to_speech("Camera not started yet")
                    else:
                        st.session_state.captured_images = []
                        add_status_message(f"Starting capture for {student_id}", 'info')
                        
                        # Create placeholders INSIDE this column
                        with st.container():
                            status_placeholder = st.empty()
                            progress_bar = st.progress(0)
                            
                            # Initial announcement with delay
                            status_placeholder.info("🎬 Starting auto capture...")
                            text_to_speech("Starting auto capture. Please follow the instructions")
                            time.sleep(5)  # Wait 3 seconds for speech to finish
    
                            # Simplified voice instructions
                            voice_instructions = [
                                "Look straight",
                                "Turn left",
                                "Turn right", 
                                "Look up",
                                "Look down"
                            ]
    
                            for i in range(5):
                                # Announce instruction with voice
                                text_to_speech(voice_instructions[i])
                                
                                # Show current instruction
                                status_placeholder.info(f"📸 {instructions[i]}")
                                
                                # Countdown with voice for last 3 seconds
                                for sec in range(5, 0, -1):
                                    # Voice countdown for 3, 2, 1
                                    if sec == 3:
                                        text_to_speech("3")
                                    elif sec == 2:
                                        text_to_speech("2")
                                    elif sec == 1:
                                        text_to_speech("1")
                                    
                                    progress_bar.progress((5-sec) / 5)
                                    time.sleep(1)
                                
                                # Say "snap" right before capture
                                text_to_speech("snap")
                                time.sleep(0.2)  # Brief pause for "snap" to play
                                
                                # Capture frame - use ORIGINAL unflipped frame
                                frame = ctx.video_processor.frame_original
                                if frame is not None:
                                    filename = f"{student_id}_{i+1}.jpg"
                                    filepath = os.path.join("face_database", filename)
                                    cv2.imwrite(filepath, frame)
    
                                    st.session_state.captured_images.append(filename)
                                    status_placeholder.success(f"✅ Captured image {i+1}/5")
                                    progress_bar.progress((i+1) / 5)
                                    
                                    # Only voice feedback on last capture
                                    if i == 4:
                                        text_to_speech("All done!")
                                else:
                                    status_placeholder.error("❌ Failed to capture frame.")
                                    add_status_message(f"Capture failed at image {i+1}", 'error')
                                    text_to_speech("Capture failed")
                                    break
                                
                                # Brief pause between captures
                                time.sleep(0.5)
    
                            if len(st.session_state.captured_images) == 5:
                                status_placeholder.success("🎉 All 5 images captured successfully!")
                                progress_bar.progress(1.0)
                                add_status_message(f"All images captured for {student_id}", 'success')
                                if all_valid:
                                    st.balloons()
                                time.sleep(2)
                                st.rerun()

    if st.session_state.captured_images:
        st.write("Captured images:")
        for fname in st.session_state.captured_images:
            st.write(f"- {fname}")

    if len(st.session_state.captured_images) == 5:
        if st.button("Generate QR, Save Student & Email"):
            
            # Final validation check before saving
            if not all_valid:
                st.error("❌ Please fix all input errors before registering the student.")
                add_status_message("Registration failed: Invalid input", 'error')
                text_to_speech("Please fix all input errors before registration")
                return

            if not all([student_id, student_name, email, programme, faculty]):
                st.error("❌ Please fill in all fields")
                return
            
            # WARN if fingerprint is missing 
            if not st.session_state.get('fingerprint_success', False):
                st.error("❌ Fingerprint Enrollment is REQUIRED. Please enroll a finger first.")
                text_to_speech("Fingerprint data missing")
                return  # <--- STOP HERE if no fingerprint
            
            # If we pass the check, get the ID
            finger_id_to_save = st.session_state.fingerprint_id_assigned

            # Build student dict with full image paths
            student = {
                'student_id': student_id,
                'student_name': student_name,
                'email': email,
                'programme': programme,
                'faculty': faculty,
                'fingerprint_id': finger_id_to_save,
                'img1': f"face_database/{st.session_state.captured_images[0]}",
                'img2': f"face_database/{st.session_state.captured_images[1]}",
                'img3': f"face_database/{st.session_state.captured_images[2]}",
                'img4': f"face_database/{st.session_state.captured_images[3]}",
                'img5': f"face_database/{st.session_state.captured_images[4]}",
            }
            
            # Append to CSV
            with open(CSV_FILE, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writerow(student)
            
            # Generate QR
            qr_path = os.path.join("student_qr_codes", f"{student_id}.png")
            q = qrcode.make(student_id)
            q.save(qr_path)
            
            st.success(f"✅ Student saved and QR generated!")
            add_status_message(f"Student {student_id} registered successfully", 'success')
            
            # Display QR code
            col1, col2 = st.columns([1, 1])
            with col1:
                st.image(qr_path, width=250, caption=f"QR Code for {student_id}")
            
            with col2:
                # Download button for QR code
                with open(qr_path, "rb") as file:
                    st.download_button(
                        label="⬇️ Download QR Code",
                        data=file,
                        file_name=f"{student_id}_QR.png",
                        mime="image/png",
                        use_container_width=True
                    )
            
            # Send email (optional)
            try:
                if "EMAIL_ADDRESS" in st.secrets and "EMAIL_PASSWORD" in st.secrets:
                    EMAIL_ADDRESS = st.secrets["EMAIL_ADDRESS"]
                    EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]
                    msg = EmailMessage()
                    msg['Subject'] = "Your Graduation QR Code"
                    msg['From'] = EMAIL_ADDRESS
                    msg['To'] = email
                    msg.set_content(f"Hi {student_name},\n\nPlease find your graduation QR code attached.")
                    with open(qr_path, 'rb') as f:
                        msg.add_attachment(f.read(), maintype='image', subtype='png', filename=os.path.basename(qr_path))
                    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                        smtp.send_message(msg)
                    st.success("📧 Email sent successfully!")
                    add_status_message(f"Email sent to {email}", 'success')
                else:
                    st.info("📧 Email not configured. QR code saved locally.")
                    add_status_message("QR saved (email not configured)", 'info')
            except Exception as e:
                st.warning(f"⚠️ QR saved locally but email failed: {str(e)}")
                add_status_message(f"Email failed: {str(e)[:50]}", 'warning')

            # -----------------------------
            # Generate Embeddings Automatically
            # -----------------------------
            image_files = [
                os.path.join("face_database", os.path.basename(student['img1'])),
                os.path.join("face_database", os.path.basename(student['img2'])),
                os.path.join("face_database", os.path.basename(student['img3'])),
                os.path.join("face_database", os.path.basename(student['img4'])),
                os.path.join("face_database", os.path.basename(student['img5'])),
            ]
            
            st.info("Generating face embeddings... (This takes ~3 seconds)")
            
            ok, msg = save_student_embedding(
                student_id,
                student_name,
                email,
                programme,
                faculty,
                finger_id_to_save,
                image_files
            )
            
            if ok:
                st.success("Face embedding saved successfully!")
            else:
                st.error(f"Embedding failed: {msg}")
                return

            # Clear captured images and reset
            st.session_state.captured_images = []
            st.session_state.current_student_id = student_id
            
            # Option to continue or go to scan
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("➕ Register Another Student", use_container_width=True):
                    st.rerun()
            with col2:
                if st.button("➡️ Go to Scan QR", use_container_width=True):
                    st.session_state.page = "scan"
                    st.rerun()

# -------------------------
# Page: Scan QR (OpenCV Window)
# -------------------------
def page_scan():
    st.header("Scan Student QR Code")
    
    # Announce when entering page
    if 'qr_page_loaded' not in st.session_state:
        st.session_state.qr_page_loaded = True
        text_to_speech("Click the button to launch QR scanner")
    
    # Use wider layout
    st.markdown("""
    <style>
    .block-container {
        max-width: 95%;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.info("📱 Click the button below to open the QR scanner in a separate window")
    
    # Instructions
    with st.expander("📖 How to use the QR Scanner"):
        st.markdown("""
        1. Click the "Launch QR Scanner" button below
        2. A new camera window will open
        3. Hold your QR code steady in front of the camera
        4. Wait for green border (QR detected)
        5. Verify your Student ID on the camera window (5 seconds)
        6. Scanner will close automatically
        7. Check your details here and proceed to face verification
        8. Press 'Q' to cancel and close scanner manually
        """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🎥 Launch QR Scanner", use_container_width=True):
            text_to_speech("Opening QR scanner")
            
            # Show loading message
            with st.spinner("Opening camera window..."):
                # Clean up old scan results
                if os.path.exists('scanned_qr.json'):
                    os.remove('scanned_qr.json')
                
                # Run external scanner
                try:
                    result = subprocess.run(
                        ["python", "camera_qr_scan.py"],
                        capture_output=True,
                        text=True,
                        timeout=120  # 2 minute timeout
                    )
                    
                except subprocess.TimeoutExpired:
                    st.error("⚠️ Scanner timed out after 2 minutes")
                except Exception as e:
                    st.error(f"❌ Error launching scanner: {e}")
                
                # Read result
                if os.path.exists('scanned_qr.json'):
                    try:
                        with open('scanned_qr.json', 'r') as f:
                            data = json.load(f)
                        
                        if 'error' in data:
                            st.error(f"❌ Scanner Error: {data['error']}")
                            text_to_speech("Camera not available")
                        elif 'student_id' in data:
                            scanned_id = data['student_id']
                            st.session_state.scanned_id = scanned_id
                            
                            st.success(f"✅ QR Code Scanned Successfully!")
                            add_status_message(f"QR scanned: {scanned_id}", 'success')
                            
                            # Lookup student info
                            try:
                                df = pd.read_csv(CSV_FILE)
                                row = df[df['student_id'] == scanned_id]
                                
                                if not row.empty:
                                    student_name = row['student_name'].values[0]
                                    programme = row['programme'].values[0]
                                    faculty = row['faculty'].values[0]
                                    email = row['email'].values[0]
                                    
                                    # Display info in WIDE, attractive format
                                    st.markdown("---")
                                    st.markdown("## 📋 Student Information Verification")
                                    st.markdown("### Please verify your details below:")
                                    
                                    # Create attractive info cards
                                    st.markdown("""
                                    <style>
                                    .info-card {
                                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                        padding: 20px;
                                        border-radius: 10px;
                                        color: white;
                                        margin: 10px 0;
                                    }
                                    .info-label {
                                        font-size: 14px;
                                        opacity: 0.9;
                                        margin-bottom: 5px;
                                    }
                                    .info-value {
                                        font-size: 24px;
                                        font-weight: bold;
                                    }
                                    </style>
                                    """, unsafe_allow_html=True)
                                    
                                    # Two column layout for info
                                    col_a, col_b = st.columns(2, gap="large")
                                    
                                    with col_a:
                                        st.markdown(f"""
                                        <div class="info-card">
                                            <div class="info-label">👤 Student Name</div>
                                            <div class="info-value">{student_name}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        st.markdown(f"""
                                        <div class="info-card">
                                            <div class="info-label">🎓 Programme</div>
                                            <div class="info-value">{programme}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    with col_b:
                                        st.markdown(f"""
                                        <div class="info-card">
                                            <div class="info-label">🆔 Student ID</div>
                                            <div class="info-value">{scanned_id}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                        
                                        st.markdown(f"""
                                        <div class="info-card">
                                            <div class="info-label">🏛️ Faculty</div>
                                            <div class="info-value">{faculty}</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    
                                    # Email in full width
                                    st.markdown(f"""
                                    <div class="info-card">
                                        <div class="info-label">📧 Email Address</div>
                                        <div class="info-value" style="font-size: 20px;">{email}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    st.markdown("---")
                                    
                                    # Voice announcement
                                    text_to_speech(f"Welcome {student_name}. Please verify your details")
                                    add_status_message(f"Student found: {student_name}", 'info')
                                    
                                    # Countdown with progress bar
                                    st.info("⏳ Please verify your details above...")
                                    progress_bar = st.progress(0)
                                    countdown_text = st.empty()
                                    
                                    for i in range(10, 0, -1):
                                        countdown_text.markdown(f"### ⏱️ Proceeding to face verification in {i} seconds...")
                                        progress_bar.progress((10 - i) / 10)
                                        time.sleep(1)
                                    
                                    countdown_text.empty()
                                    progress_bar.empty()
                                    
                                    text_to_speech("Proceeding to face verification")
                                    
                                    # Clean up and redirect
                                    os.remove('scanned_qr.json')
                                    st.session_state.qr_page_loaded = False
                                    st.session_state.page = "face"
                                    st.rerun()
                                    
                                else:
                                    st.warning(f"⚠️ Student ID '{scanned_id}' not found in database")
                                    text_to_speech("Student ID not found")
                                    add_status_message(f"Student ID {scanned_id} not found", 'warning')
                                    os.remove('scanned_qr.json')
                                    
                            except Exception as e:
                                st.error(f"❌ Database Error: {e}")
                                if os.path.exists('scanned_qr.json'):
                                    os.remove('scanned_qr.json')
                        
                    except json.JSONDecodeError:
                        st.error("❌ Error reading scan result")
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {e}")
                else:
                    st.warning("⚠️ No QR code was scanned (window may have been closed)")
                    text_to_speech("Scan cancelled")
    
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.qr_page_loaded = False
            if os.path.exists('scanned_qr.json'):
                os.remove('scanned_qr.json')
            st.rerun()
    
    # Show recent activity
    if st.session_state.scanned_id:
        st.markdown("---")
        st.info(f"💡 Last scanned ID: **{st.session_state.scanned_id}**")
        
# -------------------------
# Face recognition models loading
# -------------------------
@st.cache_resource
def initialize_models_and_data():

    try:
        embedder = FaceNet()
        detector = MTCNN()
        mask_model = load_model(MASK_MODEL_PATH)
        
        # 1. Define the full list of columns you expect now
        expected_columns = [
            'student_id', 'student_name', 'email', 'programme', 'faculty', 
            'fingerprint_id', 'img1', 'img2', 'img3', 'img4', 'img5', 'embedding'
        ]

        if not os.path.exists(EMBEDDED_CSV):
            st.warning(f"Embedding CSV not found: {EMBEDDED_CSV}")
            df = pd.DataFrame(columns=expected_columns)
            db_embeddings = []
        else:
            df = pd.read_csv(EMBEDDED_CSV)
            db_embeddings = []

            # 2. Check if 'embedding' column actually exists
            if 'embedding' in df.columns:
                for emb in df['embedding']:
                    # 3. Handle Empty/NaN rows (Skip them instead of crashing)
                    if pd.isna(emb) or str(emb).strip() == "":
                        db_embeddings.append(None)
                        continue
                    
                    try:
                        # 4. Clean the string: Remove brackets [] and whitespace
                        clean_emb = str(emb).replace('[', '').replace(']', '').strip()
                        
                        # 5. Convert to float array
                        db_embeddings.append(np.array(list(map(float, clean_emb.split(',')))))
                    except ValueError:
                        # If data is corrupted (e.g., text instead of numbers), treat as None
                        db_embeddings.append(None)
            else:
                # If column is missing entirely
                db_embeddings = [None] * len(df)

        return embedder, detector, mask_model, df, db_embeddings
        
    except Exception as e:
        st.error(f"Error initializing models: {e}")
        st.stop()

EMBEDDER, DETECTOR, MASK_MODEL, STUDENT_DF, DB_EMBEDDINGS = initialize_models_and_data()

def get_embedding(face_pixels): #Added
    """Extract embedding with same preprocessing as registration"""
    # Ensure correct size
    if face_pixels.shape != (160, 160, 3):
        face_pixels = cv2.resize(face_pixels, (160, 160))
    
    # Normalize the same way as during registration
    face_pixels = face_pixels.astype('float32')
    face_pixels = (face_pixels - face_pixels.mean()) / (face_pixels.std() + 1e-6)
    
    # Add batch dimension
    face_pixels = np.expand_dims(face_pixels, axis=0)
    
    return EMBEDDER.embeddings(face_pixels)[0]

def verify_face(live_emb):
    min_dist = float("inf")
    identity_index = -1
    for i, db_emb in enumerate(DB_EMBEDDINGS):
        if db_emb is None:
            continue
        d = euclidean(live_emb, db_emb)
        if d < min_dist:
            min_dist = d
            identity_index = i
    if min_dist < 0.83 and identity_index >= 0:
        row = STUDENT_DF.loc[identity_index].to_dict()
        row['distance'] = min_dist
        return row
    return None

class FaceScannerTransformer(VideoTransformerBase):
    def __init__(self):
        self.match_data = {'status': 'Awaiting Face'}
        self.lock = threading.Lock()
        self.last_logged_id = None

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        
        # Process with original orientation for face detection
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = DETECTOR.detect_faces(rgb)
        
        if len(faces) == 1:
            x, y, w, h = faces[0]['box']
            x, y = max(0, x), max(0, y)
            crop = img[y:y+h, x:x+w]
            if crop.size != 0:
                mask_input = cv2.resize(crop, (224, 224))
                mask_input = img_to_array(mask_input) / 255.0
                mask_input = np.expand_dims(mask_input, 0)
                with tf.device('/cpu:0'):
                    pred = MASK_MODEL.predict(mask_input, verbose=0)
                if np.argmax(pred) == 0:
                    status = "TAKE OFF MASK"
                    color = (0,0,255)
                    with self.lock:
                        self.match_data = {'status': status}
                else:
                    facenet_input = cv2.resize(crop, (160,160))
                    facenet_input = cv2.cvtColor(facenet_input, cv2.COLOR_BGR2RGB)
                    live_emb = get_embedding(facenet_input)
                    match = verify_face(live_emb)
                    if match is not None:
                        with self.lock:
                            match['status'] = "CONFIRMED"
                            self.match_data = match
                            
                            if self.last_logged_id != match.get('student_id'):
                                log_attendance(match, '✓')
                                self.last_logged_id = match.get('student_id')
                        
                        color = (0,255,0)
                        status = f"CONFIRMED: {match.get('student_name','')}"
                    else:
                        with self.lock:
                            self.match_data = {'status': 'UNKNOWN'}
                        color = (0,0,255)
                        status = "UNKNOWN IDENTITY"
                
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                cv2.putText(img, status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        elif len(faces) > 1:
            with self.lock:
                self.match_data = {'status': 'ERROR: Multiple Faces'}
            cv2.putText(img, "MULTIPLE FACES", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        # Return flipped image for mirror effect display
        return cv2.flip(img, 1)

import serial
import time
import streamlit as st

# MUST MATCH ARDUINO "Serial.begin(9600)"
SERIAL_PORT = 'COM3'  # <--- CHECK YOUR PORT (Mac/Linux: /dev/ttyUSB0)
BAUD_RATE = 9600      

def verify_finger_process(expected_fid):
    """
    Communicates with IPsk.ino to verify a specific fingerprint ID.
    Matches Arduino function: handleSpecificVerification(int expectedID)
    """
    status_text = st.empty()
    progress_bar = st.progress(0)
    
    arduino = None
    try:
        # 1. Open Connection
        # ----------------------------------------
        # print(f"🔌 Connecting to {SERIAL_PORT}...") # Uncomment for debug
        arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        
        # 2. Handshake / Wait for Reboot
        # ----------------------------------------
        # Arduino prints "STATUS:Sensor_Connected_OK" on setup.
        # We wait 3 seconds to let it finish setup, then clear the buffer.
        time.sleep(3) 
        arduino.reset_input_buffer()
        
        # 3. Send Command
        # ----------------------------------------
        command = f"VERIFY:{expected_fid}\n"
        arduino.write(command.encode())
        
        status_text.info(f"⚠️ Face Auth Failed. Place finger (Expected ID: {expected_fid})...")
        
        # 4. Listen for Response
        # ----------------------------------------
        start_time = time.time()
        while True:
            # Global Timeout (30s)
            if time.time() - start_time > 30:
                return False, "Timeout: Sensor did not respond."

            if arduino.in_waiting > 0:
                line = arduino.readline().decode('utf-8', errors='ignore').strip()
                
                # Debug: See exactly what Arduino is sending
                if line:
                    print(f"📥 Arduino: {line}") 

                # --- MATCHING LOGIC (Based on your IPsk.ino) ---

                # Arduino: Serial.println("PLACE_THUMB");
                if "PLACE_THUMB" in line:
                    status_text.warning("👉 Place your finger on the sensor...")
                    progress_bar.progress(30)
                
                # Arduino: Serial.println("VERIFYING");
                elif "VERIFYING" in line:
                    status_text.info("🔄 Verifying fingerprint...")
                    progress_bar.progress(60)
                
                # Arduino: Serial.print("SUCCESS_VERIFIED:"); Serial.println(finger.fingerID);
                elif "SUCCESS_VERIFIED" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        found_id = int(parts[1])
                        
                        # Double check ID match on Python side just in case
                        if found_id == int(expected_fid):
                            progress_bar.progress(100)
                            return True, found_id
                        else:
                            return False, f"ID Mismatch (Scanned {found_id}, Expected {expected_fid})"
                
                # Arduino: Serial.print("NOT_MATCH:Expected_");
                elif "NOT_MATCH" in line:
                    return False, "❌ Fingerprint does not match this student."
                
                # Arduino: Serial.println("FAILED:No_Match_In_Database");
                elif "FAILED" in line or "No_Match_In_Database" in line:
                    return False, "❌ Fingerprint not found in database."
                
                # Arduino: Serial.println("ERROR:No_Finger_Detected_Timeout");
                elif "No_Finger_Detected" in line:
                    return False, "❌ Time limit exceeded (No finger)."

    except serial.SerialException:
        return False, f"Could not connect to {SERIAL_PORT}. Close Arduino IDE and retry."
    except Exception as e:
        return False, f"Error: {str(e)}"
    finally:
        if arduino and arduino.is_open:
            arduino.close()

# -------------------------
# UPDATED: Page Face
# -------------------------
# -------------------------
# Page: Face Verification (Original Logic + Fallback)
# -------------------------
def page_face():
    st.header("Face Verification")

    # --- INSTRUCTIONS (Original Style) ---
    st.markdown("""
    <style>
    .info-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 10px;
    }
    .info-label { font-weight: bold; color: #555; }
    .info-value { font-size: 1.2em; color: #000; }
    </style>
    """, unsafe_allow_html=True)

    st.info("""
    **Instructions:**
    1. Click 'Launch Face Verification'
    2. Look at the camera
    3. If face verification fails, you will be prompted to use your fingerprint.
    """)

    # Check if ID is scanned (Original check logic)
    if 'scanned_id' not in st.session_state or not st.session_state.scanned_id:
        st.warning("⚠️ No Student ID. Please Scan QR first.")
        # Optional debug input
        # tid = st.text_input("DEBUG: Enter ID manually") ...
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("🎥 Launch Face Verification", use_container_width=True):
            text_to_speech("Opening face verification system")
            
            with st.spinner("Opening camera window..."):
                # Clean up old results
                if os.path.exists('face_verify_result.json'):
                    os.remove('face_verify_result.json')

                # Run external face verification script (Using the ORIGINAL WORKING FILE)
                failed_or_timeout = False
                try:
                   
                    cmd = [sys.executable, "camera_face_verify.py", str(st.session_state.scanned_id)]
                    
                    # Original timeout logic (or slightly increased to be safe)
                    subprocess.run(cmd, check=True, timeout=120)
                
                except subprocess.TimeoutExpired:
                    st.error("⚠️ Face verification timed out.")
                    failed_or_timeout = True
                except Exception as e:
                    st.error(f"❌ Error launching face verification: {e}")
                    failed_or_timeout = True

                # --- PROCESS RESULTS ---
                face_success = False
                if not failed_or_timeout and os.path.exists('face_verify_result.json'):
                    try:
                        with open('face_verify_result.json', 'r') as f:
                            data = json.load(f)

                        # --- SCENARIO A: ERROR IN SCRIPT ---
                        if 'error' in data:
                            st.error(f"❌ Verification Error: {data['error']}")
                            text_to_speech("Camera error")
                            failed_or_timeout = True

                        # --- SCENARIO B: FACE VERIFICATION SUCCESS ---
                        elif data.get('status') == 'success':
                            face_success = True
                            student_name = data.get('student_name', 'Unknown')
                            programme = data.get('programme', '-')
                            
                            st.success("✅ Face Verification Successful!")
                            add_status_message(f"Face verified: {student_name}", 'success')
                            text_to_speech(f"Welcome {student_name}")

                            # Display Success UI (Original Style)
                            st.markdown("---")
                            st.markdown("## ✅ Identity Confirmed")
                            
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.markdown(f"""
                                <div class="info-card">
                                <div class="info-label">👤 Student Name</div>
                                <div class="info-value">{student_name}</div>
                                </div>
                                """, unsafe_allow_html=True)
                            with col_b:
                                st.markdown(f"""
                                <div class="info-card">
                                <div class="info-label">🎓 Programme</div>
                                <div class="info-value">{programme}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                            # Log attendance
                            log_attendance(data, '✓')

                        # --- SCENARIO C: EXPLICIT FAILURE (Unknown Face) ---
                        else:
                            st.error("❌ Face Verification Failed.")
                            failed_or_timeout = True

                    except json.JSONDecodeError:
                        st.error("❌ Error reading verification result file")
                        failed_or_timeout = True
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {e}")
                        failed_or_timeout = True
                elif not failed_or_timeout:
                    st.warning("⚠️ No verification result found (Window closed?)")
                    failed_or_timeout = True

                # --- FINGERPRINT FALLBACK LOGIC ---
                # This runs ONLY if face verification failed or timed out
                if not face_success or failed_or_timeout:
                    st.markdown("---")
                    st.subheader("👆 Fingerprint Fallback")
                    text_to_speech("Face verification failed. Please use fingerprint.")
                    
                    # 1. Get current ID
                    current_id = st.session_state.scanned_id
                    
                    if current_id:
                        # 2. Look up Fingerprint ID
                        try:
                            df = pd.read_csv(CSV_FILE)
                            # Ensure string match
                            df['student_id'] = df['student_id'].astype(str)
                            student_row = df[df['student_id'] == str(current_id)]
                            
                            if not student_row.empty:
                                fid = student_row.iloc[0]['fingerprint_id']
                                
                                if pd.notna(fid) and str(fid).strip() != "":
                                    # 3. Call Helper Function
                                    # (Ensure verify_finger_process is defined in test3.py)
                                    fp_success, msg = verify_finger_process(int(fid))
                                    
                                    if fp_success:
                                        st.success("✅ Fingerprint Verified Successfully!")
                                        text_to_speech("Fingerprint verified")
                                        
                                        student_data = student_row.iloc[0].to_dict()
                                        if log_attendance(student_data, '✓'):
                                            st.success("Attendance Logged.")
                                            add_status_message(f"Fingerprint verified: {student_data['student_name']}", 'success')
                                        
                                        st.info(f"Identity confirmed for: {student_data['student_name']}")
                                        
                                    else:
                                        st.error(f"❌ Verification Failed: {msg}")
                                        text_to_speech("Access denied")
                                        add_status_message(f"Auth failed for {current_id}", 'error')
                                else:
                                    st.warning("⚠️ No fingerprint registered for this student.")
                            else:
                                st.error("Student ID not found in database.")
                        except Exception as e:
                            st.error(f"Database Error: {e}")

    with col2:
        if st.button("🔄 Reset / Clear", use_container_width=True):
            if os.path.exists('face_verify_result.json'):
                os.remove('face_verify_result.json')
            st.rerun()

    # Show recent activity
    st.markdown("---")
    if st.session_state.scanned_id:
        st.info(f"💡 Current Session ID: **{st.session_state.scanned_id}**")

# -------------------------
# Page: Attendance Dashboard
# -------------------------
def page_dashboard():
    st.header("📊 Live Attendance Dashboard")
    
    # Use wider layout
    st.markdown("""
    <style>
    .block-container {
        max-width: 98%;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Refresh and download buttons
    col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    with col2:
        # Download attendance CSV
        if st.button("📥 Download Attendance", use_container_width=True):
            try:
                if os.path.exists(ATTENDANCE_CSV):
                    df_att = pd.read_csv(ATTENDANCE_CSV)
                    csv_data = df_att.to_csv(index=False)
                    st.download_button(
                        label="Download",
                        data=csv_data,
                        file_name=f"attendance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_attendance"
                    )
                else:
                    st.warning("No attendance records yet")
            except Exception as e:
                st.error(f"Error: {e}")
    with col3:
        # Download all students CSV
        if st.button("📥 Download Students", use_container_width=True):
            try:
                df_students = pd.read_csv(CSV_FILE)
                csv_data = df_students.to_csv(index=False)
                st.download_button(
                    label="Download",
                    data=csv_data,
                    file_name=f"students_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="download_students"
                )
            except Exception as e:
                st.error(f"Error: {e}")
    
    try:
        # Load all registered students
        df_students = pd.read_csv(CSV_FILE)
        
        # Load attendance records
        if os.path.exists(ATTENDANCE_CSV):
            df_attendance = pd.read_csv(ATTENDANCE_CSV)
        else:
            df_attendance = pd.DataFrame(columns=ATTENDANCE_FIELDS)
        
        if df_students.empty:
            st.warning("No students registered yet.")
            return
        
        # Create combined dataframe: ALL students with their attendance status
        # Get latest verification for each student
        if not df_attendance.empty:
            # Sort by time and keep only the latest entry per student
            df_attendance_sorted = df_attendance.sort_values('verification_time', ascending=False)
            df_latest_attendance = df_attendance_sorted.drop_duplicates(subset=['student_id'], keep='first')
        else:
            df_latest_attendance = pd.DataFrame(columns=ATTENDANCE_FIELDS)
        
        # Merge students with their attendance status
        df_combined = df_students[['student_id', 'student_name', 'programme', 'faculty']].copy()
        
        # Add attendance status and time
        df_combined['status'] = '✗'  # Default: Not verified
        df_combined['verification_time'] = '-'
        
        # Update with actual attendance data
        for idx, student_row in df_combined.iterrows():
            student_id = student_row['student_id']
            attendance_record = df_latest_attendance[df_latest_attendance['student_id'] == student_id]
            
            if not attendance_record.empty:
                df_combined.at[idx, 'status'] = attendance_record.iloc[0]['status']
                df_combined.at[idx, 'verification_time'] = attendance_record.iloc[0]['verification_time']
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        total_students = len(df_combined)
        verified_count = len(df_combined[df_combined['status'] == '✓'])
        not_verified_count = total_students - verified_count
        verification_rate = (verified_count / total_students * 100) if total_students > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Students", total_students)
        with col2:
            st.metric("✅ Verified", verified_count)
        with col3:
            st.metric("❌ Not Verified", not_verified_count)
        with col4:
            st.metric("Verification Rate", f"{verification_rate:.1f}%")
        
        st.markdown("---")
        
        # Sort and filter options
        st.subheader("📋 All Students Attendance Status")
        
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            sort_by = st.selectbox(
                "Sort by:",
                ["Time (Latest First)", "Time (Oldest First)", "Student ID", "Student Name", "Status (Verified First)", "Status (Unverified First)"],
                index=0
            )
        with col2:
            filter_status = st.selectbox(
                "Filter by Status:",
                ["All", "✓ Verified Only", "✗ Not Verified Only"],
                index=0
            )
        with col3:
            filter_faculty = st.multiselect(
                "Filter by Faculty:",
                options=df_combined['faculty'].unique().tolist()
            )
        
        # Apply filters
        df_filtered = df_combined.copy()
        
        if filter_status == "✓ Verified Only":
            df_filtered = df_filtered[df_filtered['status'] == '✓']
        elif filter_status == "✗ Not Verified Only":
            df_filtered = df_filtered[df_filtered['status'] == '✗']
        
        if filter_faculty:
            df_filtered = df_filtered[df_filtered['faculty'].isin(filter_faculty)]
        
        # Apply sorting
        if sort_by == "Time (Latest First)":
            # Put unverified at bottom, sort verified by time
            df_filtered['sort_key'] = df_filtered['verification_time'].apply(
                lambda x: '0' if x == '-' else '1' + x
            )
            df_filtered = df_filtered.sort_values('sort_key', ascending=False)
            df_filtered = df_filtered.drop('sort_key', axis=1)
        elif sort_by == "Time (Oldest First)":
            df_filtered['sort_key'] = df_filtered['verification_time'].apply(
                lambda x: 'ZZZZ' if x == '-' else x
            )
            df_filtered = df_filtered.sort_values('sort_key', ascending=True)
            df_filtered = df_filtered.drop('sort_key', axis=1)
        elif sort_by == "Student ID":
            df_filtered = df_filtered.sort_values('student_id')
        elif sort_by == "Student Name":
            df_filtered = df_filtered.sort_values('student_name')
        elif sort_by == "Status (Verified First)":
            df_filtered = df_filtered.sort_values('status', ascending=False)
        elif sort_by == "Status (Unverified First)":
            df_filtered = df_filtered.sort_values('status', ascending=True)
        
        # Display dataframe with styling
        st.dataframe(
            df_filtered,
            use_container_width=True,
            hide_index=True,
            column_config={
                "student_id": st.column_config.TextColumn("Student ID", width="medium"),
                "student_name": st.column_config.TextColumn("Name", width="large"),
                "programme": st.column_config.TextColumn("Programme", width="medium"),
                "faculty": st.column_config.TextColumn("Faculty", width="medium"),
                "status": st.column_config.TextColumn("Status", width="small"),
                "verification_time": st.column_config.TextColumn("Verification Time", width="medium"),
            },
            height=500  # Set fixed height for scrolling
        )
        
        st.caption(f"Showing {len(df_filtered)} of {total_students} students")
        
    except FileNotFoundError as e:
        st.error(f"Required CSV file not found: {e}")
    except Exception as e:
        st.error(f"Error loading dashboard data: {e}")
        st.exception(e)

# -------------------------
# Page: Capture
# -------------------------
def page_capture():
    st.header("📸 Stage Photo Capture")
    
    # Announce when entering page
    if 'capture_page_loaded' not in st.session_state:
        st.session_state.capture_page_loaded = True
        text_to_speech("Stage photo capture ready. Click the button to launch camera")
    
    # Wide layout
    st.markdown("""
    <style>
    .block-container {
        max-width: 95%;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.info("📸 Click the button below to open the stage photo capture window")
    
    # Instructions
    with st.expander("📖 How to use Stage Photo Capture"):
        st.markdown("""
        1. Click the "Launch Stage Photo Capture" button below
        2. A new camera window will open
        3. **Position two people** with **certificate visible** between them
        4. **Both people must look at the camera** (green box = forward, orange = not forward)
        5. System **auto-detects** when ready
        6. **3-second countdown** starts automatically with voice "Look at camera and smile"
        7. **Photo captures automatically** after countdown
        8. **Auto-reset after 15 seconds** OR when people leave the frame
        9. Press 'Q' to close manually
        """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("🎥 Launch Stage Photo Capture", use_container_width=True):
            text_to_speech("Opening stage photo capture")
            
            with st.spinner("Opening camera window..."):
                # Clean up old results
                if os.path.exists('stage_capture_result.json'):
                    os.remove('stage_capture_result.json')
                
                # Run external capture
                try:
                    result = subprocess.run(
                        ["python", "camera_stage_capture.py"],
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minute timeout
                    )
                    
                except subprocess.TimeoutExpired:
                    st.error("⚠️ Stage capture timed out after 5 minutes")
                except Exception as e:
                    st.error(f"❌ Error launching stage capture: {e}")
                
                # Read result
                if os.path.exists('stage_capture_result.json'):
                    try:
                        with open('stage_capture_result.json', 'r') as f:
                            data = json.load(f)
                        
                        if 'error' in data:
                            st.error(f"❌ Capture Error: {data['error']}")
                            text_to_speech("Camera not available")
                        elif data.get('status') == 'success':
                            photo_path = data['photo_path']
                            timestamp = data['timestamp']
                            
                            st.success("✅ Stage Photo Captured Successfully!")
                            add_status_message(f"Stage photo captured", 'success')
                            text_to_speech("Photo captured successfully")
                            
                            # Display results
                            st.markdown("---")
                            st.markdown("## 📸 Captured Photo")
                            
                            col_a, col_b = st.columns([2, 1])
                            
                            with col_a:
                                st.image(photo_path, caption=f"Captured at {timestamp}", 
                                        use_container_width=True)
                            
                            with col_b:
                                st.info("**Photo Details:**")
                                st.write(f"📁 **File:** `{os.path.basename(photo_path)}`")
                                st.write(f"🕐 **Time:** {timestamp}")
                                st.write(f"📂 **Folder:** `stage_photos/`")
                                
                                # Download button
                                with open(photo_path, "rb") as file:
                                    st.download_button(
                                        label="⬇️ Download Photo",
                                        data=file,
                                        file_name=os.path.basename(photo_path),
                                        mime="image/jpeg",
                                        use_container_width=True
                                    )
                            
                            st.balloons()
                            
                            # Clean up
                            os.remove('stage_capture_result.json')
                            
                        elif data.get('status') == 'cancelled':
                            st.warning("⚠️ Capture cancelled (window was closed)")
                            text_to_speech("Capture cancelled")
                            os.remove('stage_capture_result.json')
                        
                    except json.JSONDecodeError:
                        st.error("❌ Error reading capture result")
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {e}")
                else:
                    st.warning("⚠️ No result file found (window may have been closed)")
                    text_to_speech("Capture cancelled")
    
    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.capture_page_loaded = False
            if os.path.exists('stage_capture_result.json'):
                os.remove('stage_capture_result.json')
            st.rerun()
    
    # Show recent captures
    st.markdown("---")
    st.subheader("📂 Recent Stage Photos")
    
    if os.path.exists("stage_photos"):
        photos = sorted([f for f in os.listdir("stage_photos") if f.endswith('.jpg')], 
                       reverse=True)[:6]  # Show last 6 photos
        
        if photos:
            cols = st.columns(3)
            for idx, photo in enumerate(photos):
                with cols[idx % 3]:
                    photo_path = os.path.join("stage_photos", photo)
                    st.image(photo_path, caption=photo, use_container_width=True)
        else:
            st.info("No photos captured yet")
    else:
        st.info("No photos captured yet")

# -------------------------
# Page: Graduation Photo Capture 
# -------------------------
def page_graduation_capture():
    st.markdown("""
    <style>
    .graduation-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    .graduation-title {
        font-size: 48px;
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .graduation-subtitle {
        font-size: 20px;
        margin-top: 10px;
        opacity: 0.95;
    }
.feature-card-compact {
        background: white;
        padding: 12px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 8px 0;
        border-left: 3px solid #667eea;
        min-height: 80px;
    }
    .feature-icon {
        font-size: 24px;
        text-align: center;
        margin-bottom: 5px;
    }
    .feature-title-compact {
        font-size: 14px;
        font-weight: bold;
        color: #333;
        margin-bottom: 3px;
        text-align: center;
    }
    .feature-desc-compact {
        color: #666;
        font-size: 11px;
        text-align: center;
    }
    .launch-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 24px;
        font-weight: bold;
        padding: 20px 40px;
        border-radius: 50px;
        border: none;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        cursor: pointer;
        transition: all 0.3s;
    }
    .status-box {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border: 2px dashed #667eea;
        text-align: center;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Beautiful graduation header
    st.markdown("""
    <div class="graduation-header">
        <div class="graduation-title">🎓 Graduation Photo Capture</div>
    </div>
    """, unsafe_allow_html=True)
    
# Announce when entering page
    if 'grad_capture_page_loaded' not in st.session_state:
        st.session_state.grad_capture_page_loaded = True
        text_to_speech("Welcome to graduation photo capture system")
    
    # Wide layout
    st.markdown("""
    <style>
    .block-container {
        max-width: 95%;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Two column layout: Instructions + Launch
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### ✨ System Features")
        
        # Compact feature cards - 2 columns
        feat_col1, feat_col2 = st.columns(2)
        
        with feat_col1:
            st.markdown("""
            <div class="feature-card-compact">
                <div class="feature-icon">🖐️</div>
                <div class="feature-title-compact">Palm Detection</div>
                <div class="feature-desc-compact">Show 5 fingers to start</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card-compact">
                <div class="feature-icon">⭕</div>
                <div class="feature-title-compact">Countdown</div>
                <div class="feature-desc-compact">Circular animation</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card-compact">
                <div class="feature-icon">🔴</div>
                <div class="feature-title-compact">Laser Navigation</div>
                <div class="feature-desc-compact">Point to select buttons</div>
            </div>
            """, unsafe_allow_html=True)
        
        with feat_col2:
            st.markdown("""
            <div class="feature-card-compact">
                <div class="feature-icon">⏱️</div>
                <div class="feature-title-compact">3-Second Hold</div>
                <div class="feature-desc-compact">Keep palm steady</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="feature-card-compact">
                <div class="feature-icon">📧</div>
                <div class="feature-title-compact">Email Delivery</div>
                <div class="feature-desc-compact">Photos sent via email</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 How to Use")

        # Instructions expander
        with st.expander("📖 Detailed Instructions", expanded=False):
            st.markdown("""
            **Step-by-Step Guide:**
            
            1. **Launch System** 🚀
               - Click the "Launch Graduation Capture" button below
               - A new camera window will open
            
            2. **Start Capture** 🖐️
               - Show your PALM with all 5 fingers visible
               - System will detect and show palm lines and accuracy
            
            3. **Hold Steady** ⏱️
               - Keep your palm in position for 3 seconds
               - Progress bar will show your progress
            
            4. **Smile!** 😊
               - After countdown, photo captures automatically
               - Audio will announce "Photo captured!"
            
            5. **Review Photo** 📸
               - Your photo will be displayed
               - Hover your hand over "DOWNLOAD" or "CANCEL"
            
            6. **Scan QR Code** 📱
               - If downloading, scan your student QR code
               - Confirm your identity
            
            7. **Receive Photo** 📧
               - Photo will be sent to your registered email
               - Success screen will show confirmation
            
            **Controls:**
            - Press **'Q'** at any time to quit the system
            - System auto-resets after each capture
            """)
        
        st.markdown("---")
        
        # Launch button
        if st.button("🎥 Launch Graduation Capture System", 
                    use_container_width=True, 
                    type="primary"):
            
            text_to_speech("Launching graduation photo capture system")
            add_status_message("Opening graduation capture", 'info')
            
            st.info("🎬 Opening camera window...")
            
            # Clean up old results
            if os.path.exists('graduation_capture_result.json'):
                os.remove('graduation_capture_result.json')
            
            # Run your graduation capture script
            try:
                result = subprocess.run(
                    ["jupyter", "nbconvert", "--to", "notebook", "--execute", "handGestureSystem.ipynb"],  
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 minute timeout
                )
                
            except subprocess.TimeoutExpired:
                st.error("⚠️ System timed out after 10 minutes")
                text_to_speech("System timeout")
                add_status_message("Capture timeout", 'error')
                
            except Exception as e:
                st.error(f"❌ Error launching system: {e}")
                text_to_speech("System error")
                add_status_message(f"Launch error: {str(e)[:50]}", 'error')
            
            # Check for results
            if os.path.exists('graduation_capture_result.json'):
                try:
                    with open('graduation_capture_result.json', 'r') as f:
                        data = json.load(f)
                    
                    if 'error' in data:
                        st.error(f"❌ System Error: {data['error']}")
                        text_to_speech("System error occurred")
                        add_status_message(f"Error: {data['error']}", 'error')
                        
                    elif data.get('status') == 'success':
                        st.success("✅ Graduation capture session completed!")
                        text_to_speech("Session completed successfully")
                        add_status_message("Capture session completed", 'success')
                        
                        # Show stats if available
                        if 'photos_captured' in data:
                            st.info(f"📸 Photos captured: {data['photos_captured']}")
                        if 'emails_sent' in data:
                            st.info(f"📧 Emails sent: {data['emails_sent']}")
                        
                        st.balloons()
                        
                    elif data.get('status') == 'cancelled':
                        st.warning("⚠️ System was closed by user")
                        text_to_speech("System closed")
                        add_status_message("User closed system", 'warning')
                    
                    # Clean up result file
                    os.remove('graduation_capture_result.json')
                    
                except json.JSONDecodeError:
                    st.error("❌ Error reading system result")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")
            else:
                # If no result file, system was likely just closed
                st.info("ℹ️ System window was closed")
                text_to_speech("System closed")

        
    # Bottom section: Recent captures or tips
    st.markdown("---")
    st.markdown("### 📂 Recent Captures")
    
    if os.path.exists("result/captures"):
        captures = sorted([f for f in os.listdir("result/captures") 
                          if f.endswith('.jpg')], reverse=True)[:6]
        
        if captures:
            cols = st.columns(3)
            for idx, photo in enumerate(captures):
                with cols[idx % 3]:
                    photo_path = os.path.join("result/captures", photo)
                    st.image(photo_path, caption=photo, use_container_width=True)
        else:
            st.info("📭 No captures yet. Launch the system to start capturing!")
    else:
        st.info("📭 Captures folder not found. System will create it on first capture.")
    
    # Tips section
    st.markdown("---")
    st.markdown("### 💡 Pro Tips")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **🖐️ Palm Detection**
        - Ensure good lighting
        - Keep hand steady
        - Show all 5 fingers clearly
        """)
    
    with col2:
        st.info("""
        **📸 Photo Quality**
        - Look directly at camera
        - Smile naturally
        - Avoid sudden movements
        """)
    
    with col3:
        st.info("""
        **📱 QR Code**
        - Have QR ready before capture
        - Hold QR steady when scanning
        - Ensure QR is not damaged
        """)

# -------------------------
# Main router
# -------------------------
if st.session_state.page == "register":
    page_register()
elif st.session_state.page == "scan":
    page_scan()
elif st.session_state.page == "face":
    page_face()
elif st.session_state.page == "dashboard":
    page_dashboard()
elif st.session_state.page == "capture":
    page_capture()
elif st.session_state.page == "graduation_capture":
    page_graduation_capture()

# Footer
st.markdown("---")
st.caption("Graduation Attendance System")  