# camera_face_verify_smoothstart.py
import cv2
import json
import os
import time
import threading
import numpy as np
import pandas as pd
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from scipy.spatial.distance import euclidean
import tensorflow as tf

# -------------------------
# CONFIG / TUNABLES
# -------------------------
EMBEDDED_CSV = "student_csv/student_with_embeddings.csv"
MASK_MODEL_PATH = "mask_detectorV2.h5"

CAM_SRC = 0
DISPLAY_W = 800            # display window size (kept relatively big for UX)
DISPLAY_H = 600
PROC_W = 400               # processing width (smaller -> faster detection)
PROC_H = 300
PROCESS_EVERY_N_FRAMES = 4  # process 1-in-N frames
EMBEDDING_COOLDOWN = 0.45   # seconds between FaceNet calls
REQUIRED_CONSECUTIVE = 3
DISTANCE_THRESHOLD = 0.83
CAM_WARMUP_FRAMES = 10      # read & discard before starting processing
SUCCESS_DISPLAY_SECONDS = 5

# -------------------------
# Threaded video class (fast read, avoids read() blocking)
# -------------------------
class ThreadedVideoStream:
    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.width = width
        self.height = height
        self.stream = cv2.VideoCapture(self.src)
        # request wanted size (driver may honor/ignore)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        self.grabbed, self.frame = self.stream.read()
        self.lock = threading.Lock()
        self.stopped = False

    def start(self):
        t = threading.Thread(target=self.update, daemon=True)
        t.start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            with self.lock:
                self.grabbed, self.frame = grabbed, frame
            if not grabbed:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.grabbed, self.frame.copy()

    def stop(self):
        self.stopped = True
        try:
            self.stream.release()
        except Exception:
            pass

# -------------------------
# Load models and DB
# -------------------------
print("Loading models...")
embedder = FaceNet()   # FaceNet
detector = MTCNN()     # MTCNN (kept per your request)
mask_model = load_model(MASK_MODEL_PATH)
print("Models loaded.")

# load db embeddings
if os.path.exists(EMBEDDED_CSV):
    df = pd.read_csv(EMBEDDED_CSV)
    db_embeddings = []
    for emb in df.get('embedding', []):
        if pd.isna(emb) or emb == "":
            db_embeddings.append(None)
        else:
            if isinstance(emb, str):
                vals = list(map(float, emb.strip("[] ").split(",")))
                db_embeddings.append(np.array(vals))
            else:
                db_embeddings.append(None)
    print(f"Loaded {len(db_embeddings)} embeddings.")
else:
    print("Embedding DB not found; continuing with empty DB.")
    df = pd.DataFrame()
    db_embeddings = []

# -------------------------
# WARM-UP MODELS BEFORE OPENING CAMERA
# (this is the key change for smooth-start)
# -------------------------
print("Warming up models (this prevents the initial freeze)...")
_dummy_face = np.zeros((160, 160, 3), dtype=np.uint8)         # RGB dummy for FaceNet warm-up
try:
    # Warm up MTCNN: it accepts RGB numpy array (uint8). run a single detect.
    _ = detector.detect_faces(_dummy_face)
except Exception:
    # some MTCNN implementations may behave differently; ignore failures here
    pass

# Warm-up FaceNet (embedding)
try:
    _ = embedder.embeddings(np.expand_dims(_dummy_face, axis=0))
except Exception:
    pass

# Warm-up mask model
try:
    _ = mask_model.predict(np.zeros((1, 224, 224, 3), dtype=np.float32), verbose=0)
except Exception:
    pass
print("Warm-up complete.")

# -------------------------
# Helper functions
# -------------------------
def get_embedding(face_pixels):
    # face_pixels expected to be 160x160 RGB uint8/float
    arr = np.expand_dims(face_pixels, axis=0)
    return embedder.embeddings(arr)[0]

def verify_face(live_emb):
    min_dist = float("inf")
    idx = -1
    for i, db_emb in enumerate(db_embeddings):
        if db_emb is None:
            continue
        d = euclidean(live_emb, db_emb)
        if d < min_dist:
            min_dist = d
            idx = i
    if idx >= 0 and min_dist < DISTANCE_THRESHOLD:
        row = df.loc[idx].to_dict()
        row['distance'] = float(min_dist)
        return row
    return None

# -------------------------
# Main loop
# -------------------------
def run_face_verification(expected_student_id=None):
    # Start threaded camera AFTER warm-up (camera warm-up as well)
    stream = ThreadedVideoStream(CAM_SRC, width=DISPLAY_W, height=DISPLAY_H).start()

    # Camera warm-up: discard a few frames so exposure/autofocus stabilize
    print(f"Stabilizing camera ({CAM_WARMUP_FRAMES} frames)...")
    for _ in range(CAM_WARMUP_FRAMES):
        grabbed, _ = stream.read()
        time.sleep(0.01)

    cv2.namedWindow("Face Verification", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Verification", DISPLAY_W, DISPLAY_H)

    print("Starting face verification. Press 'Q' to cancel.")
    if expected_student_id:
        print("Expected ID:", expected_student_id)

    verified = False
    frame_count = 0
    last_status = ""
    consecutive_matches = 0
    last_match = None
    last_embedding_time = 0.0
    success_display_counter = 0
    success_info = {}

    try:
        while True:
            grabbed, frame = stream.read()
            if not grabbed or frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            display = frame.copy()
            h, w = display.shape[:2]

            # process smaller resized image for detection -> faster
            proc_frame = cv2.resize(frame, (PROC_W, PROC_H))
            scale_x = w / PROC_W
            scale_y = h / PROC_H

            if frame_count % PROCESS_EVERY_N_FRAMES == 0 and success_display_counter == 0 and not verified:
                # detections on smaller frame (MTCNN expects RGB)
                rgb_small = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
                faces = detector.detect_faces(rgb_small)

                if len(faces) == 1:
                    x_s, y_s, w_s, h_s = faces[0]['box']
                    # scale back to original frame coords
                    x = int(max(0, x_s * scale_x))
                    y = int(max(0, y_s * scale_y))
                    w_face = int(max(1, w_s * scale_x))
                    h_face = int(max(1, h_s * scale_y))
                    x2 = min(w - 1, x + w_face)
                    y2 = min(h - 1, y + h_face)

                    crop = frame[y:y2, x:x2]
                    if crop.size != 0:
                        # Mask detection (cheap relative to FaceNet)
                        mask_input = cv2.resize(crop, (224, 224))
                        mask_input = img_to_array(mask_input) / 255.0
                        mask_input = np.expand_dims(mask_input, 0)
                        pred = mask_model.predict(mask_input, verbose=0)
                        mask_label = np.argmax(pred)

                        if mask_label == 0:
                            last_status = "REMOVE MASK"
                            consecutive_matches = 0
                            last_match = None
                            color = (0, 0, 255)
                            cv2.rectangle(display, (x, y), (x2, y2), color, 3)
                            cv2.putText(display, "REMOVE MASK", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                        else:
                            # Only call FaceNet if cooldown expired (reduces CPU/GPU thrash)
                            now = time.time()
                            if now - last_embedding_time >= EMBEDDING_COOLDOWN:
                                # prepare input for FaceNet
                                facenet_input = cv2.resize(crop, (160, 160))
                                facenet_input = cv2.cvtColor(facenet_input, cv2.COLOR_BGR2RGB)
                                live_emb = get_embedding(facenet_input)
                                last_embedding_time = now

                                match = verify_face(live_emb)
                            else:
                                # skip embedding this cycle; treat as unknown for now
                                match = None

                            if match is not None:
                                student_id = match.get('student_id', '')
                                student_name = match.get('student_name', '')
                                distance = match.get('distance', 0.0)

                                if expected_student_id and student_id != expected_student_id:
                                    last_status = "WRONG PERSON"
                                    consecutive_matches = 0
                                    last_match = None
                                    color = (0, 0, 255)
                                    cv2.rectangle(display, (x, y), (x2, y2), color, 3)
                                    cv2.putText(display, f"EXPECTED {expected_student_id}", (x, y - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                                else:
                                    # consecutive match logic (counts processed frames where a match occurred)
                                    if last_match == student_id:
                                        consecutive_matches += 1
                                    else:
                                        consecutive_matches = 1
                                        last_match = student_id

                                    last_status = "MATCHED"
                                    color = (0, 255, 0)
                                    cv2.rectangle(display, (x, y), (x2, y2), color, 3)
                                    cv2.putText(display, f"{student_name} ({distance:.3f})", (x, y - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                                    if consecutive_matches >= REQUIRED_CONSECUTIVE:
                                        # final verification SUCCESS
                                        success_info = {
                                            'student_id': student_id,
                                            'student_name': student_name,
                                            'distance': distance
                                        }
                                        # save result JSON
                                        result = {
                                            'status': 'success',
                                            'student_id': student_id,
                                            'student_name': student_name,
                                            'programme': match.get('programme', ''),
                                            'faculty': match.get('faculty', ''),
                                            'distance': distance,
                                            'match': True
                                        }
                                        with open('face_verify_result.json', 'w') as f:
                                            json.dump(result, f)
                                        print("✅ Verification Success:", student_name, student_id)
                                        success_display_counter = int(SUCCESS_DISPLAY_SECONDS * 30)  # approx frames to show
                                        verified = True
                            else:
                                # no match this cycle (either unknown or embedding skipped)
                                last_status = "UNKNOWN"
                                consecutive_matches = 0
                                last_match = None
                                color = (0, 0, 255)
                                cv2.rectangle(display, (x, y), (x2, y2), color, 2)
                                cv2.putText(display, "UNKNOWN", (x, y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    last_status = "NO FACE" if len(faces) == 0 else "MULTIPLE FACES"
                    consecutive_matches = 0
                    last_match = None

            # If success display counter active, show the full success overlay and count down
            if success_display_counter > 0 and success_info:
                # show big green frame and info
                color = (0, 200, 0)
                cv2.rectangle(display, (0, 0), (w, h), color, 8)
                cv2.putText(display, "VERIFICATION SUCCESS", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                cv2.putText(display, f"Name: {success_info['student_name']}", (50, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.putText(display, f"ID: {success_info['student_id']}", (50, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                success_display_counter -= 1
                # when counter reaches zero we will exit the loop below
            else:
                # standard overlays
                if last_status:
                    cv2.putText(display, f"Status: {last_status}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(display, "Press 'Q' to quit", (10, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Face Verification", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                print("User cancelled.")
                break

            # exit when success display finished
            if verified and success_display_counter <= 0:
                break

    finally:
        stream.stop()
        cv2.destroyAllWindows()
        print("Face verification closed.")

# -------------------------
# Run as script
# -------------------------
if __name__ == "__main__":
    import sys
    expected_id = sys.argv[1] if len(sys.argv) > 1 else None
    run_face_verification(expected_id)
