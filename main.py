import cv2
import numpy as np
import os
import csv
import mediapipe as mp
from datetime import datetime

# --- 🎛️ CALIBRATION KNOBS (Tweak these if needed) ---
DB_FOLDER = "faces_db"
ATTENDANCE_FILE = f"Attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"

# 1. DETECTION SENSITIVITY (0.0 to 1.0)
# Lower = Detects faces easier (even if blurry/turned)
# Higher = Strict, needs perfect lighting
DETECTION_CONFIDENCE = 0.4 

# 2. RECOGNITION STRICTNESS (0 to 150)
# Lower (e.g., 50) = Very strict (Must look exactly like photo)
# Higher (e.g., 100) = Forgiving (Recognizes you even with glasses/different angles)
RECOGNITION_THRESHOLD = 95 

# 3. FACE PADDING (Pixels)
# Adds extra space around the face crop. Helps if your face is too close to camera.
FACE_PADDING = 30

print("🚀 Initializing Privalens 2.0 (Calibrated Mode)...")

# --- SETUP GOOGLE MEDIAPIPE ---
mp_face_detection = mp.solutions.face_detection
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Load Models
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=DETECTION_CONFIDENCE)

# Setup OpenCV Recognition
recognizer = cv2.face.LBPHFaceRecognizer_create()

# --- HELPER FUNCTIONS ---
logged_users = set()
def log_attendance(name):
    if name in logged_users: return
    current_time = datetime.now().strftime("%H:%M:%S")
    file_exists = os.path.isfile(ATTENDANCE_FILE)
    with open(ATTENDANCE_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists: writer.writerow(["Name", "Time", "Status"])
        writer.writerow([name, current_time, "Present"])
    print(f"✅ LOGGED: {name} at {current_time}")
    logged_users.add(name)

def get_face_box(detection, w, h):
    bboxC = detection.location_data.relative_bounding_box
    x = int(bboxC.xmin * w)
    y = int(bboxC.ymin * h)
    bw = int(bboxC.width * w)
    bh = int(bboxC.height * h)
    
    # Apply Padding (Calibrated)
    x = max(0, x - FACE_PADDING)
    y = max(0, y - FACE_PADDING)
    bw = min(w - x, bw + FACE_PADDING*2)
    bh = min(h - y, bh + FACE_PADDING*2)
    
    return x, y, bw, bh

# --- TRAIN AI ---
faces_data, ids, names, current_id = [], [], {}, 0
if not os.path.exists(DB_FOLDER): os.makedirs(DB_FOLDER)
files = [f for f in os.listdir(DB_FOLDER) if f.endswith(('.jpg', '.png'))]

if files:
    print(f"🧠 Training on {len(files)} faces...")
    for filename in files:
        path = os.path.join(DB_FOLDER, filename)
        img = cv2.imread(path)
        if img is None: continue
        
        # Convert to RGB for MediaPipe Training
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_img)
        
        if results.detections:
            for detection in results.detections:
                h, w, _ = img.shape
                x, y, bw, bh = get_face_box(detection, w, h)
                # Crop logic
                if bw > 0 and bh > 0:
                    face_crop = cv2.cvtColor(img[y:y+bh, x:x+bw], cv2.COLOR_BGR2GRAY)
                    faces_data.append(face_crop)
                    ids.append(current_id)
                    names[current_id] = os.path.splitext(filename)[0]
        current_id += 1
        
    if faces_data: 
        recognizer.train(faces_data, np.array(ids))
        print("✅ System Ready!")
    else:
        print("⚠️ Warning: Could not detect faces in your saved photos. Try registering again with better light.")

# --- RUN CAMERA ---
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    # Mirror Flip & Color Convert
    image = cv2.flip(image, 1)
    h, w, c = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 1. PRIVACY BLUR (Optimized)
    seg_results = segmentation.process(image_rgb)
    condition = np.stack((seg_results.segmentation_mask,) * 3, axis=-1) > 0.1
    
    small_img = cv2.resize(image, (0,0), fx=0.1, fy=0.1)
    blurred_small = cv2.GaussianBlur(small_img, (15, 15), 0)
    bg_image = cv2.resize(blurred_small, (w, h))
    output_image = np.where(condition, image, bg_image)

    # 2. DETECT FACES
    det_results = face_detection.process(image_rgb)

    if det_results.detections:
        for detection in det_results.detections:
            x, y, bw, bh = get_face_box(detection, w, h)
            
            # Defaults
            user_name = "Unknown"
            color = (0, 0, 255)
            confidence_display = 0
            
            # 3. RECOGNIZE
            try:
                if bw > 5 and bh > 5: # Basic size check
                    face_roi = cv2.cvtColor(output_image[y:y+bh, x:x+bw], cv2.COLOR_BGR2GRAY)
                    id_pred, confidence = recognizer.predict(face_roi)
                    confidence_display = int(confidence)

                    # CHECK CONFIDENCE vs CALIBRATED THRESHOLD
                    if confidence < RECOGNITION_THRESHOLD:
                        user_name = names.get(id_pred, "Unknown")
                        color = (0, 255, 0) # Green
                        log_attendance(user_name)
                    else:
                        user_name = "Unknown"
                        color = (0, 0, 255) # Red
            except Exception as e:
                pass 

            # UI
            cv2.rectangle(output_image, (x, y), (x+bw, y+bh), color, 2)
            cv2.putText(output_image, f"{user_name} ({confidence_display})", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Privalens (Calibrated)', output_image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()