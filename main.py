import cv2
import numpy as np
import os
import csv
import mediapipe as mp
from datetime import datetime

# --- CONFIGURATION ---
DB_FOLDER = "faces_db"
BLINK_MIN_FRAMES = 2
ATTENDANCE_FILE = f"Attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"

print("🚀 Initializing Privalens with Google MediaPipe...")

# 1. SETUP GOOGLE MEDIAPIPE (Standard Way)
mp_face_detection = mp.solutions.face_detection
mp_selfie_segmentation = mp.solutions.selfie_segmentation

# Configure AI Models
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Setup OpenCV Recognition
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# 2. HELPER: LOG ATTENDANCE
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

# 3. TRAIN AI (Load your faces)
faces_data, ids, names, current_id = [], [], {}, 0
if not os.path.exists(DB_FOLDER): os.makedirs(DB_FOLDER)
files = [f for f in os.listdir(DB_FOLDER) if f.endswith(('.jpg', '.png'))]

if files:
    print(f"🧠 Training on {len(files)} faces...")
    for filename in files:
        path = os.path.join(DB_FOLDER, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        faces_rect = face_cascade.detectMultiScale(img, 1.1, 5)
        for (x, y, w, h) in faces_rect:
            faces_data.append(img[y:y+h, x:x+w])
            ids.append(current_id)
            names[current_id] = os.path.splitext(filename)[0]
        current_id += 1
    if faces_data: recognizer.train(faces_data, np.array(ids))

# 4. RUN CAMERA LOOP
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    # --- GOOGLE FEATURE: PRIVACY BLUR ---
    image = cv2.flip(image, 1) 
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = segmentation.process(image_rgb)
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    
    bg_image = cv2.GaussianBlur(image, (55, 55), 0)
    output_image = np.where(condition, image, bg_image)

    # --- FACE RECOGNITION ---
    gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        try:
            id_pred, confidence = recognizer.predict(roi_gray)
            user_name = names.get(id_pred, "Unknown") if (confidence < 85 and faces_data) else "Unknown"
        except: user_name = "Unknown"

        if user_name != "Unknown":
            log_attendance(user_name)
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        cv2.rectangle(output_image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(output_image, f"{user_name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Privalens (Google AI)', output_image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()