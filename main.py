import cv2
import numpy as np
import os
import csv
from datetime import datetime

# --- CONFIGURATION ---
DB_FOLDER = "faces_db"
BLINK_MIN_FRAMES = 2
BLINK_MAX_FRAMES = 5
ATTENDANCE_FILE = f"Attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"

print("🚀 Initializing Privalens Pro...")

# 1. SETUP CAMERA & DETECTORS
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# 2. HELPER: PRIVACY FILTER
def apply_tech_filter(img, face_rect):
    x, y, w, h = face_rect
    color = (0, 255, 0)
    d = 20 # Line length
    t = 2  # Thickness
    # Corners
    cv2.line(img, (x, y), (x+d, y), color, t)
    cv2.line(img, (x, y), (x, y+d), color, t)
    cv2.line(img, (x+w, y), (x+w-d, y), color, t)
    cv2.line(img, (x+w, y), (x+w, y+d), color, t)
    cv2.line(img, (x, y+h), (x+d, y+h), color, t)
    cv2.line(img, (x, y+h), (x, y+h-d), color, t)
    cv2.line(img, (x+w, y+h), (x+w-d, y+h), color, t)
    cv2.line(img, (x+w, y+h), (x+w, y+h-d), color, t)
    return img

# 3. HELPER: SAVE ATTENDANCE TO CSV
logged_users = set()

def log_attendance(name):
    if name in logged_users:
        return
    
    current_time = datetime.now().strftime("%H:%M:%S")
    file_exists = os.path.isfile(ATTENDANCE_FILE)
    
    with open(ATTENDANCE_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Time", "Status"])
        writer.writerow([name, current_time, "Present"])
    
    print(f"✅ LOGGED: {name} at {current_time}")
    logged_users.add(name)

# 4. TRAIN AI
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
else:
    print("⚠️ Warning: No faces found! Please run app.py first.")

# 5. RUN LOOP
eyes_missing_frames = 0
attendance_marked = False
verified_user = ""
print("📷 System Active. Press 'q' to quit.")

while True:
    success, img = cap.read()
    if not success: break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        try:
            id_pred, confidence = recognizer.predict(roi_gray)
            user_name = names.get(id_pred, "Unknown") if (confidence < 85 and faces_data) else "Unknown"
        except: user_name = "Unknown"

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
        if len(eyes) >= 1:
            if BLINK_MIN_FRAMES <= eyes_missing_frames <= BLINK_MAX_FRAMES and user_name != "Unknown":
                attendance_marked = True
                verified_user = user_name
                log_attendance(user_name)
            eyes_missing_frames = 0
        else:
            eyes_missing_frames += 1

        if user_name == "Unknown":
            color = (0, 0, 255)
            status = "UNKNOWN"
        elif attendance_marked and verified_user == user_name:
            color = (0, 255, 0)
            status = f"LOGGED: {datetime.now().strftime('%H:%M')}"
            img = apply_tech_filter(img, (x, y, w, h))
        else:
            color = (0, 255, 255)
            status = "BLINK TO VERIFY"
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

        cv2.putText(img, status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(img, user_name, (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Privalens Pro", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()