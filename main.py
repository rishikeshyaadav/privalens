import cv2
import numpy as np
import os

# CONFIGURATION
DB_FOLDER = "faces_db"
BLINK_MIN_FRAMES = 1
BLINK_MAX_FRAMES = 5 

print("Initializing System...")
cap = cv2.VideoCapture(0)
# Load Face and Eye detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

# PRIVACY FILTER FUNCTION
def apply_privacy_filter(img, face_rect):
    mask = np.zeros_like(img)
    x, y, w, h = face_rect
    center_x, center_y = x + w//2, y + h//2
    radius = int(h * 0.8)
    cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), -1)
    blurred_img = cv2.GaussianBlur(img, (51, 51), 0)
    return np.where(mask==np.array([255, 255, 255]), img, blurred_img)

# TRAIN THE AI
faces_data, ids, names, current_id = [], [], {}, 0
if not os.path.exists(DB_FOLDER): os.makedirs(DB_FOLDER)
files = [f for f in os.listdir(DB_FOLDER) if f.endswith(('.jpg', '.png'))]

if files:
    print(f"🧠 Learning from {len(files)} images...")
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

# START CAMERA
eyes_missing_frames, attendance_marked, verified_user = 0, False, ""
print("📷 Camera Active. Press 'q' to quit.")

while True:
    success, img = cap.read()
    if not success: break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0: cv2.imshow("Privalens", img)

    for (x, y, w, h) in faces:
        # Apply privacy blur first
        img = apply_privacy_filter(img, (x, y, w, h))
        roi_gray = gray[y:y+h, x:x+w]

        # Predict Identity
        try:
            id_pred, confidence = recognizer.predict(roi_gray)
            if confidence < 80 and faces_data:
                user_name = names.get(id_pred, "Unknown")
            else:
                user_name = "Unknown"
        except: user_name = "Unknown"

        # Check for Blinking (Liveness)
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 4)
        if len(eyes) >= 1:
            if BLINK_MIN_FRAMES <= eyes_missing_frames <= BLINK_MAX_FRAMES and user_name != "Unknown":
                attendance_marked = True
                verified_user = user_name
            eyes_missing_frames = 0
        else:
            eyes_missing_frames += 1

        # Display Status
        if user_name == "Unknown": status, color = "ACCESS DENIED", (0, 0, 255)
        elif attendance_marked and verified_user == user_name: status, color = "PRESENT (Verified)", (0, 255, 0)
        else: status, color = "BLINK TO VERIFY...", (0, 255, 255)

        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, status, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img, user_name, (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Privalens", img)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()