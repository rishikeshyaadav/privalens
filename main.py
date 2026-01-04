import cv2
import numpy as np
import os
import csv
import mediapipe as mp
from datetime import datetime

# --- CONFIGURATION ---
DB_FOLDER = "faces_db"
ATTENDANCE_FILE = f"Attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"
CONFIDENCE_THRESHOLD = 0.5
RECOGNITION_THRESHOLD = 95
FACE_PADDING = 30

# Blink Constants
BLINK_RATIO_THRESHOLD = 0.5  # Sensitivity (Higher = Easier to blink)
BLINK_FRAMES = 2             # How many frames eyes must be closed

print("🚀 Initializing Privalens 3.0 (Anti-Spoofing Mode)...")

# 1. SETUP GOOGLE MEDIAPIPE MODELS
mp_face_detection = mp.solutions.face_detection
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_face_mesh = mp.solutions.face_mesh  # New Model for Eyes

# Initialize
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=CONFIDENCE_THRESHOLD)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Setup OpenCV Recognition
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

# 3. HELPER: CALCULATE BLINK (Eye Aspect Ratio logic simplified)
def is_blinking(landmarks, img_h, img_w):
    # Points for eyes (Top, Bottom, Left, Right)
    # Left Eye
    le_t = landmarks[159]
    le_b = landmarks[145]
    # Right Eye
    re_t = landmarks[386]
    re_b = landmarks[374]
    
    # Calculate vertical distances
    left_dist = abs(le_t.y - le_b.y) * img_h
    right_dist = abs(re_t.y - re_b.y) * img_h
    
    # Simple threshold check
    avg_dist = (left_dist + right_dist) / 2
    return avg_dist < 6.0 # Pixels distance when closed

def get_face_box(detection, w, h):
    bboxC = detection.location_data.relative_bounding_box
    x = int(bboxC.xmin * w)
    y = int(bboxC.ymin * h)
    bw = int(bboxC.width * w)
    bh = int(bboxC.height * h)
    x = max(0, x - FACE_PADDING)
    y = max(0, y - FACE_PADDING)
    bw = min(w - x, bw + FACE_PADDING*2)
    bh = min(h - y, bh + FACE_PADDING*2)
    return x, y, bw, bh

# 4. TRAIN AI
faces_data, ids, names, current_id = [], [], {}, 0
if not os.path.exists(DB_FOLDER): os.makedirs(DB_FOLDER)
files = [f for f in os.listdir(DB_FOLDER) if f.endswith(('.jpg', '.png'))]

if files:
    print(f"🧠 Training on {len(files)} faces...")
    for filename in files:
        path = os.path.join(DB_FOLDER, filename)
        img = cv2.imread(path)
        if img is None: continue
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_img)
        if results.detections:
            for detection in results.detections:
                h, w, _ = img.shape
                x, y, bw, bh = get_face_box(detection, w, h)
                if bw>0 and bh>0:
                    face_crop = cv2.cvtColor(img[y:y+bh, x:x+bw], cv2.COLOR_BGR2GRAY)
                    faces_data.append(face_crop)
                    ids.append(current_id)
                    names[current_id] = os.path.splitext(filename)[0]
        current_id += 1
    if faces_data: 
        recognizer.train(faces_data, np.array(ids))
        print("✅ System Ready!")

# 5. RUN CAMERA
cap = cv2.VideoCapture(0)
blink_counter = 0
verified_session = False # Becomes true after blink
active_user = None

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    image = cv2.flip(image, 1)
    h, w, c = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # A. BLUR BACKGROUND
    seg_results = segmentation.process(image_rgb)
    condition = np.stack((seg_results.segmentation_mask,) * 3, axis=-1) > 0.1
    small_img = cv2.resize(image, (0,0), fx=0.1, fy=0.1)
    blurred_small = cv2.GaussianBlur(small_img, (15, 15), 0)
    bg_image = cv2.resize(blurred_small, (w, h))
    output_image = np.where(condition, image, bg_image)

    # B. DETECT FACES
    det_results = face_detection.process(image_rgb)
    
    # C. DETECT BLINK (Liveness)
    mesh_results = face_mesh.process(image_rgb)
    blink_now = False
    if mesh_results.multi_face_landmarks:
        for face_landmarks in mesh_results.multi_face_landmarks:
            if is_blinking(face_landmarks.landmark, h, w):
                blink_counter += 1
            else:
                if blink_counter >= BLINK_FRAMES:
                    blink_now = True # Blink completed
                blink_counter = 0

    if det_results.detections:
        for detection in det_results.detections:
            x, y, bw, bh = get_face_box(detection, w, h)
            
            user_name = "Unknown"
            color = (0, 0, 255)
            
            try:
                if bw > 0 and bh > 0:
                    face_roi = cv2.cvtColor(output_image[y:y+bh, x:x+bw], cv2.COLOR_BGR2GRAY)
                    id_pred, confidence = recognizer.predict(face_roi)
                    
                    if confidence < RECOGNITION_THRESHOLD:
                        temp_name = names.get(id_pred, "Unknown")
                        
                        # LOGIC: Check Liveness
                        if blink_now and not verified_session:
                            verified_session = True
                            active_user = temp_name
                            log_attendance(temp_name)
                        
                        # Display Logic
                        if verified_session and active_user == temp_name:
                            user_name = temp_name
                            color = (0, 255, 0) # Green (Access Granted)
                            status_text = f"VERIFIED: {user_name}"
                        else:
                            user_name = temp_name
                            color = (0, 255, 255) # Yellow (Wait for blink)
                            status_text = "BLINK TO VERIFY..."
                    else:
                        verified_session = False
                        status_text = "Unknown"
            except: 
                status_text = "Error"

            cv2.rectangle(output_image, (x, y), (x+bw, y+bh), color, 2)
            cv2.putText(output_image, status_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Reset session if no face
    if not det_results.detections:
        verified_session = False

    cv2.imshow('Privalens (Anti-Spoofing)', output_image)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()