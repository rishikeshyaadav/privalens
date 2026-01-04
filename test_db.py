import firebase_admin
from firebase_admin import credentials, firestore
import os

print("📂 Checking current folder:", os.getcwd())

# 1. Check if file exists
filename = "serviceAccountKey.json"
if os.path.exists(filename):
    print(f"✅ Found {filename}!")
    
    # 2. Try to Connect
    try:
        cred = credentials.Certificate(filename)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("✅ Connection successful!")
        
        # 3. Try to Write Data
        print("📝 Attempting to write test data...")
        db.collection("attendance").document("test_user").set({
            "name": "Test User",
            "status": "Connected!",
            "time": "Now"
        })
        print("🚀 SUCCESS! Data sent to cloud.")
        print("👉 Go check your Firebase Console now!")
        
    except Exception as e:
        print(f"❌ Connection Error: {e}")
else:
    print(f"❌ Error: Could not find '{filename}'")
    print("Files I can see here are:", os.listdir())