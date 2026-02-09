from ultralytics import YOLO
import cv2
import requests
import os
import time

# --- CONFIG ---
SERVER_URL = "http://127.0.0.1:5000/api/update_crowd"
# Now supports 4 coaches
COACH_IMAGES = {
    "GEN-1": "test_images/GEN-1.jpg",
    "GEN-2": "test_images/GEN-2.jpg",
    "GEN-3": "test_images/GEN-3.jpg",
    "GEN-4": "test_images/GEN-4.jpg"
}
SAVE_DIR = "static/captures"

def edge_process():
    print("🚀 Starting Edge Node (Local AI Processing)...")
    # 1. LOAD MODEL LOCALLY (This is Edge Processing)
    model = YOLO("yolov8n.pt") 

    while True:
        print("\n🔄 [EDGE] Scanning Cameras...")
        
        for coach_id, img_path in COACH_IMAGES.items():
            if not os.path.exists(img_path):
                print(f"   ⚠️ Image not found: {img_path}")
                continue

            # 2. LOCAL PROCESSING
            img = cv2.imread(img_path)
            results = model.predict(img, classes=[0], conf=0.3, verbose=False)
            count = len(results[0].boxes)
            
            # 3. LOCAL CALCULATION
            capacity = 20
            density = int((count / capacity) * 100)
            if density > 100: density = 100

            # 4. SAVE ANNOTATED IMAGE (For display only)
            annotated_frame = results[0].plot()
            save_path = os.path.join(SAVE_DIR, f"{coach_id}_live.jpg")
            cv2.imwrite(save_path, annotated_frame)

            # 5. SEND DATA (JSON)
            # We are sending the computation RESULT, not the raw task.
            payload = {
                "coach_id": coach_id,
                "crowd_percent": density
            }
            
            try:
                requests.post(SERVER_URL, json=payload)
                print(f"   ✅ [EDGE] {coach_id}: Processed {count} pax ({density}%) -> Sent JSON.")
            except:
                print("   ❌ Server Disconnected")

        time.sleep(5)

if __name__ == "__main__":
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
    edge_process()