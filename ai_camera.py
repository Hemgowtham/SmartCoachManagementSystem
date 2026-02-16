from ultralytics import YOLO
import cv2
import requests
import os
import time

# --- CONFIGURATION ---
SERVER_URL = "http://127.0.0.1:5000/api/update_crowd"
TEST_IMAGES_DIR = "test_images"       # Folder containing GEN-1.jpg, GEN-2.jpg etc.
SAVE_DIR = "static/captures"          # Where processed images go

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_all_images():
    ensure_dir(SAVE_DIR)
    
    # 1. Load YOLO Model
    print("Loading YOLOv8 Model...")
    model = YOLO("yolov8n.pt") 

    # 2. Get list of all images in 'test_images' folder
    if not os.path.exists(TEST_IMAGES_DIR):
        print(f"Error: Folder '{TEST_IMAGES_DIR}' not found. Please create it and add images.")
        return

    image_files = [f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print(f"No images found in {TEST_IMAGES_DIR}!")
        return

    print(f"Found {len(image_files)} images. Processing...")

    # 3. Loop through each image
    for img_file in image_files:
        # Extract Coach ID from filename (e.g., 'GEN-1.jpg' -> 'GEN-1')
        coach_id = os.path.splitext(img_file)[0]
        img_path = os.path.join(TEST_IMAGES_DIR, img_file)
        
        # Read Image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read {img_file}")
            continue

        # Detect People (Class 0)
        results = model.predict(img, classes=[0], conf=0.3, verbose=False)
        result = results[0]
        
        # Count People
        count = len(result.boxes)
        
        # Draw Boxes & Save to Static Folder
        annotated_frame = result.plot()
        save_filename = f"{coach_id}_live.jpg"
        save_path = os.path.join(SAVE_DIR, save_filename)
        cv2.imwrite(save_path, annotated_frame)
        
        # Calculate Dummy Density (Capacity = 20 for demo)
        capacity = 20
        density = int((count / capacity) * 100)
        if density > 100: density = 100

        # Send to Server
        payload = {
            "coach_id": coach_id,
            "crowd_percent": density,
            "image_filename": save_filename
        }
        
        try:
            requests.post(SERVER_URL, json=payload)
            print(f"{coach_id}: Detected {count} people ({density}%) -> Sent to Server")
        except:
            print(f"{coach_id}: Failed to send data (Is app.py running?)")

    print("\nAll images processed! Refresh your Admin Dashboard.")

if __name__ == "__main__":
    while True:
        process_all_images()
        print("Waiting 10 seconds before next scan...\n")
        time.sleep(10)