"""
Edge Node AI Processor
----------------------
This script simulates the onboard CCTV processing unit. 
It remains idle until triggered by a station departure event from the central server.
It utilizes YOLOv8 to perform Perspective Footprint Mapping for occlusion-resistant density estimation.
"""

from ultralytics import YOLO
import cv2
import requests
import os
import time
import numpy as np

SERVER_URL = "http://127.0.0.1:5000"
SAVE_DIR = "static/captures"
COACHES = ["GEN-1", "GEN-2", "GEN-3", "GEN-4", "SLR-1", "SLR-2", "SLR-M"]

def calculate_advanced_density(img, results):
    """
    Calculates spatial density using Perspective Footprint Mapping.
    Prioritizes the feet (base) of bounding boxes and scales their weight based on depth (occlusion).
    """
    h, w = img.shape[:2]
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    # Return 0 if coach is completely empty
    if len(boxes) == 0: 
        return 0, img.copy()

    overlay = np.zeros_like(img)
    total_weighted_mass = 0
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Extract the center coordinate of the passenger's feet
        feet_x = int((x1 + x2) / 2)
        feet_y = int(y2)
        
        # Occlusion Multiplier: Passengers further from the camera (lower Y value) 
        # have a higher statistical probability of obscuring passengers behind them.
        depth_ratio = 1.0 - (feet_y / h) 
        occlusion_weight = 1.0 + (depth_ratio * 2.0) 
        total_weighted_mass += occlusion_weight
        
        # Render the spatial footprint footprint based on perspective depth
        radius = int(w * 0.05 * (feet_y / h)) 
        if radius > 0:
            cv2.circle(overlay, (feet_x, feet_y), radius, (0, 0, 255), -1)
            cv2.rectangle(overlay, (x1, int(y1 + (y2-y1)*0.2)), (x2, y2), (0, 0, 255), 2)

    # Define the maximum mathematical threshold for crush load capacity
    MAX_CAPACITY = 35.0 
    density = min(int((total_weighted_mass / MAX_CAPACITY) * 100), 100)

    # Render final analytical image
    visual_img = cv2.addWeighted(img, 0.7, overlay, 0.6, 0)
    cv2.putText(visual_img, f"Mass Index: {round(total_weighted_mass,1)} | Density: {density}%", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return density, visual_img

def edge_process():
    """
    Main polling loop. Listens for server state changes to trigger image processing.
    """
    print("[SYSTEM] Initializing Edge AI Node...")
    model = YOLO("yolov8s.pt") 
    
    if not os.path.exists(SAVE_DIR): 
        os.makedirs(SAVE_DIR)

    last_processed_state = None

    while True:
        try:
            res = requests.get(f"{SERVER_URL}/api/get_live_status")
            state = res.json()
            
            # Remain idle if no active train is assigned
            if not state.get('active_train'):
                time.sleep(2)
                continue

            # Ensure processing only occurs once per station departure event
            current_state_key = f"{state['active_train']}_{state['station_index']}"
            if current_state_key == last_processed_state:
                time.sleep(2)
                continue

            station_idx = state['station_index']
            print(f"[EVENT] Train departed. Initiating analysis for Station Index: {station_idx}")
            
            # Fetch corresponding static image set for demonstration purposes
            folder_path = f"test_images/stop_{station_idx}"
            if not os.path.exists(folder_path):
                folder_path = "test_images"

            for coach in COACHES:
                img_path = None
                for ext in ['.jpg', '.jpeg', '.png']:
                    temp_path = os.path.join(folder_path, f"{coach}{ext}")
                    if os.path.exists(temp_path):
                        img_path = temp_path
                        break
                
                if not img_path: 
                    continue

                img = cv2.imread(img_path)
                results = model.predict(img, classes=[0], conf=0.20, verbose=False)
                density, visual_img = calculate_advanced_density(img, results)
                
                save_path = os.path.join(SAVE_DIR, f"{coach}_live.jpg")
                cv2.imwrite(save_path, visual_img)
                
                # Transmit raw YOLO density to central server
                requests.post(f"{SERVER_URL}/api/update_crowd", json={"coach_id": coach, "crowd_percent": density})
                
            last_processed_state = current_state_key
            print("[SYSTEM] Edge analysis complete. Entering standby mode.")
                
        except Exception as e: 
            pass 
            
        time.sleep(2)

if __name__ == "__main__":
    edge_process()