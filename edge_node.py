from ultralytics import YOLO
import cv2
import requests
import os
import time
import numpy as np

SERVER_URL = "http://127.0.0.1:5000/api/update_crowd"

COACH_IMAGES = {
    "GEN-1": "test_images/GEN-1.jpg",
    "GEN-2": "test_images/GEN-2.jpg",
    "GEN-3": "test_images/GEN-3.jpg",
    "GEN-4": "test_images/GEN-4.jpg",
    "SLR-1": "test_images/SLR-1.jpg",
    "SLR-2": "test_images/SLR-2.jpg",
    "SLR-M": "test_images/SLR-M.jpg"
}
SAVE_DIR = "static/captures"

def calculate_and_visualize_density(img, results):
    h, w = img.shape[:2]
    boxes = results[0].boxes.xyxy.cpu().numpy()
    count = len(boxes)

    if count == 0:
        # If empty, just return the clear image
        return 0, img.copy()

    grid_size = int(w / 35) 
    overlay = np.zeros_like(img)
    
    total_valid_cells = 0
    occupied_cells = 0

    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            total_valid_cells += 1
            
            cell_x1, cell_y1 = x, y
            cell_x2, cell_y2 = min(x + grid_size, w), min(y + grid_size, h)
            
            is_occupied = False
            
            for box in boxes:
                bx1, by1, bx2, by2 = map(int, box[:4])
                
                bw = bx2 - bx1
                bh = by2 - by1
                core_x1 = bx1 + int(bw * 0.15)
                core_x2 = bx2 - int(bw * 0.15)
                core_y1 = by1 + int(bh * 0.15)
                core_y2 = by2
                
                if (cell_x1 < core_x2 and cell_x2 > core_x1 and
                    cell_y1 < core_y2 and cell_y2 > core_y1):
                    is_occupied = True
                    break
                    
            if is_occupied:
                cv2.rectangle(overlay, (cell_x1, cell_y1), (cell_x2, cell_y2), (0, 0, 255), -1) # Red only
                occupied_cells += 1
                
    MAX_GRID_FILL = 0.45
    
    if total_valid_cells > 0:
        raw_fill = occupied_cells / total_valid_cells
        density = int((raw_fill / MAX_GRID_FILL) * 100)
    else:
        density = 0
        
    density = min(density, 100)
    
    # Visual Blend - Only applies the red squares, leaving the rest of the image natural
    visual_img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        
    return density, visual_img

def edge_process():
    print("Starting Edge Node (Clean Heatmap Analytics)...")
    model = YOLO("yolov8n.pt") 
    if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

    while True:
        print("\n[EDGE] Scanning Coach Cameras...")
        for coach_id, img_path in COACH_IMAGES.items():
            if not os.path.exists(img_path):
                continue

            img = cv2.imread(img_path)
            results = model.predict(img, classes=[0], conf=0.25, verbose=False)
            density, visual_img = calculate_and_visualize_density(img, results)
            
            save_path = os.path.join(SAVE_DIR, f"{coach_id}_live.jpg")
            cv2.imwrite(save_path, visual_img)

            try:
                requests.post(SERVER_URL, json={"coach_id": coach_id, "crowd_percent": density})
                print(f"{coach_id} Analyzed -> Space Filled: {density}%")
            except:
                pass
        time.sleep(4)

if __name__ == "__main__":
    edge_process()