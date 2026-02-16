from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import os
import base64
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# --- CONFIG ---
# Load AI Model (Simulating the Edge Node logic for the Web Demo)
print("Loading YOLOv8 Model...")
model = YOLO("yolov8n.pt") 

# DATABASE
train_state = {
    "current_station": "Vijayawada",
    "next_station": "Nuzvid",
    "is_reversed": False,
    "coaches": {
        "GEN-1": {"crowd": 0, "image": None},
        "GEN-2": {"crowd": 0, "image": None},
        "GEN-3": {"crowd": 0, "image": None},
        "GEN-4": {"crowd": 0, "image": None} # Added 4th Coach
    }
}

@app.route('/')
def passenger_dashboard():
    return render_template('index.html')

@app.route('/admin')
def admin_dashboard():
    return render_template('admin.html')

# --- API 1: SIMULATE EDGE PROCESSING (For Web Dashboard Demo) ---
# This allows the Admin to upload/capture an image.
# The Server acts as the "Edge Device" temporarily to process it.
@app.route('/api/simulate_edge_processing', methods=['POST'])
def simulate_edge_processing():
    try:
        data = request.json
        image_data = data.get('image') # Base64 string
        coach_id = data.get('coach_id')

        # 1. Decode Image
        encoded_data = image_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. Run YOLO (Processing Phase)
        results = model.predict(img, classes=[0], conf=0.3, verbose=False)
        count = len(results[0].boxes)
        
        # 3. Save Processed Image (For Display Only)
        annotated_frame = results[0].plot()
        filename = f"{coach_id}_processed.jpg"
        save_path = os.path.join("static/captures", filename)
        cv2.imwrite(save_path, annotated_frame)

        # 4. Calculate Data
        capacity = 20
        density = int((count / capacity) * 100)
        if density > 100: density = 100

        # 5. Update DB (Simulating "Sending JSON")
        train_state['coaches'][coach_id] = {
            "crowd": density,
            "image": f"/static/captures/{filename}"
        }

        return jsonify({"status": "success", "count": count, "density": density})

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"status": "error", "message": str(e)})

# --- API 2: REAL EDGE DATA RECEIVER (For Python Script) ---
# This is where the REAL Edge Node sends data.
# Note: It only receives JSON (crowd %), NOT the image processing work.
@app.route('/api/update_crowd', methods=['POST'])
def update_crowd(): 
    data = request.json
    coach_id = data.get('coach_id')
    crowd_percent = data.get('crowd_percent')
    # The edge node sends the image separately just for display, 
    # but the CALCULATION happened on the edge.
    
    if coach_id in train_state['coaches']:
        train_state['coaches'][coach_id]['crowd'] = crowd_percent
        # We assume the edge node saved the image to static/captures via a separate upload 
        # or shared folder, or we just update the text.
        # For this demo, we assume the filename matches.
        train_state['coaches'][coach_id]['image'] = f"/static/captures/{coach_id}_live.jpg"

    return jsonify({"status": "success"})

# --- DATA FETCHING APIs ---
@app.route('/api/get_status', methods=['GET'])
def get_status():
    simple_coaches = {k: v['crowd'] for k, v in train_state['coaches'].items()}
    return jsonify({
        "current_station": train_state['current_station'],
        "next_station": train_state['next_station'],
        "is_reversed": train_state['is_reversed'],
        "coaches": simple_coaches 
    })

@app.route('/api/get_admin_data', methods=['GET'])
def get_admin_data():
    return jsonify(train_state)

if __name__ == '__main__':
    if not os.path.exists('static/captures'):
        os.makedirs('static/captures')
    app.run(debug=True, host='0.0.0.0', port=5000)