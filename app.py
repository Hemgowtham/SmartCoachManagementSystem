"""
Central Management Server (RailVision)
--------------------------------------
Handles API routing, global state synchronization, historical logging, 
and advanced predictive passenger distribution modeling based on UTS data.
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import base64
import cv2
import numpy as np
import csv
from datetime import datetime
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

print("[SYSTEM] Loading YOLOv8 Model for Validation Panel...")
model = YOLO("yolov8s.pt") 

TRAIN_DB = {}
ALL_STATIONS = set()
UTS_DB = {}

GLOBAL_STATE = {
    "active_train": None,
    "station_index": 0
}

# Stores the raw YOLO density output from the edge node
# The displayed 'crowd' is mathematically adjusted from this base using UTS data.
train_state_coaches = {
    "GEN-1": {"yolo_base": 0, "image": None},
    "GEN-2": {"yolo_base": 0, "image": None},
    "GEN-3": {"yolo_base": 0, "image": None},
    "GEN-4": {"yolo_base": 0, "image": None},
    "SLR-1": {"yolo_base": 0, "image": None},
    "SLR-2": {"yolo_base": 0, "image": None},
    "SLR-M": {"yolo_base": 0, "image": None}
}

STATION_NAMES = { "SC": "Secunderabad", "GNT": "Guntur", "BZA": "Vijayawada", "VSKP": "Visakhapatnam" }
def format_station(code): return STATION_NAMES.get(code, code)

def load_train_database():
    try:
        with open('train_database.csv', mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                composition = [{"id": "LOCO", "type": "LOCO", "flags": []}]
                gen_count, rsrv_count = 1, 1
                
                if "1-Front" in row['Female_Coach_Pos']: composition.append({"id": "SLR-1", "type": "SLR", "flags": ["female"]})
                for _ in range(int(row['Gen_Front'])):
                    composition.append({"id": f"GEN-{gen_count}", "type": "GEN", "flags": []}); gen_count += 1
                for _ in range(int(row['Reserved_Coaches'])):
                    composition.append({"id": f"RSRV-{rsrv_count}", "type": "RSRV", "flags": []}); rsrv_count += 1
                if "1-Middle" in row['Female_Coach_Pos']: composition.append({"id": "SLR-M", "type": "SLR", "flags": ["female"]})
                for _ in range(int(row['Gen_Back'])):
                    composition.append({"id": f"GEN-{gen_count}", "type": "GEN", "flags": []}); gen_count += 1
                if "1-Back" in row['Female_Coach_Pos']: composition.append({"id": "SLR-2", "type": "SLR", "flags": ["female"]})
                    
                stations_data = []
                for idx, st in enumerate(row['Halts'].split('|')):
                    fname = format_station(st)
                    ALL_STATIONS.add(fname)
                    stations_data.append({"name": fname, "code": st, "arr": "--:--", "dep": "--:--"})
                
                t_no = row['Train_No']
                TRAIN_DB[t_no] = {
                    "number": t_no, "name": row['Train_Name'], "type": row['Train_Type'],
                    "route_start": format_station(row['Origin']), "route_end": format_station(row['Destination']),
                    "stations": stations_data, "composition": composition
                }
                UTS_DB[t_no] = {st['name'].split(' (')[0]: {'boarding': 0, 'departing': 0} for st in stations_data}
    except Exception as e: print(f"[ERROR] DB Load Failed: {e}")

load_train_database()

def distribute_passengers(total_passengers, base_densities, is_boarding=True):
    """
    Advanced Distribution Algorithm.
    Uses squared weighting to exponentially favor empty coaches when boarding, 
    and exponentially favor crowded coaches when departing.
    """
    distributed_changes = {k: 0 for k in base_densities.keys()}
    if total_passengers <= 0: return distributed_changes

    if is_boarding:
        # Square the empty space to heavily prioritize unoccupied coaches
        weights = {k: max(0, 100 - v)**2 for k, v in base_densities.items()}
    else:
        # Square the occupied space to prioritize people leaving crowded coaches
        weights = {k: v**2 for k, v in base_densities.items()}
        
    total_weight = sum(weights.values())
    
    # Fallback to equal distribution if weights zero out
    if total_weight == 0:
        weights = {k: 1 for k in base_densities.keys()}
        total_weight = len(base_densities)

    for k in distributed_changes.keys():
        # Conversion: 1 passenger occupies roughly 0.7% of a standard general coach
        density_change = (total_passengers * (weights[k] / total_weight)) * 0.7
        distributed_changes[k] = density_change

    return distributed_changes

@app.route('/')
def passenger_dashboard(): return render_template('index.html')

@app.route('/admin')
def admin_dashboard(): return render_template('admin.html')

@app.route('/uts')
def uts_dashboard(): return render_template('uts.html')

@app.route('/api/stations', methods=['GET'])
def get_stations(): return jsonify(sorted(list(ALL_STATIONS)))

@app.route('/api/train_list', methods=['GET'])
def get_train_list(): return jsonify([{"number": t["number"], "name": t["name"]} for t in TRAIN_DB.values()])

@app.route('/api/get_train/<train_no>', methods=['GET'])
def get_train(train_no):
    if train_no in TRAIN_DB: return jsonify(TRAIN_DB[train_no])
    return jsonify({"error": "Train not found"}), 404

@app.route('/api/search_route', methods=['GET'])
def search_route():
    src = request.args.get('src', '').lower()
    dest = request.args.get('dest', '').lower()
    results = []
    for t_no, t_data in TRAIN_DB.items():
        st_names = [s['name'].lower() for s in t_data['stations']]
        src_match = next((s for s in st_names if src in s), None)
        dest_match = next((s for s in st_names if dest in s), None)
        if src_match and dest_match and st_names.index(src_match) < st_names.index(dest_match):
            results.append({"number": t_no, "name": t_data['name'], "type": t_data['type'], "departure": t_data['stations'][st_names.index(src_match)]['dep']})
    return jsonify(results)

@app.route('/api/book_uts', methods=['POST'])
def book_uts():
    data = request.json
    t_no = data['train_no']
    src = data['source']
    dest = data['destination']
    count = data['passengers']
    
    if t_no in UTS_DB:
        if src in UTS_DB[t_no]: UTS_DB[t_no][src]['boarding'] += count
        if dest in UTS_DB[t_no]: UTS_DB[t_no][dest]['departing'] += count
    return jsonify({"status": "success"})

@app.route('/api/admin/set_train', methods=['POST'])
def set_train():
    GLOBAL_STATE['active_train'] = request.json['train_no']
    GLOBAL_STATE['station_index'] = 0
    return jsonify({"status": "success"})

@app.route('/api/admin/depart', methods=['POST'])
def depart():
    if not GLOBAL_STATE['active_train']: return jsonify({"error": "No train active"})
    
    t_no = GLOBAL_STATE['active_train']
    train_info = TRAIN_DB[t_no]
    curr_idx = GLOBAL_STATE['station_index']
    
    if curr_idx >= len(train_info['stations']) - 1:
        return jsonify({"error": "End of journey reached"})

    curr_st = train_info['stations'][curr_idx]['name'].split(' (')[0]
    next_st = train_info['stations'][curr_idx + 1]['name'].split(' (')[0]

    # Calculate average density for logging
    total_crowd = sum(c['yolo_base'] for c in train_state_coaches.values())
    avg_val = round(total_crowd / max(1, len(train_state_coaches)), 2)
    crowd_cat = "Critical" if avg_val > 85 else "High" if avg_val > 60 else "Medium" if avg_val > 30 else "Low"
    
    row = [datetime.now().strftime("%d-%m-%Y"), datetime.now().strftime("%H:%M:%S"), t_no, train_info['route_start'], train_info['route_end'], curr_st, next_st, avg_val, crowd_cat]
    file_exists = os.path.isfile('historical_crowd_data.csv')
    with open('historical_crowd_data.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists: writer.writerow(['Date', 'Time', 'Train No', 'Route Start', 'Route End', 'Current Station', 'Next Station', 'Average Crowd', 'Crowd Category'])
        writer.writerow(row)

    GLOBAL_STATE['station_index'] += 1
    return jsonify({"status": "success", "new_index": GLOBAL_STATE['station_index']})

@app.route('/api/update_crowd', methods=['POST'])
def update_crowd():
    # Called exclusively by edge_node.py
    data = request.json
    coach_id = data.get('coach_id')
    if coach_id in train_state_coaches:
        train_state_coaches[coach_id]['yolo_base'] = data.get('crowd_percent')
        train_state_coaches[coach_id]['image'] = f"/static/captures/{coach_id}_live.jpg"
    return jsonify({"status": "success"})

@app.route('/api/get_live_status', methods=['GET'])
def get_live_status(): 
    t_no = GLOBAL_STATE['active_train']
    if not t_no: return jsonify({"active_train": None})

    train_info = TRAIN_DB[t_no]
    curr_idx = GLOBAL_STATE['station_index']
    total_stations = len(train_info['stations'])
    
    is_final_station = (curr_idx == total_stations - 1)
    is_penultimate = (curr_idx == total_stations - 2)

    curr_st = train_info['stations'][curr_idx]['name'].split(' (')[0]
    next_st = train_info['stations'][curr_idx + 1]['name'].split(' (')[0] if not is_final_station else None

    # Retrieve UTS data for mathematical modifications
    curr_departing = UTS_DB[t_no][curr_st]['departing'] if curr_st in UTS_DB[t_no] else 0
    curr_boarding = UTS_DB[t_no][curr_st]['boarding'] if curr_st in UTS_DB[t_no] else 0
    
    next_departing = UTS_DB[t_no][next_st]['departing'] if next_st and next_st in UTS_DB[t_no] else 0
    next_boarding = UTS_DB[t_no][next_st]['boarding'] if next_st and next_st in UTS_DB[t_no] else 0

    base_densities = {k: v['yolo_base'] for k, v in train_state_coaches.items()}
    
    # Calculate Live Transformations based on UTS activity at CURRENT station
    curr_dep_dist = distribute_passengers(curr_departing, base_densities, is_boarding=False)
    curr_brd_dist = distribute_passengers(curr_boarding, base_densities, is_boarding=True)

    live_coaches = {}
    for coach_id, cdata in train_state_coaches.items():
        if is_final_station:
            live_crowd = 0; expected_crowd = 0
        else:
            # Live Crowd = YOLO Base - Current Departures + Current Boardings
            live_crowd = cdata['yolo_base'] - curr_dep_dist[coach_id] + curr_brd_dist[coach_id]
            live_crowd = max(0, min(int(live_crowd), 100))
            
            if is_penultimate:
                expected_crowd = 0
            else:
                # Expected Crowd = Live Crowd - Future Departures + Future Boardings
                next_dep_dist = distribute_passengers(next_departing, {coach_id: live_crowd}, is_boarding=False)
                next_brd_dist = distribute_passengers(next_boarding, {coach_id: live_crowd}, is_boarding=True)
                expected = live_crowd - next_dep_dist[coach_id] + next_brd_dist[coach_id]
                expected_crowd = max(0, min(int(expected), 100))

        live_coaches[coach_id] = {
            "crowd": live_crowd, 
            "expected": expected_crowd, 
            "image": cdata['image']
        }

    return jsonify({"active_train": t_no, "station_index": curr_idx, "train_data": train_info, "coaches": live_coaches})

@app.route('/api/demo_process', methods=['POST'])
def demo_process():
    """ Methodology Validation API for Panel Presentation """
    try:
        data = request.json
        encoded_data = data.get('image').split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        h, w = img.shape[:2]
        boxes = model.predict(img, classes=[0], conf=0.15, verbose=False)[0].boxes.xyxy.cpu().numpy()
        
        overlay = np.zeros_like(img)
        total_weighted_mass = 0
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            feet_x = int((x1 + x2) / 2)
            feet_y = int(y2)
            
            depth_ratio = 1.0 - (feet_y / h) 
            occlusion_weight = 1.0 + (depth_ratio * 2.0) 
            total_weighted_mass += occlusion_weight
            
            radius = int(w * 0.05 * (feet_y / h)) 
            if radius > 0:
                cv2.circle(overlay, (feet_x, feet_y), radius, (0, 0, 255), -1)
                cv2.rectangle(overlay, (x1, int(y1 + (y2-y1)*0.2)), (x2, y2), (0, 0, 255), 2)

        density = min(int((total_weighted_mass / 35.0) * 100), 100)
        visual_img = cv2.addWeighted(img, 0.7, overlay, 0.6, 0)
        cv2.putText(visual_img, f"Mass Index: {round(total_weighted_mass,1)} | Spatial Density: {density}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        _, buffer = cv2.imencode('.jpg', visual_img)
        processed_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({"status": "success", "density": density, "processed_image": processed_b64})
    except Exception as e: return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    if not os.path.exists('static/captures'): os.makedirs('static/captures')
    app.run(debug=True, host='0.0.0.0', port=5000)