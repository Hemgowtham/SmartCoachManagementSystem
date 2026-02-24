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

print("Loading YOLOv8 Model...")
model = YOLO("yolov8n.pt") 

TRAIN_DB = {}
ALL_STATIONS = set()

STATION_NAMES = {
    "SC": "Secunderabad", "GNT": "Guntur", "BZA": "Vijayawada", "VSKP": "Visakhapatnam",
    "BBS": "Bhubaneswar", "HWH": "Howrah", "RJY": "Rajahmundry", "WL": "Warangal",
    "HYB": "Hyderabad", "PUNE": "Pune", "CSMT": "CSMT Mumbai", "LPI": "Lingampalli",
    "TVC": "Trivandrum", "ERN": "Ernakulam", "CBE": "Coimbatore", "NDLS": "New Delhi",
    "SHM": "Shalimar", "KGP": "Kharagpur", "MAS": "Chennai Central", "CDNR": "Chandanagar",
    "HFZ": "Hafizpet", "BMT": "Begumpet", "KCG": "Kacheguda", "FM": "Falaknuma",
    "KCC": "Krishna Canal", "MAG": "Mangalagiri", "MS": "Chennai Egmore", "TBM": "Tambaram",
    "TEL": "Tenali", "OGL": "Ongole", "NLR": "Nellore", "GDR": "Gudur"
}

def format_station(code):
    name = STATION_NAMES.get(code, code)
    if name == code: return name
    return f"{name} ({code})"

def load_train_database():
    try:
        with open('train_database.csv', mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                composition = [{"id": "LOCO", "type": "LOCO", "flags": []}]
                gen_count, rsrv_count = 1, 1
                
                front_flags = []
                if "1-Front" in row['Female_Coach_Pos']: front_flags.append("female")
                if "1-Front" in row['Disabled_Coach_Pos']: front_flags.append("disabled")
                if front_flags: composition.append({"id": "SLR-1", "type": "SLR", "flags": front_flags})

                for _ in range(int(row['Gen_Front'])):
                    composition.append({"id": f"GEN-{gen_count}", "type": "GEN", "flags": []})
                    gen_count += 1
                for _ in range(int(row['Reserved_Coaches'])):
                    composition.append({"id": f"RSRV-{rsrv_count}", "type": "RSRV", "flags": []})
                    rsrv_count += 1
                    
                mid_flags = []
                if "1-Middle" in row['Female_Coach_Pos']: mid_flags.append("female")
                if "1-Middle" in row['Disabled_Coach_Pos']: mid_flags.append("disabled")
                if mid_flags: composition.append({"id": "SLR-M", "type": "SLR", "flags": mid_flags})

                for _ in range(int(row['Gen_Back'])):
                    composition.append({"id": f"GEN-{gen_count}", "type": "GEN", "flags": []})
                    gen_count += 1

                back_flags = []
                if "1-Back" in row['Female_Coach_Pos']: back_flags.append("female")
                if "1-Back" in row['Disabled_Coach_Pos']: back_flags.append("disabled")
                if back_flags: composition.append({"id": "SLR-2", "type": "SLR", "flags": back_flags})

                stations_data = []
                for idx, st in enumerate(row['Halts'].split('|')):
                    fname = format_station(st)
                    ALL_STATIONS.add(fname)
                    arr = row['Arrival_Times'].split('|')[idx] if idx < len(row['Arrival_Times'].split('|')) else "--:--"
                    dep = row['Departure_Times'].split('|')[idx] if idx < len(row['Departure_Times'].split('|')) else "--:--"
                    stations_data.append({"name": fname, "code": st, "arr": arr, "dep": dep})
                
                rev = format_station(row['Reversal_Station'])
                dest = format_station(row['Destination'])
                
                TRAIN_DB[row['Train_No']] = {
                    "number": row['Train_No'], "name": row['Train_Name'], "type": row['Train_Type'],
                    "route_str": f"{format_station(row['Origin'])} -> {dest}",
                    "route_start": format_station(row['Origin']), "route_end": dest,
                    "reversal_at": None if rev == dest else rev, "stations": stations_data, "composition": composition
                }
    except Exception as e: print(f"DB Error: {e}")

load_train_database()

train_state_coaches = {
    "GEN-1": {"crowd": 0, "image_src": "test_images/GEN-1.jpg", "image": None},
    "GEN-2": {"crowd": 0, "image_src": "test_images/GEN-2.jpg", "image": None},
    "GEN-3": {"crowd": 0, "image_src": "test_images/GEN-3.jpg", "image": None},
    "GEN-4": {"crowd": 0, "image_src": "test_images/GEN-4.jpg", "image": None},
    "SLR-1": {"crowd": 0, "image_src": "test_images/SLR-1.jpg", "image": None},
    "SLR-2": {"crowd": 0, "image_src": "test_images/SLR-2.jpg", "image": None},
    "SLR-M": {"crowd": 0, "image_src": "test_images/SLR-M.jpg", "image": None}
}

def calculate_and_visualize_density(img, results):
    h, w = img.shape[:2]
    boxes = results[0].boxes.xyxy.cpu().numpy()
    count = len(boxes)

    if count == 0:
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
                cv2.rectangle(overlay, (cell_x1, cell_y1), (cell_x2, cell_y2), (0, 0, 255), -1) 
                occupied_cells += 1
                
    MAX_GRID_FILL = 0.45
    if total_valid_cells > 0:
        raw_fill = occupied_cells / total_valid_cells
        density = int((raw_fill / MAX_GRID_FILL) * 100)
    else:
        density = 0
        
    density = min(density, 100) 
    visual_img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        
    return density, visual_img

def process_startup_images():
    print("Initializing Server: Clean Heatmap Analytics...")
    if not os.path.exists('static/captures'): os.makedirs('static/captures')
    for coach_id, data in train_state_coaches.items():
        img_path = data.get("image_src")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            results = model.predict(img, classes=[0], conf=0.25, verbose=False)
            density, visual_img = calculate_and_visualize_density(img, results)
            filename = f"{coach_id}_startup.jpg"
            cv2.imwrite(os.path.join("static/captures", filename), visual_img)
            train_state_coaches[coach_id]["crowd"] = density
            train_state_coaches[coach_id]["image"] = f"/static/captures/{filename}"

process_startup_images()

@app.route('/')
def passenger_dashboard(): return render_template('index.html')

@app.route('/admin')
def admin_dashboard(): return render_template('admin.html')

@app.route('/test_images/<filename>')
def serve_test_images(filename): return send_from_directory('test_images', filename)

@app.route('/api/get_train/<train_no>', methods=['GET'])
def get_train(train_no):
    if train_no in TRAIN_DB: return jsonify(TRAIN_DB[train_no])
    return jsonify({"error": "Train not found"}), 404

@app.route('/api/stations', methods=['GET'])
def get_stations(): return jsonify(sorted(list(ALL_STATIONS)))

@app.route('/api/train_list', methods=['GET'])
def get_train_list(): return jsonify([{"number": t["number"], "name": t["name"]} for t in TRAIN_DB.values()])

@app.route('/api/search_route', methods=['GET'])
def search_route():
    src, dest = request.args.get('src', '').lower(), request.args.get('dest', '').lower()
    results = []
    for t_no, t_data in TRAIN_DB.items():
        st_names = [s['name'].lower() for s in t_data['stations']]
        src_match = next((s for s in st_names if src in s), None)
        dest_match = next((s for s in st_names if dest in s), None)
        if src_match and dest_match and st_names.index(src_match) < st_names.index(dest_match):
            results.append({"number": t_no, "name": t_data['name'], "type": t_data['type'], "departure": t_data['stations'][st_names.index(src_match)]['dep']})
    return jsonify(results)

@app.route('/api/log_segment', methods=['POST'])
def log_segment():
    try:
        data = request.json
        clean_name = lambda val: val.split(' (')[0].strip() if val else "Unknown"
        avg_val = round(sum(c['crowd'] for c in train_state_coaches.values()) / max(1, len(train_state_coaches)), 2)
        crowd_cat = "Critical" if avg_val > 85 else "High" if avg_val > 60 else "Medium" if avg_val > 30 else "Low"
        
        row = [datetime.now().strftime("%d-%m-%Y"), datetime.now().strftime("%A"), data.get('train_no', 'Unknown'), clean_name(data.get('route_start', '')), clean_name(data.get('route_end', '')), clean_name(data.get('current_station', '')), clean_name(data.get('next_station', '')), avg_val, crowd_cat]
        file_exists = os.path.isfile('historical_crowd_data.csv')
        with open('historical_crowd_data.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists: writer.writerow(['Date', 'Day', 'Train No', 'Route Start', 'Route End', 'Current Station', 'Next Station', 'Average Crowd', 'Crowd Category'])
            writer.writerow(row)
        return jsonify({"status": "success"})
    except: return jsonify({"status": "error"})

@app.route('/api/update_crowd', methods=['POST'])
def update_crowd():
    data = request.json
    coach_id = data.get('coach_id')
    if coach_id in train_state_coaches:
        train_state_coaches[coach_id]['crowd'] = data.get('crowd_percent')
        train_state_coaches[coach_id]['image'] = f"/static/captures/{coach_id}_live.jpg"
    return jsonify({"status": "success"})

@app.route('/api/simulate_edge_processing', methods=['POST'])
def simulate_edge_processing():
    try:
        data = request.json
        coach_id = data.get('coach_id')
        img = cv2.imdecode(np.frombuffer(base64.b64decode(data.get('image').split(',')[1]), np.uint8), cv2.IMREAD_COLOR)
        results = model.predict(img, classes=[0], conf=0.25, verbose=False)
        density, visual_img = calculate_and_visualize_density(img, results)
        filename = f"{coach_id}_processed.jpg"
        cv2.imwrite(os.path.join("static/captures", filename), visual_img)
        if coach_id in train_state_coaches:
            train_state_coaches[coach_id]['crowd'] = density
            train_state_coaches[coach_id]['image'] = f"/static/captures/{filename}"
        return jsonify({"status": "success", "count": len(results[0].boxes), "density": density})
    except Exception as e: return jsonify({"status": "error", "message": str(e)})

@app.route('/api/get_status', methods=['GET'])
def get_status(): return jsonify({"coaches": {k: v['crowd'] for k, v in train_state_coaches.items()}})

@app.route('/api/get_admin_data', methods=['GET'])
def get_admin_data(): return jsonify({"coaches": train_state_coaches})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)