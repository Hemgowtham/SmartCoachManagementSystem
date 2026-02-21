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
    if name == code:
        return name
    return f"{name} ({code})"

def load_train_database():
    try:
        with open('train_database.csv', mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                composition = [{"id": "LOCO", "type": "LOCO", "flags": []}]
                gen_count = 1
                rsrv_count = 1
                
                front_flags = []
                if "1-Front" in row['Female_Coach_Pos']: front_flags.append("female")
                if "1-Front" in row['Disabled_Coach_Pos']: front_flags.append("disabled")
                if front_flags:
                    composition.append({"id": "SLR-1", "type": "SLR", "flags": front_flags})

                for _ in range(int(row['Gen_Front'])):
                    composition.append({"id": f"GEN-{gen_count}", "type": "GEN", "flags": []})
                    gen_count += 1
                    
                for _ in range(int(row['Reserved_Coaches'])):
                    composition.append({"id": f"RSRV-{rsrv_count}", "type": "RSRV", "flags": []})
                    rsrv_count += 1
                    
                mid_flags = []
                if "1-Middle" in row['Female_Coach_Pos']: mid_flags.append("female")
                if "1-Middle" in row['Disabled_Coach_Pos']: mid_flags.append("disabled")
                if mid_flags:
                    composition.append({"id": "SLR-M", "type": "SLR", "flags": mid_flags})

                for _ in range(int(row['Gen_Back'])):
                    composition.append({"id": f"GEN-{gen_count}", "type": "GEN", "flags": []})
                    gen_count += 1

                back_flags = []
                if "1-Back" in row['Female_Coach_Pos']: back_flags.append("female")
                if "1-Back" in row['Disabled_Coach_Pos']: back_flags.append("disabled")
                if back_flags:
                    composition.append({"id": "SLR-2", "type": "SLR", "flags": back_flags})

                stations_list = row['Halts'].split('|')
                arr_list = row['Arrival_Times'].split('|')
                dep_list = row['Departure_Times'].split('|')
                
                stations_data = []
                for idx, st in enumerate(stations_list):
                    formatted_name = format_station(st)
                    ALL_STATIONS.add(formatted_name)
                    arr = arr_list[idx] if idx < len(arr_list) else "--:--"
                    dep = dep_list[idx] if idx < len(dep_list) else "--:--"
                    stations_data.append({"name": formatted_name, "code": st, "arr": arr, "dep": dep})
                
                reversal = format_station(row['Reversal_Station'])
                dest_formatted = format_station(row['Destination'])
                if reversal == dest_formatted:
                    reversal = None

                TRAIN_DB[row['Train_No']] = {
                    "number": row['Train_No'],
                    "name": row['Train_Name'],
                    "type": row['Train_Type'],
                    "route_str": f"{format_station(row['Origin'])} -> {dest_formatted}",
                    "route_start": format_station(row['Origin']),
                    "route_end": dest_formatted,
                    "reversal_at": reversal,
                    "stations": stations_data,
                    "composition": composition
                }
        print(f"Loaded {len(TRAIN_DB)} trains from database.")
    except Exception as e:
        print(f"Error loading train_database.csv: {e}")

load_train_database()

# MANUAL OVERRIDE: Pre-filling the system with manual data and images for demonstration
train_state_coaches = {
    "GEN-1": {"crowd": 85, "image": "/test_images/GEN-1.jpg"},
    "GEN-2": {"crowd": 45, "image": "/test_images/GEN-2.jpg"},
    "GEN-3": {"crowd": 20, "image": "/test_images/GEN-3.jpg"},
    "GEN-4": {"crowd": 95, "image": "/test_images/GEN-4.jpg"},
    "SLR-1": {"crowd": 30, "image": "/test_images/GEN-1.jpg"},
    "SLR-2": {"crowd": 15, "image": "/test_images/GEN-2.jpg"},
    "SLR-M": {"crowd": 50, "image": "/test_images/GEN-3.jpg"}
}

@app.route('/')
def passenger_dashboard():
    return render_template('index.html')

@app.route('/admin')
def admin_dashboard():
    return render_template('admin.html')

# NEW ROUTE: Serve the manual test images directly to the dashboard
@app.route('/test_images/<filename>')
def serve_test_images(filename):
    return send_from_directory('test_images', filename)

@app.route('/api/get_train/<train_no>', methods=['GET'])
def get_train(train_no):
    if train_no in TRAIN_DB:
        return jsonify(TRAIN_DB[train_no])
    return jsonify({"error": "Train not found"}), 404

@app.route('/api/stations', methods=['GET'])
def get_stations():
    return jsonify(sorted(list(ALL_STATIONS)))

@app.route('/api/train_list', methods=['GET'])
def get_train_list():
    trains = [{"number": t["number"], "name": t["name"]} for t in TRAIN_DB.values()]
    return jsonify(trains)

@app.route('/api/search_route', methods=['GET'])
def search_route():
    src = request.args.get('src', '').lower()
    dest = request.args.get('dest', '').lower()
    results = []
    
    for t_no, t_data in TRAIN_DB.items():
        st_names = [s['name'].lower() for s in t_data['stations']]
        
        src_match = next((s for s in st_names if src in s), None)
        dest_match = next((s for s in st_names if dest in s), None)

        if src_match and dest_match:
            src_idx = st_names.index(src_match)
            dest_idx = st_names.index(dest_match)
            if src_idx < dest_idx:
                results.append({
                    "number": t_no, 
                    "name": t_data['name'], 
                    "type": t_data['type'], 
                    "departure": t_data['stations'][src_idx]['dep']
                })
    return jsonify(results)

@app.route('/api/log_segment', methods=['POST'])
def log_segment():
    try:
        data = request.json
        
        def clean_name(val):
            if not val: return "Unknown"
            return val.split(' (')[0].strip()

        train_no = data.get('train_no', 'Unknown')
        route_start = clean_name(data.get('route_start', 'Unknown'))
        route_end = clean_name(data.get('route_end', 'Unknown'))
        current_station = clean_name(data.get('current_station', 'Unknown'))
        next_station = clean_name(data.get('next_station', 'Unknown'))

        total_percent = 0
        num_coaches = len(train_state_coaches)
        for coach, cdata in train_state_coaches.items():
            total_percent += cdata['crowd']

        avg_percent = round(total_percent / num_coaches, 2) if num_coaches > 0 else 0

        if avg_percent <= 30: crowd_category = "Low"
        elif avg_percent <= 60: crowd_category = "Medium"
        elif avg_percent <= 85: crowd_category = "High"
        else: crowd_category = "Critical"

        now = datetime.now()
        row = [
            now.strftime("%d-%m-%Y"), now.strftime("%A"), train_no, route_start, route_end, 
            current_station, next_station, avg_percent, crowd_category
        ]

        csv_filename = 'historical_crowd_data.csv'
        file_exists = os.path.isfile(csv_filename)
        
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Date', 'Day', 'Train No', 'Route Start', 'Route End', 'Current Station', 'Next Station', 'Average Crowd', 'Crowd Category'])
            writer.writerow(row)

        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error"})

@app.route('/api/update_crowd', methods=['POST'])
def update_crowd():
    data = request.json
    coach_id = data.get('coach_id')
    crowd_percent = data.get('crowd_percent')
    
    if coach_id in train_state_coaches:
        train_state_coaches[coach_id]['crowd'] = crowd_percent
        train_state_coaches[coach_id]['image'] = f"/static/captures/{coach_id}_live.jpg"

    return jsonify({"status": "success"})

@app.route('/api/simulate_edge_processing', methods=['POST'])
def simulate_edge_processing():
    try:
        data = request.json
        image_data = data.get('image')
        coach_id = data.get('coach_id')

        encoded_data = image_data.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = model.predict(img, classes=[0], conf=0.3, verbose=False)
        count = len(results[0].boxes)
        
        annotated_frame = results[0].plot()
        filename = f"{coach_id}_processed.jpg"
        save_path = os.path.join("static/captures", filename)
        cv2.imwrite(save_path, annotated_frame)

        capacity = 20
        density = int((count / capacity) * 100)
        if density > 100: density = 100

        if coach_id in train_state_coaches:
            train_state_coaches[coach_id]['crowd'] = density
            train_state_coaches[coach_id]['image'] = f"/static/captures/{filename}"

        return jsonify({"status": "success", "count": count, "density": density})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/get_status', methods=['GET'])
def get_status():
    simple_coaches = {k: v['crowd'] for k, v in train_state_coaches.items()}
    return jsonify({"coaches": simple_coaches})

@app.route('/api/get_admin_data', methods=['GET'])
def get_admin_data():
    return jsonify({"coaches": train_state_coaches})

if __name__ == '__main__':
    if not os.path.exists('static/captures'):
        os.makedirs('static/captures')
    app.run(debug=True, host='0.0.0.0', port=5000)