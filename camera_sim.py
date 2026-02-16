import requests
import time
import random

# URL of your Flask Backend
URL = "http://127.0.0.1:5000/api/update_crowd"

def simulate_journey():
    stations = ["Vijayawada", "Mustabada", "Gannavaram", "Nuzvid"]
    
    while True:
        for station in stations:
            # Check if this is a reversal station (Example: Vijayawada)
            is_reversed = True if station == "Vijayawada" else False
            
            print(f"\n🚂 Train Arrived at: {station} | Reversal: {is_reversed}")
            
            # Simulate 10 updates (20 seconds) per station
            for _ in range(10):
                # Generate Random Crowd Data for each coach
                payload = {
                    "current_station": station,
                    "is_reversed": is_reversed,
                    "crowd_data": {
                        "GEN-1": random.randint(80, 100), # Crowded Front
                        "GEN-2": random.randint(60, 90),
                        "DIV-1": random.randint(10, 40),
                        "LAD-1": random.randint(20, 50),
                        "GEN-3": random.randint(0, 30),   # Empty Back
                        "GEN-4": random.randint(0, 20)
                    }
                }
                
                try:
                    requests.post(URL, json=payload)
                    print(f"Sent Data: GEN-1={payload['crowd_data']['GEN-1']}%")
                except:
                    print("Server not running!")
                
                time.sleep(2) # Wait 2 seconds

if __name__ == "__main__":
    print("Camera Simulator Started...")
    time.sleep(1)
    simulate_journey()