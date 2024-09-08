import cv2
import numpy as np
import base64
import yaml
from datetime import datetime
import openpyxl
from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")

# Initialize variables
start_time = datetime.now()
last_time = datetime.now()
total_output = 0
fastest = 0
ppm = 0
ppm_average = 0
rec_qty = 8
qty = 0
is_paused = False
current_frame_number = 0

# Frame rate control (30 FPS)
TARGET_FPS = 18
FRAME_INTERVAL = 1 / TARGET_FPS  # 33ms for 30fps

# Prepare for Excel file output
path = "./output/"
wb = openpyxl.Workbook()
ws = wb.active
ws.append(("datetime", "total_output", "minute", "average ppm", "ct", "ppm"))

# Load the area data from the YAML file
fn_yaml = "./area.yml"
with open(fn_yaml, 'r') as stream:
    area_data = yaml.safe_load(stream)

# Extract points for the counting area and create a bounding box
area_points = np.array(area_data[0]['points'])
x_min = np.min(area_points[:, 0])
x_max = np.max(area_points[:, 0])
y_min = np.min(area_points[:, 1])
y_max = np.max(area_points[:, 1])

# Red color range in HSV
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])

last_detected_frame = -50  # Initial frame value

# Function to decode base64 frame to OpenCV image
def decode_base64_frame(base64_str):
    img_bytes = base64.b64decode(base64_str)
    img_np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)

# Function to process the frame and perform counting
def process_frame(frame):
    global last_time, total_output, qty, fastest, ppm, ppm_average, last_detected_frame

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask to detect red objects
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    # Find contours of the red objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:  # Minimum area for counting
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # Check if the centroid is within the defined area
                if x_min <= cx <= x_max and y_min <= cy <= y_max:
                    # Ensure frame interval before counting again
                    if datetime.now().timestamp() - last_detected_frame > 0.5:
                        qty += 1
                        total_output += 1
                        last_detected_frame = datetime.now().timestamp()

                        # Calculate speed
                        current_time = datetime.now()
                        diff = current_time - last_time
                        ct = diff.total_seconds()
                        ppm = round(60 / ct, 2)
                        last_time = current_time

                        diff = current_time - start_time
                        minutes = diff.total_seconds() / 60
                        ppm_average = round(total_output / minutes, 2)

                        if ppm > fastest:
                            fastest = ppm

                        if qty > rec_qty:
                            data = (current_time, total_output, minutes, ppm_average, ct, ppm)
                            ws.append(data)
                            qty = 0

    # Only return metadata
    return {
        "total_output": total_output,
        "ppm": ppm,
        "ppm_average": ppm_average,
        "fastest": fastest,
        "current_frame_number": current_frame_number
    }

@socketio.on('video_info')
def handle_video_info(data):
    """Receive video information such as pause, current frame number, and process accordingly."""
    global is_paused, current_frame_number

    # Update current frame number and pause status based on frontend data
    current_frame_number = data['frame_number']
    is_paused = data['paused']

    if not is_paused:
        base64_frame = data['frame']
        frame = decode_base64_frame(base64_frame)

        # Process the frame (e.g., count objects)
        metadata = process_frame(frame)

        # Emit only metadata back to frontend
        emit('processed_meta', metadata)

@socketio.on('pause')
def handle_pause(data):
    """Handle the pause signal from the frontend."""
    global is_paused
    is_paused = data['paused']

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
