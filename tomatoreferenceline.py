import yaml
import numpy as np
import cv2
import openpyxl
from flask import Flask
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import base64
import time

app = Flask(__name__)
# CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})
# socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize variables
last_time = time.time()
ct = 0
total_output = 0
fastest = 0
ppm = 0
ppm_average = 0
rec_qty = 8
qty = 0

# Excel file output setup
wb = openpyxl.Workbook()
ws = wb.active
ws.append(("datetime", "total_output", "minute", "average ppm", "ct", "ppm"))

# Configuration settings
config = {
    'save_video': True,
    'text_overlay': True,
    'object_overlay': True,
    'object_id_overlay': False,
    'object_detection': True,
    'min_area_motion_contour': 1000,
    'park_sec_to_wait': 0.001,
    'frame_interval': 50  # Frames to wait before counting another object
}

# Reference lines for detecting objects
reference_lines = [
    [(160, 312), (965, 312)],  # Horizontal line
    [(160, 270), (965, 270)]   # Horizontal line
]

# Red color range in HSV
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])

tracked_objects = {}

FRAME_INTERVAL = 1 / 30  # 30 FPS

def is_crossing_line(centroid, line_start, line_end):
    """Check if the centroid crosses the reference line."""
    x1, y1 = line_start
    x2, y2 = line_end
    x, y = centroid
    return min(x1, x2) <= x <= max(x1, x2) and abs(y - y1) < 5


def process_frame(frame):
    """Process video frame and track objects."""
    global last_time, ct, total_output, ppm, ppm_average, fastest, qty

    frame_out = frame.copy()

    # Draw reference lines on the frame
    for line in reference_lines:
        cv2.line(frame_out, line[0], line[1], (0, 0, 255), 2)

    # Convert frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create mask to detect red objects
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    # Find contours of the red objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > config['min_area_motion_contour']:
            # Calculate centroid of the detected object
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroid = (cx, cy)

                # Check if the object is already tracked
                matched_id = None
                for obj_id, (prev_centroid, frames_since_seen) in tracked_objects.items():
                    distance = np.linalg.norm(np.array(centroid) - np.array(prev_centroid))
                    if distance < 50:  # Matching threshold
                        matched_id = obj_id
                        break

                if matched_id is None:
                    matched_id = len(tracked_objects) + 1

                # Update tracked object
                tracked_objects[matched_id] = (centroid, 0)

                # Check if the centroid is crossing any reference line
                for line in reference_lines:
                    if is_crossing_line(centroid, line[0], line[1]):
                        qty += 1
                        total_output += 1

                        current_time = time.time()
                        diff = current_time - last_time
                        ct = diff
                        ppm = round(60 / ct, 2)
                        last_time = current_time

                        diff = current_time - last_time
                        minutes = diff / 60
                        
                        # Check for zero division error before calculating ppm_average
                        if minutes > 0:
                            ppm_average = round(total_output / minutes, 2)
                        else:
                            ppm_average = 0

                        if ppm > fastest:
                            fastest = ppm
                        ws.append((current_time, total_output, minutes, ppm_average, ct, ppm))

                        if qty > rec_qty:
                            ws.append((current_time, total_output, minutes, ppm_average, ct, ppm))
                            qty = 0

                        del tracked_objects[matched_id]

    return total_output


@socketio.on('video_frame')
def handle_video_stream(data):
    """Receive base64-encoded frame from frontend, process it, and send back the count."""
    # Decode base64 frame
    base64_frame = data['frame']
    nparr = np.frombuffer(base64.b64decode(base64_frame), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process the frame (counting tomatoes)
    tomato_count = process_frame(frame)

    # Send tomato count to frontend
    emit('tomato_count', {'count': tomato_count})

@socketio.on('restart_count')
def handle_restart_count():
    """Handle the restart count signal from the frontend and reset the count."""
    global total_output
    total_output = 0  # Reset the tomato count
    emit('tomato_count', {'count': total_output})  # Send the reset count back to the frontend


if __name__ == '__main__':
    try:
        socketio.run(app, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        # Save Excel file
        wb.save("./output/output_" + time.strftime("%d-%m-%Y_%H-%M-%S") + ".xlsx")
