import cv2
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import numpy as np
import base64
import eventlet
#pip install flask opencv-python flask-socketio eventlet
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})  # Enable CORS for your frontend

socketio = SocketIO(app, cors_allowed_origins="http://localhost:3000")
# socketio = SocketIO(app, async_mode='eventlet')

def decode_base64_frame(base64_str):
    """Decode base64 string to an OpenCV image (frame)."""
    img_bytes = base64.b64decode(base64_str)
    img_np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)

@socketio.on('video_frame')
def handle_video_stream(data):
    """Receive base64-encoded frame from frontend, process it, and send back."""
    base64_frame = data['frame']
    frame = decode_base64_frame(base64_frame)

    # Process the frame using OpenCV (e.g., add a red circle)
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2
    cv2.circle(frame, (center_x, center_y), 75 // 2, (0, 0, 255), -1)

    # Encode frame back to base64 to send to frontend
    _, buffer = cv2.imencode('.jpg', frame)
    processed_frame = base64.b64encode(buffer).decode('utf-8')

    emit('processed_frame', {'frame': processed_frame})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
