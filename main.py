from flask import Flask, request, jsonify, send_from_directory
import yaml
import numpy as np
import cv2
from datetime import datetime
import openpyxl
import os
from flask_cors import CORS
app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

app.config['UPLOAD_FOLDER'] = './uploads/'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

service_url = "http://localhost:5000"
UPLOAD_FOLDER = './uploads/'
OUTPUT_DIR = './output/'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route('/uploads/<filename>')
def get_result_image1(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/output/<filename>')
def get_result_image2(filename):
    return send_from_directory(OUTPUT_DIR, filename)

# def upload_video():
#     if 'video' not in request.files:
#         return jsonify({"error": "No video file provided"}), 400
    
#     file = request.files['video']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     if file and file.filename.endswith('.mp4'):
#         timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
#         filename = f"uploaded_{timestamp}.mp4"
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(filepath)
#         return jsonify({"file_path": filepath}), 200

#     return jsonify({"error": "Unsupported file type"}), 400

@app.route('/process-video', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.mp4'):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"uploaded_{timestamp}.mp4"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

    video_path = filepath

    # Initialize variables
    start_time = datetime.now()
    last_time = datetime.now()
    ct = 0
    total_output = 0
    fastest = 0
    ppm = 0
    ppm_average = 0
    qty = 0
    rec_qty = 8

    # Store counts every 3 seconds
    time_intervals = []
    object_counts = []

    # Prepare for Excel file output
    output_dir = "./output/"
    os.makedirs(output_dir, exist_ok=True)
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(("datetime", "total_output", "minute", "average ppm", "ct", "ppm"))

    # Generate output file path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    fn_out = os.path.join(output_dir, f"output_{timestamp}.mp4")

    # Configuration settings
    config = {
        'save_video': True,
        'text_overlay': True,
        'object_overlay': True,
        'object_id_overlay': False,
        'object_detection': True,
        'min_area_motion_contour': 1000,
        'park_sec_to_wait': 0.001,
        'start_frame': 0
    }

    # Load video file
    cap = cv2.VideoCapture(video_path)

    if config['save_video']:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_info = {'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        out = cv2.VideoWriter(fn_out, fourcc, 25.0, (video_info['width'], video_info['height']))

    # Load YAML data for object areas
    fn_yaml = "./area.yml"
    with open(fn_yaml, 'r') as stream:
        object_area_data = yaml.safe_load(stream)

    object_contours = []
    object_bounding_rects = []
    object_mask = []

    for park in object_area_data:
        points = np.array(park['points'])
        rect = cv2.boundingRect(points)
        points_shifted = points.copy()
        points_shifted[:, 0] = points[:, 0] - rect[0]
        points_shifted[:, 1] = points[:, 1] - rect[1]
        object_contours.append(points)
        object_bounding_rects.append(rect)
        mask = cv2.drawContours(np.zeros((rect[3], rect[2]), dtype=np.uint8), [points_shifted], contourIdx=-1, color=255, thickness=-1, lineType=cv2.LINE_8)
        mask = mask == 255
        object_mask.append(mask)

    object_status = [False] * len(object_area_data)
    object_buffer = [None] * len(object_area_data)

    while cap.isOpened():
        try:
            video_cur_pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Current position in seconds
            ret, frame = cap.read()
            if not ret:
                break

            frame_blur = cv2.GaussianBlur(frame.copy(), (5, 5), 3)
            frame_gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
            frame_out = frame.copy()

            if config['object_detection']:
                for ind, park in enumerate(object_area_data):
                    points = np.array(park['points'])
                    rect = object_bounding_rects[ind]
                    roi_gray = frame_gray[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]
                    status = np.std(roi_gray) < 20 and np.mean(roi_gray) > 56

                    if status != object_status[ind] and object_buffer[ind] is None:
                        object_buffer[ind] = video_cur_pos

                    elif status != object_status[ind] and object_buffer[ind] is not None:
                        if video_cur_pos - object_buffer[ind] > config['park_sec_to_wait']:
                            if not status:
                                qty += 1
                                total_output += 1
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
                                    data = (current_time, total_output, minutes, ppm_average, ct, ppm)
                                    ws.append(data)

                                if qty > rec_qty:
                                    data = (current_time, total_output, minutes, ppm_average, ct, ppm)
                                    ws.append(data)
                                    qty = 0

                            object_status[ind] = status
                            object_buffer[ind] = None

                    elif status == object_status[ind] and object_buffer[ind] is not None:
                        object_buffer[ind] = None

            if config['object_overlay']:
                for ind, park in enumerate(object_area_data):
                    points = np.array(park['points'])
                    color = (0, 255, 0) if object_status[ind] else (0, 0, 255)
                    cv2.drawContours(frame_out, [points], contourIdx=-1, color=color, thickness=2, lineType=cv2.LINE_8)
                    moments = cv2.moments(points)
                    centroid = (int(moments['m10'] / moments['m00']) - 3, int(moments['m01'] / moments['m00']) + 3)
                    cv2.putText(frame_out, str(park['id']), centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)

            if config['text_overlay']:
                cv2.rectangle(frame_out, (1, 5), (350, 70), (0, 255, 0), 2)
                str_on_frame = "Object Counting:"
                cv2.putText(frame_out, str_on_frame, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                str_on_frame = f"Total Counting = {total_output}, Speed (PPM) = {ppm}"
                cv2.putText(frame_out, str_on_frame, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                str_on_frame = f"Fastest PPM: {fastest}, Average: {ppm_average}"
                cv2.putText(frame_out, str_on_frame, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

            if config['save_video']:
                out.write(frame_out)

            # Record count every 3 seconds
            if video_cur_pos >= len(time_intervals) * 3:
                time_intervals.append(f"{video_cur_pos:.2f}")
                object_counts.append(str(total_output))

        except KeyboardInterrupt:
            break

    # Ensure last second and count are included
    final_video_pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    if final_video_pos > len(time_intervals) * 3:
        time_intervals.append(f"{final_video_pos:.2f}")
        object_counts.append(str(total_output))

    cap.release()
    if config['save_video']:
        out.release()
    cv2.destroyAllWindows()

    # Create the JSON response with counts
    counts = {
        "time": time_intervals,
        "object_count": object_counts
    }

    # Normalize paths to remove './' or any redundant segments
    normalized_output_video_path = os.path.normpath(fn_out)
    normalized_original_video_path = os.path.normpath(video_path)

    # Construct the full URLs without './'
    output_video_url = f"{service_url}/{normalized_output_video_path.lstrip('./')}"
    original_video_url = f"{service_url}/{normalized_original_video_path.lstrip('./')}"

    return jsonify({
        "output_video_path":  output_video_url,
        "original_video":  original_video_url,
        "counts": counts
    })



if __name__ == '__main__':
    app.run(debug=True)
