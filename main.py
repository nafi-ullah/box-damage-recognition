from flask import Flask, request, jsonify
import yaml
import numpy as np
import cv2
from datetime import datetime
import openpyxl
import os

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = './uploads/'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024


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

    # data = request.json
    video_path =  filepath #data.get('video_path', "./testvideo.mp4")  # Default video path if not provided

    # Initialize variables
    start_time = datetime.now()
    last_time = datetime.now()

    ct = 0
    total_output = 0
    fastest = 0
    ppm = 0
    ppm_average = 0

    # Record counting every a qty
    rec_qty = 8
    qty = 0

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
        'save_video': True,  # Set to True to save the video
        'text_overlay': True,
        'object_overlay': True,
        'object_id_overlay': False,
        'object_detection': True,
        'min_area_motion_contour': 1000,  # Adjust based on the box size
        'park_sec_to_wait': 0.001,
        'start_frame': 0
    }

    # Load video file instead of capturing from a camera
    cap = cv2.VideoCapture(video_path)

    # Define the codec and create VideoWriter object if saving video
    if config['save_video']:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
        video_info = {'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        out = cv2.VideoWriter(fn_out, fourcc, 25.0, (video_info['width'], video_info['height']))

    # Read YAML data for object areas
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
        points_shifted[:, 0] = points[:, 0] - rect[0]  # Shift contour to ROI
        points_shifted[:, 1] = points[:, 1] - rect[1]
        object_contours.append(points)
        object_bounding_rects.append(rect)
        mask = cv2.drawContours(np.zeros((rect[3], rect[2]), dtype=np.uint8), [points_shifted], contourIdx=-1, color=255, thickness=-1, lineType=cv2.LINE_8)
        mask = mask == 255
        object_mask.append(mask)

    object_status = [False] * len(object_area_data)
    object_buffer = [None] * len(object_area_data)

    print("Program for counting objects crossing the line. Frame size: 960x720")

    while cap.isOpened():
        try:
            # Read frame-by-frame
            video_cur_pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Current position of the video file in seconds
            ret, frame = cap.read()
            if not ret:
                print("Capture Error")
                break

            frame_blur = cv2.GaussianBlur(frame.copy(), (5, 5), 3)
            frame_gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
            frame_out = frame.copy()

            if config['object_detection']:
                for ind, park in enumerate(object_area_data):
                    points = np.array(park['points'])
                    rect = object_bounding_rects[ind]
                    roi_gray = frame_gray[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]  # Crop ROI for faster calculation
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
                                    # Record to Excel
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

            # Save the frame to the output video
            if config['save_video']:
                out.write(frame_out)

            k = cv2.waitKey(1)
            if k == ord('q'):
                break

        except KeyboardInterrupt:
            data = (current_time, total_output, minutes, ppm_average, ct, ppm)
            ws.append(data)
            wb.save(os.path.join(output_dir, "output_" + start_time.strftime("%d-%m-%Y %H-%M-%S") + ".xlsx"))
            print("Actual Speed (PPM): " + str(ppm_average))
            break

    cap.release()
    if config['save_video']:
        out.release()
    cv2.destroyAllWindows()

    return jsonify({"output_video_path": fn_out})

if __name__ == '__main__':
    app.run(debug=True)
