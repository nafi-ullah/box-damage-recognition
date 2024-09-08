import yaml
import numpy as np
import cv2
from datetime import datetime
import openpyxl

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
path = "./output/"
wb = openpyxl.Workbook()
ws = wb.active
ws.append(("datetime", "total_output", "minute", "average ppm", "ct", "ppm"))

# File paths
fn_yaml = "./area.yml"
fn_out = r"./output/output.mp4"
video_path = "./datasets/tomatofast.mp4"

# Configuration settings
config = {
    'save_video': True, 
    'text_overlay': True,
    'object_overlay': True,
    'object_id_overlay': False,
    'object_detection': True,
    'min_area_motion_contour': 1000,  
    'park_sec_to_wait': 0.001,
    'start_frame': 0,
    'frame_interval': 50  # Number of frames to wait before counting another object
}

# Load video file
cap = cv2.VideoCapture(video_path)

# Define the codec and create VideoWriter object if saving video
if config['save_video']:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    video_info = {'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    out = cv2.VideoWriter(fn_out, fourcc, 25.0, (video_info['width'], video_info['height']))

# Load the area data from the YAML file
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

# Frame counter and flag to track when the last object entered the area
frame_count = 0
last_detected_frame = -config['frame_interval']  # Initialize to a value before the first frame

print("Program for detecting and counting red tomatoes.")

while cap.isOpened():
    try:
        ret, frame = cap.read()
        if not ret:
            print("Capture Error")
            break

        frame_out = frame.copy()

        # Draw the area with a red mark (polygon) on the frame
        cv2.polylines(frame_out, [area_points], isClosed=True, color=(0, 0, 255), thickness=2)

        # Convert frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask to detect red objects
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2

        # Find contours of the red objects
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Increment frame counter
        frame_count += 1

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > config['min_area_motion_contour']:  
                # Calculate centroid of the detected object
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    # Check if the centroid is within the defined area
                    if x_min <= cx <= x_max and y_min <= cy <= y_max:
                        # Check the frame difference to ensure no counting within 10 frames
                        if frame_count - last_detected_frame > config['frame_interval']:
                            # Count the object and update the last detected frame
                            qty += 1
                            total_output += 1
                            last_detected_frame = frame_count

                            # Update timestamps and calculate speed
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

                    # Draw the contour and centroid on the frame
                    cv2.drawContours(frame_out, [cnt], 0, (0, 255, 0), 2)
                    cv2.circle(frame_out, (cx, cy), 5, (255, 0, 0), -1)

        if config['text_overlay']:
            cv2.rectangle(frame_out, (1, 5), (350, 70), (0, 255, 0), 2)
            str_on_frame = "Tomato Counting:"
            cv2.putText(frame_out, str_on_frame, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            str_on_frame = f"Total Counting = {total_output}, Speed (PPM) = {ppm}"
            cv2.putText(frame_out, str_on_frame, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            str_on_frame = f"Fastest PPM: {fastest}, Average: {ppm_average}"
            cv2.putText(frame_out, str_on_frame, (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        # Display video
        imS = cv2.resize(frame_out, (1280, 720))
        cv2.imshow('Tomato Counting - by Jobsnavi', imS)

        # Save the frame to the output video
        if config['save_video']:
            out.write(frame_out)

        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    except KeyboardInterrupt:
        data = (current_time, total_output, minutes, ppm_average, ct, ppm)
        ws.append(data)
        wb.save(path + "output_" + start_time.strftime("%d-%m-%Y %H-%M-%S") + ".xlsx")
        print("Actual Speed (PPM): " + str(ppm_average))
        break

cap.release()
if config['save_video']:
    out.release()
cv2.destroyAllWindows()
