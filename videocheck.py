import os
import cv2

# Read video
video_path = os.path.join('datasets', 'longconvey.mp4')  # Run from the root folder
video = cv2.VideoCapture(video_path)
#longconvey file : 856x480
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video dimensions: {width}x{height}")

ret = True  
while ret:
    ret, frame = video.read()

    if ret:
       
        center_x = width // 2
        center_y = height // 2
        cv2.circle(frame, (center_x, center_y), 75 // 2, (0, 0, 255), -1) 

        cv2.imshow('frame', frame)
        cv2.waitKey(40)  

video.release() 
cv2.destroyAllWindows()
