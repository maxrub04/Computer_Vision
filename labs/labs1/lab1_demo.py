from solutions.project1 import secret_image_processing
import sys, numpy as np, cv2 as cv

capture = cv.VideoCapture( cv.CAP_ANY)

if not capture.isOpened():
    print("Failed to initialise camera capture. Check camera connection")
    sys.exit(1)

last_frame = None
while True:
    _, frame = capture.read()
    
    # implement this
    frame = secret_image_processing(frame)
    
    frame = frame if frame is not None else last_frame 
    last_frame = frame

    cv.imshow('Project 01 - Colour tracking', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
